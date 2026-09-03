"""Fit and persist the deployment classifier for infer_live.py.

    python fit_model.py                       # features_sessions.csv -> model_lda.json
    python fit_model.py --kind qda --out model_qda.json

infer_live.py loads the JSON this writes and never refits. Everything the
loaded model rests on is recorded in its "meta" block so a prediction can be
traced back: which frame, how many rows, which sessions, which two features,
which geometry and detector constants the features were computed under, and
the session-disjoint held-out accuracy of the same recipe.

Two fits happen here, in this order, and both are printed:

  1. The CHECK fit -- train on sessions _00.._02 of each class, test on _03,
     the same split bakeoff.py uses. Its accuracy and Wilson interval go into
     the meta block. This is the number that says what the recipe is worth.
  2. The DEPLOYMENT fit -- all 12 sessions, every row. This is what is saved.
     It has no held-out score of its own; the check fit's score is the
     honest proxy and is labelled as such.

The training data is SIMULATED. gait_gen's constants, the detector thresholds
and the tests over them were co-evolved, so the held-out accuracy is
internally consistent by construction and is not evidence about hardware.
sim_vs_real.py D2 measures what this recipe does on real stances; that number
is much worse.
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd

import detector as D
from discriminant import accuracy_ci, fit_lda, fit_qda, predict, save_model
from make_sessions import CLASSES, session_name

REPO = os.path.dirname(os.path.abspath(__file__))
FRAME_CSV = os.path.join(REPO, "features_sessions.csv")
DEFAULT_OUT = os.path.join(REPO, "model_lda.json")

# Same two features, same held-out session index, as bakeoff.py. Duplicated
# rather than imported because bakeoff.py runs its whole analysis at import.
FEATURES = ["cop_path_len", "cop_displacement"]
TEST_SESSION_IDX = 3


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fit(kind, X, y):
    return fit_lda(X, y) if kind == "lda" else fit_qda(X, y)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frame", default=FRAME_CSV,
                    help="feature frame built by bakeoff.py (default: features_sessions.csv)")
    ap.add_argument("--kind", choices=("lda", "qda"), default="lda")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: model_<kind>.json in the repo root)")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(REPO, f"model_{args.kind}.json")

    if not os.path.exists(args.frame):
        print(f"{args.frame} not found. Build it first:\n    python bakeoff.py")
        return 1

    frame = pd.read_csv(args.frame)
    for col in FEATURES + ["label", "session"]:
        if col not in frame.columns:
            print(f"{args.frame}: missing column {col!r}")
            return 1

    print(f"frame     : {os.path.relpath(args.frame, REPO)}  "
          f"({len(frame)} rows, {frame['session'].nunique()} sessions)")
    print(f"features  : {FEATURES}")
    print(f"kind      : {args.kind}")
    print()

    # 1. check fit, session-disjoint ---------------------------------------
    test_sessions = sorted(session_name(lab, TEST_SESSION_IDX)[:-4] for lab in CLASSES)
    is_test = frame["session"].isin(test_sessions)
    train, test = frame[~is_test], frame[is_test]
    Xtr, ytr = train[FEATURES].to_numpy(float), train["label"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(float), test["label"].to_numpy()

    m_check = fit(args.kind, Xtr, ytr)
    pred = predict(m_check, Xte)
    acc, lo, hi, se = accuracy_ci(yte, pred)
    print("CHECK FIT  (train _00.._02, test _03; the bakeoff.py split)")
    print(f"  n_train = {len(train)}   n_test = {len(test)}")
    print(f"  held-out accuracy = {acc:.4f}   ({int((pred == yte).sum())} / {len(yte)})")
    print(f"  Wilson 95% CI     = [{lo:.4f}, {hi:.4f}]   (Wald se {se:.6f})")
    labs = sorted(set(yte))
    print("  confusion (rows = true, cols = predicted, order " + str(labs) + "):")
    for t in labs:
        row = [int(((yte == t) & (pred == q)).sum()) for q in labs]
        print(f"    {t:8s} " + " ".join(f"{v:5d}" for v in row))
    print()

    # 2. deployment fit, every row -----------------------------------------
    X, y = frame[FEATURES].to_numpy(float), frame["label"].to_numpy()
    model = fit(args.kind, X, y)
    print("DEPLOYMENT FIT  (all sessions, all rows) -> " + os.path.relpath(out, REPO))
    print(f"  classes = {[str(c) for c in model['classes']]}")
    print(f"  counts  = {[int(c) for c in model['counts']]}")
    print(f"  priors  = {np.round(model['priors'], 4).tolist()}")
    print(f"  means   = {np.round(model['means'], 4).tolist()}")

    meta = {
        "purpose": "deployment classifier loaded by infer_live.py",
        "features": FEATURES,
        "training_data": (
            "SIMULATED: 12 gait_gen sessions via make_sessions.py, stances by "
            "detector.find_stances + merge_close, features by "
            "features.extract_features. Not evidence about hardware."),
        "frame": os.path.relpath(args.frame, REPO),
        "frame_sha256": sha256_of(args.frame),
        "n_rows": int(len(frame)),
        "sessions": sorted(frame["session"].unique().tolist()),
        "class_counts": {str(c): int(n) for c, n in
                         frame["label"].value_counts().sort_index().items()},
        "heldout_check": {
            "split": "session-disjoint, test = session _03 of each class",
            "test_sessions": test_sessions,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "accuracy": float(acc),
            "n_correct": int((pred == yte).sum()),
            "wilson95_lo": float(lo),
            "wilson95_hi": float(hi),
            "wald_se": float(se),
        },
        "geometry": {
            "insole_len_mm": D.INSOLE_LEN_MM,
            "insole_width_mm": D.INSOLE_WIDTH_MM,
            "sensor_mm": {k: list(v) for k, v in D.SENSOR_MM.items()},
        },
        "detector": {
            "T_ON": D.T_ON, "T_OFF": D.T_OFF, "MIN_DURATION": D.MIN_DURATION,
            "MAX_DURATION": D.MAX_DURATION, "GAP_MERGE": D.GAP_MERGE,
        },
        "fit_script": "fit_model.py",
    }
    save_model(model, out, meta=meta)
    print(f"  wrote {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
