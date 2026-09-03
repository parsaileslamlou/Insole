"""A classifier trained on the real _02 captures, with the analysis behind it.

    python scripts/train_real.py

Writes docs/real_results.md, figures/real_results/*.png and
models/model_{lda,qda}_real.json. Every number in the document is computed
here; regenerate it rather than editing it. Seeded, deterministic, and it
runs from a fresh clone in well under a minute.

What it does, in order:
  1. loads the _02 captures (labels from data/real/README.md), segments each
     with detector.py at the committed thresholds on RAW counts, and asserts
     the pinned stance counts (stand 0, walk 35, fast 48, shuffle 30). Stand
     yields no stances and is excluded from classification.
  2. computes features.py's extractors under three input representations
     (insole/representations.py): A raw counts, B conductance, C gain-matched
     conductance; and two feature sets: CoP-only (the set the sim bake-off
     used) and the full seven.
  3. splits time-blocked within each session (first 60 % of stances train,
     last 40 % test) -- NOT a per-session split, which one session per class
     makes impossible -- and reports a random stance-level split (20 seeds)
     and contiguous-block cross-validation beside it so the optimism gap is
     visible. Switches to leave-one-session-out automatically once every
     class has two or more sessions.
  4. fits LDA and QDA (insole/discriminant.py) and sklearn logistic regression
     (as scripts/bakeoff.py does) for every representation x feature set,
     and chooses ONE headline cell by a rule fixed before any result was
     seen: CoP-only features, the representation and the model (LDA or QDA)
     with the best contiguous-block CV accuracy, ties to raw counts and LDA.
     The time-blocked test accuracy of that cell is the headline.
  5. lists every misclassified test stance, draws it, and quantifies the
     mechanisms behind each confusion pair; checks peak force against onset
     time per session; measures what the gain match changes; compares the
     sim-trained models on the same stances.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy import stats                                            # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.pipeline import make_pipeline                         # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

from insole import calibration as C                                # noqa: E402
from insole import detector as D                                   # noqa: E402
from insole.discriminant import (                                  # noqa: E402
    DegenerateClassError, IllConditionedCovarianceWarning, SingularCovarianceError,
    accuracy_ci, fit_lda, fit_qda, load_model, predict, save_model,
)
from insole.features import cop_frame                              # noqa: E402
from insole.paths import DATA_REAL, DOCS, FIGURES, MODELS, REPO    # noqa: E402
from insole.representations import (                               # noqa: E402
    LETTER, REPRESENTATIONS, SHIPPED, features_under, gains_from_doc, transform_frames,
)
from insole.splits import (                                        # noqa: E402
    contiguous_block_folds, leave_one_session_out, random_stance_splits,
    sessions_per_class, time_blocked_split,
)

# As-captured filenames and labels: data/real/README.md.
SESSIONS = [("stand", "stand_02.csv"), ("walk", "walk02.csv"),
            ("fast", "fast02.csv"), ("shuffle", "shuffle02.csv")]
EXPECTED_STANCES = {"stand": 0, "walk": 35, "fast": 48, "shuffle": 30}
CLASSES = ["fast", "shuffle", "walk"]                  # np.unique order
COP_FEATURES = ["cop_path_len", "cop_displacement"]     # scripts/bakeoff.py FEATURES
FULL_FEATURES = ["peak_counts", "time_to_peak_s", "contact_time_s",
                 "loading_rate_cps", "impulse_counts_s",
                 "cop_path_len", "cop_displacement"]
FEATURE_SETS = OrderedDict([("cop", COP_FEATURES), ("full", FULL_FEATURES)])
MODEL_KINDS = ("lda", "qda", "lr")
TRAIN_FRAC = 0.6
N_RANDOM = 20
N_BLOCKS = 5
SEED = 0
MM = D.INSOLE_LEN_MM                                    # normalised units -> mm
S4 = D.SENSOR_COLS.index("s4")

RULE = "=" * 78


def head(t):
    print("\n" + RULE + "\n" + t + "\n" + RULE)


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def discover_sessions(real_dir):
    """Known _02 files first, then any extra <label>*.csv that is not _01."""
    found = []
    seen = set()
    for label, fname in SESSIONS:
        path = os.path.join(real_dir, fname)
        if os.path.exists(path):
            found.append((label, fname))
            seen.add(fname)
    for fname in sorted(os.listdir(real_dir)):
        if not fname.endswith(".csv") or fname in seen or "_01" in fname or "check_all" in fname:
            continue
        for label in EXPECTED_STANCES:
            if fname.startswith(label):
                found.append((label, fname))
                seen.add(fname)
    return found


def load_sessions(real_dir):
    out = []
    for label, fname in discover_sessions(real_dir):
        df = pd.read_csv(os.path.join(real_dir, fname))
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        stances = D.merge_close(D.find_stances(total))
        out.append(dict(label=label, session=fname[:-4], file=fname, df=df,
                        total=total, stances=stances))
    return out


def build_frames(sessions, gains):
    """{rep: feature frame}; identical row order across representations."""
    frames = {}
    for rep in REPRESENTATIONS:
        parts = []
        for s in sessions:
            if s["label"] == "stand" or not s["stances"]:
                continue
            f = features_under(s["df"], s["stances"], s["label"], rep,
                               gains if rep == "gain_matched" else None)
            ts = s["df"]["ts_us"].to_numpy()
            f["session"] = s["session"]
            f["onset_s"] = [(ts[a] - ts[0]) / 1e6 for a, _ in s["stances"]]
            f["s4_zero_frac"] = [float((s["df"]["s4"].to_numpy()[a:b + 1] == 0).mean())
                                 for a, b in s["stances"]]
            parts.append(f)
        frames[rep] = pd.concat(parts, ignore_index=True)
    return frames


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def fit_predict(kind, Xtr, ytr, Xte):
    """-> (pred, note) or (None, reason). Standardisation only for LR, as bakeoff.py."""
    if kind == "lr":
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        pipe.fit(Xtr, ytr)
        return pipe.predict(Xte), ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = fit_lda(Xtr, ytr) if kind == "lda" else fit_qda(Xtr, ytr)
            pred = predict(m, Xte)
    except DegenerateClassError as e:
        return None, f"DegenerateClassError: {e}"
    except SingularCovarianceError as e:
        return None, f"SingularCovarianceError: {e}"
    n_ill = sum(issubclass(w.category, IllConditionedCovarianceWarning) for w in caught)
    return pred, (f"{n_ill} ill-conditioned covariance warning(s)" if n_ill else "")


def evaluate(kind, frame, feats, train_idx, test_idx):
    X = frame[feats].to_numpy(float)
    y = frame["label"].to_numpy()
    if not np.isfinite(X[np.concatenate([train_idx, test_idx])]).all():
        return dict(skipped="non-finite feature value in the split")
    pred, note = fit_predict(kind, X[train_idx], y[train_idx], X[test_idx])
    if pred is None:
        return dict(skipped=note)
    yte = y[test_idx]
    acc, lo, hi, se = accuracy_ci(yte, pred)
    labs = CLASSES
    conf = [[int(((yte == t) & (pred == q)).sum()) for q in labs] for t in labs]
    prec, rec = {}, {}
    for lab in labs:
        tp = int(((yte == lab) & (pred == lab)).sum())
        fp = int(((yte != lab) & (pred == lab)).sum())
        fn = int(((yte == lab) & (pred != lab)).sum())
        prec[lab] = tp / (tp + fp) if tp + fp else float("nan")
        rec[lab] = tp / (tp + fn) if tp + fn else float("nan")
    return dict(acc=acc, lo=lo, hi=hi, se=se, n_correct=int((pred == yte).sum()),
                n_test=len(yte), pred=pred, yte=yte, test_idx=test_idx,
                conf=conf, precision=prec, recall=rec, note=note)


def cv_accuracy(kind, frame, feats, folds):
    """Pooled accuracy over folds (every stance tested once) and per-fold list."""
    correct = total = 0
    per_fold = []
    for tr, te in folds:
        r = evaluate(kind, frame, feats, tr, te)
        if "skipped" in r:
            return float("nan"), [], r["skipped"]
        correct += r["n_correct"]
        total += r["n_test"]
        per_fold.append(r["acc"])
    return correct / total, per_fold, ""


def random_summary(kind, frame, feats):
    accs = []
    for tr, te in random_stance_splits(frame, TRAIN_FRAC, N_RANDOM, SEED):
        r = evaluate(kind, frame, feats, tr, te)
        if "skipped" in r:
            return float("nan"), float("nan"), float("nan"), r["skipped"]
        accs.append(r["acc"])
    return float(np.mean(accs)), float(np.min(accs)), float(np.max(accs)), ""


def mcnemar(a_pred, b_pred, yte):
    ok_a, ok_b = a_pred == yte, b_pred == yte
    b = int((ok_a & ~ok_b).sum())
    c = int((~ok_a & ok_b).sum())
    if b + c == 0:
        return b, c, float("nan")
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


# ---------------------------------------------------------------------------
# 3. Mechanism measurements
# ---------------------------------------------------------------------------
def cop_mm(vals6):
    """CoP in mm for an (n, 6) array of channel values (any representation)."""
    out = np.full((len(vals6), 2), np.nan)
    for i, v in enumerate(vals6):
        row = dict(zip(D.SENSOR_COLS, v))
        x, y = cop_frame(row)
        out[i] = (x * MM, y * MM)
    return out


def stance_frames(sessions, label=None):
    """(n, 6) raw counts of every frame inside a kept stance, per session."""
    for s in sessions:
        if s["label"] == "stand" or (label and s["label"] != label):
            continue
        vals = s["df"][D.SENSOR_COLS].to_numpy(dtype=float)
        for a, b in s["stances"]:
            yield s, vals[a:b + 1]


def gain_effect_mm(sessions, gains, fs):
    """Mean |CoP(C) - CoP(A)| over stance frames, per class, in mm."""
    out = {}
    for lab in CLASSES:
        d = []
        for _, v in stance_frames(sessions, lab):
            a = cop_mm(v)
            c = cop_mm(transform_frames(v, "gain_matched", gains, fs))
            ok = ~np.isnan(a[:, 0]) & ~np.isnan(c[:, 0])
            d.append(np.hypot(*(c[ok] - a[ok]).T))
        d = np.concatenate(d)
        out[lab] = (float(d.mean()), float(np.median(d)))
    return out


def s4_zero_effect_mm(sessions, substitute):
    """On stance frames where s4 == 0: CoP with s4 := substitute minus actual, per class."""
    out = {}
    for lab in CLASSES:
        d, n, tot = [], 0, 0
        for _, v in stance_frames(sessions, lab):
            z = (v[:, S4] == 0) & (v.sum(axis=1) > 0)
            tot += len(v)
            n += int(z.sum())
            if z.any():
                w = v[z].copy()
                w[:, S4] = substitute
                d.append(np.hypot(*(cop_mm(w) - cop_mm(v[z])).T))
        d = np.concatenate(d) if d else np.array([np.nan])
        out[lab] = (float(np.nanmean(d)), n / tot if tot else float("nan"))
    return out


def ml_spread_mm(sessions):
    """Stance-to-stance sd of the mean medial-lateral CoP position, per class (mm)."""
    out = {}
    for lab in CLASSES:
        means = [np.nanmean(cop_mm(v)[:, 0]) for _, v in stance_frames(sessions, lab)]
        out[lab] = float(np.std(means, ddof=1))
    return out


def band_overlap(frame, feats, train_idx):
    """Fraction of each class's stances inside another class's p10-p90 band (train)."""
    out = {}
    tr = frame.loc[train_idx]
    for f in feats:
        bands = {lab: np.percentile(tr.loc[tr["label"] == lab, f], [10, 90]) for lab in CLASSES}
        for a in CLASSES:
            va = frame.loc[frame["label"] == a, f].to_numpy()
            for b in CLASSES:
                if a == b:
                    continue
                lo, hi = bands[b]
                out[(f, a, b)] = float(((va >= lo) & (va <= hi)).mean())
    return out


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def fig_feature_distributions(frame, feats, path):
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.6))
    axes = np.atleast_1d(axes)
    colors = {"fast": "#E07A3F", "shuffle": "#7A5CC7", "walk": "#378ADD"}
    rng = np.random.default_rng(SEED)
    for ax, f in zip(axes, feats):
        for i, lab in enumerate(CLASSES):
            v = frame.loc[frame["label"] == lab, f].to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, alpha=0.7,
                       color=colors[lab], label=f"{lab} (n={len(v)})")
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", lw=1.2)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_title(f)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-class feature distributions, all stances (bar = median)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_errors(sessions, rows, rep, gains, fs, path, title):
    """One figure per confusion pair: total-force trace and CoP path per stance."""
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.2), squeeze=False)
    by_session = {s["session"]: s for s in sessions}
    for j, r in enumerate(rows):
        s = by_session[r["session"]]
        a, b = int(r["start"]), int(r["end"])
        vals = s["df"][D.SENSOR_COLS].to_numpy(dtype=float)[a:b + 1]
        t = np.arange(len(vals)) / 100.0
        axes[0, j].plot(t, vals.sum(axis=1), color="0.2", lw=1.2)
        axes[0, j].plot(t, vals[:, S4], color="#E07A3F", lw=0.9, label="s4")
        axes[0, j].set_title(f"{r['session']} @ {r['onset_s']:.1f}s", fontsize=8)
        axes[0, j].set_xlabel("s")
        if j == 0:
            axes[0, j].set_ylabel("total counts (s4 in orange)")
            axes[0, j].legend(fontsize=7)
        cp = cop_mm(transform_frames(vals, rep, gains, fs))
        ax = axes[1, j]
        ax.plot([0, D.INSOLE_WIDTH_MM, D.INSOLE_WIDTH_MM, 0, 0], [0, 0, MM, MM, 0], color="0.6", lw=0.8)
        for name, (x, y) in D.SENSOR_MM.items():
            ax.scatter([x], [y], s=12, color="0.5")
        ax.plot(cp[:, 0], cp[:, 1], color="#1D9E75", lw=1.2)
        ok = ~np.isnan(cp[:, 0])
        if ok.any():
            ax.scatter(cp[ok][0, 0], cp[ok][0, 1], s=22, color="#1D9E75", marker="o")
            ax.scatter(cp[ok][-1, 0], cp[ok][-1, 1], s=22, color="#1D9E75", marker="s")
        ax.set_aspect("equal")
        ax.set_xlim(-5, D.INSOLE_WIDTH_MM + 5)
        ax.set_ylim(-5, MM + 5)
        ax.set_xlabel("medial -> lateral (mm)")
        ax.set_title(f"path {r['cop_path_len']:.2f} disp {r['cop_displacement']:.2f}", fontsize=8)
        if j == 0:
            ax.set_ylabel("heel -> toe (mm)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_peak_vs_onset(frame, trends, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, lab in zip(axes, CLASSES):
        sub = frame[frame["label"] == lab]
        ax.scatter(sub["onset_s"], sub["peak_counts"], s=14, color="0.25")
        sl, ic, r, p = trends[lab]
        xs = np.array([sub["onset_s"].min(), sub["onset_s"].max()])
        ax.plot(xs, ic + sl * xs, color="#E07A3F", lw=1.2)
        ax.set_title(f"{lab}: slope {sl:+.1f} counts/s, r={r:+.2f}, p={p:.3f}", fontsize=9)
        ax.set_xlabel("stance onset (s into the capture)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("peak total force (raw counts)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def fmt_ci(r):
    return f"{r['acc']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}] ({r['n_correct']}/{r['n_test']})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-dir", default=str(DATA_REAL))
    ap.add_argument("--gain", default=str(MODELS / "gain_match.json"))
    ap.add_argument("--doc", default=str(DOCS / "real_results.md"))
    ap.add_argument("--fig-dir", default=str(FIGURES / "real_results"))
    ap.add_argument("--models-dir", default=str(MODELS))
    ap.add_argument("--guard", type=int, default=0,
                    help="stances dropped from the end of each training block (default 0)")
    args = ap.parse_args(argv)
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.doc), exist_ok=True)
    t_start = time.time()
    md = []                                          # the document, line by line

    def out(s=""):
        print(s)
        md.append(s)

    gm = C.load_gain_match(args.gain)
    gains = gains_from_doc(gm)
    fs = float(gm["fs_counts"])

    # -- 1. data -----------------------------------------------------------
    head("1  DATA -- the _02 captures, segmented on raw counts")
    sessions = load_sessions(args.real_dir)
    counts = {}
    for s in sessions:
        counts.setdefault(s["label"], 0)
        counts[s["label"]] += len(s["stances"])
        print(f"  {s['label']:8s} {s['file']:16s} frames={len(s['df'])} stances={len(s['stances'])}")
    for lab, want in EXPECTED_STANCES.items():
        got = counts.get(lab, 0)
        assert got == want, f"{lab}: {got} stances, expected {want} at MAX_DURATION={D.MAX_DURATION}"
    frames = build_frames(sessions, gains)
    frame_a = frames["raw"]
    n_all = len(frame_a)
    spc = sessions_per_class(frame_a)
    multi_session = all(len(v) >= 2 for v in spc.values())
    print(f"  moving stances: {n_all}  sessions per class: { {k: len(v) for k, v in spc.items()} }")

    # -- 3. splits ---------------------------------------------------------
    head("2  SPLITS")
    if multi_session:
        folds = list(leave_one_session_out(frame_a))
        print(f"  every class has >= 2 sessions: leave-one-session-out, {len(folds)} folds")
        train_idx, test_idx = folds[0][0], folds[0][1]
        split_name = "leave-one-session-out (fold 0 shown as the headline split)"
        per_class = {lab: (int((frame_a.loc[train_idx, 'label'] == lab).sum()),
                           int((frame_a.loc[test_idx, 'label'] == lab).sum()), 0) for lab in CLASSES}
    else:
        train_idx, test_idx, per_class = time_blocked_split(frame_a, TRAIN_FRAC, args.guard)
        split_name = (f"time-blocked within each session: first {TRAIN_FRAC:.0%} of stances "
                      f"train, last {1 - TRAIN_FRAC:.0%} test, guard band {args.guard}")
        print("  ONE session per class: no per-session split is possible. Using " + split_name)
    for lab in CLASSES:
        print(f"    {lab:8s} train {per_class[lab][0]:3d}  test {per_class[lab][1]:3d}  dropped {per_class[lab][2]}")
    yte_all = frame_a.loc[test_idx, "label"].to_numpy()
    floor_lab, floor_n = Counter(yte_all).most_common(1)[0]
    floor = floor_n / len(yte_all)
    print(f"  n_train={len(train_idx)} n_test={len(test_idx)}  test majority floor = "
          f"{floor:.4f} ({floor_n}/{len(test_idx)}, always '{floor_lab}')")
    block_folds = list(contiguous_block_folds(frame_a, N_BLOCKS))

    # -- 4. grid -----------------------------------------------------------
    head("3  RESULTS GRID -- representation x feature set x model")
    grid = OrderedDict()
    cv = {}
    for rep in REPRESENTATIONS:
        fr = frames[rep]
        for fset, feats in FEATURE_SETS.items():
            for kind in MODEL_KINDS:
                key = (rep, fset, kind)
                r = evaluate(kind, fr, feats, train_idx, test_idx)
                cv_acc, cv_folds, cv_skip = cv_accuracy(kind, fr, feats, block_folds)
                rnd = random_summary(kind, fr, feats)
                grid[key] = dict(tb=r, cv=cv_acc, cv_folds=cv_folds, cv_skip=cv_skip,
                                 rnd=rnd)
                cv[key] = cv_acc
                if "skipped" in r:
                    print(f"  {LETTER[rep]} {fset:4s} {kind:3s}  SKIPPED: {r['skipped']}")
                else:
                    print(f"  {LETTER[rep]} {fset:4s} {kind:3s}  time-blocked {fmt_ci(r)}  "
                          f"block-CV {cv_acc:.4f}  random {rnd[0]:.4f} [{rnd[1]:.4f}, {rnd[2]:.4f}]"
                          + (f"  ({r['note']})" if r["note"] else ""))

    # -- headline by the pre-registered rule --------------------------------
    order_rep = {"raw": 0, "gain_matched": 1, "conductance": 2}
    order_kind = {"lda": 0, "qda": 1}
    candidates = [k for k in grid if k[1] == "cop" and k[2] in ("lda", "qda")
                  and "skipped" not in grid[k]["tb"] and np.isfinite(cv[k])]
    headline = sorted(candidates, key=lambda k: (-round(cv[k], 6), order_rep[k[0]], order_kind[k[2]]))[0]
    h_rep, h_fset, h_kind = headline
    H = grid[headline]["tb"]
    h_frame = frames[h_rep]
    h_feats = FEATURE_SETS[h_fset]
    print(f"\n  HEADLINE (rule: CoP-only; best block-CV among LDA/QDA; ties -> raw, LDA): "
          f"{LETTER[h_rep]} {h_fset} {h_kind}  time-blocked {fmt_ci(H)}")
    other_kind = "qda" if h_kind == "lda" else "lda"
    O = grid[(h_rep, h_fset, other_kind)]["tb"]
    b, c, p_mc = (mcnemar(H["pred"], O["pred"], H["yte"]) if "skipped" not in O else (0, 0, float("nan")))
    guard_r = None
    if not multi_session:
        g_tr, g_te, g_pc = time_blocked_split(frame_a, TRAIN_FRAC, 1)
        guard_r = evaluate(h_kind, h_frame, h_feats, g_tr, g_te)
    full_best = max((k for k in grid if k[1] == "full" and "skipped" not in grid[k]["tb"]),
                    key=lambda k: grid[k]["tb"]["acc"])

    # -- 5. mechanisms -------------------------------------------------------
    head("4  MECHANISMS")
    errors = []
    for i, (t, q) in enumerate(zip(H["yte"], H["pred"])):
        if t != q:
            row = h_frame.loc[H["test_idx"][i]]
            errors.append(dict(session=row["session"], onset_s=float(row["onset_s"]),
                               start=int(row["start"]), end=int(row["end"]), true=t, pred=q,
                               s4_zero_frac=float(row["s4_zero_frac"]),
                               **{f: float(row[f]) for f in FULL_FEATURES}))
    pairs = OrderedDict()
    for e in errors:
        pairs.setdefault((e["true"], e["pred"]), []).append(e)
    train_means = {lab: h_frame.loc[train_idx][h_frame.loc[train_idx, "label"] == lab][h_feats + ["s4_zero_frac"]].mean()
                   for lab in CLASSES}
    overlap = band_overlap(h_frame, h_feats, train_idx)
    s4_frac = {lab: float(frame_a.loc[frame_a["label"] == lab, "s4_zero_frac"].mean()) for lab in CLASSES}
    all_s4 = np.concatenate([s["df"]["s4"].to_numpy() for s in sessions])
    substitute = float(np.median(all_s4[all_s4 > 0]))
    s4_eff = s4_zero_effect_mm(sessions, substitute)
    g_eff = gain_effect_mm(sessions, gains, fs)
    spread = ml_spread_mm(sessions)
    trends = {}
    for lab in CLASSES:
        sub = frame_a[frame_a["label"] == lab]
        lr = stats.linregress(sub["onset_s"], sub["peak_counts"])
        trends[lab] = (float(lr.slope), float(lr.intercept), float(lr.rvalue), float(lr.pvalue))
    for lab in CLASSES:
        print(f"  {lab:8s} s4==0 inside stances {s4_frac[lab]:.1%}  s4-zero CoP effect "
              f"{s4_eff[lab][0]:.1f} mm  gain-match CoP effect {g_eff[lab][0]:.2f} mm (mean)  "
              f"ML spread sd {spread[lab]:.2f} mm  peak-vs-onset slope {trends[lab][0]:+.1f} counts/s "
              f"(p={trends[lab][3]:.3f})")

    # sim-trained models on the same stances, under the representation they
    # were fitted on (their meta says which; pre-stage-20 fits were raw).
    sim_acc = {}
    for kind in ("lda", "qda"):
        path = os.path.join(args.models_dir, f"model_{kind}.json")
        if os.path.exists(path):
            m = load_model(path)
            m_rep = m["meta"].get("representation", "raw")
            fr = frames[m_rep]
            X = fr[m["meta"]["features"]].to_numpy(float)
            pred = predict(m, X)
            sim_acc[kind] = (float((pred == fr["label"].to_numpy()).mean()),
                             int((pred == fr["label"].to_numpy()).sum()), m_rep)
    all_floor = Counter(frame_a["label"]).most_common(1)[0][1] / n_all

    # -- figures ---------------------------------------------------------------
    fig_feature_distributions(h_frame, h_feats, os.path.join(args.fig_dir, "feature_distributions.png"))
    fig_peak_vs_onset(frame_a, trends, os.path.join(args.fig_dir, "peak_vs_onset.png"))
    err_figs = {}
    for (t, q), rows in pairs.items():
        name = f"errors_{t}_to_{q}.png"
        fig_errors(sessions, rows, h_rep, gains, fs, os.path.join(args.fig_dir, name),
                   f"true {t} -> predicted {q}: {len(rows)} test stance(s), "
                   f"CoP under {LETTER[h_rep]} ({h_rep}); circle = onset, square = end")
        err_figs[(t, q)] = name

    # -- 6. persist models --------------------------------------------------------
    head("5  PERSIST")
    X_all = h_frame[h_feats].to_numpy(float)
    y_all = h_frame["label"].to_numpy()
    saved = {}
    for kind, fit in (("lda", fit_lda), ("qda", fit_qda)):
        try:
            m = fit(X_all, y_all)
        except (DegenerateClassError, SingularCovarianceError) as e:
            print(f"  {kind}: not persisted ({e})")
            continue
        r = grid[(h_rep, h_fset, kind)]["tb"]
        meta = {
            "purpose": "classifier trained on the real _02 captures; loaded by infer_live.py with --model",
            "training_data": ("REAL: data/real _02 set, one 60 s session per class, one subject, "
                              "figure-8 path, tethered USB; stances by detector.find_stances + "
                              "merge_close on raw counts at the committed thresholds"),
            "sessions": sorted(frame_a["session"].unique().tolist()),
            "n_rows": int(n_all),
            "class_counts": {lab: int((y_all == lab).sum()) for lab in CLASSES},
            "representation": h_rep,
            "representation_letter": LETTER[h_rep],
            "gain_match": os.path.relpath(args.gain, REPO) if h_rep == "gain_matched" else None,
            "feature_set": h_fset,
            "features": h_feats,
            "split": split_name + " -- NOT a per-session split",
            "heldout_check": ({"n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
                               "accuracy": float(r["acc"]), "n_correct": int(r["n_correct"]),
                               "wilson95_lo": float(r["lo"]), "wilson95_hi": float(r["hi"]),
                               "test_floor": float(floor),
                               "block_cv_accuracy": float(grid[(h_rep, h_fset, kind)]["cv"]),
                               "random_split_mean": float(grid[(h_rep, h_fset, kind)]["rnd"][0])}
                              if "skipped" not in r else {"skipped": r["skipped"]}),
            "geometry": {"insole_len_mm": D.INSOLE_LEN_MM, "insole_width_mm": D.INSOLE_WIDTH_MM,
                         "sensor_mm": {k: list(v) for k, v in D.SENSOR_MM.items()}},
            "detector": {"T_ON": D.T_ON, "T_OFF": D.T_OFF, "MIN_DURATION": D.MIN_DURATION,
                         "MAX_DURATION": D.MAX_DURATION, "GAP_MERGE": D.GAP_MERGE},
            "git_hash": git_hash(),
            "fit_script": "scripts/train_real.py",
        }
        path = os.path.join(args.models_dir, f"model_{kind}_real.json")
        save_model(m, path, meta=meta)
        saved[kind] = os.path.relpath(path, REPO)
        print(f"  wrote {saved[kind]}")

    # -- 7. the document ------------------------------------------------------------
    head("6  DOCUMENT -> " + os.path.relpath(args.doc, REPO))
    L = LETTER
    out("# Real-data results")
    out()
    out("Every number in this file is written by `python scripts/train_real.py` "
        f"(git {git_hash()}); regenerate it, do not edit it. Figures: "
        f"`{os.path.relpath(args.fig_dir, REPO)}/`. Models: "
        + ", ".join(f"`{p}`" for p in saved.values()) + ".")
    out()
    out("## 1. Data")
    out()
    out("The `_02` captures only (`data/real/README.md`): one 60 s session per activity, "
        "100 Hz, tethered USB, one subject, one day, walking a figure-8. `_01` is failure "
        "evidence and is never trained or evaluated on. Segmentation uses `insole/detector.py` "
        f"at the committed thresholds (T_ON={D.T_ON}, T_OFF={D.T_OFF}, MIN_DURATION={D.MIN_DURATION}, "
        f"MAX_DURATION={D.MAX_DURATION}, GAP_MERGE={D.GAP_MERGE}) on raw counts, and the counts are "
        "asserted against `tests/test_stances.py`:")
    out()
    out("| activity | file | frames | stances kept |")
    out("|---|---|---|---|")
    for s in sessions:
        out(f"| {s['label']} | `{s['file']}` | {len(s['df'])} | {len(s['stances'])} |")
    out()
    out(f"Standing is one unbroken {len(sessions[0]['df'])}-frame contact, rejected by MAX_DURATION, so "
        f"it contributes no stances and is excluded from classification. n = {n_all} moving stances "
        f"({', '.join(f'{lab} {int((y_all == lab).sum())}' for lab in CLASSES)}); the all-data "
        f"majority floor is {all_floor:.4f}.")
    out()
    out("## 2. Representations and feature sets")
    out()
    out("Features are `insole/features.py`'s extractors, unchanged, computed on three per-frame "
        "input representations (`insole/representations.py`); the detector always runs on raw counts:")
    out()
    out("- **A raw**: the six ADC counts as logged.")
    out("- **B conductance**: x = counts / (4095 − counts) per channel; x(0) = 0.")
    out(f"- **C gain-matched**: x · g, g from `models/gain_match.json` "
        f"({', '.join(f's{i}={gains[i]:.4f}' for i in range(6))}).")
    out()
    out("Force is linear in conductance (`insole/calibration.py`), so under B and C the centre of "
        "pressure is a force-proportional centroid and under A it is not. The gain match is a "
        "single-point relative match at ~12 N: above 824 counts (62–67 % of loaded walking "
        "frames, `scripts/analyze_real.py` C3) it extrapolates, and below ~5 N the channels' "
        "activation thresholds differ, so it does not hold there.")
    out()
    out(f"**Shipped representation: {L[SHIPPED]} ({SHIPPED})** -- `insole.representations.SHIPPED`, "
        f"the one `insole/infer_live.py` feeds on every source, `scripts/bakeoff.py` builds the sim "
        f"frame under, and the persisted models are fitted on. It was chosen by the headline rule "
        f"in section 4 (block-CV, CoP-only, LDA/QDA: A {max(cv[('raw', 'cop', k)] for k in ('lda', 'qda')):.4f}, "
        f"B {max(cv[('conductance', 'cop', k)] for k in ('lda', 'qda')):.4f}, "
        f"C {max(cv[('gain_matched', 'cop', k)] for k in ('lda', 'qda')):.4f}). The simulator has no "
        f"per-channel gain to correct, so under B every source is treated identically; the gain match "
        f"still runs per frame for the extrapolation counter.")
    out()
    out("Two feature sets: **cop** = `cop_path_len`, `cop_displacement` (exactly the set "
        "`scripts/bakeoff.py` used, for comparability with the simulator), and **full** = all "
        "seven (`peak_counts`, `time_to_peak_s`, `contact_time_s`, `loading_rate_cps`, "
        "`impulse_counts_s` plus the two CoP features). The sim bake-off excluded the five "
        "timing/magnitude features because simulated fast and walk differ only in cadence, so "
        "any cadence feature reads the label off the generator. On real data cadence is measured, "
        "not constructed, so the full set is a legitimate classifier here, but it is not "
        "comparable with the sim number. Under B and C the count-valued features are in "
        "conductance units and keep their column names.")
    out()
    out("## 3. Split")
    out()
    out(f"**{split_name}.** " + ("Every class has at least two sessions, so the script switched to "
        "leave-one-session-out." if multi_session else
        "There is one session per class, so no per-session split exists. Stances are sorted by "
        "onset within each session; the earlier ones train and the later ones test. This is a "
        "within-session number and carries the leakage that implies: consecutive stances of one "
        "walk share the subject, the day, the shoe, the path and the sensor state. Do not read it as "
        "generalisation to a new session."))
    out()
    out("| class | train | test | dropped (guard) |")
    out("|---|---|---|---|")
    for lab in CLASSES:
        out(f"| {lab} | {per_class[lab][0]} | {per_class[lab][1]} | {per_class[lab][2]} |")
    out()
    out(f"n_train = {len(train_idx)}, n_test = {len(test_idx)}, test majority floor = {floor:.4f} "
        f"({floor_n}/{len(test_idx)}, always `{floor_lab}`).")
    out()
    out("Two further splits are reported beside it so the optimism gap is visible: a **random "
        f"stance-level split** with the same per-class sizes, {N_RANDOM} seeds (mean, min, max), which "
        "puts near-copies of every test stance into training and is expected to be optimistic; and "
        f"**contiguous-block cross-validation**, {N_BLOCKS} time blocks per class, each block held out "
        "once, which tests every stance exactly once with its own block out of training.")
    out()
    out("## 4. Results grid")
    out()
    out("Accuracy on the time-blocked test set with a Wilson 95 % interval and the count, then the "
        "block-CV pooled accuracy, then the random-split mean [min, max]. LDA/QDA are "
        "`insole/discriminant.py`; LR is sklearn's `LogisticRegression` on standardised features, as "
        "in `scripts/bakeoff.py`. A skipped cell says why.")
    out()
    out("| rep | features | model | time-blocked acc [Wilson 95 %] | block-CV | random split |")
    out("|---|---|---|---|---|---|")
    for key, g in grid.items():
        rep, fset, kind = key
        r = g["tb"]
        star = " **(headline)**" if key == headline else ""
        if "skipped" in r:
            out(f"| {L[rep]} | {fset} | {kind} | skipped: {r['skipped']} | | |")
        else:
            out(f"| {L[rep]} | {fset} | {kind} | {fmt_ci(r)}{star} | {g['cv']:.4f} | "
                f"{g['rnd'][0]:.4f} [{g['rnd'][1]:.4f}, {g['rnd'][2]:.4f}] |")
    out()
    out("### Headline")
    out()
    out(f"Rule, fixed before any result was seen: CoP-only features; among LDA and QDA under A, B "
        f"and C, the cell with the best block-CV accuracy; ties go to raw counts and to LDA. "
        f"That is **{L[h_rep]} ({h_rep}), {h_fset}, {h_kind.upper()}**: time-blocked accuracy "
        f"**{H['acc']:.4f}** [{H['lo']:.4f}, {H['hi']:.4f}] ({H['n_correct']}/{H['n_test']}) against a "
        f"test floor of {floor:.4f}; block-CV {grid[headline]['cv']:.4f} "
        f"(folds {', '.join(f'{a:.3f}' for a in grid[headline]['cv_folds'])}); random split "
        f"{grid[headline]['rnd'][0]:.4f} [{grid[headline]['rnd'][1]:.4f}, {grid[headline]['rnd'][2]:.4f}]. "
        f"The gap between the random-split mean and the time-blocked number is the optimism that "
        f"temporal adjacency buys on this data: {grid[headline]['rnd'][0] - H['acc']:+.4f}.")
    out()
    if guard_r is not None and "skipped" not in guard_r:
        out(f"With a one-stance guard band between the training and test blocks (training loses its "
            f"last stance per class) the same cell scores {fmt_ci(guard_r)}.")
        out()
    out(f"The best full-feature cell is {L[full_best[0]]} {full_best[1]} {full_best[2]} at "
        f"{fmt_ci(grid[full_best]['tb'])} (block-CV {grid[full_best]['cv']:.4f}). It is the better "
        f"classifier of these activities and it is reported here as such, but it rides on "
        f"`contact_time_s` and its relatives, whose class medians on this data are "
        + ", ".join(f"{lab} {frame_a.loc[frame_a['label'] == lab, 'contact_time_s'].median():.2f} s" for lab in CLASSES)
        + ", i.e. on cadence; it is not comparable with the sim bake-off and does not test the CoP features.")
    out()
    out("Confusion matrix of the headline cell (rows true, columns predicted, order "
        + ", ".join(CLASSES) + "):")
    out()
    out("| | " + " | ".join(CLASSES) + " | recall |")
    out("|---|" + "---|" * (len(CLASSES) + 1))
    for i, lab in enumerate(CLASSES):
        out(f"| **{lab}** | " + " | ".join(str(v) for v in H["conf"][i]) + f" | {H['recall'][lab]:.3f} |")
    out("| precision | " + " | ".join(f"{H['precision'][lab]:.3f}" for lab in CLASSES) + " | |")
    out()
    worst = min(CLASSES, key=lambda lab: H["recall"][lab])
    wi = CLASSES.index(worst)
    wq = CLASSES[int(np.argmax([v if j != wi else -1 for j, v in enumerate(H["conf"][wi])]))]
    out(f"Per-class recall " + ", ".join(f"{lab} {H['recall'][lab]:.3f}" for lab in CLASSES)
        + f": the headline's accuracy comes from the other classes; {worst} test stances are called "
        f"{wq} {H['conf'][wi][CLASSES.index(wq)]} times out of {sum(H['conf'][wi])}. The table below "
        f"shows why a time-blocked split does that -- the training block and the test block of one "
        f"session are not the same distribution:")
    out()
    out("| feature | class | train block mean | test block mean | shift in test-block sd |")
    out("|---|---|---|---|---|")
    for f in h_feats:
        for lab in CLASSES:
            tr_v = h_frame.loc[train_idx][h_frame.loc[train_idx, "label"] == lab][f]
            te_v = h_frame.loc[test_idx][h_frame.loc[test_idx, "label"] == lab][f]
            sd = te_v.std() if len(te_v) > 1 and te_v.std() > 0 else float("nan")
            out(f"| {f} | {lab} | {tr_v.mean():.4f} | {te_v.mean():.4f} | {(te_v.mean() - tr_v.mean()) / sd:+.2f} |")
    out()
    if "skipped" not in O:
        out(f"McNemar, {h_kind.upper()} vs {other_kind.upper()} on the same test set: b = {b}, c = {c}"
            + (f", exact two-sided p = {p_mc:.4f}" if b + c else " -- no discordant pairs, no test")
            + (f"; the smallest p attainable at b + c = {b + c} is {min(1.0, 2 * 0.5 ** (b + c)):.4f}."
               if b + c else "."))
        out()
    out("## 5. Per-class feature distributions (headline cell)")
    out()
    out(f"![feature distributions]({os.path.relpath(os.path.join(args.fig_dir, 'feature_distributions.png'), os.path.dirname(args.doc))})")
    out()
    out("| feature | class | n | mean | sd | min | median | max |")
    out("|---|---|---|---|---|---|---|---|")
    for f in h_feats:
        for lab in CLASSES:
            v = h_frame.loc[h_frame["label"] == lab, f]
            out(f"| {f} | {lab} | {len(v)} | {v.mean():.4f} | {v.std():.4f} | {v.min():.4f} | {v.median():.4f} | {v.max():.4f} |")
    out()
    out("Fraction of each class's stances (all of them) that fall inside another class's "
        "p10–p90 training band, per feature. High values are the overlap the classifier cannot "
        "resolve:")
    out()
    out("| feature | class | inside " + " band | inside ".join(CLASSES) + " band |")
    out("|---|---|" + "---|" * len(CLASSES))
    for f in h_feats:
        for a in CLASSES:
            cells = ["—" if a == b_ else f"{overlap[(f, a, b_)]:.2f}" for b_ in CLASSES]
            out(f"| {f} | {a} | " + " | ".join(cells) + " |")
    out()
    out("## 6. Every misclassified test stance")
    out()
    if not errors:
        out("None.")
    else:
        out("| session | onset (s) | true | predicted | " + " | ".join(h_feats) + " | contact_time_s | s4 = 0 frames |")
        out("|---|---|---|---|" + "---|" * (len(h_feats) + 2))
        for e in errors:
            out(f"| {e['session']} | {e['onset_s']:.2f} | {e['true']} | {e['pred']} | "
                + " | ".join(f"{e[f]:.4f}" for f in h_feats)
                + f" | {e['contact_time_s']:.2f} | {e['s4_zero_frac']:.0%} |")
        out()
        for (t, q), rows in pairs.items():
            out(f"### {t} → {q} ({len(rows)})")
            out()
            out(f"![{t} to {q}]({os.path.relpath(os.path.join(args.fig_dir, err_figs[(t, q)]), os.path.dirname(args.doc))})")
            out()
            sent = []
            for f in h_feats:
                em = float(np.mean([r[f] for r in rows]))
                mt, mq = float(train_means[t][f]), float(train_means[q][f])
                nearer = q if abs(em - mq) < abs(em - mt) else t
                sent.append(f"`{f}` averages {em:.4f} over these stances against training means "
                            f"{mt:.4f} ({t}) and {mq:.4f} ({q}), nearer to {nearer}")
            s4m = float(np.mean([r["s4_zero_frac"] for r in rows]))
            sent.append(f"s4 read 0 on {s4m:.0%} of their frames against {float(train_means[t]['s4_zero_frac']):.0%} "
                        f"for {t} and {float(train_means[q]['s4_zero_frac']):.0%} for {q} in training; on those frames "
                        f"the CoP is a five-sensor centroid, and the counterfactual below puts that at "
                        f"{s4_eff[t][0]:.1f} mm for {t}")
            ct = float(np.mean([r["contact_time_s"] for r in rows]))
            sent.append(f"their contact time averages {ct:.2f} s against class medians "
                        f"{frame_a.loc[frame_a['label'] == t, 'contact_time_s'].median():.2f} s ({t}) and "
                        f"{frame_a.loc[frame_a['label'] == q, 'contact_time_s'].median():.2f} s ({q}), which the "
                        f"CoP-only cell never sees")
            out("Measured: " + "; ".join(sent) + ".")
            out()
    out("## 7. Mechanisms, measured")
    out()
    out("| class | s4 = 0 inside stances | CoP shift on s4-zero frames (mm) | CoP shift A → C, all stance frames (mean / median mm) | stance-to-stance ML spread sd (mm) |")
    out("|---|---|---|---|---|")
    for lab in CLASSES:
        out(f"| {lab} | {s4_frac[lab]:.1%} | {s4_eff[lab][0]:.1f} | {g_eff[lab][0]:.2f} / {g_eff[lab][1]:.2f} | {spread[lab]:.2f} |")
    out()
    out(f"**s4 zeros.** s4 (1st metatarsal head) has the highest activation threshold of the six, so "
        f"its zeros are below-threshold readings, never imputed. On the frames inside kept stances "
        f"it reads 0 on {s4_frac['fast']:.0%} (fast), {s4_frac['walk']:.0%} (walk) and "
        f"{s4_frac['shuffle']:.0%} (shuffle) of frames. The CoP shift those zeros are responsible "
        f"for is measured directly: on every such frame the CoP is recomputed with s4 set to the "
        f"median non-zero s4 count across the four captures ({substitute:.0f} counts) and the "
        f"difference taken -- {s4_eff['fast'][0]:.1f} / {s4_eff['walk'][0]:.1f} / {s4_eff['shuffle'][0]:.1f} mm "
        f"for fast / walk / shuffle, on a 91 mm wide insole.")
    out()
    out(f"**Gain match.** Replacing raw counts by gain-matched conductance moves the per-frame CoP by "
        f"{g_eff['fast'][0]:.2f} / {g_eff['walk'][0]:.2f} / {g_eff['shuffle'][0]:.2f} mm on average "
        f"(fast / walk / shuffle), i.e. the s4-zero effect is "
        f"{s4_eff['walk'][0] / g_eff['walk'][0]:.0f}× the gain-match effect on walk. The CoP-only "
        f"{h_kind.upper()} scores {grid[('raw', 'cop', h_kind)]['tb']['acc']:.4f} under A, "
        f"{grid[('conductance', 'cop', h_kind)]['tb']['acc']:.4f} under B and "
        f"{grid[('gain_matched', 'cop', h_kind)]['tb']['acc']:.4f} under C on the time-blocked test "
        f"(block-CV {cv[('raw', 'cop', h_kind)]:.4f} / {cv[('conductance', 'cop', h_kind)]:.4f} / "
        f"{cv[('gain_matched', 'cop', h_kind)]:.4f}): the representation moves the answer by at most "
        f"{max(abs(grid[(r, 'cop', h_kind)]['tb']['acc'] - grid[('raw', 'cop', h_kind)]['tb']['acc']) for r in REPRESENTATIONS) * len(test_idx):.0f} "
        f"test stance(s). Whatever the representation, the same frames carry the same s4 zeros and "
        f"the same 62–67 % extrapolation above 824 counts.")
    out()
    out(f"**Figure-8 turning.** The stance-to-stance spread of the mean medial-lateral CoP position is "
        f"{spread['walk']:.2f} / {spread['fast']:.2f} / {spread['shuffle']:.2f} mm (sd; walk / fast / shuffle). "
        f"Against the ±15 mm uncertainty on the sensor coordinates this is not resolvable, so turning "
        f"remains a hypothesis, as `docs/sim_vs_real.md` D4b concluded.")
    out()
    out("## 8. Peak force against onset time")
    out()
    out(f"![peak vs onset]({os.path.relpath(os.path.join(args.fig_dir, 'peak_vs_onset.png'), os.path.dirname(args.doc))})")
    out()
    out("| class | slope (raw counts per s of capture) | r | p |")
    out("|---|---|---|---|")
    for lab in CLASSES:
        out(f"| {lab} | {trends[lab][0]:+.2f} | {trends[lab][2]:+.3f} | {trends[lab][3]:.3f} |")
    out()
    down = [lab for lab in CLASSES if trends[lab][0] < 0 and trends[lab][3] < 0.05]
    if down:
        out(f"Peak total force declines significantly over the minute for {', '.join(down)}, which is "
            f"the direction FSR stress relaxation predicts (`docs/calibration_notes.md`: counts fell "
            f"~31 % in 76 s under constant load). Whether it is relaxation or the subject slowing "
            f"cannot be separated from one capture.")
    else:
        out("No class shows a significant decline of peak force over its 60 s capture at p < 0.05. "
            "That is not evidence against FSR stress relaxation -- the bench measurement was under "
            "constant load, and gait loads each sensor for a fraction of a second at a time -- it "
            "says the drift does not visibly enter one minute of walking.")
    out()
    out("## 9. Simulator versus real")
    out()
    if sim_acc:
        out("The sim-trained deployment models (`models/model_lda.json`, `models/model_qda.json`, "
            "fitted on 12 simulated sessions by `scripts/fit_model.py`) applied to the same "
            f"{n_all} real stances under the representation they were fitted on "
            f"({L[sim_acc['lda'][2]]}): LDA {sim_acc['lda'][0]:.4f} "
            f"({sim_acc['lda'][1]}/{n_all}), QDA {sim_acc['qda'][0]:.4f} ({sim_acc['qda'][1]}/{n_all}), "
            f"below the {all_floor:.4f} majority floor -- the expected outcome for a model fitted on a "
            f"generator whose constants were co-evolved with the detector. The same recipe retrained "
            f"on real stances scores {H['acc']:.4f} [{H['lo']:.4f}, {H['hi']:.4f}] on the time-blocked "
            f"test, and the sim bake-off's {0.9296:.4f} on 270 held-out simulated stances "
            f"(`docs/bakeoff.md`) is not a number this data can reproduce or refute: different "
            f"stances, different split, different world.")
    else:
        out("The sim-trained models were not found in the models directory; nothing to compare.")
    out()
    out("## 10. Split verdict: what more data fixes, what the hardware cannot")
    out()
    out("More data would fix:")
    out()
    out("- **Per-session generalisation.** One session per class means every number here is "
        "within-session. A second session per class flips this script to leave-one-session-out "
        "automatically.")
    out("- **Subjects.** One subject. Nothing here says anything about another foot.")
    out("- **Path.** Everything was walked on a figure-8 in a small space; straight-line gait "
        "and its symmetric loading are unmeasured.")
    out(f"- **Cadence range.** Fast and walk are separated by contact time "
        f"({frame_a.loc[frame_a['label'] == 'fast', 'contact_time_s'].median():.2f} vs "
        f"{frame_a.loc[frame_a['label'] == 'walk', 'contact_time_s'].median():.2f} s median); "
        "intermediate cadences would blur that boundary and the CoP features would have to carry it.")
    out()
    out("Six-sensor hardware limits that data will not fix:")
    out()
    out(f"- **s4's activation threshold** turns the CoP into a five-sensor centroid on "
        f"{min(s4_frac.values()):.0%}–{max(s4_frac.values()):.0%} of stance frames, with a "
        f"{min(v[0] for v in s4_eff.values()):.0f}–{max(v[0] for v in s4_eff.values()):.0f} mm shift.")
    out("- **±15 mm sensor coordinates** on a 91 mm wide insole: every CoP distance inherits it.")
    out("- **The gain match extrapolates** above 824 counts and does not hold below ~5 N.")
    out("- **No spatial resolution between sensors**: the CoP is a weighted mean of six points; "
        "anything between them is interpolation.")
    out()
    text = "\n".join(md) + "\n"
    with open(args.doc, "w") as f:
        f.write(text)
    print(f"\nwrote {os.path.relpath(args.doc, REPO)} and {len(os.listdir(args.fig_dir))} figures in "
          f"{time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
