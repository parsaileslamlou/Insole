"""Model bake-off on the session-disjoint insole split. Console output only.

FEATURE SUBSET -- cop_path_len and cop_displacement ONLY.
    The other five stance features (contact_time_s, impulse_counts_s,
    time_to_peak_s, loading_rate_cps, peak_counts) are all cadence one
    division removed, and the three classes are DEFINED by changing the
    stride period in make_sessions.py (walk 1.0s, fast 0.6s, shuffle 0.5s).
    Including them lets every model score 1.000, so the comparison measures
    nothing but the label-generating process. The two centre-of-pressure
    features are the only ones carrying spatial rather than temporal
    information, so they are the only ones on which a bake-off asks a real
    question.

SPLIT -- session-disjoint, holding out session _03 of each class.
    Row-wise shuffling would be WRONG here. Consecutive stances inside one
    session are near-duplicates: same simulated gait, same seed, same
    cadence, one stride apart. A random row split puts near-copies of a test
    stride into the training set, so the test score measures memorised
    session identity rather than generalisation to an unseen session. Only a
    session-disjoint split asks the question we care about.

Every number this script asserts, it prints first.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SKLDA,
    QuadraticDiscriminantAnalysis as SKQDA,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from discriminant import (
    fit_lda, fit_qda, predict, accuracy_ci,
    _log_discriminants, IllConditionedCovarianceWarning, RANK_TOL,
)
from features import SENSOR_COLS, find_stances, merge_close, extract_features
from make_sessions import CLASSES, SESSIONS_PER_CLASS, session_name

FEATURES = ["cop_path_len", "cop_displacement"]
TEST_SESSION_IDX = 3            # hold out the last session of every class
FRAME_CSV = "features_sessions.csv"

# Figures carried over from an earlier session, to be confirmed or corrected
# out loud rather than silently replaced.
PRIOR_N_TEST = 270
PRIOR_POOLED_COND = 3.39

RULE = "=" * 78


def head(t):
    print("\n" + RULE + "\n" + t + "\n" + RULE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_frame():
    """Build the feature frame with features.py. No notebook cells involved."""
    frames = []
    for label in CLASSES:
        for i in range(SESSIONS_PER_CLASS):
            stem = session_name(label, i)[:-4]
            if not os.path.exists(stem + ".csv"):
                if not os.path.exists(stem + ".txt"):
                    subprocess.run([sys.executable, "make_sessions.py"],
                                   check=True, stdout=subprocess.DEVNULL)
                subprocess.run([sys.executable, "read_serial.py",
                                stem + ".txt", stem + ".csv"],
                               check=True, stdout=subprocess.DEVNULL)
            df = pd.read_csv(stem + ".csv")
            total = df[SENSOR_COLS].sum(axis=1).to_numpy()
            feats = extract_features(df, merge_close(find_stances(total)), label)
            feats["session"] = stem
            frames.append(feats)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(FRAME_CSV, index=False)
    return out


head("DATA")
if os.path.exists(FRAME_CSV):
    frame = pd.read_csv(FRAME_CSV)
    print("loaded " + FRAME_CSV + " (built by features.py)")
else:
    frame = build_frame()
    print("rebuilt " + FRAME_CSV + " via features.py + read_serial.py CLI")

n_rows = len(frame)
n_sessions = frame["session"].nunique()
print(f"rows                 : {n_rows}")
print(f"sessions             : {n_sessions}")
print(f"feature columns used : {FEATURES}  (p = {len(FEATURES)})")
assert n_rows == 1123, n_rows
assert n_sessions == 12, n_sessions
assert len(FEATURES) == 2

print("\nper-session row counts and class composition:")
print(f"  {'session':16s} {'rows':>5s}  label")
for s in sorted(frame["session"].unique()):
    sub = frame[frame["session"] == s]
    labs = sorted(sub["label"].unique())
    print(f"  {s:16s} {len(sub):5d}  {','.join(labs)}")
    assert len(labs) == 1, s + " mixes labels"

print("\nclass totals over all 12 sessions:")
for lab, n in frame["label"].value_counts().sort_index().items():
    print(f"  {lab:8s} {n:4d}")


# ---------------------------------------------------------------------------
# Session-disjoint split
# ---------------------------------------------------------------------------
head("SPLIT (session-disjoint)")
test_sessions = sorted(session_name(lab, TEST_SESSION_IDX)[:-4] for lab in CLASSES)
is_test = frame["session"].isin(test_sessions)
train = frame[~is_test]
test = frame[is_test]

print(f"held-out sessions    : {test_sessions}")
print(f"train sessions       : {sorted(set(frame['session']) - set(test_sessions))}")
print(f"n_train              : {len(train)}")
print(f"n_test               : {len(test)}")
print(f"n_train + n_test     : {len(train) + len(test)}  (rows: {n_rows})")
assert len(train) + len(test) == n_rows
assert not (set(train["session"]) & set(test["session"])), "sessions leak across split"
print("session disjointness : OK (no session appears on both sides)")

print(f"\nprior figure for n_test : {PRIOR_N_TEST}")
print(f"n_test computed here    : {len(test)}")
if len(test) == PRIOR_N_TEST:
    print(f"VERDICT: CONFIRMED -- the earlier {PRIOR_N_TEST}-test-row figure is "
          f"reproduced exactly by this split.")
else:
    print(f"VERDICT: CORRECTED -- earlier figure {PRIOR_N_TEST} does not hold for "
          f"this split; correct value is {len(test)} "
          f"(difference {len(test) - PRIOR_N_TEST:+d}).")

print("\ntest-set class balance:")
test_counts = test["label"].value_counts().sort_index()
for lab, n in test_counts.items():
    print(f"  {lab:8s} {n:4d}   {n / len(test):.4f}")
print("\ntrain-set class balance:")
for lab, n in train["label"].value_counts().sort_index().items():
    print(f"  {lab:8s} {n:4d}   {n / len(train):.4f}")

Xtr = train[FEATURES].to_numpy(float)
ytr = train["label"].to_numpy()
Xte = test[FEATURES].to_numpy(float)
yte = test["label"].to_numpy()
print(f"\nXtrain {Xtr.shape}   Xtest {Xte.shape}")


# ---------------------------------------------------------------------------
# Baselines and models
# ---------------------------------------------------------------------------
head("BASELINES AND MODELS (same split throughout)")
majority_label = test_counts.idxmax()
majority_rate = test_counts.max() / len(test)
print(f"MAJORITY-CLASS FLOOR (test set): always predict '{majority_label}'")
print(f"  accuracy = {majority_rate:.4f}   ({test_counts.max()} / {len(test)})")
print("  every number below is read against this floor.\n")

train_majority = train["label"].value_counts().idxmax()
train_major_rate = float((yte == train_majority).mean())
print(f"train-fitted majority baseline: always predict '{train_majority}' "
      f"(most common in TRAIN)")
print(f"  accuracy on test = {train_major_rate:.4f}\n")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m_lda = fit_lda(Xtr, ytr)
    m_qda = fit_qda(Xtr, ytr)
rank_warnings = [w for w in caught
                 if issubclass(w.category, IllConditionedCovarianceWarning)]

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
logreg.fit(Xtr, ytr)

results = {}
for name, pred in [
    ("LogisticRegression (scaled)", logreg.predict(Xte)),
    ("my LDA", predict(m_lda, Xte)),
    ("my QDA", predict(m_qda, Xte)),
]:
    acc, lo, hi, se = accuracy_ci(yte, pred)
    results[name] = dict(pred=pred, acc=acc, lo=lo, hi=hi, se=se)
    print(f"  {name:29s} test accuracy = {acc:.4f}   "
          f"({int((pred == yte).sum())} / {len(yte)})   "
          f"lift over floor = {acc - majority_rate:+.4f}")

labs = list(test_counts.index)
print("\nper-class recall on the test set:")
print(f"  {'model':29s} " + "  ".join(f"{l:>9s}" for l in labs))
for name, r in results.items():
    cells = []
    for lab in labs:
        mask = yte == lab
        cells.append(f"{(r['pred'][mask] == lab).mean():9.4f}")
    print(f"  {name:29s} " + "  ".join(cells))

print(f"\nconfusion matrices (rows = true, cols = predicted, order {labs}):")
for name, r in results.items():
    print("  " + name)
    for t in labs:
        row = [int(((yte == t) & (r["pred"] == q)).sum()) for q in labs]
        print(f"    {t:8s} " + " ".join(f"{v:5d}" for v in row))


# ---------------------------------------------------------------------------
# sklearn cross-check on the same subset
# ---------------------------------------------------------------------------
head("SKLEARN CROSS-CHECK (mean-centered log-discriminants)")
print("Posteriors are NOT compared: on separable data they saturate to 0/1 and")
print("the difference is vacuous. Mean-centering removes the class-independent")
print("offset that cancels in the argmax anyway.\n")


def centered(a):
    return a - a.mean(axis=1, keepdims=True)


sk_lda = SKLDA(solver="svd").fit(Xtr, ytr)
print(f"class order mine {[str(v) for v in m_lda['classes']]} "
      f"vs sklearn {[str(v) for v in sk_lda.classes_]}")
assert list(sk_lda.classes_) == list(m_lda["classes"])
d_lda = float(np.abs(centered(_log_discriminants(m_lda, Xte))
                     - centered(sk_lda.decision_function(Xte))).max())
print(f"LDA  max|d centered log-discriminant| = {d_lda:.6e}")
print(f"LDA  label agreement on test          = "
      f"{(predict(m_lda, Xte) == sk_lda.predict(Xte)).mean():.6f}")

print()
default_tol = SKQDA().tol
try:
    sk_qda = SKQDA().fit(Xtr, ytr)
    print(f"QDA  sklearn FITS at its default tol ({default_tol:g}) -- no loosening needed.")
except np.linalg.LinAlgError as e:
    print(f"QDA  sklearn REFUSES to fit at its default tol ({default_tol:g}):")
    print(f"       {e}")
    print("     That refusal is the result. Retrying below with an EXPLICITLY")
    print("     loosened tol=1e-12 purely to obtain a comparison number; this is")
    print("     not sklearn's default behaviour and is labelled as such.")
    sk_qda = SKQDA(tol=1e-12).fit(Xtr, ytr)

assert list(sk_qda.classes_) == list(m_qda["classes"])
d_qda = float(np.abs(centered(_log_discriminants(m_qda, Xte))
                     - centered(sk_qda.decision_function(Xte))).max())
print(f"QDA  max|d centered log-discriminant| = {d_qda:.6e}")
print(f"QDA  label agreement on test          = "
      f"{(predict(m_qda, Xte) == sk_qda.predict(Xte)).mean():.6f}")


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
head("CONDITIONING (2-feature subset)")
mu = Xtr.mean(0)
sd = Xtr.std(0)
Xtr_z = (Xtr - mu) / sd
m_lda_z = fit_lda(Xtr_z, ytr)
with warnings.catch_warnings(record=True) as caught_z:
    warnings.simplefilter("always")
    m_qda_z = fit_qda(Xtr_z, ytr)
rank_warnings_z = [w for w in caught_z
                   if issubclass(w.category, IllConditionedCovarianceWarning)]

cond_pooled_raw = float(np.linalg.cond(m_lda["cov"]))
cond_pooled_z = float(np.linalg.cond(m_lda_z["cov"]))
print(f"{'matrix':28s} {'raw':>14s} {'z-scored':>14s}")
print(f"{'pooled (train split)':28s} {cond_pooled_raw:14.6g} {cond_pooled_z:14.6g}")
for i, c in enumerate(m_qda["classes"]):
    print(f"{'per-class ' + str(c):28s} "
          f"{float(np.linalg.cond(m_qda['covs'][i])):14.6g} "
          f"{float(np.linalg.cond(m_qda_z['covs'][i])):14.6g}")

Xall = frame[FEATURES].to_numpy(float)
yall = frame["label"].to_numpy()
cond_pooled_all = float(np.linalg.cond(fit_lda(Xall, yall)["cov"]))
print(f"{'pooled (all 1123 rows)':28s} {cond_pooled_all:14.6g}")

print(f"\nprior figure for pooled cond : {PRIOR_POOLED_COND}")
print(f"pooled cond, train split     : {cond_pooled_raw:.6g}")
print(f"pooled cond, all 1123 rows   : {cond_pooled_all:.6g}")
match = [n for n, v in [("train-split", cond_pooled_raw),
                        ("all-rows", cond_pooled_all)]
         if abs(v - PRIOR_POOLED_COND) < 0.005]
if match:
    print(f"VERDICT: CONFIRMED -- {PRIOR_POOLED_COND} matches the {match[0]} "
          f"figure to 2 dp.")
else:
    print(f"VERDICT: CORRECTED -- {PRIOR_POOLED_COND} matches neither figure "
          f"({cond_pooled_raw:.4f} train-split, {cond_pooled_all:.4f} all-rows).")

print(f"\nrank warning (RANK_TOL = {RANK_TOL:g}) on this subset:")
print(f"  raw      : {len(rank_warnings)} warning(s)")
for w in rank_warnings:
    print(f"    {w.message}")
print(f"  z-scored : {len(rank_warnings_z)} warning(s)")
for w in rank_warnings_z:
    print(f"    {w.message}")
if not rank_warnings and not rank_warnings_z:
    print("  DOES NOT FIRE on this subset. Both cop features are O(0.01-1), so no")
    print("  covariance eigenvalue falls under the absolute 1e-4 floor -- unlike")
    print("  the full 7-feature set, where peak_counts and loading_rate_cps span")
    print("  1e5 and drag two eigenvalues per class below it.")


# ---------------------------------------------------------------------------
# McNemar, LDA vs QDA
# ---------------------------------------------------------------------------
head("MCNEMAR: my LDA vs my QDA (test set)")
ok_lda = results["my LDA"]["pred"] == yte
ok_qda = results["my QDA"]["pred"] == yte
a = int((ok_lda & ok_qda).sum())
b = int((ok_lda & ~ok_qda).sum())
c = int((~ok_lda & ok_qda).sum())
d = int((~ok_lda & ~ok_qda).sum())

print("contingency table (test set):")
print(f"  {'':22s} {'QDA correct':>12s} {'QDA wrong':>12s}")
print(f"  {'LDA correct':22s} {a:12d} {b:12d}")
print(f"  {'LDA wrong':22s} {c:12d} {d:12d}")
print(f"  total = {a + b + c + d}  (n_test = {len(yte)})")
assert a + b + c + d == len(yte)

print("\ndiscordant pairs:")
print(f"  b (LDA right, QDA wrong) = {b}")
print(f"  c (LDA wrong, QDA right) = {c}")
print(f"  b + c                    = {b + c}")

if b + c == 0:
    print("\nNO POWER: b + c = 0. The two models agree on every single test row,")
    print("so McNemar's test is undefined -- there is no discordant evidence to")
    print("weigh in either direction. No p-value is reportable.")
else:
    pvalue = float(stats.binomtest(b, b + c, 0.5).pvalue)
    min_p = min(1.0, 2 * 0.5 ** (b + c))
    print(f"\nexact binomial (two-sided) p-value  = {pvalue:.6g}")
    print(f"smallest p attainable at b + c = {b + c} is {min_p:.6g}")
    if min_p > 0.05:
        print(f"NO POWER: with only {b + c} discordant pair(s), even the most extreme")
        print(f"possible split cannot reach p <= 0.05 (the floor is {min_p:.4g}).")
        print("The p-value above is uninformative on its own.")
    elif b + c < 25:
        print(f"LOW POWER: b + c = {b + c} < 25, so the exact test is used instead of")
        print("the chi-square approximation, and the estimate remains fragile.")
    else:
        print(f"b + c = {b + c} >= 25: the normal approximation would also apply.")


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------
head("INTERVALS")
print("accuracy_ci is an unclipped Wald interval, used exactly as written.\n")
print(f"{'model':29s} {'acc':>8s} {'n':>6s} {'se':>10s}   95% CI")
loud = []
for name, r in results.items():
    print(f"{name:29s} {r['acc']:8.4f} {len(yte):6d} {r['se']:10.6f}   "
          f"[{r['lo']:.6f}, {r['hi']:.6f}]")
    if r["hi"] - r["lo"] == 0:
        loud.append(f"{name}: ZERO-WIDTH interval [{r['lo']}, {r['hi']}] "
                    f"(acc = {r['acc']}, se = {r['se']})")
    if r["lo"] < 0.0:
        loud.append(f"{name}: lower bound {r['lo']:.6f} lies below 0")
    if r["hi"] > 1.0:
        loud.append(f"{name}: upper bound {r['hi']:.6f} lies above 1")

print("\nstride-level vs session-level standard error")
print("(stride-level treats all test strides as independent; they are not --")
print(" strides within a session are near-duplicates, which is exactly what the")
print(" session-disjoint split exists to respect)\n")
n_test_sessions = len(test_sessions)
print(f"number of test sessions = {n_test_sessions}")
print(f"{'model':29s} {'se_stride':>11s} {'se_session':>11s} {'ratio':>7s}   per-session acc")
for name, r in results.items():
    per_sess = []
    for s in test_sessions:
        m = test["session"].to_numpy() == s
        per_sess.append(float((r["pred"][m] == yte[m]).mean()))
    per_sess = np.array(per_sess)
    se_sess = float(per_sess.std(ddof=1) / np.sqrt(n_test_sessions))
    ratio = se_sess / r["se"] if r["se"] > 0 else float("inf")
    print(f"{name:29s} {r['se']:11.6f} {se_sess:11.6f} {ratio:7.2f}   "
          f"{np.array2string(per_sess, precision=4)}")
    if r["se"] == 0:
        loud.append(f"{name}: stride-level se is 0, so the se ratio is undefined")

print(f"\nsession-level n is {n_test_sessions}, not {len(yte)}. A standard error built on")
print(f"{n_test_sessions} points carries ~{n_test_sessions - 1} degrees of freedom; z = 1.96 is the wrong")
print("multiplier for it and it must not be read as a 95% interval.")

if loud:
    print("\n" + "!" * 78)
    for line in loud:
        print("!! " + line)
    print("!" * 78)
else:
    print("\nAll intervals finite, non-degenerate, and inside [0, 1].")

print("\n" + RULE + "\nEND\n" + RULE)
