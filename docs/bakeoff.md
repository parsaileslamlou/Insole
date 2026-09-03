# Bake-off: CoP-only gait classification

From `scripts/bakeoff.py`. Every number here is in that script's output;
regenerate with `python scripts/bakeoff.py` (delete
`data/sim/features_sessions.csv` first to rebuild the frame from the sim
sessions). Numbers are under the measured 274 × 91 mm geometry and under the
shipped feature representation, **B, conductance** (`insole/representations.py`,
chosen at stage 20 on the real captures, `docs/real_results.md`): the detector
runs on raw counts, the features on x = counts / (4095 − counts). The figures
this file carried under raw counts, and under the placeholder geometry before
that, are listed at the end for the record.

## What I compared

My LDA and QDA (`insole/discriminant.py`), sklearn's `LogisticRegression`, and
sklearn's own LDA/QDA, on one session-disjoint split: 1123 stances, 12
sessions, holding out `_03` of each class. n_train 853, n_test 270.

Two features only, `cop_path_len` and `cop_displacement`. I dropped
`contact_time_s`, `impulse_counts_s`, `time_to_peak_s`, `loading_rate_cps` and
`peak_counts` — each is cadence one division removed, and the classes are
*defined* by changing stride period in `insole/make_sessions.py`. On the full 7
the comparison measures the label-generating process, not the models.
**TODO:** `scripts/bakeoff.py` never fits the full set, so that figure is absent
here; on the real captures `scripts/train_real.py` fits both sets.

## Against the floor

Majority-class floor on test: **0.4296** (116/270).

| model | accuracy | correct | Wilson 95% CI |
|---|---|---|---|
| LogisticRegression | 0.9185 | 248/270 | [0.8797, 0.9456] |
| my LDA | 0.9185 | 248/270 | [0.8797, 0.9456] |
| my QDA | 0.9296 | 251/270 | [0.8927, 0.9545] |

Mine track sklearn's to 1.598721e-14 (LDA) and 3.907985e-14 (QDA) in
mean-centered log-discriminants.

Per-class recall (fast / shuffle / walk): LR 0.8854 / 0.9310 / 0.9483, LDA
0.9479 / 0.9224 / 0.8621, QDA 0.9479 / 0.9224 / 0.9138. LDA and QDA lose their
errors on walk → fast (8 and 5 rows) and shuffle → fast (6 each); LR spreads
them more evenly.

## The errors are not the same error

An earlier version of this file said all three models made the same mistake,
"9 `fast` rows into `walk`". That was a count, not a row list, and it was
wrong twice over. `scripts/bakeoff.py` prints the misclassified row indices:

| model | wrong | fast → walk | rows |
|---|---|---|---|
| LogisticRegression | 22 | 7 | 73, 75, 81, 92, 109, 120, 144 |
| my LDA | 22 | 5 | 73, 75, 92, 95, 109 |
| my QDA | 19 | 5 | 73, 75, 92, 95, 109 |

Four fast → walk rows are common to all three models; eight appear in at
least one. Across all errors, 16 rows are common to the three models and 28
appear in at least one. (Under raw counts the counts were 7 / 3 / 6 with 3
common and 7 in the union; under the old geometry 9/9/9 with intersection 8,
union 10.) `fast` and `walk` are the same `gait_gen` mode at different
`cycle_s` and `shuffle` is a different mode, so some fast/walk confusion is
expected once cadence is stripped — but it is not a fixed set of rows, and it
moved again when the representation changed. It is a property of the feature
space, not a stable structural confusion.

## McNemar: undecidable

b = 0, c = 3. Three discordant pairs, all in QDA's favour, exact two-sided
p = 0.25 — which is also the smallest p attainable at b + c = 3, so the test
has no power here. QDA's margin over LDA is three stances. Not a win.

## Two standard errors

| model | se_stride (n=270) | se_session (n=3) | ratio | per-session acc |
|---|---|---|---|---|
| LogisticRegression | 0.016649 | 0.018752 | 1.13 | 0.8854, 0.9310, 0.9483 |
| my LDA | 0.016649 | 0.025453 | 1.53 | 0.9479, 0.9224, 0.8621 |
| my QDA | 0.015566 | 0.010245 | 0.66 | 0.9479, 0.9224, 0.9138 |

Stride-level SE treats 270 strides as independent; they are not. Session-level
SE uses n = 3, carries ~2 degrees of freedom, and I never turn it into an
interval — z = 1.96 is the wrong multiplier. `se_stride` is the Wald standard
error; the intervals in the table above are Wilson (`accuracy_ci` used to be
an unclipped Wald interval, which had zero width at acc = 1 and could
exceed 1).

## Conditioning

Pooled within-class covariance condition number on the two features: 3.78
(train split), 4.29 (all 1123 rows); z-scoring brings the train figure to
3.17. The 3.39 an earlier session quoted matches neither and the script says
so. No covariance eigenvalue falls under the 1e-4 rank floor on this subset.

## Caveat

All of this is simulated data. `gait_gen`'s constants, the detector
thresholds, and the tests checking them were co-evolved, so the result is
internally consistent by construction. The real `_02` captures are the first
non-circular test of any of it: on them the sim-trained recipe scores below
the majority floor (`scripts/sim_vs_real.py`, D2), and the same recipe
retrained on real stances is in `docs/real_results.md`.

## Superseded figures

**Raw counts (representation A), same geometry, before stage 20:** LR 0.9037
(244/270), LDA 0.9296 (251/270), QDA 0.9259 (250/270); fast → walk rows
7 / 3 / 6 with 3 common and 7 in the union; McNemar b = 4, c = 3, p = 1;
se_session 0.032354 / 0.004370 / 0.012197. The conductance transform is
nonlinear, so the CoP features moved when the representation did;
`tests/test_train_real.py` still pins the raw-count LDA at 251/270.

**Placeholder geometry, before commit 4e7d34f:** LR 0.9185 (248/270), LDA
0.9333 (252/270), QDA 0.9370 (253/270); McNemar b = 0, c = 1; se_session
0.038043 / 0.014622 / 0.017185. Same 1123 stances, same split; only the two
CoP features moved (`docs/sim_vs_real.md`, D1).
