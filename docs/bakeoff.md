# Bake-off: CoP-only gait classification

From `scripts/bakeoff.py`. Every number here is in that script's output; regenerate
with `python scripts/bakeoff.py` (delete `data/sim/features_sessions.csv` first to rebuild the
frame from the sim sessions). Numbers are under the measured 274 × 91 mm
geometry; the pre-geometry figures this file used to carry are listed at the
end for the record.

## What I compared

My LDA and QDA (`insole/discriminant.py`), sklearn's `LogisticRegression`, and
sklearn's own LDA/QDA, on one session-disjoint split: 1123 stances, 12
sessions, holding out `_03` of each class. n_train 853, n_test 270.

Two features only, `cop_path_len` and `cop_displacement`. I dropped
`contact_time_s`, `impulse_counts_s`, `time_to_peak_s`, `loading_rate_cps` and
`peak_counts` — each is cadence one division removed, and the classes are
*defined* by changing stride period in `insole/make_sessions.py`. On the full 7 the
comparison measures the label-generating process, not the models.
**TODO:** `scripts/bakeoff.py` never fits the full set, so that figure is absent here.

## Against the floor

Majority-class floor on test: **0.4296** (116/270).

| model | accuracy | correct | Wilson 95% CI |
|---|---|---|---|
| LogisticRegression | 0.9037 | 244/270 | [0.8626, 0.9334] |
| my LDA | 0.9296 | 251/270 | [0.8927, 0.9545] |
| my QDA | 0.9259 | 250/270 | [0.8884, 0.9515] |

Mine track sklearn's to 6.039613e-14 (LDA) and 5.684342e-14 (QDA) in
mean-centered log-discriminants; label agreement 1.000000.

## The errors are not the same error

An earlier version of this file said all three models made the same mistake,
"9 `fast` rows into `walk`". That was a count, not a row list, and it was
wrong twice over. `scripts/bakeoff.py` now prints the misclassified row indices:

| model | wrong | fast → walk | rows |
|---|---|---|---|
| LogisticRegression | 26 | 7 | 73, 92, 109, 135, 136, 144, 149 |
| my LDA | 19 | 3 | 73, 109, 149 |
| my QDA | 20 | 6 | 73, 92, 109, 135, 136, 149 |

Three fast → walk rows are common to all three models; seven appear in at
least one. (Under the old geometry the counts were 9/9/9 but the rows were
not identical either: intersection 8, union 10.) `fast` and `walk` are the
same `gait_gen` mode at different `cycle_s` and `shuffle` is a different mode,
so some fast/walk confusion is expected once cadence is stripped — but it is
not a fixed set of rows, and it shrank when the coordinates moved. It is a
property of the feature space, not a stable structural confusion.

What does hold: `shuffle` recall is the highest or joint-highest for every
model (0.931 / 0.922 / 0.931), and the CoP features measure geometry, not
period. Were they leaking period, `fast` and `walk` would separate cleanly.

## McNemar: undecidable

b = 4, c = 3. Seven discordant pairs, exact two-sided p = 1; the smallest p
attainable at b + c = 7 is 0.0156, so the test has power in principle but the
split is as even as it can be. LDA's margin over QDA is one stance. Not a win.

## Two standard errors

| model | se_stride (n=270) | se_session (n=3) | ratio |
|---|---|---|---|
| LogisticRegression | 0.017953 | 0.032354 | 1.80 |
| my LDA | 0.015566 | 0.004370 | 0.28 |
| my QDA | 0.015938 | 0.012197 | 0.77 |

Stride-level SE treats 270 strides as independent; they are not. Session-level
SE uses n = 3, carries ~2 degrees of freedom, and I never turn it into an
interval — z = 1.96 is the wrong multiplier. `se_stride` is the Wald standard
error; the intervals in the table above are Wilson (`accuracy_ci` used to be
an unclipped Wald interval, which had zero width at acc = 1 and could
exceed 1).

## Caveat

All of this is simulated data. `gait_gen`'s constants, the detector thresholds,
and the tests checking them were co-evolved, so the result is internally
consistent by construction. The real `_02` captures are the first non-circular
test of any of it, and on them this recipe scores below the majority floor
(`scripts/sim_vs_real.py`, D2).

## Superseded figures

Under the placeholder geometry that preceded commit 4e7d34f: LR 0.9185
(248/270), LDA 0.9333 (252/270), QDA 0.9370 (253/270); McNemar b = 0, c = 1;
se_session 0.038043 / 0.014622 / 0.017185. Same 1123 stances, same split; only
the two CoP features moved (`docs/sim_vs_real.md`, D1).
