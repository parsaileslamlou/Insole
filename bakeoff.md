# Bake-off: CoP-only gait classification

From `bakeoff.py`. Every number here is in that script's output.

## What I compared

My LDA and QDA (`discriminant.py`), sklearn's `LogisticRegression`, and
sklearn's own LDA/QDA, on one session-disjoint split: 1123 stances, 12
sessions, holding out `_03` of each class. n_train 853, n_test 270.

Two features only, `cop_path_len` and `cop_displacement`. I dropped
`contact_time_s`, `impulse_counts_s`, `time_to_peak_s`, `loading_rate_cps` and
`peak_counts` — each is cadence one division removed, and the classes are
*defined* by changing stride period in `make_sessions.py`. On the full 7 the
comparison measures the label-generating process, not the models.
**TODO:** `bakeoff.py` never fits the full set, so that figure is absent here.

## Against the floor

Majority-class floor on test: **0.4296** (116/270).

| model | accuracy | correct |
|---|---|---|
| LogisticRegression | 0.9185 | 248/270 |
| my LDA | 0.9333 | 252/270 |
| my QDA | 0.9370 | 253/270 |

Mine track sklearn's to 3.375078e-14 (LDA) and 1.136868e-13 (QDA) in
mean-centered log-discriminants; label agreement 1.000000.

## The result: all three make the same mistake

Every confusion matrix puts **9 `fast` rows into `walk`**. `fast` and `walk`
are the same `gait_gen` mode at different `cycle_s`; `shuffle` is a different
mode. Strip cadence and the two walk-mode classes collapse toward each other
while `shuffle` stays separated — it has the highest recall of any class for
all three models.

That is the finding: the CoP features measure geometry, not period. Were they
leaking period, `fast` and `walk` would separate. The ranking is not the result.

## McNemar: undecidable

b = 0, c = 1. One discordant pair, p = 1 — and 1 is the *smallest* value
attainable at b + c = 1, so the test cannot reject at any threshold. QDA's
margin is one stance. Not a win.

## Two standard errors

| model | se_stride (n=270) | se_session (n=3) | ratio |
|---|---|---|---|
| LogisticRegression | 0.016649 | 0.038043 | 2.28 |
| my LDA | 0.015181 | 0.014622 | 0.96 |
| my QDA | 0.014782 | 0.017185 | 1.16 |

Stride-level SE treats 270 strides as independent; they are not. Session-level
SE uses n = 3, carries ~2 degrees of freedom, and I never turn it into an
interval — z = 1.96 is the wrong multiplier.

## Caveat

All of this is simulated data. `gait_gen`'s constants, the detector thresholds,
and the tests checking them were co-evolved, so the result is internally
consistent by construction. Prompt 13 real data is the first non-circular test
of any of it.
