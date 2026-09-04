# Real-data results

Every number in this file is produced by `python scripts/train_real.py`; regenerate it, do not edit it (the persisted models' meta records the git hash of the run). Figures: `figures\real_results/`. Models: `models\model_lda_real.json`, `models\model_qda_real.json`, `models\model_lda_real_raw.json`, `models\model_qda_real_raw.json`.

## 1. Data

Every training-grade capture in `data/real/` (`data/real/README.md`): 2 sets (`_02`, `_03`), one 60 s session per activity in each, 100 Hz, tethered USB, one subject, walking a figure-8. `_01` is failure evidence and is never trained or evaluated on. Segmentation uses `insole/detector.py` at the committed thresholds (T_ON=1200, T_OFF=450, MIN_DURATION=15, MAX_DURATION=200, GAP_MERGE=12) on raw counts, and every file's stance count is pinned twice, in `SESSIONS` here and in `tests/test_stances.py`:

| activity | file | frames | stances kept |
|---|---|---|---|
| stand | `stand_02.csv` | 6000 | 0 |
| walk | `walk02.csv` | 6000 | 35 |
| fast | `fast02.csv` | 6000 | 48 |
| shuffle | `shuffle02.csv` | 6000 | 30 |
| stand | `stand_03.csv` | 6001 | 0 |
| walk | `walk_03.csv` | 6001 | 32 |
| fast | `fast_03.csv` | 6001 | 45 |
| shuffle | `shuffle_03.csv` | 6001 | 34 |

Each standing capture is one unbroken contact the length of the file, rejected by MAX_DURATION, so it contributes no stances and is excluded from classification. n = 224 moving stances (fast 93, shuffle 64, walk 67; per session fast 48 + 45, shuffle 30 + 34, walk 35 + 32); the all-data majority floor is 0.4152 (93/224, `fast`).

## 2. Representations and feature sets

Features are `insole/features.py`'s extractors, unchanged, computed on three per-frame input representations (`insole/representations.py`); the detector always runs on raw counts:

- **A raw**: the six ADC counts as logged.
- **B conductance**: x = counts / (4095 − counts) per channel; x(0) = 0.
- **C gain-matched**: x · g, g from `models/gain_match.json` (s0=0.9900, s1=0.9616, s2=0.9741, s3=1.2513, s4=0.8692, s5=0.9538).

Force is linear in conductance (`insole/calibration.py`), so under B and C the centre of pressure is a force-proportional centroid and under A it is not. The gain match is a single-point relative match at ~12 N: above 824 counts (62–67 % of loaded walking frames of the `_02` set, `scripts/analyze_real.py` C3) it extrapolates, and below ~5 N the channels' activation thresholds differ, so it does not hold there.

**Shipped representation: B (conductance)** -- `insole.representations.SHIPPED`, the one `insole/infer_live.py` feeds on every source, `scripts/bakeoff.py` builds the sim frame under, and the persisted models are fitted on. It was chosen at stage 20 by the headline rule in section 4 on the `_02` set. On the current data the rule prefers A (raw) over B (conductance) by 2 stance(s) in 224 -- 0.9% -- (pooled leave-one-session-out accuracy, CoP-only, best of LDA/QDA: A 0.6071, B 0.5982, C 0.5982). **The rule's deciding margin is finer than the resolution of the data it is deciding on.** One stance either way is 0.4% of the test set, so the whole decision rests on 2 stances; the two cells' Wilson intervals, [0.5419, 0.6688] for A and [0.5329, 0.6602] for B, overlap over 93% of their length; and those intervals are themselves lower bounds on the uncertainty (section 3). **A and B are statistically indistinguishable on this data**, and which of them the rule ranks first at this sample size is arbitrary. **B remains the shipped representation and is FROZEN -- retained as a pre-existing freeze taken at stage 20 on the `_02` set, not as this rule's verdict.** The rule did not choose B here. It chose A, by a margin that means nothing, and the existing freeze was left standing because switching the shipped representation moves the sim bake-off frame, the sim-trained models and the streaming path together -- for a change this data cannot show to be an improvement. Any statement that the rule selected B is wrong. Both numbers are in section 4, and real models are now persisted under both representations, each labelled with its own (section 5). The simulator has no per-channel gain to correct, so under B every source is treated identically; the gain match still runs per frame for the extrapolation counter and never reaches the classifier (variant B, not C).

Two feature sets: **cop** = `cop_path_len`, `cop_displacement` (exactly the set `scripts/bakeoff.py` used, for comparability with the simulator), and **full** = all seven (`peak_counts`, `time_to_peak_s`, `contact_time_s`, `loading_rate_cps`, `impulse_counts_s` plus the two CoP features). The sim bake-off excluded the five timing/magnitude features because simulated fast and walk differ only in cadence, so any cadence feature reads the label off the generator. On real data cadence is measured, not constructed, so the full set is a legitimate classifier here, but it is not comparable with the sim number. Under B and C the count-valued features are in conductance units and keep their column names.

## 3. Split

**leave-one-session-out, 2 folds, pooled over the folds.** Every class has at least two sessions, so the script switched itself to leave-one-session-out: fold k holds out session k of every class and trains on the rest, so every stance is tested exactly once, out of its own session, and the headline pools the folds' predictions. Nothing in a test fold shares a session with anything in its training fold. It is still one subject, the same shoe, the same figure-8 path, and two sessions is the minimum that makes this split possible, not a comfortable margin: with two folds one odd session moves the number a lot.

| fold | held out | fast train / test | shuffle train / test | walk train / test | n_train | n_test |
|---|---|---|---|---|---|---|
| 0 | `fast02`, `shuffle02`, `walk02` | 45 / 48 | 34 / 30 | 32 / 35 | 111 | 113 |
| 1 | `fast_03`, `shuffle_03`, `walk_03` | 48 / 45 | 30 / 34 | 35 / 32 | 113 | 111 |

Pooled n_test = 224 (every stance once), majority floor = 0.4152 (93/224, always `fast`).

**The pooled interval is a LOWER BOUND on the uncertainty, not a confidence interval for a new session.** It is a Wilson interval computed as if the 224 pooled stances were 224 independent observations. They are not. They are 224 stances drawn from 2 sessions of 1 subject, and stances within one session share the subject, the day, the shoe, the sensor seating and the figure-8 path, so they are positively correlated; the effective number of independent observations is nearer the number of sessions than the number of stances. Treating correlated observations as independent understates the variance, so the true interval is WIDER than the one printed, by an amount this data cannot quantify: correcting it needs the between-session variance, and 2 sessions estimate that from 2 points (1 degree of freedom), which is not an estimate. The per-fold intervals are reported beside the pooled one for exactly this reason: they are the widest honest statement available here. Read the pooled interval as the floor of the uncertainty, never as its extent.

Reported beside it so the within-session optimism is visible: the **time-blocked within each session: first 60% of stances train, last 40% test, guard band 0**, pooled over sessions (fast 56/37, shuffle 38/26, walk 40/27), which was the headline recipe while there was one session per class and carries the leakage that implies (consecutive stances of one walk share the day, the sensor state and the path); and a **random stance-level split** with the same per-class sizes, 20 seeds (mean, min, max), which puts near-copies of every test stance into training and is expected to be the most optimistic of the three.

## 4. Results grid

Pooled leave-one-session-out accuracy with a Wilson 95 % interval and the count, then each fold's accuracy (fold order as in section 3), then the within-session time-blocked accuracy with its interval, then the random-split mean [min, max]. LDA/QDA are `insole/discriminant.py`; LR is sklearn's `LogisticRegression` on standardised features, as in `scripts/bakeoff.py`. A skipped cell says why.

| rep | features | model | leave-one-session-out acc [Wilson 95 %] | per fold | within-session time-blocked | random split |
|---|---|---|---|---|---|---|
| A | cop | lda | 0.5714 [0.5060, 0.6345] (128/224) | 0.6106 / 0.5315 | 0.6667 [0.5642, 0.7555] (60/90) | 0.6306 [0.5556, 0.7000] |
| A | cop | qda | 0.6071 [0.5419, 0.6688] (136/224) **(headline)** | 0.6195 / 0.5946 | 0.6667 [0.5642, 0.7555] (60/90) | 0.6272 [0.5444, 0.7000] |
| A | cop | lr | 0.5714 [0.5060, 0.6345] (128/224) | 0.6195 / 0.5225 | 0.6778 [0.5757, 0.7653] (61/90) | 0.6289 [0.5444, 0.7000] |
| A | full | lda | 0.7265 [0.6645, 0.7808] (162/223) (1 excl.) | 0.7965 / 0.6545 | 0.7667 [0.6695, 0.8420] (69/90) (1 excl.) | 0.7989 [0.7416, 0.8315] |
| A | full | qda | 0.7175 [0.6551, 0.7725] (160/223) (1 excl.) | 0.7168 / 0.7182 | 0.7889 [0.6937, 0.8605] (71/90) (1 excl.) | 0.8080 [0.7222, 0.8764] |
| A | full | lr | 0.7265 [0.6645, 0.7808] (162/223) (1 excl.) | 0.7699 / 0.6818 | 0.8000 [0.7059, 0.8696] (72/90) (1 excl.) | 0.8263 [0.7889, 0.8652] |
| B | cop | lda | 0.5670 [0.5015, 0.6302] (127/224) | 0.6018 / 0.5315 | 0.6556 [0.5528, 0.7455] (59/90) | 0.6239 [0.5222, 0.6889] |
| B | cop | qda | 0.5982 [0.5329, 0.6602] (134/224) | 0.6195 / 0.5766 | 0.6667 [0.5642, 0.7555] (60/90) | 0.6211 [0.5556, 0.6889] |
| B | cop | lr | 0.5714 [0.5060, 0.6345] (128/224) | 0.6195 / 0.5225 | 0.6778 [0.5757, 0.7653] (61/90) | 0.6244 [0.5333, 0.6889] |
| B | full | lda | 0.7399 [0.6786, 0.7931] (165/223) (1 excl.) | 0.8319 / 0.6455 | 0.8333 [0.7431, 0.8963] (75/90) (1 excl.) | 0.8118 [0.7556, 0.8652] |
| B | full | qda | 0.6457 [0.5810, 0.7056] (144/223) (1 excl.) | 0.6283 / 0.6636 | 0.8556 [0.7684, 0.9136] (77/90) (1 excl.) | 0.8074 [0.7444, 0.8652] |
| B | full | lr | 0.7578 [0.6976, 0.8094] (169/223) (1 excl.) | 0.8407 / 0.6727 | 0.8333 [0.7431, 0.8963] (75/90) (1 excl.) | 0.8324 [0.7889, 0.8764] |
| C | cop | lda | 0.5893 [0.5239, 0.6517] (132/224) | 0.6195 / 0.5586 | 0.6333 [0.5302, 0.7255] (57/90) | 0.6194 [0.5222, 0.6667] |
| C | cop | qda | 0.5982 [0.5329, 0.6602] (134/224) | 0.6372 / 0.5586 | 0.6778 [0.5757, 0.7653] (61/90) | 0.6217 [0.5333, 0.6667] |
| C | cop | lr | 0.5804 [0.5149, 0.6431] (130/224) | 0.6195 / 0.5405 | 0.6667 [0.5642, 0.7555] (60/90) | 0.6267 [0.5222, 0.6667] |
| C | full | lda | 0.7399 [0.6786, 0.7931] (165/223) (1 excl.) | 0.8142 / 0.6636 | 0.8222 [0.7306, 0.8875] (74/90) (1 excl.) | 0.8062 [0.7444, 0.8539] |
| C | full | qda | 0.6682 [0.6040, 0.7267] (149/223) (1 excl.) | 0.6460 / 0.6909 | 0.8333 [0.7431, 0.8963] (75/90) (1 excl.) | 0.8107 [0.7444, 0.8764] |
| C | full | lr | 0.7534 [0.6928, 0.8053] (168/223) (1 excl.) | 0.8319 / 0.6727 | 0.8333 [0.7431, 0.8963] (75/90) (1 excl.) | 0.8296 [0.7978, 0.8764] |

"excl." counts stances left out of that cell because a feature is not finite: `shuffle_03` stance at frame 0 (loading_rate_cps). `loading_rate_cps` is undefined when the peak is the first frame of the stance, which a capture that starts mid-contact produces; the stance stays in every cell whose features are finite.

### Headline

Rule, fixed before any result was seen: CoP-only features; among LDA and QDA under A, B and C, the cell with the best pooled leave-one-session-out accuracy; ties go to raw counts and to LDA. That is **A (raw), cop, QDA**: leave-one-session-out accuracy **0.6071** [0.5419, 0.6688] (136/224) against a majority floor of 0.4152; per fold 0.6195 [0.5274, 0.7037] (70/113) holding out `fast02`, `shuffle02`, `walk02`, 0.5946 [0.5016, 0.6813] (66/111) holding out `fast_03`, `shuffle_03`, `walk_03`; within-session time-blocked 0.6667 [0.5642, 0.7555] (60/90); random split 0.6272 [0.5444, 0.7000]. The selection metric and the reported metric are the same number here, picked among six cells, so the headline carries that much selection optimism. The gap between the within-session number and the leave-one-session-out number is what a session boundary costs on this data: +0.0595.

**The pooled interval is a LOWER BOUND on the uncertainty, not a confidence interval for a new session.** It is a Wilson interval computed as if the 224 pooled stances were 224 independent observations. They are not. They are 224 stances drawn from 2 sessions of 1 subject, and stances within one session share the subject, the day, the shoe, the sensor seating and the figure-8 path, so they are positively correlated; the effective number of independent observations is nearer the number of sessions than the number of stances. Treating correlated observations as independent understates the variance, so the true interval is WIDER than the one printed, by an amount this data cannot quantify: correcting it needs the between-session variance, and 2 sessions estimate that from 2 points (1 degree of freedom), which is not an estimate. The per-fold intervals are reported beside the pooled one for exactly this reason: they are the widest honest statement available here. Read the pooled interval as the floor of the uncertainty, never as its extent.

**Both representations are persisted, each labelled with its own.** `models/model_{lda,qda}_real.json` are fitted under **B (conductance)**, the shipped default; `models/model_{lda,qda}_real_raw.json` under **A (raw)**, the rule's headline representation. Every model JSON records a `representation` field and `insole/infer_live.py` reads it and applies the matching transform, refusing only a model whose representation it cannot honour (an absent field, an unknown name, or C with `--gain none`) -- there is no silent fallback. **The shipped default is therefore a recorded decision, not a consequence of what the loader would accept.** Under B the headline cell scores 0.5982 [0.5329, 0.6602] (134/224), 2 stance(s) apart from the A headline; that is the deployed model's number, and the two are indistinguishable on this data (section 2).

The best full-feature cell is B full lr at 0.7578 [0.6976, 0.8094] (169/223) (1 excl.) (within-session 0.8333 [0.7431, 0.8963] (75/90) (1 excl.)). It is the better classifier of these activities and it is reported here as such, but it rides on `contact_time_s` and its relatives, whose class medians on this data are fast 0.80 s, shuffle 1.27 s, walk 1.22 s, i.e. on cadence; it is not comparable with the sim bake-off and does not test the CoP features.

**The two headline numbers are on different denominators: the CoP-only 0.6071 is on n = 224, the full-feature 0.7578 on n = 223.** 1 stance (`shuffle_03` at t = 0.00 s) begins at the first frame of its capture, so it has no pre-onset frames, `loading_rate_cps` is undefined there and `features.py` returns NaN rather than imputing a value. Any cell whose feature set includes that feature drops the stance from both sides of the split; the CoP-only cells do not use it and keep it. The stance is named and dropped, never filled in. Differencing 0.6071 and 0.7578 therefore compares two accuracies measured on test sets that are not the same set.

Confusion matrix of the headline cell (rows true, columns predicted, order fast, shuffle, walk), pooled over the folds:

| | fast | shuffle | walk | recall |
|---|---|---|---|---|
| **fast** | 78 | 10 | 5 | 0.839 |
| **shuffle** | 13 | 44 | 7 | 0.688 |
| **walk** | 39 | 14 | 14 | 0.209 |
| precision | 0.600 | 0.647 | 0.538 | |

Per-class recall fast 0.839, shuffle 0.688, walk 0.209: walk test stances are called fast 39 times out of 67. The table below shows why a session boundary does that -- the two sessions of one class are not the same distribution:

| feature | class | session | n | mean | sd | shift from the class's other session(s), in pooled sd |
|---|---|---|---|---|---|---|
| cop_path_len | fast | `fast02` | 48 | 0.8843 | 0.2034 | -1.02 |
| cop_path_len | fast | `fast_03` | 45 | 1.1368 | 0.2233 | +1.02 |
| cop_path_len | shuffle | `shuffle02` | 30 | 0.9870 | 0.2110 | +0.34 |
| cop_path_len | shuffle | `shuffle_03` | 34 | 0.9051 | 0.2581 | -0.34 |
| cop_path_len | walk | `walk02` | 35 | 0.8913 | 0.2199 | -0.92 |
| cop_path_len | walk | `walk_03` | 32 | 1.2583 | 0.4579 | +0.92 |
| cop_displacement | fast | `fast02` | 48 | 0.5940 | 0.1091 | -0.52 |
| cop_displacement | fast | `fast_03` | 45 | 0.6404 | 0.0551 | +0.52 |
| cop_displacement | shuffle | `shuffle02` | 30 | 0.3428 | 0.0787 | -0.69 |
| cop_displacement | shuffle | `shuffle_03` | 34 | 0.4175 | 0.1194 | +0.69 |
| cop_displacement | walk | `walk02` | 35 | 0.4929 | 0.1852 | -0.63 |
| cop_displacement | walk | `walk_03` | 32 | 0.5893 | 0.0886 | +0.63 |

McNemar, QDA vs LDA on the same test stances: b = 16, c = 8, exact two-sided p = 0.1516; the smallest p attainable at b + c = 24 is 0.0000.

## 5. Per-class feature distributions (headline cell)

![feature distributions](..\figures\real_results\feature_distributions.png)

| feature | class | n | mean | sd | min | median | max |
|---|---|---|---|---|---|---|---|
| cop_path_len | fast | 93 | 1.0065 | 0.2471 | 0.3250 | 0.9522 | 1.5978 |
| cop_path_len | shuffle | 64 | 0.9435 | 0.2389 | 0.2985 | 0.9146 | 1.3817 |
| cop_path_len | walk | 67 | 1.0666 | 0.3969 | 0.2055 | 0.9912 | 2.9945 |
| cop_displacement | fast | 93 | 0.6165 | 0.0899 | 0.1226 | 0.6358 | 0.7354 |
| cop_displacement | shuffle | 64 | 0.3825 | 0.1083 | 0.0745 | 0.3964 | 0.5590 |
| cop_displacement | walk | 67 | 0.5389 | 0.1540 | 0.0362 | 0.5920 | 0.6970 |

Fraction of each class's stances (all of them) that fall inside another class's p10–p90 band (all sessions), per feature. High values are the overlap the classifier cannot resolve:

| feature | class | inside fast band | inside shuffle band | inside walk band |
|---|---|---|---|---|
| cop_path_len | fast | — | 0.75 | 0.88 |
| cop_path_len | shuffle | 0.83 | — | 0.86 |
| cop_path_len | walk | 0.69 | 0.69 | — |
| cop_displacement | fast | — | 0.08 | 0.56 |
| cop_displacement | shuffle | 0.05 | — | 0.77 |
| cop_displacement | walk | 0.70 | 0.12 | — |

## 6. Every misclassified test stance

Every stance was tested once, out of its own session, so this is every stance the headline cell gets wrong anywhere in the data.

| session | onset (s) | true | predicted | cop_path_len | cop_displacement | contact_time_s | s4 = 0 frames |
|---|---|---|---|---|---|---|---|
| walk02 | 0.00 | walk | shuffle | 0.6038 | 0.1143 | 0.83 | 99% |
| walk02 | 1.35 | walk | fast | 0.8911 | 0.5720 | 1.05 | 44% |
| walk02 | 4.50 | walk | fast | 1.3342 | 0.6373 | 1.02 | 18% |
| walk02 | 6.00 | walk | fast | 0.7009 | 0.6390 | 1.02 | 29% |
| walk02 | 7.65 | walk | shuffle | 1.1423 | 0.0362 | 1.08 | 28% |
| walk02 | 9.35 | walk | shuffle | 0.6981 | 0.4291 | 1.02 | 24% |
| walk02 | 10.87 | walk | shuffle | 0.9030 | 0.3576 | 1.18 | 21% |
| walk02 | 12.56 | walk | shuffle | 0.6536 | 0.1086 | 1.28 | 10% |
| walk02 | 14.38 | walk | fast | 0.8262 | 0.6260 | 1.14 | 56% |
| walk02 | 16.03 | walk | shuffle | 0.7536 | 0.3931 | 1.24 | 35% |
| walk02 | 17.70 | walk | fast | 0.7139 | 0.6446 | 1.36 | 61% |
| walk02 | 19.54 | walk | fast | 0.7488 | 0.6258 | 1.13 | 27% |
| walk02 | 21.21 | walk | fast | 0.9113 | 0.5860 | 1.22 | 33% |
| walk02 | 22.95 | walk | shuffle | 0.8921 | 0.1261 | 1.16 | 28% |
| walk02 | 24.73 | walk | fast | 0.8832 | 0.5682 | 1.10 | 32% |
| walk02 | 26.39 | walk | fast | 0.7030 | 0.6295 | 1.16 | 30% |
| walk02 | 28.10 | walk | fast | 1.1567 | 0.6081 | 1.33 | 77% |
| walk02 | 29.93 | walk | shuffle | 0.9442 | 0.4991 | 1.43 | 32% |
| walk02 | 31.91 | walk | fast | 0.7440 | 0.5955 | 1.20 | 21% |
| walk02 | 33.68 | walk | fast | 1.0507 | 0.5674 | 1.31 | 23% |
| walk02 | 35.58 | walk | shuffle | 0.9875 | 0.1572 | 1.16 | 4% |
| walk02 | 37.44 | walk | fast | 0.6839 | 0.6221 | 1.23 | 42% |
| walk02 | 39.26 | walk | fast | 0.9436 | 0.6306 | 1.16 | 15% |
| walk02 | 40.97 | walk | shuffle | 0.9305 | 0.4887 | 1.34 | 33% |
| walk02 | 42.73 | walk | shuffle | 0.9133 | 0.4596 | 1.35 | 33% |
| walk02 | 44.55 | walk | fast | 0.8683 | 0.6220 | 1.31 | 49% |
| walk02 | 46.36 | walk | fast | 1.1064 | 0.6149 | 1.36 | 18% |
| walk02 | 48.35 | walk | fast | 1.0862 | 0.6101 | 1.16 | 21% |
| walk02 | 51.97 | walk | fast | 0.8934 | 0.6458 | 1.31 | 15% |
| walk02 | 53.83 | walk | fast | 1.0136 | 0.6604 | 1.29 | 15% |
| walk02 | 57.40 | walk | shuffle | 0.9156 | 0.5406 | 1.28 | 37% |
| walk02 | 59.14 | walk | shuffle | 0.4278 | 0.2187 | 0.85 | 31% |
| fast02 | 0.24 | fast | shuffle | 0.8184 | 0.4235 | 0.91 | 10% |
| fast02 | 7.82 | fast | shuffle | 0.7032 | 0.3701 | 0.81 | 16% |
| fast02 | 13.89 | fast | walk | 0.6636 | 0.5893 | 0.75 | 34% |
| fast02 | 16.23 | fast | shuffle | 0.6572 | 0.3200 | 0.71 | 21% |
| fast02 | 19.82 | fast | shuffle | 0.9522 | 0.4241 | 0.92 | 28% |
| fast02 | 22.52 | fast | walk | 0.6541 | 0.5863 | 0.86 | 23% |
| fast02 | 23.86 | fast | shuffle | 0.3250 | 0.1226 | 0.93 | 1% |
| fast02 | 37.80 | fast | shuffle | 1.0802 | 0.5275 | 0.82 | 34% |
| fast02 | 41.58 | fast | shuffle | 0.6025 | 0.4711 | 0.74 | 27% |
| fast02 | 45.31 | fast | shuffle | 1.2238 | 0.5019 | 0.87 | 23% |
| fast02 | 57.69 | fast | shuffle | 0.9471 | 0.5180 | 0.77 | 38% |
| walk_03 | 3.53 | walk | fast | 1.0762 | 0.5905 | 1.22 | 32% |
| walk_03 | 5.26 | walk | fast | 1.1140 | 0.5636 | 1.35 | 36% |
| walk_03 | 7.35 | walk | fast | 1.1123 | 0.6885 | 1.14 | 25% |
| walk_03 | 9.09 | walk | fast | 1.0934 | 0.6444 | 1.41 | 38% |
| walk_03 | 11.23 | walk | fast | 0.7279 | 0.6463 | 1.19 | 42% |
| walk_03 | 12.97 | walk | fast | 1.1767 | 0.5738 | 1.20 | 37% |
| walk_03 | 16.75 | walk | fast | 1.2247 | 0.6599 | 1.51 | 67% |
| walk_03 | 24.54 | walk | fast | 0.9912 | 0.6355 | 1.22 | 55% |
| walk_03 | 26.41 | walk | fast | 1.1533 | 0.5229 | 1.09 | 50% |
| walk_03 | 28.28 | walk | fast | 0.8016 | 0.6748 | 1.35 | 23% |
| walk_03 | 30.10 | walk | fast | 1.4800 | 0.6402 | 1.17 | 34% |
| walk_03 | 31.89 | walk | fast | 1.3165 | 0.6232 | 1.21 | 30% |
| walk_03 | 37.31 | walk | fast | 0.9254 | 0.5969 | 1.36 | 31% |
| walk_03 | 39.16 | walk | fast | 1.2835 | 0.6210 | 1.07 | 27% |
| walk_03 | 40.85 | walk | fast | 1.0846 | 0.6970 | 1.35 | 60% |
| walk_03 | 42.97 | walk | fast | 1.3241 | 0.6062 | 1.24 | 14% |
| walk_03 | 44.92 | walk | fast | 0.9798 | 0.6392 | 1.23 | 46% |
| walk_03 | 50.32 | walk | shuffle | 1.3919 | 0.4892 | 1.28 | 34% |
| walk_03 | 52.11 | walk | fast | 1.0934 | 0.5819 | 1.18 | 27% |
| walk_03 | 56.01 | walk | fast | 1.1605 | 0.5916 | 1.28 | 24% |
| walk_03 | 57.93 | walk | fast | 0.8929 | 0.6345 | 1.10 | 21% |
| fast_03 | 0.00 | fast | shuffle | 0.8280 | 0.3665 | 0.34 | 6% |
| fast_03 | 3.52 | fast | walk | 1.5476 | 0.6217 | 1.01 | 44% |
| fast_03 | 15.09 | fast | walk | 1.5511 | 0.5590 | 1.11 | 27% |
| fast_03 | 16.63 | fast | walk | 1.5978 | 0.6039 | 1.01 | 30% |
| shuffle_03 | 0.00 | shuffle | walk | 0.2985 | 0.2204 | 0.37 | 13% |
| shuffle_03 | 2.76 | shuffle | walk | 0.6583 | 0.1952 | 1.19 | 8% |
| shuffle_03 | 6.37 | shuffle | fast | 1.0405 | 0.5180 | 1.19 | 13% |
| shuffle_03 | 8.16 | shuffle | walk | 1.0932 | 0.4998 | 1.11 | 13% |
| shuffle_03 | 9.94 | shuffle | fast | 0.7861 | 0.4466 | 1.13 | 15% |
| shuffle_03 | 13.75 | shuffle | fast | 0.6911 | 0.4798 | 1.22 | 15% |
| shuffle_03 | 19.42 | shuffle | fast | 0.8325 | 0.5075 | 1.25 | 13% |
| shuffle_03 | 21.36 | shuffle | fast | 1.0730 | 0.5567 | 1.20 | 30% |
| shuffle_03 | 23.23 | shuffle | fast | 0.9073 | 0.4869 | 1.38 | 23% |
| shuffle_03 | 25.40 | shuffle | fast | 0.8356 | 0.4738 | 1.31 | 27% |
| shuffle_03 | 31.17 | shuffle | walk | 1.3806 | 0.5125 | 1.22 | 18% |
| shuffle_03 | 34.81 | shuffle | walk | 1.1569 | 0.5187 | 1.21 | 21% |
| shuffle_03 | 42.22 | shuffle | fast | 0.6744 | 0.4494 | 1.14 | 16% |
| shuffle_03 | 45.63 | shuffle | walk | 1.3817 | 0.5399 | 1.15 | 16% |
| shuffle_03 | 49.22 | shuffle | fast | 1.0131 | 0.5013 | 1.20 | 19% |
| shuffle_03 | 50.94 | shuffle | fast | 0.9612 | 0.5046 | 1.06 | 19% |
| shuffle_03 | 52.62 | shuffle | fast | 0.7673 | 0.4899 | 1.11 | 12% |
| shuffle_03 | 56.26 | shuffle | fast | 0.8060 | 0.5590 | 1.06 | 9% |
| shuffle_03 | 57.93 | shuffle | fast | 0.9175 | 0.4902 | 0.97 | 29% |
| shuffle_03 | 59.52 | shuffle | walk | 0.3528 | 0.0745 | 0.48 | 8% |

### walk → shuffle (14)

![walk to shuffle](..\figures\real_results\errors_walk_to_shuffle.png)

Measured: `cop_path_len` averages 0.8684 over these stances against class means over all sessions 1.0666 (walk) and 0.9435 (shuffle), nearer to shuffle; `cop_displacement` averages 0.3156 over these stances against class means over all sessions 0.5389 (walk) and 0.3825 (shuffle), nearer to shuffle; s4 read 0 on 32% of their frames against 35% for walk and 20% for shuffle (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 11.8 mm for walk; their contact time averages 1.18 s against class medians 1.22 s (walk) and 1.27 s (shuffle), which the CoP-only cell never sees.

### walk → fast (39)

![walk to fast](..\figures\real_results\errors_walk_to_fast.png)

Measured: `cop_path_len` averages 1.0070 over these stances against class means over all sessions 1.0666 (walk) and 1.0065 (fast), nearer to fast; `cop_displacement` averages 0.6189 over these stances against class means over all sessions 0.5389 (walk) and 0.6165 (fast), nearer to fast; s4 read 0 on 34% of their frames against 35% for walk and 32% for fast (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 11.8 mm for walk; their contact time averages 1.22 s against class medians 1.22 s (walk) and 0.80 s (fast), which the CoP-only cell never sees.

### fast → shuffle (10)

![fast to shuffle](..\figures\real_results\errors_fast_to_shuffle.png)

Measured: `cop_path_len` averages 0.8138 over these stances against class means over all sessions 1.0065 (fast) and 0.9435 (shuffle), nearer to shuffle; `cop_displacement` averages 0.4045 over these stances against class means over all sessions 0.6165 (fast) and 0.3825 (shuffle), nearer to shuffle; s4 read 0 on 20% of their frames against 32% for fast and 20% for shuffle (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 11.8 mm for fast; their contact time averages 0.78 s against class medians 0.80 s (fast) and 1.27 s (shuffle), which the CoP-only cell never sees.

### fast → walk (5)

![fast to walk](..\figures\real_results\errors_fast_to_walk.png)

Measured: `cop_path_len` averages 1.2029 over these stances against class means over all sessions 1.0065 (fast) and 1.0666 (walk), nearer to walk; `cop_displacement` averages 0.5920 over these stances against class means over all sessions 0.6165 (fast) and 0.5389 (walk), nearer to fast; s4 read 0 on 32% of their frames against 32% for fast and 35% for walk (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 11.8 mm for fast; their contact time averages 0.95 s against class medians 0.80 s (fast) and 1.22 s (walk), which the CoP-only cell never sees.

### shuffle → walk (7)

![shuffle to walk](..\figures\real_results\errors_shuffle_to_walk.png)

Measured: `cop_path_len` averages 0.9032 over these stances against class means over all sessions 0.9435 (shuffle) and 1.0666 (walk), nearer to shuffle; `cop_displacement` averages 0.3659 over these stances against class means over all sessions 0.3825 (shuffle) and 0.5389 (walk), nearer to shuffle; s4 read 0 on 14% of their frames against 20% for shuffle and 35% for walk (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 14.5 mm for shuffle; their contact time averages 0.96 s against class medians 1.27 s (shuffle) and 1.22 s (walk), which the CoP-only cell never sees.

### shuffle → fast (13)

![shuffle to fast](..\figures\real_results\errors_shuffle_to_fast.png)

Measured: `cop_path_len` averages 0.8697 over these stances against class means over all sessions 0.9435 (shuffle) and 1.0065 (fast), nearer to shuffle; `cop_displacement` averages 0.4972 over these stances against class means over all sessions 0.3825 (shuffle) and 0.6165 (fast), nearer to shuffle; s4 read 0 on 18% of their frames against 20% for shuffle and 32% for fast (class means over all sessions); on those frames the CoP is a five-sensor centroid, and the counterfactual below puts that at 14.5 mm for shuffle; their contact time averages 1.17 s against class medians 1.27 s (shuffle) and 0.80 s (fast), which the CoP-only cell never sees.

## 7. Mechanisms, measured

| class | s4 = 0 inside stances | CoP shift on s4-zero frames (mm) | CoP shift A → C, all stance frames (mean / median mm) | stance-to-stance ML spread sd (mm) |
|---|---|---|---|---|
| fast | 31.9% | 11.8 | 3.62 / 3.36 | 2.28 |
| shuffle | 19.8% | 14.5 | 3.33 / 3.33 | 2.12 |
| walk | 35.1% | 11.8 | 3.44 / 3.26 | 2.67 |

**s4 zeros.** s4 (1st metatarsal head) has the highest activation threshold of the six, so its zeros are below-threshold readings, never imputed. On the frames inside kept stances it reads 0 on 32% (fast), 35% (walk) and 20% (shuffle) of frames. The CoP shift those zeros are responsible for is measured directly: on every such frame the CoP is recomputed with s4 set to the median non-zero s4 count across the captures (343 counts) and the difference taken -- 11.8 / 11.8 / 14.5 mm for fast / walk / shuffle, on a 91 mm wide insole.

**Gain match.** Replacing raw counts by gain-matched conductance moves the per-frame CoP by 3.62 / 3.44 / 3.33 mm on average (fast / walk / shuffle), i.e. the s4-zero effect is 3× the gain-match effect on walk. The CoP-only QDA scores 0.6071 under A, 0.5982 under B and 0.5982 under C on the leave-one-session-out split (within-session 0.6667 / 0.6667 / 0.6778): the representation moves the answer by at most 2 test stance(s) in 224. Whatever the representation, the same frames carry the same s4 zeros and the same extrapolation above 824 counts.

**Figure-8 turning.** The stance-to-stance spread of the mean medial-lateral CoP position is 2.67 / 2.28 / 2.12 mm (sd; walk / fast / shuffle). Against the ±15 mm uncertainty on the sensor coordinates this is not resolvable, so turning remains a hypothesis, as `docs/sim_vs_real.md` D4b concluded.

## 8. Peak force against onset time

![peak vs onset](..\figures\real_results\peak_vs_onset.png)

**9 tests, corrected together.** Peak total force is regressed on stance onset time once per class pooled over that class's sessions (3 tests) and once per (class, session) (6 tests) -- 9 in all, every one of them run and every one of them looked at. That is the family, so it is corrected as a family: reporting these fits and then quoting the smallest p out of them without a correction is exactly the multiple-comparisons error, and the uncorrected p values below mean nothing on their own. The correction is Holm-Bonferroni, which controls the family-wise error rate under arbitrary dependence between the tests -- necessary here, because the pooled fits and the per-session fits are fitted on overlapping stances and are not independent.

| scope | class | session | n | slope (raw counts per s of capture) | r | p | p (Holm, m = 9) |
|---|---|---|---|---|---|---|---|
| pooled | fast | `both` | 93 | +3.92 | +0.276 | 0.0073 | 0.0660 |
| pooled | shuffle | `both` | 64 | -4.66 | -0.197 | 0.1187 | 0.8310 |
| pooled | walk | `both` | 67 | +1.52 | +0.061 | 0.6264 | 1.0000 |
| per session | fast | `fast02` | 48 | +3.17 | +0.226 | 0.1218 | 0.8310 |
| per session | fast | `fast_03` | 45 | +4.71 | +0.328 | 0.0278 | 0.2222 |
| per session | shuffle | `shuffle02` | 30 | -3.99 | -0.208 | 0.2701 | 1.0000 |
| per session | shuffle | `shuffle_03` | 34 | -4.80 | -0.195 | 0.2685 | 1.0000 |
| per session | walk | `walk02` | 35 | +3.02 | +0.129 | 0.4590 | 1.0000 |
| per session | walk | `walk_03` | 32 | -0.18 | -0.007 | 0.9718 | 1.0000 |

Onset time is seconds into each session's own capture, so the sessions of one class overlay on the x axis of the figure and of the pooled fits. Stress relaxation acts within one continuous capture, so the per-session fits are the physically direct test and the pooled ones mix two captures; both are reported because both were run.

Uncorrected, fast pooled rises at +3.92 counts/s (p = 0.0073, Holm p = 0.0660); fast `fast_03` rises at +4.71 counts/s (p = 0.0278, Holm p = 0.2222). After correction over the 9 tests, **none of them survives at p < 0.05** -- the smallest adjusted p in the whole family is 0.0660.

**A rise runs opposite to the project's best-established physical finding, so it does not get reported flatly.** fast pooled, fast `fast_03` rise rather than falls. FSR stress relaxation is measured, reproduced across two independent bench sessions, and monotone in rest interval (`docs/calibration_notes.md`); it predicts peak counts *falling* across a capture at constant applied force. An apparent contradiction of it needs more than one uncorrected p value. Two explanations fit the observation and this data does not separate them:

1. **The subject accelerated across the session.** Applied force is not constant in gait -- the subject can push harder as the capture goes on, and a rising applied force can outrun a falling sensitivity. Nothing was recorded that would measure applied force independently of the sensor whose drift is in question, so this explanation cannot be checked against the captures that produced it.
2. **A multiple-comparisons artifact.** 9 tests were run; 5 of the 9 fitted slopes are positive and 4 negative, which is what noise around zero looks like. After Holm correction nothing in the family reaches p < 0.05 (smallest adjusted p 0.0660), so the correction is sufficient on its own to account for the rise.

**Verdict.** The corrected p values kill the result. After Holm correction over the 9 tests in the family, no peak-force trend in this data is significant at p < 0.05 in either direction, the rise included. It is reported here as a measurement that did not reach significance, not as a finding, and it is not evidence against stress relaxation. Separating a real acceleration from noise needs an independent measure of applied force during gait, which this hardware does not have. **Hardware-blocked.**

## 9. Simulator versus real

The sim-trained deployment models (`models/model_lda.json`, `models/model_qda.json`, fitted on 12 simulated sessions by `scripts/fit_model.py`) applied to the same 224 real stances under the representation they were fitted on (B): LDA 0.3795 (85/224), QDA 0.3438 (77/224), below the 0.4152 majority floor -- the expected outcome for a model fitted on a generator whose constants were co-evolved with the detector. The same recipe retrained on real stances scores 0.6071 [0.5419, 0.6688] leave-one-session-out (the shipped B model 0.5982 [0.5329, 0.6602]), and the sim bake-off's 0.9296 on 270 held-out simulated stances (`docs/bakeoff.md`) is not a number this data can reproduce or refute: different stances, different split, different world.

## 10. Split verdict: what more data fixes, what the hardware cannot

More data would fix:

- **Sessions.** fast 2, shuffle 2, walk 2 sessions per class is the minimum that makes leave-one-session-out possible; every headline interval here is wide and one odd session moves it a lot. More sessions narrow it; they do not change what it measures.
- **Subjects.** One subject. Nothing here says anything about another foot.
- **Path.** Everything was walked on a figure-8 in a small space; straight-line gait and its symmetric loading are unmeasured.
- **Cadence range.** Fast and walk are separated by contact time (0.80 vs 1.22 s median); intermediate cadences would blur that boundary and the CoP features would have to carry it.

Six-sensor hardware limits that data will not fix:

- **s4's activation threshold** turns the CoP into a five-sensor centroid on 20%–35% of stance frames, with a 12–14 mm shift.
- **±15 mm sensor coordinates** on a 91 mm wide insole: every CoP distance inherits it.
- **The gain match extrapolates** above 824 counts and does not hold below ~5 N.
- **No spatial resolution between sensors**: the CoP is a weighted mean of six points; anything between them is interpolation.

