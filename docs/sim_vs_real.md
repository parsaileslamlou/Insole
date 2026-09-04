# Simulator vs. real hardware — the first non-circular test

Stage 13, phases C–E. Everything below is measured on the four `_02` captures
in `data/real/`, against `insole/gait_gen.py` output, under the corrected 274 × 91 mm
geometry.

> **Superseded in part.** Sections C5, C6 and D2–D5 were measured with
> `MAX_DURATION = 120`. Finding 1 below led to raising it to 200
> (`insole/detector.py`, `scripts/sweep_max_duration.py`); re-running `scripts/analyze_real.py` and
> `scripts/sim_vs_real.py` regenerates every table that depends on it. The tables here
> are kept as the record of *why* the change was made; do not quote them as
> current. Two figures in this document were also found to be unsupported and
> are corrected in place below with a note: the `~940` calibration ceiling
> (C3, B4) and the `data/real/README.md` s4 ordering (finding 2, now fixed in
> that file). D2 was regenerated at stage 20 under the shipped feature
> representation (conductance): sim-trained LDA 0.3097 (35/113), QDA 0.2566
> (29/113), still below the 0.4248 floor; `python scripts/sim_vs_real.py`
> prints the current figures.

## Regeneration

Every number in this document is printed by one of two scripts. Nothing was
transcribed by hand from a session that cannot be re-run.

| Section | Produced by | Command |
|---|---|---|
| C1–C6 | [scripts/analyze_real.py](../scripts/analyze_real.py) | `python scripts/analyze_real.py` |
| D1 | [scripts/bakeoff.py](../scripts/bakeoff.py) | `rm data/sim/features_sessions.csv && python scripts/bakeoff.py` |
| D2–D5 | [scripts/sim_vs_real.py](../scripts/sim_vs_real.py) | `python scripts/sim_vs_real.py` |
| geometry | [insole/detector.py](../insole/detector.py) | `python tests/test_geometry.py` |

`data/real/` is read-only throughout. Neither script writes into it.

---

## Collection context

Read this before any number below, because most of the deltas trace back to it.

**Hardware.** Right-foot insole, 6 × Interlink FSR UX 402, soldered, on an
ESP32-S3 over tethered USB serial at 100 Hz. Frames carry the firmware's own
sequence number and microsecond timestamp.

**Trials.** Four activities, 60 s each, one trial per activity:

- **stand** — quiet standing.
- **walk** — a **figure-8** path around a small area. The foot was
  continuously turning and never walking straight.
- **fast** — faster cadence, **same figure-8 path**.
- **shuffle** — feet dragging, short strides, minimal ground clearance.

**The mismatch this sets up.** `insole/gait_gen.py` models straight, symmetric,
stride-identical gait: one canonical stance shape repeated at a fixed period
with a small cadence jitter. The real captures are a foot going round a
figure-8 in a confined space. Continuous turning loads the foot asymmetrically
and varies stride to stride; nothing in the simulator represents that. Where
real and sim disagree below, the figure-8 path is a live candidate explanation
and is treated as one, not as a conclusion.

**One trial per class.** There is no training set here. Section D2 fits on
simulated sessions and predicts real stances purely to prove the plumbing
carries real frames end to end. Its accuracies are not a model result.

---

## C1 — Ingest

| activity | frames | duration (s) | seq gaps | dt median (µs) | dt min | dt max | dt std | dt IQR |
|---|---|---|---|---|---|---|---|---|
| stand   | 6000 | 59.99 | 0 | 10000.0 | 9999 | 10001 | 0.0 | 0.0 |
| walk    | 6000 | 59.99 | 0 | 10000.0 | 9999 | 10001 | 0.1 | 0.0 |
| fast    | 6000 | 59.99 | 0 | 10000.0 | 9999 | 10001 | 0.2 | 0.0 |
| shuffle | 6000 | 59.99 | 0 | 10000.0 | 9999 | 10001 | 0.2 | 0.0 |

No dropped frames in any capture, and the sampling interval is uniform to
**±1 µs** — a jitter of 0.01% of the 10 ms period. This is the firmware's own
timestamp, not an assumed 100 Hz: `notebooks/insole.ipynb` cell 2 used to overwrite
`ts_us` with `index * 10000`, which would have produced this table by
construction. That line was deleted on this branch, so the uniformity above is
a measurement rather than an artifact.

Per-sensor counts, min / median / max:

| activity | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| stand   | 723 / 1045 / 1236 | 0 / 1179 / 1420 | 949 / 1132 / 1237 | 846 / 1003 / 1114 | 0 / 0 / 7 | 212 / 426 / 606 |
| walk    | 0 / 64 / 1579 | 0 / 0 / 1692 | 0 / 674 / 1365 | 0 / 714 / 1449 | 0 / 0 / 1574 | 0 / 232 / 1878 |
| fast    | 0 / 0 / 1645 | 0 / 0 / 1716 | 0 / 443 / 1426 | 0 / 490 / 1626 | 0 / 0 / 1648 | 0 / 62 / 2011 |
| shuffle | 0 / 448 / 1646 | 0 / 0 / 1650 | 8 / 786 / 1365 | 0 / 827 / 1416 | 0 / 53 / 1344 | 0 / 256 / 1497 |

No channel reaches even half of the 4095 ceiling in any capture (max observed
2011, s5 in fast). Saturation is not a limiting factor in this dataset.

## C2 — Gain match

`calibration.apply_gain_match` applied per frame, with corrections
s0 = 0.9900, s1 = 0.9616, s2 = 0.9741, s3 = 1.2513, s4 = 0.8692, s5 = 0.9538.

The correction multiplies the conductance `x = counts / (4095 − counts)`, not
the raw counts. Gain-matched conductance, median / max per sensor:

| activity | s0 | s1 | s2 | s3 | s4 | s5 | NaN cells |
|---|---|---|---|---|---|---|---|
| stand   | 0.3392 / 0.428 | 0.3906 / 0.510 | 0.3722 / 0.422 | 0.4059 / 0.468 | 0.0004 / 0.001 | 0.1107 / 0.166 | 6244 |
| walk    | 0.3230 / 0.621 | 0.4397 / 0.677 | 0.1940 / 0.487 | 0.3033 / 0.685 | 0.1682 / 0.543 | 0.2897 / 0.808 | 13290 |
| fast    | 0.4008 / 0.665 | 0.4594 / 0.694 | 0.1575 / 0.520 | 0.2629 / 0.824 | 0.2250 / 0.585 | 0.3021 / 0.920 | 14608 |
| shuffle | 0.3392 / 0.665 | 0.3958 / 0.649 | 0.2312 / 0.487 | 0.3196 / 0.661 | 0.0833 / 0.425 | 0.2174 / 0.550 | 11521 |

A zero count has conductance `0 / 4095 = 0`, which `conductance()` rejects as
non-positive, so `apply_gain_match` returns `None` and the cell becomes NaN.
Every below-threshold s4 zero lands there. That is the transform's rule about
its own domain, not a claim the sample is missing — see C4.

**Test confirmation.** The load-bearing assertion in `tests/test_calibration.py`
still passes after this call site was written:

```
PASS  correction is applied in conductance space (x = c/(fs - c))  c=1000 x=0.323102 got[0]=0.319868
PASS  result is nowhere near raw-count scaling (correction * counts)  conductance 0.319868 vs raw 990.0
```

## C3 — Extrapolation (honesty check)

The gain match was derived at a single ~12 N point, and the highest count any
calibration sample reached was **824** (`calibration.CAL_MAX_COUNTS`, from
`cal_data/`; the highest per-trial mean in the manifest is 809). Percentage of
frames whose count exceeds that ceiling, i.e. where the gain match is
extrapolating past everything it was ever sampled at:

| activity | s0 | s1 | s2 | s3 | s4 | s5 | any sensor |
|---|---|---|---|---|---|---|---|
| stand   | 99.37% | 94.98% | **100.00%** | **100.00%** | 0.00% | 0.00% | **100.00%** |
| walk    | 34.23% | 24.82% | 45.25% | 46.17% | 19.63% | 33.85% | 66.58% |
| fast    | 30.75% | 21.97% | 37.90% | 40.30% | 22.27% | 31.53% | 62.45% |
| shuffle | 40.37% | 28.98% | 48.72% | 50.10% | 5.90% | 23.58% | 61.97% |

Between **61.97% and 100%** of frames in every capture contain at least one
sensor outside the calibrated range. For standing it is every single frame, and
s2 and s3 are above the ceiling 100% of the time. Reported, not fixed: no
correction is applied and no frame is dropped on this basis.

> **Corrected.** This section originally used a ceiling of
> "~940 counts" and reported 58.55–100%. Nothing in `cal_data/` reaches 940:
> the highest raw sample is 824 (`cal_s5_t3.csv`) and the highest per-trial
> mean is 809.11. The table above is the regenerated one
> (`tests/test_infer_live.py` recomputes the constant from `cal_data/`).

## C4 — Zeros are below-threshold, not missing

s4 has the highest activation threshold of the six — calibration read s4 = 0
counts at 2.58 N while s5 read 239 at 2.49 N. Nothing in this analysis drops,
imputes, interpolates, or flags these frames.

Percentage of frames reading exactly 0 counts:

| activity | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|
| stand   | 0.00% | 4.95% | 0.00% | 0.00% | **99.12%** | 0.00% |
| walk    | 47.87% | 70.72% | 0.30% | 5.72% | 52.02% | 44.88% |
| fast    | 57.12% | 67.13% | 6.42% | 11.23% | **56.05%** | 45.52% |
| shuffle | 41.80% | 59.32% | 0.00% | 0.38% | **46.38%** | 44.13% |

### C4b — The CoP consequence

On an s4-zero frame, s4 contributes zero weight and CoP degenerates to a
5-sensor centroid. s4 is the most medial forefoot sensor (25.4 of 91 mm), so
losing it pulls CoP **laterally** (+x) and, because it also sits well forward
at 203 mm, **toward the heel** (−y).

Measured by recomputing each affected frame's CoP with s4 given a non-zero
substitute and differencing. Two substitutes are reported: each activity's own
median non-zero s4, and a common 513 counts (the median non-zero s4 pooled
across all four captures) so the columns are comparable.

| activity | s4-zero frames | own sub | own Δx (mm) | own \|Δ\| (mm) | common Δx (mm) | common Δy (mm) | common \|Δ\| (mm) |
|---|---|---|---|---|---|---|---|
| stand   | 5947 | 2   | +0.01  | 0.04  | +3.16  | −7.98  | 8.58 |
| walk    | 3121 | 664 | +24.82 | 36.51 | +23.19 | −24.25 | 33.55 |
| fast    | 3240 | 842 | +26.88 | 41.17 | +24.58 | −26.92 | 36.46 |
| shuffle | 2783 | 358 | +22.90 | 31.98 | +25.98 | −25.93 | **36.70** |

The bias is **33.6–36.7 mm** on the three moving activities under the common
substitute — roughly an eighth of the insole's length, and more than twice the
±15 mm coordinate uncertainty.

**Method caveat.** The "own" columns make stand look bias-free, but that is an
artifact: s4's median non-zero value *in stand* is itself 2 counts, so the
counterfactual asks "what if this zero were a 2" and correctly answers
"nothing". It does not show standing is unbiased — it shows the bias is
**unmeasurable from the stand capture alone**, because s4 never activates there
to calibrate a counterfactual against. The common-substitute columns are the
comparable ones.

## C5 — Stance detection, thresholds unchanged

`T_ON = 1200`, `T_OFF = 450`, `MIN_DURATION = 15`, `MAX_DURATION = 120`,
`GAP_MERGE = 12`. These were chosen by sweeping ~2016 combinations against
simulated streams whose constants were co-evolved with them. **This is their
first test against data the simulator did not produce, and they were not
retuned.**

| activity | raw | merged | median duration (fr) | min | max | median (s) | frames ≥ T_ON |
|---|---|---|---|---|---|---|---|
| stand   | 0  | **0**  | — | — | — | — | 100.0% |
| walk    | 18 | **18** | 112.5 | 84 | 119 | 1.125 | 68.1% |
| fast    | 48 | **48** | 80.0  | 64 | 94  | 0.800 | 62.9% |
| shuffle | 2  | **2**  | 99.0  | 96 | 102 | 0.990 | 65.8% |

Total-force distribution, for reading those against the thresholds:

| activity | min | p05 | median | p95 | max |
|---|---|---|---|---|---|
| stand   | 3516 | 4038 | 4808 | 4896 | 5335 |
| walk    | 2 | 94 | 3662 | 5489 | 6435 |
| fast    | 0 | 16 | 3332 | 5595 | 6667 |
| shuffle | 27 | 106 | 3533 | 5824 | 6193 |

**stand yields 0 stances — the null check passes.** Note *how*: standing never
drops below `T_OFF` (min total force 3516), so it is a single unbroken 6000-frame
run that the `MAX_DURATION` break discards. The mechanism is the one the
simulator's `sim_stand` fixture was built to exercise.

**walk and shuffle are badly under-detected.** Re-running the same T_ON/T_OFF
hysteresis with the duration limits removed shows why:

| activity | runs | median len | min | max | over MAX_DURATION | under MIN_DURATION | kept |
|---|---|---|---|---|---|---|---|
| stand   | 1  | 6000.0 | 6000 | 6000 | 1 | 0 | 0 |
| walk    | 35 | 119.0  | 84  | 144 | **17** | 0 | 18 |
| fast    | 48 | 80.0   | 64  | 94  | 0 | 0 | 48 |
| shuffle | 30 | 141.0  | 96  | 164 | **28** | 0 | 2 |

Runs longer than `MAX_DURATION` are **discarded outright**, not clipped — that
is annihilation, one true stance becoming zero detections. Real walk contacts
have a median natural length of **119 frames against a 120-frame ceiling**, so
the threshold cuts straight through the middle of the distribution. Shuffle sits
almost entirely above it (median 141), which is why 28 of 30 contacts vanish.
Fast is the only activity comfortably inside the limit and is the only one
detected cleanly.

**The survivors are a biased subsample.** Annihilation at a duration threshold
is not random dropout — it removes exactly the long contacts:

| activity | kept | kept median | natural median | kept max | natural max | first (s) | last (s) | largest gap (s) |
|---|---|---|---|---|---|---|---|---|
| walk    | 18 | 112.5 | 119.0 | 119 | 144 | 0.00 | 59.99 | 8.03 |
| fast    | 48 | 80.0  | 80.0  | 94  | 94  | 0.24 | 59.69 | 0.57 |
| shuffle | 2  | 99.0  | 141.0 | 102 | 164 | 0.00 | 59.99 | **58.03** |

Every real feature mean in C6 and D4 is computed on this truncated subsample.
For shuffle the two surviving stances sit at the two ends of the capture with
58 seconds of nothing between them.

## C6 — Features, and the uniform-dt approximation

`features.frame_dt` collapses sampling to `ts_us.diff().median()` and
`stance_features` multiplies that one dt across a whole stance. Measured against
the real timestamps:

| activity | stances | longest (fr) | approx (s) | true (s) | drift (µs) | drift (%) | worst drift (µs) |
|---|---|---|---|---|---|---|---|
| stand   | 0  | — | — | — | — | — | — |
| walk    | 18 | 119 | 1.1800 | 1.1800 | +0.0 | +0.000000% | +0.0 |
| fast    | 48 | 94  | 0.9300 | 0.9300 | +0.0 | +0.000000% | +1.0 |
| shuffle | 2  | 102 | 1.0100 | 1.0100 | +0.0 | +0.000000% | +0.0 |

**The approximation costs at most 1 µs over any stance in this dataset**, because
the sampling interval is uniform to ±1 µs (C1). On this hardware the shortcut is
free. That is a property of this firmware's timing, not of the approximation.

Feature summaries (mean ± sd):

| activity | n | peak_counts | time_to_peak_s | contact_time_s | loading_rate_cps | cop_path_len | cop_displacement |
|---|---|---|---|---|---|---|---|
| walk    | 18 | 5465.1 ± 453.7 | 0.513 ± 0.246 | 1.081 ± 0.104 | 8757 ± 5259 | 0.851 ± 0.210 | 0.448 ± 0.218 |
| fast    | 48 | 5761.4 ± 245.2 | 0.419 ± 0.152 | 0.799 ± 0.066 | 11923 ± 5291 | 0.884 ± 0.203 | 0.594 ± 0.109 |
| shuffle | 2  | 5635.0 ± 789.1 | 0.435 ± 0.516 | 0.980 ± 0.042 | 14219 ± 13308 | 0.820 ± 0.355 | 0.314 ± 0.066 |

Shuffle's n = 2 makes every shuffle statistic below decorative.

---

## D1 — The geometry invalidated the bake-off

The recorded bake-off numbers were computed under the old `SENSOR_COORDS`.
`cop_path_len` and `cop_displacement` live in a different space now, so the
frame was regenerated from the same 12 simulated sessions and the
session-disjoint split re-run.

The old figures reproduce exactly from the cached frame first, confirming the
comparison is like-for-like: the two frames describe the **same 1123 stances**
(identical `session` and `start` columns, and every non-CoP column
byte-identical). Only the two CoP features moved.

| | old geometry | new geometry | Δ |
|---|---|---|---|
| n_test | 270 | 270 | 0 |
| majority-class floor | 0.4296 | 0.4296 | 0.0000 |
| LogisticRegression (scaled) | 0.9185 | **0.9037** | −0.0148 |
| my LDA | 0.9333 | **0.9296** | −0.0037 |
| my QDA | 0.9370 | **0.9259** | −0.0111 |

All three models lose a little accuracy; the ordering LR < LDA < QDA does not
survive, since QDA now sits below LDA. The floor is unchanged because it uses no
features. Feature scale shifted as expected: mean `cop_path_len` 1.2070 → 0.8787
(× 0.728), mean `cop_displacement` 0.8027 → 0.6759 (× 0.842).

**The fast → walk error does not survive.**

| | LR | LDA | QDA | intersection | union |
|---|---|---|---|---|---|
| old geometry | 9 | 9 | 9 | 8 | 10 |
| new geometry | **7** | **3** | **6** | 3 | 7 |

Two corrections to the premise, both from the row indices rather than the counts:

1. Under the old geometry the *count* was 9 in all three models, but the *rows*
   were not identical across all three — LR misclassified row 124 where LDA and
   QDA misclassified row 66. LDA and QDA were identical to each other;
   LR differed by one row. Intersection 8, union 10.
2. Under the new geometry neither the count nor the identity survives: 7 / 3 / 6
   with only 3 rows common to all three. Of the 8 rows all three models got
   wrong before, LR still misses 4, LDA 3, QDA 4.

The error was a property of the old coordinate space, not a stable structural
confusion between fast and walk.

## D2 — Plumbing check (not a classifier result)

An LDA and a QDA fitted on all 1123 simulated stances, asked to label the 68
real stances. **This is a plumbing check.** One 60 s trial per class cannot be
trained on and is not being trained on; the point is that real frames survive
ingest → detection → features → model without a shape error or a silent NaN.
stand is excluded — it is not one of the model's three classes and yielded no
stances anyway.

Real stances scored: 68 (fast = 48, shuffle = 2, walk = 18). No non-finite CoP
features; nothing was dropped.

**sim-trained LDA — accuracy 0.1029 (7/68)**

| true ↓ / pred → | fast | shuffle | walk |
|---|---|---|---|
| fast    | 2 | 30 | 16 |
| shuffle | 0 | 2  | 0  |
| walk    | 0 | 15 | 3  |

**sim-trained QDA — accuracy 0.0882 (6/68)**

| true ↓ / pred → | fast | shuffle | walk |
|---|---|---|---|
| fast    | 1 | 34 | 13 |
| shuffle | 0 | 1  | 1  |
| walk    | 0 | 14 | 4  |

Both land *below* the 0.4296 majority floor and below chance. The pipeline
carries real data end to end — that is the entire claim. Do not quote these
accuracies as a model result.

## D3 — Comparison plots

One per activity, real per-sensor traces over `gait_gen` output for the same
activity on a shared time axis, with detected stances shaded (real blue, sim
orange). The x axis is clipped to the first 12 s; 60 s at 100 Hz is an
unreadable smear. The same unretuned detector shades both streams.

- [figures/sim_vs_real/stand.png](../figures/sim_vs_real/stand.png) — real 0 stances, sim 0
- [figures/sim_vs_real/walk.png](../figures/sim_vs_real/walk.png) — real 18, sim 60
- [figures/sim_vs_real/fast.png](../figures/sim_vs_real/fast.png) — real 48, sim 100
- [figures/sim_vs_real/shuffle.png](../figures/sim_vs_real/shuffle.png) — real 2, sim 120

The walk plot shows the shape difference at a glance: sim produces narrow
symmetric peaks reaching 2000–3200 counts, real produces broad plateaus around
1200–1500 counts lasting roughly twice as long.

## D4 — Quantitative diff, real minus sim

Signed delta is real − sim. `cop_ml_range` is the medial-lateral spread of the
CoP path within a stance, in normalised units (× 274 for mm).

### walk — real n = 18, sim n = 60

| metric | real mean | sim mean | Δ | Δ% | real sd | sim sd |
|---|---|---|---|---|---|---|
| contact_time_s | 1.0806 | 0.5605 | **+0.5201** | **+92.8%** | 0.1035 | 0.0022 |
| peak_s0 | 1249.4 | 3208.9 | −1959.5 | −61.1% | 192.3 | 18.3 |
| peak_s1 | 1307.8 | 1417.6 | −109.8 | −7.7% | 248.3 | 14.4 |
| peak_s2 | 1186.7 | 2607.9 | −1421.1 | −54.5% | 74.7 | 19.5 |
| peak_s3 | 1255.4 | 2808.7 | −1553.2 | −55.3% | 100.5 | 17.8 |
| peak_s4 | 1168.9 | 2210.7 | −1041.7 | −47.1% | 441.4 | 18.1 |
| peak_s5 | 1431.1 | 1800.1 | −369.0 | −20.5% | 371.9 | 20.9 |
| time_to_peak_s | 0.5133 | 0.4278 | +0.0855 | +20.0% | 0.2461 | 0.0108 |
| loading_rate_cps | 8756.9 | 14936.1 | −6179.1 | −41.4% | 5259.5 | 445.7 |
| cop_path_len | 0.8513 | 0.9824 | −0.1311 | −13.3% | 0.2100 | 0.0396 |
| cop_displacement | 0.4479 | 0.6991 | −0.2512 | −35.9% | 0.2178 | 0.0217 |
| cop_ml_range | 0.0651 | 0.1581 | −0.0930 | −58.8% | 0.0254 | 0.0021 |

### fast — real n = 48, sim n = 100

| metric | real mean | sim mean | Δ | Δ% | real sd | sim sd |
|---|---|---|---|---|---|---|
| contact_time_s | 0.7985 | 0.3310 | **+0.4675** | **+141.3%** | 0.0656 | 0.0030 |
| peak_s0 | 1456.5 | 3199.0 | −1742.4 | −54.5% | 254.4 | 24.5 |
| peak_s1 | 1479.4 | 1407.0 | +72.4 | +5.1% | 328.8 | 17.1 |
| peak_s2 | 1186.3 | 2599.4 | −1413.1 | −54.4% | 127.7 | 20.6 |
| peak_s3 | 1369.7 | 2797.9 | −1428.3 | −51.0% | 124.7 | 22.6 |
| peak_s4 | 1432.5 | 2201.7 | −769.2 | −34.9% | 207.3 | 22.4 |
| peak_s5 | 1747.1 | 1797.9 | −50.9 | −2.8% | 215.6 | 23.9 |
| time_to_peak_s | 0.4194 | 0.2501 | +0.1693 | +67.7% | 0.1522 | 0.0066 |
| loading_rate_cps | 11922.9 | 24483.9 | −12561.0 | −51.3% | 5290.6 | 674.9 |
| cop_path_len | 0.8843 | 0.8951 | −0.0107 | −1.2% | 0.2034 | 0.0298 |
| cop_displacement | 0.5940 | 0.7108 | −0.1168 | −16.4% | 0.1091 | 0.0208 |
| cop_ml_range | 0.0779 | 0.1561 | −0.0782 | −50.1% | 0.0300 | 0.0033 |

### shuffle — real n = 2, sim n = 120

Two real stances. Every real column here is a summary of two numbers and should
not be read as a distribution.

| metric | real mean | sim mean | Δ | Δ% | real sd | sim sd |
|---|---|---|---|---|---|---|
| contact_time_s | 0.9800 | 0.2403 | **+0.7397** | **+307.9%** | 0.0424 | 0.0016 |
| peak_s0 | 1327.0 | 1446.3 | −119.3 | −8.2% | 451.1 | 20.5 |
| peak_s1 | 1335.0 | 642.0 | +693.0 | +107.9% | 445.5 | 19.6 |
| peak_s2 | 1238.0 | 1177.6 | +60.5 | +5.1% | 14.1 | 21.7 |
| peak_s3 | 1247.5 | 1265.4 | −17.9 | −1.4% | 16.3 | 21.5 |
| peak_s4 | 535.0 | 994.2 | −459.2 | −46.2% | 277.2 | 21.3 |
| peak_s5 | 1057.0 | 811.2 | +245.8 | +30.3% | 9.9 | 21.8 |
| time_to_peak_s | 0.4350 | 0.1821 | +0.2529 | +138.9% | 0.5162 | 0.0086 |
| loading_rate_cps | 14218.7 | 12108.0 | +2110.7 | +17.4% | 13307.6 | 614.0 |
| cop_path_len | 0.8199 | 0.8223 | −0.0024 | −0.3% | 0.3553 | 0.0354 |
| cop_displacement | 0.3137 | 0.6339 | −0.3201 | −50.5% | 0.0662 | 0.0184 |
| cop_ml_range | 0.0548 | 0.1276 | −0.0728 | −57.1% | 0.0297 | 0.0035 |

**Two patterns run through all three activities.** Real contacts last
93–308% longer than simulated ones while peaking 35–61% lower on most channels
— real loading is slower and flatter than the simulator's sharp sinusoidal
pulse. And the real per-stance standard deviations are one to two orders of
magnitude larger than the simulator's on every metric (e.g. walk
`contact_time_s` sd 0.1035 real vs 0.0022 sim), which is the simulator's
stride-identical design showing up as near-zero variance.

### D4b — Stance-to-stance CoP placement spread

The within-stance range above says how far CoP travels sideways during one
contact. It says nothing about whether successive contacts were *placed*
differently — which is what a figure-8 path should produce and a straight
symmetric simulator should not. That needs a between-stance statistic: the mean
medial-lateral CoP position per stance, then the spread of that across stances.

| activity | n real | n sim | real sd | sim sd | ratio | real sd (mm) | sim sd (mm) | real range (mm) |
|---|---|---|---|---|---|---|---|---|
| walk    | 18 | 60  | 0.01037 | 0.00033 | **31.4×** | 2.84 | 0.090 | 9.24 |
| fast    | 48 | 100 | 0.00924 | 0.00046 | **19.9×** | 2.53 | 0.127 | 11.40 |
| shuffle | 2  | 120 | 0.01048 | 0.00059 | 17.7× | 2.87 | 0.162 | 4.06 |

Real stance placement varies 18–31× more than simulated, which is the direction
the figure-8 path predicts. But the absolute magnitude is **2.5–2.9 mm**, well
inside the ±15 mm uncertainty on the sensor coordinates themselves — so the
effect is not separable from the geometry error with this dataset. See finding 3.

## D5 — Sensor order check

Mean activation time within a stance for `walk_02`, relative to stance start.
Two independent timings: `t_onset`, the first frame a sensor exceeds 20% of its
own peak in that stance, and `t_peak`, the frame of its maximum.

All six sensors were present in all 18 detected walk stances.

| sensor | in n stances | t_onset (s) | sd | t_peak (s) | sd | anatomy |
|---|---|---|---|---|---|---|
| s0 | 18 | 0.0150 | 0.0432 | 0.2539 | 0.1901 | heel (medial) |
| s1 | 18 | 0.0111 | 0.0316 | 0.2222 | 0.2407 | heel (lateral) |
| s2 | 18 | 0.0594 | 0.0466 | 0.5728 | 0.2332 | lateral midfoot |
| s3 | 18 | 0.0933 | 0.0586 | 0.6428 | 0.2804 | 5th met head |
| s4 | 18 | **0.3283** | 0.1796 | 0.8039 | 0.2277 | 1st met head |
| s5 | 18 | 0.1700 | 0.1255 | 0.8567 | 0.2527 | hallux |

**Peak order: `s1 → s0 → s2 → s3 → s4 → s5`. No group-order violations.**
That is the anatomical sequence — heel pair, lateral midfoot, met heads, hallux.
**No channel pair appears swapped in firmware**, and nothing was changed. s1
leads s0 by 32 ms and both sit inside the same heel group, so their internal
order is unconstrained; the gap is far below either sensor's own spread
(sd 0.19–0.24 s) and carries no signal.

Onset order is `s1 → s0 → s2 → s3 → s5 → s4`, with exactly one group
violation: **s4 fires after s5**. This is the s4 activation threshold showing up
in the time domain rather than a wiring fault. s4 onset averages 0.3283 s into a
stance whose mean contact time is 1.0806 s (C6) — it stays below turn-on for the
first 30% of every contact, and it reads exactly zero in 52.02% of all walk
frames (C4). A sensor that cannot register the early part of a load will always
appear to fire late; the peak timing, which does not depend on when the sensor
crossed a threshold, puts s4 back in its anatomical place.

---

## Known non-faults

Documented behaviours that look like bugs and are not. None of these should be
"fixed" by dropping, imputing, or flagging data.

**s4 zeros are below-threshold, not missing data.** s4 (first metatarsal head)
has the highest activation threshold of the six; calibration measured s4 = 0
counts at 2.58 N while s5 read 239 at 2.49 N. A zero means the load is below
turn-on. It reads exactly zero in 99.12% of stand frames and 46–56% of the
moving captures (C4). Do not drop, impute, or interpolate these. The real
consequence is quantified in C4b: CoP degenerates to a 5-sensor centroid biased
33.6–36.7 mm laterally and posteriorly on the moving activities.

**The `_01` s0 dropout was a strain-relief failure, since fixed.** s0 reads flat
zero in `fast_01.csv` and `shuffle_01.csv` and comes alive mid-file in
`walk_01.csv` around seq 205–210 (`0 → 12 → 46 → 200` over a few frames). Root
cause was the FSR tail having no strain relief — the failure the repo's
open-items list had already flagged as most likely under walking. Strain relief
was added before the `_02` set. `_01` is retained as failure evidence and is
excluded from every number in this document: not analysed, not trained on, not
merged in.

**Correlated single-frame dips are a shared-reference glitch.** Scattered single
frames show s1/s2/s3/s5 dipping 5–8% together and recovering on the very next
frame. Because the dip is correlated across channels and exactly one frame wide,
it is a supply/ground-return artifact, not sensor behaviour — the
single-ground-return crosstalk already flagged for the PCB revision. Treat it as
an artifact, not real load.

**Sensor coordinates carry ±15 mm, and CoP inherits it.** Two independent
measurement passes over the same six sensors disagreed by 12–22 mm. Every CoP
number in this document carries that. It is large: ±15 mm is 5% of the insole
length and 16% of its width, and it exceeds the entire real stance-to-stance CoP
placement spread measured in D4b (2.5–2.9 mm).

---

## Top 3 sim-vs-reality findings

### 1. `MAX_DURATION` annihilates half of walking and 93% of shuffling — **SUPPORTED**

Real walk contacts have a median natural length of 119 frames against the
120-frame `MAX_DURATION` ceiling, so the threshold cuts through the middle of
the distribution and 17 of 35 walk runs and 28 of 30 shuffle runs are discarded
outright rather than clipped (C5 unbounded-run diagnostic,
[scripts/analyze_real.py](../scripts/analyze_real.py)). This could not have been predicted
before collection: `MAX_DURATION` was set to 120 because the longest *simulated*
stance was 58 frames, and nothing in the simulator suggested real contacts would
land within one frame of the limit.

### 2. The documented s4 bias ordering is wrong — **SUPPORTED** (since fixed)

`data/real/README.md` states the s4 lateral bias is "worst in stand and shuffle,
least in fast", but measured zero fractions are stand 99.12%, fast 56.05%, walk
52.02%, shuffle 46.38% (C4), and the common-substitute CoP bias is shuffle
36.70 mm, fast 36.46 mm, walk 33.55 mm, stand 8.58 mm (C4b) — fast is second
worst, not least, and shuffle is the *lowest* zero fraction of the three moving
activities. The same threshold shows up independently in the time domain, which
is what makes this a sensor property rather than a bookkeeping quirk: s4's mean
onset is 0.3283 s into a 1.0806 s stance (D5), so it misses the first 30% of
every contact.

### 3. Figure-8 asymmetry is present but below the geometry noise floor — **HYPOTHESIS**

Real stance-to-stance medial-lateral CoP placement varies 31.4× more than
simulated in walk and 19.9× more in fast (D4b, `scripts/sim_vs_real.py`), which is the
direction a continuously turning path predicts and a straight symmetric
simulator cannot produce. But the absolute spread is only 2.84 mm and 2.53 mm
against a ±15 mm coordinate uncertainty, so this run cannot separate real
turning asymmetry from measurement error on the sensor positions — it needs a
tighter geometry and a straight-line trial to compare against.

---

## Prioritized retune list

Ordered by expected impact. Confidence is stated per item, and items this
dataset **cannot** settle are marked and grouped at the end.

### A. Parameter retunes

**A1. Raise `MAX_DURATION` from 120 frames to ~200, or make it activity-aware.**
**Done: `MAX_DURATION = 200`.** The sweep is `scripts/sweep_max_duration.py`; walk
went 18 → 35 kept, shuffle 2 → 30, fast and stand unchanged.
Motivated by C5: 17/35 walk and 28/30 shuffle runs exceed 120 frames, natural
maxima are 144 (walk) and 164 (shuffle). Expected effect: walk detections rise
from 18 toward ~35 and shuffle from 2 toward ~30, and the selection bias in the
C5 survivors table disappears. **Confidence: high** — the mechanism is
arithmetic, and the numbers are unambiguous. Caveat: `MAX_DURATION` also
suppresses standing, which is a single 6000-frame run, so anything below ~1000
frames still rejects standing. There is a wide safe window here.

**A2. Re-derive `T_OFF` for shuffle.** Shuffle's natural runs have a median of
141 frames, longer than walk's 119 despite shorter strides, which means the
dragging foot is not unloading below `T_OFF = 450` between steps and adjacent
contacts merge. Expected effect: shuffle contacts separate and duration drops
toward a plausible per-step value. **Confidence: medium** — the merge is
inferred from run lengths, not directly observed; confirming it needs a
foot-off ground-truth signal this dataset does not have.

**A3. Leave `T_ON`, `MIN_DURATION` and `GAP_MERGE` alone for now.** No run in
any capture fell below `MIN_DURATION` (C5: 0 under-MIN across all four), and
`GAP_MERGE` changed nothing — raw and merged counts are identical in every
activity. **Confidence: high** that these are not currently binding; that is not
evidence they are correctly set.

**A4. Do not retune the sensor coordinates from this data.** The ±15 mm
uncertainty dominates every spatial result (finding 3), but nothing here
measures position better than the measuring passes did. This needs a third
measurement pass or a jig, not an analysis change.

**A5. IDW power is untouched and unmotivated by this data.** `IDW_POWER = 2.0`
was not exercised by anything measured here — the heatmap is a rendering aid and
no computed number depends on it. No change recommended, and none justified.

### B. Hardware fixes

**B1. s4 activation threshold.** Motivated by C4: s4 reads exactly zero in
46–99% of frames; C4b puts the resulting CoP bias at 33.6–36.7 mm on the moving
activities, and D5 shows s4's onset lagging 0.3283 s into a 1.0806 s stance —
the only anatomical-order violation in the whole capture. Options are a lower-threshold part for that position, a
preload shim, or a per-channel gain change. Expected effect: removes the single
largest known systematic error in CoP. **Confidence: high** that the problem is
real and large; **low** on which remedy is right, because this data does not
distinguish a sensor-threshold problem from a mounting/preload problem.

**B2. Shared ground return.** The correlated one-frame dips across s1/s2/s3/s5
are already attributed to a single ground return and slated for the PCB
revision. This dataset neither confirms nor refutes it — the artifact is
present, but no measurement here isolates the ground path. **Confidence: low**,
carried forward from prior work rather than established here.

**B3. Strain relief — already fixed, keep it.** The `_01` s0 dropout is the
evidence; `_02` shows no such failure in 6000 frames × 4 captures. **Confidence:
high** that the fix worked, on 4 minutes of data.

**B4. Calibrate above 824 counts.** Motivated by C3: 62–100% of real frames
sit above the highest count any calibration trial reached, and 100% of standing
frames do. Expected effect: the gain match stops extrapolating on the majority
of real data. **Confidence: high** on the need; the current single-point ~12 N
match is being asked to cover forces it never saw.

### C. Analysis changes

**C1. Leave the `insole/features.py` uniform-dt approximation as it is.** Motivated by
C6: the drift is at most 1 µs over the longest stance in any capture, because
the sampling interval is uniform to ±1 µs (C1). Expected effect of changing it:
none measurable. **Confidence: high** for this firmware; the moment sampling
becomes non-uniform (wireless, buffering, a different scheduler) this needs
re-measuring, and the check is cheap.

**C2. Re-examine the pooled covariance weighting and the Wald interval — but
not on this data.** `scripts/bakeoff.py` already flags that its stride-level standard
error treats near-duplicate strides as independent and that `accuracy_ci` is an
unclipped Wald interval. Both concerns are about the *simulated* bake-off and
are unchanged by anything measured here. **This data cannot settle either**: 68
real stances across one trial per class is not enough to fit or evaluate a model
on, which is exactly why D2 is labelled a plumbing check.

**C3. Stop reporting features from annihilated captures without the survivor
caveat.** Motivated by the C5 selection table: walk's surviving stances have a
kept median of 112.5 frames against a natural median of 119, and shuffle's two
survivors sit 58.03 s apart. Any mean over them is a mean over the short tail.
Expected effect: correctness of reporting, not of computation. **Confidence:
high.** This resolves itself once A1 lands — it has.

### What this data cannot settle

- **Whether the sensor coordinates are right.** ±15 mm swamps every spatial
  result; only a re-measurement helps (A4).
- **Whether the figure-8 asymmetry is real.** 2.5–2.9 mm of stance-to-stance
  spread against a 15 mm error bar (finding 3). Needs a straight-line trial as a
  control.
- **Any classifier question.** One 60 s trial per class. D2 is plumbing only.
- **The ground-return hypothesis** (B2) and **the right remedy for s4** (B1).
- **Whether `T_OFF` is the cause of shuffle's long runs** (A2) — no independent
  foot-off signal exists in this dataset.
- **Anything about absolute force.** `models/gain_match.json` is a single-point
  relative gain match and is not an absolute calibration; no newton figure in
  this document is anything but the calibration's own input.
