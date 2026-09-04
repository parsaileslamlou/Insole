# data/real — first measured-position gait dataset

Real-hardware captures from the **right-foot** insole. This is the first dataset
collected against **physically measured** sensor positions (below). `insole/detector.py`
now derives `SENSOR_COORDS` from these measurements (commit 4e7d34f); the
placeholder geometry it carried when this set was collected is gone.

## Hardware state

- Right-foot insole, **6× FSR (Interlink FSR UX 402)**, soldered.
- **ESP32-S3**, **USB serial (tethered)**, **100 Hz** sampling.
- Frame format and checksum per the firmware (`firmware/insole/insole.ino`).

## Sensor positions (measured 2026-09-02)

Measured with the right insole lying **top-surface-up**. Each position is
`(x, y)` in **inches**, where:

- `x` = distance from the **medial (big-toe) edge**, across the width
- `y` = distance from the **heel end**, along the length

(The axes are given in this order because the geometry requires it: the insole is
3.6 in wide and 10.8 in long, so the first value — at most 3.2 — is the width axis
and the second — up to 10.0 — is the length axis.)

| sensor | x (in, from medial edge) | y (in, from heel) | anatomy |
|--------|--------------------------|-------------------|---------|
| s0 | 1.3 | 2.0  | heel (medial of the heel pair) |
| s1 | 2.0 | 2.0  | heel (lateral of the heel pair) |
| s2 | 3.0 | 6.0  | lateral midfoot |
| s3 | 3.2 | 7.3  | fifth metatarsal head |
| s4 | 1.0 | 8.0  | first metatarsal head |
| s5 | 1.0 | 10.0 | hallux (big toe) |

Insole outline measured **10.8 × 3.6 in = 274 × 91 mm** (length × width).

> **Corrected since.** When this set was collected, the top-level `README.md`
> and `insole/detector.py` stated the insole was **295 × 74 mm** and `SENSOR_COORDS`
> was a normalized placeholder. Both were wrong; commit 4e7d34f set
> `INSOLE_LEN_MM = 274`, `INSOLE_WIDTH_MM = 91` and derived `SENSOR_COORDS`
> from the table above (`tests/test_geometry.py` pins it).

### Measurement uncertainty

Roughly **±15 mm**. Two independent measuring passes of the same sensors differed
by **12–22 mm**, so any centre-of-pressure (CoP) computed from these positions
inherits that uncertainty.

## Trials

Four activities, **60 s each, 100 Hz, USB serial (tethered)**:

- **stand** — quiet standing.
- **walk** — a **figure-8** path around a small area, **not** straight-line.
  Continuous turning loads the foot asymmetrically and will differ from the
  straight, symmetric model in `insole/gait_gen.py`.
- **fast** — faster cadence, same figure-8 path.
- **shuffle** — feet dragging, short strides, minimal ground clearance.

## Three datasets — `_01`, `_02` and `_03` (all kept)

The second ("good") set is **not** named uniformly: `stand_02.csv` has an
underscore, but `fast02.csv`, `shuffle02.csv`, and `walk02.csv` do not. Files were
committed under their **as-captured names**, unchanged.

### `_01` — DO NOT TRAIN ON THIS SET

It has an intermittent **s0** connection:

- s0 reads **flat zero** in `fast_01.csv` and `shuffle_01.csv`.
- s0 **comes alive mid-file** in `walk_01.csv` — around **seq 205–210**, climbing
  `0 → 12 → 46 → 200` over a few frames.

Root cause: the FSR tail had **no strain relief** — the failure point the repo's
open-items list had already flagged as most likely under walking. It was fixed
with strain relief before the `_02` set. `_01` is kept as **failure evidence** and
for the writeup, **not** for training.

### `_02` — the good set

Collected **after** the strain-relief fix. Use this set for analysis/training.

### `_03` — second session per class

Collected **2026-09-03**, same right-foot insole, same tethered USB serial at
100 Hz, same small figure-8 area as `_02`. Named **uniformly** with an
underscore, unlike `_02`; `scripts/train_real.py` discovers either form by label
prefix, so the inconsistency above is not repeated here.

This set exists so that every moving class has **two sessions**. With one
session per class no per-session split was possible and every real-data accuracy
was a within-session number with the leakage that implies. `scripts/train_real.py`
switches from its time-blocked within-session split to **leave-one-session-out**
on its own once each class has two or more sessions. It was **not** run as part
of this collection.

> **Known breakage — read before running the analysis.** Adding this set makes
> `scripts/train_real.py` abort *before* it reaches that multi-session path. The
> guard at lines 440-442 sums stances per **label** and asserts against
> `EXPECTED_STANCES`, which pins the `_02` counts alone, so `walk` now reports
> **67 against an expected 35**. The minimal fix is to apply the pin to the `_02`
> files only rather than to per-label totals; the multi-session branch four lines
> below it (`multi_session`, `leave_one_session_out`) already exists and is what
> the guard is blocking. `tests/test_train_real.py` fails five checks until then
> — it also pins `meta["sessions"]` to the three `_02` names and asserts the
> generated document still says "no per-session split exists", both of which stop
> being true once the split flips. Deliberately left to whoever owns the
> analysis.

| file | date | activity | path shape | duration | valid | stances | notes |
|---|---|---|---|---|---|---|---|
| `stand_03.csv` | 2026-09-03 | stand | stationary, quiet standing | 60 s | 6001 | 0 | s2 nonzero on 0.4 % of frames; s4 nonzero on 98.8 % — the reverse of `stand_02.csv` |
| `walk_03.csv` | 2026-09-03 | walk | figure-8, small area | 60 s | 6001 | 32 | `walk02.csv` gave 35 |
| `fast_03.csv` | 2026-09-03 | fast | figure-8, same area, faster cadence | 60 s | 6001 | 45 | `fast02.csv` gave 48 |
| `shuffle_03.csv` | 2026-09-03 | shuffle | figure-8, feet dragging, minimal clearance | 60 s | 6001 | 34 | `shuffle02.csv` gave 30 |

Every file: 6001 valid frames in 60.0 s with `malformed=0 bad_checksum=0
seq_breaks=0 timing_breaks=0 resets=0 lost=0 (0.00 %)`, exit 0.

**s0 is healthy across this set** — nonzero on 59.2 % (walk), 51.0 % (fast) and
57.4 % (shuffle) of frames. The `_01` flat-zero signature does not recur, so the
strain relief is still holding.

**s2 during stand is unloading, not a fault.** `stand_03.csv` has s2 nonzero on
only 0.4 % of frames while the other five sit at 97–100 %. s2 reaches 1214–1281
counts and 66–96 % nonzero in this set's three moving files, so the channel
works; the lateral midfoot simply carries no load in this standing posture.
Note that the near-zero channel during stand **differs between sessions**: in
`_02` it was s4 (zero on 99.1 % of stand frames), here s4 is nonzero on 98.8 %
and s2 takes its place. Standing posture is not repeatable session to session,
which is itself an argument for the per-session split this set enables.

## s4 zeros are NOT missing data

s4 (first metatarsal head) has the **highest activation threshold** of the six
sensors. Calibration measured s4 reading **0 counts at 2.58 N**, while s5 read
**239 counts at 2.49 N**. s4 stays below turn-on during quiet standing and much of
shuffle, and only registers when push-off drives real force through the first
metatarsal head.

**A zero at s4 means "below threshold", not "missing".** Do **not** drop or impute
these as missing data.

Consequence: during those phases CoP is effectively a **5-sensor centroid** and
carries a **lateral bias that varies by activity**. Measured on this set by
`scripts/analyze_real.py` C4/C4b (see `docs/sim_vs_real.md`), fraction of frames with
s4 = 0 and the CoP displacement the zero is responsible for (common-substitute
counterfactual):

| activity | s4 = 0 frames | CoP bias (mm) |
|---|---|---|
| stand   | 99.12% | 8.58  |
| fast    | 56.05% | 36.46 |
| walk    | 52.02% | 33.55 |
| shuffle | 46.38% | 36.70 |

So among the three moving activities **fast is second worst**, and **shuffle
has the lowest zero fraction**. An earlier version of this file said the bias
was "worst in stand and shuffle, least in fast"; that was a guess made before
the numbers were run and is wrong on both counts. Stand's zero fraction is the
highest, but its bias in millimetres is the smallest because the other five
sensors carry a nearly static load there.

`insole/infer_live.py` counts s4 = 0 frames live and prints the fraction beside every
stance, so the bias is visible while a capture runs.

## Correlated single-frame dips

Scattered single frames show **s1/s2/s3/s5 dipping 5–8% together** and recovering
on the very next frame. Because the dip is **correlated across channels and exactly
one frame wide**, it is a **shared-reference / supply glitch, not sensor
behaviour** — the same single-ground-return crosstalk flagged for the PCB revision.
Treat it as an artifact, not real load.

## Calibration state

The gain corrections in **`models/gain_match.json`** apply to these captures. It is a
**single-point (~12 N), relative gain match only — not absolute force**, and it is
applied **in conductance space, not to raw counts**.

The highest count any calibration sample reached is **824**
(`calibration.CAL_MAX_COUNTS`, from `cal_data/`; the highest per-trial mean in
the manifest is 809). Loaded frames in these captures sit above that on
62–100% of frames per activity (`scripts/analyze_real.py` C3), so the gain match is
extrapolating on most of this data. That is reported, not corrected.

## Stance detection on this set

With `MAX_DURATION = 120` (the simulator-derived value these captures were first
run against) the detector discarded 17 of 35 walk contacts and 28 of 30 shuffle
contacts outright, because real contacts here last 84–164 frames. It was raised
to 200 on this evidence (`scripts/sweep_max_duration.py`); under that value every walk
and shuffle contact is kept and standing still yields zero stances.
