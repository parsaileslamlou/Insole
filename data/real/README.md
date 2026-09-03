# data/real — first measured-position gait dataset

Real-hardware captures from the **right-foot** insole. This is the first dataset
collected against **physically measured** sensor positions (below), as opposed to
the normalized placeholder geometry still in `detector.py`.

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

> **Correction needed elsewhere.** The top-level `README.md` and `detector.py`
> still state the insole is **295 × 74 mm**. That figure is **wrong** — the
> measured insole is **274 × 91 mm** (shorter, and considerably wider). Note also
> that `SENSOR_COORDS` in `detector.py` is still the normalized placeholder
> geometry, not these measured positions. Both should be corrected from this
> dataset; that is out of scope for this commit and left as an open item.

### Measurement uncertainty

Roughly **±15 mm**. Two independent measuring passes of the same sensors differed
by **12–22 mm**, so any centre-of-pressure (CoP) computed from these positions
inherits that uncertainty.

## Trials

Four activities, **60 s each, 100 Hz, USB serial (tethered)**:

- **stand** — quiet standing.
- **walk** — a **figure-8** path around a small area, **not** straight-line.
  Continuous turning loads the foot asymmetrically and will differ from the
  straight, symmetric model in `gait_gen.py`.
- **fast** — faster cadence, same figure-8 path.
- **shuffle** — feet dragging, short strides, minimal ground clearance.

## Two datasets — `_01` and `_02` (both kept)

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

## s4 zeros are NOT missing data

s4 (first metatarsal head) has the **highest activation threshold** of the six
sensors. Calibration measured s4 reading **0 counts at 2.58 N**, while s5 read
**239 counts at 2.49 N**. s4 stays below turn-on during quiet standing and much of
shuffle, and only registers when push-off drives real force through the first
metatarsal head.

**A zero at s4 means "below threshold", not "missing".** Do **not** drop or impute
these as missing data.

Consequence: during those phases CoP is effectively a **5-sensor centroid** and
carries a **lateral bias that varies by activity** — worst in **stand** and
**shuffle**, least in **fast**.

## Correlated single-frame dips

Scattered single frames show **s1/s2/s3/s5 dipping 5–8% together** and recovering
on the very next frame. Because the dip is **correlated across channels and exactly
one frame wide**, it is a **shared-reference / supply glitch, not sensor
behaviour** — the same single-ground-return crosstalk flagged for the PCB revision.
Treat it as an artifact, not real load.

## Calibration state

The gain corrections in **`gain_match.json`** apply to these captures. It is a
**single-point (~12 N), relative gain match only — not absolute force**, and it is
applied **in conductance space, not to raw counts**.
