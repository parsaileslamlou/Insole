# Calibration notes

What is shipped in `gain_match.json` is a **single-point relative gain match**
across the six FSR channels. It is **not** an absolute force calibration, and
nothing downstream may treat it as one. It is kept in its own file, separate
from the legacy absolute fit's `calibration.json`, so the two documents -- which
have different schemas -- can never overwrite or be mistaken for one another;
`load_gain_match` refuses to read any file whose `"kind"` is not
`"relative_gain_match"`. This file records what was measured, what was not, and
the conditions under which the match holds.

## What was measured, and what was not

**Measured.** All six channels were pressed to the same force (~12 N) in a
single matched cycle, each after a comparable rest (>= 35 min since that
sensor's previous trial), and their counts recorded. From those six readings we
derive a per-channel gain that brings every channel onto a common scale. Because
all six were measured at the same force with the same recovery state, the FSR
drift (below) is common to all of them and **cancels in the channel-to-channel
ratio**. What survives is the relative gain between channels, and that is all
`gain_match.json` claims.

**Not measured.** An absolute counts -> newtons mapping. We have no bench-stable
per-channel force curve: the sensors relax too fast under sustained load (below)
for a multi-point absolute fit to mean anything in the bench time available. The
gain match tells you channel *i* reads high or low relative to the others at
~12 N. It does **not** tell you how many newtons a given count is.

### The derived numbers

From the matched cycle (all six at trial 6 in `cal_data/calibration_manifest.csv`),
with `x = count_mean / (FS_COUNTS - count_mean)`, `k = force_n / x`, and
`correction[i] = k_i / mean(k)`:

| sensor | trial | force_n (N) | count_mean | rest (min) |     k    | correction |
|:------:|:-----:|:-----------:|:----------:|:----------:|:--------:|:----------:|
|   0    |   6   |   12.0123   |   687.405  |    41.9    |  59.55   |   0.9900   |
|   1    |   6   |   11.9437   |   700.890  |    40.4    |  57.84   |   0.9616   |
|   2    |   6   |   11.7180   |   682.480  |    39.8    |  58.59   |   0.9741   |
|   3    |   6   |   11.7131   |   551.445  |    39.1    |  75.27   |   1.2513   |
|   4    |   6   |   11.8063   |   754.395  |    42.3    |  52.28   |   0.8692   |
|   5    |   6   |   11.5022   |   683.895  |    41.8    |  57.37   |   0.9538   |

`mean(k) = 60.15`, so the corrections have mean 1.0 by construction. The
correction is applied in **conductance space** -- it multiplies
`x = counts / (FS_COUNTS - counts)`, never the raw counts. Applying it to counts
is a different and wrong correction.

## Limitations

1. **Anchored at one force, ~12 N, on a nonlinear response.** The match was
   taken at a single load. An FSR's count is a nonlinear function of force, so a
   gain that equalizes the channels at 12 N does not equalize them elsewhere.
   The further the working load is from ~12 N, the more the match degrades.

2. **The working range is above anything sampled.** Walking peaks reach ~1900
   counts. The heaviest bench point (~20 N) only reached ~700-800 counts. So in
   normal use the channels sit *above* every load the match was built from --
   the match is extrapolated into a count range it never saw, on a response
   known to be nonlinear.

3. **Per-channel activation thresholds diverge at low force.** The channels do
   not turn on together. At ~2.5 N, s4 read **0 counts** while s5 read **239**.
   Below roughly 5 N the channels are not measuring the same thing and the gain
   match does not hold; treat low-force readings as unmatched.

## Drift characterization

The FSRs exhibit stress relaxation. Under repeated load at constant applied
force, counts fall by **~31% in 76 s**, recovering with a time constant of
**~20 min**. This was reproduced across **two independent bench sessions**, with
a clean gradient in count against rest interval -- the shorter the rest before a
trial, the lower the count, monotonically. This is exactly why an absolute fit
was abandoned and why the gain match is taken from a single matched cycle at a
uniform rest: the drift is real and large, and the only place it cancels is the
ratio between channels measured under the same conditions.

The 40+ rows in `cal_data/calibration_manifest.csv` are that drift-characterization
dataset. They are kept as evidence and must not be rewritten or trimmed.

## Timestamps are measured, not synthesized

The frames in `cal_data/cal_s*_t*.csv` are **live bench captures**, not synthetic.
The `ts_us` column is a real `esp_timer_get_time()` read: the firmware runs an
absolute-deadline scheduler that stamps `now` just after each `nextDueUs` deadline
passes, so `ts_us = nextDueUs + overshoot`. When that overshoot is constant the
deltas come out exactly uniform at 10 ms -- that uniformity is the *expected*
output of a working timer under this scheduler, **not** evidence of synthesis. The
proof is that other captures carry ±1 µs jitter (deltas of 9999/10001): no
`t0 + n·period` formula produces that, so the timestamps are measured, not
fabricated. The host capture path (`read_serial.parse_frame` -> `capture_calibration`)
passes `ts_us` straight through and never rewrites it.

None of this touches the gain match: `derive_gain_match` does **not** read `ts_us`
at all. Its rest intervals come from the manifest's wall-clock `timestamp_iso`
column, and its fit uses `count_mean` and the scale `force_n`.

## FS_COUNTS is still a placeholder

`FS_COUNTS = 4095` is the arithmetic full scale of a 12-bit ADC, **not** a
measured saturation count for this divider. It is an unverified placeholder. The
corrections are **FS-dependent** -- every `x = counts / (FS_COUNTS - counts)`
that feeds them changes when FS_COUNTS changes -- so if the real saturation
count is ever measured and differs from 4095, the gain match **must be
re-derived** from the manifest against the corrected value. Re-run:

```
python3 fit_calibration.py cal_data/calibration_manifest.csv --fs <measured>
```

## No transfer across hardware revisions

This calibration is specific to the sensors, divider, and board it was taken on.
It does **not** transfer across hardware revisions. After the PCB spin the whole
match must be re-captured and re-derived; do not carry `gain_match.json`
forward across a hardware change.
