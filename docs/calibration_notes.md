# Calibration notes

What is shipped in `models/gain_match.json` is a **single-point relative gain match**
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
`models/gain_match.json` claims.

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

## The estimator: unweighted OLS in conductance

The line fit that `calibration.fit_sensor` ships is **ordinary least squares,
unweighted**, linear in conductance:

```
x = c / (FS_COUNTS - c)          # c = counts, FS_COUNTS = 4095
force_n = a * x + b              # a = Sxy / Sxx, b = mean(y) - a * mean(x)
```

Every point in the sweep carries equal weight. `r2` is the ordinary
coefficient of determination over the points kept, and a channel whose `r2`
falls below `MIN_R2` (0.90) is flagged `poor_fit`.

A **weighted variant was evaluated and rejected.** The argument for it is real:
propagating a uniform count noise through the conductance transform gives
`sigma_x = sigma_c * FS_COUNTS / (FS_COUNTS - c)^2`, so the noise in `x` grows
quadratically while `x` itself grows linearly, and unweighted OLS therefore
hands the top of the sweep more leverage than its information justifies. The
variant reweights each point by `1 / sigma_x^2`. It was not adopted, for three
reasons measured on the captures this repository actually contains:

1. **It moves the slopes without changing any decision.** Weighting shifts the
   fitted slope by 2.5-7.1% on five of the six channels.
2. **It changes no flag.** Every channel that the shipped OLS fit marks
   `poor_fit` is still `poor_fit` under the weighted fit -- all five of them.
   No channel crosses `MIN_R2` in either direction, so nothing downstream sees
   a different answer.
3. **Its motivating number does not hold here.** The variant's own commit
   message justifies the change with a 19.9x information ratio between the
   lightest and heaviest load. That figure is computed for a hypothetical
   100..3000 g ladder, not for the sweep on disk. On the captures that exist
   the ratio is **1.9-2.3x** -- an order of magnitude smaller, and small enough
   that the leverage imbalance the weighting corrects is not the dominant error
   here. The limitations above -- a single-force anchor, a working range above
   everything sampled, and an unverified `FS_COUNTS` -- all dominate it.

So the weighting is a defensible refinement to an estimator whose error budget
is not currently limited by its weights. It was left out to keep the shipped
fit the simpler of two estimators that agree on every flag.

That evaluation is not on any branch of this repository. It lives in a local
git bundle, `insole_archive_branches.bundle`, on the `cal-wls-local` branch
inside it, together with the sweep-planning and range-reporting work built on
top of it. Restore it with:

```
git bundle verify insole_archive_branches.bundle
git clone insole_archive_branches.bundle wls && cd wls && git checkout cal-wls-local
```

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
python -m insole.fit_calibration cal_data/calibration_manifest.csv -o models/gain_match.json --fs <measured>
```

## No transfer across hardware revisions

This calibration is specific to the sensors, divider, and board it was taken on.
It does **not** transfer across hardware revisions. After the PCB spin the whole
match must be re-captured and re-derived; do not carry `models/gain_match.json`
forward across a hardware change.
