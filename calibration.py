"""calibration.py — per-sensor force calibration for the six FSRs.

Physical model
--------------
An FSR's *conductance* is what is linear in applied force, not its resistance
and not the raw ADC count. With the sensor in a divider against a fixed R, the
count is a monotone but nonlinear function of force; the transform

    x = counts / (FS_COUNTS - counts)

undoes the divider and leaves a quantity proportional to sensor conductance.
Force is then fit as a straight line in x:

    force_newtons = a * x + b

fit per sensor by ordinary least squares. Divider is 1 kOhm on all six
channels, so the divider constant is common to every sensor and is absorbed
into `a`; nothing here needs to know its value.

Units
-----
NEWTONS everywhere in this API. Grams appear only in capture filenames and in
what the operator types at the bench, and are converted at the boundary by
grams_to_newtons(). There is no gram-valued field in the persisted JSON.

Saturation is missing data
--------------------------
A reading at or above full scale carries no force information: the divider has
bottomed out and every force above the saturation point produces the same
count. conductance() returns None for those, and never clamps. Clamping would
silently report the saturation force for every heavier load, which reads as a
plausible flat-topped force trace rather than as the missing data it is.

This module is deliberately stdlib-only. It has to import on a bench laptop
during a capture session, where numpy may not be installed; the fit is a
closed-form least squares over at most a few dozen points and does not need it.
"""

from __future__ import annotations

import json
import math
import time

__all__ = [
    "FS_COUNTS", "G_MPS2", "CAL_MAX_COUNTS",
    "conductance", "grams_to_newtons", "is_saturated",
    "fit_sensor", "missing_fit", "is_usable",
    "save_calibration", "load_calibration", "apply_calibration",
    "load_gain_match", "apply_gain_match",
    "FLAG_OK", "FLAG_FLAT", "FLAG_FEW_POINTS", "FLAG_DEGENERATE",
    "FLAG_POOR_FIT", "FLAG_NO_DATA",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# UNMEASURED PLACEHOLDER. 4095 is the arithmetic full scale of a 12-bit ADC,
# which is NOT the same thing as the count at which this divider saturates.
# The real value is a bench measurement: load a sensor until the count stops
# rising with added weight, and record where it pins. It will come in below
# 4095 -- ESP32-S3 ADCs compress near the rails and the divider adds its own
# ceiling -- and every conductance in this file is wrong in proportion to how
# wrong this is. Measure it before trusting any fit.
#
# Overridable three ways, in increasing precedence:
#   1. this module constant
#   2. the "fs_counts" field of a loaded calibration.json
#   3. the fs= argument to conductance() / fit_sensor()
FS_COUNTS = 4095

G_MPS2 = 9.80665                # standard gravity, grams -> newtons

# A sensor whose count barely moves across the entire weight sweep is dead,
# unseated, or not under the indenter. Fitting a line to it produces a huge
# `a` from pure noise, which then reads as a hypersensitive channel rather
# than a broken one. Range is measured in raw counts over every reading in the
# sweep, saturated ones included.
FLAT_COUNT_RANGE = 100          # RETUNE: bench value, needs the real noise floor

# Below this the line does not describe the data, whatever the cause --
# hysteresis, indenter slipping between loads, a sensor going nonlinear.
MIN_R2 = 0.90                   # RETUNE: tighten once a good sensor's R2 is known

# Two points define a line exactly and leave no residual to judge it by.
MIN_FIT_POINTS = 3              # RETUNE

# Near-saturation guard. x = c/(fs - c) blows up as c approaches fs: a reading
# one count below the ceiling gives x ~ 1000x that of a mid-range reading, and
# one such point silently dominates a least-squares fit that every other point
# barely influences. Worse, at one noise sigma below the ceiling a reading is
# not distinguishable from a saturated one that noise happened to nudge down.
#
# So readings within this many counts of fs are treated as SATURATED --
# counted in n_saturated, excluded from the fit, None from apply_calibration.
# The default is one gait_gen.NOISE_STD. Set it to 0 for the literal
# counts >= fs rule and nothing more.
NEAR_SATURATION_MARGIN = 25     # RETUNE: set from the real noise floor

# The highest raw ADC count reached by ANY sample in ANY calibration capture
# (cal_data/cal_s5_t3.csv; the highest per-trial mean in
# cal_data/calibration_manifest.csv is 809.11). Above this the gain match is
# extrapolating past every load it was ever derived from. Walking peaks reach
# ~1600-2000 counts, so in normal use most loaded frames are above it; the
# number exists so that fraction can be REPORTED, not so anything is clamped
# or refused. An earlier figure of ~940 circulated in analyze_real.py and
# docs/sim_vs_real.md; nothing in cal_data/ supports it.
# test_infer_live.py recomputes this from cal_data/ and fails if it drifts.
CAL_MAX_COUNTS = 824

# --- Out-of-range policy -----------------------------------------------------
# True:  extrapolate the line past the calibrated conductance range.
# False: return None outside [x_min, x_max] recorded at fit time.
# Flip this one line to change the policy. See the DECISIONS block in the
# commit message / README -- this default is a judgement call, not physics.
EXTRAPOLATE = True

# The fitted intercept b is generally nonzero, so the model can return a small
# negative force for an unloaded sensor. False reports it as-is, because a
# persistently negative unloaded reading is a real signal that b is wrong.
CLAMP_NEGATIVE_FORCE = False

CAL_SCHEMA = 1

FLAG_OK         = "ok"
FLAG_FLAT       = "flat"            # count range below FLAT_COUNT_RANGE
FLAG_FEW_POINTS = "few_points"      # fewer than MIN_FIT_POINTS usable loads
FLAG_DEGENERATE = "degenerate"      # no spread in x; line is undefined
FLAG_POOR_FIT   = "poor_fit"        # R2 below MIN_R2
FLAG_NO_DATA    = "no_data"         # no capture files for this sensor at all

N_SENSORS = 6


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _as_float(v):
    """Best-effort float, None for anything non-numeric or non-finite."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def conductance(counts, fs=FS_COUNTS):
    """counts -> x = counts / (fs - counts), or None when the reading is unusable.

    None is returned, never a clamped value, for:
      counts >= fs   saturated; the divider has bottomed out and the true
                     force is unknowable from this sample
      counts <= 0    open circuit, unseated sensor, or no contact
      non-numeric / non-finite input

    Returning None rather than a large number is the whole point. A saturated
    sample is missing data and must propagate as missing data.
    """
    c = _as_float(counts)
    if c is None:
        return None
    f = _as_float(fs)
    if f is None or f <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}")
    if c >= f:
        return None
    if c <= 0:
        return None
    return c / (f - c)


def is_saturated(counts, fs=FS_COUNTS, margin=None):
    """True when a reading is at, or too close to, the divider ceiling.

    Kept separate from conductance() on purpose: conductance() implements the
    literal physical rule (c >= fs is unrepresentable), while this adds the
    numerical-leverage margin above it. Callers that want the raw transform
    still get exactly that.
    """
    c = _as_float(counts)
    if c is None:
        return False
    f = _as_float(fs)
    if f is None or f <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}")
    m = NEAR_SATURATION_MARGIN if margin is None else margin
    return c >= (f - m)


def grams_to_newtons(grams):
    """Bench mass in grams -> force in newtons. The only grams->N conversion."""
    g = _as_float(grams)
    if g is None:
        raise ValueError(f"grams must be numeric, got {grams!r}")
    return g * G_MPS2 / 1000.0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _result(a, b, r2, n_points, n_saturated, n_dropped, residuals, flag,
            x_min=None, x_max=None, count_range=None, fs=FS_COUNTS):
    return {
        "a": a,
        "b": b,
        "r2": r2,
        "n_points": n_points,
        "n_saturated": n_saturated,
        "n_dropped": n_dropped,
        "residuals": residuals,
        "flag": flag,
        "x_min": x_min,
        "x_max": x_max,
        "count_range": count_range,
        "fs_counts": fs,
    }


def missing_fit(fs=FS_COUNTS):
    """The fit record for a sensor with no capture files. Never usable."""
    return _result(None, None, None, 0, 0, 0, [], FLAG_NO_DATA, fs=fs)


def fit_sensor(grams, counts, fs=FS_COUNTS):
    """Least-squares fit of force_newtons = a*x + b for one sensor.

    `grams` and `counts` are parallel sequences, one entry per bench load:
    the mass placed on the sensor and the steady-state count it produced.

    Returns a dict with:
      a, b          fit coefficients, newtons per unit x and newtons. None
                    when the sensor could not be fit at all.
      r2            coefficient of determination over the points used, or None
                    when every load produced the same force (no variance to
                    explain).
      n_points      points actually used in the fit
      n_saturated   readings at or above full scale. Counted, never clamped.
      n_dropped     readings excluded for ANY reason, saturation included, so
                    n_points + n_dropped == len(counts) always holds.
      residuals     newtons, one per used point, in input order
      flag          FLAG_OK, or the reason this fit must not be applied
      x_min, x_max  calibrated conductance range, for the extrapolation policy
      count_range   raw count spread across the whole sweep

    A flat or dead sensor is FLAGGED, not silently fit. Where a line can still
    be computed the coefficients are returned alongside the flag so the numbers
    are visible on the bench, but is_usable() and apply_calibration() both
    refuse anything not FLAG_OK, so a flagged sensor can never leak into data
    as a force.
    """
    grams = list(grams)
    counts = list(counts)
    if len(grams) != len(counts):
        raise ValueError(
            f"grams and counts must be the same length, got {len(grams)} and {len(counts)}")

    fs_f = _as_float(fs)
    if fs_f is None or fs_f <= 0:
        raise ValueError(f"fs must be a positive number, got {fs!r}")

    n_total = len(counts)
    xs, ys = [], []
    n_saturated = 0
    finite_counts = []

    for g_, c_ in zip(grams, counts):
        cf = _as_float(c_)
        if cf is not None:
            finite_counts.append(cf)
            if is_saturated(cf, fs_f):
                n_saturated += 1
                continue        # excluded, never clamped
        x = conductance(c_, fs_f)
        if x is None:
            continue
        y = _as_float(g_)
        if y is None:
            continue
        xs.append(x)
        ys.append(grams_to_newtons(g_))

    n_points = len(xs)
    n_dropped = n_total - n_points
    count_range = (max(finite_counts) - min(finite_counts)) if finite_counts else 0.0
    x_min = min(xs) if xs else None
    x_max = max(xs) if xs else None

    def out(a, b, r2, residuals, flag):
        return _result(a, b, r2, n_points, n_saturated, n_dropped, residuals,
                       flag, x_min, x_max, count_range, fs_f)

    # Order matters. n_points is checked first: a sensor that saturated on
    # every load also has a tiny count range, and reporting that as "flat"
    # would point the operator at a dead channel when the real problem is that
    # every weight was too heavy.
    if n_points < MIN_FIT_POINTS:
        return out(None, None, None, [], FLAG_FEW_POINTS)

    if count_range < FLAT_COUNT_RANGE:
        return out(None, None, None, [], FLAG_FLAT)

    n = float(n_points)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))

    if sxx <= 0.0:
        return out(None, None, None, [], FLAG_DEGENERATE)

    a = sxy / sxx
    b = my - a * mx

    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None

    if r2 is not None and r2 < MIN_R2:
        return out(a, b, r2, residuals, FLAG_POOR_FIT)

    return out(a, b, r2, residuals, FLAG_OK)


def is_usable(fit):
    """True only for a fit that may be applied to real data."""
    return bool(fit) and fit.get("flag") == FLAG_OK and fit.get("a") is not None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_calibration(path, per_sensor, fs=FS_COUNTS, notes=None):
    """Write calibration.json. `per_sensor` maps sensor index -> fit dict.

    The FS_COUNTS actually used is recorded in the file, so a calibration
    taken before the saturation count was measured stays interpretable
    afterwards instead of silently inheriting a changed module constant.
    """
    doc = {
        "schema": CAL_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": "newtons",
        "model": "force_n = a * (counts / (fs_counts - counts)) + b",
        "fs_counts": _as_float(fs),
        "g_mps2": G_MPS2,
        "thresholds": {
            "flat_count_range": FLAT_COUNT_RANGE,
            "min_r2": MIN_R2,
            "min_fit_points": MIN_FIT_POINTS,
            "near_saturation_margin": NEAR_SATURATION_MARGIN,
        },
        "sensors": {str(int(k)): v for k, v in sorted(per_sensor.items())},
    }
    if notes:
        doc["notes"] = notes
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    return doc


def load_calibration(path):
    """Read calibration.json. Sensor keys come back as ints, not strings."""
    with open(path, "r") as f:
        doc = json.load(f)
    doc["sensors"] = {int(k): v for k, v in doc.get("sensors", {}).items()}
    if _as_float(doc.get("fs_counts")) is None:
        doc["fs_counts"] = FS_COUNTS
    return doc


GAIN_MATCH_KIND = "relative_gain_match"


def load_gain_match(path="gain_match.json"):
    """Read a relative-gain-match gain_match.json.

    Written by fit_calibration.derive_gain_match. The relative gain match lives
    in its OWN file, separate from the legacy absolute fit in calibration.json,
    so the two documents -- different schemas entirely -- can never overwrite or
    be mistaken for one another. Correction keys come back as ints (JSON stores
    them as strings) and their values as floats; fs_counts falls back to the
    module default if absent. apply_gain_match() consumes exactly this dict.

    Raises ValueError when the document's "kind" is absent or not
    "relative_gain_match": pointed at a legacy absolute fit (or anything else),
    it fails loudly here rather than reading a/b coefficients as if they were
    gains and returning silently wrong numbers.
    """
    with open(path, "r") as f:
        doc = json.load(f)
    kind = doc.get("kind")
    if kind != GAIN_MATCH_KIND:
        raise ValueError(
            f"{path}: not a relative gain match (kind={kind!r}, expected "
            f"{GAIN_MATCH_KIND!r}); refusing to read gains from it")
    doc["corrections"] = {int(k): _as_float(v)
                          for k, v in doc.get("corrections", {}).items()}
    if _as_float(doc.get("fs_counts")) is None:
        doc["fs_counts"] = FS_COUNTS
    return doc


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def apply_calibration(frame, cal, extrapolate=None):
    """Six raw counts -> six forces in newtons, None where unknowable.

    `frame` is the six-int list read_serial.parse_frame() already returns as
    the third element of its row tuple, unchanged and unreshaped:

        row, reason = read_serial.parse_frame(line)
        seq, ts_us, vals = row
        forces = apply_calibration(vals, cal)

    A channel is None when it saturated, read non-positive, or its sensor is
    flagged. None means "no force value exists for this sample", and must not
    be filled in downstream with a zero.
    """
    vals = list(frame)
    if len(vals) != N_SENSORS:
        raise ValueError(
            f"frame must hold {N_SENSORS} counts, got {len(vals)}")

    if extrapolate is None:
        extrapolate = EXTRAPOLATE

    fs = _as_float(cal.get("fs_counts")) or FS_COUNTS
    sensors = cal.get("sensors", {})

    out = []
    for i, c in enumerate(vals):
        fit = sensors.get(i)
        if fit is None:
            fit = sensors.get(str(i))
        if not is_usable(fit):
            out.append(None)
            continue

        if is_saturated(c, fs):             # at or near the ceiling
            out.append(None)
            continue

        x = conductance(c, fs)
        if x is None:                       # non-positive or non-numeric
            out.append(None)
            continue

        if not extrapolate:
            lo, hi = fit.get("x_min"), fit.get("x_max")
            if lo is not None and hi is not None and not (lo <= x <= hi):
                out.append(None)
                continue

        force = fit["a"] * x + fit["b"]
        if CLAMP_NEGATIVE_FORCE and force < 0.0:
            force = 0.0
        out.append(force)

    return out


def apply_gain_match(frame, cal):
    """Six raw counts -> six gain-matched conductances, None where unknowable.

    `cal` is a document from load_gain_match(). The correction is a RELATIVE
    gain derived in CONDUCTANCE space: it multiplies

        x = counts / (fs_counts - counts)

    -- the same transform conductance() computes -- and NOT the raw counts.
    Multiplying the counts instead is a different, wrong correction, because the
    counts are a nonlinear function of conductance. The result per channel is

        correction[i] * x

    a dimensionless conductance with the six channels' gains matched to their
    common mean. It is not a force: this is a relative gain match, not an
    absolute calibration.

    A channel is None when it has no correction, saturated, or read non-positive.
    Saturation uses the SAME rule as apply_calibration -- is_saturated(), i.e. at
    or within NEAR_SATURATION_MARGIN counts of full scale -> None, never a
    clamped value. Both apply paths share one saturation policy so a caller's
    result does not depend on which function it happened to import.
    """
    vals = list(frame)
    if len(vals) != N_SENSORS:
        raise ValueError(
            f"frame must hold {N_SENSORS} counts, got {len(vals)}")

    fs = _as_float(cal.get("fs_counts")) or FS_COUNTS
    corrections = cal.get("corrections", {})

    out = []
    for i, c in enumerate(vals):
        g = corrections.get(i)
        if g is None:
            g = corrections.get(str(i))
        g = _as_float(g)
        if g is None:
            out.append(None)
            continue

        if is_saturated(c, fs):             # at or near the ceiling, exactly as
            out.append(None)                # apply_calibration treats it
            continue

        x = conductance(c, fs)              # None for counts <= 0, non-finite
        if x is None:
            out.append(None)
            continue

        out.append(g * x)

    return out
