"""Regression tests for force calibration. Run from the repo root:

    python3 test_calibration.py

Stdlib only, like calibration.py itself, so these run on the bench laptop.

The synthetic sweep is generated through a KNOWN (a, b) and inverted back to
counts, so every test has a ground truth that does not come from the fitter.
Seeding a test from fit_sensor's own output would lock in today's bugs -- the
same rule test_stances.py follows with true_stances.

Noise model
-----------
Noise is put on the COUNTS, not on the masses: bench weights are known to far
better than the ADC is. Per-sample sigma is 25 counts, matching
gait_gen.NOISE_STD, which is itself a RETUNE placeholder awaiting a real
noise-floor capture. Each calibration point is then the MEDIAN of a 400-sample
window, exactly as fit_calibration.py forms it from a capture file, so what is
being tested is the noise the fit actually sees -- roughly 1.25*sigma/sqrt(N),
about 1.6 counts -- and not the raw per-sample noise.

Tolerance
---------
A_TOL_REL and B_TOL_N below are set at roughly 2x the worst error observed
over a 500-seed sweep of this exact generator (0.46% on a, 0.033 N on b). Two
reasons for the headroom rather than a tight fit to the observed worst case:
the test must not flap if the seed set or the window length is changed, and
the defects worth catching here -- a sign error, the conductance transform
inverted, grams treated as kilograms -- miss by orders of magnitude, not by
percent. A tolerance tight enough to catch a 0.5% bias would only be measuring
the RNG.
"""

import json
import math
import os
import random
import statistics
import sys
import tempfile

from calibration import (
    FS_COUNTS, G_MPS2, NEAR_SATURATION_MARGIN,
    FLAG_OK, FLAG_FLAT, FLAG_POOR_FIT, FLAG_NO_DATA,
    apply_calibration, conductance, fit_sensor, grams_to_newtons,
    is_saturated, is_usable, load_calibration, missing_fit, save_calibration,
)

# Ground truth for the synthetic sensor. Chosen so the bench sweep lands
# across 93..2200 counts: a wide, unsaturated, physically plausible span.
A_TRUE = 25.0                   # newtons per unit conductance
B_TRUE = 0.4                    # newtons

LOADS_G = [100, 200, 500, 1000, 1500, 2000, 3000]

PER_SAMPLE_NOISE = 25           # counts, matches gait_gen.NOISE_STD
WINDOW = 400                    # samples per load, ~4 s at 100 Hz after settle
SEEDS = 200

A_TOL_REL = 0.01                # 1% of a
B_TOL_N   = 0.08                # newtons


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    return bool(condition)


# ---------------------------------------------------------------------------
# Synthetic generator: known (a, b) -> counts
# ---------------------------------------------------------------------------
def counts_for(grams, a=A_TRUE, b=B_TRUE, fs=FS_COUNTS):
    """Invert force = a*x + b and x = c/(fs-c) to get the ideal count."""
    force = grams * G_MPS2 / 1000.0
    x = (force - b) / a
    return fs * x / (1.0 + x)


def noisy_sweep(seed, loads=LOADS_G, a=A_TRUE, b=B_TRUE):
    """One calibration point per load: the median of a noisy sample window."""
    rng = random.Random(seed)
    out = []
    for g in loads:
        ideal = counts_for(g, a, b)
        window = [ideal + rng.gauss(0, PER_SAMPLE_NOISE) for _ in range(WINDOW)]
        out.append(round(statistics.median(window)))
    return out


# ---------------------------------------------------------------------------
# 1. Recovery of a known line
# ---------------------------------------------------------------------------
def test_recovery():
    passed = failed = 0

    worst_a = worst_b = 0.0
    worst_r2 = 1.0
    bad = []
    for seed in range(SEEDS):
        f = fit_sensor(LOADS_G, noisy_sweep(seed))
        if f["a"] is None:
            bad.append((seed, f["flag"]))
            continue
        ea = abs(f["a"] - A_TRUE) / A_TRUE
        eb = abs(f["b"] - B_TRUE)
        worst_a, worst_b = max(worst_a, ea), max(worst_b, eb)
        worst_r2 = min(worst_r2, f["r2"])
        if ea > A_TOL_REL or eb > B_TOL_N:
            bad.append((seed, f"a={f['a']:.4f} b={f['b']:.4f}"))

    ok = check(f"{SEEDS} seeds recover a within {A_TOL_REL * 100:g}% "
               f"and b within {B_TOL_N:g} N",
               not bad,
               f"worst da/a={worst_a * 100:.3f}%  worst db={worst_b:.4f} N  "
               f"min R2={worst_r2:.6f}" if not bad else f"{len(bad)} failed: {bad[:3]}")
    passed, failed = (passed + ok, failed + (not ok))

    # Noise-free must be near exact: this catches transform errors that the
    # tolerance above is deliberately too loose to see.
    clean = fit_sensor(LOADS_G, [counts_for(g) for g in LOADS_G])
    ok = check("noise-free sweep recovers (a, b) to 1e-6",
               abs(clean["a"] - A_TRUE) < 1e-6 and abs(clean["b"] - B_TRUE) < 1e-6,
               f"a={clean['a']:.9f} b={clean['b']:.9f}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("clean fit is flagged ok with R2 == 1",
               clean["flag"] == FLAG_OK and abs(clean["r2"] - 1.0) < 1e-9)
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("bookkeeping: n_points + n_dropped == inputs",
               clean["n_points"] + clean["n_dropped"] == len(LOADS_G)
               and clean["n_points"] == len(LOADS_G))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("1000 g == 9.80665 N", abs(grams_to_newtons(1000) - G_MPS2) < 1e-12)
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


# ---------------------------------------------------------------------------
# 2. Flat / dead sensors are flagged, not fit
# ---------------------------------------------------------------------------
def test_flat_flagged():
    passed = failed = 0
    rng = random.Random(7)

    # A dead channel: reads the same regardless of load, noise only.
    flat = [900 + round(rng.gauss(0, 3)) for _ in LOADS_G]
    f = fit_sensor(LOADS_G, flat)

    ok = check("dead sensor flagged flat", f["flag"] == FLAG_FLAT,
               f"flag={f['flag']} count_range={f['count_range']:g}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("flat sensor yields NO coefficients",
               f["a"] is None and f["b"] is None,
               f"a={f['a']} b={f['b']}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("flat sensor is not usable", not is_usable(f))
    passed, failed = (passed + ok, failed + (not ok))

    # It must be the flag doing the work, not luck: an unguarded least squares
    # on this same data returns a large finite slope from pure noise.
    xs = [conductance(c) for c in flat]
    ys = [grams_to_newtons(g) for g in LOADS_G]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    naive_a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    ok = check("unguarded fit on the same data returns a bogus slope",
               abs(naive_a) > 10 * A_TRUE,
               f"naive a={naive_a:.1f} vs true {A_TRUE}")
    passed, failed = (passed + ok, failed + (not ok))

    # A sensor with a real span but a badly nonlinear response is a different
    # failure and gets a different flag.
    bent = [round(counts_for(g)) for g in LOADS_G]
    bent[3] = 3200
    bent[4] = 300
    f2 = fit_sensor(LOADS_G, bent)
    ok = check("nonlinear sensor flagged poor_fit", f2["flag"] == FLAG_POOR_FIT,
               f"flag={f2['flag']} r2={f2['r2']:.4f}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("poor_fit sensor is not usable", not is_usable(f2))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("no-data sensor flagged and unusable",
               missing_fit()["flag"] == FLAG_NO_DATA and not is_usable(missing_fit()))
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


# ---------------------------------------------------------------------------
# 3. Saturation returns None, is counted, and is never clamped
# ---------------------------------------------------------------------------
def test_saturation():
    passed = failed = 0

    for label, c in [("at full scale", FS_COUNTS),
                     ("above full scale", FS_COUNTS + 10),
                     ("zero", 0),
                     ("negative", -5)]:
        ok = check(f"conductance({label}) is None", conductance(c) is None,
                   f"got {conductance(c)}")
        passed, failed = (passed + ok, failed + (not ok))

    ok = check("conductance is finite and positive just below full scale",
               math.isfinite(conductance(FS_COUNTS - 1))
               and conductance(FS_COUNTS - 1) > 0)
    passed, failed = (passed + ok, failed + (not ok))

    # Sweep where the three heaviest loads pin the ADC, as a real one would.
    loads = LOADS_G + [5000, 7000, 9000]
    counts = [round(counts_for(g)) for g in LOADS_G] + [FS_COUNTS] * 3
    f = fit_sensor(loads, counts)

    ok = check("saturated readings counted", f["n_saturated"] == 3,
               f"n_saturated={f['n_saturated']}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("saturated readings excluded from the fit",
               f["n_points"] == len(LOADS_G)
               and f["n_points"] + f["n_dropped"] == len(loads),
               f"n_points={f['n_points']} n_dropped={f['n_dropped']}")
    passed, failed = (passed + ok, failed + (not ok))

    # Dropped, not merely down-weighted: the fit must be bit-identical to one
    # where the saturated loads were never presented.
    ref = fit_sensor(LOADS_G, [round(counts_for(g)) for g in LOADS_G])
    ok = check("fit identical to one with saturated points removed",
               f["a"] == ref["a"] and f["b"] == ref["b"],
               f"a={f['a']:.9f} vs {ref['a']:.9f}")
    passed, failed = (passed + ok, failed + (not ok))

    # And explicitly NOT what clamping would have produced. Clamp to the
    # highest count that still survives the near-saturation guard: the mildest
    # clamp available, and it still wrecks the fit.
    ceiling = FS_COUNTS - NEAR_SATURATION_MARGIN - 1
    clamped = fit_sensor(loads, [round(counts_for(g)) for g in LOADS_G]
                         + [ceiling] * 3)
    ok = check(f"fit differs from a fit clamped to {ceiling}",
               abs(clamped["a"] - f["a"]) > 1.0,
               f"dropped a={f['a']:.3f}  clamped a={clamped['a']:.3f}  "
               f"({100 * abs(clamped['a'] - f['a']) / f['a']:.0f}% off)")
    passed, failed = (passed + ok, failed + (not ok))

    # --- near-saturation leverage guard ---------------------------------
    # x = c/(fs-c) explodes as c approaches fs. One point a single count below
    # the ceiling carries ~1000x the leverage of a mid-range point and would
    # otherwise capture the whole fit.
    ok = check("conductance() keeps the literal >= fs rule",
               conductance(FS_COUNTS - 1) is not None)
    passed, failed = (passed + ok, failed + (not ok))

    ok = check(f"is_saturated() covers the {NEAR_SATURATION_MARGIN}-count margin",
               is_saturated(FS_COUNTS - 1)
               and is_saturated(FS_COUNTS - NEAR_SATURATION_MARGIN)
               and not is_saturated(FS_COUNTS - NEAR_SATURATION_MARGIN - 1))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("margin=0 restores the literal rule",
               not is_saturated(FS_COUNTS - 1, margin=0)
               and is_saturated(FS_COUNTS, margin=0))
    passed, failed = (passed + ok, failed + (not ok))

    # One near-ceiling point must not be allowed to drag the fit.
    near = [round(counts_for(g)) for g in LOADS_G] + [FS_COUNTS - 1]
    fn = fit_sensor(LOADS_G + [9000], near)
    ok = check("a point 1 count below fs is excluded, not allowed leverage",
               fn["n_saturated"] == 1 and fn["a"] == ref["a"],
               f"a={fn['a']:.6f} vs clean {ref['a']:.6f}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("near-ceiling count returns None from apply",
               _apply_one(FS_COUNTS - 1) is None
               and _apply_one(round(counts_for(1000))) is not None)
    passed, failed = (passed + ok, failed + (not ok))

    # All loads saturated is a distinct diagnosis from a dead sensor.
    allsat = fit_sensor(LOADS_G, [FS_COUNTS] * len(LOADS_G))
    ok = check("all-saturated sweep flagged, not fit",
               allsat["flag"] != FLAG_OK and allsat["a"] is None
               and allsat["n_saturated"] == len(LOADS_G),
               f"flag={allsat['flag']}")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


# ---------------------------------------------------------------------------
# 4. apply_calibration round-trip
# ---------------------------------------------------------------------------
def test_apply_round_trip():
    passed = failed = 0

    # Six sensors with distinct slopes, so a channel-ordering bug shows up.
    per_sensor = {}
    for i in range(6):
        a = A_TRUE + 5.0 * i
        b = B_TRUE + 0.1 * i
        per_sensor[i] = fit_sensor(LOADS_G, [counts_for(g, a, b) for g in LOADS_G])
    per_sensor[4] = fit_sensor(LOADS_G, [900] * len(LOADS_G))   # dead channel
    per_sensor[5] = missing_fit()                                # never captured

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "calibration.json")
        save_calibration(path, per_sensor)
        cal = load_calibration(path)

        ok = check("JSON round-trips fs_counts and units",
                   cal["fs_counts"] == FS_COUNTS and cal["units"] == "newtons"
                   and "created_utc" in cal)
        passed, failed = (passed + ok, failed + (not ok))

        ok = check("sensor keys come back as ints",
                   set(cal["sensors"]) == set(range(6)))
        passed, failed = (passed + ok, failed + (not ok))

        with open(path) as fh:
            raw = json.load(fh)
        ok = check("persisted JSON keys are strings",
                   set(raw["sensors"]) == {"0", "1", "2", "3", "4", "5"})
        passed, failed = (passed + ok, failed + (not ok))

        # A frame in exactly the shape read_serial.parse_frame() produces.
        frame = [round(counts_for(1000, A_TRUE + 5.0 * i, B_TRUE + 0.1 * i))
                 for i in range(4)] + [1500, FS_COUNTS]
        forces = apply_calibration(frame, cal)

        ok = check("apply returns six values", len(forces) == 6)
        passed, failed = (passed + ok, failed + (not ok))

        want = grams_to_newtons(1000)
        errs = [abs(forces[i] - want) for i in range(4)]
        ok = check("channels 0-3 recover the 1000 g load to 0.02 N",
                   all(e < 0.02 for e in errs),
                   "max err %.5f N" % max(errs))
        passed, failed = (passed + ok, failed + (not ok))

        ok = check("flagged (dead) channel returns None", forces[4] is None)
        passed, failed = (passed + ok, failed + (not ok))

        ok = check("no-data channel returns None", forces[5] is None)
        passed, failed = (passed + ok, failed + (not ok))

        # Saturated and zero counts on an otherwise good channel.
        sat = apply_calibration([FS_COUNTS, 0, -1, 2000, 2000, 2000], cal)
        ok = check("saturated / zero / negative counts return None",
                   sat[0] is None and sat[1] is None and sat[2] is None
                   and sat[3] is not None)
        passed, failed = (passed + ok, failed + (not ok))

        ok = check("wrong frame width is rejected",
                   _raises(lambda: apply_calibration([1, 2, 3], cal)))
        passed, failed = (passed + ok, failed + (not ok))

        # Extrapolation policy, both ways, on a count well past the sweep.
        far = round(counts_for(6000))
        on = apply_calibration([far] * 6, cal, extrapolate=True)
        off = apply_calibration([far] * 6, cal, extrapolate=False)
        ok = check("extrapolate=True returns a value past the calibrated range",
                   on[0] is not None, f"{on[0]:.3f} N")
        passed, failed = (passed + ok, failed + (not ok))

        ok = check("extrapolate=False returns None past the calibrated range",
                   off[0] is None)
        passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def _apply_one(count):
    """Run one count through apply_calibration on a known-good sensor 0."""
    cal = {"fs_counts": FS_COUNTS,
           "sensors": {0: fit_sensor(LOADS_G, [counts_for(g) for g in LOADS_G])}}
    return apply_calibration([count] + [0] * 5, cal)[0]


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in (test_recovery, test_flat_flagged, test_saturation,
                  test_apply_round_trip):
        print(f"--- {suite.__name__} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()

    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
