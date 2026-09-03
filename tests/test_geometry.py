"""Regression tests for sensor geometry and centre of pressure. Run:

    python tests/test_geometry.py

CoP is a weighted mean, so it has closed-form answers that do not depend on
the detector, on any capture, or on the simulator. Those answers are what this
file checks. Every expected value is DERIVED from detector.SENSOR_MM and
detector.INSOLE_LEN_MM rather than typed as a decimal, so a re-measurement
moves the geometry and the expectations together instead of turning this file
into a set of assertions about coordinates nobody uses any more.

The one exception is test_documented_values, which deliberately pins the
CURRENT numbers as literals. It is a canary: if a re-measurement lands, that
suite is SUPPOSED to fail, and its failure means the figures quoted in
README.md and docs/ have gone stale and need rewriting. It is not a bug.

Orientation the tests enforce, from data/real/README.md:
    x = distance from the MEDIAL edge, so x = 0 is medial, larger x is lateral
    y = distance from the HEEL end,    so y = 0 is heel,   larger y is toe
"""

import sys

import numpy as np

from insole import detector as D
from insole.features import cop_frame, cop_features

# Expected coordinates, rebuilt from the raw millimetre table rather than read
# back out of SENSOR_COORDS. Comparing SENSOR_COORDS against itself would pass
# no matter what the normalisation did; this recomputes it independently.
EXPECTED_COORDS = {
    name: (x_mm / D.INSOLE_LEN_MM, y_mm / D.INSOLE_LEN_MM)
    for name, (x_mm, y_mm) in D.SENSOR_MM.items()
}

HEEL_PAIR = ("s0", "s1")
TOE = "s5"
LATERAL_MET = "s3"      # 81.3 mm from the medial edge
MEDIAL_MET = "s4"       # 25.4 mm from the medial edge

TOL = 1e-12             # these are exact arithmetic, not fits


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    return bool(condition)


def loaded(**counts):
    """A frame with the named sensors at the given counts and the rest at 0."""
    row = {name: 0.0 for name in D.SENSOR_COLS}
    row.update(counts)
    return row


def weighted_mean(names, weights=None):
    """Hand-computed CoP: the plain weighted mean of the named coordinates."""
    w = [1.0] * len(names) if weights is None else list(weights)
    total = float(sum(w))
    xs = sum(wi * EXPECTED_COORDS[n][0] for wi, n in zip(w, names)) / total
    ys = sum(wi * EXPECTED_COORDS[n][1] for wi, n in zip(w, names)) / total
    return (xs, ys)


def close(got, want, tol=TOL):
    return (not any(np.isnan(v) for v in got)
            and abs(got[0] - want[0]) <= tol
            and abs(got[1] - want[1]) <= tol)


def test_coords_derived():
    """SENSOR_COORDS must be SENSOR_MM / INSOLE_LEN_MM on BOTH axes.

    Guards the normalisation choice itself. Dividing x by the width instead
    would stretch the short axis by 274/91 = 3.01 and silently distort every
    CoP path length; that mistake would survive every other test in this file
    because it moves the sensors and the expectations in lockstep. Here the
    expectation is rebuilt from the mm table and the length alone.
    """
    passed = failed = 0

    ok = check("SENSOR_COORDS covers exactly SENSOR_COLS",
               sorted(D.SENSOR_COORDS) == sorted(D.SENSOR_COLS),
               f"got={sorted(D.SENSOR_COORDS)}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("SENSOR_COORDS preserves s0..s5 ordering",
               list(D.SENSOR_COORDS) == D.SENSOR_COLS,
               f"got={list(D.SENSOR_COORDS)}")
    passed, failed = (passed + ok, failed + (not ok))

    for name in D.SENSOR_COLS:
        want = EXPECTED_COORDS[name]
        got = D.SENSOR_COORDS[name]
        ok = check(f"{name} normalised by length on both axes",
                   close(got, want),
                   f"got=({got[0]:.6f}, {got[1]:.6f})")
        passed, failed = (passed + ok, failed + (not ok))

    # Round-trip: coordinate * 274 must give the measured millimetres back.
    # This is the whole point of scaling both axes by the same constant.
    for name in D.SENSOR_COLS:
        mm = D.SENSOR_MM[name]
        back = tuple(v * D.INSOLE_LEN_MM for v in D.SENSOR_COORDS[name])
        ok = check(f"{name} round-trips to millimetres",
                   close(back, mm, tol=1e-9),
                   f"got=({back[0]:.4f}, {back[1]:.4f}) want={mm}")
        passed, failed = (passed + ok, failed + (not ok))

    # Every sensor must sit inside the authoritative 274 x 91 rectangle.
    for name in D.SENSOR_COLS:
        x_mm, y_mm = D.SENSOR_MM[name]
        ok = check(f"{name} lies inside the 274 x 91 mm rectangle",
                   0.0 <= x_mm <= D.INSOLE_WIDTH_MM and 0.0 <= y_mm <= D.INSOLE_LEN_MM,
                   f"({x_mm}, {y_mm}) mm")
        passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_single_sensor():
    """One sensor loaded, the other five at zero -> CoP is exactly that sensor.

    A weighted mean over a single non-zero weight has to return that point
    regardless of the weight's size, so this also varies the count to prove
    the normalisation by total_weight actually happens.
    """
    passed = failed = 0
    for name in D.SENSOR_COLS:
        for counts in (1.0, 137.0, 4095.0):
            got = cop_frame(loaded(**{name: counts}))
            want = EXPECTED_COORDS[name]
            ok = check(f"{name} alone at {counts:6.0f} counts -> {name} coords",
                       close(got, want),
                       f"got=({got[0]:.6f}, {got[1]:.6f}) want=({want[0]:.6f}, {want[1]:.6f})")
            passed, failed = (passed + ok, failed + (not ok))
    return passed, failed


def test_equal_loading():
    """Equal counts -> the unweighted mean of the loaded sensors' coordinates."""
    passed = failed = 0

    got = cop_frame(loaded(**{n: 1000.0 for n in D.SENSOR_COLS}))
    want = weighted_mean(D.SENSOR_COLS)
    ok = check("all six equal -> unweighted mean of all six",
               close(got, want),
               f"got=({got[0]:.6f}, {got[1]:.6f}) want=({want[0]:.6f}, {want[1]:.6f})")
    passed, failed = (passed + ok, failed + (not ok))

    got = cop_frame(loaded(s0=500.0, s1=500.0))
    want = weighted_mean(HEEL_PAIR)
    ok = check("heel pair equal -> midpoint of s0 and s1",
               close(got, want),
               f"got=({got[0]:.6f}, {got[1]:.6f}) want=({want[0]:.6f}, {want[1]:.6f})")
    passed, failed = (passed + ok, failed + (not ok))

    # A 3:1 split must land three-quarters of the way toward the heavier
    # sensor. Equal-weight cases alone cannot catch a weights/total mix-up.
    got = cop_frame(loaded(s0=300.0, s1=100.0))
    want = weighted_mean(HEEL_PAIR, weights=(3.0, 1.0))
    ok = check("heel pair 3:1 -> weighted toward s0",
               close(got, want),
               f"got=({got[0]:.6f}, {got[1]:.6f}) want=({want[0]:.6f}, {want[1]:.6f})")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_orientation():
    """Medial/lateral and heel/toe must not be silently transposed.

    The value tests above would all still pass if x and y were swapped
    everywhere, or if the x axis were mirrored, because they compare against
    the same table the code reads. These compare against anatomy instead.
    """
    passed = failed = 0

    x_lateral = cop_frame(loaded(**{LATERAL_MET: 1000.0}))[0]
    x_medial = cop_frame(loaded(**{MEDIAL_MET: 1000.0}))[0]
    ok = check(f"x({LATERAL_MET}) > x({MEDIAL_MET})  [lateral is larger x]",
               x_lateral > x_medial,
               f"{x_lateral:.6f} vs {x_medial:.6f}  (delta {x_lateral - x_medial:+.6f})")
    passed, failed = (passed + ok, failed + (not ok))

    y_heel = cop_frame(loaded(s0=1000.0, s1=1000.0))[1]
    y_toe = cop_frame(loaded(**{TOE: 1000.0}))[1]
    ok = check(f"y(heel pair) < y({TOE})  [heel is y = 0]",
               y_heel < y_toe,
               f"{y_heel:.6f} vs {y_toe:.6f}  (delta {y_toe - y_heel:+.6f})")
    passed, failed = (passed + ok, failed + (not ok))

    # The heel pair must both sit behind every forefoot sensor, and the toe
    # ahead of every other sensor. Catches a single transposed row, which the
    # two comparisons above would miss.
    forefoot = ("s2", "s3", "s4", "s5")
    ok = check("both heel sensors sit behind every forefoot sensor",
               all(EXPECTED_COORDS[h][1] < EXPECTED_COORDS[f][1]
                   for h in HEEL_PAIR for f in forefoot))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check(f"{TOE} is the most forward sensor",
               all(EXPECTED_COORDS[TOE][1] > EXPECTED_COORDS[n][1]
                   for n in D.SENSOR_COLS if n != TOE))
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_all_zero_is_documented():
    """All six at zero. Records CURRENT behaviour; adds no handling.

    cop_frame short-circuits on total_weight == 0 and returns (nan, nan)
    rather than raising or returning a midpoint. This matters downstream:
    cop_features drops NaN points before measuring path length, so an
    all-zero frame contributes nothing to a CoP path instead of yanking it to
    a spurious centre. Documented, not asserted as desirable.
    """
    passed = failed = 0

    row = loaded()
    try:
        got = cop_frame(row)
        raised = None
    except Exception as exc:                      # noqa: BLE001 - recording it
        got, raised = None, exc

    if raised is not None:
        ok = check("all six zero -> raises (documented behaviour)", True,
                   f"{type(raised).__name__}: {raised}")
        passed += ok
        return passed, failed

    is_nan_pair = (isinstance(got, tuple) and len(got) == 2
                   and all(isinstance(v, float) and np.isnan(v) for v in got))
    ok = check("all six zero -> returns (nan, nan), does not raise",
               is_nan_pair, f"got={got}")
    passed, failed = (passed + ok, failed + (not ok))

    # And the documented consequence: NaN frames are dropped, not counted.
    traj = np.array([cop_frame(loaded(s0=1000.0)),
                     cop_frame(row),
                     cop_frame(loaded(s5=1000.0))])
    feats = cop_features(traj)
    straight = abs(EXPECTED_COORDS[TOE][1] - EXPECTED_COORDS["s0"][1])
    expected_len = np.hypot(EXPECTED_COORDS[TOE][0] - EXPECTED_COORDS["s0"][0],
                            straight)
    ok = check("an all-zero frame is dropped from the CoP path, not counted",
               abs(feats["cop_path_len"] - expected_len) <= 1e-12,
               f"path_len={feats['cop_path_len']:.6f} want={expected_len:.6f}")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_documented_values():
    """CANARY. Pins today's numbers as literals so a re-measurement is loud.

    Everything else in this file is derived and survives a re-measurement.
    These do not, on purpose: they are the exact figures quoted in README.md,
    docs/sim_vs_real.md and the Prompt 13 writeup. If this suite fails, the
    geometry moved and those documents are stale -- rewrite them, then update
    the literals here. Do NOT relax these to match new output while leaving
    the prose alone.
    """
    passed = failed = 0
    for name, got, want in [
        ("insole length is 274.0 mm", D.INSOLE_LEN_MM, 274.0),
        ("insole width is 91.0 mm", D.INSOLE_WIDTH_MM, 91.0),
    ]:
        ok = check(name, got == want, f"got={got} want={want}")
        passed, failed = (passed + ok, failed + (not ok))

    for name, row, want in [
        ("all six equal -> (0.177676, 0.545377)",
         loaded(**{n: 1000.0 for n in D.SENSOR_COLS}), (0.177676, 0.545377)),
        ("s0+s1 equal -> (0.152920, 0.185401)",
         loaded(s0=1000.0, s1=1000.0), (0.152920, 0.185401)),
    ]:
        got = cop_frame(row)
        ok = check(name, close(got, want, tol=5e-7),
                   f"got=({got[0]:.6f}, {got[1]:.6f})")
        passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in (test_coords_derived, test_single_sensor, test_equal_loading,
                  test_orientation, test_all_zero_is_documented,
                  test_documented_values):
        print(f"--- {suite.__name__} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()

    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
