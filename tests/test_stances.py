"""Regression tests for stance detection. Run from the repo root:

    python tests/test_stances.py

One case per simulated stream, each guarding a named failure bucket.
Ground truth always comes from gait_gen.true_stances, never from the
detector -- seeding a test from the detector's current output locks in
today's bugs permanently.

Counts catch fragmentation, annihilation and merging. Nothing catches a
boundary error automatically, so stance_report is printed on every run to
keep it at least visible.

Each check_* function prints PASS/FAIL lines and returns (passed, failed);
the test_* wrapper of the same name asserts nothing failed, so pytest sees a
failure and the direct run keeps its counts.
"""

import os
import subprocess
import sys

import pandas as pd

from insole import detector as D
from insole.gait_gen import SHUFFLE_CYCLE_S, true_stances

from insole.paths import DATA_REAL, DATA_SIM, REPO as _REPO

REPO = str(_REPO)

# stream, gait_gen mode for truth, cycle_s for truth, the failure it guards
CASES = [
    ("sim_walk",    "walk",     1.0,             "baseline regression"),
    ("sim_fast",    "walk",     0.6,             "a too-long GAP_MERGE eating alternate steps"),
    ("sim_shuffle", "shuffle",  SHUFFLE_CYCLE_S, "a too-high MIN_DURATION annihilating short stances"),
    ("sim_dropout", "walk",     1.0,             "a dead heel channel delaying or killing entry"),
    ("sim_stand",   "standing", 1.0,             "missing MAX_DURATION, or MAX_DURATION fragmenting"),
]


def ensure_csv(stem):
    """Sim CSVs are gitignored, so rebuild them from the committed .txt streams."""
    csv_path = os.path.join(DATA_SIM, stem + ".csv")
    txt_path = os.path.join(DATA_SIM, stem + ".txt")
    if os.path.exists(csv_path):
        return csv_path
    if not os.path.exists(txt_path):
        raise SystemExit(f"missing {txt_path} -- the fixture is committed under data/sim/")
    subprocess.run([sys.executable, "-m", "insole.read_serial",
                    txt_path, csv_path], check=True, cwd=REPO)
    return csv_path


def total_force(csv_path):
    df = pd.read_csv(csv_path)
    return df[D.SENSOR_COLS].sum(axis=1).to_numpy()


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    return bool(condition)


def check_streams():
    passed = failed = 0
    for stem, mode, cycle_s, guards in CASES:
        total = total_force(ensure_csv(stem))
        truth = true_stances(60, mode=mode, cycle_s=cycle_s)
        detected = D.merge_close(D.find_stances(total))
        report = D.stance_report(detected, truth)

        ok = check(f"{stem:12s} detected={len(detected):4d} truth={len(truth):4d}",
                   len(detected) == len(truth), f"guards: {guards}")
        passed, failed = (passed + ok, failed + (not ok))

        # Reported, never asserted: counts are structurally blind to this.
        if report["n_matched"]:
            print(f"        boundary: start {report['mean_start_offset']:+.1f} fr, "
                  f"end {report['mean_end_offset']:+.1f} fr, "
                  f"duration {report['mean_duration_error']:+.1f} fr "
                  f"({report['n_matched']} matched)")
    return passed, failed


def test_streams():
    p, f = check_streams()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def check_merge_close():
    """merge_close as a unit, plus the fragmentation case it exists to fix."""
    passed = failed = 0

    for name, args, want in [
        ("merge_close empty",        ([],), []),
        ("merge_close no-op",        ([(0, 10), (40, 50)], 12), [(0, 10), (40, 50)]),
        ("merge_close joins pair",   ([(0, 10), (15, 25)], 12), [(0, 25)]),
        ("merge_close joins chain",  ([(0, 10), (15, 25), (30, 40)], 12), [(0, 40)]),
        ("merge_close respects gap", ([(0, 10), (23, 33)], 12), [(0, 10), (23, 33)]),
    ]:
        got = D.merge_close(*args)
        ok = check(name, got == want, f"got={got}")
        passed, failed = (passed + ok, failed + (not ok))

    # Raise T_OFF above the shuffle midstance trough (596) and drop MIN_DURATION
    # so the fragments survive: the stream shatters and merge_close must put it
    # back. This proves merge_close is load-bearing, not decoration.
    total = total_force(ensure_csv("sim_shuffle"))
    truth = true_stances(60, mode="shuffle", cycle_s=SHUFFLE_CYCLE_S)
    raw = D.find_stances(total, D.T_ON, 750, 10, D.MAX_DURATION)
    merged = D.merge_close(raw, D.GAP_MERGE)
    ok = check("merge_close repairs fragmented shuffle",
               len(raw) > len(truth) and len(merged) == len(truth),
               f"raw={len(raw)} merged={len(merged)} truth={len(truth)}")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_merge_close():
    p, f = check_merge_close()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


REAL = str(DATA_REAL)
# (as-captured filename, stances after find_stances + merge_close). One row per
# training-grade capture; scripts/train_real.py pins the same counts in
# SESSIONS and refuses a capture that is not pinned.
#
# Every count here was READ OFF THE DETECTOR, not counted independently. They
# are regression pins, and green means unchanged, not right. See
# check_real_stance_counts' docstring for what that does and does not license.
REAL_COUNTS = [("stand_02.csv", 0), ("walk02.csv", 35),
               ("fast02.csv", 48), ("shuffle02.csv", 30),
               ("stand_03.csv", 0), ("walk_03.csv", 32),
               ("fast_03.csv", 45), ("shuffle_03.csv", 34)]


def check_real_stance_counts():
    """Stance counts on the real captures at the committed thresholds.

    THESE ARE PINS MEASURED FROM THE DETECTOR, NOT INDEPENDENT GROUND TRUTH.
    Read what this test means before trusting it:

        The `_03` numbers -- stand 0, walk 32, fast 45, shuffle 34 -- were
        produced by running detector.find_stances + merge_close on those four
        captures and writing down what came out. Nobody counted 32 walk
        stances by any means independent of the detector: there is no
        hand-labelled truth for the real captures, no video, no force plate.
        So this test cannot tell you the detector is CORRECT on real gait. It
        can only tell you the detector still does what it did the day the pins
        were taken.

        GREEN HERE MEANS NO-CHANGE, NOT CORRECT. A detector that
        systematically merges two real steps into one stance would have been
        pinned merging them, and would pass this test forever.

        This is the same circularity the project already flagged for the
        Prompt 9 detector thresholds (T_ON, T_OFF, MIN_DURATION, GAP_MERGE),
        which were swept on the simulator and then used to segment the real
        captures the pins were taken from. It is recorded rather than fixed
        because fixing it needs an independent measurement the six-sensor
        hardware cannot make. The pins are still worth keeping: a regression
        pin whose limits are stated is useful, and an unpinned capture is
        worse -- scripts/train_real.py refuses to train on one.

        The sim fixtures above are the opposite case and are the reason this
        file's other tests mean something: their truth comes from
        gait_gen.true_stances, which knows where each stance was generated,
        independent of what the detector then found.

    Prediction for the `_02` set, written before the first run: stand 0, walk
    35, fast 48, shuffle 30 (analyze_real.py C5 and sweep_max_duration.py at
    MAX_DURATION = 200) -- that one was a prediction the detector could have
    failed, made before the run. The `_03` set was pinned when it landed and
    was not predicted in advance.

    This is the only test that can see MAX_DURATION: no simulated stance
    exceeds 60 frames, so every sim fixture above passes at any ceiling from
    120 to 1000, while at 120 the `_02` four read 0 / 18 / 48 / 2
    (over-ceiling runs are discarded, not clipped). Eight minutes of real data
    from one subject; a regression fixture, not evidence about generalisation.
    """
    passed = failed = 0
    for fname, want in REAL_COUNTS:
        total = total_force(os.path.join(REAL, fname))
        got = len(D.merge_close(D.find_stances(total)))
        ok = check(f"data/real/{fname:14s} stances={got:3d} want={want:3d}", got == want,
                   f"MAX_DURATION={D.MAX_DURATION}")
        passed, failed = (passed + ok, failed + (not ok))
    return passed, failed


def test_real_stance_counts():
    p, f = check_real_stance_counts()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def check_true_stances():
    """Guard the cadence bug: truth must scale with cycle_s, in count and width."""
    passed = failed = 0
    for name, kwargs, n, dur in [
        ("true_stances walk",     {"cycle_s": 1.0},                          60, 62),
        ("true_stances fast",     {"cycle_s": 0.6},                         100, 37),
        ("true_stances shuffle",  {"mode": "shuffle", "cycle_s": 0.5},      120, 31),
        ("true_stances standing", {"mode": "standing"},                       0, None),
    ]:
        got = true_stances(60, **kwargs)
        ok = len(got) == n and (dur is None or all(b - a + 1 == dur for a, b in got))
        ok = check(name, ok, f"n={len(got)} want={n}")
        passed, failed = (passed + ok, failed + (not ok))
    return passed, failed


def test_true_stances():
    p, f = check_true_stances()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in (check_true_stances, check_merge_close, check_streams,
                  check_real_stance_counts):
        print(f"--- {suite.__name__.replace('check_', 'test_', 1)} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()

    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
