"""Regression tests for the capture window anchor. Run from the repo root:

    python3 test_capture_window.py

Stdlib only, and no hardware: no board, no radio, no serial port, and no real
clock. The clock and the line source are both fakes, so the whole 600-frame
capture runs in milliseconds and gives the same answer every time.

What is being pinned
--------------------
read_serial.main() has a runaway guard that stops a live source which never
ends. It used to be anchored at PROCESS START:

    t0 = time.time()
    for line in make_source(...):
        if time.time() - t0 > DURATION_S + 5:
            break

BLE discovery runs inside make_source() and can burn the full 15 s scan before
the first line ever arrives. That time was charged against the capture window,
so a 60 s capture silently returned far less than 60 s of frames -- and exited
0 while doing it, because a short capture with no corruption and no seq gaps is
indistinguishable from a good one by every other check in the file.

The anchor is now the FIRST LINE, not process start. That is transport-
agnostic on purpose: serial_lines() also burns a second on boot chatter before
its first line.

The numbers below are the point of the test
-------------------------------------------
Scaled down 10x so it runs instantly -- DURATION_S 6 instead of 60, which makes
the guard 11 s -- with 8 s of fake discovery in front of 600 frames at 100 Hz:

    old anchor (process start)  301 of 600 frames written, exit 0
    new anchor (first line)     600 of 600 frames written, exit 0

301 is not a rounding artifact. The guard fires on the first line whose
timestamp exceeds t0 + 11 s; at 8 + 0.01*i that is i = 301, so lines 0..300
land. Frames vanish and nothing says so, which is the whole defect.

To confirm this test really does fail against the old logic, revert just the
anchor hunk in main() -- restore `t0 = time.time()` above the loop and put back
`if time.time() - t0 > DURATION_S + 5: break` -- and re-run. It reports 301
where it wants 600.
"""

import contextlib
import csv
import io
import os
import sys
import tempfile

import read_serial


PERIOD_US = read_serial.PERIOD_US       # 10000, i.e. 100 Hz
TEST_DURATION_S = 6                     # 10x scale-down of the real 60
FRAMES = 600                            # exactly TEST_DURATION_S at 100 Hz


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    return bool(condition)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeClock:
    """Stands in for the `time` module inside read_serial.

    Only time.time() and time.monotonic() are reached from the code under
    test, so those are all it provides. Nothing here ever sleeps: the source
    advances the clock explicitly, which is what makes the result identical on
    a fast laptop and on a loaded machine.
    """

    def __init__(self, start=1000.0):
        self.now = start

    def time(self):
        return self.now

    def monotonic(self):
        return self.now

    def advance(self, dt):
        self.now += dt


def frame(seq, ts_us, vals=(100, 200, 300, 400, 500, 600)):
    """One valid INS line, checksum included, matching parse_frame()."""
    nums = [seq, ts_us] + list(vals)
    ck = sum(nums) % 256
    return "INS," + ",".join(str(n) for n in nums + [ck])


def fake_source(clock, discovery_s, n_frames):
    """A line source that burns `discovery_s` before producing anything.

    This is the shape of ble_lines(): the scan and connect happen INSIDE the
    generator, so they cost wall-clock time that arrives before the first line
    rather than before the generator is created. Frames then arrive at exactly
    100 Hz on both clocks -- the host clock this advances, and the device ts_us
    field it writes -- so no seq_breaks and no timing_breaks are generated and
    the capture is clean by every check except length.
    """

    def make(source, in_path=None, **_kw):     # port=, duration_s=, stall_s= ignored
        def gen():
            clock.advance(discovery_s)
            for i in range(n_frames):
                yield frame(i, i * PERIOD_US)
                clock.advance(PERIOD_US / 1e6)
        return gen()

    return make


def run_capture(discovery_s, n_frames, source="serial"):
    """Drive read_serial.main() against the fakes.

    Returns (exit_code, rows_written, stdout_text).
    """
    clock = FakeClock()
    saved = (read_serial.time, read_serial.make_source, read_serial.DURATION_S)
    read_serial.time = clock
    read_serial.make_source = fake_source(clock, discovery_s, n_frames)
    read_serial.DURATION_S = TEST_DURATION_S

    buf = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.csv")

            # --source file takes two positionals; a live source takes one.
            # Getting this wrong does not raise: the lone path lands in in_path,
            # out_path falls back to the OUT_CSV default, and the run quietly
            # writes readings.csv in the repo root instead of the temp dir. So
            # build the argv per source.
            if source == "file":
                argv = ["--source", source, os.path.join(tmp, "in.txt"), out_path]
            else:
                argv = ["--source", source, out_path]

            with contextlib.redirect_stdout(buf):
                code = read_serial.main(argv)
            with open(out_path, newline="") as f:
                rows = list(csv.reader(f))
    finally:
        read_serial.time, read_serial.make_source, read_serial.DURATION_S = saved

    return code, len(rows) - 1, buf.getvalue()      # -1 for the CSV header


def _field(out, key):
    """Pull `key=value` out of the summary line, for failure messages."""
    for tok in out.replace("\n", " ").split():
        if tok.startswith(key + "="):
            return tok
    return key + "=<absent>"


def _note(out):
    for line in out.splitlines():
        if line.startswith("NOTE:"):
            return line
    return ""


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------
def test_slow_discovery_does_not_shorten_the_capture():
    """The regression proper. 8 s of discovery, then a full 6 s of frames."""
    passed = failed = 0

    code, rows, out = run_capture(discovery_s=8.0, n_frames=FRAMES)

    ok = check("all 600 frames survive 8 s of discovery",
               rows == FRAMES,
               f"wrote {rows}, want {FRAMES} (the old anchor wrote 301)")
    passed, failed = (passed + ok, failed + (not ok))

    # The old anchor exited 0 as well. That is exactly why the frame count,
    # and not the exit code, is the assertion that catches this defect.
    ok = check("exit 0, as it also was while losing half the capture",
               code == 0, f"exit={code}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("summary reports valid=600", "valid=600" in out,
               _field(out, "valid"))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("capture_s spans the frames, not the discovery",
               "capture_s=6.0" in out, _field(out, "capture_s"))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("device_s agrees with capture_s",
               "device_s=6.0" in out, _field(out, "device_s"))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("no NOTE: a full-length capture is not flagged short",
               "NOTE:" not in out, _note(out))
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_short_capture_is_still_reported():
    """The other half: the NOTE must fire on a real truncation, and only then.

    Committed alongside the anchor test because the two only mean something
    together. The anchor could have been "fixed" by deleting the guard, which
    would also make the first test pass -- and would leave a genuinely
    truncated capture exactly as invisible as it was before. This is the test
    that says the reporting still works.
    """
    passed = failed = 0

    # 14 s of discovery, full-length capture. Nothing is wrong here.
    code, rows, out = run_capture(discovery_s=14.0, n_frames=FRAMES)

    ok = check("14 s of discovery alone does not trigger the NOTE",
               "NOTE:" not in out and rows == FRAMES,
               f"rows={rows} {_note(out)}")
    passed, failed = (passed + ok, failed + (not ok))

    # Same 14 s discovery, but the peripheral stops after 3 s of frames.
    code, rows, out = run_capture(discovery_s=14.0, n_frames=FRAMES // 2)

    ok = check("a genuinely short capture does trigger the NOTE",
               "NOTE:" in out, f"rows={rows}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("the NOTE names the shortfall, not the discovery time",
               "3.0s" in _note(out) and "14" not in _note(out),
               _note(out))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("a short capture with no corruption still exits 0",
               code == 0, f"exit={code}")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


def test_file_replay_is_unaffected():
    """--source file must not pick up a live-source behaviour. See V5."""
    passed = failed = 0

    code, rows, out = run_capture(discovery_s=0.0, n_frames=FRAMES // 2,
                                  source="file")

    ok = check("a file replay is never flagged short",
               "NOTE:" not in out, _note(out))
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("a file replay still writes every frame",
               rows == FRAMES // 2, f"rows={rows}")
    passed, failed = (passed + ok, failed + (not ok))

    ok = check("a file replay still exits 0", code == 0, f"exit={code}")
    passed, failed = (passed + ok, failed + (not ok))

    return passed, failed


if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in (test_slow_discovery_does_not_shorten_the_capture,
                  test_short_capture_is_still_reported,
                  test_file_replay_is_unaffected):
        print(f"--- {suite.__name__} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()

    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
