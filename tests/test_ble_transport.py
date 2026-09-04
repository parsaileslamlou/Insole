"""Transport tests for the BLE line source. Run from the repo root:

    python tests/test_ble_transport.py

Two failure modes, both invisible to every other test in the suite because
both live in read_serial.py's threading and timing code rather than in the
frame codec:

  * the reader thread dies and the consumer waits forever
  * batched delivery is mistaken for a broken sample clock

Neither needs a radio. bleak is not imported, and no BLE-specific behaviour is
reimplemented here: the tests drive read_serial's real capture loop and read
its real summary line.
"""

import io
import os
import sys
import time
import shutil
import tempfile
import threading
import contextlib

from insole import read_serial
from insole.gait_gen import make_frame

# Matches FRAMES_PER_NOTIFY in firmware/insole/insole.ino. Three samples per
# notification is the whole reason arrival time is not a sample clock.
FRAMES_PER_NOTIFY = 3

N_FRAMES    = 60      # 20 notifications; keeps the test under a second
BATCH_GAP_S = 0.03    # 3 frames at 100 Hz, i.e. the real inter-notify gap
HANG_TIMEOUT_S = 10.0 # a hang must fail the suite, not stall CI forever

STATIC_VALS = [1800, 900, 1200, 1300, 1000, 400]


def run_with_timeout(fn, timeout_s):
    """Run fn() in a helper thread. Returns (finished, result, error).

    A hang is the exact failure this file exists to catch, so nothing here may
    wait on one indefinitely. The helper is a daemon: if it really is wedged on
    out.get(), the interpreter still exits and the suite still reports FAIL.
    """
    box = {}

    def target():
        try:
            box["result"] = fn()
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    return (not t.is_alive()), box.get("result"), box.get("error")


def counter_from_summary(text, name):
    """Pull one `name=value` counter out of main()'s printed summary line."""
    for token in text.split():
        if token.startswith(name + "="):
            return int(token.split("=", 1)[1])
    raise AssertionError(f"{name} missing from summary: {text!r}")


# ---------------------------------------------------------------------------
# An exception in the reader thread must not strand the consumer
# ---------------------------------------------------------------------------
def check_reader_thread_exception_reaches_the_consumer():
    boom = RuntimeError("simulated bleak disconnect")
    delivered = make_frame(0, 0, [1, 1, 1, 1, 1, 1])

    async def exploding_runner(out):
        out.put(delivered)      # one line landed before the radio died
        raise boom

    def drain():
        got, err = [], None
        try:
            for line in read_serial.ble_lines(duration_s=1,
                                              _runner=exploding_runner):
                got.append(line)
        except BaseException as exc:
            err = exc
        return got, err

    finished, result, harness_err = run_with_timeout(drain, HANG_TIMEOUT_S)

    assert finished, (
        f"consumer still blocked after {HANG_TIMEOUT_S}s: SENTINEL was not "
        "queued on the failing path. This is the reader-thread hang.")
    assert harness_err is None, f"test harness itself failed: {harness_err!r}"

    got, err = result
    assert got == [delivered], (
        f"lines queued before the failure were lost: {got!r}")
    assert err is boom, (
        f"reader exception was swallowed rather than surfaced, got {err!r}")


# ---------------------------------------------------------------------------
# Batched arrival is not a timing fault
# ---------------------------------------------------------------------------
def batched_ble_source(n_frames, arrivals):
    """Frames delivered the way the BLE path really delivers them.

    ts_us advances by exactly one period per frame: a perfectly healthy 100 Hz
    board. Arrival is clustered -- FRAMES_PER_NOTIFY frames land back to back,
    then the link is quiet for the length of the batch. That clustering is the
    thing an arrival-clock validator misreads as a broken sample clock.
    """
    for i in range(n_frames):
        arrivals.append(time.perf_counter())
        yield make_frame(i, i * read_serial.PERIOD_US, STATIC_VALS)
        if (i + 1) % FRAMES_PER_NOTIFY == 0:
            time.sleep(BATCH_GAP_S)


def arrival_clock_breaks(arrivals):
    """timing_breaks the OLD arrival-clock validator would have reported.

    The same test main() applies, fed host arrival times instead of ts_us. Its
    only job is to prove the fixture is genuinely hostile: if this returns 0
    the test above is vacuous and guards nothing.
    """
    breaks = 0
    for a, b in zip(arrivals, arrivals[1:]):
        gap_us = (b - a) * 1e6
        if abs(gap_us - read_serial.PERIOD_US) > read_serial.TIMING_TOL_US:
            breaks += 1
    return breaks


def check_batched_arrival_is_not_a_timing_break():
    arrivals = []
    tmpdir = tempfile.mkdtemp(prefix="insole_ble_")
    out_csv = os.path.join(tmpdir, "capture.csv")

    real_make_source = read_serial.make_source
    real_out_csv = read_serial.OUT_CSV
    # --source ble takes no out_path argument, and resolve_args() reads this
    # global each time it is called, so this is how the output is redirected.
    read_serial.OUT_CSV = out_csv
    read_serial.make_source = lambda *a, **k: batched_ble_source(N_FRAMES, arrivals)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = read_serial.main(["--source", "ble"])
    finally:
        read_serial.make_source = real_make_source
        read_serial.OUT_CSV = real_out_csv
        shutil.rmtree(tmpdir, ignore_errors=True)

    summary = buf.getvalue().strip()
    print("   ", summary)

    would_have = arrival_clock_breaks(arrivals)
    assert would_have >= (N_FRAMES * 2) // 3, (
        f"fixture no longer clusters arrivals (arrival-clock breaks="
        f"{would_have}); it would pass against the old validator too and so "
        "guards nothing")
    print(f"    arrival-clock validator would have reported "
          f"timing_breaks={would_have}")

    assert counter_from_summary(summary, "timing_breaks") == 0, (
        "batched delivery counted as a timing fault: the validator is reading "
        "host arrival time, not ts_us")
    assert counter_from_summary(summary, "valid") == N_FRAMES
    assert counter_from_summary(summary, "seq_breaks") == 0
    assert code == 0, f"healthy batched capture exited {code}"


def test_reader_thread_exception_reaches_the_consumer():
    check_reader_thread_exception_reaches_the_consumer()


def test_batched_arrival_is_not_a_timing_break():
    check_batched_arrival_is_not_a_timing_break()


if __name__ == "__main__":
    checks = [check_reader_thread_exception_reaches_the_consumer,
              check_batched_arrival_is_not_a_timing_break]
    failed = 0
    for fn in checks:
        print(f"--- {fn.__name__.replace('check_', 'test_', 1)} ---")
        try:
            fn()
            print("PASS", fn.__name__.replace("check_", "test_", 1))
        except AssertionError as exc:
            failed += 1
            print("FAIL", fn.__name__.replace("check_", "test_", 1), "--", exc)
        print()
    print(f"{len(checks) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
