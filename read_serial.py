"""
read_serial.py — host logger for the insole.

Three line sources behind one seam:
    "serial"  live USB CDC          (pyserial)
    "file"    replay a saved capture
    "ble"     live Nordic UART      (bleak)

Everything below make_source() is transport-agnostic: it consumes an iterator
of strings and does not know or care where they came from.

    pip install pyserial bleak

Usage:
    read_serial.py                          live BLE  -> readings.csv
    read_serial.py --source ble    out.csv  live BLE  -> out.csv
    read_serial.py --source serial out.csv  live USB  -> out.csv
    read_serial.py in.txt out.csv           replay a saved capture
    read_serial.py --source file in.txt out.csv

Both paths are optional. --source defaults to "file" when an input path is
given and "ble" otherwise, so the notebook's
`read_serial.py sim_walk.txt sim_walk.csv` works with no flags.

A live source has no input path, so its single positional is the OUTPUT --
that is what lets compare_captures.py run two captures side by side. --source
has to be given explicitly in that case, because a bare positional still means
"replay this file" and always did.
"""

import sys
import os
import time
import csv
import argparse

# ---------------------------------------------------------------------------
# Configuration. These are DEFAULTS for the CLI below, not the live values —
# nothing downstream reads them at runtime. In particular no function takes one
# as a default argument value: that binds at import time and cannot be
# overridden by assigning to the module attribute afterwards.
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = "ble"      # "serial" | "file" | "ble"

PORT       = "COM7"         # --source serial
BAUD       = 921600
IN_FILE    = "capture.txt"  # --source file, when no in_path is given
BLE_NAME   = "INSOLE"       # --source ble

DURATION_S = 60
OUT_CSV    = "readings.csv"

PERIOD_US  = 10000          # 100 Hz, must match firmware
TIMING_TOL_US = 3000        # how far a gap may stray from its expected value

# Occasional radio drops are normal over BLE and are NOT a defect. Corrupted
# or malformed frames are, on every transport. See exit_code() below.
BLE_LOSS_TOLERANCE_PCT = 2.0

NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

SENSOR_COLS = ["s0", "s1", "s2", "s3", "s4", "s5"]
CSV_HEADER  = ["seq", "ts_us"] + SENSOR_COLS


# ---------------------------------------------------------------------------
# Line sources
# ---------------------------------------------------------------------------
def serial_lines(port=PORT, baud=BAUD, duration_s=DURATION_S):
    import serial
    ser = serial.Serial(port, baud, timeout=1)
    ser.reset_input_buffer()
    t_warm = time.time()
    while time.time() - t_warm < 1.0:      # discard boot chatter
        ser.readline()
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            raw = ser.readline()
            if not raw:
                continue
            yield raw.decode("ascii", errors="ignore").strip()
    finally:
        ser.close()


def file_lines(path):
    with open(path, "r") as f:
        for raw in f:
            yield raw.strip()


def ble_lines(device_name=BLE_NAME, duration_s=DURATION_S):
    """Live BLE source.

    All asyncio and threading is confined to this function. It returns a plain
    synchronous generator of strings, exactly like the other two sources.

    Reassembly contract:
      * A notification may carry several whole lines, one line, or a fragment
        of a line. Nothing may be assumed about alignment.
      * Bytes accumulate in `buf`. We emit only up to the LAST newline in the
        buffer; whatever follows stays buffered until more bytes arrive.
      * At disconnect the trailing partial line is DISCARDED, never emitted.
        Emitting it would produce a short frame that fails the checksum and
        looks like corruption.
    """
    import asyncio
    import threading
    import queue as _queue
    from bleak import BleakScanner, BleakClient

    out = _queue.Queue(maxsize=20000)
    SENTINEL = object()

    async def _run():
        buf = bytearray()

        def on_notify(_handle, data):
            buf.extend(data)
            nl = buf.rfind(b"\n")
            if nl < 0:
                return                       # no complete line yet — wait
            chunk = bytes(buf[:nl + 1])
            del buf[:nl + 1]                 # keep the partial tail
            for raw in chunk.split(b"\n"):
                if raw:
                    out.put(raw.decode("ascii", errors="ignore").strip())

        # Discovery is timed and reported separately from the capture. It can
        # burn the full 15 s, and charging that against the capture window is
        # what used to make a requested 60 s run quietly yield far less.
        t_scan0 = time.monotonic()
        dev = await BleakScanner.find_device_by_name(device_name, timeout=15.0)
        scan_s = time.monotonic() - t_scan0
        if dev is None:
            print(f"BLE: no device named {device_name!r} found "
                  f"(scanned {scan_s:.1f}s)")
            return

        disconnected = asyncio.Event()

        def on_disconnect(_client):
            disconnected.set()

        async with BleakClient(dev, disconnected_callback=on_disconnect) as client:
            try:
                mtu = client.mtu_size
            except Exception:
                mtu = "unavailable"
            print(f"BLE: connected to {device_name} after {scan_s:.1f}s "
                  f"discovery, negotiated MTU = {mtu}")

            # The notify window opens here, AFTER discovery and connect, so it
            # gets the full duration_s it was asked for.
            t_notify0 = time.monotonic()
            await client.start_notify(NUS_TX_UUID, on_notify)
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=duration_s)
                print("BLE: peripheral disconnected before the duration elapsed")
            except asyncio.TimeoutError:
                pass                          # normal end of capture
            print(f"BLE: notify window {time.monotonic() - t_notify0:.1f}s "
                  f"(requested {duration_s}s)")
            try:
                await client.stop_notify(NUS_TX_UUID)
            except Exception:
                pass

        # buf may still hold a partial line here. It is dropped on purpose.

    def _pump():
        """Run _run() and guarantee the consumer is released, however it ends.

        _run() used to queue SENTINEL itself on each of its exit paths. Any
        path it did not cover -- a bleak connection error, a raise inside the
        notify callback -- left the consumer blocked on out.get() forever:
        a hang with no exit code at all, which is strictly worse than a FAIL.
        The sentinel is now queued in exactly one place, a finally, so no
        future exit path can miss it.

        The reassembly contract in on_notify() is untouched.
        """
        try:
            asyncio.run(_run())
        except Exception as exc:                 # noqa: BLE001 - report, never hang
            print(f"BLE: capture thread failed: {exc!r}")
        finally:
            out.put(SENTINEL)

    threading.Thread(target=_pump, daemon=True).start()

    while True:
        item = out.get()
        if item is SENTINEL:
            return
        yield item


def make_source(source, in_path=None, duration_s=DURATION_S):
    if source == "serial":
        return serial_lines(duration_s=duration_s)
    if source == "file":
        if in_path is None:
            raise ValueError("source 'file' requires an input path")
        return file_lines(in_path)
    if source == "ble":
        return ble_lines(duration_s=duration_s)
    raise ValueError(f"unknown source {source!r}")


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------
def parse_frame(line):
    """Return (row, reason). row is None when the line is not a valid frame.

    A leading '#' is a firmware status line (`# ble conn=... mtu=...`), not
    corruption. Tolerating it here as well as in main() means a direct caller
    of parse_frame() gets the same answer the capture loop does.
    """
    if line.startswith("#"):
        return None, "status"
    parts = line.split(",")
    if len(parts) != 10 or parts[0] != "INS":
        return None, "malformed"
    try:
        nums = [int(p) for p in parts[1:]]
    except ValueError:
        return None, "malformed"

    seq, ts_us = nums[0], nums[1]
    vals, ck = nums[2:8], nums[8]

    if (sum(nums[:8]) % 256) != ck:
        return None, "bad_checksum"
    if any(v < 0 or v > 4095 for v in vals):
        return None, "malformed"

    return (seq, ts_us, vals), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Capture insole frames from USB serial, BLE, or a saved file.")
    p.add_argument("in_path", nargs="?", default=None,
                   help="capture to replay; giving it implies --source file")
    p.add_argument("out_path", nargs="?", default=None,
                   help=f"output CSV (default: {OUT_CSV}). For a live source "
                        f"this is the FIRST positional, not the second.")
    p.add_argument("--source", choices=("serial", "file", "ble"), default=None,
                   help="transport; defaults to 'file' when in_path is given, "
                        "else '%s'" % DEFAULT_SOURCE)
    return p


def _same_file(a, b):
    """True if two path strings name the same file, before either is opened.

    normcase() because on Windows 'Capture.TXT' and 'capture.txt' are one
    file. os.path.samefile() is not usable here: out_path does not exist yet.
    """
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def resolve_args(argv=None):
    """Bind the two optional positionals to (in_path, out_path) per source.

    argparse fills positionals left to right, so `--source ble out.csv` puts
    out.csv in in_path -- the wrong slot, because a live source has no input
    path. The old code spotted the mismatch and REFUSED the argument, which
    left no way at all to name the output file for a live capture. That is
    exactly what compare_captures.py needs two of, so the one script whose
    whole purpose is proving BLE reassembly could not be run. Re-bind rather
    than refuse.

    The restriction that ban was standing in for is real but much narrower:
    under --source file both paths are given and must not be the same file,
    or opening out_path for writing truncates the input before it is read.
    That is now checked as the collision it actually is, on the resolved
    paths, so `capture.txt ./capture.txt` is caught too.
    """
    args = build_parser().parse_args(argv)

    if args.source is None:
        args.source = "file" if args.in_path else DEFAULT_SOURCE

    if args.source == "file":
        if args.in_path is None:
            args.in_path = IN_FILE
    else:
        # No input path exists for a live source, so a lone positional is the
        # output. Two positionals is a real mistake and still an error.
        if args.out_path is not None:
            build_parser().error(
                f"--source {args.source} takes at most one path, the output "
                f"CSV; got two: {args.in_path!r} and {args.out_path!r}")
        args.out_path, args.in_path = args.in_path, None

    if args.out_path is None:
        args.out_path = OUT_CSV

    if args.source == "file" and _same_file(args.in_path, args.out_path):
        build_parser().error(
            f"in_path and out_path are the same file ({args.in_path!r}); "
            f"the replay would truncate its own input")

    return args


def main(argv=None):
    args = resolve_args(argv)

    c = dict(valid=0, malformed=0, empty=0, bad_checksum=0,
             seq_breaks=0, timing_breaks=0, status=0, lost=0)

    prev_seq = None
    prev_ts = None

    t_first = None      # host clock at the first line of any kind
    t_last = None       # host clock at the last valid frame
    first_ts = None     # device clock (ts_us) at the first valid frame
    last_ts = None      # device clock at the last valid frame

    with open(args.out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

        for line in make_source(args.source, args.in_path):
            now = time.time()
            # This guard exists to stop a live source that never ends. It runs
            # from the FIRST LINE, not from process start, so time spent
            # discovering and connecting is not charged against the capture
            # window -- that is what silently shortened BLE runs. Anchoring on
            # the first line is transport-agnostic on purpose: serial_lines()
            # also burns a second on boot chatter before its first line.
            if t_first is None:
                t_first = now
            elif now - t_first > DURATION_S + 5:
                break

            if not line:
                c["empty"] += 1
                continue
            if line.startswith("#"):          # firmware status line, not a frame
                c["status"] += 1
                print(line)
                continue

            row, reason = parse_frame(line)
            if row is None:
                c[reason] += 1
                continue

            seq, ts_us, vals = row

            if prev_seq is not None:
                # uint16 sequence, wraps every 65536 frames (~11 min at 100 Hz)
                delta = (seq - prev_seq) % 65536
                if delta != 1:
                    c["seq_breaks"] += 1
                    c["lost"] += max(delta - 1, 0)
                # Expected time gap scales with the number of frames spanned, so
                # a dropped frame is counted ONCE (as loss) and not a second
                # time as a timing fault.
                if prev_ts is not None and delta > 0:
                    expected = PERIOD_US * delta
                    if abs((ts_us - prev_ts) - expected) > TIMING_TOL_US:
                        c["timing_breaks"] += 1

            prev_seq, prev_ts = seq, ts_us
            if first_ts is None:
                first_ts = ts_us
            last_ts, t_last = ts_us, now
            c["valid"] += 1
            w.writerow([seq, ts_us] + vals)

    expected_total = c["valid"] + c["lost"]
    loss_pct = (100.0 * c["lost"] / expected_total) if expected_total else 0.0

    # Two spans, deliberately both reported, because they are different clocks:
    #   capture_s  host wall clock, first valid frame -> last. What the session
    #              actually cost, and the number to compare against DURATION_S.
    #   device_s   the board's own ts_us span. Immune to transport jitter and
    #              batching, so a gap between the two is a link story, not a
    #              sampling story.
    capture_s = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    device_s = ((last_ts - first_ts) / 1e6) if (first_ts is not None and last_ts is not None) else 0.0

    print(f"source={args.source} valid={c['valid']} malformed={c['malformed']} "
          f"empty={c['empty']} bad_checksum={c['bad_checksum']} "
          f"seq_breaks={c['seq_breaks']} lost={c['lost']} "
          f"loss={loss_pct:.2f}% timing_breaks={c['timing_breaks']} "
          f"status={c['status']} "
          f"capture_s={capture_s:.1f} device_s={device_s:.1f}")

    # A short session should be visible, not inferred from a low frame count.
    # Reporting only -- it does not change the exit code. Live sources only:
    # a file replay finishes in well under DURATION_S by design.
    if args.source != "file" and capture_s < DURATION_S - 1.0:
        print(f"NOTE: capture ran {capture_s:.1f}s, {DURATION_S - capture_s:.1f}s "
              f"short of the requested {DURATION_S}s")

    return exit_code(c, loss_pct, args.source)


def exit_code(c, loss_pct, source):
    """Transport-aware gating.

    Two classes of anomaly, and they are not the same kind of thing:

      CORRUPTION  malformed, bad_checksum, empty
          Never acceptable on any transport. A checksum failure over BLE means
          the reassembly logic is wrong, not that the radio is busy. Zero
          tolerance, always.

      LOSS        whole frames that never arrived (seq gaps)
          On USB this means the firmware or the host stalled — a real defect.
          On BLE this is the radio doing what radios do. Gated on a RATE, not
          a count, so the check still means something: a capture that loses 30%
          of its frames still fails.
    """
    if c["malformed"] or c["bad_checksum"] or c["empty"]:
        print("FAIL: corrupted frames present")
        return 1

    if c["valid"] == 0:
        print("FAIL: no valid frames")
        return 1

    if source == "ble":
        if loss_pct > BLE_LOSS_TOLERANCE_PCT:
            print(f"FAIL: BLE frame loss {loss_pct:.2f}% "
                  f"exceeds tolerance {BLE_LOSS_TOLERANCE_PCT}%")
            return 1
        # timing_breaks over BLE that are NOT explained by loss are still real
        if c["timing_breaks"]:
            print("FAIL: timing breaks not explained by frame loss")
            return 1
        return 0

    # serial / file: unchanged strictness
    if c["lost"] or c["seq_breaks"] or c["timing_breaks"]:
        print("FAIL: sequence or timing anomaly")
        return 1
    return 0


if __name__ == "__main__":
    # main() returns the code rather than calling sys.exit() itself, so every
    # FAIL branch in exit_code() lands here and every failure leaves a nonzero
    # status. It also makes main() callable in-process without SystemExit.
    sys.exit(main())