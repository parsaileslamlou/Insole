"""
read_serial.py — host logger for the insole.

Three line sources behind one seam:
    "serial"  live USB CDC          (pyserial)
    "file"    replay a saved capture
    "ble"     live Nordic UART      (bleak)

Everything below make_source() is transport-agnostic: it consumes an iterator
of strings and does not know or care where they came from.

    pip install pyserial bleak
"""

import sys
import time
import csv

# ---------------------------------------------------------------------------
# Configuration (constants, not CLI args — matches the existing style)
# ---------------------------------------------------------------------------
SOURCE     = "ble"          # "serial" | "file" | "ble"

PORT       = "COM7"         # SOURCE == "serial"
BAUD       = 921600
IN_FILE    = "capture.txt"  # SOURCE == "file"
BLE_NAME   = "INSOLE"       # SOURCE == "ble"

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


def file_lines(path=IN_FILE):
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

        dev = await BleakScanner.find_device_by_name(device_name, timeout=15.0)
        if dev is None:
            print(f"BLE: no device named {device_name!r} found")
            out.put(SENTINEL)
            return

        disconnected = asyncio.Event()

        def on_disconnect(_client):
            disconnected.set()

        async with BleakClient(dev, disconnected_callback=on_disconnect) as client:
            try:
                mtu = client.mtu_size
            except Exception:
                mtu = "unavailable"
            print(f"BLE: connected to {device_name}, negotiated MTU = {mtu}")

            await client.start_notify(NUS_TX_UUID, on_notify)
            try:
                await asyncio.wait_for(disconnected.wait(), timeout=duration_s)
                print("BLE: peripheral disconnected before the duration elapsed")
            except asyncio.TimeoutError:
                pass                          # normal end of capture
            try:
                await client.stop_notify(NUS_TX_UUID)
            except Exception:
                pass

        # buf may still hold a partial line here. It is dropped on purpose.
        out.put(SENTINEL)

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()

    while True:
        item = out.get()
        if item is SENTINEL:
            return
        yield item


def make_source():
    if SOURCE == "serial":
        return serial_lines()
    if SOURCE == "file":
        return file_lines()
    if SOURCE == "ble":
        return ble_lines()
    raise ValueError(f"unknown SOURCE {SOURCE!r}")


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------
def parse_frame(line):
    """Return (row, reason). row is None when the line is not a valid frame."""
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
def main():
    c = dict(valid=0, malformed=0, empty=0, bad_checksum=0,
             seq_breaks=0, timing_breaks=0, status=0, lost=0)

    prev_seq = None
    prev_ts = None

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

        t0 = time.time()
        for line in make_source():
            if time.time() - t0 > DURATION_S + 5:
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
            c["valid"] += 1
            w.writerow([seq, ts_us] + vals)

    expected_total = c["valid"] + c["lost"]
    loss_pct = (100.0 * c["lost"] / expected_total) if expected_total else 0.0

    print(f"source={SOURCE} valid={c['valid']} malformed={c['malformed']} "
          f"empty={c['empty']} bad_checksum={c['bad_checksum']} "
          f"seq_breaks={c['seq_breaks']} lost={c['lost']} "
          f"loss={loss_pct:.2f}% timing_breaks={c['timing_breaks']} "
          f"status={c['status']}")

    sys.exit(exit_code(c, loss_pct))


def exit_code(c, loss_pct):
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

    if SOURCE == "ble":
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
    main()