"""Standalone 30 s noise-baseline capture.

Reads the ESP32-S3 INS frame stream from the serial port and writes valid
frames to an output CSV. Does NOT touch read_serial.py or readings.csv.

Frame format (matches gait_gen.parse_frame):
    INS,seq,ts_us,s0,s1,s2,s3,s4,s5,checksum
    checksum = (seq + ts_us + s0..s5) % 256
"""
import sys
import time

PORT = "COM12"     # CH343 USB-UART bridge
BAUD = 115200      # must match firmware Serial.begin()
DURATION_S = 30    # capture window
OUT_PATH = "data/noise_floor_30s.csv"
HEADER = "seq,ts_us,s0,s1,s2,s3,s4,s5"


def parse_frame(line):
    parts = line.strip().split(",")
    if parts[0] != "INS":
        return ("malformed", None)
    if len(parts) != 10:
        return ("malformed", None)
    field = parts[1:9]
    try:
        field = [int(f) for f in field]
    except ValueError:
        return ("malformed", None)
    if sum(field) % 256 == int(parts[-1]):
        return ("ok", field)
    return ("bad_checksum", None)


def main():
    import serial  # pyserial

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as exc:
        print(f"ERROR: could not open {PORT}: {exc}", file=sys.stderr)
        return 2

    valid = malformed = empty = bad_checksum = 0
    deadline = time.monotonic() + DURATION_S
    with ser, open(OUT_PATH, "w") as f:
        f.write(HEADER + "\n")
        ser.reset_input_buffer()
        for _ in range(10):  # warmup: discard partial/stale lines
            ser.readline()
        while time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                empty += 1
                continue
            status, payload = parse_frame(line)
            if status == "ok":
                valid += 1
                f.write(",".join(str(v) for v in payload) + "\n")
            elif status == "bad_checksum":
                bad_checksum += 1
            else:
                malformed += 1

    print(f"wrote {OUT_PATH}")
    print(f"valid={valid} malformed={malformed} empty={empty} "
          f"bad_checksum={bad_checksum}")
    if valid == 0:
        print("ERROR: no valid frames captured", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
