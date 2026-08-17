import sys
from gait_gen import parse_frame     

PORT = "COM12"     # CH343 USB-UART bridge (from list_ports)
BAUD = 115200      # must match firmware Serial.begin()
DURATION_S = 60    # capture window before clean stop (None = run forever)

def file_lines(path):
    with open(path, "r") as f:
        for line in f:
            yield line

def serial_lines(PORT, BAUD, duration_s=None):
    import serial, time
    deadline = None if duration_s is None else time.monotonic() + duration_s
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        ser.reset_input_buffer()
        for _ in range(10):  # RETUNE
            ser.readline()
        while deadline is None or time.monotonic() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            yield raw.decode("utf-8", errors="ignore")

def log_frames(source, out_path):
    """Parse `source` line-by-line into out_path; return the tally dict."""
    malformed = empty = valid = bad_checksum = seq_breaks = timing_breaks = 0
    prev_seq = prev_ts = None

    with open(out_path, "w") as f:
        f.write("seq,ts_us,s0,s1,s2,s3,s4,s5\n")
        for line in source:
            line = line.strip()
            if not line:
                empty += 1
                continue
            status, payload = parse_frame(line)
            if status == "ok":
                valid += 1
                f.write(",".join(str(v) for v in payload) + "\n")
                seq = payload[0]
                ts = payload[1]
                if prev_seq is not None:
                    if (seq - prev_seq) % 65536 != 1:
                        seq_breaks += 1
                    if abs((ts - prev_ts) - 10000) > 500:
                        timing_breaks += 1
                prev_seq = seq
                prev_ts = ts
            elif status == "bad_checksum":
                bad_checksum += 1
            else:
                malformed += 1

    return {"valid": valid, "malformed": malformed, "empty": empty,
            "bad_checksum": bad_checksum, "seq_breaks": seq_breaks,
            "timing_breaks": timing_breaks}

def main(argv):
    # No args: capture from the serial port, as before.
    # Two args: convert an already-captured .txt stream into a named .csv,
    #           which is how the sim_*.txt streams become sim_*.csv.
    if len(argv) == 3:
        in_path, out_path = argv[1], argv[2]
        source = file_lines(in_path)
    elif len(argv) == 1:
        in_path, out_path = f"{PORT}", "readings.csv"
        source = serial_lines(PORT, BAUD, DURATION_S)
    else:
        print("usage: read_serial.py [INPUT.txt OUTPUT.csv]")
        return 2

    tally = log_frames(source, out_path)
    print(f"{in_path} -> {out_path}  " +
          " ".join(f"{k}={v}" for k, v in tally.items()))

    if tally["valid"] == 0 or any(tally[k] for k in
            ("malformed", "bad_checksum", "seq_breaks", "timing_breaks")):
        print("FAIL: acceptance criteria not met")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))