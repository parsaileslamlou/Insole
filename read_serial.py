import sys
from gait_gen import parse_frame     

PORT = "COM3"      # replace with your real port from list_ports
BAUD = 115200      # must match firmware Serial.begin()

def file_lines(path):
    with open(path, "r") as f:
        for line in f:
            yield line

def serial_lines(PORT, BAUD):
    import serial
    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        ser.reset_input_buffer()
        for _ in range(10):  # RETUNE
            ser.readline()
        while True:
            raw = ser.readline()
            if not raw:
                continue
            yield raw.decode("utf-8", errors="ignore")

malformed = empty = valid = bad_checksum = seq_breaks = timing_breaks = 0
prev_seq = prev_ts = None

source = serial_lines(PORT, BAUD)
with open("readings.csv", "w") as f:
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

print(f"valid={valid} malformed={malformed} empty={empty} "
      f"bad_checksum={bad_checksum} seq_breaks={seq_breaks} "
      f"timing_breaks={timing_breaks}")

if valid == 0 or malformed or bad_checksum or seq_breaks or timing_breaks:
    print("FAIL: acceptance criteria not met")
    sys.exit(1)