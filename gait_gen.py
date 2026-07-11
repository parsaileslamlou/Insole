
def checksum(seq, timestamp, readings):
    return (seq + timestamp + sum(readings)) % 256

def make_frame(seq, timestamp, readings):
    ck = checksum(seq, timestamp, readings)
    
    readings_str = ",".join(map(str, readings))
    return f"INS,{seq},{timestamp},{readings_str},{ck}"

def parse_frame(line):
    parts = line.strip().split(",")

    sync = parts[0]
    if sync != "INS":
        return ("malformed", None)
    
    if len(parts) != 10:
        return ("malformed", None)
    
    field = parts[1:9]

    try:
        field = [int(f) for f in field]

        verified_sum = sum(field) % 256
        if verified_sum == int(parts[-1]):
            return ("ok", field)
        else:
            return ("bad_checksum", None)
    except ValueError:
        return ("malformed", None)
    

import math
import random

CYCLE_S = 1.0
SENSOR_WINDOWS = [(0.00, 0.30), (0.15, 0.45), (0.30, 0.55),
                  (0.32, 0.57), (0.35, 0.58), (0.45, 0.62)]
AMPLITUDES = [3200, 1400, 2600, 2800, 2200, 1800]
NOISE_STD = 25

STANDING_LOADS = [1800, 900, 1200, 1300, 1000, 400]

def sensor_value(t, i, mode="walk"):
    if mode == "standing":
        value = STANDING_LOADS[i]               
    else:
        phase = (t % CYCLE_S) / CYCLE_S
        start, end = SENSOR_WINDOWS[i]
        if start <= phase < end:
            progress = (phase - start) / (end - start)
            value = AMPLITUDES[i] * math.sin(math.pi * progress)
        else:
            value = 0
    value += random.gauss(0, NOISE_STD)
    return int(max(0, min(4095, value)))            

SAMPLE_HZ = 100
PERIOD_US = 1_000_000 // SAMPLE_HZ

def gait_lines(duration_s, mode="walk"):
    n = int(duration_s * SAMPLE_HZ)
    for k in range(n):
        t = k / SAMPLE_HZ                                        
        readings = [sensor_value(t, i, mode) for i in range(6)]   
        yield make_frame(k % 65536 , k * PERIOD_US, readings)           
                
with open("sim_walk.txt", "w") as f:
    for line in gait_lines(60):
        f.write(line + "\n")
