
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

# --- nasty stream parameters -------------------------------------
SHUFFLE_CYCLE_S   = 0.50           # RETUNE: short-stride shuffle stride period
SHUFFLE_AMP_SCALE = 0.45           # RETUNE: shuffling loads the foot more lightly
DROPOUT_CH        = 0              # RETUNE: which channel dies (s0 = heel, worst case)
DROPOUT_T         = (20.0, 40.0)   # seconds; dead window inside a 60 s capture

def sensor_value(t, i, mode="walk", cycle_s=CYCLE_S,
                 dropout_ch=None, dropout_t=(None, None)):
    # A disconnected FSR with a pulldown reads flat zero, not noisy zero,
    # so this returns before noise is added.
    if dropout_ch is not None and i == dropout_ch:
        t_start, t_end = dropout_t
        if t_start is not None and t_end is not None and t_start <= t < t_end:
            return 0

    if mode == "standing":
        value = STANDING_LOADS[i]
    else:
        amplitude = AMPLITUDES[i]
        if mode == "shuffle":
            # A shuffle is a normal step done small and fast: same windows,
            # same sine shape, shorter cycle and lower load.
            cycle_s = SHUFFLE_CYCLE_S
            amplitude *= SHUFFLE_AMP_SCALE
        phase = (t % cycle_s) / cycle_s
        start, end = SENSOR_WINDOWS[i]
        if start <= phase < end:
            progress = (phase - start) / (end - start)
            value = amplitude * math.sin(math.pi * progress)
        else:
            value = 0
    value += random.gauss(0, NOISE_STD)
    return int(max(0, min(4095, value)))

SAMPLE_HZ = 100
PERIOD_US = 1_000_000 // SAMPLE_HZ

def gait_lines(duration_s, mode="walk", cycle_s=CYCLE_S,
               dropout_ch=None, dropout_t=(None, None)):
    n = int(duration_s * SAMPLE_HZ)
    for k in range(n):
        t = k / SAMPLE_HZ
        readings = [sensor_value(t, i, mode, cycle_s, dropout_ch, dropout_t)
                    for i in range(6)]
        yield make_frame(k % 65536 , k * PERIOD_US, readings)

STANCE_START = min(w[0] for w in SENSOR_WINDOWS)
STANCE_END   = max(w[1] for w in SENSOR_WINDOWS)

def true_stances(duration_s, mode="walk", cycle_s=CYCLE_S):
    """Ground-truth stance intervals as (start_frame, end_frame), inclusive.

    STANCE_START/STANCE_END are phase fractions, not seconds, so they scale
    with cycle_s. A 0.6 s stride carries a 0.62 * 0.6 = 0.372 s stance.

    mode="standing" has no steps, so the answer is [] and not one long stance.
    mode="dropout" is deliberately identical to walk: a dead sensor does not
    delete a step, and truth must not be bent to match what the detector can
    see or the test becomes circular.
    """
    if mode == "standing":
        return []
    out = []
    for k in range(int(duration_s / cycle_s)):
        t0 = k * cycle_s
        a = int(round((t0 + STANCE_START * cycle_s) * SAMPLE_HZ))
        b = int(round((t0 + STANCE_END * cycle_s) * SAMPLE_HZ)) - 1
        out.append((a, b))
    return out

# (filename, gait_lines kwargs) for the five simulated streams.
STREAMS = [
    ("sim_walk.txt",    {}),
    ("sim_fast.txt",    {"cycle_s": 0.6}),
    ("sim_shuffle.txt", {"mode": "shuffle"}),
    ("sim_dropout.txt", {"dropout_ch": DROPOUT_CH, "dropout_t": DROPOUT_T}),
    ("sim_stand.txt",   {"mode": "standing"}),
]

DURATION_S = 60

if __name__ == "__main__":
    for name, kwargs in STREAMS:
        with open(name, "w") as f:
            for line in gait_lines(DURATION_S, **kwargs):
                f.write(line + "\n")
        print(f"wrote {name}")
