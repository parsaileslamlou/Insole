
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
    

import argparse
import math
import random
import sys

CYCLE_S = 1.0                      # RETUNE: nominal stride period, seconds
# RETUNE: per-sensor load windows as fractions of the stride, heel-first.
SENSOR_WINDOWS = [(0.00, 0.30), (0.15, 0.45), (0.30, 0.55),
                  (0.32, 0.57), (0.35, 0.58), (0.45, 0.62)]
AMPLITUDES = [3200, 1400, 2600, 2800, 2200, 1800]   # RETUNE: peak ADC counts per channel
NOISE_STD = 25                     # RETUNE: measure against a real noise-floor capture

STANDING_LOADS = [1800, 900, 1200, 1300, 1000, 400]  # RETUNE: static ADC counts per channel

# --- nasty stream parameters -------------------------------------
SHUFFLE_CYCLE_S   = 0.50           # RETUNE: short-stride shuffle stride period
SHUFFLE_AMP_SCALE = 0.45           # RETUNE: shuffling loads the foot more lightly
DROPOUT_CH        = 0              # RETUNE: which channel dies (s0 = heel, worst case)
DROPOUT_T         = (20.0, 40.0)   # RETUNE: seconds; dead window inside a 60 s capture

def resolve_cycle(mode, cycle_s=None):
    if cycle_s is not None:
        return cycle_s
    return SHUFFLE_CYCLE_S if mode == "shuffle" else CYCLE_S

def sensor_value(t, i, mode="walk", cycle_s=None,
                 dropout_ch=None, dropout_t=(None, None)):
    cycle_s = resolve_cycle(mode, cycle_s)

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

# --- fault injection ---------------------------------------------
# Three flag-controlled, seeded, composable fault modes on any stream. All
# three default OFF, and with all three off gait_lines() is byte-identical to
# what it produced before they existed: the fault RNG is a separate
# random.Random that is not even constructed unless a mode is on, so the
# sensor-noise draw sequence is untouched (test_faults.py pins this).
#
#   drop_rate     each frame is omitted with this probability. SEQ and TS_US
#                 still advance -- the board produced the frame, the link
#                 lost it -- so the consumer sees a sequence gap.
#   corrupt_rate  each frame is emitted with a wrong checksum with this
#                 probability. Every field is intact; only the checksum byte
#                 is off, so the line passes every host gate but the last.
#   reset_at_s    at this time the emitter behaves like a rebooted board: a
#                 couple of non-frame boot lines, then SEQ restarts at 0 and
#                 TS_US restarts at 0. The gait carries on -- the foot did
#                 not reboot -- so a stance in progress is split across it.
#   fault_seed    seeds the fault RNG. Deterministic by default.
#
# Per frame the order is reset (index-based), then drop, then corrupt. A
# dropped frame is not also corrupted: it never reaches the wire.
FAULT_SEED = 0

# What a rebooting board puts on the wire before its first frame, as seen
# over a UART bridge. The ROM line comes from the chip, not the sketch, and
# does not start with '#': the host frame gate rejects it as malformed, the
# same way it rejects the boot text at the start of a capture. The second is
# the sketch's own banner (insole.ino setup()), a '#' status line.
BOOT_LINES = [
    "rst:0x1 (POWERON),boot:0x8 (SPI_FAST_FLASH_BOOT)",
    "# ble advertising as INSOLE",
]


def corrupt_checksum(line, rng):
    """`line` with a checksum that is wrong by construction (never the right one)."""
    head, ck = line.rsplit(",", 1)
    bad = (int(ck) + 1 + rng.randrange(255)) % 256
    return f"{head},{bad}"


def gait_lines(duration_s, mode="walk", cycle_s=None,
               dropout_ch=None, dropout_t=(None, None),
               drop_rate=0.0, corrupt_rate=0.0, reset_at_s=None,
               fault_seed=FAULT_SEED):
    n = int(duration_s * SAMPLE_HZ)
    faults = drop_rate > 0 or corrupt_rate > 0 or reset_at_s is not None
    rng = random.Random(fault_seed) if faults else None
    k_reset = None if reset_at_s is None else int(reset_at_s * SAMPLE_HZ)
    base = 0                       # frame index at which the board last booted
    for k in range(n):
        t = k / SAMPLE_HZ
        readings = [sensor_value(t, i, mode, cycle_s, dropout_ch, dropout_t)
                    for i in range(6)]
        if k_reset is not None and k == k_reset:
            for boot in BOOT_LINES:
                yield boot
            base = k
        seq, ts = (k - base) % 65536, (k - base) * PERIOD_US
        if not faults:
            yield make_frame(seq, ts, readings)
            continue
        if drop_rate > 0 and rng.random() < drop_rate:
            continue
        line = make_frame(seq, ts, readings)
        if corrupt_rate > 0 and rng.random() < corrupt_rate:
            line = corrupt_checksum(line, rng)
        yield line

STANCE_START = min(w[0] for w in SENSOR_WINDOWS)
STANCE_END   = max(w[1] for w in SENSOR_WINDOWS)

def true_stances(duration_s, mode="walk", cycle_s=None):
    """Ground-truth stance intervals as (start_frame, end_frame), inclusive.

    STANCE_START/STANCE_END are phase fractions, not seconds, so they scale
    with cycle_s. A 0.6 s stride carries a 0.62 * 0.6 = 0.372 s stance.

    mode="standing" has no steps, so the answer is [] and not one long stance.
    mode="dropout" is deliberately identical to walk: a dead sensor does not
    delete a step, and truth must not be bent to match what the detector can
    see or the test becomes circular.
    """
    cycle_s = resolve_cycle(mode, cycle_s)
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


def build_parser():
    p = argparse.ArgumentParser(
        description="Simulated insole frames. With no arguments, write the five "
                    "regression streams (sim_walk/fast/shuffle/dropout/stand.txt). "
                    "With --out, write one stream, optionally with injected faults.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="write a single stream here ('-' for stdout) instead of the five files")
    p.add_argument("--mode", choices=("walk", "shuffle", "standing"), default="walk")
    p.add_argument("--cycle", type=float, default=None, metavar="SECONDS",
                   help="stride period (default: the mode's; 0.6 gives the 'fast' stream)")
    p.add_argument("--duration", type=float, default=DURATION_S, metavar="SECONDS")
    p.add_argument("--dropout", action="store_true",
                   help=f"channel s{DROPOUT_CH} dead over {DROPOUT_T[0]:g}-{DROPOUT_T[1]:g} s, as sim_dropout")
    p.add_argument("--drop-rate", type=float, default=0.0, metavar="P",
                   help="omit each frame with probability P (sequence gaps)")
    p.add_argument("--corrupt-rate", type=float, default=0.0, metavar="P",
                   help="emit each frame with a wrong checksum with probability P")
    p.add_argument("--reset-at", type=float, default=None, metavar="SECONDS",
                   help="board reboot at this time: boot lines, SEQ and TS_US restart at 0")
    p.add_argument("--fault-seed", type=int, default=FAULT_SEED,
                   help=f"seed for the fault RNG (default {FAULT_SEED})")
    p.add_argument("--noise-seed", type=int, default=None,
                   help="seed the sensor-noise RNG; default unseeded, as always")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    faults = args.drop_rate > 0 or args.corrupt_rate > 0 or args.reset_at is not None
    if args.out is None:
        if faults or args.mode != "walk" or args.cycle is not None or args.dropout:
            build_parser().error("--mode/--cycle/--dropout and the fault flags need --out")
        if args.noise_seed is not None:
            random.seed(args.noise_seed)
        for name, kwargs in STREAMS:
            with open(name, "w") as f:
                for line in gait_lines(args.duration, **kwargs):
                    f.write(line + "\n")
            print(f"wrote {name}")
        return 0

    if args.noise_seed is not None:
        random.seed(args.noise_seed)
    kwargs = dict(mode=args.mode, cycle_s=args.cycle,
                  drop_rate=args.drop_rate, corrupt_rate=args.corrupt_rate,
                  reset_at_s=args.reset_at, fault_seed=args.fault_seed)
    if args.dropout:
        kwargs.update(dropout_ch=DROPOUT_CH, dropout_t=DROPOUT_T)
    lines = gait_lines(args.duration, **kwargs)
    if args.out == "-":
        for line in lines:
            sys.stdout.write(line + "\n")
        return 0
    n = 0
    with open(args.out, "w") as f:
        for line in lines:
            f.write(line + "\n")
            n += 1
    print(f"wrote {args.out}: {n} lines, {int(args.duration * SAMPLE_HZ)} frames generated"
          + (f", drop_rate={args.drop_rate:g}" if args.drop_rate else "")
          + (f", corrupt_rate={args.corrupt_rate:g}" if args.corrupt_rate else "")
          + (f", reset_at={args.reset_at:g}s" if args.reset_at is not None else "")
          + (f", fault_seed={args.fault_seed}" if faults else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
