"""capture_calibration.py — record one steady bench load per file.

    python3 capture_calibration.py --source serial            # live, prompts
    python3 capture_calibration.py --source file sim_walk.txt # rehearsal
    python3 capture_calibration.py --sensor 3 --grams 500 --source serial

Session shape: place a known mass on one sensor, record for --seconds, write
cal_s{N}_{grams}g.csv, repeat. When the sweep is done, run fit_calibration.py
over the directory.

Every frame source, and the frame parser itself, comes from read_serial. This
script does not know the wire format and must never learn it -- a second
parser is a second thing to keep in step with framespec.md, and the two would
diverge the first time the format moved.

--source file replays a saved capture through the identical code path, so the
whole flow can be rehearsed end to end before anyone is standing at the bench
waiting on it.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import read_serial as RS
from fit_calibration import SETTLE_S, steady_median

SAMPLE_HZ = 100                 # must match firmware
DEFAULT_SECONDS = 5.0           # RETUNE: long enough for a stable median

FILENAME = "cal_s%d_%sg.csv"


def format_grams(grams):
    """500.0 -> '500', 12.5 -> '12.5'. Must satisfy fit_calibration.NAME_RE."""
    if float(grams) == int(float(grams)):
        return str(int(float(grams)))
    return ("%.3f" % float(grams)).rstrip("0").rstrip(".")


def capture(source, in_path, seconds, quiet=False):
    """Pull frames from a read_serial source. -> (rows, counters).

    Stops on whichever comes first: the frame budget for `seconds`, the wall
    clock, or the source running dry. The frame budget is what makes --source
    file terminate -- a file yields its whole contents instantly, so a
    wall-clock limit alone would swallow the entire capture every time.
    """
    budget = max(1, int(seconds * SAMPLE_HZ))
    rows = []
    c = dict(valid=0, malformed=0, bad_checksum=0, status=0, empty=0)

    t0 = time.time()
    for line in RS.make_source(source, in_path, duration_s=seconds):
        if len(rows) >= budget:
            break
        if time.time() - t0 > seconds + 5:
            break

        if not line:
            c["empty"] += 1
            continue

        row, reason = RS.parse_frame(line)
        if row is None:
            c[reason] += 1
            if reason == "status" and not quiet:
                print(line)
            continue

        seq, ts_us, vals = row
        rows.append([seq, ts_us] + vals)
        c["valid"] += 1

        if not quiet and c["valid"] % SAMPLE_HZ == 0:
            done = len(rows) / float(budget)
            print(f"\r  {done * 100:5.1f}%  {c['valid']} frames", end="", flush=True)

    if not quiet:
        print()
    return rows, c


def write_capture(directory, sensor, grams, rows):
    path = os.path.join(directory, FILENAME % (sensor, format_grams(grams)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RS.CSV_HEADER)
        w.writerows(rows)
    return path


def report(path, rows, c, sensor):
    """Print what landed, and preview the point the fit will actually take.

    Everything below is measured over the STEADY-STATE window only, by calling
    fit_calibration's own steady_median() on the file just written. Judging the
    capture over the whole file instead would flag every single one: the
    placement transient is in there by design, and it is wider than any fault
    this is meant to catch.
    """
    print(f"  {c['valid']} valid frames, malformed={c['malformed']} "
          f"bad_checksum={c['bad_checksum']} empty={c['empty']}")
    if not rows:
        return

    med, n_used, n_total = steady_median(path, sensor, SETTLE_S)
    if med is None:
        print(f"  s{sensor}: no steady-state window")
        return

    col = 2 + sensor
    t0 = rows[0][1]
    steady = [r[col] for r in rows if (r[1] - t0) / 1e6 >= SETTLE_S] or [r[col] for r in rows]
    lo, hi = min(steady), max(steady)
    print(f"  s{sensor}: median={med:g} min={lo} max={hi} spread={hi - lo} "
          f"over {n_used}/{n_total} steady samples")

    if hi >= 4095:
        print("  WARNING: channel is at full scale. That load saturates this "
              "sensor; the point will be DROPPED from the fit, not clamped. "
              "Use a lighter mass.")
    if hi - lo > 200:
        print("  WARNING: wide spread after settling -- indenter may have "
              "shifted mid-capture. Consider re-taking this load.")


def ask(prompt):
    """Read one line. None on EOF or a quit word."""
    try:
        s = input(prompt).strip()
    except EOFError:
        print()
        return None
    return None if s.lower() in ("q", "quit", "exit") else s


def prompt_one(default_sensor=None):
    """-> (sensor, grams), or None to end the session."""
    while True:
        s = ask("sensor index 0-5 (blank=repeat last, q=quit): ")
        if s is None:
            return None
        if not s and default_sensor is not None:
            sensor = default_sensor
        else:
            try:
                sensor = int(s)
            except ValueError:
                print("  not a number")
                continue
            if not (0 <= sensor < 6):
                print("  sensor index must be 0..5")
                continue

        g = ask("mass in grams (q=quit): ")
        if g is None:
            return None
        try:
            grams = float(g)
        except ValueError:
            print("  not a number")
            continue
        if grams < 0:
            print("  mass cannot be negative")
            continue
        return sensor, grams


def confirm_overwrite(path, interactive, force):
    if not os.path.exists(path) or force:
        return True
    if not interactive:
        print(f"  refusing to overwrite {path} (pass --force)")
        return False
    a = ask(f"  {os.path.basename(path)} exists. overwrite? [y/N]: ")
    return bool(a) and a.lower().startswith("y")


def run_one(args, sensor, grams, interactive):
    path = os.path.join(args.dir, FILENAME % (sensor, format_grams(grams)))
    if not confirm_overwrite(path, interactive, args.force):
        return None

    if interactive:
        a = ask(f"place {format_grams(grams)} g on s{sensor}, ENTER to record "
                f"{args.seconds:g}s (q=quit): ")
        if a is None:
            return None

    print(f"recording s{sensor} @ {format_grams(grams)} g for {args.seconds:g}s...")
    rows, c = capture(args.source, args.in_path, args.seconds, args.quiet)

    if not rows:
        print("  no valid frames -- nothing written. Check the link and retry.")
        return None

    written = write_capture(args.dir, sensor, grams, rows)
    report(written, rows, c, sensor)
    print(f"  wrote {written}")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Record steady bench loads into cal_s{N}_{grams}g.csv.")
    p.add_argument("in_path", nargs="?", default=None,
                   help="capture to replay; giving it implies --source file")
    p.add_argument("--source", choices=("serial", "file", "ble"), default=None,
                   help="transport; defaults to 'file' when in_path is given, "
                        f"else '{RS.DEFAULT_SOURCE}'")
    p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                   help=f"seconds per load (default: {DEFAULT_SECONDS:g})")
    p.add_argument("--dir", default=".", help="output directory (default: .)")
    p.add_argument("--sensor", type=int, default=None,
                   help="skip the prompt and capture this sensor once")
    p.add_argument("--grams", type=float, default=None,
                   help="skip the prompt and capture this mass once")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing capture without asking")
    p.add_argument("--quiet", action="store_true", help="no progress output")
    args = p.parse_args(argv)

    # Same defaulting rule as read_serial, so the two CLIs behave alike.
    if args.source is None:
        args.source = "file" if args.in_path else RS.DEFAULT_SOURCE
    if args.source == "file" and args.in_path is None:
        args.in_path = RS.IN_FILE
    if args.source != "file" and args.in_path is not None:
        p.error(f"in_path is only meaningful with --source file, "
                f"got --source {args.source}")

    if not os.path.isdir(args.dir):
        print(f"not a directory: {args.dir}")
        return 1

    one_shot = args.sensor is not None or args.grams is not None
    if one_shot:
        if args.sensor is None or args.grams is None:
            p.error("--sensor and --grams must be given together")
        if not (0 <= args.sensor < 6):
            p.error("--sensor must be 0..5")
        return 0 if run_one(args, args.sensor, args.grams, False) else 1

    interactive = sys.stdin.isatty()
    print(f"source={args.source} seconds={args.seconds:g} dir={args.dir}")
    print("One load per capture. Ctrl-D or 'q' ends the session.\n")

    last_sensor = None
    n = 0
    while True:
        got = prompt_one(last_sensor)
        if got is None:
            break
        sensor, grams = got
        if run_one(args, sensor, grams, interactive):
            n += 1
            last_sensor = sensor
        print()

    print(f"session over, {n} capture(s) written to {args.dir}")
    print(f"next: python3 fit_calibration.py {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
