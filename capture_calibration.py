"""capture_calibration.py — interactive scale-referenced bench capture.

    python3 capture_calibration.py --source serial          # live bench session
    python3 capture_calibration.py sim_walk.txt             # rehearse via file
    python3 capture_calibration.py --source ble --dwell 3   # longer dwell

One trial per press. Unlike the old known-mass sweep, the applied force is NOT
known in advance: you press the indentor on a sensor, the load lands on a
scale, and the scale reading is entered AFTER the press as an interval. Each
kept trial is written as cal_s{N}_t{K}.csv (raw frames) and summarised as one
row in calibration_manifest.csv, which is what a scale-aware fit consumes.

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
import datetime
import math
import os
import re
import statistics
import sys
import time

import read_serial as RS
from calibration import FS_COUNTS

SAMPLE_HZ = 100                 # must match firmware
DEFAULT_DWELL = 2.0             # seconds captured per press; --dwell overrides

MANIFEST = "calibration_manifest.csv"
MANIFEST_COLS = [
    "sensor", "trial", "csv_path", "g_min", "g_max", "force_n",
    "sigma_force_n", "n_samples", "count_mean", "count_sd",
    "saturated_frac", "timestamp_iso",
]

# cal_s3_t0.csv -> sensor 3, trial 0. Trial index only, no grams: the applied
# force lives in calibration_manifest.csv now, not in the filename.
TRIAL_RE = re.compile(r"^cal_s(\d+)_t(\d+)\.csv$")

FILENAME = "cal_s%d_t%d.csv"


def capture(source, in_path, dwell, quiet=False):
    """Pull frames from a read_serial source for one dwell window.

    -> (rows, counters). Stops on whichever comes first: the frame budget for
    `dwell`, the wall clock, or the source running dry. The frame budget is
    what makes --source file terminate -- a file yields its whole contents
    instantly, so a wall-clock limit alone would swallow the entire capture
    every time.
    """
    budget = max(1, int(dwell * SAMPLE_HZ))
    rows = []
    c = dict(valid=0, malformed=0, bad_checksum=0, status=0, empty=0)

    t0 = time.time()
    for line in RS.make_source(source, in_path, duration_s=dwell):
        if len(rows) >= budget:
            break
        if time.time() - t0 > dwell + 5:
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


def window_stats(rows, sensor):
    """Count statistics for one sensor's channel over the WHOLE dwell window.

    No steady-state trimming: a hand press has no settling transient to drop
    the way a placed weight did, and the spread is now data, not something to
    discard. Assumes `rows` is non-empty (the caller checks).
    """
    col = 2 + sensor
    counts = [r[col] for r in rows]
    n = len(counts)
    n_sat = sum(1 for v in counts if v == FS_COUNTS)
    return {
        "n": n,
        "mean": sum(counts) / n,
        "sd": statistics.stdev(counts) if n >= 2 else 0.0,
        "min": min(counts),
        "max": max(counts),
        "n_sat": n_sat,
        "sat_frac": n_sat / n,
    }


def print_window(sensor, st):
    """Report the channel over the window, then the two warnings."""
    print(f"  s{sensor}: n={st['n']} mean={st['mean']:.1f} sd={st['sd']:.1f} "
          f"min={st['min']} max={st['max']} sat_frac={st['sat_frac']:.3f}")

    # Saturation is a hard warning: at full scale the divider has bottomed out
    # and the load is unknowable, so any fit point from this press is dropped.
    if st["n_sat"]:
        print(f"  WARNING: full scale (count == {FS_COUNTS}) on {st['n_sat']} "
              f"of {st['n']} samples -- this press saturates the sensor and any "
              f"fit point from it will be DROPPED. Press lighter.")

    # Wide spread used to gate a re-take. With hand pressing it is normal, and
    # the spread is carried into the fit as sigma rather than thrown away, so it
    # is now informational only.
    spread = st["max"] - st["min"]
    if spread > 200:
        print(f"  note: wide count spread ({spread}). Hand pressing makes this "
              f"normal; it is carried into the fit as sigma, not a fault.")


def ask_enter(prompt):
    """Blocking prompt that only waits for Enter. -> True, or False on EOF."""
    try:
        input(prompt)
    except EOFError:
        print()
        return False
    return True


def prompt_sensor():
    """-> sensor int 0-5, or None to end the session."""
    while True:
        try:
            s = input("sensor number 0-5 (q=quit): ").strip()
        except EOFError:
            print()
            return None
        if s.lower() in ("q", "quit", "exit"):
            return None
        try:
            sensor = int(s)
        except ValueError:
            print("  not a number")
            continue
        if not (0 <= sensor < 6):
            print("  sensor number must be 0..5")
            continue
        return sensor


def prompt_grams(label, floor=None):
    """Read a gram value for `label`. Re-prompts on non-numeric input, and on
    a value below `floor` when given (that is the max < min guard). None on EOF.
    """
    while True:
        try:
            s = input(f"scale {label} (grams): ").strip()
        except EOFError:
            print()
            return None
        try:
            g = float(s)
        except ValueError:
            print("  not a number")
            continue
        if floor is not None and g < floor:
            print(f"  maximum {g:g} is below minimum {floor:g}; re-enter")
            continue
        return g


def prompt_decision():
    """-> 'keep' | 'redo' | 'discard'. EOF is treated as discard: nothing is
    written on an interrupted trial."""
    while True:
        try:
            s = input("  keep / redo / discard [k/r/d]: ").strip().lower()
        except EOFError:
            print()
            return "discard"
        if s in ("k", "keep"):
            return "keep"
        if s in ("r", "redo"):
            return "redo"
        if s in ("d", "discard"):
            return "discard"
        print("  answer k, r, or d")


def next_trial_index(directory, sensor):
    """One past the highest existing cal_s{sensor}_t{K}.csv in `directory`."""
    hi = -1
    for name in os.listdir(directory):
        m = TRIAL_RE.match(name)
        if m and int(m.group(1)) == sensor:
            hi = max(hi, int(m.group(2)))
    return hi + 1


def write_capture(directory, sensor, trial, rows):
    path = os.path.join(directory, FILENAME % (sensor, trial))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RS.CSV_HEADER)
        w.writerows(rows)
    return path


def append_manifest(directory, sensor, trial, csv_path, g_min, g_max,
                    force_n, sigma_force_n, st):
    """Append one row to calibration_manifest.csv, writing the header first if
    the file does not exist yet. Append-only: existing rows are never rewritten,
    so re-running a session adds trials without disturbing anything already fit.

    g_min and g_max are stored raw so force_n and sigma_force_n stay
    recomputable from them; the derived columns are a convenience.
    """
    manifest = os.path.join(directory, MANIFEST)
    exists = os.path.exists(manifest)
    with open(manifest, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(MANIFEST_COLS)
        w.writerow([
            sensor,
            trial,
            os.path.basename(csv_path),
            g_min,
            g_max,
            force_n,
            sigma_force_n,
            st["n"],
            round(st["mean"], 4),
            round(st["sd"], 4),
            round(st["sat_frac"], 6),
            datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        ])
    return manifest


def run_trial(args, sensor):
    """One scale-referenced trial for `sensor`.

    -> 'kept' | 'discarded' | 'quit'. 'redo' at the decision prompt re-runs
    from the press with the same sensor, so a fat-fingered scale entry or a bad
    press costs one more press, not a restart of the session.
    """
    while True:
        # Step 2: position the indentor and press Enter (blocking).
        if not ask_enter(f"s{sensor}: position indentor, press Enter to start "
                          f"({args.dwell:g}s dwell), Ctrl-D to quit: "):
            return "quit"

        # Step 3: capture the fixed dwell window.
        print(f"recording s{sensor} for {args.dwell:g}s...")
        rows, _ = capture(args.source, args.in_path, args.dwell, args.quiet)
        if not rows:
            print("  no valid frames captured -- check the link. Re-positioning.")
            continue

        # Step 4: report the channel over the window.
        st = window_stats(rows, sensor)
        print_window(sensor, st)

        # Step 5: the scale reading, entered as the interval it actually is.
        g_min = prompt_grams("minimum")
        if g_min is None:
            return "quit"
        g_max = prompt_grams("maximum", floor=g_min)
        if g_max is None:
            return "quit"

        # Step 6: interval -> force. The scale reading is known only to lie
        # within [g_min, g_max]; modelling that interval as uniform gives mean
        # (g_min+g_max)/2 and SD (g_max-g_min)/sqrt(12) -- the uniform SD, not
        # the interval half-width. Equal bounds leave sigma_force_n at exactly
        # 0.0; it is neither clamped nor floored, only flagged below so a
        # single-value entry is caught before it is written as certainty.
        G_PER_N = 9.81 / 1000.0
        force_n = G_PER_N * (g_min + g_max) / 2.0
        sigma_force_n = G_PER_N * (g_max - g_min) / math.sqrt(12.0)

        # Step 7: summary, then keep/redo/discard. force_n and sigma_force_n are
        # shown so a fat-fingered pair is caught before anything is written.
        print("  --- trial summary ---")
        print(f"  s{sensor}  scale [{g_min:g}, {g_max:g}] g")
        print(f"  force_n = {force_n:.4f} N   sigma_force_n = {sigma_force_n:.4f} N")
        if sigma_force_n == 0.0:
            print("  zero force uncertainty - check entry")

        decision = prompt_decision()
        if decision == "redo":
            print("  redo -- re-press the same sensor")
            continue
        if decision == "discard":
            print("  discarded, nothing written")
            return "discarded"

        # Step 8: keep -> write the raw frames and one manifest row.
        trial = next_trial_index(args.dir, sensor)
        path = write_capture(args.dir, sensor, trial, rows)
        append_manifest(args.dir, sensor, trial, path, g_min, g_max,
                        force_n, sigma_force_n, st)
        print(f"  wrote {path}")
        print(f"  appended trial {trial} to {os.path.join(args.dir, MANIFEST)}")
        return "kept"


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Interactive scale-referenced calibration capture. One "
                    "trial per press; force is read off a scale after the "
                    "press, not known in advance.")
    p.add_argument("in_path", nargs="?", default=None,
                   help="capture to replay; giving it implies --source file")
    p.add_argument("--source", choices=("serial", "file", "ble"), default=None,
                   help="transport; defaults to 'file' when in_path is given, "
                        f"else '{RS.DEFAULT_SOURCE}'")
    p.add_argument("--dwell", type=float, default=DEFAULT_DWELL,
                   help=f"seconds captured per press (default: {DEFAULT_DWELL:g})")
    p.add_argument("--dir", default=".", help="output directory (default: .)")
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

    print(f"source={args.source} dwell={args.dwell:g}s dir={args.dir}")
    print("One trial per press. The scale reading is entered after each press. "
          "'q' at the sensor prompt ends the session.\n")

    kept = 0
    while True:
        sensor = prompt_sensor()
        if sensor is None:
            break
        outcome = run_trial(args, sensor)
        if outcome == "quit":
            break
        if outcome == "kept":
            kept += 1
        print()

    print(f"session over, {kept} trial(s) kept in {args.dir}")
    print(f"next: python3 fit_calibration.py {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
