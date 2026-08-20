"""fit_calibration.py — fit all six sensors from a directory of bench captures.

    python3 fit_calibration.py                     # cwd -> calibration.json
    python3 fit_calibration.py caldata -o cal.json -p residuals.png

Input is the set of files capture_calibration.py writes:

    cal_s{N}_{grams}g.csv        N in 0..5, grams as typed at the bench

each holding one steady load on one sensor, in read_serial's CSV layout
(seq, ts_us, s0..s5). Sensor index and applied mass are read from the
filename, so a mislabelled file is a mislabelled point -- the contents carry
no record of which sensor was loaded.

Per file this takes the MEDIAN of the relevant channel over the steady-state
window, not the mean. Seating the indenter produces a spike at the start of
every capture and often a smaller one as it is lifted; the mean folds those
transients into the calibration point, while the median ignores them as long
as they occupy less than half the window. The first SETTLE_S seconds are
dropped outright as placement time.

Only matplotlib is needed, and only for the residual PNG. The table still
prints without it.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

from calibration import (
    FS_COUNTS, conductance, fit_sensor, is_saturated, missing_fit, save_calibration,
    N_SENSORS, FLAG_OK, FLAG_NO_DATA,
)

# cal_s3_500g.csv -> sensor 3, 500 g. Fractional masses are allowed for
# small trim weights; the 'g' suffix is required so a stray CSV in the
# directory is ignored rather than half-parsed.
NAME_RE = re.compile(r"^cal_s(\d+)_(\d+(?:\.\d+)?)g\.csv$")

SAMPLE_HZ = 100                 # must match firmware; only used as a ts_us fallback

# Placement transient. Everything before this is discarded, the rest of the
# file is the steady-state window.
SETTLE_S = 1.0                  # RETUNE: measure how long the indenter takes to settle

# A capture shorter than this after settling has too few samples for the
# median to reject anything.
MIN_STEADY_SAMPLES = 20         # RETUNE


def discover(directory):
    """-> {sensor_index: [(grams, path), ...]}, sorted by mass."""
    found = {}
    for name in sorted(os.listdir(directory)):
        m = NAME_RE.match(name)
        if not m:
            continue
        idx = int(m.group(1))
        grams = float(m.group(2))
        if not (0 <= idx < N_SENSORS):
            print(f"  skip {name}: sensor index {idx} outside 0..{N_SENSORS - 1}")
            continue
        found.setdefault(idx, []).append((grams, os.path.join(directory, name)))
    for idx in found:
        found[idx].sort()
    return found


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def steady_median(path, sensor, settle_s=SETTLE_S):
    """Median count on channel `sensor` over the steady-state window.

    Returns (median, n_used, n_total). median is None when the file has no
    usable steady-state window at all.
    """
    col = "s%d" % sensor
    times, counts = [], []

    with open(path, "r", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if col not in row:
                raise ValueError(f"{path}: no column {col!r}; header is {list(row)}")
            try:
                c = int(row[col])
            except (TypeError, ValueError):
                continue
            try:
                t = int(row["ts_us"]) / 1e6
            except (TypeError, ValueError, KeyError):
                t = i / float(SAMPLE_HZ)    # fall back to nominal cadence
            times.append(t)
            counts.append(c)

    n_total = len(counts)
    if n_total == 0:
        return None, 0, 0

    t0 = times[0]
    steady = [c for t, c in zip(times, counts) if (t - t0) >= settle_s]

    # A capture barely longer than the settle time leaves nothing to take a
    # median over. Fall back to the whole file and say so, rather than
    # silently dropping the load from the sweep.
    if len(steady) < MIN_STEADY_SAMPLES:
        print(f"  {os.path.basename(path)}: only {len(steady)} samples after "
              f"{settle_s:g}s settle; using all {n_total} instead")
        steady = counts

    return _median(steady), len(steady), n_total


def used_points(grams, counts, fs):
    """The (grams, counts) pairs fit_sensor() will actually keep, same order.

    Must apply the same two filters fit_sensor does, in the same order, or the
    residual plot pairs residuals with the wrong loads.
    """
    return [(g, c) for g, c in zip(grams, counts)
            if not is_saturated(c, fs) and conductance(c, fs) is not None]


def fit_all(directory, fs=FS_COUNTS, settle_s=SETTLE_S):
    """-> (per_sensor fits, per_sensor sweep points)."""
    found = discover(directory)
    fits, sweeps = {}, {}

    for idx in range(N_SENSORS):
        files = found.get(idx, [])
        if not files:
            fits[idx] = missing_fit(fs)
            sweeps[idx] = ([], [])
            continue

        grams, counts = [], []
        for g, path in files:
            med, n_used, n_total = steady_median(path, idx, settle_s)
            if med is None:
                print(f"  {os.path.basename(path)}: no usable rows, skipped")
                continue
            grams.append(g)
            counts.append(med)

        fits[idx] = fit_sensor(grams, counts, fs)
        sweeps[idx] = (grams, counts)

    return fits, sweeps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(fits, fs):
    def fmt(v, spec):
        if v is not None:
            return format(v, spec)
        return format("--", ">" + spec.split(".")[0])     # keep the columns aligned

    print()
    print(f"fs_counts = {fs:g}    force_n = a * counts/(fs - counts) + b")
    print()
    head = f"{'s':>2}  {'a (N)':>12}  {'b (N)':>10}  {'R2':>7}  {'n':>3}  {'sat':>3}  flag"
    print(head)
    print("-" * len(head))
    for i in range(N_SENSORS):
        f = fits[i]
        flag = f["flag"]
        mark = "" if flag == FLAG_OK else "   <-- FLAGGED"
        print(f"{i:>2}  {fmt(f['a'], '12.4f')}  {fmt(f['b'], '10.4f')}  "
              f"{fmt(f['r2'], '7.4f')}  {f['n_points']:>3}  {f['n_saturated']:>3}  "
              f"{flag}{mark}")
    print()

    bad = [i for i in range(N_SENSORS) if fits[i]["flag"] != FLAG_OK]
    if bad:
        print(f"{len(bad)} of {N_SENSORS} sensors flagged: "
              + ", ".join(f"s{i} ({fits[i]['flag']})" for i in bad))
        print("Flagged sensors return None from apply_calibration(). Re-seat, "
              "re-load and re-run before capturing gait.")
    else:
        print(f"all {N_SENSORS} sensors fit cleanly")


def plot_residuals(fits, sweeps, path, fs):
    """2x3 residual grid, one panel per sensor. Returns True if written."""
    try:
        import matplotlib
        matplotlib.use("Agg")           # headless: no display at the bench
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed, skipping {path}")
        return False

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    for i in range(N_SENSORS):
        ax = axes[i // 3][i % 3]
        f = fits[i]
        grams, counts = sweeps[i]
        resid = f["residuals"]

        # Residual against APPLIED LOAD, not against the conductance x: a bad
        # point is then read straight off the axis as "the 500 g capture" and
        # can be re-taken on the spot.
        loads = [g for g, _ in used_points(grams, counts, fs)]

        if resid and len(loads) == len(resid):
            ax.axhline(0.0, lw=1, color="0.6")
            ax.plot(loads, resid, "o-", ms=5, lw=1)
            rng = max(abs(r) for r in resid) or 1.0
            ax.set_ylim(-1.6 * rng, 1.6 * rng)
        else:
            ax.text(0.5, 0.5, f["flag"], ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="crimson")
            ax.set_xticks([])
            ax.set_yticks([])

        r2 = f["r2"]
        r2s = f"R2={r2:.4f}" if r2 is not None else "R2=--"
        colour = "black" if f["flag"] == FLAG_OK else "crimson"
        ax.set_title(f"s{i}  {r2s}  n={f['n_points']}  sat={f['n_saturated']}",
                     fontsize=10, color=colour)
        ax.set_xlabel("applied load (g)", fontsize=8)
        ax.set_ylabel("residual (N)", fontsize=8)
        ax.tick_params(labelsize=8)

    fig.suptitle("Calibration residuals -- curvature here means the linear-in-"
                 "conductance model is wrong for that sensor", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"wrote {path}")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Fit all six FSRs from cal_s{N}_{grams}g.csv captures.")
    p.add_argument("directory", nargs="?", default=".",
                   help="directory holding the capture CSVs (default: .)")
    p.add_argument("-o", "--out", default="calibration.json",
                   help="output JSON (default: calibration.json)")
    p.add_argument("-p", "--plot", default="calibration_residuals.png",
                   help="residual grid PNG (default: calibration_residuals.png)")
    p.add_argument("--no-plot", action="store_true", help="skip the PNG")
    p.add_argument("--fs", type=float, default=FS_COUNTS,
                   help=f"saturation count (default: {FS_COUNTS}, a PLACEHOLDER)")
    p.add_argument("--settle", type=float, default=SETTLE_S,
                   help=f"seconds discarded at the start of each capture "
                        f"(default: {SETTLE_S})")
    args = p.parse_args(argv)

    if not os.path.isdir(args.directory):
        print(f"not a directory: {args.directory}")
        return 1

    print(f"reading {args.directory}/cal_s*_*g.csv")
    fits, sweeps = fit_all(args.directory, args.fs, args.settle)

    if all(fits[i]["flag"] == FLAG_NO_DATA for i in range(N_SENSORS)):
        print("no cal_s{N}_{grams}g.csv files found -- nothing to fit")
        return 1

    save_calibration(args.out, fits, fs=args.fs,
                     notes=f"fit from {os.path.abspath(args.directory)}")
    print(f"wrote {args.out}")

    print_table(fits, args.fs)

    if not args.no_plot:
        plot_residuals(fits, sweeps, args.plot, args.fs)

    # Flagged sensors are a bench result, not a crash: the file is written
    # either way so the good channels are usable, but the exit code is nonzero
    # so a scripted session stops rather than moving on to gait capture.
    return 0 if all(fits[i]["flag"] == FLAG_OK for i in range(N_SENSORS)) else 2


if __name__ == "__main__":
    sys.exit(main())
