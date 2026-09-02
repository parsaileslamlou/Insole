"""fit_calibration.py — derive the relative gain match, or run the legacy fit.

The DEFAULT action is the single-point RELATIVE gain match across the six
channels (the calibration we ship), read from the manifest and written to its
own file, gain_match.json -- never calibration.json:

    python3 fit_calibration.py                             # cwd manifest -> gain_match.json
    python3 fit_calibration.py cal_data/calibration_manifest.csv -o gain_match.json

The legacy multi-point ABSOLUTE fit -- infeasible under FSR drift, see
docs/calibration_notes.md -- fits all six sensors from a directory of bench
captures to calibration.json and stays reachable under `legacy-fit`:

    python3 fit_calibration.py legacy-fit caldata -o calibration.json -p residuals.png

Input to the legacy fit is the set of files capture_calibration.py used to write:

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
import datetime
import json
import os
import re
import sys
import time

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


# ---------------------------------------------------------------------------
# Relative gain match (the calibration we ship)
# ---------------------------------------------------------------------------
# The FSRs relax under sustained load: counts fall ~31% in 76 s at constant
# applied force and recover with a time constant of ~20 min. That is what makes
# the multi-point ABSOLUTE fit above infeasible -- there is no bench-stable
# counts->newtons mapping in the time available. What survives the drift is the
# RATIO between channels measured at the same force with the same rest interval:
# the relaxation is common to all six and cancels in the channel-to-channel
# ratio. So we ship a single-point RELATIVE GAIN MATCH, not an absolute force
# calibration, and gain_match.json records it as exactly that -- its own file,
# never calibration.json, so it can never collide with the legacy absolute fit.
# The limitations are written up in docs/calibration_notes.md.

# calibration_manifest.csv column order, mirrored from
# capture_calibration.MANIFEST_COLS. Duplicated rather than imported so this fit
# depends only on calibration + stdlib and never drags in read_serial's
# transport stack (pyserial / bleak), which capture_calibration pulls in.
MANIFEST_COLS = [
    "sensor", "trial", "csv_path", "g_min", "g_max", "force_n",
    "sigma_force_n", "n_samples", "count_mean", "count_sd",
    "saturated_frac", "timestamp_iso",
]

# The matched cycle. A manifest row qualifies only if the sensor had rested at
# least REST_MIN_MINUTES since its previous trial -- so every channel starts
# from a comparably recovered state and the shared drift really does cancel --
# AND the applied force sits in [FORCE_N_LOW, FORCE_N_HIGH], the ~12 N anchor
# all six were pressed to. These bounds are the selection contract, not tuning
# knobs: widen them and select_matched_cycle() will catch a bad selection.
REST_MIN_MINUTES = 35.0
FORCE_N_LOW = 11.4
FORCE_N_HIGH = 12.1

GAIN_MATCH_SCHEMA = 1
GAIN_MATCH_METHOD = (
    "single-point relative gain match: each channel's gain is matched to the "
    "six-channel mean at one ~12 N force with a >=35 min rest interval, so "
    "stress-relaxation drift cancels in the channel-to-channel ratio. This is "
    "NOT an absolute force calibration and must not be reported as one."
)


def load_manifest(path):
    """calibration_manifest.csv -> list of row dicts keyed by MANIFEST_COLS.

    Tolerates a file with or without the header row capture_calibration writes:
    a first cell equal to MANIFEST_COLS[0] is treated as the header and skipped.
    sensor/trial/force_n/count_mean are parsed to numbers and timestamp_iso to
    an aware datetime. Fails loudly -- a short row, an unparseable field, a
    missing or empty file all raise -- rather than silently dropping a point.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibration manifest not found: {path}")

    rows = []
    with open(path, "r", newline="") as f:
        for lineno, rec in enumerate(csv.reader(f), start=1):
            if not rec or all(c.strip() == "" for c in rec):
                continue                                    # blank line
            if rec[0].strip() == MANIFEST_COLS[0]:
                continue                                    # header, if present
            if len(rec) < len(MANIFEST_COLS):
                raise ValueError(
                    f"{path}:{lineno}: expected {len(MANIFEST_COLS)} columns "
                    f"{MANIFEST_COLS}, got {len(rec)}: {rec}")
            d = dict(zip(MANIFEST_COLS, rec))
            try:
                d["sensor"] = int(d["sensor"])
                d["trial"] = int(d["trial"])
                d["force_n"] = float(d["force_n"])
                d["count_mean"] = float(d["count_mean"])
                d["timestamp"] = datetime.datetime.fromisoformat(
                    d["timestamp_iso"])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: unparseable field: {e}") from e
            rows.append(d)

    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def with_rest_min(rows):
    """Annotate each row in place with rest_min = minutes since that sensor's
    previous trial, chronologically. The first trial per sensor gets None: it
    has no earlier trial to have rested from. Returns the same list.
    """
    by_sensor = {}
    for r in rows:
        by_sensor.setdefault(r["sensor"], []).append(r)
    for sensor_rows in by_sensor.values():
        sensor_rows.sort(key=lambda r: r["timestamp"])
        prev = None
        for r in sensor_rows:
            r["rest_min"] = (None if prev is None
                             else (r["timestamp"] - prev).total_seconds() / 60.0)
            prev = r["timestamp"]
    return rows


def select_matched_cycle(rows):
    """The rows the gain match is derived from: one per sensor, each rested
    >= REST_MIN_MINUTES and loaded within [FORCE_N_LOW, FORCE_N_HIGH].

    Asserts exactly N_SENSORS rows, exactly one per sensor 0..N_SENSORS-1, and
    raises a readable ValueError otherwise. Requires with_rest_min() to have run.
    """
    selected = [r for r in rows
                if r.get("rest_min") is not None
                and r["rest_min"] >= REST_MIN_MINUTES
                and FORCE_N_LOW <= r["force_n"] <= FORCE_N_HIGH]

    by_sensor = {}
    for r in selected:
        by_sensor.setdefault(r["sensor"], []).append(r)

    problems = []
    if len(selected) != N_SENSORS:
        problems.append(f"selected {len(selected)} rows, expected {N_SENSORS}")
    dupes = {s: [r["trial"] for r in v] for s, v in by_sensor.items() if len(v) > 1}
    if dupes:
        problems.append(f"more than one row for sensor(s) {dupes}")
    missing = [s for s in range(N_SENSORS) if s not in by_sensor]
    if missing:
        problems.append(f"no row for sensor(s) {missing}")

    if problems:
        chosen = ", ".join(
            f"s{r['sensor']}/t{r['trial']}(force={r['force_n']:.3f},"
            f"rest={r['rest_min']:.1f}m)"
            for r in sorted(selected, key=lambda r: (r["sensor"], r["trial"])))
        raise ValueError(
            "matched-cycle selection failed: " + "; ".join(problems)
            + f". criteria: rest_min >= {REST_MIN_MINUTES} min and "
            f"{FORCE_N_LOW} <= force_n <= {FORCE_N_HIGH}. "
            f"selected [{chosen}]")

    return [by_sensor[s][0] for s in range(N_SENSORS)]


def derive_gain_match(manifest_path="calibration_manifest.csv", fs=FS_COUNTS):
    """Derive per-channel relative gain corrections from the manifest.

    Returns the full gain_match.json document (a dict) with the corrections and
    the provenance needed to defend them: the FS_COUNTS used, the six rows the
    fit consumed as (sensor, trial, force_n, count_mean, rest_min), the selection
    criteria, and a method string naming this a single-point relative gain match.
    Writes nothing.

    Per matched-cycle row: x = count_mean / (fs - count_mean) and k = force_n / x.
    Correction[i] = k_i / mean(k over the six). The corrections therefore have
    mean 1.0 by construction and carry no absolute force scale -- only the
    relative gain between channels.
    """
    rows = with_rest_min(load_manifest(manifest_path))
    matched = select_matched_cycle(rows)

    ks, points = {}, []
    for r in matched:
        cm = r["count_mean"]
        x = conductance(cm, fs)                 # same transform apply uses
        if x is None:
            raise ValueError(
                f"s{r['sensor']} t{r['trial']}: count_mean {cm} is unusable at "
                f"fs={fs} (saturated or non-positive); cannot derive a gain")
        ks[r["sensor"]] = r["force_n"] / x
        points.append({
            "sensor": r["sensor"],
            "trial": r["trial"],
            "force_n": r["force_n"],
            "count_mean": cm,
            "rest_min": round(r["rest_min"], 4),
        })

    k_mean = sum(ks.values()) / len(ks)
    corrections = {s: ks[s] / k_mean for s in ks}

    return {
        "schema": GAIN_MATCH_SCHEMA,
        "kind": "relative_gain_match",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": "dimensionless relative gain",
        "method": GAIN_MATCH_METHOD,
        "model": "corrected_x = correction[i] * (counts / (fs_counts - counts))",
        "fs_counts": float(fs),
        "selection": {
            "rest_min_minutes": REST_MIN_MINUTES,
            "force_n_low": FORCE_N_LOW,
            "force_n_high": FORCE_N_HIGH,
        },
        "k": {str(s): ks[s] for s in sorted(ks)},
        "k_mean": k_mean,
        "corrections": {str(s): corrections[s] for s in sorted(corrections)},
        "points": sorted(points, key=lambda p: p["sensor"]),
    }


def write_gain_match(path, doc):
    """Write the gain-match document to `path` as JSON. Returns the doc."""
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")
    return doc


def print_gain_table(doc):
    corr = doc["corrections"]
    ks = doc["k"]
    print()
    print(f"relative gain match   fs_counts = {doc['fs_counts']:g}")
    print(f"  {doc['method']}")
    print()
    head = (f"{'s':>2}  {'trial':>5}  {'force_n':>8}  {'count_mean':>10}  "
            f"{'rest_min':>8}  {'k':>8}  {'correction':>10}")
    print(head)
    print("-" * len(head))
    for p in doc["points"]:
        s = str(p["sensor"])
        print(f"{p['sensor']:>2}  {p['trial']:>5}  {p['force_n']:>8.4f}  "
              f"{p['count_mean']:>10.3f}  {p['rest_min']:>8.2f}  "
              f"{float(ks[s]):>8.2f}  {float(corr[s]):>10.4f}")
    print()
    mean_corr = sum(float(v) for v in corr.values()) / len(corr)
    print(f"  mean k = {doc['k_mean']:.4f}   mean correction = {mean_corr:.6f} "
          f"(1.0 by construction)")


def main_gain_match(argv=None):
    p = argparse.ArgumentParser(
        description="Derive the relative gain match from calibration_manifest.csv "
                    "and write it to gain_match.json (its own file, kept separate "
                    "from the legacy absolute fit's calibration.json). This is a "
                    "single-point RELATIVE gain match across the six channels, "
                    "NOT an absolute force calibration.")
    p.add_argument("manifest", nargs="?", default="calibration_manifest.csv",
                   help="path to calibration_manifest.csv "
                        "(default: ./calibration_manifest.csv)")
    p.add_argument("-o", "--out", default="gain_match.json",
                   help="output JSON (default: gain_match.json)")
    p.add_argument("--fs", type=float, default=FS_COUNTS,
                   help=f"saturation count (default: {FS_COUNTS}, a PLACEHOLDER; "
                        f"the corrections are FS-dependent, re-derive if it moves)")
    args = p.parse_args(argv)

    try:
        doc = derive_gain_match(args.manifest, args.fs)
    except (FileNotFoundError, ValueError) as e:
        print(f"gain match failed: {e}")
        return 1

    write_gain_match(args.out, doc)
    print(f"wrote {args.out}")
    print_gain_table(doc)
    return 0


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
    # The relative gain match is the calibration we ship, so it is the default.
    # The legacy multi-point absolute fit -- infeasible under FSR drift, see
    # docs/calibration_notes.md -- stays reachable as `legacy-fit <dir>` for
    # anyone re-running an old grams sweep.
    _argv = sys.argv[1:]
    if _argv and _argv[0] == "legacy-fit":
        sys.exit(main(_argv[1:]))
    sys.exit(main_gain_match(_argv))
