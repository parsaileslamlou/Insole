"""Phase C of Prompt 13: the four _02 real captures through the pipeline.

Run from the repo root:

    python analyze_real.py

Every number quoted in docs/sim_vs_real.md sections C1-C6 is printed by this
script, so a reader can regenerate the writeup's inputs rather than trusting
the prose. Nothing here writes into data/real/; that directory is read-only.

Why only the _02 set: the _01 captures have an intermittent s0 (flat zero in
fast_01 and shuffle_01, coming alive mid-file in walk_01 around seq 205-210).
Root-caused to a missing strain relief on the FSR tail, since fixed. _01 is
kept as failure evidence and is deliberately not analysed, trained on, or
merged in -- see data/real/README.md.

The detector thresholds are whatever detector.py holds when this runs; the
script reads them, never sets them. When it was first run (Prompt 13) all five
were the simulator-swept values, and its C5 diagnostic showed MAX_DURATION = 120
discarding 17/35 walk and 28/30 shuffle contacts. MAX_DURATION was then raised
to 200 on the strength of that diagnostic (see its comment in detector.py and
sweep_max_duration.py); re-running this script is how the change was verified.
T_ON, T_OFF, MIN_DURATION and GAP_MERGE are still the simulator-swept values.
"""

import os
import sys

import numpy as np
import pandas as pd

import calibration as C
import detector as D
from features import cop_frame, extract_features, frame_dt

REPO = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(REPO, "data", "real")

# As-captured filenames. stand_02 has an underscore and the other three do not;
# they are committed under the names the capture tool produced, so the mapping
# is spelled out here rather than guessed from a pattern that does not hold.
ACTIVITIES = [
    ("stand",   "stand_02.csv"),
    ("walk",    "walk02.csv"),
    ("fast",    "fast02.csv"),
    ("shuffle", "shuffle02.csv"),
]

# The highest count reached by any calibration sample. Above this the gain
# match is extrapolating past every force it was derived from. Defined once, in
# calibration.py, from cal_data/; the 940 that used to be typed here was not.
CAL_MAX_COUNTS = float(C.CAL_MAX_COUNTS)

# The single force the relative gain match was derived at, for the record.
GAIN_MATCH_FORCE_N = 12.0


def rule(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load(fname):
    df = pd.read_csv(os.path.join(REAL, fname))
    missing = [c for c in ["seq", "ts_us"] + D.SENSOR_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{fname}: missing columns {missing}")
    return df


def _runs_unbounded(total, t_on, t_off):
    """T_ON/T_OFF hysteresis with NO duration limits, for diagnostics only.

    find_stances applies MIN_DURATION and MAX_DURATION inside the same loop, so
    it cannot report what it threw away. This reproduces only the two-state
    crossing logic, which is what "how long is a real contact actually" needs.
    """
    runs, in_run, start = [], False, 0
    for i, x in enumerate(total):
        if in_run:
            if x < t_off:
                runs.append((start, i - 1))
                in_run = False
        elif x >= t_on:
            start, in_run = i, True
    if in_run:
        runs.append((start, len(total) - 1))
    return runs


def c1_ingest(data):
    """Frame count, duration, sampling interval and its jitter, per-sensor range."""
    rule("C1  INGEST -- frame counts, timing, per-sensor ranges")

    print(f"{'activity':9s} {'frames':>7s} {'dur_s':>7s} {'seq_gaps':>9s} "
          f"{'dt_med_us':>10s} {'dt_min':>7s} {'dt_max':>8s} {'dt_std':>8s} {'dt_iqr':>7s}")
    for name, df in data.items():
        ts = df["ts_us"].to_numpy(dtype=float)
        d = np.diff(ts)
        dur = (ts[-1] - ts[0]) / 1e6
        # A dropped frame shows up as a jump in seq, which is not the same
        # thing as timing jitter; report both so they are not conflated.
        seq_gaps = int((np.diff(df["seq"].to_numpy(dtype=np.int64)) != 1).sum())
        q75, q25 = np.percentile(d, [75, 25])
        print(f"{name:9s} {len(df):7d} {dur:7.2f} {seq_gaps:9d} "
              f"{np.median(d):10.1f} {d.min():7.0f} {d.max():8.0f} "
              f"{d.std():8.1f} {q75 - q25:7.1f}")

    print()
    print("per-sensor counts (min / median / max):")
    print(f"{'activity':9s} " + " ".join(f"{s:>18s}" for s in D.SENSOR_COLS))
    for name, df in data.items():
        cells = []
        for s in D.SENSOR_COLS:
            v = df[s].to_numpy(dtype=float)
            cells.append(f"{v.min():5.0f}/{np.median(v):6.0f}/{v.max():5.0f}")
        print(f"{name:9s} " + " ".join(f"{c:>18s}" for c in cells))

    print()
    print("Timing note: dt is the firmware's own ts_us delta, not an assumed")
    print("100 Hz. The notebook's load_sessions used to overwrite ts_us with")
    print("index * 10000, which would have printed dt_std = 0 here by")
    print("construction. That line was deleted in this branch.")


def c2_gain_match(data):
    """Apply the relative gain match, in conductance space."""
    rule("C2  GAIN MATCH -- applied to conductance, not to counts")

    cal = C.load_gain_match(os.path.join(REPO, "gain_match.json"))
    print(f"loaded gain_match.json: kind={cal['kind']!r} fs_counts={cal['fs_counts']}")
    print("corrections: " + "  ".join(
        f"s{i}={cal['corrections'][i]:.4f}" for i in range(6)))

    out = {}
    for name, df in data.items():
        frames = df[D.SENSOR_COLS].to_numpy(dtype=float)
        applied = np.array([
            [np.nan if v is None else v for v in C.apply_gain_match(row, cal)]
            for row in frames
        ])
        g = pd.DataFrame(applied, columns=D.SENSOR_COLS, index=df.index)
        g["ts_us"] = df["ts_us"].to_numpy()
        g["seq"] = df["seq"].to_numpy()
        out[name] = g

    print()
    print("gain-matched conductance x' = correction[i] * counts/(4095 - counts)")
    print(f"{'activity':9s} " + " ".join(f"{s:>16s}" for s in D.SENSOR_COLS)
          + f" {'None/NaN':>10s}")
    for name, g in out.items():
        cells = []
        for s in D.SENSOR_COLS:
            v = g[s].to_numpy(dtype=float)
            fin = v[np.isfinite(v)]
            cells.append(f"{np.median(fin):7.4f}/{fin.max():7.3f}" if fin.size
                         else "     --/     --")
        n_nan = int((~np.isfinite(g[D.SENSOR_COLS].to_numpy())).sum())
        print(f"{name:9s} " + " ".join(f"{c:>16s}" for c in cells)
              + f" {n_nan:10d}")
    print("(median / max per sensor; None -> NaN counts saturated or <= 0 counts)")

    print()
    print("A count of 0 has conductance 0/(4095-0) = 0, which conductance()")
    print("rejects as non-positive and apply_gain_match returns None for. So")
    print("every s4 below-threshold zero becomes a NaN here. That is the")
    print("transform's rule, not a claim the sample is missing -- see C4.")
    return out


def c3_extrapolation(data):
    """HONESTY CHECK. Fraction of real frames above the calibrated range."""
    rule("C3  EXTRAPOLATION -- fraction of frames beyond any calibrated force")

    print(f"The gain match was derived at a single ~{GAIN_MATCH_FORCE_N:.0f} N point, and the")
    print(f"highest count reached by ANY calibration trial was ~{CAL_MAX_COUNTS:.0f}. Every")
    print("frame above that is the gain match extrapolating past everything it")
    print("was ever sampled at. Reported, not fixed.")
    print()
    print(f"{'activity':9s} " + " ".join(f"{s:>8s}" for s in D.SENSOR_COLS)
          + f" {'any':>8s}")
    for name, df in data.items():
        v = df[D.SENSOR_COLS].to_numpy(dtype=float)
        over = v > CAL_MAX_COUNTS
        cells = [f"{100.0 * over[:, i].mean():7.2f}%" for i in range(6)]
        print(f"{name:9s} " + " ".join(f"{c:>8s}" for c in cells)
              + f" {100.0 * over.any(axis=1).mean():7.2f}%")
    print("(% of frames whose count exceeds the calibrated ceiling; 'any' = the")
    print(" frame has at least one such sensor)")


def c4_s4_zeros(data):
    """s4 zeros are below-threshold, not missing. Quantify the CoP consequence."""
    rule("C4  ZEROS -- below-threshold, not missing data")

    print("s4 has the highest activation threshold of the six: calibration read")
    print("s4 = 0 counts at 2.58 N while s5 read 239 at 2.49 N. A zero at s4")
    print("means 'below turn-on'. Nothing here drops, imputes, interpolates or")
    print("flags those frames.")
    print()
    print(f"{'activity':9s} " + " ".join(f"{s:>8s}" for s in D.SENSOR_COLS))
    for name, df in data.items():
        v = df[D.SENSOR_COLS].to_numpy(dtype=float)
        cells = [f"{100.0 * (v[:, i] == 0).mean():7.2f}%" for i in range(6)]
        print(f"{name:9s} " + " ".join(f"{c:>8s}" for c in cells))
    print("(% of frames reading exactly 0 counts)")

    # What CoP does on an s4-zero frame: s4 drops out of the weighted mean, so
    # the centroid is pulled toward whatever remains. s4 is the most medial
    # forefoot sensor, so losing it biases CoP LATERALLY (+x).
    rule("C4b  the lateral CoP bias on s4-zero frames")
    print("On a frame where s4 reads 0 it contributes zero weight, so CoP is a")
    print("5-sensor centroid. s4 is the most medial forefoot sensor (25.4 mm of")
    print("91), so dropping it pulls CoP toward +x, i.e. laterally.")
    print()
    print("Measured directly: for every frame where s4 == 0 and the other five")
    print("are not all zero, recompute CoP with s4 given the median non-zero s4")
    print("count from the SAME activity, and take the difference. That is the")
    print("displacement the zero is responsible for, using this capture's own")
    print("numbers rather than an assumed replacement force.")
    print()
    def bias_for(v, mask, sub):
        dxs, dys = [], []
        for row in v[mask]:
            got = cop_frame(dict(zip(D.SENSOR_COLS, row)))
            alt = row.copy()
            alt[4] = sub
            want = cop_frame(dict(zip(D.SENSOR_COLS, alt)))
            dxs.append(got[0] - want[0])
            dys.append(got[1] - want[1])
        return float(np.mean(dxs)), float(np.mean(dys))

    # A single substitute shared by all four activities, so the columns are
    # comparable. Using each activity's OWN median non-zero s4 makes stand look
    # bias-free purely because s4 barely ever leaves zero there -- the
    # counterfactual collapses to "replace a zero with a 2". Both are reported.
    all_s4 = np.concatenate([d[D.SENSOR_COLS].to_numpy(dtype=float)[:, 4]
                             for d in data.values()])
    common_sub = float(np.median(all_s4[all_s4 > 0]))
    print(f"common substitute = median non-zero s4 across all four = {common_sub:.0f} counts")
    print()
    print(f"{'activity':9s} {'n_frames':>9s} {'own_sub':>8s} {'own_dx_mm':>10s} "
          f"{'own_d_mm':>9s} {'com_dx_mm':>10s} {'com_dy_mm':>10s} {'com_d_mm':>9s}")
    for name, df in data.items():
        v = df[D.SENSOR_COLS].to_numpy(dtype=float)
        s4 = v[:, 4]
        nz = s4[s4 > 0]
        mask = (s4 == 0) & (v.sum(axis=1) > 0)
        if not mask.any():
            print(f"{name:9s} {0:9d}  no s4-zero frames")
            continue

        own_sub = float(np.median(nz)) if nz.size else float("nan")
        odx, ody = bias_for(v, mask, own_sub) if nz.size else (float("nan"),) * 2
        cdx, cdy = bias_for(v, mask, common_sub)
        print(f"{name:9s} {int(mask.sum()):9d} {own_sub:8.0f} {odx * D.INSOLE_LEN_MM:+10.2f} "
              f"{np.hypot(odx, ody) * D.INSOLE_LEN_MM:9.2f} "
              f"{cdx * D.INSOLE_LEN_MM:+10.2f} {cdy * D.INSOLE_LEN_MM:+10.2f} "
              f"{np.hypot(cdx, cdy) * D.INSOLE_LEN_MM:9.2f}")
    print("(positive dx = biased laterally, away from the medial edge;")
    print(" negative dy = biased toward the heel, since s4 sits forward at 203 mm)")
    print()
    print("Caveat on the 'own' columns: in stand, s4's median non-zero value is")
    print("itself ~2 counts, so that counterfactual asks 'what if the zero were")
    print("a 2' and unsurprisingly answers 'nothing'. It does NOT show stand is")
    print("unbiased -- it shows the bias is unmeasurable from stand alone,")
    print("because s4 never activates there to calibrate the counterfactual")
    print("against. The common-substitute columns are the comparable ones.")


def c5_stances(data):
    """Prompt 9 thresholds, unchanged, against real force."""
    rule("C5  STANCE DETECTION -- detector.py thresholds as currently set")

    print(f"T_ON={D.T_ON}  T_OFF={D.T_OFF}  MIN_DURATION={D.MIN_DURATION}  "
          f"MAX_DURATION={D.MAX_DURATION}  GAP_MERGE={D.GAP_MERGE}")
    print("T_ON, T_OFF, MIN_DURATION and GAP_MERGE were chosen by sweeping ~2016")
    print("combinations against simulated streams whose constants were co-evolved")
    print("with them. MAX_DURATION was re-set from this script's own diagnostic")
    print("below (120 -> 200). Nothing is retuned by this script.")
    print()

    out = {}
    print(f"{'activity':9s} {'raw':>5s} {'merged':>7s} {'dur_med_fr':>11s} "
          f"{'dur_min':>8s} {'dur_max':>8s} {'dur_med_s':>10s} {'above_T_ON':>11s}")
    for name, df in data.items():
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        raw = D.find_stances(total)
        merged = D.merge_close(raw)
        out[name] = merged
        dt = frame_dt(df["ts_us"])
        frac_on = 100.0 * (total >= D.T_ON).mean()
        if merged:
            durs = np.array([e - s + 1 for s, e in merged], dtype=float)
            print(f"{name:9s} {len(raw):5d} {len(merged):7d} {np.median(durs):11.1f} "
                  f"{durs.min():8.0f} {durs.max():8.0f} "
                  f"{np.median(durs) * dt:10.3f} {frac_on:10.1f}%")
        else:
            print(f"{name:9s} {len(raw):5d} {len(merged):7d} {'--':>11s} "
                  f"{'--':>8s} {'--':>8s} {'--':>10s} {frac_on:10.1f}%")

    print()
    print("total-force distribution, for reading the counts above against the")
    print(f"thresholds (T_ON={D.T_ON}, T_OFF={D.T_OFF}):")
    print(f"{'activity':9s} {'min':>7s} {'p05':>7s} {'median':>7s} {'p95':>7s} {'max':>7s}")
    for name, df in data.items():
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        p5, p95 = np.percentile(total, [5, 95])
        print(f"{name:9s} {total.min():7.0f} {p5:7.0f} {np.median(total):7.0f} "
              f"{p95:7.0f} {total.max():7.0f}")
    print()
    print("stand is the null check: quiet standing should yield ~zero stances.")

    # Why the counts come out where they do. find_stances discards any run that
    # reaches MAX_DURATION, so a real contact longer than 120 frames is
    # annihilated rather than clipped. Count the runs and their natural lengths
    # with the duration limits removed, to separate "the detector saw nothing"
    # from "the detector saw it and threw it away".
    print()
    print("diagnostic -- threshold crossings with the duration limits removed")
    print("(same T_ON/T_OFF hysteresis, no MIN_DURATION, no MAX_DURATION):")
    print(f"{'activity':9s} {'runs':>5s} {'len_med':>8s} {'len_min':>8s} "
          f"{'len_max':>8s} {'over_MAX':>9s} {'under_MIN':>10s} {'kept':>5s}")
    for name, df in data.items():
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        runs = _runs_unbounded(total, D.T_ON, D.T_OFF)
        if not runs:
            print(f"{name:9s} {0:5d}  no threshold crossings at all")
            continue
        L = np.array([e - s + 1 for s, e in runs], dtype=float)
        n_over = int((L > D.MAX_DURATION).sum())
        n_under = int((L < D.MIN_DURATION).sum())
        print(f"{name:9s} {len(runs):5d} {np.median(L):8.1f} {L.min():8.0f} "
              f"{L.max():8.0f} {n_over:9d} {n_under:10d} {len(out[name]):5d}")
    print("Runs over MAX_DURATION are DISCARDED outright by the max_duration")
    print("break, not clipped -- see the comment in find_stances. That is")
    print("annihilation: a true stance that becomes zero detections.")

    # Annihilation at a DURATION threshold is not random dropout: it removes
    # exactly the long contacts and keeps the short ones. So the surviving
    # stances are a biased subsample, and every mean computed from them is
    # biased with it. Two ways to see that, both from this capture:
    print()
    print("selection effect -- what survives is the SHORT end of the")
    print("distribution, and it is not spread evenly through the capture:")
    print(f"{'activity':9s} {'kept':>5s} {'kept_med':>9s} {'nat_med':>8s} "
          f"{'kept_max':>9s} {'nat_max':>8s} {'first_s':>8s} {'last_s':>7s} "
          f"{'max_gap_s':>10s}")
    for name, df in data.items():
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        runs = _runs_unbounded(total, D.T_ON, D.T_OFF)
        kept = out[name]
        if not kept or not runs:
            print(f"{name:9s} {len(kept):5d}  nothing kept")
            continue
        ts = df["ts_us"].to_numpy(dtype=float)
        t0 = ts[0]
        nat = np.array([e - s + 1 for s, e in runs], dtype=float)
        kp = np.array([e - s + 1 for s, e in kept], dtype=float)
        starts = np.array([(ts[s] - t0) / 1e6 for s, _ in kept])
        ends = np.array([(ts[e] - t0) / 1e6 for _, e in kept])
        gaps = starts[1:] - ends[:-1] if len(kept) > 1 else np.array([0.0])
        print(f"{name:9s} {len(kept):5d} {np.median(kp):9.1f} {np.median(nat):8.1f} "
              f"{kp.max():9.0f} {nat.max():8.0f} {starts[0]:8.2f} "
              f"{ends[-1]:7.2f} {gaps.max():10.2f}")
    print("kept_max can never exceed MAX_DURATION by construction; nat_max is")
    print("what the foot actually did. A large max_gap_s means whole stretches")
    print("of the capture contributed no stances at all, so the surviving")
    print("stances are not a uniform sample of the trial.")
    return out


def c6_features(data, stances):
    """Features per stance, plus the cost of the uniform-dt approximation."""
    rule("C6  FEATURES -- and the uniform-dt approximation's drift")

    print("features.frame_dt collapses sampling to ts_us.diff().median() and")
    print("stance_features multiplies that one dt across a whole stance. Below:")
    print("how far that lands from the real elapsed time on the LONGEST stance")
    print("of each activity, measured against the actual timestamps.")
    print()
    print(f"{'activity':9s} {'n_st':>5s} {'longest_fr':>11s} {'approx_s':>9s} "
          f"{'true_s':>8s} {'drift_us':>9s} {'drift_%':>10s} {'worst_us':>9s}")

    feats = {}
    for name, df in data.items():
        st = stances[name]
        if not st:
            print(f"{name:9s} {0:5d}  no stances -- nothing to extract")
            feats[name] = pd.DataFrame([])
            continue

        feats[name] = extract_features(df, st, name)
        dt = frame_dt(df["ts_us"])
        ts = df["ts_us"].to_numpy(dtype=float)

        # Worst drift over ANY stance, and the drift on the longest one.
        drifts = []
        for s, e in st:
            approx = (e - s) * dt
            true = (ts[e] - ts[s]) / 1e6
            drifts.append(approx - true)
        longest = int(np.argmax([e - s for s, e in st]))
        s, e = st[longest]
        approx = (e - s) * dt
        true = (ts[e] - ts[s]) / 1e6
        worst = max(drifts, key=abs)
        print(f"{name:9s} {len(st):5d} {e - s + 1:11d} {approx:9.4f} {true:8.4f} "
              f"{1e6 * (approx - true):+9.1f} "
              f"{100.0 * (approx - true) / true if true else float('nan'):+9.6f}% "
              f"{1e6 * worst:+9.1f}")

    print()
    for name, f in feats.items():
        if f.empty:
            continue
        print(f"--- {name}: {len(f)} stances ---")
        cols = ["peak_counts", "time_to_peak_s", "contact_time_s",
                "loading_rate_cps", "impulse_counts_s",
                "cop_path_len", "cop_displacement"]
        print(f.reindex(columns=cols).describe().loc[
            ["mean", "std", "min", "50%", "max"]].to_string(
                float_format=lambda v: f"{v:10.4f}"))
        print()
    return feats


def main():
    data = {}
    for name, fname in ACTIVITIES:
        path = os.path.join(REAL, fname)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        data[name] = load(fname)

    print("Prompt 13 Phase C -- real captures through the pipeline")
    print("source: data/real/  (read-only; nothing is written back there)")
    print("set:    _02 only. _01 is excluded -- intermittent s0, kept as evidence.")
    for name, fname in ACTIVITIES:
        print(f"  {name:9s} <- data/real/{fname}")

    c1_ingest(data)
    c2_gain_match(data)
    c3_extrapolation(data)
    c4_s4_zeros(data)
    stances = c5_stances(data)
    c6_features(data, stances)

    rule("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
