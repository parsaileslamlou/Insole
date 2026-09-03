"""Phase D of Prompt 13: the simulator against the four real _02 captures.

Run from the repo root, after scripts/analyze_real.py:

    python scripts/sim_vs_real.py

Writes the comparison plots into figures/sim_vs_real/ and prints every table
quoted in docs/sim_vs_real.md sections D1-D5. Reads data/real/ and never
writes to it.

D1 (the bake-off regeneration) lives in bakeoff.py, not here -- delete
data/sim/features_sessions.csv and re-run `python scripts/bakeoff.py` to rebuild it under the
current geometry. This script consumes the same regenerated frame for D2.

Standing caveat that applies to every number below: one 60 s trial per class
is not a training set and is not treated as one. D2 exists to prove the
plumbing carries real frames end to end, not to report a classifier result.
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from insole import detector as D                                              # noqa: E402
from insole.features import (                                            # noqa: E402
    cop_frame, extract_features, frame_dt,
)
from insole.discriminant import fit_lda, fit_qda, predict                # noqa: E402
from insole.representations import LETTER, SHIPPED, features_under       # noqa: E402

from insole.paths import DATA_REAL, DATA_SIM, FIGURES, REPO as _REPO   # noqa: E402

REPO = str(_REPO)
REAL = str(DATA_REAL)
FIGDIR = os.path.join(FIGURES, "sim_vs_real")

# real activity -> (real csv, sim csv). stand has no simulator counterpart
# beyond sim_stand, which gait_gen emits as an unbroken load with no stances.
PAIRS = [
    ("stand",   "stand_02.csv",   "sim_stand.csv"),
    ("walk",    "walk02.csv",     "sim_walk.csv"),
    ("fast",    "fast02.csv",     "sim_fast.csv"),
    ("shuffle", "shuffle02.csv",  "sim_shuffle.csv"),
]

# The three classes the sim-trained model knows. stand is not among them.
MODEL_CLASSES = ["fast", "shuffle", "walk"]
FEATURES = ["cop_path_len", "cop_displacement"]

# Anatomical firing order a right foot should show, heel to toe.
EXPECTED_ORDER = [("s0", "s1"), ("s2",), ("s3", "s4"), ("s5",)]

RULE = "=" * 78


def head(t):
    print("\n" + RULE + "\n" + t + "\n" + RULE)


def load_real(fname):
    return pd.read_csv(os.path.join(REAL, fname))


def load_sim(fname):
    path = os.path.join(DATA_SIM, fname)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run: python -m insole.read_serial "
                         f"{fname[:-4]}.txt {fname}")
    return pd.read_csv(path)


def stances_of(df):
    total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
    return D.merge_close(D.find_stances(total))


# ---------------------------------------------------------------------------
# D2 -- plumbing check
# ---------------------------------------------------------------------------
def d2_plumbing(real_feats):
    head("D2  PLUMBING CHECK -- sim-trained model applied to real stances")

    print("THIS IS NOT A CLASSIFIER RESULT. One 60 s trial per class cannot be")
    print("trained on and is not being trained on. The model is fitted on the")
    print("12 SIMULATED sessions and asked to label real stances purely to")
    print("prove that real frames survive ingest -> detection -> features ->")
    print("model without a shape error or a silent NaN. Expect garbage; the")
    print("only failure that matters here is a crash or an empty frame.")
    print()

    frame_path = os.path.join(DATA_SIM, "features_sessions.csv")
    if not os.path.exists(frame_path):
        print("features_sessions.csv missing -- run `python scripts/bakeoff.py` first "
              "to rebuild it under the current geometry. D2 skipped.")
        return
    sim = pd.read_csv(frame_path)
    print(f"training frame: {frame_path}  ({len(sim)} sim stances, "
          f"{sim['session'].nunique()} sessions)")
    print(f"feature representation on both sides: {LETTER[SHIPPED]} ({SHIPPED}), "
          f"insole.representations.SHIPPED")

    Xtr = sim[FEATURES].to_numpy(float)
    ytr = sim["label"].to_numpy()

    real = pd.concat([f for f in real_feats.values() if not f.empty],
                     ignore_index=True)
    real = real[real["label"].isin(MODEL_CLASSES)]
    if real.empty:
        print("no real stances in a class the model knows -- D2 skipped.")
        return
    Xte = real[FEATURES].to_numpy(float)
    yte = real["label"].to_numpy()

    finite = np.isfinite(Xte).all(axis=1)
    if not finite.all():
        print(f"NOTE: {int((~finite).sum())} real stance(s) had a non-finite "
              f"CoP feature and are excluded from the matrix.")
        Xte, yte = Xte[finite], yte[finite]

    print(f"real stances scored: {len(yte)}  "
          f"({', '.join(f'{c}={int((yte == c).sum())}' for c in MODEL_CLASSES)})")
    print("stand is excluded: it is not one of the model's three classes, and")
    print("the detector found no stances in it anyway.")

    for tag, model in [("LDA", fit_lda(Xtr, ytr)), ("QDA", fit_qda(Xtr, ytr))]:
        pred = predict(model, Xte)
        acc = float((pred == yte).mean())
        print(f"\nsim-trained {tag} on real stances -- accuracy {acc:.4f} "
              f"({int((pred == yte).sum())}/{len(yte)})")
        print("confusion (rows = true real activity, cols = predicted):")
        print(f"  {'':10s}" + "".join(f"{c:>9s}" for c in MODEL_CLASSES))
        for c in MODEL_CLASSES:
            row = [int(((yte == c) & (pred == p)).sum()) for p in MODEL_CLASSES]
            print(f"  {c:10s}" + "".join(f"{v:9d}" for v in row))
    print("\nLabelled a PLUMBING CHECK. Do not quote these accuracies as a")
    print("model result anywhere.")


# ---------------------------------------------------------------------------
# D3 -- comparison plots
# ---------------------------------------------------------------------------
def d3_plots(pairs_data):
    head("D3  COMPARISON PLOTS -> figures/sim_vs_real/")
    os.makedirs(FIGDIR, exist_ok=True)

    written = []
    for name, real, sim, r_st, s_st in pairs_data:
        fig, axes = plt.subplots(6, 1, sharex=True, sharey=True,
                                 figsize=(11, 9))
        t_real = (real["ts_us"].to_numpy(float) - real["ts_us"].iloc[0]) / 1e6
        t_sim = (sim["ts_us"].to_numpy(float) - sim["ts_us"].iloc[0]) / 1e6

        for i, col in enumerate(D.SENSOR_COLS):
            ax = axes[i]
            # Shade real detections first so the traces draw over them.
            for s, e in r_st:
                ax.axvspan(t_real[s], t_real[e], color="#4C9BE8", alpha=0.13,
                           lw=0)
            for s, e in s_st:
                ax.axvspan(t_sim[s], t_sim[e], color="#E8A34C", alpha=0.10,
                           lw=0)
            ax.plot(t_sim, sim[col], lw=0.7, color="#D2691E", alpha=0.75,
                    label="sim (gait_gen)" if i == 0 else None)
            ax.plot(t_real, real[col], lw=0.8, color="#1F4E79",
                    label="real (_02)" if i == 0 else None)
            ax.set_ylabel(col, rotation=0, labelpad=18, va="center")
            ax.grid(alpha=0.3)

        axes[0].set_title(
            f"{name}: real data/real vs simulated gait_gen  "
            f"(real stances shaded blue n={len(r_st)}, "
            f"sim shaded orange n={len(s_st)})")
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        axes[-1].set_xlabel("time within capture (s)")
        axes[-1].set_ylim(0, 4095)
        axes[-1].set_xlim(0, 12)      # first 12 s; 60 s of 100 Hz is unreadable
        fig.tight_layout()

        out = os.path.join(FIGDIR, f"{name}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        written.append(out)
        print(f"  wrote {os.path.relpath(out, REPO)}  "
              f"(real stances {len(r_st)}, sim stances {len(s_st)})")

    print("\nx axis is clipped to the first 12 s of each 60 s capture; at")
    print("100 Hz the full trace is an unreadable smear. Stance shading is")
    print("the SAME unretuned detector on both streams.")
    return written


# ---------------------------------------------------------------------------
# D4 -- quantitative diff table
# ---------------------------------------------------------------------------
def per_stance_metrics(df, stances, label):
    """Feature rows plus the two extras D4 needs that features.py lacks."""
    if not stances:
        return pd.DataFrame([])
    feats = extract_features(df, stances, label)

    dt = frame_dt(df["ts_us"])
    ml_range, peaks = [], []
    for s, e in stances:
        seg = df.iloc[s:e]
        xs = [cop_frame(r)[0] for _, r in seg.iterrows()]
        xs = [v for v in xs if not np.isnan(v)]
        ml_range.append((max(xs) - min(xs)) if len(xs) >= 2 else np.nan)
        peaks.append([float(seg[c].max()) for c in D.SENSOR_COLS])

    feats["cop_ml_range"] = ml_range
    for i, c in enumerate(D.SENSOR_COLS):
        feats[f"peak_{c}"] = [p[i] for p in peaks]
    feats["_dt"] = dt
    return feats


def d4_table(pairs_data):
    head("D4  QUANTITATIVE DIFF -- real minus sim, per activity")

    metrics = (["contact_time_s"]
               + [f"peak_{c}" for c in D.SENSOR_COLS]
               + ["time_to_peak_s", "loading_rate_cps",
                  "cop_path_len", "cop_displacement", "cop_ml_range"])

    for name, real, sim, r_st, s_st in pairs_data:
        rf = per_stance_metrics(real, r_st, name)
        sf = per_stance_metrics(sim, s_st, name)
        print(f"\n--- {name}: real n={len(r_st)} stances, sim n={len(s_st)} ---")
        if rf.empty or sf.empty:
            missing = "real" if rf.empty else "sim"
            print(f"    no {missing} stances -- no comparison possible for "
                  f"this activity.")
            continue

        print(f"  {'metric':20s} {'real_mean':>12s} {'sim_mean':>12s} "
              f"{'delta':>12s} {'delta_%':>9s} {'real_sd':>10s} {'sim_sd':>10s}")
        for m in metrics:
            r, s = rf[m].to_numpy(float), sf[m].to_numpy(float)
            r, s = r[np.isfinite(r)], s[np.isfinite(s)]
            if r.size == 0 or s.size == 0:
                print(f"  {m:20s} {'--':>12s} {'--':>12s} {'--':>12s}")
                continue
            rm, sm = r.mean(), s.mean()
            pct = 100.0 * (rm - sm) / sm if sm else float("nan")
            print(f"  {m:20s} {rm:12.4f} {sm:12.4f} {rm - sm:+12.4f} "
                  f"{pct:+8.1f}% {r.std(ddof=1) if r.size > 1 else 0:10.4f} "
                  f"{s.std(ddof=1) if s.size > 1 else 0:10.4f}")

    print("\nSigned delta is real minus sim. cop_ml_range is the medial-lateral")
    print("spread of the CoP path within a stance, in normalised units")
    print(f"(multiply by {D.INSOLE_LEN_MM:.0f} for mm).")

    # D4b. The within-stance range above says how far CoP travels sideways
    # during one contact. It says nothing about whether successive contacts
    # were placed differently -- which is the thing a figure-8 path should
    # produce and a straight symmetric simulator should not. That is a
    # BETWEEN-stance spread, so it needs its own statistic.
    head("D4b  STANCE-TO-STANCE CoP PLACEMENT SPREAD (the figure-8 signature)")
    print("Per stance, the mean medial-lateral CoP position; then the spread of")
    print("that quantity ACROSS stances. gait_gen models straight, symmetric,")
    print("stride-identical gait, so its stances should land on top of each")
    print("other. A foot continuously turning around a figure-8 should not.")
    print()
    print(f"{'activity':9s} {'n_real':>7s} {'n_sim':>6s} {'real_sd':>9s} "
          f"{'sim_sd':>9s} {'ratio':>7s} {'real_sd_mm':>11s} {'sim_sd_mm':>10s} "
          f"{'real_range_mm':>14s}")
    for name, real, sim, r_st, s_st in pairs_data:
        if not r_st or not s_st:
            continue

        def per_stance_mean_x(df, stances):
            out = []
            for a, b in stances:
                xs = [cop_frame(r)[0] for _, r in df.iloc[a:b].iterrows()]
                xs = [v for v in xs if not np.isnan(v)]
                if xs:
                    out.append(float(np.mean(xs)))
            return np.array(out)

        rx = per_stance_mean_x(real, r_st)
        sx = per_stance_mean_x(sim, s_st)
        if rx.size < 2 or sx.size < 2:
            print(f"{name:9s} too few stances to take a spread")
            continue
        r_sd, s_sd = float(rx.std(ddof=1)), float(sx.std(ddof=1))
        print(f"{name:9s} {rx.size:7d} {sx.size:6d} {r_sd:9.5f} {s_sd:9.5f} "
              f"{(r_sd / s_sd if s_sd else float('inf')):7.1f} "
              f"{r_sd * D.INSOLE_LEN_MM:11.2f} {s_sd * D.INSOLE_LEN_MM:10.3f} "
              f"{(rx.max() - rx.min()) * D.INSOLE_LEN_MM:14.2f}")
    print("ratio = real sd / sim sd. sd_mm multiplies by the insole length.")
    print("real_range_mm is peak-to-peak across all of that activity's stances.")
    print()
    print("Read this against the +/-15 mm coordinate uncertainty: a spread")
    print("smaller than that is not distinguishable from the measurement error")
    print("on the sensor positions themselves.")


# ---------------------------------------------------------------------------
# D5 -- sensor order
# ---------------------------------------------------------------------------
def d5_order(real, stances, activity="walk"):
    head(f"D5  SENSOR ORDER -- mean activation time within a stance ({activity}_02)")

    if not stances:
        print(f"no stances detected in {activity}_02 -- order cannot be checked.")
        return

    print("Two independent timings per sensor, both relative to stance start:")
    print("  t_onset = first frame the sensor exceeds 20% of its own peak")
    print("            within that stance")
    print("  t_peak  = frame of the sensor's maximum within that stance")
    print("A sensor that never rises in a stance contributes to neither.")
    print()

    dt = frame_dt(real["ts_us"])
    onsets = {c: [] for c in D.SENSOR_COLS}
    peaks = {c: [] for c in D.SENSOR_COLS}
    n_present = {c: 0 for c in D.SENSOR_COLS}

    for s, e in stances:
        seg = real.iloc[s:e]
        for c in D.SENSOR_COLS:
            v = seg[c].to_numpy(float)
            if v.max() <= 0:
                continue
            n_present[c] += 1
            thr = 0.20 * v.max()
            above = np.flatnonzero(v >= thr)
            if above.size:
                onsets[c].append(above[0] * dt)
            peaks[c].append(int(np.argmax(v)) * dt)

    print(f"{'sensor':7s} {'in_n':>5s} {'t_onset_s':>10s} {'sd':>7s} "
          f"{'t_peak_s':>9s} {'sd':>7s}  anatomy")
    anat = {"s0": "heel (medial)", "s1": "heel (lateral)",
            "s2": "lateral midfoot", "s3": "5th met head",
            "s4": "1st met head", "s5": "hallux"}
    rows = []
    for c in D.SENSOR_COLS:
        on = np.array(onsets[c]) if onsets[c] else np.array([np.nan])
        pk = np.array(peaks[c]) if peaks[c] else np.array([np.nan])
        rows.append((c, float(np.nanmean(on)), float(np.nanmean(pk))))
        print(f"{c:7s} {n_present[c]:5d} {np.nanmean(on):10.4f} "
              f"{np.nanstd(on):7.4f} {np.nanmean(pk):9.4f} "
              f"{np.nanstd(pk):7.4f}  {anat[c]}")

    for tag, key in (("onset", 1), ("peak", 2)):
        order = [r[0] for r in sorted(rows, key=lambda r: r[key])]
        print(f"\nobserved {tag} order (earliest first): {' -> '.join(order)}")
        expected_flat = [s for grp in EXPECTED_ORDER for s in grp]
        print(f"anatomically expected                : "
              f"{' -> '.join(expected_flat)}   (groups: "
              f"{' -> '.join('/'.join(g) for g in EXPECTED_ORDER)})")

        rank = {s: i for i, s in enumerate(order)}
        violations = []
        for gi in range(len(EXPECTED_ORDER) - 1):
            for a in EXPECTED_ORDER[gi]:
                for b in EXPECTED_ORDER[gi + 1]:
                    if rank[a] > rank[b]:
                        violations.append(f"{a} after {b}")
        if violations:
            print(f"VIOLATIONS ({tag}): " + "; ".join(violations))
        else:
            print(f"no group-order violations on {tag} timing.")

    print("\nIf the order is violated, a channel pair may be swapped in firmware.")
    print("REPORTED ONLY -- nothing in this repo is changed on the strength of")
    print("it, and firmware/ is out of scope for this branch.")


def main():
    pairs_data, real_feats = [], {}
    for name, rfile, sfile in PAIRS:
        real, sim = load_real(rfile), load_sim(sfile)
        r_st, s_st = stances_of(real), stances_of(sim)
        pairs_data.append((name, real, sim, r_st, s_st))
        # D2 feeds the sim-trained model, so its features are computed under the
        # shipped representation the model was fitted on; D4's tables stay on
        # raw counts (per_stance_metrics), where peak counts mean counts.
        real_feats[name] = (features_under(real, r_st, name, SHIPPED) if r_st
                            else pd.DataFrame([]))

    print("Prompt 13 Phase D -- simulator vs the real _02 captures")
    for name, rfile, sfile in PAIRS:
        print(f"  {name:9s} real=data/real/{rfile:16s} sim={sfile}")

    d2_plumbing(real_feats)
    d3_plots(pairs_data)
    d4_table(pairs_data)

    # pairs_data rows are (name, real_df, sim_df, real_stances, sim_stances).
    # D5 is a statement about the REAL capture, so it takes index 3, not 4.
    name, real_df, _sim_df, real_st, _sim_st = next(
        p for p in pairs_data if p[0] == "walk")
    d5_order(real_df, real_st, name)

    print("\n" + RULE + "\nEND\n" + RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
