"""The MAX_DURATION sweep behind the 120 -> 200 retune.

    python sweep_max_duration.py

Prints, for each candidate ceiling, how many stances find_stances + merge_close
keeps on the four real _02 captures and on the five sim fixtures, plus the
natural (T_ON/T_OFF only) run-length distribution of each real capture. Reads
data/real/ and the sim CSVs; writes nothing.

The sim columns exist to show what the sim tests CAN'T see: no simulated stance
exceeds 60 frames, so every candidate from 120 to 1000 leaves the sim counts
untouched. The change rests on the real data alone.
"""

import os
import sys

import numpy as np
import pandas as pd

import detector as D
from gait_gen import SHUFFLE_CYCLE_S, true_stances

REPO = os.path.dirname(os.path.abspath(__file__))
REAL = {"stand": "stand_02.csv", "walk": "walk02.csv",
        "fast": "fast02.csv", "shuffle": "shuffle02.csv"}
SIM = [("sim_walk", "walk", 1.0), ("sim_fast", "walk", 0.6),
       ("sim_shuffle", "shuffle", SHUFFLE_CYCLE_S),
       ("sim_dropout", "walk", 1.0), ("sim_stand", "standing", 1.0)]
CANDIDATES = [120, 150, 165, 180, 200, 250, 300, 500, 1000]


def total_of(path):
    df = pd.read_csv(path)
    return df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)


def runs_unbounded(total):
    runs, in_run, start = [], False, 0
    for i, x in enumerate(total):
        if in_run:
            if x < D.T_OFF:
                runs.append((start, i - 1))
                in_run = False
        elif x >= D.T_ON:
            start, in_run = i, True
    if in_run:
        runs.append((start, len(total) - 1))
    return runs


def main():
    print(f"T_ON={D.T_ON} T_OFF={D.T_OFF} MIN_DURATION={D.MIN_DURATION} "
          f"GAP_MERGE={D.GAP_MERGE}; detector.MAX_DURATION is currently {D.MAX_DURATION}")
    print()
    print("natural run lengths (T_ON/T_OFF hysteresis only), real _02 captures:")
    print(f"{'activity':9s} {'runs':>5s} {'min':>5s} {'p50':>7s} {'p90':>7s} {'max':>5s}")
    real = {}
    for name, f in REAL.items():
        total = total_of(os.path.join(REPO, "data", "real", f))
        real[name] = total
        L = np.array([e - s + 1 for s, e in runs_unbounded(total)])
        print(f"{name:9s} {len(L):5d} {L.min():5d} {np.median(L):7.1f} "
              f"{np.percentile(L, 90):7.1f} {L.max():5d}")

    sims = {}
    for stem, mode, cyc in SIM:
        path = os.path.join(REPO, stem + ".csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run: python read_serial.py {stem}.txt {stem}.csv")
        sims[stem] = (total_of(path), len(true_stances(60, mode=mode, cycle_s=cyc)))
    print()
    print("longest raw stance in each sim fixture (no ceiling):")
    for stem, (total, _) in sims.items():
        st = D.find_stances(total, max_duration=10 ** 9)
        print(f"  {stem:12s} {max(e - s + 1 for s, e in st) if st else 0}")

    print()
    print("stances kept by find_stances + merge_close per candidate MAX_DURATION:")
    print(f"{'max':>5s} " + " ".join(f"{n:>8s}" for n in REAL) + " | "
          + " ".join(f"{s[0]:>12s}" for s in SIM) + "  sim_ok")
    for m in CANDIDATES:
        row = [len(D.merge_close(D.find_stances(t, max_duration=m))) for t in real.values()]
        sim_cells, ok = [], True
        for stem, (total, want) in sims.items():
            got = len(D.merge_close(D.find_stances(total, max_duration=m)))
            sim_cells.append(f"{got}/{want}")
            ok &= got == want
        print(f"{m:5d} " + " ".join(f"{v:8d}" for v in row) + " | "
              + " ".join(f"{c:>12s}" for c in sim_cells) + f"  {ok}")
    print()
    print("sim_ok = every fixture still matches gait_gen.true_stances. The real")
    print("columns saturate at 165: every walk and shuffle contact is kept from")
    print("there up, and standing (one 6000-frame run) is rejected until ~6000.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
