"""Per-channel noise statistics for an insole capture CSV.

Usage:
    python scripts/noise_stats.py [path_to_capture.csv]

Expects columns: seq,ts_us,s0,s1,s2,s3,s4,s5 (raw 12-bit ADC counts).
Reports per channel: mean, std, min, max, peak-to-peak, and the count of
samples more than 5 std from that channel's mean. Also prints the frame
count and the capture duration derived from the ts_us span.
"""
import sys

import numpy as np
import pandas as pd

CHANNELS = ["s0", "s1", "s2", "s3", "s4", "s5"]
DEFAULT_PATH = "data/noise_floor_30s.csv"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    df = pd.read_csv(path)

    n_frames = len(df)
    duration_s = (df["ts_us"].max() - df["ts_us"].min()) / 1_000_000.0

    print(f"file:     {path}")
    print(f"frames:   {n_frames}")
    print(f"duration: {duration_s:.3f} s  (from ts_us span)")
    print()

    hdr = (f"{'ch':<4}{'mean':>11}{'std':>10}{'min':>8}{'max':>8}"
           f"{'p2p':>8}{'spikes(>5std)':>15}")
    print(hdr)
    print("-" * len(hdr))

    for ch in CHANNELS:
        col = df[ch].to_numpy()
        mean = col.mean()
        std = col.std(ddof=1)                       # sample standard deviation
        cmin = int(col.min())
        cmax = int(col.max())
        p2p = cmax - cmin
        spikes = int(np.sum(np.abs(col - mean) > 5.0 * std)) if std > 0 else 0
        print(f"{ch:<4}{mean:>11.3f}{std:>10.3f}{cmin:>8d}{cmax:>8d}"
              f"{p2p:>8d}{spikes:>15d}")


if __name__ == "__main__":
    main()
