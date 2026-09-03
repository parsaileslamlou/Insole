"""Stance and centre-of-pressure feature extraction.

Lifted verbatim out of insole.ipynb so that scripts and tests can import the
same definitions the notebook uses. A test cannot import a notebook cell, and a
second copy would drift until the test passed against a stale extractor -- the
same reason find_stances moved into detector.py.

The function bodies below are byte-identical to the notebook cells they came
from, mixed indentation included. Do not tidy them here; edit the notebook and
re-lift, or the two copies diverge again.

detector.py stays the single source of truth for the stance detector and the
sensor geometry -- they are imported, never redefined.
"""

import numpy as np
import pandas as pd

from insole.detector import (
    SENSOR_COLS, SENSOR_COORDS,
    find_stances, merge_close,
)

__all__ = [
    "frame_dt", "stance_features", "cop_frame", "cop_trajectory",
    "cop_features", "extract_features",
    "SENSOR_COLS", "SENSOR_COORDS", "find_stances", "merge_close",
]


def frame_dt(ts_us):
    return ts_us.diff().median() / 1_000_000.0


def stance_features(total, dt, start, end):
    seg = np.asarray(total)[start:end]

    peak_idx = int(np.argmax(seg))
    peak_counts = float(seg[peak_idx])
    time_to_peak_s = peak_idx * dt
    contact_time_s = (end - start) * dt
    impulse_counts_s = float(seg.sum()) * dt

    if time_to_peak_s > 0:
        loading_rate_cps = (peak_counts - float(seg[0])) / time_to_peak_s
    else:
        loading_rate_cps = np.nan

    return {
        "peak_counts": peak_counts,
        "time_to_peak_s": time_to_peak_s,
        "contact_time_s": contact_time_s,
        "loading_rate_cps": loading_rate_cps,
        "impulse_counts_s": impulse_counts_s,
    }


def cop_frame(row):

  total_weight = sum(row[name] for name in SENSOR_COLS)
  if total_weight == 0:
    return (np.nan,np.nan)

  cop_x = 0.0
  cop_y = 0.0

  for name in SENSOR_COLS:
      x, y = SENSOR_COORDS[name]
      cop_x += row[name]*x
      cop_y += row[name]*y

  cop_x /= total_weight
  cop_y /= total_weight

  return (cop_x, cop_y)


def cop_trajectory(df, start, end):
  ls = []
  for _, row in df.iloc[start:end].iterrows():
    ls.append(cop_frame(row))
  return np.array(ls)


def cop_features(traj):
    NAN_RESULT = {"cop_path_len": np.nan, "cop_displacement": np.nan}

    valid = [(x, y) for x, y in traj if not (np.isnan(x) or np.isnan(y))]
    if len(valid) < 2:
        return NAN_RESULT

    path_len = 0.0
    for i in range(len(valid) - 1):
        x0, y0 = valid[i]
        x1, y1 = valid[i + 1]
        path_len += np.hypot(x1 - x0, y1 - y0)

    x_first, y_first = valid[0]
    x_last, y_last = valid[-1]
    displacement = np.hypot(x_last - x_first, y_last - y_first)

    return {"cop_path_len": path_len, "cop_displacement": displacement}


def extract_features(df, stances, label):

  if len(stances) == 0:
    return pd.DataFrame([])

  dt = frame_dt(df["ts_us"])
  total = df[SENSOR_COLS].sum(axis=1)
  rows = []

  for start, end in stances:
    stance_ft = stance_features(total, dt, start, end)
    cop_traj = cop_trajectory(df, start, end)
    cop_ft = cop_features(cop_traj)

    row = {**stance_ft, **cop_ft, "start": start, "end": end, "label": label}
    rows.append(row)

  return pd.DataFrame(rows)
