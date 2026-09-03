"""Per-frame input representations for the feature extractor.

The stance detector always runs on RAW COUNTS (T_ON / T_OFF are in counts).
What the features see is a separate choice, made once here so the batch path
(scripts/train_real.py) and the streaming path (insole/infer_live.py) cannot
disagree:

    "raw"           A  the six ADC counts as logged
    "conductance"   B  x = counts / (FS_COUNTS - counts) per channel
    "gain_matched"  C  x * g_i, g from models/gain_match.json (real board) or
                       identity (simulator, which has no per-channel gain to
                       correct)

Force is linear in conductance, not in counts (insole/calibration.py), so B
and C make the centre of pressure a force-proportional centroid and A does
not. Whether that helps a classifier is an empirical question that
scripts/train_real.py answers on the real captures; this module only makes
the three representations computable and identical everywhere.

A count of 0 maps to x = 0: an open channel has zero conductance, so a
below-threshold s4 contributes zero weight under every representation, which
is the same treatment it already gets under raw counts. Nothing is imputed.
A count at or above FS_COUNTS (never observed: the real captures peak at
2011, the simulator clips at 4095) is mapped to the conductance one count
below full scale rather than to infinity, and counted by the caller if it
cares.
"""

import numpy as np

from insole.calibration import FS_COUNTS
from insole.detector import SENSOR_COLS
from insole.features import extract_features

REPRESENTATIONS = ("raw", "conductance", "gain_matched")
LETTER = {"raw": "A", "conductance": "B", "gain_matched": "C"}
N_SENSORS = len(SENSOR_COLS)

# The representation the classifier consumes, on every source: the streaming
# path (infer_live.features_for), the sim bake-off (scripts/bakeoff.py), the
# persisted models (scripts/fit_model.py) and the batch side of the
# streaming-equals-batch test all go through this one name.
#
# Chosen by scripts/train_real.py's rule, fixed before any result was seen:
# CoP-only features, the representation and model with the best contiguous-
# block CV accuracy on the real captures. That was B, conductance, and the
# physics agrees (force is linear in conductance). The gain match, C, scored
# no better than B there while extrapolating on most loaded frames, so it is
# not part of what the classifier sees; it is still applied per frame for the
# extrapolation counter. The simulator has no per-channel gain to correct, so
# under B every source is treated identically and no gains are attached.
# docs/real_results.md carries the numbers behind this choice.
SHIPPED = "conductance"


def identity_gains():
    """The simulator's gains: nothing to correct."""
    return {i: 1.0 for i in range(N_SENSORS)}


def gains_from_doc(doc):
    """Per-channel corrections from a load_gain_match() document, as a dict."""
    return {int(k): float(v) for k, v in doc["corrections"].items()}


def conductance_array(counts, fs=FS_COUNTS):
    """counts (any shape) -> x = c / (fs - c), with x(0) = 0 and c >= fs capped."""
    c = np.asarray(counts, dtype=float)
    c = np.minimum(c, fs - 1.0)
    return c / (fs - c)


def transform_frames(values, rep, gains=None, fs=FS_COUNTS):
    """(n, 6) counts -> (n, 6) values under `rep`. Gains apply to channel i."""
    v = np.asarray(values, dtype=float)
    if rep == "raw":
        return v
    x = conductance_array(v, fs)
    if rep == "conductance":
        return x
    if rep == "gain_matched":
        g = identity_gains() if gains is None else gains
        gvec = np.array([g[i] for i in range(N_SENSORS)], dtype=float)
        return x * gvec
    raise ValueError(f"unknown representation {rep!r}; one of {REPRESENTATIONS}")


def transform_df(df, rep, gains=None, fs=FS_COUNTS):
    """A copy of a capture frame with s0..s5 replaced by the representation."""
    out = df.copy()
    out[SENSOR_COLS] = transform_frames(df[SENSOR_COLS].to_numpy(dtype=float), rep, gains, fs)
    return out


def features_under(df, stances, label, rep, gains=None, fs=FS_COUNTS):
    """features.extract_features on the transformed frame.

    `stances` must come from the RAW total (detector thresholds are counts);
    only the values the extractor sees change. Under B and C the count-valued
    features (peak_counts, loading_rate_cps, impulse_counts_s) are in
    conductance units and keep their column names.
    """
    return extract_features(transform_df(df, rep, gains, fs), stances, label)
