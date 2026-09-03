import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation, PillowWriter

# Single source of truth: these used to be repeated here, in the notebook and
# in detector.py. Editing one copy and not the others silently moved the CoP.
from detector import (
    SENSOR_COLS, SENSOR_COORDS, INSOLE_LEN_MM, INSOLE_WIDTH_MM,
)

# The hand-drawn foot silhouette, in the arbitrary units it was digitised in.
# Only its shape matters; the extents below are measured, not assumed, so
# editing a vertex here cannot silently desynchronise it from the box.
_FOOT_OUTLINE_RAW = np.array([
    (0.30, 0.02), (0.20, 0.08), (0.16, 0.20), (0.18, 0.34),
    (0.22, 0.48), (0.20, 0.62), (0.18, 0.76), (0.20, 0.88),
    (0.28, 0.96), (0.42, 0.99), (0.58, 0.99), (0.72, 0.96),
    (0.80, 0.88), (0.82, 0.76), (0.80, 0.62), (0.80, 0.48),
    (0.82, 0.34), (0.80, 0.20), (0.72, 0.08), (0.58, 0.02),
])

# detector.py normalises both sensor axes by INSOLE_LEN_MM, so the insole
# occupies x in [0, 91/274] = [0, 0.332] and y in [0, 1]. The silhouette has
# to be mapped onto that same box or the six sensors draw outside the foot.
_raw_min = _FOOT_OUTLINE_RAW.min(axis=0)
_raw_span = np.ptp(_FOOT_OUTLINE_RAW, axis=0)
_TARGET_SPAN = np.array([INSOLE_WIDTH_MM / INSOLE_LEN_MM, 1.0])
OUTLINE_SCALE = _TARGET_SPAN / _raw_span

FOOT_OUTLINE = [
    tuple(v) for v in (_FOOT_OUTLINE_RAW - _raw_min) * OUTLINE_SCALE
]

GRID_NX, GRID_NY = 120, 240
IDW_POWER = 2.0

_gx = np.linspace(0.0, 1.0, GRID_NX)
_gy = np.linspace(0.0, 1.0, GRID_NY)
_GX, _GY = np.meshgrid(_gx, _gy)

_pts = np.column_stack([_GX.ravel(), _GY.ravel()])
FOOT_MASK = Path(FOOT_OUTLINE).contains_points(_pts).reshape(_GY.shape)

_coords = np.array([SENSOR_COORDS[c] for c in SENSOR_COLS])
_dist = np.sqrt((_GX[..., None] - _coords[:, 0]) ** 2
                + (_GY[..., None] - _coords[:, 1]) ** 2)
_ON_SENSOR = _dist < 1e-9
_safe = np.where(_ON_SENSOR, 1.0, _dist)
_W = 1.0 / _safe ** IDW_POWER
_W = np.where(_ON_SENSOR.any(axis=-1)[..., None],
              _ON_SENSOR.astype(float), _W)
IDW_WEIGHTS = _W / _W.sum(axis=-1, keepdims=True)


def field(values):
    """values: (6,) sensor readings -> (GRID_NY, GRID_NX) masked field."""
    grid = IDW_WEIGHTS @ np.asarray(values, dtype=float)
    return np.where(FOOT_MASK, grid, np.nan)


CAPTION = ("6 FSR sensors (white circles); values between them are "
           "estimated by inverse-distance weighting, not measured.")


def _draw_foot(ax):
    poly = np.array(FOOT_OUTLINE + [FOOT_OUTLINE[0]])
    ax.plot(poly[:, 0], poly[:, 1], color="0.35", lw=1.2)
    ax.scatter(_coords[:, 0], _coords[:, 1], s=28, facecolor="white",
               edgecolor="0.2", lw=0.8, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render(values, vmin=0, vmax=4095, title="", unit="ADC counts", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 7.0))
    else:
        fig = ax.figure
    im = ax.imshow(field(values), origin="lower", extent=(0, 1, 0, 1),
                   cmap="magma", vmin=vmin, vmax=vmax, interpolation="bilinear")
    _draw_foot(ax)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit)
    fig.text(0.5, 0.015, CAPTION, ha="center", fontsize=7, color="0.35",
             wrap=True)
    return fig, ax, im


def animate(df, stance, coords_cols=SENSOR_COLS, cop_fn=None,
            show_cop=True, out="stance.gif", unit="ADC counts", vmax=None):
    start, end = stance
    seg = df.iloc[start:end]
    vals = seg[coords_cols].to_numpy(dtype=float)
    ts = seg["ts_us"].to_numpy(dtype=float)
    t_rel = (ts - ts[0]) / 1e6

    if vmax is None:
        vmax = float(vals.max())

    fig, ax = plt.subplots(figsize=(4.6, 7.4))
    im = ax.imshow(field(vals[0]), origin="lower", extent=(0, 1, 0, 1),
                   cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
    _draw_foot(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(unit)
    fig.text(0.5, 0.015, CAPTION, ha="center", fontsize=7, color="0.35")

    trail, = ax.plot([], [], color="#39FF9E", lw=1.4, alpha=0.9, zorder=6)
    head, = ax.plot([], [], "o", color="#39FF9E", ms=7, mec="black",
                    mew=0.8, zorder=7)

    cop = None
    if show_cop and cop_fn is not None:
        cop = np.array([cop_fn(r) for _, r in seg.iterrows()], dtype=float)

    def update(k):
        im.set_data(field(vals[k]))
        ax.set_title(f"t = {t_rel[k]:.3f} s   (frame {k + 1}/{len(vals)})")
        if cop is not None:
            good = ~np.isnan(cop[:k + 1, 0])
            trail.set_data(cop[:k + 1, 0][good], cop[:k + 1, 1][good])
            if not np.isnan(cop[k, 0]):
                head.set_data([cop[k, 0]], [cop[k, 1]])
        return im, trail, head

    dt_ms = float(np.median(np.diff(ts)) / 1000.0)
    anim = FuncAnimation(fig, update, frames=len(vals),
                         interval=dt_ms, blit=False)
    anim.save(out, writer=PillowWriter(fps=1000.0 / dt_ms))
    plt.close(fig)
    return out, dt_ms
