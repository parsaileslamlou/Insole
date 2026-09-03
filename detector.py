"""Stance detection on total-force streams.

This lives in a file, not a notebook cell, so that both the notebook and
test_stances.py import the same definition. Copying the detector into the
test would let the two drift until the test passes against a stale detector.

Failure taxonomy used in the comments below:
  fragmentation  - one true event becomes N detections   (1 FN + N-1 FP)
  annihilation   - one true event becomes 0 detections   (1 FN, precision looks clean)
  merging        - N true events become 1 detection      (N-1 FN)
  boundary error - 1 detection, wrong edges              (invisible to P/R)
"""

SENSOR_COLS = ["s0", "s1", "s2", "s3", "s4", "s5"]

# --- insole geometry ---------------------------------------------
# Measured outline of the physical insole: 274 x 91 mm (10.8 x 3.6 in).
# An earlier 295 x 74 mm figure circulated in these docs and was wrong in
# both axes -- the insole is shorter and considerably wider than that.
INSOLE_LEN_MM   = 274.0
INSOLE_WIDTH_MM = 91.0

# Sensor centres in millimetres on a RIGHT insole lying top-up:
#   y measured from the heel end, increasing toward the toe
#   x measured from the MEDIAL edge, so x = 0 is medial, x = 91 is lateral
#
# Measurement uncertainty: two independent passes over the same six sensors
# disagreed by 12-22 mm, so treat every number below as +/-15 mm. That is
# ~5% of the insole length and ~16% of its width; any CoP figure derived
# from these coordinates inherits it and should be quoted with it.
SENSOR_MM = {
    "s0": (33.0,  50.8),
    "s1": (50.8,  50.8),
    "s2": (76.2, 152.4),
    "s3": (81.3, 185.4),
    "s4": (25.4, 203.2),
    "s5": (25.4, 254.0),
}

# Derived from SENSOR_MM, never hand-typed -- decimals typed twice drift.
#
# BOTH axes are divided by INSOLE_LEN_MM, not each by its own extent.
# Normalising x by the width instead would stretch the short axis by
# 274/91 = 3.01 and distort every distance, angle and CoP path length
# computed on these points. Dividing both by the same constant is a
# similarity transform: shape is preserved, and multiplying any coordinate
# by 274 recovers millimetres.
#
# The price is that x spans only [0, 91/274] = [0, 0.332] rather than the
# full unit interval. heatmap.py's FOOT_OUTLINE is rescaled to that same
# box; the two must move together or the sensors render outside the foot.
SENSOR_COORDS = {
    name: (x_mm / INSOLE_LEN_MM, y_mm / INSOLE_LEN_MM)
    for name, (x_mm, y_mm) in SENSOR_MM.items()
}

# --- detector constants ------------------------------------------
# Every value here is hardware-sensitive. All five were picked by measuring
# the simulated streams, not by guessing. The measurements they rest on:
#
#   stream    truth dur   in-stance peak   in-stance trough   between-stance peak
#   walk        62 fr       7637..7874        950..1137              213
#   fast        37 fr       7617..7847       1371..1558              285
#   shuffle     31 fr       3384..3596        596.. 790              220
#   dropout     62 fr       7623..7812          0..1147              219
#   standing     -          6381..6824             (never falls)       -
#
T_ON         = 1200   # RETUNE: entry threshold, total force across six channels.
                      # Must clear the 285 between-stance noise peak but sit well
                      # under the shuffle stance peak (~3400). At 2000 the shuffle
                      # spends only ~12 frames above it, under MIN_DURATION, and
                      # every shuffle stance is annihilated.
T_OFF        =  450   # RETUNE: exit threshold; must be < T_ON or the state machine
                      # chatters. Binding constraint is the shuffle midstance
                      # trough (596): above it, every shuffle stance fragments.
MIN_DURATION =   15   # RETUNE: frames; must be BELOW the shortest stance you intend
                      # to support. Shortest detected stance is the shuffle at 25
                      # frames. Still rejects the single-frame noise spike, which
                      # is what this guard was actually for.
MAX_DURATION =  120   # RETUNE: frames; must be ABOVE the longest real stance
                      # (58 detected) and below "standing" (unbounded).
GAP_MERGE    =   12   # RETUNE: frames; two detections closer than this are one
                      # stance. Tightest true separation is the shuffle at 19
                      # frames of truth / ~24 frames as detected, so this must
                      # stay under ~24 or it merges genuine adjacent shuffle steps.


def find_stances(total, t_on=T_ON, t_off=T_OFF,
                 min_duration=MIN_DURATION, max_duration=MAX_DURATION):
    """Two-state hysteresis detector over a total-force sequence.

    Returns a list of inclusive (start_idx, end_idx) pairs.
    """
    stances = []
    in_stance = False
    start = 0
    # Cleared by a max_duration break. While False, re-entry is blocked until
    # force has actually fallen below t_off at least once. Without this latch,
    # max_duration alone converts the standing annihilation into 25 back-to-back
    # bogus stances -- i.e. it trades one failure bucket for a worse one.
    armed = True
    i = -1

    for i, x in enumerate(total):
        if in_stance:
            if x < t_off:
                if i - start >= min_duration:
                    stances.append((start, i - 1))
                in_stance = False
            elif i - start >= max_duration:
                # Longer than any real stance: this is standing, or a sensor
                # stuck high. Discard rather than emit -- emitting would report
                # standing as a stance, which is the failure being fixed.
                in_stance = False
                armed = False
        else:
            if not armed:
                if x < t_off:
                    armed = True
            elif x >= t_on:
                start = i
                in_stance = True

    # A stance still open when the recording ends is a real stance. The old
    # loop dropped it as a side effect of its shape, not as a decision.
    #
    # This is also why max_duration is load-bearing rather than cosmetic. The
    # old loop scored standing correctly (0 stances) only by accident: force
    # never fell below t_off, so the open stance was silently discarded at EOF.
    # Emitting it here without the max_duration break turns 60 s of standing
    # into one 6000-frame "stance".
    if in_stance and min_duration <= i - start + 1 <= max_duration:
        stances.append((start, i))

    return stances


def merge_close(stances, gap=GAP_MERGE):
    """Join detections separated by fewer than `gap` frames.

    This is the fragmentation fix: a stance whose midstance trough dips below
    t_off is split in two, and this glues it back together with both true
    outer boundaries intact.

    Chosen over a refractory lockout because a lockout blocks *entry* for R
    frames after a stance ends, which (a) just re-enters a few frames later if
    force is still above t_on -- turning a visible fragmentation into an
    invisible boundary error -- and (b) needs R above the trough width but
    below the shortest stride interval, a window that narrows with cadence.

    Ordering caveat: find_stances drops sub-min_duration runs before this ever
    sees them, so merge_close can only rejoin fragments that individually clear
    min_duration. If a trough splits a stance into two pieces both shorter than
    min_duration, the failure is annihilation, not fragmentation, and no amount
    of gap merging recovers it -- t_off has to come down instead.
    """
    if not stances:
        return []
    out = [stances[0]]
    for start, end in stances[1:]:
        prev_start, prev_end = out[-1]
        if start - prev_end - 1 < gap:
            out[-1] = (prev_start, end)
        else:
            out.append((start, end))
    return out


def _overlap(x, y):
    return max(0, min(x[1], y[1]) - max(x[0], y[0]) + 1)


def stance_report(detected, truth):
    """Mean boundary and duration error, in frames, over matched pairs.

    This is the only instrument that can see a boundary error. Counts, and
    therefore precision and recall, are blind to it forever -- a detector that
    finds every stance but clips 15% off each contact time scores 1.000.
    Print this beside every P/R number.
    """
    empty = {"n_matched": 0, "mean_start_offset": float("nan"),
             "mean_end_offset": float("nan"), "mean_duration_error": float("nan")}
    if not detected or not truth:
        return empty

    claimed = set()
    start_offsets, end_offsets, duration_errors = [], [], []
    for d in detected:
        best, best_ov = None, 0
        for j, t in enumerate(truth):
            if j in claimed:
                continue
            ov = _overlap(d, t)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is None:
            continue
        claimed.add(best)
        t = truth[best]
        start_offsets.append(d[0] - t[0])
        end_offsets.append(d[1] - t[1])
        duration_errors.append((d[1] - d[0] + 1) - (t[1] - t[0] + 1))

    if not start_offsets:
        return empty
    n = len(start_offsets)
    return {
        "n_matched": n,
        "mean_start_offset": sum(start_offsets) / n,
        "mean_end_offset": sum(end_offsets) / n,
        "mean_duration_error": sum(duration_errors) / n,
    }
