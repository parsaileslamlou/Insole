"""infer_live.py -- streaming stance detection and classification.

One frame at a time, from whichever line source read_serial.make_source()
hands back, through the same validation, detector and feature code the batch
pipeline uses, to a persisted classifier. Switching between a saved file, USB
serial and BLE is a command-line argument, not a code change.

    python -m insole.infer_live data/sim/sim_walk.txt                  replay a frame log
    python -m insole.infer_live data/real/walk02.csv --label walk      replay a CSV capture
    python -m insole.infer_live --source serial --port COM13 --duration 60
    python -m insole.infer_live --source ble --duration 60
    python -m insole.infer_live --source serial --out preds.csv        also write per-stance rows

What it prints, and why each line is there
------------------------------------------
Per completed stance: frame span, length, features, predicted class, and the
running counters. Every STATUS_EVERY_S seconds regardless: the counters alone,
so a long silence is visibly "frames arriving, nothing completing" rather
than "nothing arriving". The counters that matter most:

    stances completed / discarded>MAX
        A stance longer than detector.MAX_DURATION is DISCARDED by the
        detector, not clipped (armed-latch design, see find_stances). At the
        old 120-frame ceiling that was 17/35 walk and 28/30 shuffle contacts.
        Without this counter beside the completed count, sparse predictions
        look like a classifier failure when they are a segmentation one.
    extrap
        Fraction of valid frames with at least one sensor above
        calibration.CAL_MAX_COUNTS, the highest count any calibration sample
        reached. The gain match is extrapolating on those frames. They are
        counted, never clamped or refused.
    s4=0
        Fraction of valid frames where s4 reads exactly 0. That is BELOW
        ACTIVATION THRESHOLD, not missing data (measured: s4 = 0 counts at
        2.58 N while s5 read 239 at 2.49 N). CoP on those frames is a
        5-sensor centroid with a 33.6-36.7 mm lateral bias. Passed through,
        counted, never imputed or dropped.
    allzero
        Frames where all six sensors read 0. features.cop_frame returns
        (nan, nan) for those by documented behaviour; cop_features skips
        them. Nothing here adds handling to detector.py or features.py.
    gap_frames / resets
        Link faults (gait_gen's fault modes; README "Fault handling"). A
        frame that never arrived or failed its checksum is dropped and
        counted by read_serial.FrameValidator, never reconstructed. A stance
        whose span has a hole in it, for either reason, is FLAGGED with the
        number of frames missing strictly inside it -- gap_frames, printed
        beside the prediction and written to --out -- and its features are computed
        over the frames that did arrive, exactly as the batch path treats
        the same CSV. A board reset (device clock ran backwards) is counted
        once: the stance in progress is discarded (StanceTracker.reset), a
        stance already complete is released, the running-median dt and the
        frame buffer are cleared, and t_start_s restarts with the new epoch
        (the `epoch` column). Sensor values are never imputed.

What the model sees
-------------------
THE MODEL DECIDES, not this script. Every persisted model records the input
representation it was fitted on in `meta.representation`
(insole/representations.py: A raw counts, B conductance
x = counts / (4095 - counts) per channel, C gain-matched conductance). This
script reads that field and applies the matching transform, so a model fitted
under any of the three loads and runs here. That is deliberate: the file
format must not be what decides which representation the project can ship.
Before, features were always computed under representations.SHIPPED and a
model naming anything else was rejected, which meant the persisted models had
to be fitted under B for this script to accept them -- the loader silently
constraining the science. It no longer does. SHIPPED is now only the default
model's representation, a recorded decision (docs/real_results.md section 5),
and `--model models/model_qda_real_raw.json` runs the A-fitted model correctly
rather than being refused.

There is NO silent fallback. A model is refused at startup, naming both
sides, when its representation cannot be honoured:

    * `meta.representation` absent -- a pre-stage-20 fit. What it was trained
      on is unrecorded, so feeding it anything is a guess.
    * an unknown name -- not one of insole.representations.REPRESENTATIONS.
    * "gain_matched" under `--gain none` -- C is conductance times the
      per-channel corrections, and with no gain document there are no
      corrections to apply. Substituting identity gains would silently
      demote C to B.

The detector always runs on raw counts whatever the model's representation
(T_ON / T_OFF are in counts). The gain match is applied every frame in
conductance space via calibration.apply_gain_match; its output is carried in
the buffer and drives the extrap counter. Under A and B it does not reach the
features -- on the real captures C scored no better than plain B while
extrapolating on most loaded frames (docs/real_results.md). Under C it is what
the features are computed on.

The default model is the one trained on the real captures
(scripts/train_real.py -> models/model_qda_real.json, leave-one-session-out
over two sessions per class; docs/real_results.md). The sim-trained model
(scripts/fit_model.py -> models/model_lda.json) stays reachable with --model;
on real captures it scores below the majority floor. The banner names the
kind of model loaded and what its predictions are good for.

Equivalence
-----------
test_infer_live.py asserts that this script's per-stance features on a file
are IDENTICAL to extract_features(df, merge_close(find_stances(total))) on
the CSV read_serial.py writes from the same file. The one documented
deviation: the batch path takes dt as the median ts_us step over the WHOLE
file; this script takes the median over the frames seen so far, which is
all a stream can know. On every capture in the repo both are exactly
10000 us, so the features agree bit for bit.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import Counter, deque
from itertools import islice
from time import perf_counter

import numpy as np
import pandas as pd

from insole import calibration as C
from insole import detector as D
from insole import read_serial as RS
from insole.discriminant import load_model, predict
from insole.features import cop_features, cop_trajectory, stance_features
from insole.representations import (
    LETTER, REPRESENTATIONS, SHIPPED, gains_from_doc, transform_frames,
)

from insole.paths import MODELS, REPO as _REPO

REPO = str(_REPO)
DEFAULT_MODEL = str(MODELS / "model_qda_real.json")   # trained on the real captures
SIM_MODEL = str(MODELS / "model_lda.json")             # sim-trained, --model for the simulator demo
DEFAULT_GAIN = str(MODELS / "gain_match.json")

STATUS_EVERY_S = 5.0
STAGES = ("read", "validate", "calibrate", "detect", "feature", "predict")
FEATURE_COLS = ["peak_counts", "time_to_peak_s", "contact_time_s",
                "loading_rate_cps", "impulse_counts_s",
                "cop_path_len", "cop_displacement"]
S4 = D.SENSOR_COLS.index("s4")


# ---------------------------------------------------------------------------
# Line sources beyond read_serial's three: a CSV capture replayed as frames
# ---------------------------------------------------------------------------
def frame_line(seq, ts_us, vals):
    """Encode one frame exactly as the firmware does (docs/frame_spec.md section 4)."""
    nums = [int(seq), int(ts_us)] + [int(v) for v in vals]
    return "INS," + ",".join(str(n) for n in nums) + f",{sum(nums) % 256}"


def csv_lines(path):
    """Replay a read_serial CSV (seq,ts_us,s0..s5) as frame lines.

    The real captures in data/real/ exist only as CSV, so this is how they
    reach the same parse_frame path a live board does. Re-encoding through the
    checksum means a corrupt CSV row fails validation instead of being trusted.
    """
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        want = RS.CSV_HEADER
        if [h.strip() for h in header] != want:
            raise ValueError(f"{path}: header {header} is not {want}")
        for row in r:
            if not row:
                continue
            yield frame_line(row[0], row[1], row[2:8])


def make_lines(source, in_path, duration_s, port):
    if source == "file" and in_path and in_path.lower().endswith(".csv"):
        return csv_lines(in_path)
    return RS.make_source(source, in_path, duration_s=duration_s, port=port)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class Timers:
    """perf_counter accumulators, one per stage."""

    def __init__(self):
        self.n = Counter()
        self.total = Counter()
        self.worst = Counter()

    def tick(self, stage, t0):
        dt = perf_counter() - t0
        self.n[stage] += 1
        self.total[stage] += dt
        if dt > self.worst[stage]:
            self.worst[stage] = dt

    def report(self):
        print(f"{'stage':10s} {'calls':>8s} {'total_ms':>10s} {'mean_us':>9s} {'max_us':>9s}")
        for st in STAGES:
            n = self.n[st]
            if n == 0:
                print(f"{st:10s} {0:8d} {'--':>10s} {'--':>9s} {'--':>9s}")
                continue
            print(f"{st:10s} {n:8d} {1e3 * self.total[st]:10.1f} "
                  f"{1e6 * self.total[st] / n:9.1f} {1e6 * self.worst[st]:9.1f}")


class RunningMedianDt:
    """Median ts_us step over the frames seen so far, pandas-median semantics.

    Steps are integers, so a Counter of them gives an exact median without
    keeping the list. Even count -> mean of the two middle values, as
    pandas.Series.median does in features.frame_dt.
    """

    def __init__(self):
        self.steps = Counter()
        self.n = 0
        self.prev = None

    def push(self, ts_us):
        if self.prev is not None:
            self.steps[ts_us - self.prev] += 1
            self.n += 1
        self.prev = ts_us

    def dt_s(self):
        if self.n == 0:
            return float("nan")
        keys = sorted(self.steps)
        lo_rank, hi_rank = (self.n - 1) // 2, self.n // 2
        seen, lo_val, hi_val = 0, None, None
        for k in keys:
            seen += self.steps[k]
            if lo_val is None and seen > lo_rank:
                lo_val = k
            if hi_val is None and seen > hi_rank:
                hi_val = k
                break
        return (lo_val + hi_val) / 2.0 / 1_000_000.0


def features_for(rows, dt, rep=SHIPPED, gains=None):
    """The batch feature functions applied to one stance's frames.

    `rows` is the (ts_us, vals) list for frames start..end INCLUSIVE, vals
    being raw counts; they are transformed to `rep` here, once, on the way in.
    `rep` is the loaded model's own `meta.representation`, not a constant --
    resolve_representation() has already checked it can be honoured, and
    `gains` is non-None exactly when rep is "gain_matched". The
    notebook-lifted extractors slice [start:end], which drops the last frame;
    passing (0, n-1) here reproduces that exactly.
    """
    n = len(rows)
    df = pd.DataFrame(transform_frames([r[1] for r in rows], rep, gains),
                      columns=D.SENSOR_COLS)
    total = df[D.SENSOR_COLS].sum(axis=1)
    out = stance_features(total, dt, 0, n - 1)
    out.update(cop_features(cop_trajectory(df, 0, n - 1)))
    return out


class RepresentationError(Exception):
    """A model names an input representation this run cannot honour.

    Carries the message printed at startup. Raised, never swallowed: the
    alternative to refusing is feeding the classifier a distribution it was
    not fitted on, which produces confident labels off a silent unit change.
    """


def resolve_representation(meta, model_path, gain_doc, gain_arg):
    """(rep, gains) for a loaded model, or raise RepresentationError.

    The model's `meta.representation` decides, and this function's only job is
    to say whether this run can honour it. It never substitutes one
    representation for another -- there is no fallback path, silent or
    otherwise, because every fallback here is a wrong answer delivered
    quietly. Both sides are named in every message: what the model asks for,
    and why this run cannot give it.

    `gain_doc` is the loaded gain-match document or None when --gain none was
    passed; `gain_arg` is the flag's text, for the message.
    """
    rep = meta.get("representation")
    if rep is None:
        raise RepresentationError(
            f"FAIL: {model_path} records no representation.\n"
            f"      Its meta has no 'representation' field, so what the features were\n"
            f"      computed on when it was fitted is not recorded anywhere. That is a\n"
            f"      pre-stage-20 fit. Feeding it raw counts, conductance or gain-matched\n"
            f"      conductance would be a guess, and a wrong guess produces confident\n"
            f"      labels off a silent unit change, so this script refuses instead.\n"
            f"      Refit it: scripts/train_real.py (real captures) or scripts/fit_model.py\n"
            f"      (simulated). Both stamp the representation into the model.")
    if rep not in REPRESENTATIONS:
        raise RepresentationError(
            f"FAIL: {model_path} names representation {rep!r}, which this build does not have.\n"
            f"      insole.representations.REPRESENTATIONS is {list(REPRESENTATIONS)}.\n"
            f"      Either the model comes from a newer tree than this checkout, or the\n"
            f"      field is corrupt. This script will not guess which transform was meant.")
    if rep == "gain_matched" and gain_doc is None:
        raise RepresentationError(
            f"FAIL: {model_path} was fitted on representation {LETTER[rep]} ({rep}), and\n"
            f"      this run has no gain match: --gain {gain_arg!r}.\n"
            f"      {LETTER[rep]} is conductance multiplied by the per-channel corrections in\n"
            f"      models/gain_match.json. With no gain document there are no corrections,\n"
            f"      and applying identity gains instead would silently demote {LETTER[rep]} to\n"
            f"      {LETTER['conductance']} (conductance) while still reporting {LETTER[rep]}.\n"
            f"      Either drop --gain none, or load a model fitted under "
            f"{LETTER['raw']} or {LETTER['conductance']}.")
    gains = gains_from_doc(gain_doc) if rep == "gain_matched" else None
    return rep, gains


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Stream frames through detector -> features -> classifier.")
    p.add_argument("in_path", nargs="?", default=None,
                   help="frame log (.txt) or read_serial CSV to replay; implies --source file")
    p.add_argument("--source", choices=("serial", "file", "ble"), default=None,
                   help="transport; defaults to 'file' when in_path is given, else 'serial'")
    p.add_argument("--port", default=None,
                   help=f"serial port for --source serial (default: read_serial.PORT = {RS.PORT})")
    p.add_argument("--duration", type=float, default=None, metavar="SECONDS",
                   help=f"live capture length (default: read_serial.DURATION_S = {RS.DURATION_S})")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="persisted classifier (default: models/model_qda_real.json, trained on the "
                        "real captures by scripts/train_real.py; models/model_lda.json is the "
                        "sim-trained one from scripts/fit_model.py)")
    p.add_argument("--gain", default=DEFAULT_GAIN,
                   help="relative gain match JSON (default: models/gain_match.json); 'none' to skip")
    p.add_argument("--out", default=None,
                   help="write one CSV row per completed stance here")
    p.add_argument("--label", default=None,
                   help="true activity of a replayed file, for the agreement line at exit")
    p.add_argument("--status-every", type=float, default=STATUS_EVERY_S, metavar="SECONDS",
                   help=f"print the counters this often even with no stance (default {STATUS_EVERY_S})")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-stance and status lines; summary only")
    return p


def resolve_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.source is None:
        args.source = "file" if args.in_path else "serial"
    if args.source == "file" and args.in_path is None:
        build_parser().error("--source file needs a path to replay")
    if args.source != "file" and args.in_path is not None:
        build_parser().error(f"--source {args.source} takes no input path")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def counters_line(v, tr, k):
    c = v.c
    n = max(c["valid"], 1)
    return (f"valid={c['valid']} bad={c['malformed'] + c['bad_checksum'] + c['empty']} "
            f"seq_breaks={c['seq_breaks']} lost={c['lost']} resets={c['resets']} | "
            f"stances={tr.n_stances} discarded>MAX={tr.n_discarded_max} "
            f"discarded@reset={tr.n_discarded_reset} rejected<MIN={tr.n_rejected_min} | "
            f"extrap={100.0 * k['extrap'] / n:.1f}% s4=0:{100.0 * k['s4zero'] / n:.1f}% "
            f"allzero={k['allzero']}")


def main(argv=None):
    args = resolve_args(argv)
    duration_s = RS._resolve(args.duration, RS.DURATION_S)

    model = load_model(args.model)
    meta = model.get("meta", {})
    feat_names = meta.get("features", ["cop_path_len", "cop_displacement"])

    gm = None
    if args.gain.lower() != "none":
        gm = C.load_gain_match(args.gain)

    # The model's own representation drives the feature path from here on.
    # Refused rather than approximated -- see resolve_representation.
    try:
        model_rep, model_gains = resolve_representation(meta, args.model, gm, args.gain)
    except RepresentationError as exc:
        print(exc)
        return 2

    # -- banner -------------------------------------------------------------
    print(f"infer_live: source={args.source}"
          + (f" in={args.in_path}" if args.in_path else "")
          + (f" port={RS._resolve(args.port, RS.PORT)}" if args.source == "serial" else "")
          + (f" duration={duration_s:g}s stall={RS.STALL_S:g}s" if args.source != "file" else ""))
    print(f"model     : {os.path.relpath(args.model, REPO) if args.model.startswith(REPO) else args.model}"
          f"  kind={model['kind']}  classes={[str(c) for c in model['classes']]}  features={feat_names}")
    hc = meta.get("heldout_check", {})
    simulated = str(meta.get("training_data", "")).upper().startswith("SIMULATED")
    if hc and "accuracy" in hc:
        print(f"            trained on {meta.get('n_rows')} rows / "
              f"{len(meta.get('sessions', []))} sessions; {hc.get('split', 'held-out check')}: "
              f"{hc['accuracy']:.4f} "
              f"[{hc.get('wilson95_lo', float('nan')):.4f}, {hc.get('wilson95_hi', float('nan')):.4f}] "
              f"on {hc.get('n_test')} {'SIMULATED' if simulated else 'real'} stances"
              + (f", majority floor {hc['test_floor']:.4f}" if "test_floor" in hc else ""))
    if simulated:
        print("WARNING   : the model is fitted on simulated gait. On real captures this")
        print("            recipe scored below the majority floor (docs/real_results.md, section 9).")
        print("            Predictions on real gait are a plumbing check, not a result.")
    else:
        print("NOTE      : the model is fitted on the real captures (one subject, figure-8 path,")
        print("            docs/real_results.md). On simulated frames its labels are a plumbing")
        print("            check, not a result; --model models/model_lda.json is the sim-trained one.")
    print(f"detector  : T_ON={D.T_ON} T_OFF={D.T_OFF} MIN_DURATION={D.MIN_DURATION} "
          f"MAX_DURATION={D.MAX_DURATION} GAP_MERGE={D.GAP_MERGE}  "
          f"(a run > MAX_DURATION is DISCARDED, not clipped)")
    print(f"features  : representation {LETTER[model_rep]} ({model_rep}), read from the model's "
          f"meta and applied on every source; the detector sees raw counts"
          + ("" if model_rep == SHIPPED else
             f"  [NOT the shipped default {LETTER[SHIPPED]} ({SHIPPED})]"))
    if gm is not None:
        print("gain match: " + "  ".join(f"s{i}={gm['corrections'][i]:.4f}" for i in range(6))
              + f"  applied to x = counts/({gm['fs_counts']:g} - counts); "
              f"extrapolating above {C.CAL_MAX_COUNTS} counts")
    else:
        print("gain match: skipped (--gain none)")
    print()

    # -- state --------------------------------------------------------------
    v = RS.FrameValidator()
    tr = D.StanceTracker()
    timers = Timers()
    dts = RunningMedianDt()
    buf = deque()                      # (idx, ts_us, vals, gain, gap_before) for frames still live
    k = Counter()                      # extrap, s4zero, allzero, predicted, no_pred
    preds = []                         # per-stance records
    idx = -1
    epoch = 0                          # incremented at every board reset
    first_ts = None
    t_first = t_last = None
    last_status = time.time()
    stalled = None
    interrupted = False
    RS.SOURCE_DROPS["n"] = 0

    def handle(events, now_s):
        for kind, s, e in events:
            if kind in ("discarded_max", "discarded_reset"):
                if not args.quiet:
                    why = (f"ran {e - s + 1} frames >= MAX_DURATION={D.MAX_DURATION}: "
                           f"DISCARDED, not clipped" if kind == "discarded_max" else
                           f"in progress at a board reset: DISCARDED, end not observable")
                    print(f"discard  frames {s:6d}..{e:6d} {why}  | " + counters_line(v, tr, k))
                continue
            if kind != "stance":
                continue
            # feature ------------------------------------------------------
            t0 = perf_counter()
            first_idx = buf[0][0]
            rows = [(r[1], r[2]) for r in islice(buf, s - first_idx, e - first_idx + 1)]
            assert len(rows) == e - s + 1, (len(rows), s, e, first_idx)
            # Frames missing strictly inside the span (lost or rejected): the
            # slots absent before each frame after the first. A gap before
            # frame s itself is outside the stance.
            gap_frames = sum(r[4] for r in islice(buf, s - first_idx + 1, e - first_idx + 1))
            ft = features_for(rows, dts.dt_s(), model_rep, model_gains)
            timers.tick("feature", t0)
            # predict ------------------------------------------------------
            t0 = perf_counter()
            x = np.array([[float(ft[f]) for f in feat_names]])
            if np.isfinite(x).all():
                label = str(predict(model, x)[0])
                k["predicted"] += 1
            else:
                label = "no-prediction(nan)"
                k["no_pred"] += 1
            timers.tick("predict", t0)
            t_start = (rows[0][0] - first_ts) / 1e6
            rec = {"stance": tr.n_stances, "start": s, "end": e,
                   "n_frames": e - s + 1, "t_start_s": t_start,
                   "epoch": epoch, "gap_frames": gap_frames, "pred": label}
            rec.update({c: ft[c] for c in FEATURE_COLS})
            preds.append(rec)
            if not args.quiet:
                print(f"stance {tr.n_stances:4d}  frames {s:6d}..{e:6d} "
                      f"({e - s + 1:3d} fr, {ft['contact_time_s']:.2f} s) t={t_start:7.2f}s  "
                      f"path={ft['cop_path_len']:.4f} disp={ft['cop_displacement']:.4f}"
                      f"  -> {label:8s} gap_frames={gap_frames} | " + counters_line(v, tr, k))

    lines = make_lines(args.source, args.in_path, duration_s, args.port)
    it = iter(lines)
    try:
        while True:
            # read --------------------------------------------------------
            t0 = perf_counter()
            try:
                line = next(it)
            except StopIteration:
                break
            except RS.StallError as exc:
                stalled = str(exc)
                break
            timers.tick("read", t0)

            now = time.time()
            if t_first is None:
                t_first = now
            elif args.source != "file" and now - t_first > duration_s + 5:
                break                       # runaway guard, as read_serial.main

            if line.startswith("#"):
                print(line)

            # validate ----------------------------------------------------
            t0 = perf_counter()
            row = v.feed(line)
            timers.tick("validate", t0)
            if row is None:
                continue
            seq, ts_us, vals = row
            if v.last_reset:
                # The board rebooted. Settle the tracker first (it may release
                # a complete stance whose frames are still in buf), then drop
                # every frame from before the reset: no future stance can
                # reach across it, and the device clock has restarted.
                handle(tr.reset(), now)
                buf.clear()
                dts = RunningMedianDt()
                first_ts = None
                epoch += 1
                if not args.quiet:
                    print(f"reset    board rebooted (SEQ and ts_us restarted): epoch {epoch}, "
                          f"run in progress discarded, dt state cleared  | "
                          + counters_line(v, tr, k))
            idx += 1
            t_last = now
            if first_ts is None:
                first_ts = ts_us

            # calibrate ---------------------------------------------------
            t0 = perf_counter()
            if gm is not None:
                g = C.apply_gain_match(vals, gm)
            else:
                g = None
            if any(c > C.CAL_MAX_COUNTS for c in vals):
                k["extrap"] += 1
            if vals[S4] == 0:
                k["s4zero"] += 1
            if not any(vals):
                k["allzero"] += 1
            dts.push(ts_us)
            timers.tick("calibrate", t0)

            # detect ------------------------------------------------------
            t0 = perf_counter()
            buf.append((idx, ts_us, vals, g, v.last_gap))
            events = tr.push(sum(vals))
            timers.tick("detect", t0)

            handle(events, now)

            keep = tr.earliest_live_index()
            if keep is None:
                buf.clear()
            else:
                while buf and buf[0][0] < keep:
                    buf.popleft()

            if not args.quiet and now - last_status >= args.status_every:
                last_status = now
                print(f"status  t={now - t_first:6.1f}s  " + counters_line(v, tr, k))
    except KeyboardInterrupt:
        interrupted = True

    handle(tr.flush(), time.time())

    # -- summary ------------------------------------------------------------
    c = v.c
    loss_pct = v.loss_pct()
    capture_s = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    device_s = ((v.last_ts - v.first_ts) / 1e6
                if (v.first_ts is not None and v.last_ts is not None) else 0.0)
    print()
    print(f"source={args.source} valid={c['valid']} malformed={c['malformed']} "
          f"empty={c['empty']} bad_checksum={c['bad_checksum']} "
          f"seq_breaks={c['seq_breaks']} lost={c['lost']} loss={loss_pct:.2f}% "
          f"timing_breaks={c['timing_breaks']} resets={c['resets']} status={c['status']} "
          f"source_drops={RS.SOURCE_DROPS['n']} capture_s={capture_s:.1f} device_s={device_s:.1f}")
    n = max(c["valid"], 1)
    print(f"stances completed={tr.n_stances} discarded>MAX_DURATION({D.MAX_DURATION})="
          f"{tr.n_discarded_max} discarded@reset={tr.n_discarded_reset} "
          f"rejected<MIN_DURATION({D.MIN_DURATION})={tr.n_rejected_min} "
          f"predicted={k['predicted']} no_prediction={k['no_pred']} "
          f"stances_with_gaps={sum(1 for r in preds if r['gap_frames'])}")
    print(f"frames extrapolating (any sensor > {C.CAL_MAX_COUNTS} counts)={k['extrap']} "
          f"({100.0 * k['extrap'] / n:.1f}%)  s4=0 frames={k['s4zero']} "
          f"({100.0 * k['s4zero'] / n:.1f}%)  all-zero frames={k['allzero']}")
    if preds:
        counts = Counter(r["pred"] for r in preds)
        print("predictions: " + "  ".join(f"{lab}={counts[lab]}" for lab in sorted(counts)))
        if args.label:
            agree = sum(1 for r in preds if r["pred"] == args.label)
            print(f"agreement with --label {args.label!r}: {agree}/{len(preds)} "
                  f"= {agree / len(preds):.4f}  (agreement with a typed label, not an accuracy)")
    print()
    print("per-stage timing (perf_counter; 'read' is time waiting on the source):")
    timers.report()

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stance", "start", "end", "n_frames",
                                              "t_start_s", "epoch", "gap_frames"]
                               + FEATURE_COLS + ["pred"])
            w.writeheader()
            for r in preds:
                w.writerow(r)
        print(f"wrote {len(preds)} stance rows to {args.out}")

    if args.source != "file" and capture_s < duration_s - 1.0 and not stalled:
        print(f"NOTE: capture ran {capture_s:.1f}s, {duration_s - capture_s:.1f}s "
              f"short of the requested {duration_s:g}s")
    if RS.SOURCE_DROPS["n"]:
        print(f"NOTE: {RS.SOURCE_DROPS['n']} lines dropped at the source hand-off")

    if stalled:
        print(f"FAIL: stalled -- {stalled}")
        return 1
    if interrupted:
        print("FAIL: interrupted (Ctrl-C); partial run above")
        return 130
    return RS.exit_code(c, loss_pct, args.source)


if __name__ == "__main__":
    sys.exit(main())
