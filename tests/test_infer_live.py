"""Tests for the streaming path (infer_live.py, detector.StanceTracker,
read_serial's seam changes). Run from the repo root:

    python tests/test_infer_live.py

Also collected by pytest. No hardware: the serial and BLE sources are
replaced by fakes that yield the same bytes a file would.

The one that matters is check_equivalence: on every capture in the repo, the
per-stance features infer_live.py prints must be IDENTICAL to what
features.extract_features computes on the CSV read_serial.py writes from the
same file. If they ever differ, the model is being fed a distribution it was
not trained on, and every other test here is decoration.

Each check_* function prints PASS/FAIL lines and returns (passed, failed);
the test_* wrapper of the same name asserts nothing failed, so pytest sees a
failure and the direct run keeps its counts.
"""

import contextlib
import csv
import glob
import inspect
import io
import os
import random
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import pandas as pd

from insole import calibration as C
from insole import detector as D
from insole import infer_live as IL
from insole import read_serial as RS
from insole.discriminant import fit_lda, fit_qda, load_model, predict, save_model
from insole.representations import SHIPPED, features_under

from insole.paths import CAL_DATA, DATA_REAL, DATA_SIM, MODELS, REPO as _REPO

REPO = str(_REPO)
REAL = str(DATA_REAL)

SIM_FIXTURES = ["sim_walk", "sim_fast", "sim_shuffle", "sim_dropout", "sim_stand"]
REAL_FILES = ["stand_02.csv", "walk02.csv", "fast02.csv", "shuffle02.csv"]

# Columns the equivalence check compares, bit for bit.
FEATS = IL.FEATURE_COLS


def check(name, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    return bool(condition)


class Tally:
    def __init__(self):
        self.passed = self.failed = 0

    def __call__(self, name, condition, detail=""):
        ok = check(name, condition, detail)
        self.passed += ok
        self.failed += (not ok)
        return ok

    def result(self):
        return self.passed, self.failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_csv(stem):
    """Sim CSVs are gitignored; rebuild from the committed .txt via read_serial."""
    csv_path = os.path.join(DATA_SIM, stem + ".csv")
    txt_path = os.path.join(DATA_SIM, stem + ".txt")
    if not os.path.exists(csv_path):
        subprocess.run([sys.executable, "-m", "insole.read_serial",
                        txt_path, csv_path], check=True, cwd=REPO,
                       stdout=subprocess.DEVNULL)
    return csv_path


def batch_features(csv_path, label="x"):
    """The batch path: detect on raw counts, features under the shipped representation."""
    df = pd.read_csv(csv_path)
    total = df[D.SENSOR_COLS].sum(axis=1).to_numpy()
    return features_under(df, D.merge_close(D.find_stances(total)), label, SHIPPED)


def run_infer(argv, quiet=True):
    """infer_live.main in-process. Returns (exit_code, stdout, per-stance rows)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "preds.csv")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = IL.main(list(argv) + ["--out", out] + (["--quiet"] if quiet else []))
        rows = []
        if os.path.exists(out):
            with open(out, newline="") as f:
                rows = list(csv.DictReader(f))
    return code, buf.getvalue(), rows


def same_values(a, b):
    """Exact equality with NaN == NaN. No tolerance: the code paths are the same."""
    a, b = float(a), float(b)
    return (a == b) or (np.isnan(a) and np.isnan(b))


def stream_features_match(rows, feats):
    """infer_live rows vs an extract_features frame: same count, spans, values."""
    if len(rows) != len(feats):
        return False, f"n stream={len(rows)} batch={len(feats)}"
    for r, (_, b) in zip(rows, feats.iterrows()):
        if int(r["start"]) != int(b["start"]) or int(r["end"]) != int(b["end"]):
            return False, f"span stream=({r['start']},{r['end']}) batch=({b['start']},{b['end']})"
        for col in FEATS:
            if not same_values(r[col], b[col]):
                return False, f"{col} at {r['start']}: stream={r[col]} batch={b[col]}"
    return True, f"{len(rows)} stances, {len(FEATS)} features each, bit-identical"


def stances_via_tracker(total, **kw):
    t = D.StanceTracker(**kw)
    out = []
    for x in total:
        out += t.push(x)
    out += t.flush()
    return [(s, e) for k, s, e in out if k == "stance"], t


def pulse_lines(pulses, seq0=0, ts0=0, per_sensor=400, quiet=(0, 0, 0, 0, 0, 0)):
    """Frame lines: `pulses` is a list of (n_on, n_off) run lengths.

    On-frames put `per_sensor` counts on every channel (total 6*per_sensor,
    well above T_ON = 1200 at the default 400); off-frames use `quiet`.
    """
    lines, seq, ts = [], seq0, ts0
    for n_on, n_off in pulses:
        for _ in range(n_on):
            lines.append(IL.frame_line(seq, ts, [per_sensor] * 6))
            seq, ts = seq + 1, ts + RS.PERIOD_US
        for _ in range(n_off):
            lines.append(IL.frame_line(seq, ts, list(quiet)))
            seq, ts = seq + 1, ts + RS.PERIOD_US
    return lines


@contextlib.contextmanager
def fake_live_source(lines, source="serial", raise_after=None):
    """Replace read_serial's live source with a generator over `lines`.

    make_source is left alone so the real dispatch runs; only the transport
    function it calls is swapped. raise_after=N raises StallError after N
    lines, which is what a board going quiet looks like to the consumer.
    """
    name = "serial_lines" if source == "serial" else "ble_lines"
    saved = getattr(RS, name)

    def fake(*_a, **_kw):
        for i, line in enumerate(lines):
            if raise_after is not None and i >= raise_after:
                raise RS.StallError(f"fake {source}: no data for {RS.STALL_S:.1f}s")
            yield line

    setattr(RS, name, fake)
    try:
        yield
    finally:
        setattr(RS, name, saved)


def write_lines(path, lines):
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 1. Equivalence: streaming features == batch features, every file
# ---------------------------------------------------------------------------
def check_equivalence():
    t = Tally()

    for stem in SIM_FIXTURES:
        csv_path = ensure_csv(stem)
        feats = batch_features(csv_path)
        code, out, rows = run_infer([os.path.join(DATA_SIM, stem + ".txt")])
        ok, detail = stream_features_match(rows, feats)
        t(f"equivalence {stem}.txt (frame log)", ok and code == 0, detail + f" exit={code}")

    for fname in REAL_FILES:
        path = os.path.join(REAL, fname)
        feats = batch_features(path)
        code, out, rows = run_infer([path])
        ok, detail = stream_features_match(rows, feats)
        t(f"equivalence data/real/{fname} (CSV replay)", ok and code == 0,
          detail + f" exit={code}")

    # The tracker alone, on every sim session that exists and on random input
    # with random thresholds, against find_stances + merge_close.
    n_files = 0
    all_eq = True
    for path in sorted(glob.glob(os.path.join(DATA_SIM, "sim_*_[0-9][0-9].csv"))):
        df = pd.read_csv(path)
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy()
        got, _ = stances_via_tracker(total)
        all_eq &= got == D.merge_close(D.find_stances(total))
        n_files += 1
    t("StanceTracker == find_stances+merge_close on every sim session", all_eq,
      f"{n_files} session files")

    rng = random.Random(0)
    bad = 0
    for _ in range(2000):
        n = rng.randint(1, 300)
        lvl, seq = 0.0, []
        for _ in range(n):
            lvl = max(0.0, lvl + rng.gauss(0, 900))
            seq.append(lvl)
        kw = dict(t_on=rng.choice([1200, 800]), t_off=rng.choice([450, 300]),
                  min_duration=rng.choice([1, 2, 5, 15]),
                  max_duration=rng.choice([3, 10, 30, 120, 200]),
                  gap=rng.choice([0, 1, 3, 12]))
        got, _ = stances_via_tracker(seq, **kw)
        want = D.merge_close(D.find_stances(seq, kw["t_on"], kw["t_off"],
                                            kw["min_duration"], kw["max_duration"]),
                             kw["gap"])
        bad += got != want
    t("StanceTracker == find_stances+merge_close on 2000 random sequences", bad == 0,
      f"mismatches={bad}")
    return t.result()


# ---------------------------------------------------------------------------
# 2. A stance spanning a read boundary
# ---------------------------------------------------------------------------
def check_read_boundary():
    t = Tally()
    txt = os.path.join(DATA_SIM, "sim_walk.txt")
    lines = [ln for ln in RS.file_lines(txt)]
    feats = batch_features(ensure_csv("sim_walk"))
    s0, e0 = int(feats.iloc[0]["start"]), int(feats.iloc[0]["end"])
    cut = (s0 + e0) // 2                    # inside the first stance

    # (a) the tracker: same object, fed in two reads, with an idle gap between.
    df = pd.read_csv(ensure_csv("sim_walk"))
    total = df[D.SENSOR_COLS].sum(axis=1).to_numpy()
    tr = D.StanceTracker()
    ev = []
    for x in total[:cut]:
        ev += tr.push(x)
    t("tracker is mid-stance at the boundary", tr.in_stance and tr.start == s0,
      f"in_stance={tr.in_stance} start={tr.start} want {s0}")
    for x in total[cut:]:
        ev += tr.push(x)
    ev += tr.flush()
    got = [(s, e) for k, s, e in ev if k == "stance"]
    t("tracker: stance spanning the boundary has the batch boundaries",
      got == D.merge_close(D.find_stances(total)), f"first={got[:1]} want=({s0},{e0})")

    # (b) the whole script, with the live source delivering the file in two
    # reads separated by a firmware status line (what a BLE notify boundary
    # or a serial readline boundary looks like to the consumer).
    chunked = lines[:cut] + ["# boundary: second read starts here"] + lines[cut:]
    with fake_live_source(chunked, "serial"):
        code, out, rows = run_infer(["--source", "serial"])
    ok, detail = stream_features_match(rows, feats)
    t("infer_live: stance spanning a read boundary is bit-identical to batch",
      ok and code == 0, detail + f" exit={code}")
    t("status line at the boundary counted, not treated as corruption",
      "status=1" in out and "malformed=0" in out, detail)
    return t.result()


# ---------------------------------------------------------------------------
# 3. Malformed frames arriving mid-stance
# ---------------------------------------------------------------------------
def check_malformed_mid_stance():
    t = Tally()
    lines = [ln for ln in RS.file_lines(os.path.join(DATA_SIM, "sim_walk.txt"))]
    feats0 = batch_features(ensure_csv("sim_walk"))
    s0, e0 = int(feats0.iloc[0]["start"]), int(feats0.iloc[0]["end"])
    mid = (s0 + e0) // 2

    bad = list(lines)
    bad[mid] = bad[mid][:-1] + ("0" if bad[mid][-1] != "0" else "1")   # checksum wrong
    bad.insert(mid + 5, "INS,garbage,line")                            # malformed
    bad.insert(mid + 9, "")                                            # empty
    bad.insert(mid + 12, "INS,1,2,3,4,5,6,7,8,9,10,11")                 # wrong field count

    with tempfile.TemporaryDirectory() as tmp:
        bad_txt = os.path.join(tmp, "bad.txt")
        bad_csv = os.path.join(tmp, "bad.csv")
        write_lines(bad_txt, bad)
        # What the batch path sees: read_serial.py's CSV of the same bytes.
        r = subprocess.run([sys.executable, "-m", "insole.read_serial",
                            bad_txt, bad_csv], cwd=REPO, capture_output=True, text=True)
        t("read_serial.py replay of the corrupted log exits nonzero", r.returncode != 0,
          f"exit={r.returncode}")
        feats = batch_features(bad_csv)
        code, out, rows = run_infer([bad_txt])

    t("no crash; bad lines counted (malformed=2 bad_checksum=1 empty=1)",
      "malformed=2" in out and "bad_checksum=1" in out and "empty=1" in out,
      out.strip().splitlines()[-12] if out else "")
    t("infer_live exits nonzero on corruption, as read_serial does", code == 1,
      f"exit={code}")
    ok, detail = stream_features_match(rows, feats)
    t("features after skipping the bad frames equal batch on the same skips",
      ok, detail)
    t("the stance containing the bad frames is one frame shorter, not dropped",
      len(rows) == len(feats0) and int(rows[0]["end"]) == e0 - 1,
      f"stream first end={rows[0]['end'] if rows else None} original={e0}")
    return t.result()


# ---------------------------------------------------------------------------
# 4. All-zero frames: the (nan, nan) CoP path
# ---------------------------------------------------------------------------
def check_all_zero_frames():
    t = Tally()

    # Two runs close enough to merge, with all-zero frames in the gap. The
    # merged stance therefore CONTAINS all-zero frames, which is the only way
    # a stance can: inside a run every frame is >= T_OFF > 0.
    gap = D.GAP_MERGE - 2
    lines = pulse_lines([(40, gap), (40, 60)])
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "zeros.txt")
        csvp = os.path.join(tmp, "zeros.csv")
        write_lines(txt, lines)
        subprocess.run([sys.executable, "-m", "insole.read_serial", txt, csvp],
                       cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        feats = batch_features(csvp)
        code, out, rows = run_infer([txt])

    t("all-zero frames inside a merged stance: counted", f"all-zero frames={gap + 60}" in out,
      [l for l in out.splitlines() if "all-zero" in l][:1])
    t("one merged stance, CoP features finite (nan CoP frames skipped by cop_features)",
      len(rows) == 1 and all(np.isfinite(float(rows[0][c]))
                             for c in ("cop_path_len", "cop_displacement")),
      f"n={len(rows)} cop={[rows[0][c] for c in ('cop_path_len', 'cop_displacement')] if rows else None}")
    ok, detail = stream_features_match(rows, feats)
    t("equals batch through the same nan path", ok and code == 0, detail)

    # The whole stream all-zero: nothing detected, nothing crashes.
    zeros = pulse_lines([(0, 300)])
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "allzero.txt")
        write_lines(txt, zeros)
        code, out, rows = run_infer([txt])
    t("300 all-zero frames: 0 stances, counter=300, exit 0",
      len(rows) == 0 and "all-zero frames=300" in out and code == 0, f"exit={code}")
    return t.result()


# ---------------------------------------------------------------------------
# 5. A stance exceeding MAX_DURATION
# ---------------------------------------------------------------------------
def check_max_duration():
    t = Tally()
    long_run = D.MAX_DURATION + 50

    # A run longer than the ceiling, then a real stance after force drops.
    lines = pulse_lines([(long_run, 30), (40, 30)])
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "long.txt")
        csvp = os.path.join(tmp, "long.csv")
        write_lines(txt, lines)
        subprocess.run([sys.executable, "-m", "insole.read_serial", txt, csvp],
                       cwd=REPO, check=True, stdout=subprocess.DEVNULL)
        feats = batch_features(csvp)
        code, out, rows = run_infer([txt], quiet=False)

    t("batch: the long run is discarded, the later stance kept",
      len(feats) == 1 and int(feats.iloc[0]["start"]) == long_run + 30,
      f"batch spans={[(int(a), int(b)) for a, b in feats[['start', 'end']].to_numpy()]}")
    t("stream: same stances as batch", stream_features_match(rows, feats)[0],
      stream_features_match(rows, feats)[1])
    t("discard reported live, naming the ceiling",
      any(l.startswith("discard") and f"MAX_DURATION={D.MAX_DURATION}" in l
          for l in out.splitlines()),
      [l for l in out.splitlines() if l.startswith("discard")][:1])
    t("summary counter: discarded>MAX_DURATION=1 beside completed=1",
      f"stances completed=1 discarded>MAX_DURATION({D.MAX_DURATION})=1" in out,
      [l for l in out.splitlines() if l.startswith("stances completed")][:1])

    # The latch: after a discard, staying between T_OFF and T_ON must NOT
    # re-enter. Force has to fall below T_OFF first.
    hover = (D.T_OFF + D.T_ON) // 2 // 6
    seq = [400] * long_run + [hover] * 50 + [0] * 5 + [400] * 40 + [0] * 5
    got, tr = stances_via_tracker(np.array(seq) * 6)
    t("armed latch: no re-entry until force < T_OFF after a discard",
      got == [(long_run + 55, long_run + 94)] and tr.n_discarded_max == 1,
      f"got={got} discarded={tr.n_discarded_max}")

    # Real data: the standing capture is one 6000-frame run, discarded once.
    code, out, rows = run_infer([os.path.join(REAL, "stand_02.csv")])
    t("data/real/stand_02.csv: 0 stances, discarded>MAX_DURATION=1",
      len(rows) == 0 and f"discarded>MAX_DURATION({D.MAX_DURATION})=1" in out,
      [l for l in out.splitlines() if l.startswith("stances completed")][:1])
    return t.result()


# ---------------------------------------------------------------------------
# 6. Source swap: file vs serial vs ble on identical bytes
# ---------------------------------------------------------------------------
def check_source_swap():
    t = Tally()
    txt = os.path.join(DATA_SIM, "sim_fast.txt")
    lines = [ln for ln in RS.file_lines(txt)]

    code_f, out_f, rows_f = run_infer([txt])
    with fake_live_source(lines, "serial"):
        code_s, out_s, rows_s = run_infer(["--source", "serial", "--duration", "60"])
    with fake_live_source(lines, "ble"):
        code_b, out_b, rows_b = run_infer(["--source", "ble", "--duration", "60"])

    def strip(o):
        # The lines that carry results; transport name and clocks excluded.
        heads = ("stances completed", "frames extrapolating", "predictions:",
                 "model ", "detector ", "gain match")
        return "\n".join(l for l in o.splitlines() if l.startswith(heads))

    t("file vs serial: identical stance rows", rows_f == rows_s,
      f"n={len(rows_f)}/{len(rows_s)}")
    t("file vs ble: identical stance rows", rows_f == rows_b,
      f"n={len(rows_f)}/{len(rows_b)}")
    t("file vs serial: identical result lines (counters, predictions, model, gain)",
      strip(out_f) == strip(out_s) and strip(out_f).count("\n") >= 4,
      f"{strip(out_f).count(chr(10)) + 1} lines compared")
    t("all three exit 0", code_f == code_s == code_b == 0,
      f"file={code_f} serial={code_s} ble={code_b}")
    t("serial summary line names its transport", "source=serial" in out_s)
    return t.result()


# ---------------------------------------------------------------------------
# 7. No state leaks between consecutive stances
# ---------------------------------------------------------------------------
def check_no_state_leak():
    t = Tally()

    # (a) K identical pulses -> K identical stances at the expected offsets.
    K, on, off = 8, 40, 30
    total = np.array([x for _ in range(K) for x in [2400] * on + [0] * off])
    got, tr = stances_via_tracker(total)
    want = [(i * (on + off), i * (on + off) + on - 1) for i in range(K)]
    t("8 identical pulses -> 8 identical stances, exact spans", got == want,
      f"got={got[:3]}...")
    t("tracker idle and empty after the last stance",
      tr.earliest_live_index() is None and tr.pending is None and not tr.in_stance)

    # (b) features of stance k do not depend on stance k-1: the second stance
    # of a two-stance stream equals the only stance of the same pulse alone.
    lines2 = pulse_lines([(40, 30), (40, 30)], per_sensor=400)
    lines1 = pulse_lines([(40, 30)], per_sensor=400, seq0=70, ts0=70 * RS.PERIOD_US)
    with tempfile.TemporaryDirectory() as tmp:
        p2, p1 = os.path.join(tmp, "two.txt"), os.path.join(tmp, "one.txt")
        write_lines(p2, lines2)
        write_lines(p1, lines1)
        _, _, rows2 = run_infer([p2])
        _, _, rows1 = run_infer([p1])
    same = (len(rows2) == 2 and len(rows1) == 1
            and all(same_values(rows2[1][c], rows1[0][c]) for c in FEATS))
    t("second stance's features equal the same pulse fed alone", same,
      f"n2={len(rows2)} n1={len(rows1)}")

    # (c) a discard does not poison the next stance's boundaries or features.
    long_run = D.MAX_DURATION + 20
    linesA = pulse_lines([(long_run, 30), (40, 30)], per_sensor=400)
    linesB = pulse_lines([(40, 30)], per_sensor=400,
                         seq0=long_run + 30, ts0=(long_run + 30) * RS.PERIOD_US)
    with tempfile.TemporaryDirectory() as tmp:
        pa, pb = os.path.join(tmp, "a.txt"), os.path.join(tmp, "b.txt")
        write_lines(pa, linesA)
        write_lines(pb, linesB)
        _, _, rowsA = run_infer([pa])
        _, _, rowsB = run_infer([pb])
    same = (len(rowsA) == 1 and len(rowsB) == 1
            and all(same_values(rowsA[0][c], rowsB[0][c]) for c in FEATS))
    t("stance after a discarded run equals the same stance with no discard before it",
      same, f"nA={len(rowsA)} nB={len(rowsB)}")

    # (d) a fresh tracker per run: counters start at zero.
    fresh = D.StanceTracker()
    t("fresh tracker starts armed, idle, zero counters",
      fresh.armed and not fresh.in_stance and fresh.n_stances == 0
      and fresh.n_discarded_max == 0 and fresh.pending is None)
    return t.result()


# ---------------------------------------------------------------------------
# 8. The seam: call-time defaults, --port/--duration, the stall watchdog
# ---------------------------------------------------------------------------
def check_seam_and_watchdog():
    t = Tally()

    for fn in (RS.serial_lines, RS.ble_lines, RS.make_source):
        sig = inspect.signature(fn)
        bound = [n for n, p in sig.parameters.items()
                 if p.default is not inspect.Parameter.empty and p.default is not None]
        t(f"{fn.__name__}: no default argument binds a module constant at import",
          not bound, f"import-bound defaults: {bound}")

    # A fake pyserial: readline yields frames for a while, then silence.
    lines = pulse_lines([(20, 20)])

    class FakeSerial:
        instances = []

        def __init__(self, port, baud, timeout=1):
            self.port, self.baud, self.timeout = port, baud, timeout
            self.t_open = time.time()
            self.i = 0
            self.closed = False
            FakeSerial.instances.append(self)

        def reset_input_buffer(self):
            pass

        def readline(self):
            # serial_lines() discards every line for 1.0 s after open (boot
            # chatter). Act like a board: chatter during that second, frames
            # after it, then silence.
            if time.time() - self.t_open < 1.2:
                return b"# boot\n"
            if self.i < len(lines):
                self.i += 1
                return (lines[self.i - 1] + "\n").encode()
            return b""

        def close(self):
            self.closed = True

    fake_mod = types.SimpleNamespace(Serial=FakeSerial)
    saved_mod = sys.modules.get("serial")
    saved = (RS.DURATION_S, RS.STALL_S, RS.PORT)
    sys.modules["serial"] = fake_mod
    try:
        RS.STALL_S = 0.3
        RS.DURATION_S = 30
        RS.PORT = "COMX"
        got, err = [], None
        try:
            for ln in RS.make_source("serial", port="COM42"):
                if ln.startswith("INS"):        # boot chatter after warm-up is
                    got.append(ln)              # yielded as status lines
        except RS.StallError as exc:
            err = str(exc)
        t("--port reaches pyserial (COM42, not the module default)",
          FakeSerial.instances[-1].port == "COM42", FakeSerial.instances[-1].port)
        t("all 40 frames delivered before the board went quiet", len(got) == 40, len(got))
        t("StallError after STALL_S of silence, naming the port",
          err is not None and "COM42" in err and "no data" in err, err)
        t("the port is closed after the stall", FakeSerial.instances[-1].closed)

        # The trap: setting the module attribute after import must take
        # effect on the next call. DURATION_S = 0 -> the window is over
        # before the first readline, so nothing is yielded and no stall fires.
        RS.DURATION_S = 0
        got = list(RS.make_source("serial"))
        t("read_serial.DURATION_S set after import changes the capture length",
          got == [] and FakeSerial.instances[-1].port == "COMX", f"got {len(got)} lines")
    finally:
        RS.DURATION_S, RS.STALL_S, RS.PORT = saved
        if saved_mod is None:
            sys.modules.pop("serial", None)
        else:
            sys.modules["serial"] = saved_mod

    # infer_live and read_serial.main both turn a stall into exit 1.
    lines = [ln for ln in RS.file_lines(os.path.join(DATA_SIM, "sim_walk.txt"))]
    with fake_live_source(lines, "serial", raise_after=1500):
        code, out, rows = run_infer(["--source", "serial", "--duration", "60"])
    t("infer_live: stall -> 'FAIL: stalled' and exit 1",
      code == 1 and "FAIL: stalled" in out, f"exit={code}")
    t("infer_live: stances completed before the stall are still reported",
      len(rows) > 0 and "stances completed=" in out, f"n={len(rows)}")

    with fake_live_source(lines, "serial", raise_after=1500), \
            tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = RS.main(["--source", "serial", os.path.join(tmp, "o.csv")])
    t("read_serial.main: stall -> 'FAIL: stalled' and exit 1",
      code == 1 and "FAIL: stalled" in buf.getvalue(), f"exit={code}")
    return t.result()


# ---------------------------------------------------------------------------
# 9. Calibration anchor and the persisted model
# ---------------------------------------------------------------------------
def check_anchor_and_model():
    t = Tally()

    # CAL_MAX_COUNTS is a number about cal_data/, so recompute it from there.
    raw_max = 0
    for path in glob.glob(os.path.join(CAL_DATA, "cal_s*_t*.csv")):
        df = pd.read_csv(path)
        raw_max = max(raw_max, int(df[D.SENSOR_COLS].to_numpy().max()))
    t("calibration.CAL_MAX_COUNTS equals the highest raw count in cal_data/",
      raw_max == C.CAL_MAX_COUNTS, f"cal_data max={raw_max} constant={C.CAL_MAX_COUNTS}")
    man = pd.read_csv(os.path.join(CAL_DATA, "calibration_manifest.csv"), header=None)
    t("manifest count_mean maximum lies below it",
      man.iloc[:, 8].max() < C.CAL_MAX_COUNTS, f"count_mean max={man.iloc[:, 8].max()}")

    # save/load round trip, both kinds, predictions identical.
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 0.2, (60, 2)) for m in ([0, 0], [1, 1], [0, 1])])
    y = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 60)
    with tempfile.TemporaryDirectory() as tmp:
        for kind, fit in (("lda", fit_lda), ("qda", fit_qda)):
            m = fit(X, y)
            path = os.path.join(tmp, f"{kind}.json")
            save_model(m, path, meta={"features": ["f0", "f1"]})
            m2 = load_model(path)
            t(f"{kind}: save/load round trip predicts identically",
              np.array_equal(predict(m, X), predict(m2, X))
              and m2["meta"]["features"] == ["f0", "f1"])

    # The committed deployment model, if present, reproduces the frame fit.
    mpath = os.path.join(MODELS, "model_lda.json")
    fpath = os.path.join(DATA_SIM, "features_sessions.csv")
    if os.path.exists(mpath) and os.path.exists(fpath):
        m = load_model(mpath)
        frame = pd.read_csv(fpath)
        Xf = frame[m["meta"]["features"]].to_numpy(float)
        yf = frame["label"].to_numpy()
        refit = fit_lda(Xf, yf)
        t("model_lda.json predicts as a fresh fit_lda on features_sessions.csv",
          np.array_equal(predict(m, Xf), predict(refit, Xf))
          and m["meta"]["n_rows"] == len(frame),
          f"n_rows={m['meta']['n_rows']}")
    else:
        t("models/model_lda.json / data/sim/features_sessions.csv present", False,
          "run: python scripts/bakeoff.py && python scripts/fit_model.py")
    return t.result()



def test_equivalence():
    p, f = check_equivalence()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_read_boundary():
    p, f = check_read_boundary()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_malformed_mid_stance():
    p, f = check_malformed_mid_stance()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_all_zero_frames():
    p, f = check_all_zero_frames()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_max_duration():
    p, f = check_max_duration()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_source_swap():
    p, f = check_source_swap()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_no_state_leak():
    p, f = check_no_state_leak()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_seam_and_watchdog():
    p, f = check_seam_and_watchdog()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


def test_anchor_and_model():
    p, f = check_anchor_and_model()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"

SUITES = [check_equivalence, check_read_boundary, check_malformed_mid_stance,
          check_all_zero_frames, check_max_duration, check_source_swap,
          check_no_state_leak, check_seam_and_watchdog, check_anchor_and_model]

if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in SUITES:
        print(f"--- {suite.__name__.replace('check_', 'test_', 1)} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()
    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
