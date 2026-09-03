"""Fault-injection tests: gait_gen's three fault modes through the logger
(read_serial.main) and the streamer (infer_live.main). Run from the repo root:

    python test_faults.py

Also collected by pytest. No hardware: the live source is replaced by a fake
yielding the same lines a file would, as test_infer_live.py does.

Predictions, written before the first run
-----------------------------------------
faults off   gait_lines() with every fault argument at its default is
             byte-identical to the pre-fault generator: the twelve
             make_sessions.py sessions rebuild to the SHA-256s pinned below,
             which were hashed from files the pre-fault code wrote, and the
             fault RNG is never constructed.
drop 1 %     the board emits 6000 frames; the host sees valid + lost == 6000,
             bad_checksum == 0, resets == 0, seq_breaks > 0, and exits 1
             (file/serial loss semantics, unchanged). The stance count equals
             the fault-free count -- one missing frame can neither split nor
             create a hysteresis stance (troughs sit far above T_OFF, stances
             far above MIN_DURATION); +/-2 is the stated tolerance so a
             boundary coincidence is not read as a regression. Every flagged
             stance's gap_frames equals its seq span minus its frame span.
corrupt 1 %  valid + bad_checksum == 6000 with lost == 0 and seq_breaks == 0:
             a corrupt frame consumed its sequence slot and is counted once,
             under bad_checksum. Exit 1 (corruption). Stance count within +/-2.
reset        resets == 1; seq_breaks == lost == timing_breaks == 0;
             valid == 6000; malformed == 2 (the ROM boot line, plus one
             status line). The stance in progress at the reset is absent
             and its remainder is not reported as a new one, so the streamer
             reports one stance fewer than fault-free; stances after it carry
             epoch 1. The running-median dt is re-seeded after the reset.
all three    the logger and the streamer report identical counters on the
             same stream, over a file and over the fake serial source, and
             valid + lost + bad_checksum == 6000 minus the frames lost at the
             reset boundary, which no host can see.
"""

import contextlib
import csv
import hashlib
import io
import os
import random
import subprocess
import sys
import tempfile
import types

import pandas as pd

import detector as D
import gait_gen as G
import infer_live as IL
import read_serial as RS
from make_sessions import CLASSES, JITTER, SEED_BASE, SESSIONS_PER_CLASS, session_name

REPO = os.path.dirname(os.path.abspath(__file__))
N_FRAMES = G.DURATION_S * G.SAMPLE_HZ            # frames the board emits in 60 s
KEYS = ("valid", "malformed", "empty", "bad_checksum", "seq_breaks", "lost",
        "timing_breaks", "resets", "status")

# SHA-256 of each make_sessions.py session as written by the generator BEFORE
# the fault modes existed (same fixed seeds). Byte identity with faults off.
PINNED = {
    "sim_walk_00.txt":    "40d9c1c913b1cc958c01e124e8c81ee074fb07177af49fff02708599e3a8883a",
    "sim_walk_01.txt":    "39a32112f2b8bd769423b7b949653b07fbb15e0aec98765a76f8f25ef2aba1ed",
    "sim_walk_02.txt":    "a08d559a56659b57448ed814107c302a03ffb896a97e073de227190dc9d7dba9",
    "sim_walk_03.txt":    "aba491c478ae3b3edcf2fb399e00d59918a148c6f0be4fa9d1450073b0e112ee",
    "sim_fast_00.txt":    "8c1910a50f28a23ab6733f842f58f48761c6c41af4667d9f29a476d60b10f6c5",
    "sim_fast_01.txt":    "aecd24e8c5e47791685edf609ca00d1522495f1038c52ae7aa3616490834fc10",
    "sim_fast_02.txt":    "c7479617a1a106ae4fbd64c54a732d54d267d04d04f608588da5bea17529a635",
    "sim_fast_03.txt":    "01372124dc9f70c493145dbc7e87a340782fd46bc414b997038b8fab9d322847",
    "sim_shuffle_00.txt": "bb0f90d221c776cae0cb45e264d10839638a3bb1abac673c7467d016d980d7a9",
    "sim_shuffle_01.txt": "e7d061b1d1cc3f607ee129f863fb6d4bf1e271a2a84854ae93e8cfd7be14db6c",
    "sim_shuffle_02.txt": "cc48f52356439e845f78a7eac69e8b9172d970c00797090d0d30b315a5aea8e2",
    "sim_shuffle_03.txt": "df8445845669ab0c6f3ddca2c40acb9c3983c0e664e629a09b6073e73059a89c",
}


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
def gen(noise_seed, **kw):
    """One 60 s walk stream, noise seeded so the fault-free twin is exact."""
    random.seed(noise_seed)
    return list(G.gait_lines(G.DURATION_S, **kw))


def write_lines(path, lines):
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def counters_of(out):
    """The counter dict from a read_serial / infer_live summary line."""
    for line in out.splitlines():
        if line.startswith("source="):
            d = {}
            for kv in line.split():
                key, val = kv.split("=", 1)
                d[key] = int(val) if val.lstrip("-").isdigit() else val
            return d
    raise AssertionError("no summary line in output:\n" + out)


def run_logger(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = RS.main(list(argv))
    return code, buf.getvalue()


def run_streamer(argv, quiet=True):
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


@contextlib.contextmanager
def fake_live_source(lines, source="serial"):
    """Swap read_serial's live transport for a generator over `lines`."""
    name = "serial_lines" if source == "serial" else "ble_lines"
    saved = getattr(RS, name)

    def fake(*_a, **_kw):
        yield from lines

    setattr(RS, name, fake)
    try:
        yield
    finally:
        setattr(RS, name, saved)


def same_counters(a, b):
    return {k: a[k] for k in KEYS} == {k: b[k] for k in KEYS}


def n_stances_fault_free(noise_seed):
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "clean.txt")
        write_lines(txt, gen(noise_seed))
        _, _, rows = run_streamer([txt])
    return len(rows)


# ---------------------------------------------------------------------------
# 1. Faults off: byte identity with the pre-fault generator
# ---------------------------------------------------------------------------
def test_faults_off_identity():
    t = Tally()

    for name, kwargs in G.STREAMS:
        random.seed(11)
        a = list(G.gait_lines(5, **kwargs))
        random.seed(11)
        b = list(G.gait_lines(5, drop_rate=0.0, corrupt_rate=0.0, reset_at_s=None,
                              fault_seed=99, **kwargs))
        t(f"{name}: explicit zero faults == defaults, byte for byte", a == b,
          f"{len(a)} lines")

    class Boom(random.Random):
        def __init__(self, *_a, **_k):
            raise AssertionError("fault RNG constructed with every fault off")

    saved = G.random
    G.random = types.SimpleNamespace(gauss=random.gauss, Random=Boom)
    try:
        random.seed(11)
        n = sum(1 for _ in G.gait_lines(2))
        ok = n == 200
    except AssertionError as exc:
        ok, n = False, str(exc)
    finally:
        G.random = saved
    t("faults off: the fault RNG is never constructed", ok, f"lines={n}")

    for label, (mode, base_cycle) in CLASSES.items():
        for idx in range(SESSIONS_PER_CLASS):
            name = session_name(label, idx)
            random.seed(SEED_BASE[label] + idx)
            text = "\n".join(G.gait_lines(60, mode=mode, cycle_s=base_cycle * JITTER[idx])) + "\n"
            digest = hashlib.sha256(text.encode()).hexdigest()
            t(f"{name} rebuilds to its pre-fault SHA-256", digest == PINNED[name],
              digest[:12])
    return t.result()


# ---------------------------------------------------------------------------
# 2. drop_rate: frames omitted -> sequence gaps
# ---------------------------------------------------------------------------
def test_drop_mode():
    t = Tally()
    lines = gen(1, drop_rate=0.01, fault_seed=1)
    n_dropped = N_FRAMES - sum(1 for ln in lines if ln.startswith("INS"))
    t("1 % drop rate omitted frames", 30 <= n_dropped <= 90, f"dropped={n_dropped}")

    with tempfile.TemporaryDirectory() as tmp:
        txt, csvp = os.path.join(tmp, "drop.txt"), os.path.join(tmp, "drop.csv")
        write_lines(txt, lines)
        code_l, out_l = run_logger([txt, csvp])
        c = counters_of(out_l)
        t("logger: valid + lost + bad_checksum == frames the board emitted",
          c["valid"] + c["lost"] + c["bad_checksum"] == N_FRAMES,
          f"{c['valid']} + {c['lost']} + {c['bad_checksum']} vs {N_FRAMES}")
        t("logger: lost == frames dropped; bad_checksum == resets == 0",
          c["lost"] == n_dropped and c["bad_checksum"] == 0 and c["resets"] == 0,
          f"lost={c['lost']} dropped={n_dropped}")
        t("logger: seq gaps reported and the file replay exits 1 (loss semantics unchanged)",
          c["seq_breaks"] > 0 and code_l == 1 and c["timing_breaks"] == 0,
          f"seq_breaks={c['seq_breaks']} exit={code_l}")
        seqs = pd.read_csv(csvp)["seq"].to_numpy()

        code_s, out_s, rows = run_streamer([txt])
    cs = counters_of(out_s)
    t("streamer: identical counters to the logger, same exit code",
      same_counters(c, cs) and code_s == code_l, f"exit={code_s}")

    n0 = n_stances_fault_free(1)
    t(f"stance count under 1 % drop within +/-2 of fault-free ({n0})",
      abs(len(rows) - n0) <= 2, f"got={len(rows)}")

    gaps = [int(r["gap_frames"]) for r in rows]
    independent = [int(seqs[int(r["end"])] - seqs[int(r["start"])]) - (int(r["end"]) - int(r["start"]))
                   for r in rows]
    t("gap_frames per stance == seq span minus frame span (independent of the validator)",
      gaps == independent, f"flagged={sum(g > 0 for g in gaps)}/{len(rows)}")
    t("some stances flagged, gaps inside stances never exceed total loss",
      any(g > 0 for g in gaps) and sum(gaps) <= n_dropped,
      f"sum={sum(gaps)} lost={n_dropped}")
    t("summary line counts the flagged stances",
      f"stances_with_gaps={sum(g > 0 for g in gaps)}" in out_s)
    return t.result()


# ---------------------------------------------------------------------------
# 3. corrupt_rate: wrong checksums
# ---------------------------------------------------------------------------
def test_corrupt_mode():
    t = Tally()
    lines = gen(2, corrupt_rate=0.01, fault_seed=2)
    n_corrupt = sum(1 for ln in lines if G.parse_frame(ln)[0] == "bad_checksum")
    t("1 % corrupt rate produced bad checksums, no other rejection",
      30 <= n_corrupt <= 90 and sum(1 for ln in lines if G.parse_frame(ln)[0] == "ok")
      == N_FRAMES - n_corrupt, f"corrupt={n_corrupt}")

    with tempfile.TemporaryDirectory() as tmp:
        txt, csvp = os.path.join(tmp, "corrupt.txt"), os.path.join(tmp, "corrupt.csv")
        write_lines(txt, lines)
        code_l, out_l = run_logger([txt, csvp])
        code_s, out_s, rows = run_streamer([txt])
    c, cs = counters_of(out_l), counters_of(out_s)
    t("logger: valid + bad_checksum == frames emitted, lost == 0",
      c["valid"] + c["bad_checksum"] == N_FRAMES and c["lost"] == 0,
      f"valid={c['valid']} bad_checksum={c['bad_checksum']} lost={c['lost']}")
    t("logger: a corrupt frame is counted once -- no seq break, no timing break",
      c["seq_breaks"] == 0 and c["timing_breaks"] == 0 and c["bad_checksum"] == n_corrupt,
      f"seq_breaks={c['seq_breaks']} timing_breaks={c['timing_breaks']}")
    t("logger: corruption exits 1 on every transport (unchanged)",
      code_l == 1 and "corrupted" in out_l)
    t("streamer: identical counters to the logger, same exit code",
      same_counters(c, cs) and code_s == code_l, f"exit={code_s}")
    n0 = n_stances_fault_free(2)
    t(f"stance count under 1 % corruption within +/-2 of fault-free ({n0})",
      abs(len(rows) - n0) <= 2, f"got={len(rows)}")
    t("corrupt frames inside stances are flagged as gaps, none imputed",
      any(int(r["gap_frames"]) > 0 for r in rows)
      and all(int(r["n_frames"]) == int(r["end"]) - int(r["start"]) + 1 for r in rows),
      f"flagged={sum(int(r['gap_frames']) > 0 for r in rows)}/{len(rows)}")
    return t.result()


# ---------------------------------------------------------------------------
# 4. reset_at_s: the board reboots mid-capture
# ---------------------------------------------------------------------------
def test_reset_mode():
    t = Tally()
    reset_s = 20.3                                  # inside walk stance 20 (20.0-20.62 s)
    k_reset = int(reset_s * G.SAMPLE_HZ)
    lines = gen(3, reset_at_s=reset_s, fault_seed=3)
    t("boot lines precede the restarted frame; SEQ and TS_US restart at 0",
      lines[k_reset:k_reset + 2] == G.BOOT_LINES and lines[k_reset + 2].startswith("INS,0,0,")
      and lines[k_reset - 1].startswith(f"INS,{k_reset - 1},{(k_reset - 1) * G.PERIOD_US},"),
      lines[k_reset + 2][:20])

    with tempfile.TemporaryDirectory() as tmp:
        txt, csvp = os.path.join(tmp, "reset.txt"), os.path.join(tmp, "reset.csv")
        write_lines(txt, lines)
        code_l, out_l = run_logger([txt, csvp])
        code_s, out_s, rows = run_streamer([txt])
        code_v, out_v, _ = run_streamer([txt], quiet=False)
    c, cs = counters_of(out_l), counters_of(out_s)
    t("logger: resets == 1 and nothing else moved (no seq break, loss or timing break)",
      c["resets"] == 1 and c["seq_breaks"] == 0 and c["lost"] == 0
      and c["timing_breaks"] == 0 and c["valid"] == N_FRAMES,
      f"{ {k: c[k] for k in KEYS} }")
    t("logger: the ROM boot line is malformed (1), the sketch banner a status line (1)",
      c["malformed"] == 1 and c["status"] == 1, f"malformed={c['malformed']} status={c['status']}")
    t("logger: a mid-capture reset exits 1 and says so",
      code_l == 1 and "reset" in out_l.lower(), f"exit={code_l}")
    t("streamer: identical counters to the logger, same exit code",
      same_counters(c, cs) and code_s == code_l, f"exit={code_s}")

    n0 = n_stances_fault_free(3)
    spans = any(int(r["start"]) < k_reset <= int(r["end"]) for r in rows)
    t(f"the stance spanning the reset is absent and not re-reported: {n0} - 1 stances",
      len(rows) == n0 - 1 and not spans, f"got={len(rows)} spans_reset={spans}")
    t("streamer reports the discard", "discarded@reset=1" in out_s
      and any(ln.startswith("discard") and "reset" in ln for ln in out_v.splitlines()))
    before = [r for r in rows if int(r["end"]) < k_reset]
    after = [r for r in rows if int(r["start"]) >= k_reset]
    t("stances before the reset carry epoch 0, after it epoch 1 with t_start_s restarted",
      all(r["epoch"] == "0" for r in before) and all(r["epoch"] == "1" for r in after)
      and after and float(after[0]["t_start_s"]) < 2.0,
      f"before={len(before)} after={len(after)} first_after_t={after[0]['t_start_s'] if after else None}")

    # The validator alone: the first post-reset frame itself lost.
    vals = [400] * 6
    v = RS.FrameValidator()
    for ln in (G.make_frame(10, 100000, vals), G.make_frame(11, 110000, vals),
               G.make_frame(1, 10000, vals), G.make_frame(2, 20000, vals)):
        v.feed(ln)
    t("validator: a reset whose first frame was lost is one reset, not loss",
      v.c["resets"] == 1 and v.c["lost"] == 0 and v.c["seq_breaks"] == 0
      and v.c["timing_breaks"] == 0 and v.c["valid"] == 4, str(v.c))

    # The running-median dt is re-seeded: 10 ms steps before the reset,
    # 20 ms steps after it, a pulse only after it. contact_time_s is
    # (end - start) * dt by the batch definition, so 39 frames at 20 ms.
    quiet, loud = [0] * 6, [400] * 6
    pre = [G.make_frame(k, k * 10000, quiet) for k in range(300)]
    post = list(G.BOOT_LINES) + [G.make_frame(k, k * 20000, loud if 50 <= k < 90 else quiet)
                                 for k in range(300)]
    with tempfile.TemporaryDirectory() as tmp:
        txt = os.path.join(tmp, "dt.txt")
        write_lines(txt, pre + post)
        _, out_dt, rows_dt = run_streamer([txt])
    ct = float(rows_dt[0]["contact_time_s"]) if rows_dt else float("nan")
    t("running-median dt re-seeded after the reset (20 ms steps -> 0.78 s, not 0.39 s)",
      len(rows_dt) == 1 and abs(ct - 39 * 0.02) < 1e-9, f"contact_time_s={ct}")
    return t.result()


# ---------------------------------------------------------------------------
# 5. All three at once: logger == streamer, file == live source
# ---------------------------------------------------------------------------
def test_logger_streamer_consistency():
    t = Tally()
    reset_s = 30.5
    k_reset = int(reset_s * G.SAMPLE_HZ)
    lines = gen(5, drop_rate=0.02, corrupt_rate=0.01, reset_at_s=reset_s, fault_seed=5)

    # Board frame index of every emitted frame line, valid or corrupt.
    emitted = []
    after_boot = False
    for ln in lines:
        if ln == G.BOOT_LINES[-1]:
            after_boot = True
        if ln.startswith("INS"):
            seq = int(ln.split(",")[1])
            emitted.append(seq + k_reset if after_boot else seq)
    last_pre = max(k for k in emitted if k < k_reset)
    first_post = min(k for k in emitted if k >= k_reset)
    invisible = (k_reset - 1 - last_pre) + (first_post - k_reset)

    with tempfile.TemporaryDirectory() as tmp:
        txt, csvp = os.path.join(tmp, "all.txt"), os.path.join(tmp, "all.csv")
        write_lines(txt, lines)
        code_lf, out_lf = run_logger([txt, csvp])
        code_sf, out_sf, rows_f = run_streamer([txt])
        with fake_live_source(lines, "serial"):
            code_ls, out_ls = run_logger(["--source", "serial", os.path.join(tmp, "live.csv")])
        with fake_live_source(lines, "serial"):
            code_ss, out_ss, rows_s = run_streamer(["--source", "serial", "--duration", "60"])

    cl, cs, cls, css = (counters_of(o) for o in (out_lf, out_sf, out_ls, out_ss))
    t("logger == streamer counters over the file", same_counters(cl, cs),
      str({k: cl[k] for k in KEYS}))
    t("logger == streamer counters over the fake serial source", same_counters(cls, css))
    t("file == serial for both consumers", same_counters(cl, cls) and same_counters(cs, css))
    t("all four exit 1", code_lf == code_sf == code_ls == code_ss == 1,
      f"{code_lf} {code_sf} {code_ls} {code_ss}")
    t("valid + lost + bad_checksum == 6000 minus frames lost at the reset boundary",
      cl["valid"] + cl["lost"] + cl["bad_checksum"] == N_FRAMES - invisible,
      f"{cl['valid']} + {cl['lost']} + {cl['bad_checksum']} = "
      f"{cl['valid'] + cl['lost'] + cl['bad_checksum']}, invisible at boundary = {invisible}")
    t("every mode registered: resets == 1, bad_checksum > 0, lost > 0",
      cl["resets"] == 1 and cl["bad_checksum"] > 0 and cl["lost"] > 0)
    t("streamer rows identical over file and serial", rows_f == rows_s,
      f"n={len(rows_f)}/{len(rows_s)}")
    return t.result()


# ---------------------------------------------------------------------------
# 6. The CLI: python gait_gen.py --out ... with the fault flags
# ---------------------------------------------------------------------------
def test_cli():
    t = Tally()
    py = sys.executable
    with tempfile.TemporaryDirectory() as tmp:
        txt, csvp = os.path.join(tmp, "cli.txt"), os.path.join(tmp, "cli.csv")
        r = subprocess.run([py, os.path.join(REPO, "gait_gen.py"), "--out", txt,
                            "--duration", "10", "--drop-rate", "0.01",
                            "--corrupt-rate", "0.01", "--reset-at", "5",
                            "--fault-seed", "5", "--noise-seed", "5"],
                           cwd=REPO, capture_output=True, text=True)
        random.seed(5)
        want = list(G.gait_lines(10, drop_rate=0.01, corrupt_rate=0.01,
                                 reset_at_s=5, fault_seed=5))
        with open(txt) as f:
            got = f.read().splitlines()
        t("CLI stream equals the in-process generator with the same seeds",
          r.returncode == 0 and got == want, r.stdout.strip())

        r2 = subprocess.run([py, os.path.join(REPO, "read_serial.py"), txt, csvp],
                            cwd=REPO, capture_output=True, text=True)
        t("read_serial.py on the CLI stream: resets=1 reported, exit 1",
          r2.returncode == 1 and "resets=1" in r2.stdout, r2.stdout.strip().splitlines()[-2:])

        r3 = subprocess.run([py, os.path.join(REPO, "gait_gen.py"), "--drop-rate", "0.1"],
                            cwd=REPO, capture_output=True, text=True)
        t("fault flags without --out are refused", r3.returncode != 0 and "--out" in r3.stderr)

        r4 = subprocess.run([py, os.path.join(REPO, "gait_gen.py")],
                            cwd=tmp, capture_output=True, text=True)
        wrote = sorted(f for f in os.listdir(tmp) if f.startswith("sim_"))
        t("no arguments still writes the five regression streams",
          r4.returncode == 0 and wrote == sorted(n for n, _ in G.STREAMS), str(wrote))
    return t.result()


SUITES = [test_faults_off_identity, test_drop_mode, test_corrupt_mode,
          test_reset_mode, test_logger_streamer_consistency, test_cli]

if __name__ == "__main__":
    total_pass = total_fail = 0
    for suite in SUITES:
        print(f"--- {suite.__name__} ---")
        p, f = suite()
        total_pass, total_fail = total_pass + p, total_fail + f
        print()
    print(f"{total_pass} passed, {total_fail} failed")
    if total_fail:
        sys.exit(1)
