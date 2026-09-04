"""Tests for the real-data training path: insole/representations.py,
insole/splits.py and scripts/train_real.py. Run from the repo root:

    python tests/test_train_real.py

Predictions, written before the first run
-----------------------------------------
splits       the time-blocked split of the 113 `_02` stances has no train/test
             overlap, every test stance starts after every training stance
             of its class, and the sizes are walk 21/14, fast 29/19,
             shuffle 18/12 (n_test 45); with guard=1 training loses one
             stance per class and the test side is unchanged. Contiguous-block
             folds test every stance exactly once, and each held-out block is
             a contiguous run in onset order. The random split keeps the
             per-class sizes and is a permutation of the same indices.
loso         on the 224 stances of the `_02` and `_03` sets,
             leave_one_session_out gives two folds, every stance is tested
             exactly once, no session sits on both sides of a fold, fold 0
             tests the `_02` set (fast 48, shuffle 30, walk 35) and fold 1
             the `_03` set (fast 45, shuffle 34, walk 32). The time-blocked
             split groups per (class, session), so on the same frame its
             sizes are the per-session sizes summed: fast 56/37, shuffle
             38/26, walk 40/27, and inside every (class, session) every test
             stance starts after every training stance.
identity     representation C with identity gains equals B exactly (bit for
             bit) on a sim fixture and on walk02; A is the frame unchanged.
sim numbers  the SHIPPED representation on the 12 simulated sessions
             reproduces data/sim/features_sessions.csv (the frame the bake-off
             and the persisted sim models were built from) row for row on the
             two CoP features, and the session-disjoint LDA on it scores what
             docs/bakeoff.md states for the shipped representation. Under raw
             counts (representation A) the same recipe scores 251/270 =
             0.9296, the figure the bake-off carried before stage 20 switched
             the shipped representation to conductance.
script       scripts/train_real.py runs end to end into a temporary output
             set, exits 0, writes the document with every pinned stance count
             (stand 0 / walk 35 / fast 48 / shuffle 30 for `_02`, 0 / 32 /
             45 / 34 for `_03`), names one headline cell, states the split
             the data on disk allows (leave-one-session-out once every class
             has two sessions, the within-session caveat otherwise), writes
             at least the two headline figures and both real models, and the
             models' meta lists exactly the sessions on disk and the shipped
             representation.

Each check_* function prints PASS/FAIL lines and returns (passed, failed);
the test_* wrapper of the same name asserts nothing failed, so pytest sees a
failure and the direct run keeps its counts.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from insole import detector as D
from insole.discriminant import fit_lda, predict
from insole.features import extract_features
from insole.make_sessions import CLASSES, SESSIONS_PER_CLASS, session_name
from insole.paths import DATA_REAL, DATA_SIM, REPO
from insole.representations import SHIPPED, features_under, identity_gains, transform_df
from insole.splits import (contiguous_block_folds, leave_one_session_out,
                           random_stance_splits, time_blocked_split)

# Session-disjoint LDA correct counts on the sim bake-off split, per
# representation: raw counts is the pre-stage-20 figure (docs/bakeoff.md,
# superseded section); the shipped one is what the doc now states.
BAKEOFF_LDA_CORRECT = {"raw": 251}

REAL_FILES_02 = [("walk", "walk02.csv"), ("fast", "fast02.csv"), ("shuffle", "shuffle02.csv")]
REAL_FILES_03 = [("walk", "walk_03.csv"), ("fast", "fast_03.csv"), ("shuffle", "shuffle_03.csv")]
# (activity, file, frames, stances) as the document's data table prints them.
PINNED_ROWS = [("stand", "stand_02.csv", 6000, 0), ("walk", "walk02.csv", 6000, 35),
               ("fast", "fast02.csv", 6000, 48), ("shuffle", "shuffle02.csv", 6000, 30),
               ("stand", "stand_03.csv", 6001, 0), ("walk", "walk_03.csv", 6001, 32),
               ("fast", "fast_03.csv", 6001, 45), ("shuffle", "shuffle_03.csv", 6001, 34)]
COP = ["cop_path_len", "cop_displacement"]
LABELS = ("walk", "fast", "shuffle")


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


def real_frame(files=REAL_FILES_02):
    parts = []
    for label, fname in files:
        df = pd.read_csv(os.path.join(DATA_REAL, fname))
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        st = D.merge_close(D.find_stances(total))
        f = extract_features(df, st, label)
        f["session"] = fname[:-4]
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def sessions_on_disk():
    """{label: [session stems]} for every training-grade moving capture in data/real."""
    out = {lab: [] for lab in LABELS}
    for fname in sorted(os.listdir(DATA_REAL)):
        if not fname.endswith(".csv") or "_01" in fname or "check_all" in fname:
            continue
        for lab in LABELS:
            if fname.startswith(lab):
                out[lab].append(fname[:-4])
    return out


# ---------------------------------------------------------------------------
# 1. Splits
# ---------------------------------------------------------------------------
def check_splits():
    t = Tally()
    fr = real_frame()
    tr, te, pc = time_blocked_split(fr, 0.6, 0)
    t("time-blocked: no overlap and every stance on one side",
      not (set(tr) & set(te)) and len(tr) + len(te) == len(fr), f"{len(tr)}+{len(te)}")
    t("time-blocked sizes walk 21/14, fast 29/19, shuffle 18/12",
      pc == {"fast": (29, 19, 0), "shuffle": (18, 12, 0), "walk": (21, 14, 0)}, str(pc))
    ordered = all(fr.loc[[i for i in tr if fr.loc[i, "label"] == lab], "start"].max()
                  < fr.loc[[i for i in te if fr.loc[i, "label"] == lab], "start"].min()
                  for lab in ("walk", "fast", "shuffle"))
    t("time-blocked: every test stance starts after every training stance of its class", ordered)
    trg, teg, pcg = time_blocked_split(fr, 0.6, 1)
    t("guard=1 drops the last training stance per class, test unchanged",
      list(teg) == list(te) and all(pcg[l][0] == pc[l][0] - 1 and pcg[l][2] == 1 for l in pc)
      and not (set(trg) & set(teg)), str(pcg))

    folds = list(contiguous_block_folds(fr, 5))
    tested = np.concatenate([te_ for _, te_ in folds])
    t("block CV: every stance tested exactly once, 5 folds",
      len(folds) == 5 and sorted(tested) == sorted(fr.index), f"tested={len(tested)}")
    contiguous = True
    for _, te_ in folds:
        for lab in ("walk", "fast", "shuffle"):
            cls_idx = fr[fr["label"] == lab].sort_values("start").index.to_numpy()
            pos = np.sort([np.flatnonzero(cls_idx == i)[0] for i in te_ if fr.loc[i, "label"] == lab])
            contiguous &= bool(np.all(np.diff(pos) == 1))
    t("block CV: each held-out block is contiguous in onset order", contiguous)
    disjoint = all(not (set(tr_) & set(te_)) for tr_, te_ in folds)
    t("block CV: train and test disjoint in every fold", disjoint)

    rnd = list(random_stance_splits(fr, 0.6, 3, seed=1))
    sizes_ok = all(sorted(np.concatenate([a, b])) == sorted(fr.index)
                   and all((fr.loc[a, "label"] == lab).sum() == pc[lab][0] for lab in pc)
                   for a, b in rnd)
    t("random split: same per-class sizes, a permutation of the same stances", sizes_ok)
    return t.result()


def test_splits():
    p, f = check_splits()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


# ---------------------------------------------------------------------------
# 2. Leave-one-session-out on two sessions per class
# ---------------------------------------------------------------------------
def check_loso():
    t = Tally()
    fr = real_frame(REAL_FILES_02 + REAL_FILES_03)
    t("two sessions per class: 224 moving stances", len(fr) == 224, f"n={len(fr)}")
    folds = list(leave_one_session_out(fr))
    t("LOSO: two folds", len(folds) == 2, f"folds={len(folds)}")
    tested = np.concatenate([te for _, te, _ in folds])
    t("LOSO: every stance tested exactly once", sorted(tested) == sorted(fr.index), f"tested={len(tested)}")
    for tr, te, held in folds:
        s_tr = set(fr.loc[tr, "session"])
        s_te = set(fr.loc[te, "session"])
        t(f"LOSO fold holding out {held}: no session on both sides, test = held out",
          not (s_tr & s_te) and s_te == set(held) and not (set(tr) & set(te)))
    pc = [{lab: int((fr.loc[te, "label"] == lab).sum()) for lab in LABELS} for _, te, _ in folds]
    t("LOSO fold 0 tests the _02 set: fast 48, shuffle 30, walk 35",
      pc[0] == {"fast": 48, "shuffle": 30, "walk": 35}, str(pc[0]))
    t("LOSO fold 1 tests the _03 set: fast 45, shuffle 34, walk 32",
      pc[1] == {"fast": 45, "shuffle": 34, "walk": 32}, str(pc[1]))

    tr, te, pcs = time_blocked_split(fr, 0.6, 0)
    t("time-blocked on two sessions: per-(class, session) sizes summed, fast 56/37, shuffle 38/26, walk 40/27",
      pcs == {"fast": (56, 37, 0), "shuffle": (38, 26, 0), "walk": (40, 27, 0)}, str(pcs))
    ordered = True
    for (lab, ses), g in fr.groupby(["label", "session"]):
        a = g.index[g.index.isin(tr)]
        b = g.index[g.index.isin(te)]
        ordered &= bool(len(a) and len(b) and fr.loc[a, "start"].max() < fr.loc[b, "start"].min())
    t("time-blocked: inside every (class, session) every test stance starts after every training stance", ordered)
    return t.result()


def test_loso():
    p, f = check_loso()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


# ---------------------------------------------------------------------------
# 3. Representations
# ---------------------------------------------------------------------------
def check_identity_gains():
    t = Tally()
    for path in (os.path.join(DATA_SIM, "sim_walk.csv"), os.path.join(DATA_REAL, "walk02.csv")):
        if not os.path.exists(path):
            subprocess.run([sys.executable, "-m", "insole.read_serial",
                            path[:-4] + ".txt", path], check=True, cwd=REPO,
                           stdout=subprocess.DEVNULL)
        df = pd.read_csv(path)
        total = df[D.SENSOR_COLS].sum(axis=1).to_numpy(dtype=float)
        st = D.merge_close(D.find_stances(total))
        a = transform_df(df, "raw")
        b = transform_df(df, "conductance")
        c = transform_df(df, "gain_matched", identity_gains())
        name = os.path.basename(path)
        t(f"{name}: A is the frame unchanged", a[D.SENSOR_COLS].equals(df[D.SENSOR_COLS].astype(float)))
        t(f"{name}: C with identity gains == B bit for bit",
          np.array_equal(b[D.SENSOR_COLS].to_numpy(), c[D.SENSOR_COLS].to_numpy()))
        t(f"{name}: a zero count has zero conductance, nothing is imputed",
          bool((b[D.SENSOR_COLS].to_numpy()[df[D.SENSOR_COLS].to_numpy() == 0] == 0).all()))
        fb = features_under(df, st, "x", "conductance")
        fc = features_under(df, st, "x", "gain_matched", identity_gains())
        t(f"{name}: features under C (identity) == features under B",
          fb[COP].equals(fc[COP]) and len(fb) == len(st), f"n={len(fb)}")
        g2 = {i: 2.0 for i in range(6)}
        fc2 = features_under(df, st, "x", "gain_matched", g2)
        t(f"{name}: a uniform gain leaves the CoP features unchanged (weights cancel)",
          np.allclose(fb[COP].to_numpy(), fc2[COP].to_numpy(), rtol=0, atol=1e-12))
    return t.result()


def test_identity_gains():
    p, f = check_identity_gains()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


# ---------------------------------------------------------------------------
# 4. Representation A reproduces the sim bake-off frame and number
# ---------------------------------------------------------------------------
def check_sim_reproduction():
    t = Tally()
    fpath = os.path.join(DATA_SIM, "features_sessions.csv")
    if not os.path.exists(fpath):
        t("data/sim/features_sessions.csv present", False, "run: python scripts/bakeoff.py")
        return t.result()
    ref = pd.read_csv(fpath)
    test_sessions = [session_name(lab, 3)[:-4] for lab in CLASSES]

    def sim_frame(rep):
        parts = []
        for label in CLASSES:
            for i in range(SESSIONS_PER_CLASS):
                stem = os.path.join(DATA_SIM, session_name(label, i)[:-4])
                df = pd.read_csv(stem + ".csv")
                total = df[D.SENSOR_COLS].sum(axis=1).to_numpy()
                st = D.merge_close(D.find_stances(total))
                f = features_under(df, st, label, rep)
                f["session"] = session_name(label, i)[:-4]
                parts.append(f)
        return pd.concat(parts, ignore_index=True)

    def lda_correct(frame):
        is_test = frame["session"].isin(test_sessions).to_numpy()
        m = fit_lda(frame.loc[~is_test, COP].to_numpy(float), frame.loc[~is_test, "label"].to_numpy())
        pred = predict(m, frame.loc[is_test, COP].to_numpy(float))
        return int((pred == frame.loc[is_test, "label"].to_numpy()).sum()), int(is_test.sum())

    mine = sim_frame(SHIPPED)
    t(f"shipped representation ({SHIPPED}) on the 12 sim sessions: same rows as features_sessions.csv",
      len(mine) == len(ref) == 1123 and list(mine["session"]) == list(ref["session"]),
      f"{len(mine)} vs {len(ref)}")
    t("shipped representation: CoP features equal the bake-off frame's",
      np.allclose(mine[COP].to_numpy(), ref[COP].to_numpy(), rtol=0, atol=1e-12))
    correct, n = lda_correct(mine)
    ref_correct, ref_n = lda_correct(ref)
    t("session-disjoint LDA on it equals the LDA on the persisted frame",
      correct == ref_correct and n == ref_n == 270, f"{correct}/{n} vs {ref_correct}/{ref_n}")
    raw_correct, raw_n = lda_correct(sim_frame("raw"))
    t("representation A (raw counts) still gives the pre-stage-20 bake-off figure 251/270 = 0.9296",
      raw_correct == BAKEOFF_LDA_CORRECT["raw"] and raw_n == 270, f"{raw_correct}/{raw_n}")
    return t.result()


def test_sim_reproduction():
    p, f = check_sim_reproduction()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


# ---------------------------------------------------------------------------
# 5. The script end to end
# ---------------------------------------------------------------------------
def check_script():
    t = Tally()
    on_disk = sessions_on_disk()
    multi = all(len(v) >= 2 for v in on_disk.values())
    with tempfile.TemporaryDirectory() as tmp:
        doc = os.path.join(tmp, "docs", "real_results.md")
        figs = os.path.join(tmp, "figs")
        models = os.path.join(tmp, "models")
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "train_real.py"),
                            "--doc", doc, "--fig-dir", figs, "--models-dir", models],
                           cwd=REPO, capture_output=True, text=True, encoding="utf-8")
        t("scripts/train_real.py exits 0", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-300:])
        text = open(doc, encoding="utf-8").read() if os.path.exists(doc) else ""
        t("document written with every pinned stance count",
          all(f"| {lab} | `{f}` | {n} | {k} |" in text for lab, f, n, k in PINNED_ROWS))
        if multi:
            t("document names one headline cell and the leave-one-session-out split (two sessions per class on disk)",
              "**(headline)**" in text and "leave-one-session-out" in text
              and "no per-session split exists" not in text and "NOT a per-session split" not in text)
        else:
            t("document names one headline cell and the within-session caveat (one session per class on disk)",
              "**(headline)**" in text and "NOT a per-session split" not in text
              and "no per-session split exists" in text)
        wrote = sorted(os.listdir(figs)) if os.path.isdir(figs) else []
        t("headline figures written", "feature_distributions.png" in wrote and "peak_vs_onset.png" in wrote, str(wrote))
        t("both real models persisted with meta",
          all(os.path.exists(os.path.join(models, f"model_{k}_real.json")) for k in ("lda", "qda")))
        if os.path.exists(os.path.join(models, "model_lda_real.json")):
            meta = json.load(open(os.path.join(models, "model_lda_real.json")))["meta"]
            expected = {s for v in on_disk.values() for s in v}
            t("meta records exactly the sessions on disk, the split, the shipped representation, feature set and git hash",
              set(meta["sessions"]) == expected and "split" in meta
              and meta["representation"] == SHIPPED
              and meta["feature_set"] == "cop" and len(meta["git_hash"]) >= 7,
              str({k: meta[k] for k in ("sessions", "representation", "feature_set", "git_hash")}))
    return t.result()


def test_script():
    p, f = check_script()
    assert f == 0, f"{f} check(s) failed; see the FAIL lines above"


SUITES = [check_splits, check_loso, check_identity_gains, check_sim_reproduction, check_script]

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
