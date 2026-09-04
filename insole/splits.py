"""Train/test splits for per-stance feature frames.

Every function takes a pandas frame with a `label` column, a `session`
column and a `start` column (frame index of stance onset), and returns index
arrays into that frame. Nothing here shuffles unless asked to, and nothing
here knows about models.

Two families. The within-session splits (time_blocked_split,
random_stance_splits, contiguous_block_folds) work per (class, session)
group, so with one session per class they are what an honest substitute for
a per-session split looks like -- earlier stances train, later stances test
-- and with several sessions per class they still never cross a session
boundary, which is what makes them the optimistic reference beside
leave_one_session_out(). The training script switches to
leave_one_session_out() on its own once every class has at least two
sessions; sessions_per_class() is how it checks.
"""

import numpy as np


def _per_group_order(frame):
    """{(label, session): index array sorted by start} -- time order per group."""
    out = {}
    for label in sorted(frame["label"].unique()):
        sub = frame[frame["label"] == label]
        for session in sorted(sub["session"].unique()):
            g = sub[sub["session"] == session].sort_values("start")
            out[(label, session)] = g.index.to_numpy()
    return out


def _per_class_order(frame):
    """{label: index array sorted by (session, start)} -- time order per class."""
    out = {}
    for label in sorted(frame["label"].unique()):
        sub = frame[frame["label"] == label].sort_values(["session", "start"])
        out[label] = sub.index.to_numpy()
    return out


def time_blocked_split(frame, train_frac=0.6, guard=0):
    """Earlier stances train, later ones test, within every (class, session).

    train = the first round(train_frac * n) stances of each (class, session)
    group in onset order, test = the rest. `guard` drops that many stances
    from the END of each group's training block, so no test stance is
    adjacent in time to a training stance; the dropped stances belong to
    neither side. Returns (train_idx, test_idx, per_class) where per_class
    maps label -> (n_train, n_test, n_dropped), summed over that class's
    sessions.
    """
    train, test, per_class = [], [], {}
    for (label, _session), idx in _per_group_order(frame).items():
        n = len(idx)
        n_train = int(round(train_frac * n))
        tr = idx[:max(n_train - guard, 0)]
        te = idx[n_train:]
        train.extend(tr)
        test.extend(te)
        a, b, c = per_class.get(label, (0, 0, 0))
        per_class[label] = (a + len(tr), b + len(te), c + n_train - len(tr))
    return np.array(train), np.array(test), per_class


def random_stance_splits(frame, train_frac=0.6, n_repeats=20, seed=0):
    """Per-(class, session) random splits with the same sizes as the time-blocked one.

    The optimism of this split is the point: consecutive stances in one
    session are near-duplicates, so a random split puts near-copies of every
    test stance into training. Yields (train_idx, test_idx) n_repeats times.
    """
    rng = np.random.default_rng(seed)
    order = _per_group_order(frame)
    for _ in range(n_repeats):
        train, test = [], []
        for _key, idx in order.items():
            n_train = int(round(train_frac * len(idx)))
            perm = rng.permutation(idx)
            train.extend(perm[:n_train])
            test.extend(perm[n_train:])
        yield np.array(train), np.array(test)


def contiguous_block_folds(frame, n_blocks=5):
    """Cross-validation over contiguous time blocks, per (class, session).

    Each group's stances are cut, in onset order, into n_blocks contiguous
    chunks; fold k holds out chunk k of EVERY group. Every stance is tested
    exactly once, and no test stance sits inside its own training block --
    though the two blocks adjacent to it are training, so adjacency leakage
    is reduced, not removed. Yields (train_idx, test_idx) per fold.
    """
    order = _per_group_order(frame)
    chunks = {key: np.array_split(idx, n_blocks) for key, idx in order.items()}
    for k in range(n_blocks):
        test = np.concatenate([chunks[key][k] for key in order])
        train = np.concatenate([np.concatenate([c for j, c in enumerate(chunks[key]) if j != k])
                                for key in order])
        yield train, test


def sessions_per_class(frame):
    """{label: sorted list of session ids}."""
    return {label: sorted(frame.loc[frame["label"] == label, "session"].unique())
            for label in sorted(frame["label"].unique())}


def leave_one_session_out(frame):
    """Hold out session k of every class, for k in range(min sessions per class).

    Only meaningful when every class has at least two sessions; the caller
    checks sessions_per_class() first. Yields (train_idx, test_idx, held_out)
    where held_out lists the session ids in the test fold. Over all folds
    every stance of the first min-sessions-per-class sessions of each class
    is tested exactly once, out of its own session.
    """
    spc = sessions_per_class(frame)
    n_folds = min(len(v) for v in spc.values())
    for k in range(n_folds):
        held = [spc[label][k] for label in spc]
        is_test = frame["session"].isin(held).to_numpy()
        yield frame.index.to_numpy()[~is_test], frame.index.to_numpy()[is_test], held
