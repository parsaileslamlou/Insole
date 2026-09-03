"""Train/test splits for per-stance feature frames.

Every function takes a pandas frame with a `label` column, a `session`
column and a `start` column (frame index of stance onset), and returns index
arrays into that frame. Nothing here shuffles unless asked to, and nothing
here knows about models.

Why these and not a per-session split: the real dataset has ONE session per
class, so no split can hold out a session. The honest substitute is a
time-blocked split within each session -- earlier stances train, later
stances test -- with a random stance-level split reported beside it to show
how much optimism temporal adjacency buys. leave_one_session_out() exists so
the training script switches to it automatically the day a second session
per class lands.
"""

import numpy as np


def _per_class_order(frame):
    """{label: index array sorted by (session, start)} -- time order per class."""
    out = {}
    for label in sorted(frame["label"].unique()):
        sub = frame[frame["label"] == label].sort_values(["session", "start"])
        out[label] = sub.index.to_numpy()
    return out


def time_blocked_split(frame, train_frac=0.6, guard=0):
    """Earlier stances train, later ones test, per class.

    train = the first round(train_frac * n) stances of each class in onset
    order, test = the rest. `guard` drops that many stances from the END of
    each class's training block, so no test stance is adjacent in time to a
    training stance; the dropped stances belong to neither side. Returns
    (train_idx, test_idx, per_class) where per_class maps label ->
    (n_train, n_test, n_dropped).
    """
    train, test, per_class = [], [], {}
    for label, idx in _per_class_order(frame).items():
        n = len(idx)
        n_train = int(round(train_frac * n))
        tr = idx[:max(n_train - guard, 0)]
        te = idx[n_train:]
        train.extend(tr)
        test.extend(te)
        per_class[label] = (len(tr), len(te), n_train - len(tr))
    return np.array(train), np.array(test), per_class


def random_stance_splits(frame, train_frac=0.6, n_repeats=20, seed=0):
    """Per-class random splits with the same sizes as the time-blocked one.

    The optimism of this split is the point: consecutive stances in one
    session are near-duplicates, so a random split puts near-copies of every
    test stance into training. Yields (train_idx, test_idx) n_repeats times.
    """
    rng = np.random.default_rng(seed)
    order = _per_class_order(frame)
    for _ in range(n_repeats):
        train, test = [], []
        for label, idx in order.items():
            n_train = int(round(train_frac * len(idx)))
            perm = rng.permutation(idx)
            train.extend(perm[:n_train])
            test.extend(perm[n_train:])
        yield np.array(train), np.array(test)


def contiguous_block_folds(frame, n_blocks=5):
    """Cross-validation over contiguous time blocks, per class.

    Each class's stances are cut, in onset order, into n_blocks contiguous
    chunks; fold k holds out chunk k of EVERY class. Every stance is tested
    exactly once, and no test stance sits inside its own training block --
    though the two blocks adjacent to it are training, so adjacency leakage
    is reduced, not removed. Yields (train_idx, test_idx) per fold.
    """
    order = _per_class_order(frame)
    chunks = {label: np.array_split(idx, n_blocks) for label, idx in order.items()}
    for k in range(n_blocks):
        test = np.concatenate([chunks[label][k] for label in order])
        train = np.concatenate([np.concatenate([c for j, c in enumerate(chunks[label]) if j != k])
                                for label in order])
        yield train, test


def sessions_per_class(frame):
    """{label: sorted list of session ids}."""
    return {label: sorted(frame.loc[frame["label"] == label, "session"].unique())
            for label in sorted(frame["label"].unique())}


def leave_one_session_out(frame):
    """Hold out session k of every class, for k in range(min sessions per class).

    Only meaningful when every class has at least two sessions; the caller
    checks sessions_per_class() first. Yields (train_idx, test_idx, held_out)
    where held_out lists the session ids in the test fold.
    """
    spc = sessions_per_class(frame)
    n_folds = min(len(v) for v in spc.values())
    for k in range(n_folds):
        held = [spc[label][k] for label in spc]
        is_test = frame["session"].isin(held).to_numpy()
        yield frame.index.to_numpy()[~is_test], frame.index.to_numpy()[is_test], held
