"""Generate the multi-session dataset for the Prompt 10 baseline model.

One file per (class, session). Sessions of the same class differ by RNG seed
and a small cadence jitter, so a session-disjoint split has something to be
disjoint about. Kept separate from gait_gen's STREAMS, which are the Prompt 9
regression fixtures and must not move.
"""
import random

from gait_gen import gait_lines

DURATION_S = 60
SESSIONS_PER_CLASS = 4

# label -> (gait_gen mode, base cycle_s). "fast" is walk mode at a shorter
# period, so the base cycle is spelled out for every class, not just that one.
CLASSES = {
    "walk":    ("walk",    1.0),
    "fast":    ("walk",    0.6),
    "shuffle": ("shuffle", 0.5),
}

# RETUNE: per-session cadence jitter, +/-4%. Wide enough that sessions are not
# copies, narrow enough that the classes stay separated in contact time.
JITTER = [0.96, 0.985, 1.015, 1.04]

# Fixed seed per class so runs are reproducible across machines and processes.
SEED_BASE = {"walk": 100, "fast": 200, "shuffle": 300}


def session_name(label, idx):
    return f"sim_{label}_{idx:02d}.txt"


if __name__ == "__main__":
    for label, (mode, base_cycle) in CLASSES.items():
        for idx in range(SESSIONS_PER_CLASS):
            cycle_s = base_cycle * JITTER[idx]
            random.seed(SEED_BASE[label] + idx)
            name = session_name(label, idx)
            with open(name, "w") as f:
                for line in gait_lines(DURATION_S, mode=mode, cycle_s=cycle_s):
                    f.write(line + "\n")
            print(f"wrote {name}  mode={mode}  cycle_s={cycle_s:.4f}")