"""Print the run list for run_all.sh: one line per run, `run_id fault batch level F n_ticks slot`, longest first.

Full design (spec §3/§8): faults 1, 14 x batches 9, 10 x {k2, k5} x F {1, 5, 15, 60} (32 runs) + control {none} x
F {1, 15} (8 runs) = 40, plus one repeat per factor level (4 runs that together cover every fault, batch,
retention level and F). --smoke: three 300-tick runs and a repeat of each, ids prefixed `smoke_`.
"""
from __future__ import annotations

import sys

from run_one import run_id_for

REPEATS = [(1, 9, "k2", 1), (14, 10, "k5", 5), (1, 10, "none", 15), (14, 9, "k2", 60)]
SMOKE = [(1, 9, "k2", 15), (14, 10, "k5", 1), (1, 9, "none", 1)]
SLOT0 = 10


def full() -> list[tuple]:
    runs = []
    for fault in (1, 14):
        for batch in (9, 10):
            for level in ("k2", "k5"):
                for F in (1, 5, 15, 60):
                    runs.append((run_id_for(fault, batch, level, F), fault, batch, level, F, 3001))
            for F in (1, 15):
                runs.append((run_id_for(fault, batch, "none", F), fault, batch, "none", F, 3001))
    assert len(runs) == 40
    runs += [(run_id_for(f, b, lv, F, rep=1), f, b, lv, F, 3001) for f, b, lv, F in REPEATS]
    return sorted(runs, key=lambda r: (r[4], r[0]))  # F = 1 (most flushes) first


def smoke(n_ticks: int = 300) -> list[tuple]:
    runs = []
    for rep in (0, 1):
        runs += [("smoke_" + run_id_for(f, b, lv, F, rep), f, b, lv, F, n_ticks) for f, b, lv, F in SMOKE]
    return runs


if __name__ == "__main__":
    rows = smoke(int(sys.argv[2]) if len(sys.argv) > 2 else 300) if sys.argv[1:2] == ["--smoke"] else full()
    for i, r in enumerate(rows):
        print(*r, SLOT0 + i)
