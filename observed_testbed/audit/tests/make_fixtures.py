"""Write the hand-specified fixture runs into audit/tests/fixtures/ (or a given directory).

Known answers for each scenario are stated in test_audit.py next to the assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_sim import simulate  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def values(n: int) -> dict[str, list[float]]:
    # Nonlinear A so interpolation differs from the sample; B with a step pattern.
    return {"A": [1.0 + 0.1 * t * t for t in range(n)],
            "B": [5.0 - 0.3 * t + (0.25 if t % 3 == 0 else 0.0) for t in range(n)]}


A_RETAINED = {0, 4, 5, 9, 10, 14, 15, 19}


def build(root: Path = FIXTURES) -> list[str]:
    root = Path(root)
    # 1. No deadband, F = 1: every view equals the truth.
    simulate(root, "nodb_f1", ["A", "B"], 20, 1, values(20))
    # 2. No deadband, F = 5: commit lag only. V3 != S_commit between flushes; Replay_commit == S_commit.
    simulate(root, "lag_f5", ["A", "B"], 20, 5, values(20))
    # 3. SDT-like discards on A at each flush (provisional rows visible, then discarded); B keeps everything
    #    but its OPC UA server stamp lags by one tick, so V_server != V3 on B.
    simulate(root, "provisional_discard", ["A", "B"], 20, 5, values(20),
             discard={("A", t) for t in range(20) if t not in A_RETAINED},
             server_offset={("B", t): 1 for t in range(20)}, deadband="SDT", compdev={"A": 0.5, "B": None})
    # 4. F = 12: origins 8..10 precede the first flush (tick 11); origins 21, 22 need the commit_anchor.
    #    A discards rows 9..11 at the first flush, so its latest committed row before o-8 is row 8.
    late = dict(discard={("A", 9), ("A", 10), ("A", 11)})
    simulate(root, "late_first_flush", ["A", "B"], 26, 12, values(26), **late)
    simulate(root, "late_first_flush_noca", ["A", "B"], 26, 12, values(26), emit_commit_anchor=False, **late)
    # 5. Violations: A row 12 committed at flush 14, removed at flush 19; B row 7 committed at 9, changed at 19.
    simulate(root, "violation", ["A", "B"], 25, 5, values(25),
             removals={("A", 12): 19}, changes={("B", 7): (19, 99.5)})
    # 6. Late commit: A row 6 is held back at flush 9 (absent from that membership) and stored at flush 14.
    simulate(root, "late_commit", ["A", "B"], 20, 5, values(20), late_commit={("A", 6): 14})
    # 7. IoTDB one-row SDT segments: at F = 1 every stored A row has two identical copies. No mismatch allowed.
    simulate(root, "dup_f1", ["A", "B"], 20, 1, values(20), dup={("A", t) for t in range(20)},
             deadband="SDT", compdev={"A": 0.5, "B": None})
    # 8. One time stored with two different values (A row 6 at flush 9): both paths must report an error.
    simulate(root, "dup_conflict", ["A", "B"], 20, 5, values(20), conflict={("A", 6): 123.0})
    # 9. An automatic flush at tick 6 (between commanded FLUSHes at 4 and 9) seals rows 5..6 unobserved: the
    #    FLUSH at 9 sees tsfiles_before = 2 against tsfiles_after = 1 of the FLUSH at 4, so the run is invalid.
    simulate(root, "unscheduled_flush", ["A", "B"], 20, 5, values(20), unscheduled={6})
    return sorted(p.name for p in root.iterdir() if p.is_dir())


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURES))
