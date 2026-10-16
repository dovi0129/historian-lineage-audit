"""Determinism check: two runs of the same cell must agree on every file once wall-clock fields are dropped.

Dropped: columns ending in `_wall`, `opc_server_ts` (a server wall clock), run_id; in run_meta: run_id, created_wall,
iotdb_settings entries that name ports or the instance directory. events.log is free text and is not compared.
Usage: compare_runs.py <run_a> <run_b>   exit 1 on any difference.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

FILES = ["samples", "flushes", "membership_delta", "membership_digest", "snapshots"]


def rows(run: Path, name: str) -> list[tuple]:
    with (run / f"{name}.csv").open(encoding="utf-8", newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        keep = [i for i, c in enumerate(head) if not c.endswith("_wall") and c not in ("opc_server_ts", "run_id")]
        return [tuple(head[i] for i in keep)] + [tuple(x[i] for i in keep) for x in r]


def meta(run: Path) -> dict:
    m = json.loads((run / "run_meta.json").read_text())
    for k in ("run_id", "created_wall"):
        m.pop(k, None)
    m["iotdb_settings"] = {k: v for k, v in m["iotdb_settings"].items()
                           if not k.endswith("_port") and k != "instance_root"}
    return m


def main(a: Path, b: Path) -> int:
    diffs = 0
    if meta(a) != meta(b):
        ma, mb = meta(a), meta(b)
        print("run_meta differs:", {k for k in ma.keys() | mb.keys() if ma.get(k) != mb.get(k)})
        diffs += 1
    for name in FILES:
        ra, rb = rows(a, name), rows(b, name)
        if ra != rb:
            first = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
            print(f"{name}.csv differs: {len(ra)} vs {len(rb)} rows; first difference at row {first}")
            diffs += 1
        else:
            print(f"{name}.csv identical ({len(ra) - 1} rows, wall columns dropped)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
