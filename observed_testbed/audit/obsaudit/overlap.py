"""Descriptive overlap between IoTDB's final retained set and the paper's declared local Swinging Door rule.

The local rule is imported from the public package (src/historian_vintage.py of the historian-lineage-audit
repository) and run with epsilon = the run's CompDev for that channel over the whole series. The Jaccard index is
reported as is, not as a pass/fail check.

The package directory is taken from the environment variable LINEAGE_SRC; by default it is <repo root>/src, where
<repo root> is the parent of observed_testbed/ when this file sits at observed_testbed/audit/obsaudit/overlap.py.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

PUBLIC_SRC = Path(os.environ.get("LINEAGE_SRC") or Path(__file__).resolve().parents[3] / "src")
if str(PUBLIC_SRC) not in sys.path:
    sys.path.append(str(PUBLIC_SRC))
from historian_vintage import swinging_door_trace  # noqa: E402  (public package; imported, not copied)


NOTE = ("IoTDB 1.3.7 resets SDT at every FLUSH: each flushed segment keeps its first and last rows plus interior "
        "corridor breaks (spec §9), so its retained set depends on F. The declared local rule runs once over the "
        "whole series and keeps both series endpoints. The two segmentations differ by construction; the overlap is "
        "descriptive.")


def sdt_overlap(ledger_rows, meta: dict) -> dict:
    compdev = meta.get("compdev") or {}
    out = {}
    for ch, g in ledger_rows.groupby("channel", sort=True):
        eps = compdev.get(ch)
        if eps is None or (isinstance(eps, float) and math.isnan(eps)):
            out[ch] = {"status": "no_deadband"}
            continue
        g = g.sort_values("tick")
        values = g["value_sent"].to_numpy(float)
        local = set(int(i) for i in swinging_door_trace(values, float(eps)).event_indices)
        ticks = g["tick"].to_numpy(np.int64)
        iotdb = set(int(t) for t in ticks[g["final_retained"].to_numpy(bool)])
        local = {int(ticks[i]) for i in local}
        union = local | iotdb
        out[ch] = {
            "compdev": float(eps), "n_samples": int(len(values)),
            "n_iotdb": len(iotdb), "n_local_rule": len(local), "n_both": len(local & iotdb),
            "n_iotdb_only": len(iotdb - local), "n_local_only": len(local - iotdb),
            "jaccard": len(local & iotdb) / len(union) if union else None,
            "iotdb_retention": len(iotdb) / len(values), "local_retention": len(local) / len(values),
            "last_tick_iotdb": int(ticks[-1]) in iotdb, "last_tick_local": int(ticks[-1]) in local,
        }
    return {"note": NOTE, "flush_every": meta.get("flush_every"), "channels": out}
