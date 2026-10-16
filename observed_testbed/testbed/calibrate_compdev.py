"""Calibrate SDT CompDev per channel with IoTDB 1.3.7 itself (spec §8): F = 15, batch 1 of each fault,
targets 1/2 (k2) and 1/5 (k5) retained, then freeze in config/compdev.json.

Retention = distinct retained row_times in the final post-flush query / 3001. Writes are one insert_record per tick
holding all five channels (the memtable content equals the run path's per-channel writes); FLUSH every F ticks from
tick F-1 plus a final FLUSH, as in run_one.py. Every evaluation uses a fresh device because CompDev is fixed at
series creation. Per channel, bisection on log10(CompDev); the closer of the two final bracket ends is frozen.

After freezing, achieved retention is measured on batch 1 for F in {1, 5, 15, 60} at the frozen CompDev, together
with the floor (CompDev = 1e12: only segment first/last rows survive). A target is infeasible at F when floor > target.

Usage: calibrate_compdev.py --fault 1 --level k2 --slot 3   (one process per fault x level)
       calibrate_compdev.py --merge                          (writes config/compdev.json, config/achieved_retention.json)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

TESTBED = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTBED))
from historian_io import Historian, meas_name, open_session, tick_to_ms  # noqa: E402
from iotdb_instance import IOTDB_VERSION, IoTDBInstance  # noqa: E402
from iotdb.utils.IoTDBConstants import TSDataType  # noqa: E402
from run_one import CHANNELS, flush_schedule  # noqa: E402
from tep_data import file_sha256, load_tep  # noqa: E402

CONFIG = TESTBED / "config"
TARGET = {"k2": 0.5, "k5": 0.2}
CAL_F, CAL_BATCH, N = 15, 1, 3001
FLOOR_COMPDEV = 1e12
ITERS = 40


class Evaluator:
    def __init__(self, session, values: dict[str, list[float]]):
        self.s = session
        self.values = values
        self.k = 0

    def retention(self, compdev: dict[str, float], F: int) -> dict[str, dict]:
        self.k += 1
        h = Historian(self.s, "cal", device=f"e{self.k}")
        h.create(CHANNELS, compdev, create_db=self.k == 1)
        flush_set = set(flush_schedule(N, F))
        meas = [meas_name(c) for c in CHANNELS]
        types = [TSDataType.DOUBLE] * len(CHANNELS)
        for t in range(N):
            self.s.insert_record(h.device, tick_to_ms(t), meas, types, [self.values[c][t] for c in CHANNELS])
            if t in flush_set:
                h.flush()
        h.flush()
        full = h.full(CHANNELS)
        return {c: {"retention": len({t for t, _ in full[c]}) / N, "rows": len(full[c])} for c in CHANNELS}


def calibrate(fault: int, level: str, slot: int) -> None:
    values = load_tep(fault, CAL_BATCH)
    values = {c: values[c][:N] for c in CHANNELS}
    target = TARGET[level]
    inst = IoTDBInstance(f"calib_f{fault:02d}_{level}", slot=slot)
    inst.prepare()
    log: list[dict] = []
    t0 = time.time()
    try:
        inst.start()
        ev = Evaluator(open_session(inst.rpc_port), values)
        lo = {}
        hi = {}
        for c in CHANNELS:
            span = max(values[c]) - min(values[c])
            lo[c], hi[c] = math.log10(span * 1e-9), math.log10(span * 10)
        r_lo = ev.retention({c: 10 ** lo[c] for c in CHANNELS}, CAL_F)
        r_hi = ev.retention({c: 10 ** hi[c] for c in CHANNELS}, CAL_F)
        log.append({"compdev": {c: 10 ** lo[c] for c in CHANNELS}, "result": r_lo})
        log.append({"compdev": {c: 10 ** hi[c] for c in CHANNELS}, "result": r_hi})
        for c in CHANNELS:
            assert r_lo[c]["retention"] > target > r_hi[c]["retention"], (c, r_lo[c], r_hi[c])
        res_lo = {c: r_lo[c] for c in CHANNELS}
        res_hi = {c: r_hi[c] for c in CHANNELS}
        for _ in range(ITERS):
            mid = {c: (lo[c] + hi[c]) / 2 for c in CHANNELS}
            r = ev.retention({c: 10 ** mid[c] for c in CHANNELS}, CAL_F)
            log.append({"compdev": {c: 10 ** mid[c] for c in CHANNELS}, "result": r})
            for c in CHANNELS:
                if r[c]["retention"] > target:
                    lo[c], res_lo[c] = mid[c], r[c]
                else:
                    hi[c], res_hi[c] = mid[c], r[c]
        frozen, cal = {}, {}
        for c in CHANNELS:
            d_lo, d_hi = abs(res_lo[c]["retention"] - target), abs(res_hi[c]["retention"] - target)
            pick_lo = d_lo < d_hi
            frozen[c] = float(10 ** (lo[c] if pick_lo else hi[c]))
            cal[c] = {"compdev": frozen[c], "achieved_retention_F15_batch1": (res_lo if pick_lo else res_hi)[c],
                      "bracket": [10 ** lo[c], 10 ** hi[c]],
                      "bracket_retention": [res_lo[c]["retention"], res_hi[c]["retention"]]}
        achieved = {}
        for F in (1, 5, 15, 60):
            a = ev.retention(frozen, F)
            fl = ev.retention({c: FLOOR_COMPDEV for c in CHANNELS}, F)
            achieved[str(F)] = {c: {"achieved": a[c]["retention"], "rows": a[c]["rows"],
                                    "floor": fl[c]["retention"], "target": target,
                                    "feasible": fl[c]["retention"] <= target} for c in CHANNELS}
    finally:
        inst.stop()
    out = {"fault": fault, "level": level, "target": target, "calibration_F": CAL_F, "batch": CAL_BATCH,
           "source_sha256": file_sha256(fault, CAL_BATCH), "iotdb_version": IOTDB_VERSION, "iterations": ITERS,
           "channels": cal, "achieved_by_F": achieved, "wall_s": round(time.time() - t0, 1), "evaluations": log}
    CONFIG.mkdir(exist_ok=True)
    (CONFIG / f"calibration_f{fault:02d}_{level}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({c: [cal[c]["compdev"], cal[c]["achieved_retention_F15_batch1"]] for c in CHANNELS}))


def merge() -> None:
    compdev: dict = {}
    achieved: dict = {}
    for fault in (1, 14):
        for level in TARGET:
            d = json.loads((CONFIG / f"calibration_f{fault:02d}_{level}.json").read_text())
            compdev.setdefault(str(fault), {})[level] = {c: d["channels"][c]["compdev"] for c in CHANNELS}
            achieved.setdefault(str(fault), {})[level] = d["achieved_by_F"]
    note = ("SDT CompDev per fault, retention level and channel; calibrated with IoTDB 1.3.7 at F=15 on batch 1 "
            "(calibrate_compdev.py); frozen. Achieved retention and feasibility per F: achieved_retention.json.")
    (CONFIG / "compdev.json").write_text(json.dumps({"note": note, "targets": TARGET, "compdev": compdev}, indent=1))
    (CONFIG / "achieved_retention.json").write_text(json.dumps(
        {"note": "batch 1, distinct retained row_times / 3001; floor = CompDev 1e12; feasible = floor <= target",
         "achieved": achieved}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fault", type=int)
    p.add_argument("--level", choices=list(TARGET))
    p.add_argument("--slot", type=int)
    p.add_argument("--merge", action="store_true")
    a = p.parse_args()
    merge() if a.merge else calibrate(a.fault, a.level, a.slot)
