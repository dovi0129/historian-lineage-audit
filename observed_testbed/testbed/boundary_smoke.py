"""Boundary smoke (spec §8): what IoTDB 1.3.7 returns before and after FLUSH under SDT.

Cases: single point, straight line, one kink, a short TEP segment; F in {1, 15}; SDT on and off.
All cells share database root.bs (IoTDB 1.3.7 refused a 9th database: no free DataRegionGroup); each cell is its
own device. Cells run one after another and each ends with a final FLUSH, so a FLUSH only meets the current cell's
unflushed rows.
After every write the full series is queried (pre-flush view); after every FLUSH it is queried at once, again
after a 1 s pause, and the on-disk TsFile list is recorded. Raw outputs go to boundary_smoke/raw.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from historian_io import Historian, open_session  # noqa: E402
from iotdb_instance import IoTDBInstance  # noqa: E402
from tep_data import load_tep  # noqa: E402

OUT = Path(__file__).resolve().parent / "boundary_smoke"


def cases() -> dict[str, tuple[list[float], float]]:
    tep = load_tep(1, 1)["XMEAS-7"][:45]
    return {
        "single": ([1.0], 0.01),
        "line": ([0.1 * t for t in range(20)], 0.01),
        "kink": ([float(t) if t <= 9 else 9.0 for t in range(20)], 0.01),
        "tep": ([float(v) for v in tep], 0.5),
    }


def tsfiles(inst: IoTDBInstance) -> list[str]:
    root = inst.root / "data" / "datanode" / "data"
    return sorted(f"{p.relative_to(root)}:{p.stat().st_size}" for p in root.rglob("*.tsfile*"))


def run_cell(sess, inst, name, values, compdev, F, sdt) -> dict:
    cell = f"{name}_f{F}_{'sdt' if sdt else 'raw'}"
    h = Historian(sess, "bs", device=cell)
    ddl = h.create(["x"], {"x": compdev if sdt else None}, create_db=False)
    log: list[dict] = []
    n = len(values)
    flush_ticks = [t for t in range(n) if t >= F - 1 and (t - (F - 1)) % F == 0]
    flush_ticks.append(n)  # final flush after the last write
    for t in range(n + 1):
        if t < n:
            h.write("x", t, values[t])
            log.append({"tick": t, "event": "write", "value": repr(values[t]),
                        "visible": [[a, repr(b)] for a, b in h.full(["x"])["x"]]})
        if t in flush_ticks:
            w0 = time.time()
            h.flush()
            w1 = time.time()
            now = h.full(["x"])["x"]
            files_now = tsfiles(inst)
            time.sleep(1.0)
            later = h.full(["x"])["x"]
            log.append({"tick": t, "event": "flush", "flush_wall_s": round(w1 - w0, 4),
                        "visible_after": [[a, repr(b)] for a, b in now],
                        "same_after_1s": now == later, "tsfiles": files_now, "tsfiles_after_1s": tsfiles(inst)})
    return {"cell": cell, "ddl": ddl, "show": h.show_timeseries(), "n": n, "F": F, "sdt": sdt,
            "compdev": compdev if sdt else None, "flush_ticks": flush_ticks, "log": log}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    inst = IoTDBInstance("boundary_smoke", slot=1)
    inst.prepare()
    results = []
    try:
        inst.start()
        sess = open_session(inst.rpc_port)
        sess.execute_non_query_statement("CREATE DATABASE root.bs")
        for name, (values, compdev) in cases().items():
            for F in (1, 15):
                for sdt in (True, False):
                    results.append(run_cell(sess, inst, name, values, compdev, F, sdt))
                    print(results[-1]["cell"], "done", flush=True)
        # SDT state across a server restart is out of scope; settings are recorded instead.
        sess.close()
    finally:
        inst.stop()
    (OUT / "raw.json").write_text(json.dumps({"instance": inst.settings(), "cells": results}, indent=1))


if __name__ == "__main__":
    main()
