"""Follow-up probe to the boundary smoke: when does an SDT flush return the same row twice?

Straight line 0.1*t, n = 16, CompDev 0.01, SDT on, F in {2, 3, 15}; F = 15 leaves a one-row last segment (tick 15).
Records the full query, count(x), a window query and a latest-row query after the final flush.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from historian_io import Historian, _rows, open_session, tick_to_ms  # noqa: E402
from iotdb_instance import IoTDBInstance  # noqa: E402

OUT = Path(__file__).resolve().parent / "boundary_smoke"
inst = IoTDBInstance("boundary_probe", slot=2)
inst.prepare()
res = []
try:
    inst.start()
    s = open_session(inst.rpc_port)
    s.execute_non_query_statement("CREATE DATABASE root.bp")
    for F in (2, 3, 15):
        h = Historian(s, "bp", device=f"line_f{F}")
        h.create(["x"], {"x": 0.01}, create_db=False)
        flushes = []
        for t in range(16):
            h.write("x", t, 0.1 * t)
            if t >= F - 1 and (t - (F - 1)) % F == 0:
                h.flush(); flushes.append(t)
        h.flush(); flushes.append(16)
        full = h.full(["x"])["x"]
        cnt = _rows(s, f"SELECT count(x) FROM {h.device}", 1)
        win = h.window(["x"], 13, 15)["x"]
        last = h.latest_le("x", 15)
        raw_lines = []
        ds = s.execute_query_statement(f"SELECT x FROM {h.device}")
        while ds.has_next():
            raw_lines.append(str(ds.next()))
        ds.close_operation_handle()
        res.append({"F": F, "flush_ticks": flushes, "full": [[a, repr(b)] for a, b in full],
                    "count": cnt[0][1][0] if cnt else None, "window_13_15": [[a, repr(b)] for a, b in win],
                    "latest_le_15": last, "raw_result_lines": raw_lines})
        print(F, [a for a, _ in full], "count", res[-1]["count"], "win", [a for a, _ in win], "last", last)
    s.close()
finally:
    inst.stop()
(OUT / "raw_probe.json").write_text(json.dumps(res, indent=1))
