"""Boundary probe (finding 7): a multi-series raw query hangs once one SDT series holds a doubled row.

Usage: boundary_multiseries_probe.py <n_rows> sdt|raw <F> <slot>; n=300, F=15 has no one-row segment, n=301 has one."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from historian_io import Historian, _rows, meas_name, open_session, tick_to_ms
from iotdb_instance import IoTDBInstance
from iotdb.utils.IoTDBConstants import TSDataType
from run_one import CHANNELS
from tep_data import load_tep

n = int(sys.argv[1]); sdt = sys.argv[2] == "sdt"; F = int(sys.argv[3])
vals = load_tep(1, 1)
inst = IoTDBInstance(f"qprobe_{sys.argv[2]}_{n}_{F}", slot=int(sys.argv[4])); inst.prepare()
try:
    inst.start(); s = open_session(inst.rpc_port)
    h = Historian(s, "q")
    h.create(CHANNELS, {c: (1e-6 if sdt else None) for c in CHANNELS})
    meas = [meas_name(c) for c in CHANNELS]
    t0 = time.time()
    for t in range(n):
        s.insert_record(h.device, tick_to_ms(t), meas, [TSDataType.DOUBLE]*5, [vals[c][t] for c in CHANNELS])
        if t >= F - 1 and (t - (F - 1)) % F == 0:
            h.flush()
    h.flush()
    print("write+flush", round(time.time() - t0, 2), "s", flush=True)
    for c in CHANNELS[:2]:
        t0 = time.time(); r = _rows(s, f"SELECT {meas_name(c)} FROM {h.device}", 1); print("single", c, len(r), round(time.time()-t0, 3), "s", flush=True)
    t0 = time.time(); r = _rows(s, f"SELECT {', '.join(meas)} FROM {h.device}", 5); print("5 cols no order", len(r), round(time.time()-t0, 3), "s", flush=True)
    t0 = time.time(); r = _rows(s, f"SELECT {', '.join(meas)} FROM {h.device} ORDER BY TIME ASC", 5); print("5 cols order", len(r), round(time.time()-t0, 3), "s", flush=True)
finally:
    inst.stop()
