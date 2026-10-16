"""One testbed run: TEP replay over OPC UA -> collector -> IoTDB 1.3.7, with the spec §8 barrier at every tick.

Barrier at tick t: publish -> receive (all channels) -> write ack (per channel) -> if t is a flush tick: FLUSH, wait
for return, membership query -> origin snapshot for o = t. After the last tick: FLUSH + membership, labelled
tick = n_ticks, no snapshot. Ordering comes only from this loop; wall times are logged, never used for ordering.

Outputs (INTERFACE.md): runs/<run_id>/{run_meta.json, samples.csv, flushes.csv, membership_delta.csv,
membership_digest.csv, snapshots.csv, events.log, DONE}.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

TESTBED = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTBED))
from collector import Collector, iso, now_utc  # noqa: E402
from historian_io import Historian, open_session  # noqa: E402
from iotdb_instance import IOTDB_VERSION, IoTDBInstance  # noqa: E402
from opcua_source import TEPReplayServer  # noqa: E402
from query_client import QueryClient  # noqa: E402
from tep_data import file_sha256, load_tep  # noqa: E402

RUNS = TESTBED.parent / "runs"
CHANNELS = ["XMEAS-7", "XMEAS-9", "XMV-10", "XMEAS-4", "XMEAS-21"]
CODE_FILES = ["collector.py", "historian_io.py", "iotdb_instance.py", "opcua_source.py", "query_client.py",
              "run_one.py", "tep_data.py", "run_all.sh", "run_list.py"]


def code_digest() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        p = TESTBED / name
        h.update(name.encode() + b"\0" + (p.read_bytes() if p.exists() else b"<missing>") + b"\0")
    return h.hexdigest()


def flush_schedule(n_ticks: int, F: int) -> list[int]:
    ticks = [t for t in range(F - 1, n_ticks, F)]
    return ticks + [n_ticks]  # final flush after the last tick, labelled n_ticks


def run_id_for(fault: int, batch: int, level: str, F: int, rep: int = 0) -> str:
    return f"f{fault:02d}_b{batch:02d}_{level}_F{F:02d}" + (f"_rep{rep}" if rep else "")


def compdev_for(fault: int, level: str) -> dict[str, float | None]:
    if level == "none":
        return {c: None for c in CHANNELS}
    cfg = json.loads((TESTBED / "config" / "compdev.json").read_text())
    entry = cfg["compdev"][str(fault)][level]
    return {c: float(entry[c]) for c in CHANNELS}


async def run(args) -> None:
    run_id = args.run_id or run_id_for(args.fault, args.batch, args.level, args.flush_every, args.rep)
    out = (Path(args.out_root) if args.out_root else RUNS) / run_id
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} exists and is not empty; refusing to overwrite (use --overwrite)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "events.log").open("w", encoding="utf-8")

    def ev(msg: str) -> None:
        log.write(f"{iso(now_utc())} {msg}\n")
        log.flush()

    tep = load_tep(args.fault, args.batch)
    n = args.n_ticks
    values = {c: tep[c][:n] for c in CHANNELS}
    compdev = compdev_for(args.fault, args.level)
    F = args.flush_every
    flush_ticks = flush_schedule(n, F)
    flush_set = set(flush_ticks)
    inst = IoTDBInstance(run_id, slot=args.slot)
    inst.prepare()
    ev(f"run {run_id} fault={args.fault} batch={args.batch} level={args.level} F={F} n_ticks={n} slot={args.slot}")
    ev(f"source mode1_{args.fault}_{args.batch}.xlsx sha256={file_sha256(args.fault, args.batch)} (verified at load)")
    ev(f"compdev {json.dumps(compdev)}")
    meta = {
        "run_id": run_id, "fault": args.fault, "batch": args.batch, "n_ticks": n, "channels": CHANNELS,
        "deadband": "none" if args.level == "none" else "SDT", "compdev": compdev, "flush_every": F,
        "flush_ticks": flush_ticks, "iotdb_version": IOTDB_VERSION, "iotdb_settings": inst.settings(),
        "asyncua_version": version("asyncua"), "tick_wall_ms": args.tick_wall_ms,
        "testbed_code_digest": code_digest(), "created_wall": iso(now_utc()),
        "retention_level": args.level, "iotdb_client_version": version("apache-iotdb"),
    }
    files = {k: (out / f"{k}.csv").open("w", newline="", encoding="utf-8") for k in
             ("samples", "flushes", "membership_delta", "membership_digest", "snapshots")}
    w = {k: csv.writer(f, lineterminator="\n") for k, f in files.items()}
    w["samples"].writerow(["run_id", "channel", "tick", "value_sent", "opc_source_ts", "opc_server_ts", "server_tick",
                           "recv_wall", "write_req_wall", "write_ack_wall", "write_ack_tick"])
    w["flushes"].writerow(["run_id", "flush_id", "tick", "issue_wall", "return_wall", "tsfiles_before",
                           "tsfiles_after"])
    w["membership_delta"].writerow(["run_id", "flush_id", "tick", "channel", "row_time", "value", "change"])
    w["membership_digest"].writerow(["run_id", "flush_id", "tick", "channel", "n_rows", "digest"])
    w["snapshots"].writerow(["run_id", "origin", "channel", "row_time", "value", "kind", "query_wall"])

    server = coll = None
    sessions = []
    t_start = time.time()
    try:
        ev(f"iotdb start {inst.start():.1f}s port={inst.rpc_port} root={inst.root}")
        s_coll, s_query = open_session(inst.rpc_port), open_session(inst.rpc_port)
        sessions = [s_coll, s_query]
        h_coll, h_query = Historian(s_coll, "tep"), Historian(s_query, "tep")
        for sql in h_coll.create(CHANNELS, compdev):
            ev(f"ddl {sql}")
        for line in h_coll.show_timeseries():
            ev(f"show {line}")
        server = TEPReplayServer(inst.base_port + 15, CHANNELS)
        await server.start()
        coll = Collector(server.endpoint, CHANNELS, h_coll)
        drained = await coll.connect()
        ev(f"opcua {server.endpoint} subscribed; initial notifications drained: {drained}")
        qc = QueryClient(h_query, CHANNELS)
        tick_starts = []
        flush_id = 0
        sealed: list[int] = []
        t_f = None
        loop_t0 = time.time()

        def do_flush(tick: int) -> None:
            nonlocal flush_id
            before = inst.sealed_tsfiles()
            issue, ret = coll.flush()
            after = inst.sealed_tsfiles()
            sealed.append(after - before)
            w["flushes"].writerow([run_id, flush_id, tick, iso(issue), iso(ret), before, after])
            delta, summary = coll.membership()
            for ch, rt, v, change in delta:
                w["membership_delta"].writerow([run_id, flush_id, tick, ch, rt, repr(v), change])
            for ch in CHANNELS:
                w["membership_digest"].writerow([run_id, flush_id, tick, ch, *summary[ch]])
            flush_id += 1

        for t in range(n):
            tick_starts.append(now_utc())
            await server.publish(t, {c: values[c][t] for c in CHANNELS})
            got = await coll.receive(t)
            for ch in CHANNELS:
                v, src, srv, recv = got[ch]
                if repr(v) != repr(values[ch][t]):
                    raise RuntimeError(f"value changed in transit at tick {t} {ch}: {v!r} != {values[ch][t]!r}")
                srv_aware = srv if srv.tzinfo else srv.replace(tzinfo=src.tzinfo)
                server_tick = bisect.bisect_right(tick_starts, srv_aware) - 1
                req, ack = coll.write(ch, t, v)
                w["samples"].writerow([run_id, ch, t, repr(values[ch][t]), iso(src), iso(srv), server_tick,
                                       iso(recv), iso(req), iso(ack), t])
            if t in flush_set:
                do_flush(t)
                t_f = t
            for row in qc.snapshot(t, t_f):
                w["snapshots"].writerow([run_id, row[0], row[1], row[2], repr(row[3]), row[4], row[5]])
            if args.tick_wall_ms:
                rest = args.tick_wall_ms / 1000 - (time.time() - tick_starts[-1].timestamp())
                if rest > 0:
                    await asyncio.sleep(rest)
            if t % 500 == 0:
                ev(f"tick {t} wall {time.time() - loop_t0:.1f}s")
        do_flush(n)
        loop_s = time.time() - loop_t0
        meta["duplicate_rows_observed"] = coll.duplicate_rows()
        ev(f"loop done: {n} ticks in {loop_s:.1f}s ({1000 * loop_s / max(n, 1):.1f} ms/tick); "
           f"{flush_id} flushes; duplicate_rows_observed {json.dumps(meta['duplicate_rows_observed'])}")
        retained = {ch: len({t for t, _ in rows}) for ch, rows in coll.last_members.items()}
        ev(f"retained distinct rows / n_ticks: " + json.dumps({ch: round(k / n, 6) for ch, k in retained.items()}))
    except BaseException:
        ev("FAILED\n" + traceback.format_exc())
        raise
    finally:
        for f in files.values():
            f.close()
        if coll is not None:
            try:
                await coll.close()
            except Exception as exc:  # noqa: BLE001 - shutdown only, recorded
                ev(f"collector close error: {exc!r}")
        if server is not None:
            await server.stop()
        for s in sessions:
            s.close()
        inst.stop()
        ev(f"iotdb stopped; total wall {time.time() - t_start:.1f}s")
    # Unscheduled flush guard: every memtable flush is logged once by IoTDB; the count must equal the files sealed
    # by the scheduled FLUSHes (check_run.py checks tsfiles_before/after in flushes.csv).
    meta["memtable_flushes_logged"] = inst.memtable_flushes_logged()
    ev(f"memtable flushes logged by IoTDB {meta['memtable_flushes_logged']}; files sealed by scheduled FLUSH "
       f"{sum(sealed)}")
    if meta["memtable_flushes_logged"] != sum(sealed):
        ev("FAILED: memtable flush count differs from scheduled-FLUSH sealed files (unscheduled flush)")
        log.close()
        raise SystemExit(1)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    ev("DONE")
    log.close()
    (out / "DONE").write_text(iso(now_utc()) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fault", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--level", choices=["k2", "k5", "none"], required=True)
    p.add_argument("--flush-every", type=int, required=True)
    p.add_argument("--n-ticks", type=int, default=3001)
    p.add_argument("--slot", type=int, required=True, help="port block index; unique among concurrent runs")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-root", default=None, help="default: ../runs")
    p.add_argument("--overwrite", action="store_true", help="delete an existing run directory of the same id first")
    p.add_argument("--tick-wall-ms", type=float, default=0.0, help="minimum wall time per tick; 0 = no pacing")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
