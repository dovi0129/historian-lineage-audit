"""Check one run directory against INTERFACE.md (format and internal consistency only; no audit logic).

Usage: check_run.py <run_dir> [<run_dir> ...]   exit 1 if any check fails.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

COLS = {
    "samples": ["run_id", "channel", "tick", "value_sent", "opc_source_ts", "opc_server_ts", "server_tick",
                "recv_wall", "write_req_wall", "write_ack_wall", "write_ack_tick"],
    "flushes": ["run_id", "flush_id", "tick", "issue_wall", "return_wall", "tsfiles_before", "tsfiles_after"],
    "membership_delta": ["run_id", "flush_id", "tick", "channel", "row_time", "value", "change"],
    "membership_digest": ["run_id", "flush_id", "tick", "channel", "n_rows", "digest"],
    "snapshots": ["run_id", "origin", "channel", "row_time", "value", "kind", "query_wall"],
}
META_KEYS = ["run_id", "fault", "batch", "n_ticks", "channels", "deadband", "compdev", "flush_every", "flush_ticks",
             "iotdb_version", "iotdb_settings", "asyncua_version", "tick_wall_ms", "testbed_code_digest",
             "created_wall", "duplicate_rows_observed", "memtable_flushes_logged"]


def read(path: Path, name: str) -> list[dict]:
    with (path / f"{name}.csv").open(encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        if r.fieldnames != COLS[name]:
            raise AssertionError(f"{name}.csv columns {r.fieldnames} != {COLS[name]}")
        return list(r)


def digest(lines: list[tuple[int, str]]) -> str:
    s = "\n".join(f"{t},{v}" for t, v in sorted(lines))
    return hashlib.sha256(s.encode("utf-8")).hexdigest() if lines else ""


def check(run: Path) -> list[str]:
    errs: list[str] = []
    ok = lambda cond, msg: None if cond else errs.append(msg)  # noqa: E731
    ok((run / "DONE").exists(), "DONE missing")
    meta = json.loads((run / "run_meta.json").read_text())
    ok(all(k in meta for k in META_KEYS), f"run_meta missing {[k for k in META_KEYS if k not in meta]}")
    n, chans, rid = meta["n_ticks"], meta["channels"], meta["run_id"]
    ok(rid == run.name, "run_id != directory name")
    F = meta["flush_every"]
    ok(meta["flush_ticks"] == list(range(F - 1, n, F)) + [n], "flush_ticks schedule")

    S = read(run, "samples")
    ok(len(S) == n * len(chans), f"samples rows {len(S)} != {n}*{len(chans)}")
    keys = Counter((r["channel"], int(r["tick"])) for r in S)
    ok(set(keys) == {(c, t) for c in chans for t in range(n)} and max(keys.values()) == 1, "samples key set")
    ok(all(r["write_ack_tick"] == r["tick"] for r in S), "write_ack_tick != tick")
    ok(all(r["server_tick"] == r["tick"] for r in S), "server_tick != tick in some rows")
    ok(all(r["value_sent"] == repr(float(r["value_sent"])) for r in S), "value_sent not repr(float)")
    ok(all(r["run_id"] == rid for r in S), "samples run_id")

    Fl = read(run, "flushes")
    ok([int(r["tick"]) for r in Fl] == meta["flush_ticks"], "flushes.csv ticks != flush_ticks")
    ok([int(r["flush_id"]) for r in Fl] == list(range(len(Fl))), "flush_id sequence")
    fid_tick = {int(r["flush_id"]): int(r["tick"]) for r in Fl}
    # unscheduled-flush gate (INTERFACE.md item 3): sealed-file count unchanged between scheduled FLUSHes,
    # at most one file sealed per FLUSH, and IoTDB's memtable-flush log count equal to the files sealed.
    prev_after = 0
    for r in Fl:
        b, a = int(r["tsfiles_before"]), int(r["tsfiles_after"])
        ok(b == prev_after, f"unscheduled flush before flush {r['flush_id']}: tsfiles_before {b} != {prev_after}")
        ok(a - b in (0, 1), f"flush {r['flush_id']} sealed {a - b} files")
        prev_after = a
    n_sealed = sum(int(r["tsfiles_after"]) - int(r["tsfiles_before"]) for r in Fl)
    ok(meta.get("memtable_flushes_logged") == n_sealed,
       f"memtable_flushes_logged {meta.get('memtable_flushes_logged')} != sealed by FLUSH {n_sealed}")

    D = read(run, "membership_delta")
    G = read(run, "membership_digest")
    ok(len(G) == len(Fl) * len(chans), "membership_digest rows")
    cum: dict[str, Counter] = {c: Counter() for c in chans}
    by_fid = defaultdict(list)
    for r in D:
        by_fid[int(r["flush_id"])].append(r)
        ok(fid_tick[int(r["flush_id"])] == int(r["tick"]), "delta tick label")
    removed = sum(r["change"] == "removed" for r in D)
    for fid in range(len(Fl)):
        for r in by_fid.get(fid, []):
            k = (int(r["row_time"]), r["value"])
            if r["change"] == "added":
                cum[r["channel"]][k] += 1
            else:
                cum[r["channel"]][k] -= 1
                ok(cum[r["channel"]][k] >= 0, f"removed absent row {k}")
        for g in (x for x in G if int(x["flush_id"]) == fid):
            lines = list(cum[g["channel"]].elements())
            ok(int(g["n_rows"]) == len(lines), f"n_rows mismatch flush {fid} {g['channel']}")
            ok(g["digest"] == digest(lines), f"digest mismatch flush {fid} {g['channel']}")
    final_dups = {c: sum(cum[c].values()) - len(+cum[c]) for c in chans}
    ok(final_dups == meta["duplicate_rows_observed"], f"duplicate_rows_observed {meta['duplicate_rows_observed']} "
                                                      f"!= deltas {final_dups}")

    P = read(run, "snapshots")
    origins = Counter(int(r["origin"]) for r in P)
    ok(set(origins) == set(range(n)), "snapshot origins != 0..n_ticks-1")
    flush_ticks = meta["flush_ticks"][:-1]
    kinds = Counter()
    for r in P:
        o, t, kind = int(r["origin"]), int(r["row_time"]), r["kind"]
        kinds[kind] += 1
        if kind == "window":
            ok(o - 8 <= t <= o, f"window row out of range o={o} t={t}")
        elif kind == "anchor":
            ok(t < o - 8, f"anchor row o={o} t={t}")
        elif kind == "commit_anchor":
            tf = max([x for x in flush_ticks if x <= o], default=None)
            ok(tf is not None and t <= min(tf, o - 9), f"commit_anchor o={o} t={t} t_f={tf}")
        else:
            errs.append(f"unknown kind {kind}")
    per = Counter((int(r["origin"]), r["channel"], r["kind"]) for r in P if r["kind"] != "window")
    ok(all(v == 1 for v in per.values()), "more than one anchor/commit_anchor per origin and channel")
    print(f"{run.name}: n_ticks={n} F={F} flushes={len(Fl)} sealed={n_sealed} samples={len(S)} delta={len(D)} (removed={removed}) "
          f"snapshots={len(P)} {dict(kinds)} duplicates={final_dups} -> {'OK' if not errs else 'FAIL'}")
    return errs


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        e = check(Path(p))
        for x in e[:20]:
            print("  ", x)
        bad += bool(e)
    sys.exit(1 if bad else 0)
