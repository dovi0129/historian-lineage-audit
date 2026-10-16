"""Toy historian that writes INTERFACE.md run directories for hand-specified scenarios.

Deliberately independent of the audit package (no imports from obsaudit). Barrier per tick t: write every channel's
sample t (visible, provisional) -> if t is a flush tick, flush (pending rows are stored unless the scenario discards
or delays them; scheduled removals/changes of stored rows apply) and record the membership -> snapshot for o = t.
A final flush labelled last_tick + 1 follows the last tick; it has no snapshot. Stored rows may carry two copies
(INTERFACE.md "Duplicate rows"): membership deltas, digests and snapshot windows list every copy, anchors one.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def _wall(t: int, frac: int = 0) -> str:
    return f"2026-10-16T{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}.{frac:06d}+00:00"


def _digest(rows: dict[int, list[float]]) -> str:
    text = "\n".join(f"{t},{repr(v)}" for t in sorted(rows) for v in rows[t])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simulate(out_dir: Path, run_id: str, channels: list[str], n_ticks: int, F: int, values: dict[str, list[float]],
             discard=(), late_commit=None, removals=None, changes=None, server_offset=None,
             emit_commit_anchor: bool = True, deadband: str = "none", compdev=None, dup=(), conflict=None,
             unscheduled=()) -> None:
    discard = set(discard)
    late_commit = dict(late_commit or {})      # (ch, t) -> flush tick at which the held row is finally stored
    removals = dict(removals or {})            # (ch, t) -> flush tick at which a stored row is removed
    changes = dict(changes or {})              # (ch, t) -> (flush tick, new value)
    server_offset = dict(server_offset or {})  # (ch, t) -> server_tick - tick
    dup = set(dup)                             # (ch, t) stored as two identical copies (one-row SDT segment)
    conflict = dict(conflict or {})            # (ch, t) -> value of a second stored copy (must be rejected)
    unscheduled = set(unscheduled)             # ticks with an automatic flush after the write (no membership query)
    last = n_ticks - 1
    regular = list(range(F - 1, last + 1, F))
    flush_ticks = regular + [last + 1]
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    stored = {c: {} for c in channels}  # time -> list of stored copies
    pending = {c: {} for c in channels}
    held: dict[tuple[str, int], float] = {}
    previous = {c: {} for c in channels}
    samples, flushes, deltas, digests, snaps = [], [], [], [], []
    t_f = None
    sealed = [0]  # sealed TsFiles; each flush with pending rows seals exactly one

    def seal() -> None:
        if any(pending[c] for c in channels):
            sealed[0] += 1
        for c in channels:
            for t, v in sorted(pending[c].items()):
                if (c, t) in discard:
                    continue
                if (c, t) in late_commit:
                    held[(c, t)] = v
                    continue
                stored[c][t] = [v, v] if (c, t) in dup else [v, conflict[(c, t)]] if (c, t) in conflict else [v]
            pending[c].clear()

    def flush(tick: int, flush_id: int) -> None:
        tsfiles_before = sealed[0]
        seal()
        for (c, t), when in late_commit.items():
            if when == tick and (c, t) in held:
                stored[c][t] = [held.pop((c, t))]
        for (c, t), when in removals.items():
            if when == tick:
                del stored[c][t]
        for (c, t), (when, new) in changes.items():
            if when == tick:
                stored[c][t] = [new] * len(stored[c][t])
        flushes.append([run_id, flush_id, tick, _wall(tick, 100), _wall(tick, 200), tsfiles_before, sealed[0]])
        for c in channels:
            before, now = previous[c], stored[c]
            for t in sorted(set(before) | set(now)):
                if t in before and (t not in now or now[t] != before[t]):
                    deltas.extend([run_id, flush_id, tick, c, t, repr(v), "removed"] for v in before[t])
                if t in now and (t not in before or now[t] != before[t]):
                    deltas.extend([run_id, flush_id, tick, c, t, repr(v), "added"] for v in now[t])
            previous[c] = dict(now)
            digests.append([run_id, flush_id, tick, c, sum(len(v) for v in now.values()), _digest(now)])

    flush_id = 0
    for t in range(n_ticks):
        for c in channels:
            v = float(values[c][t])
            pending[c][t] = v
            samples.append([run_id, c, t, repr(v), _wall(t), _wall(t, 1), t + server_offset.get((c, t), 0),
                            _wall(t, 2), _wall(t, 3), _wall(t, 4), t])
        if t in unscheduled:
            seal()
        if t in regular:
            flush(t, flush_id)
            flush_id += 1
            t_f = t
        for c in channels:
            copies = {**stored[c], **{s: [v] for s, v in pending[c].items()}}
            visible = {s: vs[0] for s, vs in copies.items()}  # anchor queries (LIMIT 1) return one copy
            times = sorted(visible)
            for s in times:
                if t - 8 <= s <= t:
                    snaps.extend([run_id, t, c, s, repr(v), "window", _wall(t, 5)] for v in copies[s])
            before = [s for s in times if s < t - 8]
            anchor = before[-1] if before else None
            if anchor is not None:
                snaps.append([run_id, t, c, anchor, repr(visible[anchor]), "anchor", _wall(t, 5)])
            if emit_commit_anchor and t_f is not None:
                eligible = [s for s in times if s <= min(t_f, t - 9)]
                if eligible and eligible[-1] != anchor:
                    snaps.append([run_id, t, c, eligible[-1], repr(visible[eligible[-1]]), "commit_anchor",
                                  _wall(t, 5)])
    flush(last + 1, flush_id)

    meta = {"run_id": run_id, "fault": 0, "batch": 0, "n_ticks": n_ticks, "channels": channels,
            "deadband": deadband, "compdev": compdev or {c: None for c in channels}, "flush_every": F,
            "flush_ticks": flush_ticks, "iotdb_version": "fixture", "iotdb_settings": {"auto_flush": "off"},
            "asyncua_version": "fixture", "tick_wall_ms": 0, "testbed_code_digest": "fixture",
            "created_wall": _wall(0)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tables = {
        "samples.csv": (["run_id", "channel", "tick", "value_sent", "opc_source_ts", "opc_server_ts", "server_tick",
                         "recv_wall", "write_req_wall", "write_ack_wall", "write_ack_tick"], samples),
        "flushes.csv": (["run_id", "flush_id", "tick", "issue_wall", "return_wall", "tsfiles_before",
                         "tsfiles_after"], flushes),
        "membership_delta.csv": (["run_id", "flush_id", "tick", "channel", "row_time", "value", "change"], deltas),
        "membership_digest.csv": (["run_id", "flush_id", "tick", "channel", "n_rows", "digest"], digests),
        "snapshots.csv": (["run_id", "origin", "channel", "row_time", "value", "kind", "query_wall"], snaps),
    }
    for name, (header, rows) in tables.items():
        with open(out / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
    (out / "events.log").write_text("fixture\n", encoding="utf-8")
    (out / "DONE").write_text("", encoding="utf-8")
