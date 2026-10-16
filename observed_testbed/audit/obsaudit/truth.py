"""Snapshot-side truth built only from snapshots.csv, flushes.csv and run_meta.json.

Snapshot rows are a multiset: identical copies of a row collapse to one row (counted); one time with two values is
an error. S_all(o) = rows returned at origin o (window and anchor). S_commit(o) = returned rows with time <= t_f(o), the tick
of the last barrier flush at or before o; its anchor is the snapshot anchor when that row is committed, otherwise
the commit_anchor row. Without a committed anchor where one is needed the cell is 'truth_undetermined'.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field

from .io_run import Issues, TruthInputs
from .reconstruct import WINDOW, Row

BEFORE_FIRST_FLUSH = "before_first_flush"
UNDETERMINED = "truth_undetermined"


@dataclass
class TruthCell:
    s_all: tuple[Row, ...]
    s_commit: tuple[Row, ...] | None
    status: str  # "ok", BEFORE_FIRST_FLUSH or UNDETERMINED
    t_f: int | None
    window_copies: dict[int, int] = field(default_factory=dict)  # copies returned per window row time


@dataclass
class Truth:
    cells: dict[tuple[int, str], TruthCell]
    violations: list[dict] = field(default_factory=list)
    n_commit_anchor_rows: int = 0
    duplicates: dict[str, dict[str, int]] = field(default_factory=dict)  # channel -> kind -> (origin, time) count


def build_truth(inp: TruthInputs, issues: Issues) -> Truth:
    channels = list(inp.meta["channels"])
    n_ticks = int(inp.meta["n_ticks"])
    schedule = inp.schedule
    snaps = inp.snapshots
    grouped = {key: g for key, g in snaps.groupby(["origin", "channel"], sort=False)}
    truth = Truth(cells={}, n_commit_anchor_rows=int((snaps["kind"] == "commit_anchor").sum()))
    bad_rows: list[dict] = []
    conflicts: list[dict] = []
    # Committed rows seen so far, for the "never disappear or change" check over later snapshots.
    known: dict[str, dict[int, tuple[float, int]]] = {c: {} for c in channels}
    known_times: dict[str, list[int]] = {c: [] for c in channels}

    for origin in range(n_ticks):
        t_f = schedule.last_flush_at_or_before(origin)
        for ch in channels:
            g = grouped.get((origin, ch))
            copies: dict[str, dict[int, list]] = {"window": {}, "anchor": {}, "commit_anchor": {}}
            if g is not None:
                for r in g.itertuples(index=False):
                    if r.kind not in copies:
                        continue  # reported by the loader
                    t, v = int(r.row_time), float(r.value)
                    entry = copies[r.kind].get(t)
                    if entry is None:
                        copies[r.kind][t] = [v, 1]
                    elif entry[0] == v:
                        entry[1] += 1  # identical copy (IoTDB one-row SDT segment); collapses to one row
                    else:
                        conflicts.append({"origin": origin, "channel": ch, "kind": r.kind, "row_time": t,
                                          "values": [entry[0], v]})
            window = sorted((t, e[0]) for t, e in copies["window"].items())
            anchors = sorted((t, e[0]) for t, e in copies["anchor"].items())
            c_anchors = sorted((t, e[0]) for t, e in copies["commit_anchor"].items())
            limit = min(t_f, origin - WINDOW - 1) if t_f is not None else None
            for t, _ in window:
                if not origin - WINDOW <= t <= origin:
                    bad_rows.append({"origin": origin, "channel": ch, "kind": "window", "row_time": t})
            if len(anchors) > 1 or any(t >= origin - WINDOW for t, _ in anchors):
                bad_rows.append({"origin": origin, "channel": ch, "kind": "anchor", "rows": anchors})
            if len(c_anchors) > 1 or any(limit is None or t > limit for t, _ in c_anchors):
                bad_rows.append({"origin": origin, "channel": ch, "kind": "commit_anchor", "rows": c_anchors})
            anchor = anchors[-1] if anchors else None
            commit_anchor = c_anchors[-1] if c_anchors else None
            window_copies = {t: e[1] for t, e in copies["window"].items()}
            for kind, rows in copies.items():
                n_dup = sum(1 for e in rows.values() if e[1] > 1)
                if n_dup:
                    dup = truth.duplicates.setdefault(ch, {})
                    dup[kind] = dup.get(kind, 0) + n_dup
            s_all = tuple(([anchor] if anchor else []) + window)
            if t_f is None:
                truth.cells[(origin, ch)] = TruthCell(s_all, None, BEFORE_FIRST_FLUSH, None, window_copies)
                continue
            if anchor is not None and anchor[0] <= t_f:
                if commit_anchor is not None:  # interface: omitted when it is the same row as the anchor
                    bad_rows.append({"origin": origin, "channel": ch, "kind": "commit_anchor", "redundant": True})
                c_anchor, status = anchor, "ok"
            elif anchor is None:
                c_anchor, status = None, "ok"  # no visible row before o-8, hence no committed one
            elif commit_anchor is not None:
                c_anchor, status = commit_anchor, "ok"
            else:
                c_anchor, status = None, UNDETERMINED
            committed_window = [row for row in window if row[0] <= t_f]
            s_commit = tuple(([c_anchor] if c_anchor else []) + committed_window)
            truth.cells[(origin, ch)] = TruthCell(s_all, s_commit if status == "ok" else None, status, t_f,
                                                  window_copies)
            _check_committed_persist(truth, known[ch], known_times[ch], ch, origin, window, anchor, commit_anchor)
            for t, v in s_commit:
                if t not in known[ch]:
                    known[ch][t] = (v, origin)
                    known_times[ch].insert(bisect_left(known_times[ch], t), t)
    if conflicts:
        issues.add("error", "snap:same_time_values", f"{len(conflicts)} snapshot times carry different values",
                   conflicts)
    if bad_rows:
        issues.add("error", "snap:structure", f"{len(bad_rows)} snapshot rows break the window/anchor rules", bad_rows)
    return truth


def _check_committed_persist(truth: Truth, known: dict, known_times: list, ch: str, origin: int,
                             window: list[Row], anchor: Row | None, commit_anchor: Row | None) -> None:
    """Every committed row seen earlier must still be returned, unchanged, wherever this snapshot would show it."""
    returned = dict(window)
    for row in (anchor, commit_anchor):
        if row is not None:
            returned[row[0]] = row[1]
    # Rows older than the oldest returned row before o-8 are not shown by construction. Every committed row from
    # that row on must be returned: a committed row between it and o-8 would itself have been the (commit) anchor.
    if commit_anchor is not None:
        floor = commit_anchor[0]
    elif anchor is not None:
        floor = anchor[0]
    else:
        floor = None
    start = 0 if floor is None else bisect_left(known_times, floor)
    for t in known_times[start:]:
        if t > origin:
            break
        value, first_origin = known[t]
        if t not in returned:
            truth.violations.append({"type": "committed_row_missing_in_snapshot", "channel": ch, "row_time": t,
                                     "value": value, "first_committed_origin": first_origin, "origin": origin})
        elif returned[t] != value:
            truth.violations.append({"type": "committed_row_changed_in_snapshot", "channel": ch, "row_time": t,
                                     "value": value, "new_value": returned[t], "first_committed_origin": first_origin,
                                     "origin": origin})
