"""Replay-side ledger built only from samples.csv, membership files, flushes.csv and run_meta.json.

Membership is treated as a multiset (identical copies of a row are counted and digested, then collapsed).
Per sample: visible from its write-ack tick; observed commit tick c_obs = tick of the first post-flush membership
query containing it; discard tick = tick of the first membership query after its write that lacks it (before any
commit); final retained flag = presence in the final membership. Committed rows must never disappear or change.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left, insort
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io_run import Issues, LedgerInputs

INF = math.inf


@dataclass
class ChannelArrays:
    """Rows of one view for one channel, sorted by time; row k is active at origin o iff start <= o < end."""

    time: np.ndarray
    value: np.ndarray
    start: np.ndarray
    end: np.ndarray
    copies: np.ndarray | None = None  # stored copies (Replay_visible only; identical copies count once in R)


@dataclass
class Ledger:
    rows: pd.DataFrame
    violations: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    visible: dict[str, ChannelArrays] = field(default_factory=dict)
    n_memberships: int = 0
    n_digests_checked: int = 0
    duplicates: dict = field(default_factory=dict)


def digest_lines(lines_in_time_order: list[str]) -> str:
    """Pinned convention: numeric row_time order, '\\n'-joined, no trailing newline, UTF-8, lowercase hex sha256."""
    return hashlib.sha256("\n".join(lines_in_time_order).encode("utf-8")).hexdigest()


def _replay_membership(inp: LedgerInputs, issues: Issues):
    """Apply cumulative deltas in flush order, verify every digest, and collect per-row membership events."""
    schedule = inp.schedule
    channels = list(inp.meta["channels"])
    # Membership is a multiset: IoTDB 1.3.7 stores a one-row SDT segment twice (BOUNDARY_SMOKE.md finding 5).
    state: dict[str, dict[int, list]] = {c: {} for c in channels}  # time -> [value, copies]
    order: dict[str, list[int]] = {c: [] for c in channels}
    events: dict[tuple[str, int], list[tuple[int, int, float]]] = {}  # (flush index, copies after, value)
    duplicated: dict[tuple[str, int], int] = {}  # max copies seen, for rows stored more than once
    delta_by_flush = {k: g for k, g in inp.delta.groupby("flush_id", sort=False)}
    digest_by_key = {(int(r.flush_id), r.channel): r for r in inp.digest.itertuples(index=False)}
    n_checked = 0
    bad_digest: list[dict] = []
    for k_index, flush_id in enumerate(schedule.flush_ids):
        group = delta_by_flush.get(flush_id)
        if group is not None:
            # Removals first so that a value change (removed + added of one row in one flush) applies cleanly.
            group = group.assign(_order=(group["change"] == "added").astype(int)).sort_values("_order", kind="stable")
            for r in group.itertuples(index=False):
                ch, t, v = r.channel, int(r.row_time), float(r.value)
                if ch not in state:
                    issues.add("error", "delta:channel", f"unknown channel {ch}", [r._asdict()])
                    continue
                entry = state[ch].get(t)
                if r.change == "removed":
                    if entry is None:
                        issues.add("error", "delta:remove_absent", "removed row was not present", [r._asdict()])
                        continue
                    if entry[0] != v:
                        issues.add("error", "delta:remove_value", "removed value differs from the stored value",
                                   [r._asdict()])
                    entry[1] -= 1
                    if entry[1] == 0:
                        del state[ch][t]
                        order[ch].pop(bisect_left(order[ch], t))
                    events.setdefault((ch, t), []).append((k_index, entry[1], entry[0]))
                elif entry is None:
                    state[ch][t] = [v, 1]
                    insort(order[ch], t)
                    events.setdefault((ch, t), []).append((k_index, 1, v))
                elif entry[0] != v:
                    issues.add("error", "delta:same_time_values", "added a second value at a stored time",
                               [r._asdict()])
                else:
                    entry[1] += 1
                    duplicated[(ch, t)] = max(duplicated.get((ch, t), 0), entry[1])
                    events.setdefault((ch, t), []).append((k_index, entry[1], v))
        for ch in channels:
            expected = digest_by_key.get((flush_id, ch))
            if expected is None:
                issues.add("error", "digest:missing", f"no digest row for flush {flush_id} channel {ch}")
                continue
            lines = []
            for t in order[ch]:
                value, copies = state[ch][t]
                lines.extend([f"{t},{repr(value)}"] * copies)
            n_checked += 1
            if digest_lines(lines) != expected.digest or len(lines) != int(expected.n_rows):
                bad_digest.append({"flush_id": flush_id, "channel": ch, "n_rows": len(lines),
                                   "expected_n_rows": int(expected.n_rows)})
    if bad_digest:
        issues.add("error", "digest:mismatch", f"{len(bad_digest)} cumulative memberships do not reproduce the digest",
                   bad_digest)
    extra = set(digest_by_key) - {(f, c) for f in schedule.flush_ids for c in channels}
    if extra:
        issues.add("error", "digest:extra", "digest rows for unknown flush/channel", sorted(extra)[:10])
    return events, n_checked, duplicated


def build_ledger(inp: LedgerInputs, issues: Issues) -> Ledger:
    schedule = inp.schedule
    last_tick = int(inp.meta["n_ticks"]) - 1
    # Barrier position of each membership query; the final one is post-run (labelled last_tick + 1).
    eff = np.asarray(schedule.ticks, dtype=float)
    eff[-1] = last_tick + 1
    n_flush = len(eff)
    final_index = n_flush - 1
    events, n_checked, duplicated = _replay_membership(inp, issues)
    ledger = Ledger(rows=pd.DataFrame(), n_memberships=n_flush, n_digests_checked=n_checked,
                    duplicates={"n_rows_stored_more_than_once": len(duplicated),
                                "max_copies": max(duplicated.values(), default=1),
                                "by_channel": {c: sum(1 for (ch, _) in duplicated if ch == c)
                                               for c in inp.meta["channels"]}})

    samples = inp.samples.sort_values(["channel", "tick"], kind="stable")
    sample_keys = set(zip(samples["channel"], samples["tick"]))
    for (ch, t), ev in events.items():
        if (ch, t) not in sample_keys:
            ledger.violations.append({"type": "phantom_row", "channel": ch, "row_time": t,
                                      "first_flush_tick": float(eff[ev[0][0]])})

    records = []
    visible_parts: dict[str, list[tuple[int, float, float, float, int]]] = {}
    for r in samples.itertuples(index=False):
        ch, tau, ack, sent = r.channel, int(r.tick), int(r.write_ack_tick), float(r.value_sent)
        ev = events.get((ch, tau), [])
        m1 = int(np.searchsorted(eff, ack, side="left"))  # first membership query at or after the write ack
        # Membership state after each flush that touched the row: (flush index, copies, value).
        timeline: list[tuple[int, int, float]] = []
        for k_index, copies, v in ev:
            if k_index < m1:
                ledger.violations.append({"type": "row_before_write", "channel": ch, "row_time": tau,
                                          "flush_tick": float(eff[k_index]), "write_ack_tick": ack})
            if timeline and timeline[-1][0] == k_index:
                timeline[-1] = (k_index, copies, v)
            else:
                timeline.append((k_index, copies, v))

        def state_at(k: int) -> tuple[int, float]:
            copies, val = 0, math.nan
            for k_index, c_after, v_after in timeline:
                if k_index > k:
                    break
                copies, val = c_after, v_after
            return copies, val

        copies_m1, value_m1 = state_at(m1)
        c_index = m1 if copies_m1 else next((k for k, c, _ in timeline if k > m1 and c), None)
        discard_index = None if copies_m1 else m1
        commit_value = state_at(c_index)[1] if c_index is not None else math.nan
        final_copies, final_value = state_at(final_index)
        removed_after = None
        if c_index is not None:
            for k_index, c_after, v_after in timeline:
                if k_index <= c_index:
                    continue
                if not c_after and removed_after is None:
                    removed_after = k_index
                    ledger.violations.append({
                        "type": "committed_row_removed", "channel": ch, "row_time": tau, "value": commit_value,
                        "c_obs": float(eff[c_index]), "removed_at_tick": float(eff[k_index])})
                elif c_after and v_after != commit_value:
                    ledger.violations.append({
                        "type": "committed_row_changed", "channel": ch, "row_time": tau, "value": commit_value,
                        "new_value": v_after, "c_obs": float(eff[c_index]), "changed_at_tick": float(eff[k_index])})
            if discard_index is not None:
                ledger.anomalies.append({"type": "committed_after_discard", "channel": ch, "row_time": tau,
                                         "discard_tick": float(eff[discard_index]), "c_obs": float(eff[c_index])})
            if commit_value != sent:
                ledger.anomalies.append({"type": "stored_value_differs_from_sent", "channel": ch, "row_time": tau,
                                         "value_sent": sent, "stored": commit_value})
        records.append({
            "channel": ch, "tick": tau, "value_sent": sent, "write_ack_tick": ack, "server_tick": int(r.server_tick),
            "c_obs": float(eff[c_index]) if c_index is not None else math.nan,
            "commit_at_final_flush": c_index == final_index,
            "commit_value": commit_value,
            "discard_tick": float(eff[discard_index]) if discard_index is not None else math.nan,
            "removed_after_commit_tick": float(eff[removed_after]) if removed_after is not None else math.nan,
            "final_retained": final_copies > 0, "final_value": final_value if final_copies else math.nan,
            "final_copies": final_copies,
        })
        # Visibility intervals (time, value, start, end, copies): one provisional copy from the write ack until the
        # first membership query, then the membership state; a new interval starts when copies or value change.
        parts = visible_parts.setdefault(ch, [])
        if ack < eff[m1]:
            parts.append((tau, sent, float(ack), float(eff[m1]), 1))
        seg = (float(eff[m1]), value_m1, copies_m1) if copies_m1 else None
        for k_index, c_after, v_after in timeline:
            if k_index <= m1:
                continue
            if seg is not None and (c_after, v_after) != (seg[2], seg[1]):
                parts.append((tau, seg[1], seg[0], float(eff[k_index]), seg[2]))
                seg = None
            if c_after and seg is None:
                seg = (float(eff[k_index]), v_after, c_after)
        if seg is not None:
            parts.append((tau, seg[1], seg[0], INF, seg[2]))
    ledger.rows = pd.DataFrame.from_records(records)
    for ch, parts in visible_parts.items():
        parts.sort(key=lambda p: (p[0], p[2]))
        arr = np.asarray(parts, dtype=float).reshape(-1, 5)
        ledger.visible[ch] = ChannelArrays(arr[:, 0].astype(np.int64), arr[:, 1], arr[:, 2], arr[:, 3],
                                           arr[:, 4].astype(np.int64))
    return ledger


def view_arrays(ledger: Ledger, view: str) -> dict[str, ChannelArrays]:
    """Per-channel availability arrays for the replay-side views of BRIEF task 5."""
    if view == "Replay_visible":
        return ledger.visible
    out = {}
    for ch, g in ledger.rows.groupby("channel", sort=False):
        g = g.sort_values("tick")
        if view == "Replay_commit":
            g = g.loc[g["c_obs"].notna()]
            start, value = g["c_obs"].to_numpy(float), g["commit_value"].to_numpy(float)
        else:
            g = g.loc[g["final_retained"]]
            value = g["final_value"].to_numpy(float)
            if view == "V3":
                start = g["tick"].to_numpy(float)
            elif view == "V_server":
                start = g["server_tick"].to_numpy(float)
            elif view == "V1":
                start = np.full(len(g), -INF)
            else:
                raise ValueError(view)
        out[ch] = ChannelArrays(g["tick"].to_numpy(np.int64), value, start, np.full(len(g), INF))
    return out


def commit_lag_summary(ledger: Ledger) -> dict:
    """Distribution of c_obs - tau over rows committed by a barrier flush; final-flush commits counted apart."""
    rows = ledger.rows
    committed = rows.loc[rows["c_obs"].notna()]
    regular = committed.loc[~committed["commit_at_final_flush"]]
    lag = (regular["c_obs"] - regular["tick"]).to_numpy(float)
    out: dict = {"n_committed_regular": int(len(lag)), "n_committed_at_final_flush": int(committed["commit_at_final_flush"].sum())}
    if len(lag):
        q = np.percentile(lag, [50, 90, 99])
        values, counts = np.unique(lag.astype(int), return_counts=True)
        out.update({"min": float(lag.min()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                    "max": float(lag.max()), "mean": float(lag.mean()),
                    "histogram": {str(int(v)): int(c) for v, c in zip(values, counts)}})
    return out


def retention_summary(ledger: Ledger) -> dict:
    out = {}
    for ch, g in ledger.rows.groupby("channel", sort=True):
        out[ch] = {"n_samples": int(len(g)), "n_final_retained": int(g["final_retained"].sum()),
                   "achieved_retention": float(g["final_retained"].mean()),
                   "n_ever_committed": int(g["c_obs"].notna().sum()),
                   "n_discarded": int(g["discard_tick"].notna().sum())}
    return out
