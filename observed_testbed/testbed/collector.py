"""Collector: OPC UA subscription client -> IoTDB writes, scheduled FLUSH and post-flush membership query.

Notifications are queued by the subscription handler; receive(tick) blocks until every channel delivered the value
whose SourceTimestamp is the process time of that tick (anything else is an error, never silently skipped).
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from datetime import datetime, timezone

from asyncua import Client, ua

from historian_io import Historian
from opcua_source import process_time


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    if dt.tzinfo is None:  # asyncua may return naive UTC datetimes
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def digest(rows: list[tuple[int, float]]) -> str:
    """INTERFACE.md pinned convention: numeric sort by row_time, f"{t},{repr(v)}", "\\n"-joined, sha256 hex."""
    lines = [f"{t},{v!r}" for t, v in sorted(rows, key=lambda r: (r[0], repr(r[1])))]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() if lines else ""


class _Handler:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def datachange_notification(self, node, val, data) -> None:
        self.queue.put_nowait((node.nodeid.Identifier, data.monitored_item.Value, now_utc()))

    def status_change_notification(self, status) -> None:
        self.queue.put_nowait(("__status__", status, now_utc()))


class Collector:
    def __init__(self, endpoint: str, channels: list[str], historian: Historian, recv_timeout_s: float = 30.0):
        self.endpoint = endpoint
        self.channels = channels
        self.h = historian
        self.recv_timeout_s = recv_timeout_s
        self.queue: asyncio.Queue = asyncio.Queue()
        self.client = Client(endpoint, timeout=30)
        self.prev_members: dict[str, Counter] = {c: Counter() for c in channels}
        self.last_members: dict[str, list[tuple[int, float]]] = {c: [] for c in channels}

    async def connect(self) -> list[str]:
        await self.client.connect()
        idx = await self.client.get_namespace_index("urn:historian-observed-testbed:tep")
        nodes = [self.client.get_node(ua.NodeId(ch, idx)) for ch in self.channels]
        # publishing interval 0: the server publishes each change as soon as it is queued (no timer).
        self.sub = await self.client.create_subscription(0, _Handler(self.queue))
        flt = ua.DataChangeFilter(Trigger=ua.DataChangeTrigger.StatusValueTimestamp)
        handles = await self.sub._subscribe(nodes, ua.AttributeIds.Value, mfilter=flt, queuesize=100,
                                            sampling_interval=0.0)
        if any(isinstance(h, ua.StatusCode) for h in handles):
            raise RuntimeError(f"monitored item creation failed: {handles}")
        # the server sends each item's current (initial) value once on subscription; drain exactly those.
        drained = []
        for _ in self.channels:
            ch, dv, _w = await asyncio.wait_for(self.queue.get(), self.recv_timeout_s)
            drained.append(f"{ch}:{dv.Value.Value!r}@{dv.SourceTimestamp}")
        return drained

    async def receive(self, tick: int) -> dict[str, tuple[float, datetime, datetime, datetime]]:
        want = process_time(tick)
        got: dict[str, tuple[float, datetime, datetime, datetime]] = {}
        while len(got) < len(self.channels):
            ch, dv, recv = await asyncio.wait_for(self.queue.get(), self.recv_timeout_s)
            if ch == "__status__":
                raise RuntimeError(f"subscription status change at tick {tick}: {dv}")
            src = dv.SourceTimestamp if dv.SourceTimestamp.tzinfo else dv.SourceTimestamp.replace(tzinfo=timezone.utc)
            if ch not in self.channels or ch in got or src != want:
                raise RuntimeError(f"unexpected notification at tick {tick}: {ch} {src} (want {want})")
            if dv.StatusCode is not None and not dv.StatusCode.is_good():
                raise RuntimeError(f"bad status at tick {tick}: {ch} {dv.StatusCode}")
            got[ch] = (float(dv.Value.Value), src, dv.ServerTimestamp, recv)
        return got

    def write(self, channel: str, tick: int, value: float) -> tuple[datetime, datetime]:
        req = now_utc()
        self.h.write(channel, tick, value)
        return req, now_utc()

    def flush(self) -> tuple[datetime, datetime]:
        issue = now_utc()
        self.h.flush()
        return issue, now_utc()

    def membership(self) -> tuple[list[tuple[str, int, float, str]], dict[str, tuple[int, str]]]:
        """Full post-flush query. Returns delta lines (channel, row_time, value, change), one per stored copy,
        and {channel: (n_rows, digest)}."""
        full = self.h.full(self.channels)
        delta: list[tuple[str, int, float, str]] = []
        summary: dict[str, tuple[int, str]] = {}
        for ch in self.channels:
            rows = full[ch]
            cur = Counter((t, repr(v)) for t, v in rows)
            prev = self.prev_members[ch]
            for change, diff in (("removed", prev - cur), ("added", cur - prev)):
                for (t, vr), k in sorted(diff.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                    delta.extend([(ch, t, float(vr), change)] * k)
            self.prev_members[ch] = cur
            self.last_members[ch] = rows
            summary[ch] = (len(rows), digest(rows))
        return delta, summary

    def duplicate_rows(self) -> dict[str, int]:
        """Extra stored copies in the last membership query: n_rows - number of distinct (row_time, value)."""
        return {ch: len(rows) - len({(t, repr(v)) for t, v in rows}) for ch, rows in self.last_members.items()}

    async def close(self) -> None:
        try:
            await self.sub.delete()
        finally:
            await self.client.disconnect()
