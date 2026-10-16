"""Query client: origin snapshot of what IoTDB returns at origin o (own session; never reads membership results).

For each channel: window rows o-8 <= time <= o; anchor = latest row with time <= o-9; commit_anchor = latest row with
time <= min(t_f, o-9), t_f = last scheduled flush tick <= o, omitted when no flush has happened, when no such row
exists, or when it is the anchor row (INTERFACE.md item 6).
"""
from __future__ import annotations

from datetime import datetime, timezone

from historian_io import Historian

LAG_MAX = 8


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class QueryClient:
    def __init__(self, historian: Historian, channels: list[str]):
        self.h = historian
        self.channels = channels

    def snapshot(self, origin: int, t_f: int | None) -> list[tuple[int, str, int, float, str, str]]:
        """Rows (origin, channel, row_time, value, kind, query_wall) in channel order, window rows ascending."""
        out: list[tuple[int, str, int, float, str, str]] = []
        win = self.h.window(self.channels, origin - LAG_MAX, origin)
        w_wall = now_iso()
        for ch in self.channels:
            for t, v in win[ch]:
                out.append((origin, ch, t, v, "window", w_wall))
            hi = origin - LAG_MAX - 1
            if hi < 0:
                continue
            anchor = self.h.latest_le(ch, hi)
            a_wall = now_iso()
            if anchor is not None:
                out.append((origin, ch, anchor[0], anchor[1], "anchor", a_wall))
            if t_f is None or t_f >= hi:
                continue  # no flush yet, or min(t_f, o-9) = o-9: the commit anchor would be the anchor row
            ca = self.h.latest_le(ch, t_f)
            ca_wall = now_iso()
            if ca is not None and (anchor is None or (ca[0], repr(ca[1])) != (anchor[0], repr(anchor[1]))):
                out.append((origin, ch, ca[0], ca[1], "commit_anchor", ca_wall))
        return out
