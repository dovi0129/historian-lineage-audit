"""Thin IoTDB 1.3 tree-model helpers shared by the collector, the query client and the boundary smoke.

Times inside IoTDB are milliseconds: time_ms = T0_MS + tick * 60000. T0_MS is the start of a 7-day time partition
(time_partition_interval = 604800000 ms, origin 0), so a 3001-tick run never crosses a partition boundary.
"""
from __future__ import annotations

from iotdb.Session import Session
from iotdb.utils.IoTDBConstants import TSDataType

TICK_MS = 60_000
T0_MS = 2_800 * 604_800_000  # partition start, 2023-08-31T00:00:00Z
assert T0_MS % 604_800_000 == 0


def tick_to_ms(tick: int) -> int:
    return T0_MS + tick * TICK_MS


def ms_to_tick(ms: int) -> int:
    q, r = divmod(ms - T0_MS, TICK_MS)
    if r:
        raise ValueError(f"time {ms} is not on the tick grid")
    return q


def meas_name(channel: str) -> str:
    return channel.replace("-", "_")


def open_session(port: int) -> Session:
    s = Session("127.0.0.1", port, "root", "root", fetch_size=100_000)
    s.open(False)
    return s


def _rows(session: Session, sql: str, n_cols: int) -> list[tuple[int, list[float | None]]]:
    ds = session.execute_query_statement(sql)
    out = []
    try:
        while ds.has_next():
            rec = ds.next()
            vals: list[float | None] = []
            for f in rec.get_fields():
                v = getattr(f, "value", None)
                vals.append(None if v is None or f.get_data_type() is None else float(v))
            if len(vals) != n_cols:
                raise RuntimeError(f"expected {n_cols} columns, got {len(vals)}: {sql}")
            out.append((rec.get_timestamp(), vals))
    finally:
        ds.close_operation_handle()
    return out


class Historian:
    """Database `root.<db>`, device `root.<db>.<device>`; one DOUBLE series per channel."""

    def __init__(self, session: Session, db: str, device: str = "d"):
        self.s = session
        self.db = f"root.{db}"
        self.device = f"{self.db}.{device}"

    def create(self, channels: list[str], compdev: dict[str, float | None], create_db: bool = True) -> list[str]:
        stmts = [f"CREATE DATABASE {self.db}"] if create_db else []
        for ch in channels:
            cd = compdev.get(ch)
            sql = f"CREATE TIMESERIES {self.device}.{meas_name(ch)} WITH DATATYPE=DOUBLE, ENCODING=PLAIN"
            if cd is not None:
                sql += f", DEADBAND=SDT, COMPDEV={cd!r}"
            stmts.append(sql)
        for sql in stmts:
            self.s.execute_non_query_statement(sql)
        return stmts

    def show_timeseries(self) -> list[str]:
        ds = self.s.execute_query_statement(f"SHOW TIMESERIES {self.device}.**")
        out = []
        try:
            while ds.has_next():
                out.append(str(ds.next()))
        finally:
            ds.close_operation_handle()
        return out

    def write(self, channel: str, tick: int, value: float) -> None:
        self.s.insert_record(self.device, tick_to_ms(tick), [meas_name(channel)], [TSDataType.DOUBLE], [value])

    def flush(self) -> None:
        self.s.execute_non_query_statement(f"FLUSH {self.db}")

    # One query per series: in IoTDB 1.3.7 a multi-series raw query on this device times out (60 s) as soon as one
    # series holds a doubled row from a one-row SDT flush segment; single-series queries return correctly
    # (BOUNDARY_SMOKE.md finding 7).
    def _series(self, channel: str, where: str = "") -> list[tuple[int, float]]:
        sql = f"SELECT {meas_name(channel)} FROM {self.device}{where} ORDER BY TIME ASC"
        return [(ms_to_tick(ms), vals[0]) for ms, vals in _rows(self.s, sql, 1) if vals[0] is not None]

    def full(self, channels: list[str]) -> dict[str, list[tuple[int, float]]]:
        """All rows of every series, per channel, ascending time (ticks), duplicates as returned."""
        return {c: self._series(c) for c in channels}

    def window(self, channels: list[str], lo: int, hi: int) -> dict[str, list[tuple[int, float]]]:
        """Rows with lo <= tick <= hi, per channel, duplicates as returned."""
        where = f" WHERE time >= {tick_to_ms(lo)} AND time <= {tick_to_ms(hi)}"
        return {c: self._series(c, where) for c in channels}

    def latest_le(self, channel: str, hi: int) -> tuple[int, float] | None:
        """Latest row with tick <= hi (raw rows, not aggregate statistics)."""
        sql = (f"SELECT {meas_name(channel)} FROM {self.device} WHERE time <= {tick_to_ms(hi)} "
               f"ORDER BY TIME DESC LIMIT 1")
        rows = _rows(self.s, sql, 1)
        if not rows or rows[0][1][0] is None:
            return None
        return ms_to_tick(rows[0][0]), rows[0][1][0]
