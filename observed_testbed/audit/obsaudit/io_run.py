"""Reader and structural validator for one run directory (INTERFACE.md).

The two loaders below are the only places that open run files. `load_ledger_inputs` never opens snapshots.csv and
`load_truth_inputs` never opens samples.csv or the membership files (spec §8, two independent paths).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SAMPLES_COLS = [
    "run_id", "channel", "tick", "value_sent", "opc_source_ts", "opc_server_ts", "server_tick", "recv_wall",
    "write_req_wall", "write_ack_wall", "write_ack_tick",
]
FLUSH_COLS = ["run_id", "flush_id", "tick", "issue_wall", "return_wall", "tsfiles_before", "tsfiles_after"]
DELTA_COLS = ["run_id", "flush_id", "tick", "channel", "row_time", "value", "change"]
DIGEST_COLS = ["run_id", "flush_id", "tick", "channel", "n_rows", "digest"]
SNAP_COLS = ["run_id", "origin", "channel", "row_time", "value", "kind", "query_wall"]
META_KEYS = [
    "run_id", "fault", "batch", "n_ticks", "channels", "deadband", "compdev", "flush_every", "flush_ticks",
    "iotdb_version", "iotdb_settings", "asyncua_version", "tick_wall_ms", "testbed_code_digest", "created_wall",
]
SNAP_KINDS = {"window", "anchor", "commit_anchor"}


class RunFormatError(Exception):
    """A run directory cannot be loaded at all."""


@dataclass
class Issues:
    """Validation findings; level 'error' invalidates the run, 'warning' is reported only."""

    items: list[dict] = field(default_factory=list)

    def add(self, level: str, check: str, message: str, rows: list | None = None) -> None:
        item = {"level": level, "check": check, "message": message, "rows": (rows or [])[:20]}
        if item not in self.items:  # both loaders read flushes.csv; report each finding once
            self.items.append(item)

    @property
    def n_errors(self) -> int:
        return sum(item["level"] == "error" for item in self.items)


@dataclass
class FlushSchedule:
    """Flush ticks split into barrier flushes (count toward t_f(o)) and the post-run final flush."""

    flush_ids: list[int]
    ticks: list[int]
    final_flush_id: int
    regular_ticks: list[int]

    def last_flush_at_or_before(self, origin: int) -> int | None:
        best = None
        for tick in self.regular_ticks:
            if tick <= origin:
                best = tick
            else:
                break
        return best


@dataclass
class LedgerInputs:
    meta: dict
    flushes: pd.DataFrame
    schedule: FlushSchedule
    samples: pd.DataFrame
    delta: pd.DataFrame
    digest: pd.DataFrame


@dataclass
class TruthInputs:
    meta: dict
    flushes: pd.DataFrame
    schedule: FlushSchedule
    snapshots: pd.DataFrame


def _read_csv(path: Path, columns: list[str], issues: Issues) -> pd.DataFrame:
    if not path.is_file():
        raise RunFormatError(f"missing file {path.name}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(frame.columns) != columns:
        missing = [c for c in columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in columns]
        level = "error" if missing else "warning"
        issues.add(level, f"columns:{path.name}", f"expected {columns}; missing {missing}; extra {extra}")
        if missing:
            raise RunFormatError(f"{path.name} lacks columns {missing}")
    return frame


def _to_int(frame: pd.DataFrame, column: str, name: str) -> None:
    try:
        frame[column] = frame[column].astype("int64")
    except (ValueError, TypeError) as exc:
        raise RunFormatError(f"{name}.{column} is not integer: {exc}") from exc


def _to_float(frame: pd.DataFrame, column: str, name: str, issues: Issues) -> None:
    """Parse with Python float (exact for repr output) and flag strings that are not repr(float)."""
    text = frame[column]
    try:
        values = [float(s) for s in text]
    except ValueError as exc:
        raise RunFormatError(f"{name}.{column} is not float: {exc}") from exc
    bad = [i for i, (s, v) in enumerate(zip(text, values)) if repr(v) != s]
    if bad:
        issues.add("warning", f"repr:{name}.{column}", f"{len(bad)} values are not written as repr(float)",
                   [frame.iloc[i].to_dict() for i in bad[:5]])
    frame[column] = pd.Series(values, index=frame.index, dtype="float64")


def read_meta(run_dir: Path, issues: Issues) -> dict:
    path = run_dir / "run_meta.json"
    if not path.is_file():
        raise RunFormatError("missing run_meta.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        issues.add("error", "meta:keys", f"run_meta.json lacks {missing}")
        for key in ("run_id", "n_ticks", "channels", "flush_every"):
            if key in missing:
                raise RunFormatError(f"run_meta.json lacks {key}")
    if meta["run_id"] != run_dir.name:
        issues.add("warning", "meta:run_id", f"run_id {meta['run_id']} differs from directory {run_dir.name}")
    return meta


def read_flushes(run_dir: Path, meta: dict, issues: Issues) -> tuple[pd.DataFrame, FlushSchedule]:
    flushes = _read_csv(run_dir / "flushes.csv", FLUSH_COLS, issues)
    for column in ("flush_id", "tick", "tsfiles_before", "tsfiles_after"):
        _to_int(flushes, column, "flushes")
    if flushes.empty:
        raise RunFormatError("flushes.csv is empty (the final flush is always required)")
    flushes = flushes.sort_values("flush_id", kind="stable").reset_index(drop=True)
    if flushes["flush_id"].duplicated().any():
        raise RunFormatError("duplicate flush_id in flushes.csv")
    last_tick = int(meta["n_ticks"]) - 1
    ticks = flushes["tick"].tolist()
    regular = ticks[:-1]
    schedule = FlushSchedule(flushes["flush_id"].tolist(), ticks, int(flushes["flush_id"].iloc[-1]), regular)
    if ticks[-1] != last_tick + 1:
        issues.add("error", "flush:final", f"final flush tick {ticks[-1]} != last_tick + 1 = {last_tick + 1}")
    _check_sealed_files(flushes, issues)
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        issues.add("error", "flush:order", "flush ticks are not strictly increasing in flush_id order")
    F = int(meta["flush_every"])
    expected = list(range(F - 1, last_tick + 1, F)) if F >= 1 else []
    if regular != expected:
        issues.add("error", "flush:schedule", f"regular flush ticks differ from F={F} schedule",
                   [{"first_extra": sorted(set(regular) - set(expected))[:5],
                     "first_missing": sorted(set(expected) - set(regular))[:5]}])
    if "flush_ticks" in meta and list(meta["flush_ticks"]) != ticks:
        issues.add("error", "flush:meta", "run_meta flush_ticks differ from flushes.csv")
    if (flushes["run_id"] != meta["run_id"]).any():
        issues.add("error", "run_id:flushes", "run_id column differs from run_meta")
    return flushes, schedule


def _check_sealed_files(flushes: pd.DataFrame, issues: Issues) -> None:
    """Unscheduled flushes invalidate a run (INTERFACE.md flushes.csv; code audit finding 4).

    Sealed TsFiles may change only inside a commanded FLUSH, by at most one file. A fresh data directory has none, so
    tsfiles_before > 0 at the first FLUSH means a flush happened before it.
    """
    rows = flushes[["flush_id", "tick", "tsfiles_before", "tsfiles_after"]].to_dict("records")
    gaps = [{"flush_id": r["flush_id"], "tick": r["tick"], "tsfiles_before": r["tsfiles_before"],
             "previous_tsfiles_after": prev} for prev, r in
            zip([0] + [r["tsfiles_after"] for r in rows[:-1]], rows) if r["tsfiles_before"] != prev]
    if gaps:
        issues.add("error", "flush:unscheduled", f"{len(gaps)} sealed-file counts changed between commanded FLUSHes",
                   gaps)
    many = [r for r in rows if not 0 <= r["tsfiles_after"] - r["tsfiles_before"] <= 1]
    if many:
        issues.add("error", "flush:sealed_count", "a FLUSH sealed more than one file (or the count decreased)", many)
    none = [r for r in rows[:-1] if r["tsfiles_after"] == r["tsfiles_before"]]
    if none:
        issues.add("warning", "flush:sealed_none", f"{len(none)} scheduled FLUSHes with data sealed no file", none)


def load_ledger_inputs(run_dir: Path, issues: Issues) -> LedgerInputs:
    run_dir = Path(run_dir)
    meta = read_meta(run_dir, issues)
    flushes, schedule = read_flushes(run_dir, meta, issues)
    samples = _read_csv(run_dir / "samples.csv", SAMPLES_COLS, issues)
    for column in ("tick", "server_tick", "write_ack_tick"):
        _to_int(samples, column, "samples")
    _to_float(samples, "value_sent", "samples", issues)
    delta = _read_csv(run_dir / "membership_delta.csv", DELTA_COLS, issues)
    for column in ("flush_id", "tick", "row_time"):
        _to_int(delta, column, "membership_delta")
    _to_float(delta, "value", "membership_delta", issues)
    digest = _read_csv(run_dir / "membership_digest.csv", DIGEST_COLS, issues)
    for column in ("flush_id", "tick", "n_rows"):
        _to_int(digest, column, "membership_digest")
    for name, frame in (("samples", samples), ("membership_delta", delta), ("membership_digest", digest)):
        if (frame["run_id"] != meta["run_id"]).any():
            issues.add("error", f"run_id:{name}", "run_id column differs from run_meta")
    _check_samples(samples, meta, issues)
    bad_change = delta.loc[~delta["change"].isin(["added", "removed"])]
    if not bad_change.empty:
        issues.add("error", "delta:change", "change must be 'added' or 'removed'", bad_change.head().to_dict("records"))
    flush_tick = dict(zip(schedule.flush_ids, schedule.ticks))
    for name, frame in (("membership_delta", delta), ("membership_digest", digest)):
        wrong = frame.loc[frame["flush_id"].map(flush_tick) != frame["tick"]]
        if not wrong.empty:
            issues.add("error", f"flush_tick:{name}", "tick differs from flushes.csv for the same flush_id",
                       wrong.head().to_dict("records"))
    return LedgerInputs(meta, flushes, schedule, samples, delta, digest)


def _check_samples(samples: pd.DataFrame, meta: dict, issues: Issues) -> None:
    channels = list(meta["channels"])
    n_ticks = int(meta["n_ticks"])
    if sorted(samples["channel"].unique()) != sorted(channels):
        issues.add("error", "samples:channels", f"channels {sorted(samples['channel'].unique())} != meta {channels}")
    dup = samples.loc[samples.duplicated(["channel", "tick"], keep=False)]
    if not dup.empty:
        issues.add("error", "samples:duplicate", "more than one row per channel per tick", dup.head().to_dict("records"))
    observed_last = int(samples["tick"].max()) if not samples.empty else -1
    if observed_last != n_ticks - 1 or int(samples["tick"].min()) != 0:
        issues.add("error", "samples:n_ticks", f"ticks span {samples['tick'].min()}..{observed_last}, meta n_ticks={n_ticks}")
    for channel in channels:
        ticks = set(samples.loc[samples["channel"] == channel, "tick"])
        missing = sorted(set(range(n_ticks)) - ticks)
        if missing:
            issues.add("error", "samples:coverage", f"{channel} lacks {len(missing)} ticks", missing[:10])
    early = samples.loc[samples["write_ack_tick"] < samples["tick"]]
    if not early.empty:
        issues.add("error", "samples:ack_before_tick", "write_ack_tick precedes tick", early.head().to_dict("records"))
    late = samples.loc[samples["write_ack_tick"] != samples["tick"]]
    if not late.empty:
        issues.add("warning", "samples:ack_not_same_tick", f"{len(late)} acks not in their own barrier tick",
                   late.head().to_dict("records"))


def load_truth_inputs(run_dir: Path, issues: Issues) -> TruthInputs:
    run_dir = Path(run_dir)
    meta = read_meta(run_dir, issues)
    flushes, schedule = read_flushes(run_dir, meta, issues)
    snapshots = _read_csv(run_dir / "snapshots.csv", SNAP_COLS, issues)
    for column in ("origin", "row_time"):
        _to_int(snapshots, column, "snapshots")
    _to_float(snapshots, "value", "snapshots", issues)
    if (snapshots["run_id"] != meta["run_id"]).any():
        issues.add("error", "run_id:snapshots", "run_id column differs from run_meta")
    bad_kind = snapshots.loc[~snapshots["kind"].isin(SNAP_KINDS)]
    if not bad_kind.empty:
        issues.add("error", "snap:kind", f"kind must be one of {sorted(SNAP_KINDS)}", bad_kind.head().to_dict("records"))
    origins = set(snapshots["origin"].unique())
    missing = sorted(set(range(int(meta["n_ticks"]))) - origins)
    if missing:
        # An origin with no visible row in any channel has no snapshot rows; that only happens before the first write.
        issues.add("warning", "snap:origins", f"{len(missing)} origins have no snapshot rows", missing[:10])
    extra = sorted(o for o in origins if not 0 <= o < int(meta["n_ticks"]))
    if extra:
        issues.add("error", "snap:origin_range", "snapshot origins outside 0..n_ticks-1", extra[:10])
    return TruthInputs(meta, flushes, schedule, snapshots)


def run_is_complete(run_dir: Path) -> bool:
    return (Path(run_dir) / "DONE").is_file()
