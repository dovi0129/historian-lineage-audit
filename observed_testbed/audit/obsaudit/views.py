"""Views per origin and channel, their comparison with the snapshot truth, and the negative checks (spec §8).

A view's row set at origin o is: its active rows with o-8 <= time <= o (window), the latest active row with time
< o-8 (anchor) and the first active row with time > o (next; only the completed export V1 or a corrupted replay has
one). These rows determine R at every feature time s in [o-8, o], so features are computed from the row set alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .ledger import ChannelArrays, Ledger, view_arrays
from .reconstruct import LAGS, MIN_ORIGIN, TOL, WINDOW, Row, features, same_value
from .truth import BEFORE_FIRST_FLUSH, Truth

REPLAY_VIEWS = ("Replay_commit", "Replay_visible", "V3", "V_server", "V1")
# (name, view, truth reference); the last pair is truth against truth (descriptive share of S_all != S_commit).
COMPARISONS = (
    ("Replay_commit_vs_S_commit", "Replay_commit", "S_commit"),
    ("Replay_visible_vs_S_all", "Replay_visible", "S_all"),
    ("V3_vs_S_commit", "V3", "S_commit"),
    ("V_server_vs_S_commit", "V_server", "S_commit"),
    ("V1_vs_S_commit", "V1", "S_commit"),
    ("S_all_vs_S_commit", "S_all", "S_commit"),
)
FULL_ROWS = {"Replay_commit_vs_S_commit", "Replay_visible_vs_S_all"}


def view_rows(arr: ChannelArrays | None, origin: int, unused_next: bool = True) -> tuple[Row, ...]:
    """Window, anchor and next row of the rows active at origin.

    A replay view should never hold a row after o, so its next row is kept whether or not R uses it (that is a
    leak the row-set comparison must see). For the completed export V1 (unused_next=False) the next row is kept
    only when R uses it, i.e. when no row lies exactly at o but one lies before it.
    """
    if arr is None or len(arr.time) == 0:
        return ()
    active = (arr.start <= origin) & (origin < arr.end)
    t, v = arr.time[active], arr.value[active]
    lo = int(np.searchsorted(t, origin - WINDOW, side="left"))
    hi = int(np.searchsorted(t, origin, side="right"))
    stop = hi + 1 if (unused_next or (hi > 0 and t[hi - 1] < origin)) else hi
    rows = [(int(t[k]), float(v[k])) for k in range(max(lo - 1, 0), min(stop, len(t)))]
    return tuple(rows)


def window_copies(arr: ChannelArrays | None, origin: int) -> dict[int, int]:
    """Stored copies of each active row with o-8 <= time <= o (Replay_visible arrays carry copies)."""
    if arr is None or len(arr.time) == 0:
        return {}
    active = (arr.start <= origin) & (origin < arr.end) & (arr.time >= origin - WINDOW) & (arr.time <= origin)
    copies = arr.copies if arr.copies is not None else np.ones(len(arr.time), dtype=np.int64)
    return {int(t): int(c) for t, c in zip(arr.time[active], copies[active])}


@dataclass
class CellResult:
    rowset_equal: bool
    only_a: tuple[Row, ...]
    only_b: tuple[Row, ...]
    feat_a: tuple[float, ...]
    feat_b: tuple[float, ...]
    bad_lags: tuple[int, ...]

    @property
    def any_mismatch(self) -> bool:
        return not self.rowset_equal or bool(self.bad_lags)


def compare_cell(a: tuple[Row, ...], b: tuple[Row, ...], origin: int) -> CellResult:
    da, db = dict(a), dict(b)
    only_a = tuple(r for r in a if r[0] not in db or abs(r[1] - db[r[0]]) > TOL)
    only_b = tuple(r for r in b if r[0] not in da or abs(r[1] - da[r[0]]) > TOL)
    fa, fb = features(a, origin), features(b, origin)
    bad = tuple(lag for lag, x, y in zip(LAGS, fa, fb) if not same_value(x, y))
    return CellResult(not only_a and not only_b, only_a, only_b, fa, fb, bad)


def fmt_rows(rows) -> str:
    return " ".join(f"{t}:{v!r}" for t, v in rows)


@dataclass
class Tally:
    n_cells: int = 0
    n_excluded_before_first_flush: int = 0
    n_excluded_undetermined: int = 0
    n_rowset: int = 0
    n_feature_cells: int = 0
    n_any: int = 0
    n_features: int = 0
    n_feature_mismatch: int = 0
    n_missing_mismatch: int = 0
    diffs: list[float] = field(default_factory=list)
    per_channel: dict[str, int] = field(default_factory=dict)
    origins_eval: set[int] = field(default_factory=set)
    origins_bad: set[int] = field(default_factory=set)
    flags: dict[str, list[tuple[int, bool]]] = field(default_factory=dict)  # channel -> (origin, mismatch)

    def summary(self) -> dict:
        diffs = np.asarray(self.diffs, dtype=float)
        runs = []
        for seq in self.flags.values():
            length, prev = 0, None
            for origin, bad in seq:
                if bad and length and origin == prev + 1:
                    length += 1
                else:
                    if length:
                        runs.append(length)
                    length = 1 if bad else 0
                prev = origin
            if length:
                runs.append(length)
        runs_arr = np.asarray(runs, dtype=float)
        return {
            "n_cells_evaluated": self.n_cells,
            "n_cells_excluded_before_first_flush": self.n_excluded_before_first_flush,
            "n_cells_excluded_truth_undetermined": self.n_excluded_undetermined,
            "n_cells_rowset_mismatch": self.n_rowset,
            "n_cells_feature_mismatch": self.n_feature_cells,
            "n_cells_any_mismatch": self.n_any,
            "cell_mismatch_rate": rate(self.n_any, self.n_cells),
            "n_origins_evaluated": len(self.origins_eval),
            "n_origins_any_mismatch": len(self.origins_bad),
            "origin_mismatch_rate": rate(len(self.origins_bad), len(self.origins_eval)),
            "n_features": self.n_features,
            "n_feature_mismatch": self.n_feature_mismatch,
            "feature_mismatch_rate": rate(self.n_feature_mismatch, self.n_features),
            "n_feature_mismatch_one_side_missing": self.n_missing_mismatch,
            "abs_diff_both_present": _dist(diffs),
            "mismatch_run_lengths_origins": _dist(runs_arr) | {"n_runs": len(runs)},
            "cells_any_mismatch_by_channel": dict(sorted(self.per_channel.items())),
        }


def rate(k: int, n: int) -> float | None:
    return k / n if n else None


def _dist(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    q = np.percentile(x, [50, 90, 99])
    return {"n": int(len(x)), "min": float(x.min()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(x.max()), "mean": float(x.mean())}


@dataclass
class ComparisonOutput:
    tallies: dict[str, Tally]
    mismatches: list[dict]
    arrays: dict[str, dict[str, ChannelArrays]]
    v_server_equals_v3_cells: int
    v_server_differs_v3_cells: int
    multiplicity: dict = field(default_factory=dict)


def run_comparisons(ledger: Ledger, truth: Truth, meta: dict) -> ComparisonOutput:
    channels = list(meta["channels"])
    n_ticks = int(meta["n_ticks"])
    run_id = meta["run_id"]
    arrays = {view: view_arrays(ledger, view) for view in REPLAY_VIEWS}
    tallies = {name: Tally() for name, _, _ in COMPARISONS}
    mismatches: list[dict] = []
    vs_eq = vs_ne = 0
    # Descriptive (not a criterion): window-row copies, ledger membership multiset vs snapshot multiset.
    mult = {"n_cells": 0, "n_cells_agree": 0, "n_cells_disagree": 0, "n_cells_with_duplicates_ledger": 0,
            "n_cells_with_duplicates_snapshot": 0, "examples": []}
    for origin in range(MIN_ORIGIN, n_ticks):
        for ch in channels:
            cell = truth.cells[(origin, ch)]
            ledger_copies = window_copies(arrays["Replay_visible"].get(ch), origin)
            mult["n_cells"] += 1
            mult["n_cells_with_duplicates_ledger"] += any(c > 1 for c in ledger_copies.values())
            mult["n_cells_with_duplicates_snapshot"] += any(c > 1 for c in cell.window_copies.values())
            if ledger_copies == cell.window_copies:
                mult["n_cells_agree"] += 1
            else:
                mult["n_cells_disagree"] += 1
                if len(mult["examples"]) < 10:
                    mult["examples"].append({"origin": origin, "channel": ch, "ledger": ledger_copies,
                                             "snapshot": cell.window_copies})
            rows = {view: view_rows(arrays[view].get(ch), origin, unused_next=view != "V1") for view in REPLAY_VIEWS}
            rows["S_all"] = cell.s_all
            rows["S_commit"] = cell.s_commit
            if compare_cell(rows["V_server"], rows["V3"], origin).any_mismatch:
                vs_ne += 1
            else:
                vs_eq += 1
            for name, view, ref in COMPARISONS:
                tally = tallies[name]
                if rows[ref] is None:
                    if cell.status == BEFORE_FIRST_FLUSH:
                        tally.n_excluded_before_first_flush += 1
                    else:
                        tally.n_excluded_undetermined += 1
                    tally.flags.setdefault(ch, []).append((origin, False))
                    continue
                res = compare_cell(rows[view], rows[ref], origin)
                _tally(tally, res, origin, ch)
                if res.any_mismatch:
                    mismatches.append(_mismatch_record(run_id, name, origin, ch, cell, res, rows[view], rows[ref]))
    return ComparisonOutput(tallies, mismatches, arrays, vs_eq, vs_ne, mult)


def _tally(tally: Tally, res: CellResult, origin: int, ch: str) -> None:
    tally.n_cells += 1
    tally.origins_eval.add(origin)
    tally.n_features += len(LAGS)
    if not res.rowset_equal:
        tally.n_rowset += 1
    if res.bad_lags:
        tally.n_feature_cells += 1
        tally.n_feature_mismatch += len(res.bad_lags)
        for lag in res.bad_lags:
            k = LAGS.index(lag)
            a, b = res.feat_a[k], res.feat_b[k]
            if math.isnan(a) or math.isnan(b):
                tally.n_missing_mismatch += 1
            else:
                tally.diffs.append(abs(a - b))
    if res.any_mismatch:
        tally.n_any += 1
        tally.origins_bad.add(origin)
        tally.per_channel[ch] = tally.per_channel.get(ch, 0) + 1
    tally.flags.setdefault(ch, []).append((origin, res.any_mismatch))


def _mismatch_record(run_id, name, origin, ch, cell, res: CellResult, rows_a, rows_b) -> dict:
    idx = [LAGS.index(lag) for lag in res.bad_lags]
    diffs = [abs(res.feat_a[k] - res.feat_b[k]) for k in idx]
    diffs = [d for d in diffs if not math.isnan(d)]
    full = name in FULL_ROWS
    return {
        "run_id": run_id, "comparison": name, "origin": origin, "channel": ch,
        "t_f": cell.t_f,
        "rowset_mismatch": int(not res.rowset_equal), "n_feature_mismatch": len(res.bad_lags),
        "mismatched_lags": ";".join(str(lag) for lag in res.bad_lags),
        "features_view": ";".join(f"{LAGS[k]}:{res.feat_a[k]!r}" for k in idx),
        "features_truth": ";".join(f"{LAGS[k]}:{res.feat_b[k]!r}" for k in idx),
        "max_abs_diff": max(diffs) if diffs else math.nan,
        "rows_only_view": fmt_rows(res.only_a), "rows_only_truth": fmt_rows(res.only_b),
        "rows_view": fmt_rows(rows_a) if full else "", "rows_truth": fmt_rows(rows_b) if full else "",
    }


# ---------------------------------------------------------------- negative checks (spec §8)

def _matched_cells(out: ComparisonOutput, truth: Truth, meta: dict):
    """(origin, channel, replay rows, S_commit rows) where the primary comparison matched, in origin order."""
    bad = {(m["origin"], m["channel"]) for m in out.mismatches if m["comparison"] == "Replay_commit_vs_S_commit"}
    for origin in range(MIN_ORIGIN, int(meta["n_ticks"])):
        for ch in meta["channels"]:
            cell = truth.cells.get((origin, ch))
            arr = out.arrays["Replay_commit"].get(ch)
            if arr is None or cell is None or cell.s_commit is None or (origin, ch) in bad:
                continue
            yield origin, ch, arr, cell.s_commit


def _perturbed(arr: ChannelArrays, index: int, new_start: float) -> ChannelArrays:
    start = arr.start.copy()
    start[index] = new_start
    return replace(arr, start=start)


def negative_check_shift(out: ComparisonOutput, truth: Truth, meta: dict) -> dict:
    """(a) Shift the commit tick of a used endpoint by +1; the comparison must report row-set and feature mismatch."""
    fallback = None
    for origin, ch, arr, s_commit in _matched_cells(out, truth, meta):
        rows = view_rows(arr, origin)
        times = [t for t, _ in rows]
        for lag in LAGS:
            s = origin - lag
            left = max((t for t in times if t <= s), default=None)
            right = min((t for t in times if t > s), default=None)
            if left is None:
                continue
            mode = "interpolation" if (left < s and right is not None) else ("exact" if left == s else None)
            if mode is None:
                continue
            for endpoint in ((left, right) if mode == "interpolation" else (left,)):
                k = int(np.searchsorted(arr.time, endpoint))
                if arr.start[k] != origin:
                    continue  # +1 only changes availability at this origin if c_obs == origin
                shifted = _perturbed(arr, k, arr.start[k] + 1)
                res = compare_cell(view_rows(shifted, origin), s_commit, origin)
                record = {"origin": origin, "channel": ch, "lag": lag, "mode": mode, "endpoint_time": endpoint,
                          "c_obs": float(arr.start[k]), "shifted_c_obs": float(arr.start[k] + 1),
                          "rowset_detected": not res.rowset_equal, "feature_detected": bool(res.bad_lags),
                          "feature_before": res.feat_b[LAGS.index(lag)], "feature_after": res.feat_a[LAGS.index(lag)]}
                if not same_value(record["feature_before"], record["feature_after"]):
                    if mode == "interpolation":
                        return _verdict(record, require_feature=True)
                    fallback = fallback or record
    if fallback is not None:
        return _verdict(fallback, require_feature=True)
    return {"status": "not_applicable", "reason": "no used endpoint with c_obs == origin changes a feature"}


def negative_check_insert(out: ComparisonOutput, truth: Truth, meta: dict) -> dict:
    """(b) Make one not-yet-available row available at an origin; the row-set comparison must flag it.

    future_time: the next committed row with time > o (unused when a row exists at o).
    future_commit: a row in the window with o-8 <= time <= o committed only after o.
    Unused insertions (features unchanged) are preferred; otherwise the first used one is reported.
    """
    result = {}
    for kind in ("future_time", "future_commit"):
        used_case = None
        found = None
        for origin, ch, arr, s_commit in _matched_cells(out, truth, meta):
            if kind == "future_time":
                k = int(np.searchsorted(arr.time, origin, side="right"))
                candidates = [k] if k < len(arr.time) and arr.start[k] > origin else []
            else:
                candidates = [k for k in range(int(np.searchsorted(arr.time, origin - WINDOW)),
                                               int(np.searchsorted(arr.time, origin, side="right")))
                              if arr.start[k] > origin]
            for k in candidates:
                res = compare_cell(view_rows(_perturbed(arr, k, origin), origin), s_commit, origin)
                record = {"origin": origin, "channel": ch, "inserted_time": int(arr.time[k]),
                          "true_c_obs": float(arr.start[k]), "rowset_detected": not res.rowset_equal,
                          "features_changed": bool(res.bad_lags)}
                if not res.bad_lags:
                    found = record
                    break
                used_case = used_case or record
            if found:
                break
        record = found or used_case
        if record is None:
            result[kind] = {"status": "not_applicable", "reason": "no row available later than its origin"}
        else:
            record["unused"] = found is not None
            result[kind] = _verdict(record, require_feature=False)
    return result


def _verdict(record: dict, require_feature: bool) -> dict:
    ok = record["rowset_detected"] and (record.get("feature_detected", True) if require_feature else True)
    return {"status": "detected" if ok else "NOT_DETECTED", **record}
