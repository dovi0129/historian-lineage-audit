"""Reconstruction rule R (spec §8), implemented without endpoint assumptions.

For feature time s, R interpolates linearly between the nearest available rows at or before s and after s.
If no available row lies after s it holds the latest row at or before s; if none lies at or before s the
feature is missing (NaN). Rows are never invented: only the rows passed in are used.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Sequence

WINDOW = 8
LAGS = (0, 2, 4, 6, 8)
TOL = 1e-12
MIN_ORIGIN = 8

Row = tuple[int, float]


def reconstruct(rows: Sequence[Row], s: int) -> float:
    """Apply R to rows sorted by strictly increasing time."""
    times = [t for t, _ in rows]
    i = bisect_right(times, s)
    if i == 0:
        return math.nan
    t_left, v_left = rows[i - 1]
    if t_left == s or i == len(rows):
        return v_left
    t_right, v_right = rows[i]
    return v_left + (v_right - v_left) * (s - t_left) / (t_right - t_left)


def features(rows: Sequence[Row], origin: int) -> tuple[float, ...]:
    """Features at s = origin - lag for every lag in LAGS."""
    return tuple(reconstruct(rows, origin - lag) for lag in LAGS)


def same_value(a: float, b: float) -> bool:
    """Feature equality: both missing, or both present with |a - b| <= TOL."""
    a_missing, b_missing = math.isnan(a), math.isnan(b)
    if a_missing or b_missing:
        return a_missing and b_missing
    return abs(a - b) <= TOL
