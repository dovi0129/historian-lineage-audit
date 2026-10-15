"""Source-event provenance and origin-vintage replay for historian views.

The module keeps event time and commit/knowledge time separate.  It models the
explicit forward operators used by the local historian experiments; it does not
attempt to emulate a vendor-specific historian API or runtime snapshot service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RetentionTrace:
    """Retained event indices and the indices at which they become known."""

    event_indices: np.ndarray
    commit_indices: np.ndarray

    def validate(self, n: int) -> None:
        events = np.asarray(self.event_indices, dtype=np.int64)
        commits = np.asarray(self.commit_indices, dtype=np.int64)
        if events.ndim != 1 or commits.ndim != 1 or len(events) != len(commits):
            raise ValueError("retention arrays must be aligned one-dimensional arrays")
        if len(events) < 2 or events[0] != 0 or events[-1] != n - 1:
            raise ValueError("retention trace must include both endpoints")
        if np.any(np.diff(events) <= 0):
            raise ValueError("event indices must be strictly increasing")
        if np.any(np.diff(commits) < 0):
            raise ValueError("commit indices must be nondecreasing")
        if np.any(commits < events):
            raise ValueError("commit indices must not precede their source events")


@dataclass(frozen=True)
class FinalReconstruction:
    """A final linear reconstruction and cell-level source lineage."""

    values: np.ndarray
    left_source: np.ndarray
    right_source: np.ndarray
    left_commit: np.ndarray
    right_commit: np.ndarray
    availability: np.ndarray


@dataclass(frozen=True)
class VintageFeatureBundle:
    """Aligned final- and origin-vintage supervised feature representations."""

    origins: np.ndarray
    feature_times: np.ndarray
    final_features: np.ndarray
    origin_features: np.ndarray
    final_current: np.ndarray
    origin_current: np.ndarray
    strict_input_availability: np.ndarray
    active_input_availability: np.ndarray
    final_feature_availability: np.ndarray
    structural_future_share: np.ndarray
    active_future_share: np.ndarray
    active_future_mask: np.ndarray
    final_source_left: np.ndarray
    final_source_right: np.ndarray
    origin_source_left: np.ndarray
    origin_source_right: np.ndarray
    origin_source_commit: np.ndarray
    target_final_availability: np.ndarray
    target_active_availability: np.ndarray
    target_final_values: np.ndarray
    target_origin_values: np.ndarray


def identity_trace(n: int) -> RetentionTrace:
    if n < 2:
        raise ValueError("identity retention requires n >= 2")
    indices = np.arange(n, dtype=np.int64)
    trace = RetentionTrace(indices, indices.copy())
    trace.validate(n)
    return trace


def periodic_trace(n: int, stride: int) -> RetentionTrace:
    if n < 2 or stride < 1:
        raise ValueError("periodic retention requires n >= 2 and stride >= 1")
    events = list(range(0, int(n), int(stride)))
    if events[-1] != n - 1:
        events.append(n - 1)
    indices = np.asarray(events, dtype=np.int64)
    trace = RetentionTrace(indices, indices.copy())
    trace.validate(n)
    return trace


def event_time_only_trace(trace: RetentionTrace, n: int) -> RetentionTrace:
    """Return the same retained events while treating event time as knowledge time.

    This is an explicit ablation for comparing event-time-only point-in-time
    replay with replay that respects the original commit/knowledge indices.  It
    does not modify which values were retained.
    """

    trace.validate(n)
    events = np.asarray(trace.event_indices, dtype=np.int64).copy()
    output = RetentionTrace(event_indices=events, commit_indices=events.copy())
    output.validate(n)
    return output


def swinging_door_trace(values: np.ndarray, epsilon: float) -> RetentionTrace:
    """Return the local swinging-door retention path with decision times.

    A corridor violation observed at scan index ``i`` commits the retained event
    selected by the existing algorithm at ``i``.  This reproduces the event
    indices of ``historian_inferential_instability.swinging_door_indices`` while
    exposing the additional knowledge-time coordinate.
    """

    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2 or not np.all(np.isfinite(x)):
        raise ValueError("swinging-door retention requires a finite series of length >= 2")
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")

    kept = [0]
    committed = [0]
    anchor = 0
    lower = -np.inf
    upper = np.inf
    i = 1
    while i < n:
        width = i - anchor
        candidate_lower = (x[i] - epsilon - x[anchor]) / width
        candidate_upper = (x[i] + epsilon - x[anchor]) / width
        new_lower = max(lower, candidate_lower)
        new_upper = min(upper, candidate_upper)
        if new_lower > new_upper:
            retain = i - 1
            if retain <= anchor:
                retain = i
            if retain != kept[-1]:
                kept.append(retain)
                committed.append(i)
            anchor = retain
            lower = -np.inf
            upper = np.inf
            if anchor == i:
                i += 1
            continue
        lower = new_lower
        upper = new_upper
        i += 1
    if kept[-1] != n - 1:
        kept.append(n - 1)
        committed.append(n - 1)

    trace = RetentionTrace(
        np.asarray(kept, dtype=np.int64), np.asarray(committed, dtype=np.int64)
    )
    trace.validate(n)
    return trace


def final_linear(values: np.ndarray, trace: RetentionTrace) -> FinalReconstruction:
    """Linearly reconstruct a complete series and retain exact source lineage."""

    x = np.asarray(values, dtype=float)
    n = len(x)
    trace.validate(n)
    events = np.asarray(trace.event_indices, dtype=np.int64)
    commits = np.asarray(trace.commit_indices, dtype=np.int64)
    grid = np.arange(n, dtype=np.int64)

    right_position = np.searchsorted(events, grid, side="left")
    exact = events[right_position] == grid
    left_position = np.where(exact, right_position, right_position - 1)
    left = events[left_position]
    right = events[right_position]
    availability = np.maximum(commits[left_position], commits[right_position])
    reconstructed = np.interp(grid.astype(float), events.astype(float), x[events])
    return FinalReconstruction(
        values=np.asarray(reconstructed, dtype=float),
        left_source=left,
        right_source=right,
        left_commit=commits[left_position],
        right_commit=commits[right_position],
        availability=np.asarray(availability, dtype=np.int64),
    )


def _channel_tolerance(values: np.ndarray) -> float:
    scale = max(float(np.std(values)), 1.0)
    return max(1e-12, 1e-10 * scale)


def _origin_vintage_cells(
    values: np.ndarray,
    trace: RetentionTrace,
    reconstruction: FinalReconstruction,
    event_times: np.ndarray,
    origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve cells from information committed no later than each origin."""

    x = np.asarray(values, dtype=float)
    s = np.asarray(event_times, dtype=np.int64)
    o = np.asarray(origins, dtype=np.int64)
    if s.shape != o.shape or np.any(s < 0) or np.any(s > o) or np.any(o >= len(x)):
        raise ValueError("invalid event-time/origin geometry")

    final_availability = reconstruction.availability[s]
    ready = final_availability <= o
    event_bound = np.searchsorted(trace.event_indices, s, side="right") - 1
    commit_bound = np.searchsorted(trace.commit_indices, o, side="right") - 1
    hold_position = np.minimum(event_bound, commit_bound)
    if np.any(hold_position < 0):
        raise RuntimeError("no retained source is available for an origin-vintage cell")
    hold_source = trace.event_indices[hold_position]

    result = np.where(ready, reconstruction.values[s], x[hold_source])
    left = np.where(ready, reconstruction.left_source[s], hold_source)
    right = np.where(ready, reconstruction.right_source[s], hold_source)
    source_commit = np.where(
        ready,
        final_availability,
        trace.commit_indices[hold_position],
    )
    if np.any(source_commit > o):
        raise AssertionError("origin-vintage cell uses a source committed after its origin")
    return result, left, right, source_commit


def build_vintage_features(
    panel: np.ndarray,
    traces: Sequence[RetentionTrace],
    origins: np.ndarray,
    lags: Sequence[int],
    *,
    target_channel: int,
    horizon: int,
) -> VintageFeatureBundle:
    """Build final and causal origin-vintage feature rows on identical support.

    Columns are channel-major and then lag-major, matching the existing local
    forecasting runners.
    """

    x = np.asarray(panel, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("panel must be a finite two-dimensional array")
    n, n_channels = x.shape
    if len(traces) != n_channels or not 0 <= target_channel < n_channels:
        raise ValueError("trace/channel mismatch")
    origin = np.asarray(origins, dtype=np.int64)
    lag_array = np.asarray(tuple(lags), dtype=np.int64)
    if len(origin) == 0 or len(lag_array) == 0 or np.any(lag_array < 0):
        raise ValueError("origins and non-negative lags are required")
    if not np.any(lag_array == 0):
        raise ValueError("lags must include zero to define the current-value baseline")
    if horizon < 1:
        raise ValueError("horizon must be a positive number of grid steps")
    if np.min(origin) < np.max(lag_array) or np.max(origin) + horizon >= n:
        raise ValueError("origins do not support the requested lags and horizon")

    reconstructions = [final_linear(x[:, channel], traces[channel]) for channel in range(n_channels)]
    n_rows = len(origin)
    n_features = n_channels * len(lag_array)
    final_features = np.empty((n_rows, n_features), dtype=float)
    origin_features = np.empty_like(final_features)
    final_left = np.empty((n_rows, n_features), dtype=np.int32)
    final_right = np.empty_like(final_left)
    origin_left = np.empty_like(final_left)
    origin_right = np.empty_like(final_left)
    origin_commit = np.empty_like(final_left)
    final_availability = np.empty((n_rows, n_features), dtype=np.int32)
    active = np.empty((n_rows, n_features), dtype=bool)
    feature_times = origin[:, None] - lag_array[None, :]

    column = 0
    for channel in range(n_channels):
        reconstruction = reconstructions[channel]
        tolerance = _channel_tolerance(x[:, channel])
        for lag_index in range(len(lag_array)):
            times = feature_times[:, lag_index]
            asof, asof_left, asof_right, asof_commit = _origin_vintage_cells(
                x[:, channel], traces[channel], reconstruction, times, origin
            )
            final = reconstruction.values[times]
            availability = reconstruction.availability[times]
            final_features[:, column] = final
            origin_features[:, column] = asof
            final_left[:, column] = reconstruction.left_source[times]
            final_right[:, column] = reconstruction.right_source[times]
            origin_left[:, column] = asof_left
            origin_right[:, column] = asof_right
            origin_commit[:, column] = asof_commit
            final_availability[:, column] = availability
            active[:, column] = (availability > origin) & (np.abs(final - asof) > tolerance)
            column += 1

    structural_future = final_availability > origin[:, None]
    strict_input_availability = np.max(final_availability, axis=1).astype(np.int64)
    active_availability = np.max(
        np.where(active, final_availability, origin[:, None]), axis=1
    ).astype(np.int64)

    target_times = origin + int(horizon)
    target_reconstruction = reconstructions[target_channel]
    target_origin_values, _, _, _ = _origin_vintage_cells(
        x[:, target_channel],
        traces[target_channel],
        target_reconstruction,
        target_times,
        target_times,
    )
    target_final_values = target_reconstruction.values[target_times]
    target_final_availability = target_reconstruction.availability[target_times].astype(np.int64)
    target_tolerance = _channel_tolerance(x[:, target_channel])
    target_active = (target_final_availability > target_times) & (
        np.abs(target_final_values - target_origin_values) > target_tolerance
    )
    target_active_availability = np.where(
        target_active, target_final_availability, target_times
    ).astype(np.int64)

    current_column = target_channel * len(lag_array) + int(np.flatnonzero(lag_array == 0)[0])
    return VintageFeatureBundle(
        origins=origin,
        feature_times=feature_times,
        final_features=final_features,
        origin_features=origin_features,
        final_current=final_features[:, current_column],
        origin_current=origin_features[:, current_column],
        strict_input_availability=strict_input_availability,
        active_input_availability=active_availability,
        final_feature_availability=final_availability,
        structural_future_share=np.mean(structural_future, axis=1),
        active_future_share=np.mean(active, axis=1),
        active_future_mask=active,
        final_source_left=final_left,
        final_source_right=final_right,
        origin_source_left=origin_left,
        origin_source_right=origin_right,
        origin_source_commit=origin_commit,
        target_final_availability=target_final_availability,
        target_active_availability=target_active_availability,
        target_final_values=target_final_values,
        target_origin_values=target_origin_values,
    )


def unique_source_ids(
    bundle: VintageFeatureBundle,
    rows: np.ndarray,
    *,
    view: str,
    n_time: int,
) -> np.ndarray:
    """Return unique channel/time source IDs for selected supervised rows."""

    selected = np.asarray(rows, dtype=np.int64)
    if view == "FINAL":
        left, right = bundle.final_source_left[selected], bundle.final_source_right[selected]
    elif view == "ORIGIN":
        left, right = bundle.origin_source_left[selected], bundle.origin_source_right[selected]
    else:
        raise ValueError(f"unknown view: {view}")
    feature = np.arange(left.shape[1], dtype=np.int64)
    n_lags = bundle.feature_times.shape[1]
    channels = feature // n_lags
    left_ids = channels[None, :] * int(n_time) + left.astype(np.int64)
    right_ids = channels[None, :] * int(n_time) + right.astype(np.int64)
    return np.unique(np.concatenate((left_ids.ravel(), right_ids.ravel())))
