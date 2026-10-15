#!/usr/bin/env python3
"""Differentiate reconstructed-grid PIT checks from lineage-aware vintage audit.

The script reuses the fixed CSTR/TEP geometry and FINAL-versus-COMMIT results
from script 101.  It adds the source-event-time-only ablation, test-unit
geometry/fidelity summaries, and a descriptive remedy attribution.  It models
the declared forward operators, not a vendor-specific historian API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import historian_vintage as hv  # noqa: E402


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import parent script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s101 = load_script(
    ROOT / "scripts" / "101_historian_vintage_replay_kill_test.py",
    "historian_vintage_script101",
)

SPEC = (
    "reports/swarm/specs/"
    "2026-09-03_historian_reconstruction_lineage_method_differentiation.approved.md"
)
DEFAULT_OUT = ROOT / "artifacts" / "historian_reconstruction_lineage_differentiation"
SMOKE_OUT = ROOT / "artifacts" / "historian_reconstruction_lineage_differentiation_smoke"
LOG = ROOT / "artifacts" / "logs" / "102_historian_reconstruction_lineage_differentiation.log"
PARENT_FULL = ROOT / "artifacts" / "historian_vintage_replay_kill_test"
PARENT_SMOKE = ROOT / "artifacts" / "historian_vintage_replay_kill_test_smoke"

META_COLUMNS = (
    "process",
    "region",
    "target",
    "horizon_minutes",
    "seed",
    "fold",
    "fault",
    "operator",
    "severity_k",
    "nominal_retention",
)
KEY_COLUMNS = META_COLUMNS[:-1] + ("score", "model")
DIAGNOSTICS = (
    "compression_fraction",
    "final_native_feature_sd_nrmse",
    "final_commit_feature_sd_nrmse",
    "strict_future_cell_share",
    "active_future_cell_share",
    "naive_grid_pit_miss_share",
    "median_horizon_erosion_minutes",
    "p95_horizon_erosion_minutes",
    "event_commit_feature_sd_nrmse",
    "event_commit_origin_difference_share",
)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def choose_workers(requested: int | None, tasks: int, cap: int) -> int:
    cores = os.cpu_count() or 8
    if requested is not None:
        return max(1, min(int(requested), cores, tasks))
    return max(1, min(cap, max(1, cores - 2), tasks))


def direct_features(panel: np.ndarray, origins: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Return native channel-major, lag-major features."""

    x = np.asarray(panel, dtype=float)
    origin = np.asarray(origins, dtype=np.int64)
    columns = [
        x[origin - int(lag), channel]
        for channel in range(x.shape[1])
        for lag in lags
    ]
    return np.column_stack(columns)


def sd_nrmse(left: np.ndarray, right: np.ndarray, reference: np.ndarray) -> float:
    """Column-standardized RMS difference with constant columns excluded."""

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.shape != ref.shape or a.ndim != 2:
        raise ValueError("sd_nrmse arrays must be aligned two-dimensional matrices")
    scales = np.std(ref, axis=0)
    usable = scales > 1e-15
    if not np.any(usable):
        return 0.0 if np.allclose(a, b, atol=1e-12, rtol=0.0) else math.nan
    normalized = (a[:, usable] - b[:, usable]) / scales[usable]
    return float(np.sqrt(np.mean(np.square(normalized))))


def value_difference_mask(
    left: np.ndarray, right: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    ref = np.asarray(reference, dtype=float)
    scales = np.std(ref, axis=0)
    tolerances = np.maximum(1e-12, 1e-10 * np.maximum(scales, 1.0))
    return np.abs(a - b) > tolerances[None, :]


def retained_ratio(traces: Sequence[hv.RetentionTrace], n_time: int) -> float:
    return float(np.mean([len(trace.event_indices) / n_time for trace in traces]))


def diagnostic_row(
    *,
    final_features: np.ndarray,
    event_features: np.ndarray,
    commit_features: np.ndarray,
    native_features: np.ndarray,
    final_availability: np.ndarray,
    active_future_mask: np.ndarray,
    origins: np.ndarray,
    rows: np.ndarray,
    horizon_steps: int,
    dt_minutes: float,
    retained_ratio_by_origin: np.ndarray,
    channels: Sequence[str],
    n_lags: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    selected = np.asarray(rows, dtype=np.int64)
    if len(selected) == 0:
        raise ValueError("diagnostic unit has no rows")
    o = np.asarray(origins[selected], dtype=np.int64)
    final = np.asarray(final_features[selected], dtype=float)
    event = np.asarray(event_features[selected], dtype=float)
    commit = np.asarray(commit_features[selected], dtype=float)
    native = np.asarray(native_features[selected], dtype=float)
    availability = np.asarray(final_availability[selected], dtype=np.int64)
    active = np.asarray(active_future_mask[selected], dtype=bool)
    if final.shape[1] != len(channels) * n_lags:
        raise ValueError("feature geometry does not match channels and lags")

    structural = availability > o[:, None]
    latest = np.max(availability, axis=1)
    active_latest = np.max(np.where(active, availability, o[:, None]), axis=1)
    target_times = o + int(horizon_steps)
    strict_h_eff = (target_times - np.maximum(o, latest)) * dt_minutes
    active_h_eff = (target_times - np.maximum(o, active_latest)) * dt_minutes
    event_commit_difference = value_difference_mask(event, commit, native)

    channel_latest = np.max(
        availability.reshape(len(selected), len(channels), n_lags), axis=2
    )
    strict_latest = np.max(channel_latest, axis=1)
    tied = channel_latest == strict_latest[:, None]
    tie_count = np.sum(tied, axis=1)
    dominance_credit = np.sum(tied / tie_count[:, None], axis=0)
    top_channel_index = int(np.argmax(dominance_credit))

    nominal_horizon = float(horizon_steps * dt_minutes)
    strict_erosion = nominal_horizon - strict_h_eff
    active_erosion = nominal_horizon - active_h_eff
    ratio = float(np.mean(np.asarray(retained_ratio_by_origin[selected], dtype=float)))
    return {
        **metadata,
        "n_test_origins": int(len(selected)),
        "n_feature_cells": int(final.size),
        "retained_event_ratio": ratio,
        "compression_fraction": 1.0 - ratio,
        "final_native_feature_sd_nrmse": sd_nrmse(final, native, native),
        "final_commit_feature_sd_nrmse": sd_nrmse(final, commit, native),
        "event_commit_feature_sd_nrmse": sd_nrmse(event, commit, native),
        "strict_future_cell_share": float(np.mean(structural)),
        "active_future_cell_share": float(np.mean(active)),
        "naive_grid_pit_pass_share": 1.0,
        "naive_grid_pit_miss_share": float(np.mean(np.any(structural, axis=1))),
        "naive_grid_pit_active_miss_share": float(np.mean(np.any(active, axis=1))),
        "event_commit_cell_difference_share": float(np.mean(event_commit_difference)),
        "event_commit_origin_difference_share": float(
            np.mean(np.any(event_commit_difference, axis=1))
        ),
        "strict_effective_horizon_mean_minutes": float(np.mean(strict_h_eff)),
        "strict_effective_horizon_median_minutes": float(np.median(strict_h_eff)),
        "strict_effective_horizon_p05_minutes": float(np.quantile(strict_h_eff, 0.05)),
        "strict_effective_horizon_min_minutes": float(np.min(strict_h_eff)),
        "active_effective_horizon_mean_minutes": float(np.mean(active_h_eff)),
        "active_effective_horizon_median_minutes": float(np.median(active_h_eff)),
        "active_effective_horizon_p05_minutes": float(np.quantile(active_h_eff, 0.05)),
        "active_effective_horizon_min_minutes": float(np.min(active_h_eff)),
        "mean_horizon_erosion_minutes": float(np.mean(strict_erosion)),
        "median_horizon_erosion_minutes": float(np.median(strict_erosion)),
        "p95_horizon_erosion_minutes": float(np.quantile(strict_erosion, 0.95)),
        "max_horizon_erosion_minutes": float(np.max(strict_erosion)),
        "active_mean_horizon_erosion_minutes": float(np.mean(active_erosion)),
        "strict_nonpositive_horizon_share": float(np.mean(strict_h_eff <= 0.0)),
        "strict_source_at_or_after_target_share": float(np.mean(latest >= target_times)),
        "top_strict_availability_channel": str(channels[top_channel_index]),
        "top_strict_availability_channel_share": float(
            dominance_credit[top_channel_index] / len(selected)
        ),
        "mean_number_of_tied_max_channels": float(np.mean(tie_count)),
    }


def genericize_cells(
    frame: pd.DataFrame, *, comparison: str, left_view: str, right_view: str
) -> pd.DataFrame:
    output = frame.loc[:, list(META_COLUMNS) + ["score", "model"]].copy()
    output.insert(len(META_COLUMNS), "comparison", comparison)
    output.insert(len(META_COLUMNS) + 1, "left_view", left_view)
    output.insert(len(META_COLUMNS) + 2, "right_view", right_view)
    mapping = {
        "loss_left_train_left_eval": "loss_tf_ef",
        "loss_left_train_right_eval": "loss_tf_eo",
        "loss_right_train_left_eval": "loss_to_ef",
        "loss_right_train_right_eval": "loss_to_eo",
        "evaluation_view_gap": "deployment_gap",
        "right_training_repair": "origin_training_repair",
        "diagonal_gap": "diagonal_gap",
        "absolute_material_gap_1pct": "absolute_material_gap_1pct",
        "optimistic_evaluation_gap_1pct": "optimistic_deployment_gap_1pct",
        "practical_diagonal_winner_change_1pct": (
            "practical_diagonal_winner_change_1pct"
        ),
        "practical_evaluation_winner_change_1pct": (
            "practical_deployment_winner_change_1pct"
        ),
        "practical_training_repair_winner_change_1pct": (
            "practical_training_repair_winner_change_1pct"
        ),
        "prediction_view_nrmse_left_trained": (
            "prediction_view_nrmse_final_trained"
        ),
        "prediction_view_nrmse_right_trained": (
            "prediction_view_nrmse_origin_trained"
        ),
        "winner_left_train_left_eval": "winner_tf_ef",
        "winner_left_train_right_eval": "winner_tf_eo",
        "winner_right_train_left_eval": "winner_to_ef",
        "winner_right_train_right_eval": "winner_to_eo",
    }
    for target, source in mapping.items():
        output[target] = frame[source].to_numpy()
    return output


def run_cstr_event_commit(task: tuple[int, str, int, bool]) -> dict[str, Any]:
    seed, operator, k, smoke = task
    geometry = s101.SMOKE_CSTR_GEOMETRY if smoke else s101.FULL_CSTR_GEOMETRY
    trajectory = s101.load_or_generate_cstr(seed, geometry, smoke)
    calibration = np.column_stack(
        [trajectory[channel][geometry.calibration_slice] for channel in s101.CSTR_CHANNELS]
    )
    panel = np.column_stack(
        [trajectory[channel][geometry.analysis_slice] for channel in s101.CSTR_CHANNELS]
    )
    parameters: list[tuple[float, bool]] | None = None
    if operator == "SWINGING_DOOR":
        parameters = []
        for channel in range(panel.shape[1]):
            epsilon, _ = s101.hii.calibrate_swinging_door(
                calibration[:, channel], 1.0 / k
            )
            parameters.append((epsilon, True))
    commit_traces = s101.make_traces(
        panel, operator=operator, k=k, sda_parameters=parameters
    )
    event_traces = [
        hv.event_time_only_trace(trace, len(panel)) for trace in commit_traces
    ]
    origins = np.arange(
        max(s101.CSTR_LAGS), len(panel) - s101.CSTR_HORIZON, dtype=np.int64
    )
    commit_bundle = hv.build_vintage_features(
        panel,
        commit_traces,
        origins,
        s101.CSTR_LAGS,
        target_channel=0,
        horizon=s101.CSTR_HORIZON,
    )
    event_bundle = hv.build_vintage_features(
        panel,
        event_traces,
        origins,
        s101.CSTR_LAGS,
        target_channel=0,
        horizon=s101.CSTR_HORIZON,
    )
    native = direct_features(panel, origins, s101.CSTR_LAGS)
    target = panel[origins + s101.CSTR_HORIZON, 0]
    train, _, test = s101.hii.chronological_split(len(origins))
    metadata = {
        "process": "CSTR",
        "region": "CSTR_OUTPUT_Q1",
        "target": "measured_output",
        "horizon_minutes": 1.0,
        "seed": seed,
        "fold": -1,
        "fault": -1,
        "operator": operator,
        "severity_k": k,
        "nominal_retention": 1.0 / k,
    }
    _, cells = s101.evaluate_prediction_cell(
        final_features=event_bundle.origin_features,
        origin_features=commit_bundle.origin_features,
        final_current=event_bundle.origin_current,
        origin_current=commit_bundle.origin_current,
        target=target,
        train=train,
        test=test,
        process="CSTR",
        metadata=metadata,
        smoke=smoke,
    )
    ratio = retained_ratio(commit_traces, len(panel))
    diagnostic = diagnostic_row(
        final_features=commit_bundle.final_features,
        event_features=event_bundle.origin_features,
        commit_features=commit_bundle.origin_features,
        native_features=native,
        final_availability=commit_bundle.final_feature_availability,
        active_future_mask=commit_bundle.active_future_mask,
        origins=origins,
        rows=test,
        horizon_steps=s101.CSTR_HORIZON,
        dt_minutes=s101.CSTR_DT,
        retained_ratio_by_origin=np.full(len(origins), ratio),
        channels=s101.CSTR_CHANNELS,
        n_lags=len(s101.CSTR_LAGS),
        metadata=metadata,
    )
    checks = {
        "retained_event_indices_identical": bool(
            all(
                np.array_equal(commit.event_indices, event.event_indices)
                for commit, event in zip(commit_traces, event_traces)
            )
        ),
        "max_final_export_difference": float(
            np.max(
                np.abs(
                    event_bundle.final_features - commit_bundle.final_features
                )
            )
        ),
        "max_grid_time_minus_origin": int(
            np.max(commit_bundle.feature_times - origins[:, None])
        ),
        "max_commit_source_minus_origin": int(
            np.max(commit_bundle.origin_source_commit - origins[:, None])
        ),
        "max_event_commit_feature_difference": float(
            np.max(
                np.abs(
                    event_bundle.origin_features - commit_bundle.origin_features
                )
            )
        ),
    }
    return {"cells": cells, "diagnostics": [diagnostic], "checks": [checks]}


def run_tep_event_commit(
    task: tuple[str, int] | tuple[str, int, str]
) -> dict[str, Any]:
    operator, k, *panel_arg = task
    target_panel = panel_arg[0] if panel_arg else "selected"
    if s101._TEP_SEGMENTS is None or s101._TEP_CALIBRATION_MAP is None:
        raise RuntimeError("TEP worker was not initialized")
    segments = s101._TEP_SEGMENTS
    smoke = s101._TEP_SMOKE
    folds = s101.active_tep_folds(smoke)
    output_cells: list[dict[str, Any]] = []
    output_diagnostics: list[dict[str, Any]] = []
    output_checks: list[dict[str, Any]] = []

    segment_traces: dict[int, list[hv.RetentionTrace]] = {}
    for segment in segments:
        parameters = None
        if operator == "SWINGING_DOOR":
            parameters = [
                s101._TEP_CALIBRATION_MAP[(k, channel)]
                for channel in range(len(s101.TEP_CHANNELS))
            ]
        segment_traces[segment.trajectory] = s101.make_traces(
            segment.values, operator=operator, k=k, sda_parameters=parameters
        )

    for region, target_name, horizon in s101.active_tep_targets(
        smoke, target_panel
    ):
        target_channel = s101.TEP_CHANNEL_INDEX[target_name]
        final_parts: list[np.ndarray] = []
        event_parts: list[np.ndarray] = []
        commit_parts: list[np.ndarray] = []
        native_parts: list[np.ndarray] = []
        availability_parts: list[np.ndarray] = []
        active_parts: list[np.ndarray] = []
        origin_parts: list[np.ndarray] = []
        ratio_parts: list[np.ndarray] = []
        target_parts: list[np.ndarray] = []
        event_current_parts: list[np.ndarray] = []
        commit_current_parts: list[np.ndarray] = []
        fault_parts: list[np.ndarray] = []
        batch_parts: list[np.ndarray] = []

        max_grid_delta = -10**9
        max_commit_delta = -10**9
        max_event_commit_difference = 0.0
        max_final_export_difference = 0.0
        retained_event_indices_identical = True
        for segment in segments:
            origins = np.arange(
                max(s101.TEP_LAGS), len(segment.values) - horizon, dtype=np.int64
            )
            commit_traces = segment_traces[segment.trajectory]
            event_traces = [
                hv.event_time_only_trace(trace, len(segment.values))
                for trace in commit_traces
            ]
            retained_event_indices_identical = (
                retained_event_indices_identical
                and all(
                    np.array_equal(commit.event_indices, event.event_indices)
                    for commit, event in zip(commit_traces, event_traces)
                )
            )
            commit_bundle = hv.build_vintage_features(
                segment.values,
                commit_traces,
                origins,
                s101.TEP_LAGS,
                target_channel=target_channel,
                horizon=horizon,
            )
            event_bundle = hv.build_vintage_features(
                segment.values,
                event_traces,
                origins,
                s101.TEP_LAGS,
                target_channel=target_channel,
                horizon=horizon,
            )
            final_parts.append(commit_bundle.final_features.astype(np.float32))
            event_parts.append(event_bundle.origin_features.astype(np.float32))
            commit_parts.append(commit_bundle.origin_features.astype(np.float32))
            native_parts.append(
                direct_features(segment.values, origins, s101.TEP_LAGS).astype(np.float32)
            )
            availability_parts.append(commit_bundle.final_feature_availability)
            active_parts.append(commit_bundle.active_future_mask)
            origin_parts.append(origins)
            ratio = retained_ratio(commit_traces, len(segment.values))
            ratio_parts.append(np.full(len(origins), ratio))
            target_parts.append(segment.values[origins + horizon, target_channel])
            event_current_parts.append(event_bundle.origin_current)
            commit_current_parts.append(commit_bundle.origin_current)
            fault_parts.append(np.full(len(origins), segment.fault, dtype=np.int16))
            batch_parts.append(np.full(len(origins), segment.batch, dtype=np.int16))
            max_grid_delta = max(
                max_grid_delta,
                int(np.max(commit_bundle.feature_times - origins[:, None])),
            )
            max_commit_delta = max(
                max_commit_delta,
                int(
                    np.max(
                        commit_bundle.origin_source_commit - origins[:, None]
                    )
                ),
            )
            max_event_commit_difference = max(
                max_event_commit_difference,
                float(
                    np.max(
                        np.abs(
                            event_bundle.origin_features
                            - commit_bundle.origin_features
                        )
                    )
                ),
            )
            max_final_export_difference = max(
                max_final_export_difference,
                float(
                    np.max(
                        np.abs(
                            event_bundle.final_features
                            - commit_bundle.final_features
                        )
                    )
                ),
            )

        final_features = np.concatenate(final_parts)
        event_features = np.concatenate(event_parts)
        commit_features = np.concatenate(commit_parts)
        native_features = np.concatenate(native_parts)
        final_availability = np.concatenate(availability_parts)
        active_mask = np.concatenate(active_parts)
        origins_all = np.concatenate(origin_parts)
        retention_by_origin = np.concatenate(ratio_parts)
        target = np.concatenate(target_parts)
        faults = np.concatenate(fault_parts)
        batches = np.concatenate(batch_parts)
        event_current = np.concatenate(event_current_parts)
        commit_current = np.concatenate(commit_current_parts)

        for fold, train_batches, test_batches in folds:
            prior = np.flatnonzero(np.isin(batches, train_batches))
            train = s101.even_subsample(prior)
            fitted_panels = {
                "FINAL": s101.fit_panel(
                    event_features, target, train, process="TEP", smoke=smoke
                ),
                "ORIGIN": s101.fit_panel(
                    commit_features, target, train, process="TEP", smoke=smoke
                ),
            }
            for fault in sorted(np.unique(faults)):
                test = np.flatnonzero(
                    np.isin(batches, test_batches) & (faults == fault)
                )
                metadata = {
                    "process": "TEP",
                    "region": region,
                    "target": target_name,
                    "horizon_minutes": float(horizon),
                    "seed": -1,
                    "fold": fold,
                    "fault": int(fault),
                    "operator": operator,
                    "severity_k": k,
                    "nominal_retention": 1.0 / k,
                }
                _, cells = s101.evaluate_prediction_cell(
                    final_features=event_features,
                    origin_features=commit_features,
                    final_current=event_current,
                    origin_current=commit_current,
                    target=target,
                    train=train,
                    test=test,
                    process="TEP",
                    metadata=metadata,
                    smoke=smoke,
                    fitted_panels=fitted_panels,
                )
                output_cells.extend(cells)
                output_diagnostics.append(
                    diagnostic_row(
                        final_features=final_features,
                        event_features=event_features,
                        commit_features=commit_features,
                        native_features=native_features,
                        final_availability=final_availability,
                        active_future_mask=active_mask,
                        origins=origins_all,
                        rows=test,
                        horizon_steps=horizon,
                        dt_minutes=1.0,
                        retained_ratio_by_origin=retention_by_origin,
                        channels=s101.TEP_CHANNELS,
                        n_lags=len(s101.TEP_LAGS),
                        metadata=metadata,
                    )
                )

        output_checks.append(
            {
                "process": "TEP",
                "region": region,
                "operator": operator,
                "severity_k": k,
                "retained_event_indices_identical": bool(
                    retained_event_indices_identical
                ),
                "max_final_export_difference": max_final_export_difference,
                "max_grid_time_minus_origin": max_grid_delta,
                "max_commit_source_minus_origin": max_commit_delta,
                "max_event_commit_feature_difference": max_event_commit_difference,
            }
        )
    return {
        "cells": output_cells,
        "diagnostics": output_diagnostics,
        "checks": output_checks,
    }


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if np.std(x_rank) <= 1e-15 or np.std(y_rank) <= 1e-15:
        return math.nan
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def diagnostic_associations(
    diagnostics: pd.DataFrame, final_cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ridge = final_cells.loc[
        final_cells["score"].eq("MAE") & final_cells["model"].str.contains("RIDGE")
    ].copy()
    join_keys = list(META_COLUMNS[:-1])
    selected = ridge.loc[
        :,
        join_keys
        + [
            "evaluation_view_gap",
            "right_training_repair",
            "diagonal_gap",
            "practical_evaluation_winner_change_1pct",
            "practical_training_repair_winner_change_1pct",
        ],
    ]
    merged = diagnostics.merge(selected, on=join_keys, how="inner", validate="one_to_one")
    if len(merged) != len(diagnostics):
        raise AssertionError("diagnostic and primary Ridge units are not aligned")
    merged["absolute_evaluation_view_gap"] = np.abs(merged["evaluation_view_gap"])

    rows: list[dict[str, Any]] = []
    grouping_specs: list[tuple[str, list[str]]] = [
        ("REGION_OPERATOR", ["process", "region", "operator"]),
        ("REGION_ALL_OPERATORS", ["process", "region"]),
    ]
    for scope, group_columns in grouping_specs:
        for keys, group in merged.groupby(group_columns, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            metadata = {column: value for column, value in zip(group_columns, keys)}
            for diagnostic in DIAGNOSTICS:
                values = group[diagnostic].to_numpy(dtype=float)
                signed = group["evaluation_view_gap"].to_numpy(dtype=float)
                absolute = group["absolute_evaluation_view_gap"].to_numpy(dtype=float)
                lower = float(np.quantile(values, 0.25))
                upper = float(np.quantile(values, 0.75))
                low_mask = values <= lower
                high_mask = values >= upper
                rows.append(
                    {
                        "scope": scope,
                        **metadata,
                        "diagnostic": diagnostic,
                        "n_units": int(len(group)),
                        "n_unique_values": int(len(np.unique(values))),
                        "spearman_signed_deployment_gap": rank_correlation(values, signed),
                        "spearman_absolute_deployment_gap": rank_correlation(values, absolute),
                        "upper_minus_lower_signed_gap": float(
                            np.mean(signed[high_mask]) - np.mean(signed[low_mask])
                        ),
                        "upper_minus_lower_absolute_gap": float(
                            np.mean(absolute[high_mask]) - np.mean(absolute[low_mask])
                        ),
                        "upper_minus_lower_winner_change_rate": float(
                            group.loc[
                                high_mask,
                                "practical_evaluation_winner_change_1pct",
                            ].mean()
                            - group.loc[
                                low_mask,
                                "practical_evaluation_winner_change_1pct",
                            ].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows), merged


def classify_remedy(row: pd.Series) -> str:
    deployment = float(row["evaluation_view_gap"])
    repair = float(row["right_training_repair"])
    diagonal = float(row["diagonal_gap"])
    if max(abs(deployment), abs(diagonal)) < 0.01:
        return "PRACTICALLY_UNCHANGED"
    if deployment >= 0.01 and repair >= 0.01:
        return "EVALUATION_AND_RETRAINING_REPAIR"
    if deployment >= 0.01 and abs(repair) < 0.01:
        return "EVALUATION_CORRECTION_DOMINANT"
    if deployment <= -0.01:
        return "ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS"
    if repair <= -0.01:
        return "RETRAINING_ADVERSE_OR_HETEROGENEOUS"
    return "MIXED_SMALL_COMPONENTS"


def remedy_table(final_cells: pd.DataFrame) -> pd.DataFrame:
    output = final_cells.loc[
        final_cells["score"].eq("MAE") & final_cells["model"].str.contains("RIDGE")
    ].copy()
    output["remedy_class"] = output.apply(classify_remedy, axis=1)
    return output


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def summarize_results(
    *,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
    associations: pd.DataFrame,
    remedies: pd.DataFrame,
    checks: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    geometry_summary: list[dict[str, Any]] = []
    for (region, operator), group in diagnostics.groupby(["region", "operator"]):
        geometry_summary.append(
            {
                "region": region,
                "operator": operator,
                "n_units": int(len(group)),
                "median_grid_pit_miss_share": float(
                    group["naive_grid_pit_miss_share"].median()
                ),
                "median_active_grid_pit_miss_share": float(
                    group["naive_grid_pit_active_miss_share"].median()
                ),
                "median_horizon_erosion_minutes": float(
                    group["median_horizon_erosion_minutes"].median()
                ),
                "median_final_native_feature_sd_nrmse": float(
                    group["final_native_feature_sd_nrmse"].median()
                ),
                "median_event_commit_feature_sd_nrmse": float(
                    group["event_commit_feature_sd_nrmse"].median()
                ),
            }
        )

    event = comparisons.loc[
        comparisons["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & comparisons["score"].eq("MAE")
        & comparisons["model"].str.contains("RIDGE")
    ]
    commit_summary: list[dict[str, Any]] = []
    for (region, operator), group in event.groupby(["region", "operator"]):
        commit_summary.append(
            {
                "region": region,
                "operator": operator,
                "n_units": int(len(group)),
                "median_evaluation_gap": float(group["evaluation_view_gap"].median()),
                "max_absolute_evaluation_gap": float(
                    np.max(np.abs(group["evaluation_view_gap"]))
                ),
                "practical_winner_changes": int(
                    group["practical_evaluation_winner_change_1pct"].sum()
                ),
            }
        )

    remedy_summary: list[dict[str, Any]] = []
    counts = (
        remedies.groupby(["region", "operator", "remedy_class"])
        .size()
        .rename("n_units")
        .reset_index()
    )
    for row in counts.to_dict("records"):
        remedy_summary.append({key: finite_or_none(value) for key, value in row.items()})

    association_highlights: list[dict[str, Any]] = []
    selected = associations.loc[associations["scope"].eq("REGION_OPERATOR")]
    for (region, operator), group in selected.groupby(["region", "operator"]):
        valid = group.dropna(subset=["spearman_absolute_deployment_gap"])
        if len(valid) == 0:
            continue
        best = valid.iloc[
            np.argmax(np.abs(valid["spearman_absolute_deployment_gap"].to_numpy()))
        ]
        association_highlights.append(
            {
                "region": region,
                "operator": operator,
                "diagnostic_with_largest_absolute_rho": str(best["diagnostic"]),
                "spearman_absolute_deployment_gap": float(
                    best["spearman_absolute_deployment_gap"]
                ),
            }
        )

    return {
        "specification": SPEC,
        "run_mode": "SMOKE" if smoke else "FULL",
        "claim_boundary": (
            "Explicit periodic and swinging-door source-event operators; not vendor API "
            "behavior or industrial prevalence."
        ),
        "counts": {
            "view_comparison_rows": int(len(comparisons)),
            "diagnostic_units": int(len(diagnostics)),
            "association_rows": int(len(associations)),
            "remedy_units": int(len(remedies)),
        },
        "checks": checks,
        "geometry_by_region_operator": geometry_summary,
        "event_vs_commit_by_region_operator": commit_summary,
        "remedy_counts": remedy_summary,
        "association_highlights": association_highlights,
        "contribution_state": "PENDING_MANAGER_INTERPRETATION",
    }


def validate_outputs(
    *,
    parent_cells: pd.DataFrame,
    event_cells: pd.DataFrame,
    comparisons: pd.DataFrame,
    diagnostics: pd.DataFrame,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(parent_cells) != len(event_cells):
        raise AssertionError("event/commit and parent cell counts differ")
    for name, frame in (
        ("parent", parent_cells),
        ("event", event_cells),
    ):
        if frame.duplicated(list(KEY_COLUMNS)).any():
            raise AssertionError(f"{name} cells contain duplicate keys")
    if diagnostics.duplicated(list(META_COLUMNS[:-1])).any():
        raise AssertionError("diagnostic units contain duplicate keys")
    numeric = diagnostics.select_dtypes(include=[np.number])
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise AssertionError("diagnostics contain non-finite numeric values")

    merged = event_cells.merge(
        parent_cells,
        on=list(KEY_COLUMNS),
        suffixes=("_event", "_parent"),
        validate="one_to_one",
    )
    commit_replay_delta = float(
        np.max(
            np.abs(
                merged["loss_to_eo_event"].to_numpy()
                - merged["loss_to_eo_parent"].to_numpy()
            )
        )
    )
    if commit_replay_delta > 1e-10:
        raise AssertionError(
            f"event/commit right-right score differs from script 101: {commit_replay_delta}"
        )

    periodic = event_cells["operator"].eq("PERIODIC")
    periodic_columns = ["loss_tf_ef", "loss_tf_eo", "loss_to_ef", "loss_to_eo"]
    periodic_delta = float(
        np.max(
            np.abs(
                event_cells.loc[periodic, periodic_columns].to_numpy()
                - event_cells.loc[periodic, ["loss_to_eo"] * 4].to_numpy()
            )
        )
    )
    if periodic_delta > 1e-10:
        raise AssertionError(f"periodic event/commit views differ: {periodic_delta}")

    max_grid_delta = max(int(row["max_grid_time_minus_origin"]) for row in checks)
    max_commit_delta = max(int(row["max_commit_source_minus_origin"]) for row in checks)
    same_retained_events = all(
        bool(row["retained_event_indices_identical"]) for row in checks
    )
    max_final_export_delta = max(
        float(row["max_final_export_difference"]) for row in checks
    )
    if not same_retained_events or max_final_export_delta > 1e-12:
        raise AssertionError(
            "event-time and commit-time ablations changed retained events or final export"
        )
    if max_grid_delta > 0 or max_commit_delta > 0:
        raise AssertionError("point-in-time source check failed")
    expected_comparisons = 2 * len(parent_cells)
    if len(comparisons) != expected_comparisons:
        raise AssertionError("unexpected comparison row count")
    return {
        "parent_and_event_cell_rows": int(len(parent_cells)),
        "commit_replay_score_max_abs_delta": commit_replay_delta,
        "periodic_event_commit_score_max_abs_delta": periodic_delta,
        "event_commit_retained_events_identical": same_retained_events,
        "event_commit_final_export_max_abs_delta": max_final_export_delta,
        "max_grid_time_minus_origin": max_grid_delta,
        "max_commit_source_minus_origin": max_commit_delta,
    }


def plot_geometry(diagnostics: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )
    grouped = (
        diagnostics.groupby(["region", "operator", "severity_k"])
        .agg(
            pit_miss=("naive_grid_pit_miss_share", "median"),
            erosion=("median_horizon_erosion_minutes", "median"),
        )
        .reset_index()
    )
    regions = list(grouped["region"].drop_duplicates())
    markers = {"PERIODIC": "o", "SWINGING_DOOR": "s"}
    colors = ["#1f4e79", "#9c3d10", "#4d4d4d"]
    region_labels = {
        "CSTR_OUTPUT_Q1": "CSTR",
        "PRESSURE_Q16": "TEP pressure",
        "TEMPERATURE_Q6": "TEP temperature",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    for color, region in zip(colors, regions):
        for operator in s101.OPERATORS:
            cell = grouped.loc[
                grouped["region"].eq(region) & grouped["operator"].eq(operator)
            ].sort_values("severity_k")
            if len(cell) == 0:
                continue
            operator_label = "periodic" if operator == "PERIODIC" else "swinging door"
            label = f"{region_labels.get(region, region)} / {operator_label}"
            axes[0].plot(
                cell["severity_k"],
                cell["pit_miss"],
                marker=markers[operator],
                color=color,
                linestyle="-" if operator == "PERIODIC" else "--",
                label=label,
            )
            axes[1].plot(
                cell["severity_k"],
                cell["erosion"],
                marker=markers[operator],
                color=color,
                linestyle="-" if operator == "PERIODIC" else "--",
            )
    axes[0].set_xlabel("Retention severity, k")
    axes[0].set_ylabel("Origins missed by grid-time PIT")
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_xlabel("Retention severity, k")
    axes[1].set_ylabel("Median horizon erosion (min; symlog)")
    axes[1].set_yscale("symlog", linthresh=1.0, linscale=0.8)
    axes[1].set_yticks([0, 1, 5, 10, 30, 100, 300])
    axes[1].set_yticklabels(["0", "1", "5", "10", "30", "100", "300"])
    axes[0].text(0.0, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.0, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    for axis in axes:
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.22, 1.0, 1.0))
    fig.savefig(out_dir / "pit_miss_and_horizon_erosion.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "pit_miss_and_horizon_erosion.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_geometry_consequence(merged: pd.DataFrame, out_dir: Path) -> None:
    regions = list(merged["region"].drop_duplicates())
    colors = ["#1f4e79", "#9c3d10", "#4d4d4d"]
    markers = {"PERIODIC": "o", "SWINGING_DOOR": "s"}
    region_labels = {
        "CSTR_OUTPUT_Q1": "CSTR",
        "PRESSURE_Q16": "TEP pressure",
        "TEMPERATURE_Q6": "TEP temperature",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    for color, region in zip(colors, regions):
        for operator in s101.OPERATORS:
            cell = merged.loc[
                merged["region"].eq(region) & merged["operator"].eq(operator)
            ]
            if len(cell) == 0:
                continue
            operator_label = "periodic" if operator == "PERIODIC" else "swinging door"
            label = f"{region_labels.get(region, region)} / {operator_label}"
            axes[0].scatter(
                cell["final_native_feature_sd_nrmse"],
                100.0 * cell["evaluation_view_gap"],
                s=18,
                alpha=0.65,
                color=color,
                marker=markers[operator],
                label=label,
            )
            axes[1].scatter(
                cell["median_horizon_erosion_minutes"],
                100.0 * cell["evaluation_view_gap"],
                s=18,
                alpha=0.65,
                color=color,
                marker=markers[operator],
            )
    axes[0].set_xlabel("Final-vs-native feature SD-NRMSE")
    axes[0].set_ylabel("Ridge deployment gap (%)")
    axes[1].set_xlabel("Median horizon erosion (min; symlog)")
    axes[1].set_xscale("symlog", linthresh=1.0, linscale=0.8)
    axes[1].set_xticks([0, 1, 5, 10, 30, 100, 300])
    axes[1].set_xticklabels(["0", "1", "5", "10", "30", "100", "300"])
    axes[1].set_ylabel("Ridge deployment gap (%)")
    axes[0].text(0.0, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")
    axes[1].text(0.0, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    for axis in axes:
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.grid(color="0.9", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0.0, 0.22, 1.0, 1.0))
    fig.savefig(out_dir / "geometry_vs_deployment_gap.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "geometry_vs_deployment_gap.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_method_schematic(out_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.1, 2.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.25")
    items = (
        (0.10, "Retained events\nevent τ, commit c"),
        (0.30, "Cell reconstruction\n+ source lineage"),
        (0.50, "Window availability\n+ effective horizon"),
        (0.70, "Origin replay\nFINAL ↔ ORIGIN"),
        (0.90, "Loss, ranking,\n+ remedy"),
    )
    for x, label in items:
        axis.text(x, 0.55, label, ha="center", va="center", bbox=box, fontsize=7.8)
    for (x0, _), (x1, _) in zip(items[:-1], items[1:]):
        axis.annotate(
            "",
            xy=(x1 - 0.08, 0.55),
            xytext=(x0 + 0.08, 0.55),
            arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 1.0},
        )
    axis.text(
        0.31,
        0.17,
        "Grid timestamp alone can precede the latest source commit",
        ha="center",
        va="center",
        fontsize=9,
        color="#9c3d10",
    )
    axis.annotate(
        "",
        xy=(0.31, 0.39),
        xytext=(0.31, 0.23),
        arrowprops={"arrowstyle": "->", "color": "#9c3d10", "lw": 1.0},
    )
    fig.tight_layout()
    fig.savefig(out_dir / "reconstruction_lineage_method_schematic.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "reconstruction_lineage_method_schematic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run(
    smoke: bool,
    workers: int | None,
    out_dir: Path,
    *,
    parent_dir: Path | None = None,
    target_panel: str = "selected",
) -> dict[str, Any]:
    parent_dir = parent_dir or (PARENT_SMOKE if smoke else PARENT_FULL)
    parent_path = parent_dir / "forecast_cells.csv"
    if not parent_path.exists():
        raise FileNotFoundError(f"missing script-101 cells: {parent_path}")
    parent_cells = pd.read_csv(parent_path)

    configs = (
        (("PERIODIC", 2), ("SWINGING_DOOR", 2))
        if smoke
        else tuple((operator, k) for operator in s101.OPERATORS for k in s101.LEVELS)
    )
    cstr_seeds = (0,) if smoke else tuple(range(8))
    cstr_tasks = [
        (seed, operator, k, smoke)
        for seed in cstr_seeds
        for operator, k in configs
    ]
    outputs: dict[str, list[Any]] = {"cells": [], "diagnostics": [], "checks": []}
    cstr_workers = choose_workers(workers, len(cstr_tasks), cap=16)
    log(f"CSTR event/commit tasks={len(cstr_tasks)} workers={cstr_workers}")
    with ProcessPoolExecutor(max_workers=cstr_workers) as executor:
        futures = {executor.submit(run_cstr_event_commit, task): task for task in cstr_tasks}
        for count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            for key in outputs:
                outputs[key].extend(result[key])
            if count == len(futures) or count % max(1, len(futures) // 4) == 0:
                log(f"CSTR progress {count}/{len(futures)}")

    segments = s101.load_tep_segments(smoke)
    calibration = s101.load_tep_calibration()
    tep_workers = choose_workers(workers, len(configs), cap=8)
    log(f"TEP event/commit configs={len(configs)} workers={tep_workers}")
    with ProcessPoolExecutor(
        max_workers=tep_workers,
        initializer=s101.init_tep_worker,
        initargs=(segments, calibration, smoke),
    ) as executor:
        tep_tasks = [(*task, target_panel) for task in configs]
        futures = {
            executor.submit(run_tep_event_commit, task): task for task in tep_tasks
        }
        for count, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            for key in outputs:
                outputs[key].extend(result[key])
            log(f"TEP progress {count}/{len(futures)}")

    event_cells = pd.DataFrame(outputs["cells"])
    diagnostics = pd.DataFrame(outputs["diagnostics"])
    final_comparison = genericize_cells(
        parent_cells,
        comparison="FINAL_VS_EVENT_COMMIT",
        left_view="FINAL_EXPORT",
        right_view="SOURCE_EVENT_COMMIT_PIT",
    )
    event_comparison = genericize_cells(
        event_cells,
        comparison="EVENT_TIME_VS_EVENT_COMMIT",
        left_view="SOURCE_EVENT_TIME_PIT",
        right_view="SOURCE_EVENT_COMMIT_PIT",
    )
    comparisons = pd.concat([final_comparison, event_comparison], ignore_index=True)
    validation = validate_outputs(
        parent_cells=parent_cells,
        event_cells=event_cells,
        comparisons=comparisons,
        diagnostics=diagnostics,
        checks=outputs["checks"],
    )
    associations, merged = diagnostic_associations(diagnostics, final_comparison)
    remedies = remedy_table(final_comparison)
    checks = {
        **validation,
        "event_commit_task_checks": len(outputs["checks"]),
        "periodic_event_commit_feature_max_abs_delta": float(
            diagnostics.loc[
                diagnostics["operator"].eq("PERIODIC"),
                "event_commit_feature_sd_nrmse",
            ].max()
        ),
    }
    if checks["periodic_event_commit_feature_max_abs_delta"] > 1e-12:
        raise AssertionError("periodic event/commit feature ablation is not identical")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(out_dir / "view_comparison_cells.csv", index=False)
    diagnostics.to_csv(out_dir / "geometry_differentiation.csv", index=False)
    associations.to_csv(out_dir / "diagnostic_associations.csv", index=False)
    remedies.to_csv(out_dir / "remedy_attribution.csv", index=False)
    summary = summarize_results(
        comparisons=comparisons,
        diagnostics=diagnostics,
        associations=associations,
        remedies=remedies,
        checks=checks,
        smoke=smoke,
    )
    summary["tep_target_panel"] = target_panel
    summary["parent_result_dir"] = str(parent_dir.relative_to(ROOT) if parent_dir.is_absolute() else parent_dir)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    plot_method_schematic(out_dir)
    plot_geometry(diagnostics, out_dir)
    plot_geometry_consequence(merged, out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tep-panel",
        choices=tuple(s101.TEP_TARGET_PANELS),
        default="selected",
        help="TEP target--horizon panel; full2x2 requires explicit paths",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=None,
        help="script-101 result directory containing forecast_cells.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = bool(args.smoke)
    if args.tep_panel != "selected" and (args.out is None or args.parent is None):
        raise SystemExit(
            "--out and --parent are required when --tep-panel is not selected"
        )
    out_dir = args.out or (SMOKE_OUT if smoke else DEFAULT_OUT)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(
        "starting historian lineage differentiation; "
        f"smoke={smoke} tep_panel={args.tep_panel}"
    )
    summary = run(
        smoke=smoke,
        workers=args.workers,
        out_dir=out_dir,
        parent_dir=args.parent,
        target_panel=args.tep_panel,
    )
    log(
        "completed historian lineage differentiation; "
        f"units={summary['counts']['diagnostic_units']}"
    )


if __name__ == "__main__":
    main()
