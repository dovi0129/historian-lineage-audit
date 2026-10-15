#!/usr/bin/env python3
"""Run the approved historian final-vintage versus origin-vintage kill test.

The experiment models explicit source-event lineage and commit/knowledge time.
It does not emulate or make claims about a vendor-specific historian API.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import historian_inferential_instability as hii  # noqa: E402
import historian_vintage as hv  # noqa: E402
from preview_stage_a import CSTRTraceConfig, run_cstr_preview  # noqa: E402

SPEC = "reports/swarm/specs/2026-09-03_historian_vintage_replay_kill_test.approved.md"
DEFAULT_OUT = ROOT / "artifacts" / "historian_vintage_replay_kill_test"
SMOKE_OUT = ROOT / "artifacts" / "historian_vintage_replay_kill_test_smoke"
LOG = ROOT / "artifacts" / "logs" / "101_historian_vintage_replay_kill_test.log"
TEP_DATA = ROOT / "data" / "tep"
TEP_CALIBRATION = ROOT / "config" / "operator_calibration.csv"

LEVELS = (2, 3, 5, 8)
OPERATORS = ("PERIODIC", "SWINGING_DOOR")
SCORES = ("MAE", "RMSE")
QUADRANTS = ("TF_EF", "TF_EO", "TO_EF", "TO_EO")
RANDOM_SEED = 20260903
TRAIN_CAP = 50_000

CSTR_DT = 0.1
CSTR_CHANNELS = ("measured_output", "action")
CSTR_LAGS = (0, 10, 20, 30, 40, 50)
CSTR_HORIZON = 10
CSTR_MODELS = ("PERSISTENCE", "RIDGE_AR", "SHALLOW_GBR")

TEP_XMEAS = tuple(f"XMEAS-{index}" for index in range(1, 23))
TEP_XMV = tuple(f"XMV-{index}" for index in range(1, 13))
TEP_CHANNELS = TEP_XMEAS + TEP_XMV
TEP_CHANNEL_INDEX = {name: index for index, name in enumerate(TEP_CHANNELS)}
TEP_LAGS = (0, 2, 4, 6, 8)
TEP_MODELS = ("PERSISTENCE", "RIDGE_ARX", "SHALLOW_HGB")
TEP_TARGETS = (("PRESSURE_Q16", "XMEAS-7", 16), ("TEMPERATURE_Q6", "XMEAS-9", 6))
TEP_FULL_PANEL_TARGETS = (
    ("PRESSURE_Q6", "XMEAS-7", 6),
    ("PRESSURE_Q16", "XMEAS-7", 16),
    ("TEMPERATURE_Q6", "XMEAS-9", 6),
    ("TEMPERATURE_Q16", "XMEAS-9", 16),
)
TEP_TARGET_PANELS = {
    "selected": TEP_TARGETS,
    "full2x2": TEP_FULL_PANEL_TARGETS,
}
TEP_FAULTS = (1, 4, 5, 11, 14, 15)
TEP_BATCHES = tuple(range(1, 11))
TEP_FOLDS = (
    (1, tuple(range(1, 5)), (5, 6)),
    (2, tuple(range(1, 7)), (7, 8)),
    (3, tuple(range(1, 9)), (9, 10)),
)


class Geometry:
    def __init__(self, n_steps: int, burn: int, calibration: int, analysis: int):
        self.n_steps = int(n_steps)
        self.burn = int(burn)
        self.calibration = int(calibration)
        self.analysis = int(analysis)

    @property
    def calibration_slice(self) -> slice:
        return slice(self.burn, self.burn + self.calibration)

    @property
    def analysis_slice(self) -> slice:
        start = self.burn + self.calibration
        return slice(start, start + self.analysis)


FULL_CSTR_GEOMETRY = Geometry(1536, 256, 256, 1024)
SMOKE_CSTR_GEOMETRY = Geometry(768, 128, 128, 512)


@dataclass(frozen=True)
class TEPSegment:
    fault: int
    batch: int
    values: np.ndarray

    @property
    def trajectory(self) -> int:
        return self.fault * 100 + self.batch


_TEP_SEGMENTS: list[TEPSegment] | None = None
_TEP_CALIBRATION_MAP: dict[tuple[int, int], tuple[float, bool]] | None = None
_TEP_SMOKE = False


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def choose_workers(requested: int | None, tasks: int) -> int:
    cores = os.cpu_count() or 8
    if requested is not None:
        return max(1, min(int(requested), cores, tasks))
    return max(1, min(8, cores - 2, tasks))


def even_subsample(indices: np.ndarray, cap: int = TRAIN_CAP) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= cap:
        return indices
    positions = (np.arange(cap, dtype=np.int64) * (len(indices) - 1)) // (cap - 1)
    selected = indices[positions]
    if len(np.unique(selected)) != cap:
        raise AssertionError("deterministic training subsample is not unique")
    return selected


def point_score(residual: np.ndarray, score: str) -> float:
    if score == "MAE":
        return float(np.mean(np.abs(residual)))
    if score == "RMSE":
        return float(np.sqrt(np.mean(np.square(residual))))
    raise ValueError(score)


def safe_relative(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 1e-15 else math.nan


def rank_models(values: dict[str, float], order: Sequence[str]) -> list[str]:
    priority = {name: index for index, name in enumerate(order)}
    return sorted(order, key=lambda name: (values[name], priority[name]))


def fit_panel(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    *,
    process: str,
    smoke: bool,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    x_train = np.asarray(features[train], dtype=float)
    y_train = np.asarray(target[train], dtype=float)
    scaler = StandardScaler().fit(x_train)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), y_train)
    if process == "CSTR":
        nonlinear = GradientBoostingRegressor(
            n_estimators=20 if smoke else 100,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        ridge_name, nonlinear_name = "RIDGE_AR", "SHALLOW_GBR"
    elif process == "TEP":
        nonlinear = HistGradientBoostingRegressor(
            max_iter=10 if smoke else 100,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
        ).fit(x_train, y_train)
        ridge_name, nonlinear_name = "RIDGE_ARX", "SHALLOW_HGB"
    else:
        raise ValueError(process)

    return {
        ridge_name: lambda values: np.asarray(
            ridge.predict(scaler.transform(np.asarray(values, dtype=float))), dtype=float
        ),
        nonlinear_name: lambda values: np.asarray(
            nonlinear.predict(np.asarray(values, dtype=float)), dtype=float
        ),
    }


def evaluate_prediction_cell(
    *,
    final_features: np.ndarray,
    origin_features: np.ndarray,
    final_current: np.ndarray,
    origin_current: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    process: str,
    metadata: dict[str, Any],
    smoke: bool,
    fitted_panels: dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit both vintages and score all four train/evaluation quadrants."""

    models = CSTR_MODELS if process == "CSTR" else TEP_MODELS
    fitted = fitted_panels or {
        "FINAL": fit_panel(final_features, target, train, process=process, smoke=smoke),
        "ORIGIN": fit_panel(origin_features, target, train, process=process, smoke=smoke),
    }
    features = {"FINAL": final_features, "ORIGIN": origin_features}
    current = {"FINAL": final_current, "ORIGIN": origin_current}
    quadrant_views = {
        "TF_EF": ("FINAL", "FINAL"),
        "TF_EO": ("FINAL", "ORIGIN"),
        "TO_EF": ("ORIGIN", "FINAL"),
        "TO_EO": ("ORIGIN", "ORIGIN"),
    }
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for quadrant, (train_view, eval_view) in quadrant_views.items():
        for model in models:
            if model == "PERSISTENCE":
                pred = np.asarray(current[eval_view][test], dtype=float)
            else:
                pred = fitted[train_view][model](features[eval_view][test])
            if pred.shape != target[test].shape or not np.all(np.isfinite(pred)):
                raise RuntimeError(f"invalid prediction for {quadrant}/{model}")
            predictions[(quadrant, model)] = pred

    y_test = np.asarray(target[test], dtype=float)
    y_sd = float(np.std(y_test))
    sensitivity: dict[tuple[str, str], float] = {}
    for train_view, prefix in (("FINAL", "TF"), ("ORIGIN", "TO")):
        for model in models:
            difference = predictions[(f"{prefix}_EF", model)] - predictions[(f"{prefix}_EO", model)]
            rms = float(np.sqrt(np.mean(np.square(difference))))
            sensitivity[(train_view, model)] = safe_relative(rms, y_sd)

    score_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for score in SCORES:
        values: dict[str, dict[str, float]] = {}
        rankings: dict[str, list[str]] = {}
        margins: dict[str, float] = {}
        for quadrant in QUADRANTS:
            values[quadrant] = {
                model: point_score(y_test - predictions[(quadrant, model)], score)
                for model in models
            }
            ranking = rank_models(values[quadrant], models)
            rankings[quadrant] = ranking
            margins[quadrant] = safe_relative(
                values[quadrant][ranking[1]] - values[quadrant][ranking[0]],
                values[quadrant][ranking[0]],
            )
            train_view = "FINAL" if quadrant.startswith("TF") else "ORIGIN"
            for model in models:
                score_rows.append(
                    {
                        **metadata,
                        "score": score,
                        "quadrant": quadrant,
                        "train_view": train_view,
                        "evaluation_view": "FINAL" if quadrant.endswith("EF") else "ORIGIN",
                        "model": model,
                        "loss": values[quadrant][model],
                        "prediction_view_nrmse": sensitivity[(train_view, model)],
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                    }
                )

        practical_change = int(
            rankings["TF_EF"][0] != rankings["TO_EO"][0]
            and max(margins["TF_EF"], margins["TO_EO"]) >= 0.01
        )
        practical_deployment_change = int(
            rankings["TF_EF"][0] != rankings["TF_EO"][0]
            and max(margins["TF_EF"], margins["TF_EO"]) >= 0.01
        )
        practical_training_repair_change = int(
            rankings["TF_EO"][0] != rankings["TO_EO"][0]
            and max(margins["TF_EO"], margins["TO_EO"]) >= 0.01
        )
        winner_fields: dict[str, Any] = {}
        for quadrant in QUADRANTS:
            winner_fields[f"winner_{quadrant.lower()}"] = rankings[quadrant][0]
            winner_fields[f"ranking_{quadrant.lower()}"] = "|".join(rankings[quadrant])
            winner_fields[f"margin_{quadrant.lower()}"] = margins[quadrant]

        for model in models:
            ff = values["TF_EF"][model]
            fo = values["TF_EO"][model]
            of = values["TO_EF"][model]
            oo = values["TO_EO"][model]
            deployment_gap = safe_relative(fo - ff, ff)
            origin_repair = safe_relative(fo - oo, fo)
            diagonal_gap = safe_relative(oo - ff, ff)
            cell_rows.append(
                {
                    **metadata,
                    "score": score,
                    "model": model,
                    "loss_tf_ef": ff,
                    "loss_tf_eo": fo,
                    "loss_to_ef": of,
                    "loss_to_eo": oo,
                    "deployment_gap": deployment_gap,
                    "origin_training_repair": origin_repair,
                    "diagonal_gap": diagonal_gap,
                    "absolute_material_gap_1pct": int(
                        max(abs(deployment_gap), abs(diagonal_gap)) >= 0.01
                    ),
                    "optimistic_deployment_gap_1pct": int(deployment_gap >= 0.01),
                    "practical_diagonal_winner_change_1pct": practical_change,
                    "practical_deployment_winner_change_1pct": practical_deployment_change,
                    "practical_training_repair_winner_change_1pct": practical_training_repair_change,
                    "prediction_view_nrmse_final_trained": sensitivity[("FINAL", model)],
                    "prediction_view_nrmse_origin_trained": sensitivity[("ORIGIN", model)],
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    **winner_fields,
                }
            )
    return score_rows, cell_rows


def make_traces(
    panel: np.ndarray,
    *,
    operator: str,
    k: int,
    sda_parameters: Sequence[tuple[float, bool]] | None = None,
) -> list[hv.RetentionTrace]:
    traces: list[hv.RetentionTrace] = []
    for channel in range(panel.shape[1]):
        if operator == "PERIODIC":
            trace = hv.periodic_trace(len(panel), k)
        elif operator == "SWINGING_DOOR":
            if sda_parameters is None:
                raise ValueError("swinging-door parameters are required")
            epsilon, feasible = sda_parameters[channel]
            trace = hv.swinging_door_trace(panel[:, channel], epsilon) if feasible else hv.identity_trace(len(panel))
        else:
            raise ValueError(operator)
        traces.append(trace)
    return traces


def normalized_feature_divergence(
    final: np.ndarray, origin: np.ndarray, columns: np.ndarray
) -> float:
    left = np.asarray(final[:, columns], dtype=float)
    right = np.asarray(origin[:, columns], dtype=float)
    scales = np.std(left, axis=0)
    usable = scales > 1e-15
    if not np.any(usable):
        return 0.0 if np.allclose(left, right) else math.nan
    normalized = (left[:, usable] - right[:, usable]) / scales[usable]
    return float(np.sqrt(np.mean(np.square(normalized))))


def geometry_rows(
    bundle: hv.VintageFeatureBundle,
    *,
    channels: Sequence[str],
    lags: Sequence[int],
    horizon: int,
    n_time: int,
    dt_minutes: float,
    metadata: dict[str, Any],
    traces: Sequence[hv.RetentionTrace],
) -> list[dict[str, Any]]:
    n_lags = len(lags)
    target_times = bundle.origins + int(horizon)
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [("__ALL__", np.arange(bundle.final_features.shape[1]))]
    scopes.extend(
        (name, np.arange(index * n_lags, (index + 1) * n_lags))
        for index, name in enumerate(channels)
    )
    for channel_name, columns in scopes:
        availability = bundle.final_feature_availability[:, columns]
        active = bundle.active_future_mask[:, columns]
        structural = availability > bundle.origins[:, None]
        latest = np.max(availability, axis=1)
        active_latest = np.max(
            np.where(active, availability, bundle.origins[:, None]), axis=1
        )
        strict_h_eff = (target_times - np.maximum(bundle.origins, latest)) * dt_minutes
        active_h_eff = (target_times - np.maximum(bundle.origins, active_latest)) * dt_minutes
        if channel_name == "__ALL__":
            retained_ratio = float(
                np.mean([len(trace.event_indices) / n_time for trace in traces])
            )
        else:
            channel_index = channels.index(channel_name)
            retained_ratio = float(len(traces[channel_index].event_indices) / n_time)
        rows.append(
            {
                **metadata,
                "channel": channel_name,
                "scope": "STRICT_ALL_CHANNELS" if channel_name == "__ALL__" else "CHANNEL_ONLY",
                "n_origins": int(len(bundle.origins)),
                "n_feature_cells": int(len(bundle.origins) * len(columns)),
                "nominal_horizon_minutes": float(horizon * dt_minutes),
                "retained_event_ratio": retained_ratio,
                "structural_future_cell_share": float(np.mean(structural)),
                "active_future_cell_share": float(np.mean(active)),
                "origins_any_structural_future_share": float(np.mean(np.any(structural, axis=1))),
                "origins_any_active_future_share": float(np.mean(np.any(active, axis=1))),
                "strict_latest_at_or_after_target_share": float(np.mean(latest >= target_times)),
                "active_latest_at_or_after_target_share": float(np.mean(active_latest >= target_times)),
                "strict_effective_horizon_mean_minutes": float(np.mean(strict_h_eff)),
                "strict_effective_horizon_median_minutes": float(np.median(strict_h_eff)),
                "strict_effective_horizon_p05_minutes": float(np.quantile(strict_h_eff, 0.05)),
                "strict_effective_horizon_min_minutes": float(np.min(strict_h_eff)),
                "active_effective_horizon_mean_minutes": float(np.mean(active_h_eff)),
                "active_effective_horizon_median_minutes": float(np.median(active_h_eff)),
                "active_effective_horizon_p05_minutes": float(np.quantile(active_h_eff, 0.05)),
                "active_effective_horizon_min_minutes": float(np.min(active_h_eff)),
                "feature_view_divergence_sd_nrmse": normalized_feature_divergence(
                    bundle.final_features, bundle.origin_features, columns
                ),
                "native_target_availability_delay_mean_minutes": 0.0,
                "final_target_future_source_share": float(
                    np.mean(bundle.target_final_availability > target_times)
                ),
                "active_target_future_source_share": float(
                    np.mean(bundle.target_active_availability > target_times)
                ),
                "final_target_availability_delay_mean_minutes": float(
                    np.mean(bundle.target_final_availability - target_times) * dt_minutes
                ),
            }
        )
    return rows


def source_support_rows(
    bundle: hv.VintageFeatureBundle,
    train: np.ndarray,
    test: np.ndarray,
    *,
    n_time: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_features = bundle.final_features.shape[1]
    origin_commit_delta = bundle.origin_source_commit - bundle.origins[:, None]
    origin_event_delta = np.maximum(
        bundle.origin_source_left, bundle.origin_source_right
    ) - bundle.origins[:, None]
    for view in ("FINAL", "ORIGIN"):
        fit_sources = hv.unique_source_ids(bundle, train, view=view, n_time=n_time)
        test_sources = hv.unique_source_ids(bundle, test, view=view, n_time=n_time)
        overlap = np.intersect1d(fit_sources, test_sources, assume_unique=True)
        rows.append(
            {
                **metadata,
                "view": view,
                "n_fit_rows": int(len(train)),
                "n_test_rows": int(len(test)),
                "n_fit_lineage_references": int(len(train) * n_features * 2),
                "n_test_lineage_references": int(len(test) * n_features * 2),
                "n_unique_fit_sources": int(len(fit_sources)),
                "n_unique_test_sources": int(len(test_sources)),
                "n_overlapping_sources": int(len(overlap)),
                "test_source_overlap_share": safe_relative(len(overlap), len(test_sources)),
                "fit_source_reuse_ratio": safe_relative(len(train) * n_features * 2, len(fit_sources)),
                "test_source_reuse_ratio": safe_relative(len(test) * n_features * 2, len(test_sources)),
                "n_post_origin_commit_violations": int(np.sum(origin_commit_delta > 0)),
                "max_origin_commit_minus_origin": int(np.max(origin_commit_delta)),
                "n_post_origin_event_violations": int(np.sum(origin_event_delta > 0)),
                "max_origin_event_minus_origin": int(np.max(origin_event_delta)),
                "overlap_reason": "SAME_TRAJECTORY_CHRONOLOGICAL_SPLIT",
            }
        )
    return rows


def cstr_cache_path(seed: int, geometry: Geometry, smoke: bool) -> Path:
    if smoke:
        directory = ROOT / "artifacts" / "cstr_pathway_severity_replication_smoke" / "trajectories"
    else:
        directory = ROOT / "artifacts" / "historian_inferential_instability_kill" / "trajectories"
    return directory / f"cstr_on_seed{seed}_n{geometry.n_steps}.npz"


def load_or_generate_cstr(seed: int, geometry: Geometry, smoke: bool) -> dict[str, np.ndarray]:
    path = cstr_cache_path(seed, geometry, smoke)
    if path.exists():
        with np.load(path) as loaded:
            arrays = {key: np.asarray(loaded[key], dtype=float) for key in loaded.files}
        if all(len(value) == geometry.n_steps for value in arrays.values()):
            return arrays
    _, trace = run_cstr_preview(
        cfg=CSTRTraceConfig(n_steps=geometry.n_steps),
        seed=seed,
        max_preview=1,
        controller_h=12,
        preview_h=0,
        preview_noise_scale=0.0,
        osc_scale=1.0,
    )
    arrays = {
        "measured_output": np.asarray(trace["output"] + trace["obs"], dtype=float),
        "latent_output_truth": np.asarray(trace["output"], dtype=float),
        "action": np.asarray(trace["action"], dtype=float),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def run_cstr_task(task: tuple[int, str, int, bool]) -> dict[str, list[dict[str, Any]]]:
    seed, operator, k, smoke = task
    geometry = SMOKE_CSTR_GEOMETRY if smoke else FULL_CSTR_GEOMETRY
    trajectory = load_or_generate_cstr(seed, geometry, smoke)
    calibration = np.column_stack(
        [trajectory[channel][geometry.calibration_slice] for channel in CSTR_CHANNELS]
    )
    panel = np.column_stack(
        [trajectory[channel][geometry.analysis_slice] for channel in CSTR_CHANNELS]
    )
    parameters: list[tuple[float, bool]] | None = None
    if operator == "SWINGING_DOOR":
        parameters = []
        for channel in range(panel.shape[1]):
            epsilon, _ = hii.calibrate_swinging_door(calibration[:, channel], 1.0 / k)
            parameters.append((epsilon, True))
    traces = make_traces(panel, operator=operator, k=k, sda_parameters=parameters)
    origins = np.arange(max(CSTR_LAGS), len(panel) - CSTR_HORIZON, dtype=np.int64)
    bundle = hv.build_vintage_features(
        panel,
        traces,
        origins,
        CSTR_LAGS,
        target_channel=0,
        horizon=CSTR_HORIZON,
    )
    target = panel[origins + CSTR_HORIZON, 0]
    train, _, test = hii.chronological_split(len(origins))
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
    score_rows, cell_rows = evaluate_prediction_cell(
        final_features=bundle.final_features,
        origin_features=bundle.origin_features,
        final_current=bundle.final_current,
        origin_current=bundle.origin_current,
        target=target,
        train=train,
        test=test,
        process="CSTR",
        metadata=metadata,
        smoke=smoke,
    )
    geometry_rows_out = geometry_rows(
        bundle,
        channels=CSTR_CHANNELS,
        lags=CSTR_LAGS,
        horizon=CSTR_HORIZON,
        n_time=len(panel),
        dt_minutes=CSTR_DT,
        metadata=metadata,
        traces=traces,
    )
    support_rows = source_support_rows(
        bundle, train, test, n_time=len(panel), metadata=metadata
    )
    return {
        "scores": score_rows,
        "cells": cell_rows,
        "geometry": geometry_rows_out,
        "support": support_rows,
    }


def load_tep_segments(smoke: bool) -> list[TEPSegment]:
    faults = (1,) if smoke else TEP_FAULTS
    batches = tuple(range(1, 7)) if smoke else TEP_BATCHES
    segments: list[TEPSegment] = []
    for fault in faults:
        for batch in batches:
            path = TEP_DATA / f"mode1_{fault}_{batch}.xlsx"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_excel(path)
            required = ("Time (h)", *TEP_CHANNELS)
            missing = [name for name in required if name not in frame.columns]
            if missing or len(frame) != 3001:
                raise RuntimeError(f"invalid TEP file {path.name}: missing={missing}, rows={len(frame)}")
            minutes = frame["Time (h)"].to_numpy(dtype=float) * 60.0
            if not np.allclose(minutes, np.arange(3001), atol=1e-4):
                raise RuntimeError(f"non one-minute grid in {path.name}")
            values = frame.loc[:, list(TEP_CHANNELS)].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"non-finite TEP values in {path.name}")
            segments.append(TEPSegment(fault, batch, values))
    return segments


def load_tep_calibration() -> dict[tuple[int, int], tuple[float, bool]]:
    frame = pd.read_csv(TEP_CALIBRATION)
    selected = frame.loc[frame["severity_k"].isin(LEVELS)]
    expected = len(LEVELS) * len(TEP_CHANNELS)
    if len(selected) != expected:
        raise RuntimeError(f"unexpected TEP calibration geometry: {len(selected)} != {expected}")
    output: dict[tuple[int, int], tuple[float, bool]] = {}
    for row in selected.itertuples(index=False):
        key = (int(row.severity_k), int(row.channel_index))
        if key in output or str(row.channel) != TEP_CHANNELS[key[1]]:
            raise RuntimeError(f"invalid TEP calibration row {key}")
        output[key] = (float(row.epsilon), bool(row.calibration_feasible))
    return output


def init_tep_worker(
    segments: list[TEPSegment],
    calibration: dict[tuple[int, int], tuple[float, bool]],
    smoke: bool,
) -> None:
    global _TEP_SEGMENTS, _TEP_CALIBRATION_MAP, _TEP_SMOKE
    _TEP_SEGMENTS = segments
    _TEP_CALIBRATION_MAP = calibration
    _TEP_SMOKE = smoke


def active_tep_targets(
    smoke: bool, target_panel: str = "selected"
) -> tuple[tuple[str, str, int], ...]:
    try:
        targets = TEP_TARGET_PANELS[target_panel]
    except KeyError as exc:
        raise ValueError(f"unknown TEP target panel: {target_panel}") from exc
    # The selected-panel smoke test remains intentionally minimal. The full
    # 2x2 smoke test exercises every added target--horizon path.
    return targets[:1] if smoke and target_panel == "selected" else targets


def active_tep_folds(smoke: bool) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    return ((1, tuple(range(1, 5)), (5, 6)),) if smoke else TEP_FOLDS


def tep_source_stats(
    bundle: hv.VintageFeatureBundle, segment: TEPSegment
) -> list[dict[str, Any]]:
    all_rows = np.arange(len(bundle.origins), dtype=np.int64)
    n_features = bundle.final_features.shape[1]
    origin_commit_delta = bundle.origin_source_commit - bundle.origins[:, None]
    origin_event_delta = np.maximum(
        bundle.origin_source_left, bundle.origin_source_right
    ) - bundle.origins[:, None]
    output: list[dict[str, Any]] = []
    for view in ("FINAL", "ORIGIN"):
        unique = hv.unique_source_ids(bundle, all_rows, view=view, n_time=len(segment.values))
        output.append(
            {
                "fault": segment.fault,
                "batch": segment.batch,
                "trajectory": segment.trajectory,
                "view": view,
                "n_rows": int(len(all_rows)),
                "n_lineage_references": int(len(all_rows) * n_features * 2),
                "n_unique_sources": int(len(unique)),
                "n_post_origin_commit_violations": int(np.sum(origin_commit_delta > 0)),
                "max_origin_commit_minus_origin": int(np.max(origin_commit_delta)),
                "n_post_origin_event_violations": int(np.sum(origin_event_delta > 0)),
                "max_origin_event_minus_origin": int(np.max(origin_event_delta)),
            }
        )
    return output


def aggregate_tep_support(
    stats: pd.DataFrame,
    *,
    folds: Sequence[tuple[int, tuple[int, ...], tuple[int, ...]]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, train_batches, test_batches in folds:
        for view in ("FINAL", "ORIGIN"):
            fit = stats.loc[stats["batch"].isin(train_batches) & stats["view"].eq(view)]
            test = stats.loc[stats["batch"].isin(test_batches) & stats["view"].eq(view)]
            rows.append(
                {
                    **metadata,
                    "fold": fold,
                    "view": view,
                    "n_fit_rows": int(fit["n_rows"].sum()),
                    "n_test_rows": int(test["n_rows"].sum()),
                    "n_fit_lineage_references": int(fit["n_lineage_references"].sum()),
                    "n_test_lineage_references": int(test["n_lineage_references"].sum()),
                    "n_unique_fit_sources": int(fit["n_unique_sources"].sum()),
                    "n_unique_test_sources": int(test["n_unique_sources"].sum()),
                    "n_overlapping_sources": 0,
                    "test_source_overlap_share": 0.0,
                    "fit_source_reuse_ratio": safe_relative(
                        float(fit["n_lineage_references"].sum()),
                        float(fit["n_unique_sources"].sum()),
                    ),
                    "test_source_reuse_ratio": safe_relative(
                        float(test["n_lineage_references"].sum()),
                        float(test["n_unique_sources"].sum()),
                    ),
                    "n_post_origin_commit_violations": int(
                        fit["n_post_origin_commit_violations"].sum()
                        + test["n_post_origin_commit_violations"].sum()
                    ),
                    "max_origin_commit_minus_origin": int(
                        max(
                            fit["max_origin_commit_minus_origin"].max(),
                            test["max_origin_commit_minus_origin"].max(),
                        )
                    ),
                    "n_post_origin_event_violations": int(
                        fit["n_post_origin_event_violations"].sum()
                        + test["n_post_origin_event_violations"].sum()
                    ),
                    "max_origin_event_minus_origin": int(
                        max(
                            fit["max_origin_event_minus_origin"].max(),
                            test["max_origin_event_minus_origin"].max(),
                        )
                    ),
                    "overlap_reason": "DISJOINT_FAULT_BATCH_TRAJECTORIES",
                }
            )
    return rows


def run_tep_config(
    task: tuple[str, int] | tuple[str, int, str]
) -> dict[str, list[dict[str, Any]]]:
    operator, k, *panel_arg = task
    target_panel = panel_arg[0] if panel_arg else "selected"
    if _TEP_SEGMENTS is None or _TEP_CALIBRATION_MAP is None:
        raise RuntimeError("TEP worker was not initialized")
    segments = _TEP_SEGMENTS
    smoke = _TEP_SMOKE
    folds = active_tep_folds(smoke)
    all_scores: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_geometry: list[dict[str, Any]] = []
    all_support: list[dict[str, Any]] = []

    segment_traces: dict[int, list[hv.RetentionTrace]] = {}
    for segment in segments:
        parameters = None
        if operator == "SWINGING_DOOR":
            parameters = [_TEP_CALIBRATION_MAP[(k, channel)] for channel in range(len(TEP_CHANNELS))]
        segment_traces[segment.trajectory] = make_traces(
            segment.values, operator=operator, k=k, sda_parameters=parameters
        )

    for region, target_name, horizon in active_tep_targets(smoke, target_panel):
        target_channel = TEP_CHANNEL_INDEX[target_name]
        bundles: list[tuple[TEPSegment, hv.VintageFeatureBundle]] = []
        geometry_out: list[dict[str, Any]] = []
        source_stats: list[dict[str, Any]] = []
        for segment in segments:
            origins = np.arange(max(TEP_LAGS), len(segment.values) - horizon, dtype=np.int64)
            traces = segment_traces[segment.trajectory]
            bundle = hv.build_vintage_features(
                segment.values,
                traces,
                origins,
                TEP_LAGS,
                target_channel=target_channel,
                horizon=horizon,
            )
            bundles.append((segment, bundle))
            base_metadata = {
                "process": "TEP",
                "region": region,
                "target": target_name,
                "horizon_minutes": float(horizon),
                "seed": -1,
                "fold": -1,
                "fault": segment.fault,
                "batch": segment.batch,
                "trajectory": segment.trajectory,
                "operator": operator,
                "severity_k": k,
                "nominal_retention": 1.0 / k,
            }
            geometry_out.extend(
                geometry_rows(
                    bundle,
                    channels=TEP_CHANNELS,
                    lags=TEP_LAGS,
                    horizon=horizon,
                    n_time=len(segment.values),
                    dt_minutes=1.0,
                    metadata=base_metadata,
                    traces=traces,
                )
            )
            source_stats.extend(tep_source_stats(bundle, segment))

        final_features = np.concatenate(
            [bundle.final_features.astype(np.float32) for _, bundle in bundles]
        )
        origin_features = np.concatenate(
            [bundle.origin_features.astype(np.float32) for _, bundle in bundles]
        )
        final_current = np.concatenate([bundle.final_current for _, bundle in bundles])
        origin_current = np.concatenate([bundle.origin_current for _, bundle in bundles])
        target = np.concatenate(
            [segment.values[bundle.origins + horizon, target_channel] for segment, bundle in bundles]
        )
        faults = np.concatenate(
            [np.full(len(bundle.origins), segment.fault, dtype=np.int16) for segment, bundle in bundles]
        )
        batches = np.concatenate(
            [np.full(len(bundle.origins), segment.batch, dtype=np.int16) for segment, bundle in bundles]
        )

        for fold, train_batches, test_batches in folds:
            prior = np.flatnonzero(np.isin(batches, train_batches))
            train = even_subsample(prior)
            fitted_panels = {
                "FINAL": fit_panel(
                    final_features, target, train, process="TEP", smoke=smoke
                ),
                "ORIGIN": fit_panel(
                    origin_features, target, train, process="TEP", smoke=smoke
                ),
            }
            for fault in sorted(np.unique(faults)):
                test = np.flatnonzero(np.isin(batches, test_batches) & (faults == fault))
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
                scores, cells = evaluate_prediction_cell(
                    final_features=final_features,
                    origin_features=origin_features,
                    final_current=final_current,
                    origin_current=origin_current,
                    target=target,
                    train=train,
                    test=test,
                    process="TEP",
                    metadata=metadata,
                    smoke=smoke,
                    fitted_panels=fitted_panels,
                )
                all_scores.extend(scores)
                all_cells.extend(cells)

        support_metadata = {
            "process": "TEP",
            "region": region,
            "target": target_name,
            "horizon_minutes": float(horizon),
            "seed": -1,
            "fault": -1,
            "operator": operator,
            "severity_k": k,
            "nominal_retention": 1.0 / k,
        }
        all_support.extend(
            aggregate_tep_support(pd.DataFrame(source_stats), folds=folds, metadata=support_metadata)
        )
        all_geometry.extend(geometry_out)

    return {
        "scores": all_scores,
        "cells": all_cells,
        "geometry": all_geometry,
        "support": all_support,
    }


def run_self_checks() -> dict[str, Any]:
    rng = np.random.default_rng(20260903)
    arrays = [np.arange(21, dtype=float), np.sin(np.arange(37) / 3.0)]
    arrays.extend(rng.normal(size=length).cumsum() for length in (17, 64, 129))
    checked = 0
    for values in arrays:
        for epsilon in (0.0, 0.05, 0.5, 2.0):
            trace = hv.swinging_door_trace(values, epsilon)
            expected = hii.swinging_door_indices(values, epsilon)
            if not np.array_equal(trace.event_indices, expected):
                raise AssertionError("swinging-door retained indices differ")
            reconstructed = hv.final_linear(values, trace)
            reference = hii.reconstruct(values, expected, "LINEAR")
            if not np.allclose(reconstructed.values, reference, atol=1e-12, rtol=0.0):
                raise AssertionError("final linear reconstruction differs")
            checked += 1

    values = np.arange(9, dtype=float)[:, None]
    origins = np.arange(1, 8, dtype=np.int64)
    known = hv.build_vintage_features(
        values, [hv.periodic_trace(len(values), 2)], origins, (0,), target_channel=0, horizon=1
    )
    expected_future = np.array([True, False, True, False, True, False, True])
    observed_future = known.strict_input_availability > known.origins
    if not np.array_equal(observed_future, expected_future):
        raise AssertionError("periodic future-source known answer failed")
    if not np.all(known.origin_source_commit <= known.origins[:, None]):
        raise AssertionError("origin-vintage source commit exceeds origin")
    # Target cells at even indices are retained; odd target cells need the next event.
    target_expected = np.array([False, True, False, True, False, True, False])
    if not np.array_equal(known.target_final_availability > origins + 1, target_expected):
        raise AssertionError("target-vintage known answer failed")
    return {
        "randomized_sda_cases": checked,
        "sda_retained_index_equality": True,
        "final_linear_equality": True,
        "periodic_future_source_known_answer": True,
        "origin_commit_not_after_origin": True,
        "target_availability_separated": True,
    }


def append_result(target: dict[str, list[dict[str, Any]]], result: dict[str, list[dict[str, Any]]]) -> None:
    for key in target:
        target[key].extend(result[key])


def validate_outputs(
    scores: pd.DataFrame,
    cells: pd.DataFrame,
    geometry: pd.DataFrame,
    support: pd.DataFrame,
) -> dict[str, Any]:
    score_key = [
        "process", "region", "seed", "fold", "fault", "operator", "severity_k",
        "score", "quadrant", "model",
    ]
    cell_key = [
        "process", "region", "seed", "fold", "fault", "operator", "severity_k",
        "score", "model",
    ]
    geometry_key = [
        "process", "region", "seed", "fold", "fault", "batch", "operator", "severity_k", "channel",
    ]
    if scores.duplicated(score_key).any():
        raise RuntimeError("duplicate forecast score key")
    if cells.duplicated(cell_key).any():
        raise RuntimeError("duplicate forecast cell key")
    if geometry.duplicated(geometry_key).any():
        raise RuntimeError("duplicate geometry key")
    numeric_score = scores["loss"].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_score)) or np.any(numeric_score < 0):
        raise RuntimeError("non-finite or negative forecast loss")
    if not set(scores["quadrant"]) == set(QUADRANTS):
        raise RuntimeError("missing train/evaluation quadrant")
    if (support["n_overlapping_sources"] < 0).any():
        raise RuntimeError("invalid source overlap count")
    if bool((support["n_post_origin_commit_violations"] != 0).any()):
        raise RuntimeError("origin-vintage lineage contains a post-origin commit")
    if bool((support["n_post_origin_event_violations"] != 0).any()):
        raise RuntimeError("origin-vintage lineage contains a post-origin source event")
    return {
        "forecast_scores_rows": int(len(scores)),
        "forecast_cells_rows": int(len(cells)),
        "geometry_rows": int(len(geometry)),
        "source_support_rows": int(len(support)),
        "unique_score_keys": True,
        "unique_cell_keys": True,
        "unique_geometry_keys": True,
        "finite_nonnegative_losses": True,
        "four_quadrants_present": True,
        "origin_lineage_asof_feasible": True,
    }


def recurrent_effects(cells: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    primary = cells.loc[cells["score"].eq("MAE")].copy()
    primary["material"] = (
        primary["absolute_material_gap_1pct"].eq(1)
        | primary["practical_diagonal_winner_change_1pct"].eq(1)
    )
    records: list[dict[str, Any]] = []
    qualified_regions: list[str] = []
    for region, region_frame in primary.groupby("region"):
        unit = "seed" if region_frame["process"].iloc[0] == "CSTR" else "fold"
        group_keys = ["operator", "severity_k", "model"]
        if unit == "fold":
            # Require temporal recurrence within the same predeclared TEP fault family.
            group_keys.append("fault")
        best_record: dict[str, Any] | None = None
        for key, group in region_frame.groupby(group_keys):
            material_units = int(group.loc[group["material"], unit].nunique())
            optimistic_units = int(
                group.loc[group["optimistic_deployment_gap_1pct"].eq(1), unit].nunique()
            )
            record = {
                "region": region,
                "operator": key[0],
                "severity_k": int(key[1]),
                "model": key[2],
                "fault": int(key[3]) if unit == "fold" else -1,
                "recurrence_unit": unit,
                "n_material_units": material_units,
                "n_optimistic_units": optimistic_units,
                "median_deployment_gap": float(group["deployment_gap"].median()),
                "median_diagonal_gap": float(group["diagonal_gap"].median()),
                "practical_winner_changes": int(
                    group["practical_diagonal_winner_change_1pct"].sum()
                ),
            }
            records.append(record)
            if best_record is None or (
                record["n_material_units"], record["n_optimistic_units"]
            ) > (best_record["n_material_units"], best_record["n_optimistic_units"]):
                best_record = record
        if best_record is not None and best_record["n_material_units"] >= 2:
            qualified_regions.append(str(region))
    return records, sorted(qualified_regions)


def summarize(
    *,
    scores: pd.DataFrame,
    cells: pd.DataFrame,
    geometry: pd.DataFrame,
    support: pd.DataFrame,
    checks: dict[str, Any],
    validation: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    all_geometry = geometry.loc[geometry["channel"].eq("__ALL__")]
    region_geometry: list[dict[str, Any]] = []
    for region, group in all_geometry.groupby("region"):
        region_geometry.append(
            {
                "region": region,
                "median_origins_any_structural_future_share": float(
                    group["origins_any_structural_future_share"].median()
                ),
                "median_origins_any_active_future_share": float(
                    group["origins_any_active_future_share"].median()
                ),
                "median_strict_effective_horizon_minutes": float(
                    group["strict_effective_horizon_median_minutes"].median()
                ),
                "median_active_effective_horizon_minutes": float(
                    group["active_effective_horizon_median_minutes"].median()
                ),
                "max_strict_target_crossing_share": float(
                    group["strict_latest_at_or_after_target_share"].max()
                ),
                "max_active_target_crossing_share": float(
                    group["active_latest_at_or_after_target_share"].max()
                ),
            }
        )
    recurrent, qualified_regions = recurrent_effects(cells)
    learned_qualified_regions = sorted(
        {
            row["region"]
            for row in recurrent
            if row["model"] != "PERSISTENCE" and row["n_material_units"] >= 2
        }
    )
    periodic_learned_qualified_regions = sorted(
        {
            row["region"]
            for row in recurrent
            if row["operator"] == "PERIODIC"
            and row["model"] != "PERSISTENCE"
            and row["n_material_units"] >= 2
        }
    )
    geometry_regions = {
        row["region"]
        for row in region_geometry
        if row["median_origins_any_active_future_share"] >= 0.01
    }
    qualifying = sorted(set(qualified_regions) & geometry_regions)
    if len(qualifying) >= 2:
        decision = "GO_VINTAGE_METHOD"
    elif len(qualifying) == 1:
        decision = "BOUNDED_VINTAGE_SIGNAL"
    elif geometry_regions:
        decision = "GEOMETRY_ONLY"
    else:
        decision = "NO_VINTAGE_SIGNAL"

    primary = cells.loc[cells["score"].eq("MAE")]
    return {
        "specification": SPEC,
        "run_mode": "SMOKE" if smoke else "FULL",
        "claim_boundary": (
            "Explicit forward-operator source provenance and as-of availability; "
            "not vendor-specific API behavior or population prevalence."
        ),
        "self_checks": checks,
        "validation": validation,
        "counts": {
            "processes": sorted(scores["process"].unique().tolist()),
            "regions": sorted(scores["region"].unique().tolist()),
            "forecast_score_rows": int(len(scores)),
            "forecast_cell_rows": int(len(cells)),
            "geometry_rows": int(len(geometry)),
            "source_support_rows": int(len(support)),
        },
        "geometry_by_region": region_geometry,
        "forecast_primary_mae": {
            "material_cell_share": float(primary["absolute_material_gap_1pct"].mean()),
            "optimistic_deployment_cell_share": float(
                primary["optimistic_deployment_gap_1pct"].mean()
            ),
            "practical_diagonal_winner_changes": int(
                primary.drop_duplicates(
                    [
                        "process", "region", "seed", "fold", "fault", "operator",
                        "severity_k", "score",
                    ]
                )["practical_diagonal_winner_change_1pct"].sum()
            ),
            "practical_deployment_winner_changes": int(
                primary.drop_duplicates(
                    [
                        "process", "region", "seed", "fold", "fault", "operator",
                        "severity_k", "score",
                    ]
                )["practical_deployment_winner_change_1pct"].sum()
            ),
            "practical_training_repair_winner_changes": int(
                primary.drop_duplicates(
                    [
                        "process", "region", "seed", "fold", "fault", "operator",
                        "severity_k", "score",
                    ]
                )["practical_training_repair_winner_change_1pct"].sum()
            ),
            "median_deployment_gap": float(primary["deployment_gap"].median()),
            "deployment_gap_min": float(primary["deployment_gap"].min()),
            "deployment_gap_max": float(primary["deployment_gap"].max()),
        },
        "recurrent_effects": recurrent,
        "geometry_qualified_regions": sorted(geometry_regions),
        "effect_qualified_regions": qualified_regions,
        "learned_model_effect_qualified_regions": learned_qualified_regions,
        "periodic_learned_model_effect_qualified_regions": periodic_learned_qualified_regions,
        "jointly_qualified_regions": qualifying,
        "decision": decision,
        "decision_note": (
            "The automated route is a graded exploratory summary. Positive, negative, and heterogeneous "
            "cells must still be inspected before a paper-direction decision."
        ),
    }


def plot_horizon_erosion(geometry: pd.DataFrame, out_dir: Path) -> None:
    data = geometry.loc[geometry["channel"].eq("__ALL__")].copy()
    grouped = (
        data.groupby(["region", "operator", "severity_k"], as_index=False)
        .agg(
            any_active=("origins_any_active_future_share", "median"),
            effective=("active_effective_horizon_mean_minutes", "median"),
        )
    )
    regions = sorted(grouped["region"].unique())
    fig, axes = plt.subplots(len(regions), 2, figsize=(8.6, 2.7 * len(regions)), squeeze=False)
    markers = {"PERIODIC": "o", "SWINGING_DOOR": "s"}
    for row, region in enumerate(regions):
        for operator in OPERATORS:
            subset = grouped.loc[(grouped["region"] == region) & (grouped["operator"] == operator)]
            axes[row, 0].plot(
                subset["severity_k"], 100 * subset["any_active"], marker=markers[operator], label=operator
            )
            axes[row, 1].plot(
                subset["severity_k"], subset["effective"], marker=markers[operator], label=operator
            )
        axes[row, 0].set_ylabel("Origins with active\nfuture source (%)")
        axes[row, 1].set_ylabel("Active effective\nhorizon (min)")
        axes[row, 0].set_title(region, loc="left", fontsize=9)
        axes[row, 1].set_title(region, loc="left", fontsize=9)
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[-1, 0].set_xlabel("Retention severity k")
    axes[-1, 1].set_xlabel("Retention severity k")
    axes[0, 1].legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_erosion.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "horizon_erosion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vintage_gap(cells: pd.DataFrame, out_dir: Path) -> None:
    data = cells.loc[cells["score"].eq("MAE") & cells["model"].ne("PERSISTENCE")].copy()
    grouped = (
        data.groupby(["region", "operator", "severity_k", "model"], as_index=False)["deployment_gap"]
        .median()
    )
    regions = sorted(grouped["region"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(3.6 * len(regions), 3.6), squeeze=False)
    for axis, region in zip(axes[0], regions):
        subset = grouped.loc[grouped["region"].eq(region)]
        models = sorted(subset["model"].unique())
        colors = {model: f"C{index}" for index, model in enumerate(models)}
        for operator, marker, linestyle in (
            ("PERIODIC", "o", "-"),
            ("SWINGING_DOOR", "s", "--"),
        ):
            for model in models:
                part = subset.loc[
                    subset["operator"].eq(operator) & subset["model"].eq(model)
                ]
                axis.plot(
                    part["severity_k"],
                    100 * part["deployment_gap"],
                    marker=marker,
                    linestyle=linestyle,
                    color=colors[model],
                    label=f"{model} · {operator}",
                )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.axhline(1.0, color="0.5", linewidth=0.7, linestyle="--")
        axis.axhline(-1.0, color="0.5", linewidth=0.7, linestyle="--")
        axis.set_xlabel("Retention severity k")
        axis.set_ylabel("Median deployment gap (%)")
        axis.set_title(region, loc="left", fontsize=9)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=6.5, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "vintage_forecast_gap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "vintage_forecast_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    output: dict[str, list[dict[str, Any]]],
    *,
    out_dir: Path,
    checks: dict[str, Any],
    smoke: bool,
    target_panel: str = "selected",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(output["scores"])
    cells = pd.DataFrame(output["cells"])
    geometry = pd.DataFrame(output["geometry"])
    support = pd.DataFrame(output["support"])
    validation = validate_outputs(scores, cells, geometry, support)
    scores.to_csv(out_dir / "forecast_scores.csv", index=False)
    cells.to_csv(out_dir / "forecast_cells.csv", index=False)
    geometry.to_csv(out_dir / "geometry_cells.csv", index=False)
    support.to_csv(out_dir / "source_support.csv", index=False)
    summary = summarize(
        scores=scores,
        cells=cells,
        geometry=geometry,
        support=support,
        checks=checks,
        validation=validation,
        smoke=smoke,
    )
    summary["tep_target_panel"] = target_panel
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    plot_horizon_erosion(geometry, out_dir)
    plot_vintage_gap(cells, out_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run the bounded real-file smoke geometry")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tep-panel",
        choices=tuple(TEP_TARGET_PANELS),
        default="selected",
        help="TEP target--horizon panel; full2x2 requires an explicit --out",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = bool(args.smoke)
    if args.tep_panel != "selected" and args.out is None:
        raise SystemExit("--out is required when --tep-panel is not selected")
    out_dir = args.out or (SMOKE_OUT if smoke else DEFAULT_OUT)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log(
        "starting vintage replay kill test; "
        f"smoke={smoke} tep_panel={args.tep_panel}"
    )
    checks = run_self_checks()
    log(f"self-checks passed: {checks}")

    output: dict[str, list[dict[str, Any]]] = {
        "scores": [], "cells": [], "geometry": [], "support": []
    }
    cstr_seeds = (0,) if smoke else tuple(range(8))
    configs = (("PERIODIC", 2), ("SWINGING_DOOR", 2)) if smoke else tuple(
        (operator, k) for operator in OPERATORS for k in LEVELS
    )
    cstr_tasks = [(seed, operator, k, smoke) for seed in cstr_seeds for operator, k in configs]
    cstr_workers = choose_workers(args.workers, len(cstr_tasks))
    log(f"CSTR tasks={len(cstr_tasks)} workers={cstr_workers}")
    with ProcessPoolExecutor(max_workers=cstr_workers) as executor:
        futures = {executor.submit(run_cstr_task, task): task for task in cstr_tasks}
        for count, future in enumerate(as_completed(futures), start=1):
            append_result(output, future.result())
            if count == len(futures) or count % max(1, len(futures) // 4) == 0:
                log(f"CSTR progress {count}/{len(futures)}")

    segments = load_tep_segments(smoke)
    calibration = load_tep_calibration()
    tep_workers = choose_workers(args.workers, len(configs))
    log(f"TEP segments={len(segments)} configs={len(configs)} workers={tep_workers}")
    with ProcessPoolExecutor(
        max_workers=tep_workers,
        initializer=init_tep_worker,
        initargs=(segments, calibration, smoke),
    ) as executor:
        tep_tasks = [(*task, args.tep_panel) for task in configs]
        futures = {executor.submit(run_tep_config, task): task for task in tep_tasks}
        for count, future in enumerate(as_completed(futures), start=1):
            append_result(output, future.result())
            log(f"TEP progress {count}/{len(futures)} config={futures[future]}")

    summary = write_outputs(
        output,
        out_dir=out_dir,
        checks=checks,
        smoke=smoke,
        target_panel=args.tep_panel,
    )
    log(f"completed: decision={summary['decision']} outputs={out_dir.relative_to(ROOT) if out_dir.is_absolute() else out_dir}")
    print(json.dumps({"decision": summary["decision"], "out_dir": str(out_dir.relative_to(ROOT) if out_dir.is_absolute() else out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
