#!/usr/bin/env python3
"""Build bounded manuscript-strengthening evidence from frozen lineage runs.

This script does not refit forecasting models.  It (i) re-aggregates the
matched event-time versus commit-time scores by model, (ii) recomputes
feature-scope feasibility from the same public TEP trajectories and frozen
Swinging Door tolerances, and (iii) creates one deterministic worked example
for explaining the two-clock mechanism.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
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


PAPER = Path(__file__).resolve().parents[1]
WORKSPACE = PAPER
OSC_ROOT = PAPER
OSC_SRC = OSC_ROOT / "src"
if str(OSC_SRC) not in sys.path:
    sys.path.insert(0, str(OSC_SRC))

import historian_vintage as hv  # noqa: E402


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import analysis script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s101 = load_script(
    OSC_ROOT / "scripts/101_historian_vintage_replay_kill_test.py",
    "historian_vintage_stage1_source",
)

LINEAGE_DIR = (
    OSC_ROOT
    / "artifacts/historian_reconstruction_lineage_tep_full_panel_breadth"
)
TEP_DATA = PAPER / "data/tep"
DEFAULT_OUTPUT = PAPER / "artifacts/stage1_strengthening"
DEFAULT_FIGURES = PAPER / "figures/generated"
DEFAULT_TABLES = PAPER / "tables/generated"

REGIONS = (
    ("PRESSURE_Q6", "XMEAS-7", 6),
    ("PRESSURE_Q16", "XMEAS-7", 16),
    ("TEMPERATURE_Q6", "XMEAS-9", 6),
    ("TEMPERATURE_Q16", "XMEAS-9", 16),
)
REGION_LABELS = {
    "CSTR_OUTPUT_Q1": "CSTR output, 1 min",
    "PRESSURE_Q6": "TEP pressure, 6 min",
    "PRESSURE_Q16": "TEP pressure, 16 min",
    "TEMPERATURE_Q6": "TEP temperature, 6 min",
    "TEMPERATURE_Q16": "TEP temperature, 16 min",
}
MODEL_LABELS = {
    "PERSISTENCE": "Persistence",
    "RIDGE_AR": "Ridge",
    "RIDGE_ARX": "Ridge",
    "SHALLOW_GBR": "Shallow tree",
    "SHALLOW_HGB": "Shallow tree",
}
MODEL_ORDER = ("Persistence", "Ridge", "Shallow tree")
SCOPE_ORDER = (
    "target current",
    "target history",
    "all-channel current",
    "full lag window",
)
SEVERITIES = (2, 3, 5, 8)
TEST_BATCHES = (5, 6, 7, 8, 9, 10)


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def summarize_model_clock_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-aggregate frozen matched-clock scores without fitting models."""

    frame = pd.read_csv(LINEAGE_DIR / "view_comparison_cells.csv")
    require_columns(
        frame,
        {
            "region",
            "operator",
            "severity_k",
            "comparison",
            "score",
            "model",
            "loss_left_train_left_eval",
            "loss_left_train_right_eval",
            "evaluation_view_gap",
        },
        "matched-clock comparison",
    )
    selected = frame.loc[
        frame["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & frame["operator"].eq("SWINGING_DOOR")
        & frame["score"].eq("MAE")
    ].copy()
    selected["model_label"] = selected["model"].map(MODEL_LABELS)
    if selected["model_label"].isna().any():
        unknown = sorted(selected.loc[selected["model_label"].isna(), "model"].unique())
        raise RuntimeError(f"unmapped models: {unknown}")
    selected["paired_absolute_mae_change"] = (
        selected["loss_left_train_right_eval"]
        - selected["loss_left_train_left_eval"]
    )

    by_level = (
        selected.groupby(
            ["region", "model_label", "severity_k"], sort=False
        )
        .agg(
            n_units=("evaluation_view_gap", "size"),
            median_relative_change=("evaluation_view_gap", "median"),
            q25_relative_change=("evaluation_view_gap", lambda x: x.quantile(0.25)),
            q75_relative_change=("evaluation_view_gap", lambda x: x.quantile(0.75)),
            positive_units=("evaluation_view_gap", lambda x: int((x > 0).sum())),
            median_absolute_mae_change=("paired_absolute_mae_change", "median"),
            q25_absolute_mae_change=(
                "paired_absolute_mae_change",
                lambda x: x.quantile(0.25),
            ),
            q75_absolute_mae_change=(
                "paired_absolute_mae_change",
                lambda x: x.quantile(0.75),
            ),
        )
        .reset_index()
    )
    across_levels = (
        selected.groupby(["region", "model_label"], sort=False)
        .agg(
            n_cells=("evaluation_view_gap", "size"),
            median_relative_change=("evaluation_view_gap", "median"),
            q25_relative_change=("evaluation_view_gap", lambda x: x.quantile(0.25)),
            q75_relative_change=("evaluation_view_gap", lambda x: x.quantile(0.75)),
            positive_cells=("evaluation_view_gap", lambda x: int((x > 0).sum())),
            median_absolute_mae_change=("paired_absolute_mae_change", "median"),
        )
        .reset_index()
    )
    return by_level, across_levels


def fold_for_batch(batch: int) -> int:
    if batch in (5, 6):
        return 1
    if batch in (7, 8):
        return 2
    if batch in (9, 10):
        return 3
    raise ValueError(f"batch {batch} is not a held-out batch")


def load_tep_values(fault: int, batch: int) -> np.ndarray:
    path = TEP_DATA / f"mode1_{fault}_{batch}.xlsx"
    frame = pd.read_excel(path, usecols=list(s101.TEP_CHANNELS))
    values = frame.loc[:, list(s101.TEP_CHANNELS)].to_numpy(dtype=float)
    if values.shape[1] != len(s101.TEP_CHANNELS) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid TEP panel: {path}")
    return values


def scope_indices(target_channel: int, n_channels: int, n_lags: int) -> dict[str, np.ndarray]:
    lag_zero = 0
    return {
        "target current": np.asarray([target_channel * n_lags + lag_zero]),
        "target history": np.arange(
            target_channel * n_lags, (target_channel + 1) * n_lags
        ),
        "all-channel current": np.asarray(
            [channel * n_lags + lag_zero for channel in range(n_channels)]
        ),
        "full lag window": np.arange(n_channels * n_lags),
    }


def append_scope_arrays(
    store: dict[tuple[Any, ...], dict[str, list[np.ndarray]]],
    *,
    region: str,
    severity: int,
    fold: int,
    fault: int,
    scope: str,
    bundle: hv.VintageFeatureBundle,
    indices: np.ndarray,
) -> None:
    origins = bundle.origins
    availability = bundle.final_feature_availability[:, indices]
    active = bundle.active_future_mask[:, indices]
    structural_fail = np.any(availability > origins[:, None], axis=1)
    active_fail = np.any(active, axis=1)
    strict_erosion = np.maximum(
        0, np.max(availability, axis=1) - origins
    ).astype(float)
    active_availability = np.max(
        np.where(active, availability, origins[:, None]), axis=1
    )
    active_erosion = np.maximum(0, active_availability - origins).astype(float)
    key = (region, severity, fold, fault, scope)
    payload = store[key]
    payload["structural_fail"].append(structural_fail.astype(float))
    payload["active_fail"].append(active_fail.astype(float))
    payload["strict_erosion"].append(strict_erosion)
    payload["active_erosion"].append(active_erosion)


def fixed_origin_replay(
    values: np.ndarray,
    trace: hv.RetentionTrace,
    reconstruction: hv.FinalReconstruction,
    times: np.ndarray,
    origin: int,
) -> np.ndarray:
    """Apply the declared committed-prefix replay at one fixed origin."""

    event_times = np.asarray(times, dtype=np.int64)
    origins = np.full_like(event_times, int(origin))
    final_availability = reconstruction.availability[event_times]
    ready = final_availability <= origins
    event_bound = np.searchsorted(trace.event_indices, event_times, side="right") - 1
    commit_bound = np.searchsorted(trace.commit_indices, origins, side="right") - 1
    hold_position = np.minimum(event_bound, commit_bound)
    if np.any(hold_position < 0):
        raise RuntimeError("worked example has no admissible prefix source")
    hold_source = trace.event_indices[hold_position]
    return np.where(ready, reconstruction.values[event_times], values[hold_source])


def build_worked_example(
    values: np.ndarray,
    traces: Sequence[hv.RetentionTrace],
    bundle: hv.VintageFeatureBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose a deterministic, temporally central commit-delay example."""

    channel_name = "XMEAS-7"
    channel = s101.TEP_CHANNEL_INDEX[channel_name]
    n_lags = len(s101.TEP_LAGS)
    feature_column = channel * n_lags  # target current, lag 0
    origins = bundle.origins
    interior = (origins >= int(0.1 * len(values))) & (
        origins <= int(0.9 * len(values))
    )
    candidate = (
        interior
        & (bundle.final_source_right[:, feature_column] <= origins)
        & (bundle.final_feature_availability[:, feature_column] > origins)
        & bundle.active_future_mask[:, feature_column]
    )
    candidate_rows = np.flatnonzero(candidate)
    if len(candidate_rows) == 0:
        raise RuntimeError("no interior event-before-commit example was found")
    midpoint = 0.5 * (origins[0] + origins[-1])
    ordering = np.lexsort(
        (origins[candidate_rows], np.abs(origins[candidate_rows] - midpoint))
    )
    row = int(candidate_rows[ordering[0]])
    origin = int(origins[row])
    feature_time = origin
    trace = traces[channel]
    reconstruction = hv.final_linear(values[:, channel], trace)
    event_trace = hv.event_time_only_trace(trace, len(values))
    event_reconstruction = hv.final_linear(values[:, channel], event_trace)

    event_position = int(
        np.searchsorted(trace.event_indices, feature_time, side="right") - 1
    )
    previous_position = max(0, event_position - 1)
    next_position = min(len(trace.event_indices) - 1, event_position + 1)
    previous_event = int(trace.event_indices[previous_position])
    selected_event = int(bundle.final_source_right[row, feature_column])
    selected_commit = int(bundle.final_feature_availability[row, feature_column])
    next_event = int(trace.event_indices[next_position])
    start = max(0, min(previous_event - 5, origin - 24))
    stop = min(len(values) - 1, max(next_event + 5, origin + 12))
    times = np.arange(start, stop + 1, dtype=np.int64)
    visible = times <= origin
    event_replay = np.full(len(times), np.nan)
    commit_replay = np.full(len(times), np.nan)
    event_replay[visible] = fixed_origin_replay(
        values[:, channel],
        event_trace,
        event_reconstruction,
        times[visible],
        origin,
    )
    commit_replay[visible] = fixed_origin_replay(
        values[:, channel], trace, reconstruction, times[visible], origin
    )

    retained_lookup = {
        int(event): int(commit)
        for event, commit in zip(trace.event_indices, trace.commit_indices)
    }
    frame = pd.DataFrame(
        {
            "time_index": times,
            "native_value": values[times, channel],
            "final_reconstruction": reconstruction.values[times],
            "event_time_replay_at_origin": event_replay,
            "commit_time_replay_at_origin": commit_replay,
            "retained_event": [int(time in retained_lookup) for time in times],
            "retained_commit_index": [retained_lookup.get(int(time), np.nan) for time in times],
        }
    )
    metadata = {
        "selection_rule": (
            "Fixed TEP pressure/6-min, fault 1, batch 5, Swinging Door k=5; "
            "among interior lag-0 XMEAS-7 cells whose event existed but whose "
            "retention decision was not committed, choose the origin nearest "
            "the trajectory midpoint, breaking ties toward the earlier origin."
        ),
        "region": "PRESSURE_Q6",
        "fault": 1,
        "batch": 5,
        "severity_k": 5,
        "channel": channel_name,
        "origin": origin,
        "feature_time": feature_time,
        "previous_committed_event": previous_event,
        "selected_source_event": selected_event,
        "selected_source_commit": selected_commit,
        "next_retained_event": next_event,
        "event_time_value": float(event_replay[times == origin][0]),
        "commit_time_value": float(commit_replay[times == origin][0]),
        "absolute_value_difference": float(
            event_replay[times == origin][0] - commit_replay[times == origin][0]
        ),
    }
    return frame, metadata


def summarize_feature_scopes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    calibration = s101.load_tep_calibration()
    store: dict[tuple[Any, ...], dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    worked_frame: pd.DataFrame | None = None
    worked_metadata: dict[str, Any] | None = None

    for fault in s101.TEP_FAULTS:
        for batch in TEST_BATCHES:
            values = load_tep_values(fault, batch)
            fold = fold_for_batch(batch)
            for severity in SEVERITIES:
                parameters = [
                    calibration[(severity, channel)]
                    for channel in range(len(s101.TEP_CHANNELS))
                ]
                traces = s101.make_traces(
                    values,
                    operator="SWINGING_DOOR",
                    k=severity,
                    sda_parameters=parameters,
                )
                for horizon in (6, 16):
                    origins = np.arange(
                        max(s101.TEP_LAGS), len(values) - horizon, dtype=np.int64
                    )
                    bundle = hv.build_vintage_features(
                        values,
                        traces,
                        origins,
                        s101.TEP_LAGS,
                        target_channel=s101.TEP_CHANNEL_INDEX["XMEAS-7"],
                        horizon=horizon,
                    )
                    for region, target_name, region_horizon in REGIONS:
                        if region_horizon != horizon:
                            continue
                        target_channel = s101.TEP_CHANNEL_INDEX[target_name]
                        scopes = scope_indices(
                            target_channel,
                            len(s101.TEP_CHANNELS),
                            len(s101.TEP_LAGS),
                        )
                        for scope, indices in scopes.items():
                            append_scope_arrays(
                                store,
                                region=region,
                                severity=severity,
                                fold=fold,
                                fault=fault,
                                scope=scope,
                                bundle=bundle,
                                indices=indices,
                            )
                    if (
                        fault == 1
                        and batch == 5
                        and severity == 5
                        and horizon == 6
                    ):
                        worked_frame, worked_metadata = build_worked_example(
                            values, traces, bundle
                        )

    unit_rows: list[dict[str, Any]] = []
    for (region, severity, fold, fault, scope), values in store.items():
        structural = np.concatenate(values["structural_fail"])
        active = np.concatenate(values["active_fail"])
        strict_erosion = np.concatenate(values["strict_erosion"])
        active_erosion = np.concatenate(values["active_erosion"])
        unit_rows.append(
            {
                "region": region,
                "severity_k": severity,
                "fold": fold,
                "fault": fault,
                "feature_scope": scope,
                "n_origins": len(structural),
                "strict_failure_share": float(np.mean(structural)),
                "active_failure_share": float(np.mean(active)),
                "median_strict_erosion_minutes": float(np.median(strict_erosion)),
                "median_active_erosion_minutes": float(np.median(active_erosion)),
            }
        )
    units = pd.DataFrame(unit_rows)
    summary = (
        units.groupby(["region", "severity_k", "feature_scope"], sort=False)
        .agg(
            n_units=("strict_failure_share", "size"),
            median_strict_failure_share=("strict_failure_share", "median"),
            q25_strict_failure_share=(
                "strict_failure_share",
                lambda x: x.quantile(0.25),
            ),
            q75_strict_failure_share=(
                "strict_failure_share",
                lambda x: x.quantile(0.75),
            ),
            median_active_failure_share=("active_failure_share", "median"),
            median_strict_erosion_minutes=(
                "median_strict_erosion_minutes",
                "median",
            ),
            median_active_erosion_minutes=(
                "median_active_erosion_minutes",
                "median",
            ),
        )
        .reset_index()
    )
    if worked_frame is None or worked_metadata is None:
        raise RuntimeError("worked example was not generated")
    return units, summary, worked_frame, worked_metadata


def plot_worked_example(
    frame: pd.DataFrame, metadata: dict[str, Any], output: Path
) -> None:
    configure_plotting()
    fig, axis = plt.subplots(figsize=(4.25, 2.75))
    t = frame["time_index"].to_numpy(float)
    axis.plot(t, frame["native_value"], color="0.78", linewidth=1.0, label="Native samples")
    axis.plot(
        t,
        frame["final_reconstruction"],
        color="0.15",
        linewidth=1.35,
        label="Completed linear reconstruction",
    )
    axis.plot(
        t,
        frame["event_time_replay_at_origin"],
        color="0.35",
        linewidth=1.25,
        linestyle="--",
        label="Event-time replay at origin",
    )
    axis.plot(
        t,
        frame["commit_time_replay_at_origin"],
        color="0.05",
        linewidth=1.45,
        drawstyle="steps-post",
        linestyle=":",
        label="Availability-aware replay at origin",
    )
    retained = frame.loc[frame["retained_event"].eq(1)].copy()
    admissible = retained["retained_commit_index"] <= metadata["origin"]
    axis.scatter(
        retained.loc[admissible, "time_index"],
        retained.loc[admissible, "native_value"],
        s=22,
        color="black",
        zorder=5,
        label="Retained and committed",
    )
    axis.scatter(
        retained.loc[~admissible, "time_index"],
        retained.loc[~admissible, "native_value"],
        s=26,
        facecolors="white",
        edgecolors="black",
        linewidth=0.9,
        zorder=5,
        label="Retained later",
    )
    origin = int(metadata["origin"])
    commit = int(metadata["selected_source_commit"])
    axis.axvline(origin, color="0.20", linewidth=0.9)
    axis.annotate(
        "origin",
        xy=(origin, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -3),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.3,
    )
    axis.annotate(
        "available one minute later",
        xy=(commit, frame.loc[frame["time_index"].eq(origin), "native_value"].iloc[0]),
        xytext=(commit + 3, axis.get_ylim()[1]),
        arrowprops={"arrowstyle": "->", "color": "0.25", "lw": 0.8},
        ha="left",
        va="top",
        fontsize=7.8,
    )
    axis.set_xlabel("Trajectory minute")
    axis.set_ylabel("XMEAS-7")
    axis.grid(color="0.91", linewidth=0.5)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    order = [0, 1, 2, 3, 4, 5]
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        fontsize=7.2,
    )
    fig.tight_layout()
    save_figure(fig, output / "lineage_worked_example")


def write_model_table(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Study target & Model & $N$ & Median [IQR] (\%) & Positive & $\Delta$MAE \\",
        r"\midrule",
    ]
    for region in REGION_LABELS:
        group = summary.loc[summary["region"].eq(region)].set_index("model_label")
        for index, model in enumerate(MODEL_ORDER):
            record = group.loc[model]
            label = REGION_LABELS[region] if index == 0 else ""
            rows.append(
                f"{label} & {model} & {int(record.n_cells)} & "
                f"{100*record.median_relative_change:+.2f} "
                f"[{100*record.q25_relative_change:+.2f}, "
                f"{100*record.q75_relative_change:+.2f}] & "
                f"{int(record.positive_cells)}/{int(record.n_cells)} & "
                f"{record.median_absolute_mae_change:+.4g} \\\\" 
            )
        if region != tuple(REGION_LABELS)[-1]:
            rows.append(r"\addlinespace")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_scope_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[
        summary["feature_scope"].isin(["target current", "full lag window"])
        & summary["severity_k"].isin([2, 8])
    ].copy()
    rows = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Task & Feature scope & Fail $k=2$ & Fail $k=8$ & Erosion $k=2$ & Erosion $k=8$ \\",
        r"\midrule",
    ]
    for region, _, _ in REGIONS:
        group = selected.loc[selected["region"].eq(region)]
        for index, scope in enumerate(("target current", "full lag window")):
            scope_rows = group.loc[group["feature_scope"].eq(scope)].set_index("severity_k")
            label = REGION_LABELS[region] if index == 0 else ""
            rows.append(
                f"{label} & {scope.title()} & "
                f"{100*scope_rows.loc[2, 'median_strict_failure_share']:.1f} & "
                f"{100*scope_rows.loc[8, 'median_strict_failure_share']:.1f} & "
                f"{scope_rows.loc[2, 'median_strict_erosion_minutes']:.1f} & "
                f"{scope_rows.loc[8, 'median_strict_erosion_minutes']:.1f} \\\\" 
            )
        if region != REGIONS[-1][0]:
            rows.append(r"\addlinespace")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate(
    model_level: pd.DataFrame,
    model_summary: pd.DataFrame,
    scope_units: pd.DataFrame,
    scope_summary: pd.DataFrame,
    worked_metadata: dict[str, Any],
) -> dict[str, Any]:
    expected_model_level = len(REGION_LABELS) * len(MODEL_ORDER) * len(SEVERITIES)
    expected_model_summary = len(REGION_LABELS) * len(MODEL_ORDER)
    expected_scope_units = len(REGIONS) * len(SEVERITIES) * 3 * 6 * len(SCOPE_ORDER)
    expected_scope_summary = len(REGIONS) * len(SEVERITIES) * len(SCOPE_ORDER)
    if len(model_level) != expected_model_level:
        raise AssertionError((len(model_level), expected_model_level))
    if len(model_summary) != expected_model_summary:
        raise AssertionError((len(model_summary), expected_model_summary))
    if len(scope_units) != expected_scope_units:
        raise AssertionError((len(scope_units), expected_scope_units))
    if len(scope_summary) != expected_scope_summary:
        raise AssertionError((len(scope_summary), expected_scope_summary))
    if not np.isfinite(
        model_level.select_dtypes(include=[np.number]).to_numpy(float)
    ).all():
        raise AssertionError("model summary contains non-finite values")
    if not np.isfinite(
        scope_summary.select_dtypes(include=[np.number]).to_numpy(float)
    ).all():
        raise AssertionError("scope summary contains non-finite values")
    if worked_metadata["selected_source_commit"] <= worked_metadata["origin"]:
        raise AssertionError("worked example does not separate event and commit time")
    if abs(worked_metadata["absolute_value_difference"]) <= 1e-12:
        raise AssertionError("worked example has no active value difference")

    target_current = scope_summary.loc[
        scope_summary["feature_scope"].eq("target current")
    ]
    return {
        "model_level_rows": len(model_level),
        "model_summary_rows": len(model_summary),
        "scope_unit_rows": len(scope_units),
        "scope_summary_rows": len(scope_summary),
        "target_current_min_median_failure_share": float(
            target_current["median_strict_failure_share"].min()
        ),
        "worked_example": worked_metadata,
    }


def run(output: Path, figures: Path, tables: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    model_level, model_summary = summarize_model_clock_effects()
    scope_units, scope_summary, worked_frame, worked_metadata = (
        summarize_feature_scopes()
    )
    checks = validate(
        model_level, model_summary, scope_units, scope_summary, worked_metadata
    )

    model_level.to_csv(output / "model_clock_sensitivity_by_level.csv", index=False)
    model_summary.to_csv(output / "model_clock_sensitivity_summary.csv", index=False)
    scope_units.to_csv(output / "feature_scope_units.csv", index=False)
    scope_summary.to_csv(output / "feature_scope_summary.csv", index=False)
    worked_frame.to_csv(output / "worked_example_trace.csv", index=False)
    (output / "worked_example_metadata.json").write_text(
        json.dumps(worked_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_model_table(model_summary, tables / "lineage_model_sensitivity.tex")
    write_scope_table(scope_summary, tables / "lineage_feature_scope.tex")
    plot_worked_example(worked_frame, worked_metadata, figures)
    (output / "checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args.output_dir, args.figures_dir, args.tables_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
