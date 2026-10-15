#!/usr/bin/env python3
"""Verify the isolated TEP full-panel breadth extension.

This check does not fit models. It confirms that the selected-panel rows are
scientifically identical to the frozen reference and that the extension has
the declared 2 x 2 TEP geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_PARENT = ROOT / "artifacts" / "historian_vintage_replay_kill_test"
DEFAULT_FULL_PARENT = ROOT / "artifacts" / "historian_vintage_replay_tep_full_panel_breadth"
DEFAULT_REFERENCE_LINEAGE = (
    ROOT / "artifacts" / "historian_reconstruction_lineage_differentiation"
)
DEFAULT_FULL_LINEAGE = (
    ROOT / "artifacts" / "historian_reconstruction_lineage_tep_full_panel_breadth"
)

TEP_COMBINATIONS = {
    ("PRESSURE_Q6", "XMEAS-7", 6.0),
    ("PRESSURE_Q16", "XMEAS-7", 16.0),
    ("TEMPERATURE_Q6", "XMEAS-9", 6.0),
    ("TEMPERATURE_Q16", "XMEAS-9", 16.0),
}
OPERATORS = {"PERIODIC", "SWINGING_DOOR"}
SEVERITIES = {2, 3, 5, 8}
FOLDS = {1, 2, 3}
FAULTS = {1, 4, 5, 11, 14, 15}
MODELS = {"PERSISTENCE", "RIDGE_ARX", "SHALLOW_HGB"}
SCORES = {"MAE", "RMSE"}
COMPARISONS = {"FINAL_VS_EVENT_COMMIT", "EVENT_TIME_VS_EVENT_COMMIT"}


def read_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compare_reference(
    reference: pd.DataFrame,
    extended: pd.DataFrame,
    *,
    key_columns: list[str],
    tolerance: float,
) -> dict[str, Any]:
    if reference.duplicated(key_columns).any() or extended.duplicated(key_columns).any():
        raise AssertionError("comparison keys are not unique")
    if list(reference.columns) != list(extended.columns):
        raise AssertionError("reference and extended schemas differ")

    merged = reference.merge(
        extended,
        on=key_columns,
        how="left",
        suffixes=("_reference", "_extended"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        missing = int(merged["_merge"].ne("both").sum())
        raise AssertionError(f"extended result is missing {missing} reference rows")

    numeric_columns = [
        column
        for column in reference.columns
        if column not in key_columns and pd.api.types.is_numeric_dtype(reference[column])
    ]
    text_columns = [
        column
        for column in reference.columns
        if column not in key_columns and column not in numeric_columns
    ]
    max_abs_delta = 0.0
    for column in numeric_columns:
        left = merged[f"{column}_reference"].to_numpy(dtype=float)
        right = merged[f"{column}_extended"].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(left), np.isnan(right)):
            raise AssertionError(f"NaN pattern changed for {column}")
        finite = np.isfinite(left) & np.isfinite(right)
        if finite.any():
            delta = float(np.max(np.abs(left[finite] - right[finite])))
            max_abs_delta = max(max_abs_delta, delta)
    text_mismatches = 0
    for column in text_columns:
        left = merged[f"{column}_reference"].fillna("<NA>").astype(str)
        right = merged[f"{column}_extended"].fillna("<NA>").astype(str)
        text_mismatches += int(left.ne(right).sum())
    if max_abs_delta > tolerance or text_mismatches:
        raise AssertionError(
            "reference regression mismatch: "
            f"max_abs_delta={max_abs_delta}, text_mismatches={text_mismatches}"
        )
    return {
        "reference_rows": int(len(reference)),
        "matched_rows": int(len(merged)),
        "max_abs_numeric_delta": max_abs_delta,
        "text_mismatches": text_mismatches,
        "tolerance": tolerance,
    }


def verify_geometry(cells: pd.DataFrame, lineage: pd.DataFrame) -> dict[str, Any]:
    tep = cells.loc[cells["process"].eq("TEP")].copy()
    combinations = set(
        tep[["region", "target", "horizon_minutes"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    checks = {
        "target_horizon_combinations": combinations == TEP_COMBINATIONS,
        "operators": set(tep["operator"].unique()) == OPERATORS,
        "severities": set(tep["severity_k"].unique()) == SEVERITIES,
        "folds": set(tep["fold"].unique()) == FOLDS,
        "faults": set(tep["fault"].unique()) == FAULTS,
        "models": set(tep["model"].unique()) == MODELS,
        "scores": set(tep["score"].unique()) == SCORES,
        "lineage_comparisons": set(lineage["comparison"].unique()) == COMPARISONS,
        "parent_rows": len(cells) == 3840,
        "lineage_rows": len(lineage) == 7680,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"full-panel geometry checks failed: {failed}")
    return {
        **checks,
        "tep_combinations": [
            {"region": region, "target": target, "horizon_minutes": horizon}
            for region, target, horizon in sorted(combinations)
        ],
        "parent_rows_observed": int(len(cells)),
        "lineage_rows_observed": int(len(lineage)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-parent", type=Path, default=DEFAULT_REFERENCE_PARENT)
    parser.add_argument("--full-parent", type=Path, default=DEFAULT_FULL_PARENT)
    parser.add_argument("--reference-lineage", type=Path, default=DEFAULT_REFERENCE_LINEAGE)
    parser.add_argument("--full-lineage", type=Path, default=DEFAULT_FULL_LINEAGE)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_parent = read_csv(args.reference_parent, "forecast_cells.csv")
    full_parent = read_csv(args.full_parent, "forecast_cells.csv")
    reference_lineage = read_csv(args.reference_lineage, "view_comparison_cells.csv")
    full_lineage = read_csv(args.full_lineage, "view_comparison_cells.csv")

    parent_keys = [
        "process", "region", "target", "horizon_minutes", "seed", "fold",
        "fault", "operator", "severity_k", "score", "model",
    ]
    lineage_keys = ["comparison", *parent_keys]
    summary_path = args.full_lineage / "summary.json"
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    internal_checks = summary["checks"]
    if internal_checks["periodic_event_commit_score_max_abs_delta"] > args.tolerance:
        raise AssertionError("periodic event/commit control is not exact")
    if internal_checks["max_grid_time_minus_origin"] > 0:
        raise AssertionError("feature grid time exceeds origin")
    if internal_checks["max_commit_source_minus_origin"] > 0:
        raise AssertionError("origin replay uses a post-origin commit")

    result = {
        "status": "PASS",
        "interpretation": "post-selection breadth analysis; not independent confirmation",
        "parent_regression": compare_reference(
            reference_parent,
            full_parent,
            key_columns=parent_keys,
            tolerance=args.tolerance,
        ),
        "lineage_regression": compare_reference(
            reference_lineage,
            full_lineage,
            key_columns=lineage_keys,
            tolerance=args.tolerance,
        ),
        "full_panel_geometry": verify_geometry(full_parent, full_lineage),
        "internal_lineage_checks": {
            "periodic_event_commit_score_max_abs_delta": internal_checks[
                "periodic_event_commit_score_max_abs_delta"
            ],
            "commit_replay_score_max_abs_delta": internal_checks[
                "commit_replay_score_max_abs_delta"
            ],
            "event_commit_retained_events_identical": internal_checks[
                "event_commit_retained_events_identical"
            ],
            "event_commit_final_export_max_abs_delta": internal_checks[
                "event_commit_final_export_max_abs_delta"
            ],
            "max_grid_time_minus_origin": internal_checks["max_grid_time_minus_origin"],
            "max_commit_source_minus_origin": internal_checks[
                "max_commit_source_minus_origin"
            ],
        },
    }
    output_path = args.full_lineage / "full_panel_verification.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
