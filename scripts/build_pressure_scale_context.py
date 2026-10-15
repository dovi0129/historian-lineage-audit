#!/usr/bin/env python3
"""Put the matched-clock pressure MAE changes on the native XMEAS-7 scale.

No model is refitted.  Saved Swinging Door, MAE, event-time versus commit-time
cells supply the V3-trained losses L33 (V3 evaluation) and L34 (V4 evaluation).
The public TEP files, loaded with the original runner's loader, supply the
standard deviation of the scored pressure targets in each fold--fault cell.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
WORKSPACE = PAPER
OSC_ROOT = PAPER
LINEAGE_DIR = OSC_ROOT / "artifacts/historian_reconstruction_lineage_tep_full_panel_breadth"
EXPORT_DIR = PAPER / "artifacts/manuscript_export"
TABLE_PATH = PAPER / "tables/generated/pressure_scale_context.tex"
MODELS = {"PERSISTENCE": "Persistence", "RIDGE_ARX": "Ridge", "SHALLOW_HGB": "Shallow tree"}


def load_runner():
    path = OSC_ROOT / "scripts/101_historian_vintage_replay_kill_test.py"
    spec = importlib.util.spec_from_file_location("historian_s101", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module here
    spec.loader.exec_module(module)
    return module


def target_scale(s101) -> pd.DataFrame:
    """SD of the scored native targets over the two test trajectories of each cell."""
    segments = s101.load_tep_segments(smoke=False)
    channel = s101.TEP_CHANNEL_INDEX["XMEAS-7"]
    rows = []
    for region, target_name, horizon in s101.TEP_FULL_PANEL_TARGETS:
        if target_name != "XMEAS-7":
            continue
        for fold, _, test_batches in s101.TEP_FOLDS:
            for fault in s101.TEP_FAULTS:
                targets = []
                for segment in segments:
                    if segment.fault != fault or segment.batch not in test_batches:
                        continue
                    # Same origin range as the runner: max lag .. n - horizon - 1.
                    origins = np.arange(
                        max(s101.TEP_LAGS), len(segment.values) - horizon, dtype=np.int64
                    )
                    targets.append(segment.values[origins + horizon, channel])
                if len(targets) != len(test_batches):
                    raise RuntimeError(f"missing test trajectory: {region} fold {fold} fault {fault}")
                y = np.concatenate(targets)
                rows.append(
                    {
                        "region": region,
                        "fold": fold,
                        "fault": fault,
                        "n_targets": int(y.size),
                        "target_sd": float(np.std(y, ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def saved_losses() -> pd.DataFrame:
    frame = pd.read_csv(LINEAGE_DIR / "view_comparison_cells.csv")
    selected = frame.loc[
        frame["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & frame["operator"].eq("SWINGING_DOOR")
        & frame["score"].eq("MAE")
        & frame["region"].isin(["PRESSURE_Q6", "PRESSURE_Q16"])
    ].copy()
    selected["L33"] = selected["loss_left_train_left_eval"]
    selected["L34"] = selected["loss_left_train_right_eval"]
    selected["delta_mae"] = selected["L34"] - selected["L33"]
    wide = selected.pivot_table(
        index=["region", "fold", "fault", "severity_k"],
        columns="model",
        values=["L33", "delta_mae"],
    )
    wide.columns = [f"{value}_{model}" for value, model in wide.columns]
    return wide.reset_index()


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in cells.groupby("region", sort=False):
        if len(group) != 72:
            raise RuntimeError(f"{region}: expected 72 cells, found {len(group)}")
        record = {"region": region, "horizon_min": int(region.split("Q")[1]), "n_cells": len(group)}
        for column in (
            "target_sd",
            "L33_PERSISTENCE",
            "L33_RIDGE_ARX",
            "delta_mae_RIDGE_ARX",
            "delta_over_sd_RIDGE_ARX",
            "delta_over_persistence_RIDGE_ARX",
        ):
            record[f"{column}_median"] = float(group[column].median())
            record[f"{column}_q25"] = float(group[column].quantile(0.25))
            record[f"{column}_q75"] = float(group[column].quantile(0.75))
        rows.append(record)
    return pd.DataFrame(rows).sort_values("horizon_min", ignore_index=True)


def write_table(summary: pd.DataFrame, path: Path) -> None:
    def cell(record, column, fmt):
        return (
            f"{record[f'{column}_median']:{fmt}} "
            f"[{record[f'{column}_q25']:{fmt}}, {record[f'{column}_q75']:{fmt}}]"
        )

    rows = (
        ("Target SD (kPa)", "target_sd", ".3f"),
        ("Persistence $L_{33}$ (kPa)", "L33_PERSISTENCE", ".3f"),
        ("Ridge $L_{33}$ (kPa)", "L33_RIDGE_ARX", ".3f"),
        ("Ridge $\\Delta$MAE (kPa)", "delta_mae_RIDGE_ARX", ".4f"),
        ("$\\Delta$MAE / target SD (\\%)", "delta_over_sd_RIDGE_ARX", ".2f"),
        ("$\\Delta$MAE / persistence $L_{33}$ (\\%)", "delta_over_persistence_RIDGE_ARX", ".2f"),
    )
    records = summary.to_dict("records")
    lines = [
        r"\begin{tabular}{@{}l" + "r" * len(records) + "@{}}",
        r"\toprule",
        "Quantity & " + " & ".join(f"{r['horizon_min']}-min pressure" for r in records) + r" \\",
        r"\midrule",
    ]
    for label, column, fmt in rows:
        lines.append(label + " & " + " & ".join(cell(r, column, fmt) for r in records) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    s101 = load_runner()
    scale = target_scale(s101)
    losses = saved_losses()
    cells = losses.merge(scale, on=["region", "fold", "fault"], how="left", validate="many_to_one")
    if cells["target_sd"].isna().any():
        raise RuntimeError("unmatched cells between saved losses and target scale")
    cells["delta_over_sd_RIDGE_ARX"] = 100.0 * cells["delta_mae_RIDGE_ARX"] / cells["target_sd"]
    cells["delta_over_persistence_RIDGE_ARX"] = (
        100.0 * cells["delta_mae_RIDGE_ARX"] / cells["L33_PERSISTENCE"]
    )
    summary = summarize(cells)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    cells.to_csv(EXPORT_DIR / "pressure_scale_cells.csv", index=False)
    summary.to_csv(EXPORT_DIR / "pressure_scale_summary.csv", index=False)
    write_table(summary, TABLE_PATH)

    # The Ridge Delta-MAE medians must reproduce Supplementary Table S4.
    reference = {"PRESSURE_Q6": 0.03746, "PRESSURE_Q16": 0.02196}
    checks = {
        region: {
            "median_delta_mae": float(summary.loc[summary.region.eq(region), "delta_mae_RIDGE_ARX_median"].iloc[0]),
            "table_value": value,
        }
        for region, value in reference.items()
    }
    for region, check in checks.items():
        if abs(check["median_delta_mae"] - check["table_value"]) > 5e-6:
            raise RuntimeError(f"{region}: Delta-MAE does not reproduce the supplementary table: {check}")
    (EXPORT_DIR / "pressure_scale_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
