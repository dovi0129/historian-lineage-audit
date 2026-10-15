#!/usr/bin/env python3
"""Export public lineage tables and figures from regenerated cells."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
PAPER = Path(__file__).resolve().parents[1]
LINEAGE_DIR = PAPER / "artifacts/historian_reconstruction_lineage_tep_full_panel_breadth"
OUT = PAPER / "artifacts/manuscript_export"
TABLES = PAPER / "tables/generated"
FIGURES = PAPER / "figures/generated"
LINEAGE_REGION_ORDER = ("CSTR_OUTPUT_Q1", "PRESSURE_Q6", "PRESSURE_Q16", "TEMPERATURE_Q6", "TEMPERATURE_Q16")
LINEAGE_REGION_LABELS = {"CSTR_OUTPUT_Q1": "CSTR output, 1 min", "PRESSURE_Q6": "TEP pressure, 6 min", "PRESSURE_Q16": "TEP pressure, 16 min", "TEMPERATURE_Q6": "TEP temperature, 6 min", "TEMPERATURE_Q16": "TEP temperature, 16 min"}
GEOMETRY_REGION_ORDER = ("CSTR_OUTPUT_Q1", "PRESSURE_Q6", "PRESSURE_Q16")
GEOMETRY_REGION_LABELS = {"CSTR_OUTPUT_Q1": "CSTR, 1-min horizon", "PRESSURE_Q6": "TEP, 6-min horizon", "PRESSURE_Q16": "TEP, 16-min horizon"}
EXPECTED_REMEDY_UNITS = 640
REMEDY_ORDER = ("PRACTICALLY_UNCHANGED", "EVALUATION_AND_RETRAINING_REPAIR", "EVALUATION_CORRECTION_DOMINANT", "ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS", "RETRAINING_ADVERSE_OR_HETEROGENEOUS", "MIXED_SMALL_COMPONENTS")
REMEDY_LABELS = dict(zip(REMEDY_ORDER, ("Practically unchanged", "Evaluation + retraining", "Evaluation correction", "Origin view improves / heterogeneous", "Retraining adverse / heterogeneous", "Mixed, small")))
REMEDY_FIGURE_LABELS = {"PRACTICALLY_UNCHANGED": "Practically unchanged (<1%)", "EVALUATION_AND_RETRAINING_REPAIR": r"Both contrasts $\geq1\%$", "EVALUATION_CORRECTION_DOMINANT": "Evaluation contrast only", "ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS": "Origin view lowers loss", "RETRAINING_ADVERSE_OR_HETEROGENEOUS": "Retraining raises loss", "MIXED_SMALL_COMPONENTS": "Mixed / small components"}
REMEDY_MARGINS = (0.005, 0.010, 0.020)
LINE_STYLES = {"CSTR_OUTPUT_Q1": ("0.10", "o", "-"), "PRESSURE_Q6": ("0.28", "s", "-"), "PRESSURE_Q16": ("0.45", "D", "--"), "TEMPERATURE_Q6": ("0.64", "^", "-."), "TEMPERATURE_Q16": ("0.80", "v", ":")}

def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{label} missing columns: {missing}")


def load_lineage_artifacts() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    comparisons = pd.read_csv(LINEAGE_DIR / "view_comparison_cells.csv")
    geometry = pd.read_csv(LINEAGE_DIR / "geometry_differentiation.csv")
    remedies = pd.read_csv(LINEAGE_DIR / "remedy_attribution.csv")
    summary = json.loads((LINEAGE_DIR / "summary.json").read_text(encoding="utf-8"))
    require_columns(
        comparisons,
        {
            "region",
            "operator",
            "severity_k",
            "comparison",
            "score",
            "model",
            "evaluation_view_gap",
            "practical_evaluation_winner_change_1pct",
        },
        "lineage view comparisons",
    )
    require_columns(
        geometry,
        {
            "region",
            "operator",
            "severity_k",
            "naive_grid_pit_miss_share",
            "naive_grid_pit_active_miss_share",
            "median_horizon_erosion_minutes",
            "p95_horizon_erosion_minutes",
            "final_native_feature_sd_nrmse",
            "event_commit_feature_sd_nrmse",
        },
        "lineage geometry",
    )
    require_columns(
        remedies,
        {"region", "operator", "severity_k", "score", "model", "remedy_class"},
        "lineage remedies",
    )
    checks = summary.get("checks", {})
    required_checks = {
        "event_commit_retained_events_identical": True,
        "event_commit_final_export_max_abs_delta": 0.0,
        "periodic_event_commit_score_max_abs_delta": 0.0,
        "periodic_event_commit_feature_max_abs_delta": 0.0,
        "max_grid_time_minus_origin": 0,
        "max_commit_source_minus_origin": 0,
    }
    for key, expected in required_checks.items():
        value = checks.get(key)
        if isinstance(expected, bool):
            if value is not expected:
                raise RuntimeError(f"lineage invariant failed: {key}={value}")
        elif value is None or abs(float(value) - expected) > 1e-12:
            raise RuntimeError(f"lineage invariant failed: {key}={value}")
    return comparisons, geometry, remedies, summary


def build_lineage_geometry_summary(geometry: pd.DataFrame) -> pd.DataFrame:
    output = (
        geometry.groupby(["region", "operator", "severity_k"], sort=False)
        .agg(
            n_units=("naive_grid_pit_miss_share", "size"),
            median_grid_pit_miss_share=("naive_grid_pit_miss_share", "median"),
            median_active_grid_pit_miss_share=(
                "naive_grid_pit_active_miss_share",
                "median",
            ),
            median_horizon_erosion_minutes=(
                "median_horizon_erosion_minutes",
                "median",
            ),
            median_p95_horizon_erosion_minutes=(
                "p95_horizon_erosion_minutes",
                "median",
            ),
            median_final_native_feature_sd_nrmse=(
                "final_native_feature_sd_nrmse",
                "median",
            ),
            median_event_commit_feature_sd_nrmse=(
                "event_commit_feature_sd_nrmse",
                "median",
            ),
        )
        .reset_index()
    )
    return output


def build_event_commit_summary(comparisons: pd.DataFrame) -> pd.DataFrame:
    selected = comparisons.loc[
        comparisons["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & comparisons["score"].eq("MAE")
        & comparisons["model"].str.startswith("RIDGE")
    ].copy()
    output = (
        selected.groupby(["region", "operator", "severity_k"], sort=False)
        .agg(
            n_units=("evaluation_view_gap", "size"),
            median_evaluation_gap=("evaluation_view_gap", "median"),
            q25_evaluation_gap=("evaluation_view_gap", lambda x: x.quantile(0.25)),
            q75_evaluation_gap=("evaluation_view_gap", lambda x: x.quantile(0.75)),
            max_absolute_evaluation_gap=(
                "evaluation_view_gap",
                lambda x: x.abs().max(),
            ),
            practical_winner_changes=(
                "practical_evaluation_winner_change_1pct",
                "sum",
            ),
        )
        .reset_index()
    )
    return output


def classify_remedy_values(
    deployment: float,
    repair: float,
    diagonal: float,
    margin: float,
) -> str:
    """Apply the frozen ordered remedy rule on the relative-loss scale."""

    if margin <= 0 or not np.isfinite(margin):
        raise ValueError("remedy margin must be finite and positive")
    if max(abs(deployment), abs(diagonal)) < margin:
        return "PRACTICALLY_UNCHANGED"
    if deployment >= margin and repair >= margin:
        return "EVALUATION_AND_RETRAINING_REPAIR"
    if deployment >= margin and abs(repair) < margin:
        return "EVALUATION_CORRECTION_DOMINANT"
    if deployment <= -margin:
        return "ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS"
    if repair <= -margin:
        return "RETRAINING_ADVERSE_OR_HETEROGENEOUS"
    return "MIXED_SMALL_COMPONENTS"


def remedy_margin_column(margin: float) -> str:
    return f"remedy_class_m{int(round(1000 * margin)):03d}"


def build_remedy_cells(remedies: pd.DataFrame) -> pd.DataFrame:
    """Validate and augment the full-panel Ridge/MAE remedy stratum."""

    require_columns(
        remedies,
        {
            "process",
            "region",
            "target",
            "seed",
            "fold",
            "fault",
            "operator",
            "severity_k",
            "comparison",
            "score",
            "model",
            "loss_left_train_left_eval",
            "loss_left_train_right_eval",
            "loss_right_train_left_eval",
            "loss_right_train_right_eval",
            "evaluation_view_gap",
            "right_training_repair",
            "diagonal_gap",
            "remedy_class",
        },
        "lineage remedy cells",
    )
    selected = remedies.loc[
        remedies["comparison"].eq("FINAL_VS_EVENT_COMMIT")
        & remedies["score"].eq("MAE")
        & remedies["model"].str.startswith("RIDGE")
    ].copy()
    if len(selected) != EXPECTED_REMEDY_UNITS:
        raise RuntimeError(
            f"expected {EXPECTED_REMEDY_UNITS} Ridge/MAE remedy cells, "
            f"found {len(selected)}"
        )
    if set(selected["region"]) != set(LINEAGE_REGION_ORDER):
        raise RuntimeError("unexpected remedy-region support")

    loss_columns = [
        "loss_left_train_left_eval",
        "loss_left_train_right_eval",
        "loss_right_train_left_eval",
        "loss_right_train_right_eval",
    ]
    losses = selected[loss_columns].to_numpy(float)
    if not np.all(np.isfinite(losses)) or np.any(losses <= 0):
        raise RuntimeError("remedy log contrasts require finite positive losses")

    ff = selected["loss_left_train_left_eval"].to_numpy(float)
    fo = selected["loss_left_train_right_eval"].to_numpy(float)
    of = selected["loss_right_train_left_eval"].to_numpy(float)
    oo = selected["loss_right_train_right_eval"].to_numpy(float)
    expected = {
        "evaluation_view_gap": (fo - ff) / ff,
        "right_training_repair": (fo - oo) / fo,
        "diagonal_gap": (oo - ff) / ff,
    }
    for column, values in expected.items():
        if not np.allclose(
            selected[column].to_numpy(float), values, rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"saved remedy contrast does not reproduce: {column}")

    selected["ell_e_final_train"] = np.log(fo / ff)
    selected["ell_b_origin_eval"] = np.log(fo / oo)
    selected["ell_diagonal"] = np.log(oo / ff)
    selected["ell_e_origin_train"] = np.log(oo / of)
    selected["ell_b_final_eval"] = np.log(ff / of)
    selected["i_log"] = (
        selected["ell_e_origin_train"] - selected["ell_e_final_train"]
    )
    interaction_check = selected["ell_b_final_eval"] - selected["ell_b_origin_eval"]
    if not np.allclose(selected["i_log"], interaction_check, rtol=1e-10, atol=1e-12):
        raise RuntimeError("four-loss log interaction identity failed")
    if not np.allclose(
        selected["ell_diagonal"],
        selected["ell_e_final_train"] - selected["ell_b_origin_eval"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("ordered-path log identity failed")

    for margin in REMEDY_MARGINS:
        column = remedy_margin_column(margin)
        selected[column] = [
            classify_remedy_values(deployment, repair, diagonal, margin)
            for deployment, repair, diagonal in zip(
                selected["evaluation_view_gap"],
                selected["right_training_repair"],
                selected["diagonal_gap"],
            )
        ]
        if not set(selected[column]).issubset(REMEDY_ORDER):
            raise RuntimeError(f"unrecognized remedy class at margin {margin}")

    one_percent = remedy_margin_column(0.01)
    mismatches = selected[one_percent].ne(selected["remedy_class"])
    if mismatches.any():
        raise RuntimeError(
            f"1% remedy classifier differs from frozen labels in {int(mismatches.sum())} rows"
        )
    return selected.reset_index(drop=True)


def build_remedy_threshold_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("ALL", "ALL", cells)]
    scopes.extend(
        (region, operator, group)
        for (region, operator), group in cells.groupby(
            ["region", "operator"], sort=False
        )
    )
    for margin in REMEDY_MARGINS:
        column = remedy_margin_column(margin)
        for region, operator, group in scopes:
            counts = group[column].value_counts()
            total = len(group)
            for remedy_class in REMEDY_ORDER:
                count = int(counts.get(remedy_class, 0))
                rows.append(
                    {
                        "margin": margin,
                        "margin_percent": 100 * margin,
                        "region": region,
                        "operator": operator,
                        "remedy_class": remedy_class,
                        "n_units": count,
                        "total_units": total,
                        "share": count / total,
                    }
                )
    output = pd.DataFrame(rows)
    overall = output.loc[output["region"].eq("ALL")]
    sums = overall.groupby("margin")["n_units"].sum()
    if not (sums == len(cells)).all():
        raise RuntimeError(f"remedy classes are not exhaustive: {sums.to_dict()}")
    return output


def build_interaction_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("ALL", "ALL", cells)]
    for region in LINEAGE_REGION_ORDER:
        for operator in ("PERIODIC", "SWINGING_DOOR"):
            group = cells.loc[
                cells["region"].eq(region) & cells["operator"].eq(operator)
            ]
            if group.empty:
                raise RuntimeError(f"missing interaction group: {region}/{operator}")
            scopes.append((region, operator, group))
    threshold = math.log1p(0.01)
    for region, operator, group in scopes:
        rows.append(
            {
                "region": region,
                "operator": operator,
                "n_units": len(group),
                "median_evaluation_view_gap": group["evaluation_view_gap"].median(),
                "median_origin_training_benefit": group[
                    "right_training_repair"
                ].median(),
                "median_diagonal_gap": group["diagonal_gap"].median(),
                "median_i_log": group["i_log"].median(),
                "q25_i_log": group["i_log"].quantile(0.25),
                "q75_i_log": group["i_log"].quantile(0.75),
                "path_sensitive_n_1pct": int((group["i_log"].abs() >= threshold).sum()),
                "path_sensitive_share_1pct": float(
                    (group["i_log"].abs() >= threshold).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_fault_summary(cells: pd.DataFrame) -> pd.DataFrame:
    selected = cells.loc[cells["process"].eq("TEP")].copy()
    column = remedy_margin_column(0.01)
    rows: list[dict[str, Any]] = []
    for (region, operator, fault), group in selected.groupby(
        ["region", "operator", "fault"], sort=True
    ):
        counts = group[column].value_counts()
        if len(group) != 12:
            raise RuntimeError(
                f"expected 12 TEP fold-severity cells for {region}/{operator}/fault {fault}"
            )
        row: dict[str, Any] = {
            "region": region,
            "operator": operator,
            "fault": int(fault),
            "n_units": len(group),
        }
        for remedy_class in REMEDY_ORDER:
            row[remedy_class] = int(counts.get(remedy_class, 0))
        rows.append(row)
    output = pd.DataFrame(rows)
    class_columns = list(REMEDY_ORDER)
    if not (output[class_columns].sum(axis=1) == output["n_units"]).all():
        raise RuntimeError("fault-stratified remedy counts are not exhaustive")
    return output


def write_lineage_geometry_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, str]] = []
    for region in GEOMETRY_REGION_ORDER:
        for operator in ("PERIODIC", "SWINGING_DOOR"):
            group = summary.loc[
                summary["region"].eq(region) & summary["operator"].eq(operator)
            ].set_index("severity_k")
            rows.append(
                {
                    "Study target": GEOMETRY_REGION_LABELS[region],
                    "Retention": "Periodic"
                    if operator == "PERIODIC"
                    else "Swinging Door",
                    r"\shortstack{Row-time miss\\$k=2/8$}": (
                        f"{100 * float(group.loc[2, 'median_grid_pit_miss_share']):.1f}/"
                        f"{100 * float(group.loc[8, 'median_grid_pit_miss_share']):.1f}"
                    ),
                    r"\shortstack{Erosion (min)\\$k=2/8$}": (
                        f"{float(group.loc[2, 'median_horizon_erosion_minutes']):.1f}/"
                        f"{float(group.loc[8, 'median_horizon_erosion_minutes']):.1f}"
                    ),
                }
            )
    display = pd.DataFrame(rows)
    latex = display.to_latex(
        index=False,
        escape=False,
        column_format=r"@{}llcc@{}",
    )
    path.write_text(latex, encoding="utf-8")


def write_event_commit_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[summary["operator"].eq("SWINGING_DOOR")].copy()
    rows: list[dict[str, str | int]] = []
    for region in LINEAGE_REGION_ORDER:
        group = selected.loc[selected["region"].eq(region)].set_index("severity_k")
        row: dict[str, str | int] = {"Study target": LINEAGE_REGION_LABELS[region]}
        for severity in (2, 3, 5, 8):
            row[f"$k={severity}$"] = (
                f"{100 * float(group.loc[severity, 'median_evaluation_gap']):+.2f}"
            )
        row["Winner changes"] = int(group["practical_winner_changes"].sum())
        rows.append(row)
    latex = pd.DataFrame(rows).to_latex(
        index=False,
        escape=False,
        column_format=r"@{}lccccc@{}",
    )
    path.write_text(latex, encoding="utf-8")


def write_tep_selection_context_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, str | int]] = []
    target_labels = {"XMEAS-7": "Pressure", "XMEAS-9": "Temperature"}
    frozen = {("XMEAS-7", 16), ("XMEAS-9", 6)}
    for target in ("XMEAS-7", "XMEAS-9"):
        for horizon in (6, 16):
            group = summary.loc[
                summary["target"].eq(target)
                & summary["horizon_minutes"].eq(horizon)
            ].set_index("analysis_spec")
            if set(group.index) != {"C3_MAE", "C3_RMSE"}:
                raise ValueError(f"incomplete TEP selection cell: {target}/{horizon}")
            if not (group["n_cells"] == 432).all():
                raise ValueError(f"unexpected TEP selection support: {target}/{horizon}")
            rows.append(
                {
                    "Target": target_labels[target],
                    "Horizon (min)": horizon,
                    r"\shortstack{MAE\\practical/supported}": (
                        f"{int(group.loc['C3_MAE', 'practical'])}/"
                        f"{int(group.loc['C3_MAE', 'supported'])}"
                    ),
                    r"\shortstack{RMSE\\practical/supported}": (
                        f"{int(group.loc['C3_RMSE', 'practical'])}/"
                        f"{int(group.loc['C3_RMSE', 'supported'])}"
                    ),
                    r"\shortstack{Original lineage\\follow-up}": (
                        "Yes" if (target, horizon) in frozen else "No"
                    ),
                }
            )
    path.write_text(
        pd.DataFrame(rows).to_latex(
            index=False,
            escape=False,
            column_format=r"@{}lcccc@{}",
        ),
        encoding="utf-8",
    )


def write_remedy_rule_table(path: Path) -> None:
    display = pd.DataFrame(
        [
            (1, REMEDY_LABELS["PRACTICALLY_UNCHANGED"], r"$\max(|g_E|,|g_D|)<m$"),
            (
                2,
                REMEDY_LABELS["EVALUATION_AND_RETRAINING_REPAIR"],
                r"$g_E\geq m$ and $g_R\geq m$",
            ),
            (
                3,
                REMEDY_LABELS["EVALUATION_CORRECTION_DOMINANT"],
                r"$g_E\geq m$ and $|g_R|<m$",
            ),
            (
                4,
                REMEDY_LABELS["ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS"],
                r"$g_E\leq-m$",
            ),
            (
                5,
                REMEDY_LABELS["RETRAINING_ADVERSE_OR_HETEROGENEOUS"],
                r"$g_R\leq-m$",
            ),
            (6, REMEDY_LABELS["MIXED_SMALL_COMPONENTS"], "Otherwise"),
        ],
        columns=["Priority", "Response pattern", "First satisfied condition"],
    )
    path.write_text(
        display.to_latex(
            index=False,
            escape=False,
            column_format=r"@{}cll@{}",
        ),
        encoding="utf-8",
    )


def write_remedy_threshold_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary.loc[
        summary["region"].eq("ALL") & summary["operator"].eq("ALL")
    ].copy()
    rows: list[dict[str, str]] = []
    for remedy_class in REMEDY_ORDER:
        row = {"Response pattern": REMEDY_LABELS[remedy_class]}
        for margin in REMEDY_MARGINS:
            cell = selected.loc[
                np.isclose(selected["margin"], margin)
                & selected["remedy_class"].eq(remedy_class)
            ].iloc[0]
            row[f"$m={100 * margin:g}\\%$"] = (
                f"{int(cell['n_units'])}/{int(cell['total_units'])} "
                f"({100 * float(cell['share']):.1f}\\%)"
            )
        rows.append(row)
    path.write_text(
        pd.DataFrame(rows).to_latex(
            index=False,
            escape=False,
            column_format=r"@{}lccc@{}",
        ),
        encoding="utf-8",
    )


def write_interaction_summary_table(summary: pd.DataFrame, path: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for record in summary.itertuples(index=False):
        if record.region == "ALL":
            region_label = "All study targets"
            operator_label = "Both"
        else:
            region_label = LINEAGE_REGION_LABELS[record.region]
            operator_label = (
                "Periodic" if record.operator == "PERIODIC" else "Swinging Door"
            )
        group_label = rf"\shortstack[l]{{{region_label}\\{operator_label}}}"
        rows.append(
            {
                "Group": group_label,
                "$N$": int(record.n_units),
                "$g_E/g_R/g_D$ (\\%)": (
                    f"{100 * record.median_evaluation_view_gap:+.1f}/"
                    f"{100 * record.median_origin_training_benefit:+.1f}/"
                    f"{100 * record.median_diagonal_gap:+.1f}"
                ),
                "$100I_{\\log}$ [IQR]": (
                    f"{100 * record.median_i_log:+.1f} "
                    f"[{100 * record.q25_i_log:+.1f}, {100 * record.q75_i_log:+.1f}]"
                ),
                "Sensitive": (
                    f"{int(record.path_sensitive_n_1pct)}/{int(record.n_units)}"
                ),
            }
        )
    path.write_text(
        pd.DataFrame(rows).to_latex(
            index=False,
            escape=False,
            column_format=r"@{}lrccr@{}",
        ),
        encoding="utf-8",
    )


def write_fault_summary_table(summary: pd.DataFrame, path: Path) -> None:
    class_columns = list(REMEDY_ORDER)
    combined = (
        summary.groupby(["region", "fault"], sort=True)[["n_units", *class_columns]]
        .sum()
        .reset_index()
    )
    rows: list[dict[str, str | int]] = []
    for record in combined.itertuples(index=False):
        denominator = int(record.n_units)
        rows.append(
            {
                "TEP target": LINEAGE_REGION_LABELS[record.region].replace("TEP ", ""),
                "Fault": int(record.fault),
                "$N$": denominator,
                "Unchanged": f"{int(getattr(record, 'PRACTICALLY_UNCHANGED'))}/{denominator}",
                "Eval.+train": f"{int(getattr(record, 'EVALUATION_AND_RETRAINING_REPAIR'))}/{denominator}",
                "Eval. only": f"{int(getattr(record, 'EVALUATION_CORRECTION_DOMINANT'))}/{denominator}",
                "Origin better": f"{int(getattr(record, 'ORIGIN_VIEW_IMPROVES_OR_HETEROGENEOUS'))}/{denominator}",
                "Retrain adverse": f"{int(getattr(record, 'RETRAINING_ADVERSE_OR_HETEROGENEOUS'))}/{denominator}",
                "Mixed": f"{int(getattr(record, 'MIXED_SMALL_COMPONENTS'))}/{denominator}",
            }
        )
    path.write_text(
        pd.DataFrame(rows).to_latex(
            index=False,
            escape=False,
            column_format=r"@{}lrrrrrrrr@{}",
        ),
        encoding="utf-8",
    )


def write_availability_provenance_table(
    provenance: pd.DataFrame, path: Path
) -> None:
    labels = {
        "ASSUMED_ZERO_LATENCY": "Assumed zero latency",
        "RECONSTRUCTED_DECISION": "Reconstructed from decision",
        "OBSERVED_AVAILABILITY": "Observed availability",
        "UNAVAILABLE": "Unavailable",
    }
    display = provenance.copy()
    display["state"] = display["state"].map(labels)
    display = display.rename(
        columns={
            "evidence": "Evidence",
            "state": "Availability provenance",
            "use": "Interpretation",
        }
    )
    latex = display.to_latex(
        index=False,
        escape=True,
        column_format=(
            r"@{}>{\raggedright\arraybackslash}p{2.7cm}"
            r">{\raggedright\arraybackslash}p{2.9cm}"
            r">{\raggedright\arraybackslash}p{6.0cm}@{}"
        ),
    )
    path.write_text(latex, encoding="utf-8")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12.5,
            "axes.labelsize": 12.5,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_lineage_geometry(geometry: pd.DataFrame, output: Path) -> None:
    """Show unit-level horizon erosion without repeating deterministic shares."""

    configure_plotting()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )
    selected = geometry.loc[geometry["region"].isin(GEOMETRY_REGION_ORDER)].copy()
    selected["erosion_ratio"] = (
        selected["median_horizon_erosion_minutes"] / selected["horizon_minutes"]
    )
    colors = {
        "CSTR_OUTPUT_Q1": "#222222",
        "PRESSURE_Q6": "#33658A",
        "PRESSURE_Q16": "#C45A34",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    for axis, operator, title in zip(
        axes,
        ("PERIODIC", "SWINGING_DOOR"),
        ("(a) Periodic retention", "(b) Swinging Door retention"),
    ):
        subset = selected.loc[selected["operator"].eq(operator)]
        for region in GEOMETRY_REGION_ORDER:
            points = subset.loc[subset["region"].eq(region)]
            marker = LINE_STYLES[region][1]
            axis.scatter(
                100 * points["retained_event_ratio"],
                points["erosion_ratio"],
                s=22,
                marker=marker,
                facecolor=colors[region],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.45,
                zorder=2,
            )
            medians = (
                points.groupby("severity_k", sort=True)
                .agg(
                    x=("retained_event_ratio", "median"),
                    y=("erosion_ratio", "median"),
                )
                .sort_index()
            )
            axis.plot(
                100 * medians["x"],
                medians["y"],
                color=colors[region],
                marker=marker,
                markersize=4.5,
                linewidth=1.25,
                zorder=3,
                label=GEOMETRY_REGION_LABELS[region],
            )
        axis.axhspan(1.0, 55, color="0.96", zorder=0)
        axis.axhline(1.0, color="0.25", linestyle="--", linewidth=0.85, zorder=1)
        axis.text(
            0.03,
            0.95,
            "Inputs available\nafter target time",
            transform=axis.transAxes,
            fontsize=8.4,
            ha="left",
            va="top",
            color="0.25",
        )
        axis.set_yscale("symlog", linthresh=0.1, linscale=0.75, base=10)
        axis.set_ylim(-0.015, 55)
        axis.set_yticks((0, 0.1, 0.5, 1, 5, 10, 50))
        axis.set_yticklabels(("0", "0.1", "0.5", "1", "5", "10", "50"))
        axis.set_xlim(56, 8)
        axis.set_xticks((12.5, 20, 33.3, 50))
        axis.set_xticklabels(("12.5", "20", "33", "50"))
        axis.set_title(title, loc="left", fontsize=10.2)
        axis.set_xlabel("Achieved retained events (%)")
        axis.grid(color="0.91", linewidth=0.5)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Horizon consumed, $E_h/h$\n(unit median)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3)
    fig.text(0.53, 0.015,
             "$E_h/h=1$: final inputs become available at the target time",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.88), w_pad=1.0)
    save_figure(fig, output / "lineage_pit_horizon")


def contrast_text_color(rgba: tuple[float, ...]) -> str:
    """Choose the higher-contrast black/white label on an opaque sRGB cell."""
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, (0.2126, 0.7152, 0.0722)))
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "black" if black_contrast >= white_contrast else "white"


def plot_event_commit_effect(comparisons: pd.DataFrame, output: Path) -> None:
    """Map paired event-time/commit-time effects across tasks and retention levels."""

    configure_plotting()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )
    selected = comparisons.loc[
        comparisons["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & comparisons["operator"].eq("SWINGING_DOOR")
        & comparisons["score"].eq("MAE")
        & comparisons["model"].str.startswith("RIDGE")
        & comparisons["region"].isin(LINEAGE_REGION_ORDER)
    ].copy()
    grouped = (
        selected.groupby(["region", "severity_k"], sort=False)
        .agg(
            median_effect=("evaluation_view_gap", "median"),
            positive=("evaluation_view_gap", lambda values: int((values > 0).sum())),
            changed=("practical_evaluation_winner_change_1pct", "sum"),
            total=("evaluation_view_gap", "size"),
        )
        .reset_index()
    )
    severities = (2, 3, 5, 8)
    expected_cells = len(LINEAGE_REGION_ORDER) * len(severities)
    if len(grouped) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} task-by-severity cells, found {len(grouped)}"
        )

    effect_matrix = np.empty((len(LINEAGE_REGION_ORDER), len(severities)))
    winner_matrix = np.empty_like(effect_matrix)
    effect_text: list[list[str]] = []
    winner_text: list[list[str]] = []
    for row, region in enumerate(LINEAGE_REGION_ORDER):
        effect_row: list[str] = []
        winner_row: list[str] = []
        for column, severity in enumerate(severities):
            cell = grouped.loc[
                grouped["region"].eq(region)
                & grouped["severity_k"].eq(severity)
            ].iloc[0]
            effect = 100 * float(cell["median_effect"])
            total = int(cell["total"])
            effect_matrix[row, column] = effect
            winner_matrix[row, column] = float(cell["changed"])
            effect_row.append(
                f"{effect:+.2f}%\n{int(cell['positive'])}/{total} > 0"
            )
            winner_row.append(f"{int(cell['changed'])}/{total}")
        effect_text.append(effect_row)
        winner_text.append(winner_row)

    display_labels = {
        "CSTR_OUTPUT_Q1": "CSTR, 1 min",
        "PRESSURE_Q6": "Pressure, 6 min",
        "PRESSURE_Q16": "Pressure, 16 min",
        "TEMPERATURE_Q6": "Temperature, 6 min",
        "TEMPERATURE_Q16": "Temperature, 16 min",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.35),
        gridspec_kw={"width_ratios": (1.10, 1.0)},
        sharey=True,
    )

    # Equal absolute changes must have equal distance from the neutral colour.
    effect_limit = max(7.0, float(np.ceil(np.abs(effect_matrix).max())))
    effect_norm = mcolors.TwoSlopeNorm(vmin=-effect_limit, vcenter=0.0, vmax=effect_limit)
    effect_cmap = plt.get_cmap("RdBu_r")
    effect_image = axes[0].imshow(
        effect_matrix,
        aspect="auto",
        cmap=effect_cmap,
        norm=effect_norm,
    )
    for row in range(effect_matrix.shape[0]):
        for column in range(effect_matrix.shape[1]):
            background = effect_cmap(effect_norm(effect_matrix[row, column]))
            axes[0].text(
                column,
                row,
                effect_text[row][column],
                ha="center",
                va="center",
                fontsize=8.3,
                color=contrast_text_color(background),
            )
    axes[0].set_title("(a) Ridge / MAE", loc="left")
    effect_colorbar = fig.colorbar(
        effect_image, ax=axes[0], fraction=0.050, pad=0.030
    )
    effect_colorbar.set_label("Median MAE change (%)", fontsize=8.2)
    effect_colorbar.set_ticks((-effect_limit, 0, effect_limit))
    effect_colorbar.ax.tick_params(labelsize=8.2)

    winner_norm = mcolors.Normalize(vmin=0, vmax=max(6, winner_matrix.max()))
    winner_cmap = plt.get_cmap("YlOrBr")
    winner_image = axes[1].imshow(
        winner_matrix,
        aspect="auto",
        cmap=winner_cmap,
        norm=winner_norm,
    )
    for row in range(winner_matrix.shape[0]):
        for column in range(winner_matrix.shape[1]):
            background = winner_cmap(winner_norm(winner_matrix[row, column]))
            axes[1].text(
                column,
                row,
                winner_text[row][column],
                ha="center",
                va="center",
                fontsize=8.8,
                color=contrast_text_color(background),
            )
    axes[1].set_title("(b) Model choice", loc="left")
    winner_colorbar = fig.colorbar(
        winner_image, ax=axes[1], fraction=0.047, pad=0.030
    )
    winner_colorbar.set_label("Changed units", fontsize=8.2)
    winner_colorbar.ax.tick_params(labelsize=8.2)

    for axis in axes:
        axis.set_xticks(range(len(severities)), ("50", "33.3", "20", "12.5"))
        axis.set_yticks(
            range(len(LINEAGE_REGION_ORDER)),
            [display_labels[region] for region in LINEAGE_REGION_ORDER],
        )
        axis.tick_params(length=0)
        axis.axhline(0.5, color="white", linewidth=1.2)
        axis.axhline(2.5, color="white", linewidth=1.2)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
    axes[1].tick_params(labelleft=False)
    axes[0].set_xlabel("Target retained events (%)")
    axes[1].set_xlabel("Target retained events (%)")
    fig.tight_layout(w_pad=0.8)
    save_figure(fig, output / "lineage_event_commit_effect")

def plot_pressure_fault_map(comparisons: pd.DataFrame, output: Path) -> None:
    """Show which TEP pressure fault families generate the clock-effect upper tail."""

    configure_plotting()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.2,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )
    selected = comparisons.loc[
        comparisons["comparison"].eq("EVENT_TIME_VS_EVENT_COMMIT")
        & comparisons["operator"].eq("SWINGING_DOOR")
        & comparisons["score"].eq("MAE")
        & comparisons["model"].str.startswith("RIDGE")
        & comparisons["region"].isin(("PRESSURE_Q6", "PRESSURE_Q16"))
    ].copy()
    grouped = (
        selected.groupby(["region", "fault", "severity_k"], sort=False)
        .agg(median_effect=("evaluation_view_gap", "median"))
        .reset_index()
    )
    faults = (1, 4, 5, 11, 14, 15)
    severities = (2, 3, 5, 8)
    expected_cells = 2 * len(faults) * len(severities)
    if len(grouped) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} pressure fault-map cells, found {len(grouped)}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    norm = mcolors.LogNorm(vmin=0.8, vmax=70)
    cmap = plt.get_cmap("YlOrBr")
    image = None
    for axis, region in zip(axes, ("PRESSURE_Q6", "PRESSURE_Q16")):
        matrix = np.empty((len(faults), len(severities)))
        for row, fault in enumerate(faults):
            for column, severity in enumerate(severities):
                cell = grouped.loc[
                    grouped["region"].eq(region)
                    & grouped["fault"].eq(fault)
                    & grouped["severity_k"].eq(severity)
                ].iloc[0]
                matrix[row, column] = 100 * float(cell["median_effect"])
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                red, green, blue, _ = cmap(norm(matrix[row, column]))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="black" if luminance > 0.52 else "white",
                )
        horizon = 6 if region.endswith("Q6") else 16
        axis.set_title(f"TEP pressure, {horizon} min", loc="left")
        axis.set_xticks(range(len(severities)), ("50", "33.3", "20", "12.5"))
        axis.set_xlabel("Target retained events (%)")
        axis.set_yticks(range(len(faults)), [f"Fault {fault}" for fault in faults])
        axis.tick_params(length=0)
    axes[0].set_ylabel("Fault family (median across three folds)")
    if image is None:
        raise RuntimeError("pressure fault map produced no image")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.027, pad=0.025)
    colorbar.set_label("Median relative MAE increase (%)", fontsize=8.5)
    colorbar.set_ticks((1, 2, 5, 10, 20, 50))
    colorbar.set_ticklabels(("1", "2", "5", "10", "20", "50"))
    colorbar.ax.tick_params(labelsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.89, bottom=0.18, top=0.88, wspace=0.12)
    save_figure(fig, output / "lineage_pressure_fault_map")


def plot_remedy_effects(cells: pd.DataFrame, output: Path) -> None:
    """Show the continuous ordered-path effects without classifying cells."""

    configure_plotting()
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.3), sharex=True, sharey=True)
    markers = {2: "o", 3: "s", 5: "^", 8: "D"}
    shades = {2: "0.12", 3: "0.34", 5: "0.56", 8: "0.78"}
    flat_axes = axes.ravel()
    for axis, region in zip(flat_axes, LINEAGE_REGION_ORDER):
        group = cells.loc[cells["region"].eq(region)]
        for operator in ("PERIODIC", "SWINGING_DOOR"):
            operator_points = group.loc[group["operator"].eq(operator)]
            for severity in (2, 3, 5, 8):
                points = operator_points.loc[
                    operator_points["severity_k"].eq(severity)
                ]
                axis.scatter(
                    100 * points["evaluation_view_gap"],
                    100 * points["right_training_repair"],
                    s=20,
                    marker=markers[severity],
                    facecolor=(shades[severity] if operator == "SWINGING_DOOR" else "none"),
                    edgecolor="black",
                    linewidth=0.55,
                    alpha=0.70,
                )
        axis.axvline(0, color="0.25", linewidth=0.65)
        axis.axhline(0, color="0.25", linewidth=0.65)
        for boundary in (-1, 1):
            axis.axvline(boundary, color="0.55", linewidth=0.55, linestyle="--")
            axis.axhline(boundary, color="0.55", linewidth=0.55, linestyle="--")
        axis.set_xscale("symlog", linthresh=1, linscale=0.8, base=10)
        axis.set_yscale("symlog", linthresh=1, linscale=0.8, base=10)
        axis.set_xlim(-10, 220)
        axis.set_ylim(-100, 100)
        axis.set_xticks((-5, -1, 0, 1, 10, 100))
        axis.set_yticks((-50, -10, -1, 0, 1, 10, 50))
        axis.tick_params(labelsize=10.0, labelbottom=True, labelleft=True)
        axis.grid(color="0.91", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.text(
            0.04,
            0.93,
            f"$N={len(group)}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
        )
        axis.set_title(LINEAGE_REGION_LABELS[region], fontsize=9.8)

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    legend_axis.text(0.04, 1.02,
                     "Right: origin evaluation raises loss\nUp: origin-view retraining lowers loss",
                     transform=legend_axis.transAxes, fontsize=9.4, va="top")
    legend_axis.text(
        0.06,
        0.82,
        "Retention rule",
        transform=legend_axis.transAxes,
        fontweight="bold",
        fontsize=9.3,
    )
    legend_axis.scatter(
        [0.10], [0.69], s=32, marker="o", facecolor="none", edgecolor="black",
        transform=legend_axis.transAxes,
    )
    legend_axis.text(0.20, 0.69, "Periodic (open)", transform=legend_axis.transAxes, va="center", fontsize=8.8)
    legend_axis.scatter(
        [0.10], [0.58], s=32, marker="o", facecolor="0.35", edgecolor="black",
        transform=legend_axis.transAxes,
    )
    legend_axis.text(0.20, 0.58, "Swinging Door (filled)", transform=legend_axis.transAxes, va="center", fontsize=8.8)
    legend_axis.text(
        0.06,
        0.42,
        "Retained-fraction marker",
        transform=legend_axis.transAxes,
        fontweight="bold",
        fontsize=9.3,
    )
    for index, severity in enumerate((2, 3, 5, 8)):
        y = 0.31 - 0.09 * index
        legend_axis.scatter(
            [0.10], [y], s=30, marker=markers[severity], facecolor=shades[severity],
            edgecolor="black", transform=legend_axis.transAxes,
        )
        legend_axis.text(
            0.20,
            y,
            f"$k={severity}$ ({100 / severity:.3g}%)",
            transform=legend_axis.transAxes,
            va="center",
            fontsize=8.5,
        )
    fig.supxlabel(
        "Loss increase on origin-view evaluation, $100g_E$ (%; symlog)",
        fontsize=10.8,
        y=0.035,
    )
    fig.supylabel(
        "Training-view loss contrast under origin evaluation, $100g_R$ (%; symlog)",
        fontsize=10.8,
    )
    fig.text(
        0.53,
        0.012,
        "Upper right: both contrasts positive; these are paired comparisons, not a causal partition.",
        ha="center",
        fontsize=9.0,
    )
    fig.tight_layout(rect=(0.04, 0.08, 1.0, 0.98), w_pad=0.7, h_pad=0.75)
    save_figure(fig, output / "lineage_remedy_effects")


def plot_pressure_remedy_effects(cells: pd.DataFrame, output: Path) -> None:
    """Main-text view of crossed-vintage effects for the two pressure tasks."""

    configure_plotting()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
        }
    )
    selected = cells.loc[
        cells["region"].isin(("PRESSURE_Q6", "PRESSURE_Q16"))
    ].copy()
    operator_styles = {
        "PERIODIC": {
            "label": "Periodic observations",
            "facecolor": "white",
            "edgecolor": "#4f83a3",
            "alpha": 0.58,
        },
        "SWINGING_DOOR": {
            "label": "Swinging Door observations",
            "facecolor": "#1f4f6b",
            "edgecolor": "#17394b",
            "alpha": 0.58,
        },
    }
    median_offsets = {
        ("PRESSURE_Q6", "PERIODIC"): (10, -34),
        ("PRESSURE_Q6", "SWINGING_DOOR"): (12, 18),
        ("PRESSURE_Q16", "PERIODIC"): (8, -34),
        ("PRESSURE_Q16", "SWINGING_DOOR"): (12, 18),
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), sharey=True)
    for axis, region in zip(axes, ("PRESSURE_Q6", "PRESSURE_Q16")):
        group = selected.loc[selected["region"].eq(region)]
        x_values = 100 * group["evaluation_view_gap"]
        y_values = 100 * group["right_training_repair"]
        x_max = 10 * math.ceil(1.04 * x_values.max() / 10)
        y_min, y_max = -15.0, 40.0

        axis.fill_between(
            (1, x_max),
            1,
            y_max,
            color="0.96",
            zorder=0,
        )
        axis.axvline(0, color="0.30", linewidth=0.75)
        axis.axhline(0, color="0.30", linewidth=0.75)
        axis.axvline(1, color="0.65", linewidth=0.6, linestyle="--")
        for boundary in (-1, 1):
            axis.axhline(boundary, color="0.65", linewidth=0.6, linestyle="--")

        for operator in ("PERIODIC", "SWINGING_DOOR"):
            points = group.loc[group["operator"].eq(operator)]
            style = operator_styles[operator]
            axis.scatter(
                100 * points["evaluation_view_gap"],
                100 * points["right_training_repair"],
                s=22,
                marker="o",
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=0.7,
                alpha=style["alpha"],
                zorder=2,
            )

            median_x = 100 * points["evaluation_view_gap"].median()
            median_y = 100 * points["right_training_repair"].median()
            axis.scatter(
                [median_x],
                [median_y],
                s=92,
                marker="o",
                facecolor=style["facecolor"],
                edgecolor="black",
                linewidth=1.35,
                zorder=5,
            )
            axis.scatter(
                [median_x],
                [median_y],
                s=27,
                marker="+",
                color=("black" if operator == "PERIODIC" else "white"),
                linewidth=1.15,
                zorder=6,
            )
            offset = median_offsets[(region, operator)]
            axis.annotate(
                f"{'Periodic' if operator == 'PERIODIC' else 'Swinging Door'} median\n"
                f"({median_x:.1f}, {median_y:.1f})%",
                xy=(median_x, median_y),
                xytext=offset,
                textcoords="offset points",
                ha="left",
                va=("top" if offset[1] < 0 else "bottom"),
                fontsize=7.1,
                color="0.12",
                arrowprops={
                    "arrowstyle": "-",
                    "color": style["edgecolor"],
                    "linewidth": 0.7,
                    "shrinkA": 1.5,
                    "shrinkB": 4.0,
                },
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": style["edgecolor"],
                    "linewidth": 0.55,
                    "alpha": 0.92,
                },
                zorder=7,
            )

        axis.set_xlim(0, x_max)
        axis.set_ylim(y_min, y_max)
        axis.text(
            0.97,
            0.95,
            "Shaded: both effects >1%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="0.35",
        )
        axis.set_title(
            f"TEP pressure, {6 if region.endswith('Q6') else 16} min",
            loc="left",
            fontsize=10.2,
        )
        axis.set_xlabel(r"Origin-view evaluation penalty, $100g_E$ (%)")
        axis.grid(color="0.92", linewidth=0.5)
        axis.set_axisbelow(True)

    axes[0].set_ylabel(r"Origin-view retraining contrast, $100g_R$ (%)")
    operator_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=operator_styles["PERIODIC"]["facecolor"],
            markeredgecolor=operator_styles["PERIODIC"]["edgecolor"],
            label=operator_styles["PERIODIC"]["label"],
            markersize=5.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=operator_styles["SWINGING_DOOR"]["facecolor"],
            markeredgecolor=operator_styles["SWINGING_DOOR"]["edgecolor"],
            label=operator_styles["SWINGING_DOOR"]["label"],
            markersize=5.5,
        ),
    ]
    fig.legend(handles=operator_handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=1.0)
    save_figure(fig, output / "lineage_remedy_effects_pressure")


def plot_remedy_categories(summary: pd.DataFrame, output: Path) -> None:
    """Appendix display of the exhaustive 1% categorical partition."""

    configure_plotting()
    summary = summary.loc[
        np.isclose(summary["margin"], 0.01) & summary["region"].ne("ALL")
    ].copy()
    row_order = [
        (region, operator)
        for region in LINEAGE_REGION_ORDER
        for operator in ("PERIODIC", "SWINGING_DOOR")
    ]
    y = np.arange(len(row_order), dtype=float)
    fig, axis = plt.subplots(figsize=(7.4, 6.4))
    left = np.zeros(len(row_order), dtype=float)
    colors = ("0.98", "0.12", "0.32", "0.52", "0.72", "0.90")
    hatches = ("..", "", "///", "xx", "\\\\", "++")
    for remedy, color, hatch in zip(REMEDY_ORDER, colors, hatches):
        values: list[float] = []
        counts: list[int] = []
        denominators: list[int] = []
        for region, operator in row_order:
            group = summary.loc[
                summary["region"].eq(region)
                & summary["operator"].eq(operator)
                & summary["remedy_class"].eq(remedy)
            ]
            values.append(100 * float(group["share"].iloc[0]) if len(group) else 0.0)
            counts.append(int(group["n_units"].iloc[0]) if len(group) else 0)
            denominators.append(int(group["total_units"].iloc[0]) if len(group) else 0)
        bars = axis.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            hatch=hatch,
            label=REMEDY_FIGURE_LABELS[remedy],
        )
        for bar, value, count, denominator in zip(
            bars, values, counts, denominators
        ):
            if count == 0:
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{count}",
                ha="center",
                va="center",
                fontsize=8,
                rotation=0,
                color="white" if float(color) < .6 else "black",
                bbox={
                    "boxstyle": "round,pad=0.10",
                    "facecolor": "0.10" if float(color) < .6 else "white",
                    "edgecolor": "none",
                    "alpha": 1.0,
                },
            )
        left += np.asarray(values)
    row_labels = [
        f"{LINEAGE_REGION_LABELS[region]}\n"
        f"{'Periodic' if operator == 'PERIODIC' else 'Swinging Door'}"
        f" ($N={int(summary.loc[summary.region.eq(region) & summary.operator.eq(operator), 'total_units'].iloc[0])}$)"
        for region, operator in row_order
    ]
    axis.set_yticks(y, row_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of Ridge/MAE design cells (%); labels show counts $n$")
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="0.90", linewidth=0.55)
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.008,
        "The same units recur at four retained fractions; bars summarize fixed design cells.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    save_figure(fig, output / "lineage_remedy_categories")



def main() -> None:
    for p in (OUT, TABLES, FIGURES): p.mkdir(parents=True, exist_ok=True)
    comparisons, geometry, remedies, payload = load_lineage_artifacts()
    geom = build_lineage_geometry_summary(geometry)
    events = build_event_commit_summary(comparisons)
    cells = build_remedy_cells(remedies)
    threshold = build_remedy_threshold_summary(cells)
    interaction = build_interaction_summary(cells)
    fault = build_fault_summary(cells)
    outputs = {"lineage_geometry_summary": geom, "event_commit_summary": events, "remedy_cells_augmented": cells, "remedy_threshold_summary": threshold, "remedy_interaction_summary": interaction, "remedy_fault_summary": fault}
    for name, frame in outputs.items(): frame.to_csv(OUT / f"{name}.csv", index=False)
    threshold.loc[np.isclose(threshold["margin"], .01)].to_csv(OUT / "remedy_summary.csv", index=False)
    write_lineage_geometry_table(geom, TABLES / "lineage_geometry_summary.tex")
    write_event_commit_table(events, TABLES / "lineage_event_commit_summary.tex")
    write_remedy_rule_table(TABLES / "lineage_remedy_rules.tex")
    write_remedy_threshold_table(threshold, TABLES / "lineage_remedy_summary.tex")
    write_interaction_summary_table(interaction, TABLES / "lineage_interaction_summary.tex")
    write_fault_summary_table(fault, TABLES / "lineage_fault_summary.tex")
    plot_lineage_geometry(geometry, FIGURES)
    plot_event_commit_effect(comparisons, FIGURES)
    plot_pressure_fault_map(comparisons, FIGURES)
    plot_remedy_effects(cells, FIGURES)
    plot_pressure_remedy_effects(cells, FIGURES)
    plot_remedy_categories(threshold, FIGURES)
    print(json.dumps({"comparison_rows":len(comparisons),"remedy_units":len(cells),"checks":payload["checks"]}, indent=2))

if __name__ == "__main__": main()
