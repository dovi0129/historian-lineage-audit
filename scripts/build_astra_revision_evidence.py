#!/usr/bin/env python3
"""Post-process frozen V3/V4 losses for the PDF-only reviewer revision; no fitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER
SOURCE = ROOT / "artifacts/historian_reconstruction_lineage_tep_full_panel_breadth"
OUT = PAPER / "artifacts/astra_revision"
TABLES = PAPER / "tables/generated"
FIGURES = PAPER / "figures/generated"
KEY = ["process", "region", "target", "horizon_minutes", "seed", "fold", "fault", "operator", "severity_k", "score"]
LOSSES = {
    "L33": "loss_left_train_left_eval", "L34": "loss_left_train_right_eval",
    "L43": "loss_right_train_left_eval", "L44": "loss_right_train_right_eval",
}
WINNERS = {
    "L33": "winner_left_train_left_eval", "L34": "winner_left_train_right_eval",
    "L43": "winner_right_train_left_eval", "L44": "winner_right_train_right_eval",
}
REGIONS = ["CSTR_OUTPUT_Q1", "PRESSURE_Q6", "PRESSURE_Q16", "TEMPERATURE_Q6", "TEMPERATURE_Q16"]
LABEL = dict(zip(REGIONS, ["CSTR, 1 min", "Pressure, 6 min", "Pressure, 16 min", "Temperature, 6 min", "Temperature, 16 min"]))
MODEL = {"PERSISTENCE": "Persistence", "RIDGE_AR": "Ridge", "RIDGE_ARX": "Ridge", "SHALLOW_GBR": "Tree", "SHALLOW_HGB": "Tree"}
RANK = {"PERSISTENCE": 0, "RIDGE_AR": 1, "RIDGE_ARX": 1, "SHALLOW_GBR": 2, "SHALLOW_HGB": 2}


def table(name: str, columns: str, header: str, rows: list[str]) -> None:
    (TABLES / name).write_text(
        "\\begin{tabular}{@{}" + columns + "@{}}\n\\toprule\n" + header
        + " \\\\\n\\midrule\n" + "\n".join(r + r" \\" for r in rows)
        + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8",
    )


def load_cells() -> pd.DataFrame:
    x = pd.read_csv(SOURCE / "view_comparison_cells.csv")
    x = x[x.comparison.eq("EVENT_TIME_VS_EVENT_COMMIT")].copy()
    assert len(x) == 3840 and not x.duplicated(KEY + ["model"]).any()
    assert x.groupby("score").size().to_dict() == {"MAE": 1920, "RMSE": 1920}
    assert set(x.region) == set(REGIONS)
    for col, source in LOSSES.items():
        x[col] = x[source]
    assert np.isfinite(x[list(LOSSES)].to_numpy()).all() and (x[list(LOSSES)] > 0).all().all()
    x["d_eval"] = x.L34 / x.L33 - 1
    x["d_train"] = 1 - x.L44 / x.L34
    x["d_total"] = x.L44 / x.L33 - 1
    for new, old in [("d_eval", "evaluation_view_gap"), ("d_train", "right_training_repair"), ("d_total", "diagonal_gap")]:
        np.testing.assert_allclose(x[new], x[old], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(1+x.d_total, (1+x.d_eval)*(1-x.d_train), rtol=1e-8, atol=1e-12)
    periodic = x[x.operator.eq("PERIODIC")]
    np.testing.assert_allclose(periodic[["d_eval", "d_train", "d_total"]], 0, atol=1e-12)
    for loss in LOSSES:
        x[f"normalized_{loss}"] = 100*x[loss]/x.L33
    return x


def selection_cells(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in x.groupby(KEY, sort=True):
        assert len(frame) == 3 and set(frame.model.map(RANK)) == {0, 1, 2}
        frame = frame.assign(priority=frame.model.map(RANK)).sort_values("priority").set_index("model")
        w = {col: frame[col].idxmin() for col in LOSSES}
        for col, saved in WINNERS.items():
            assert frame[saved].eq(w[col]).all(), (key, col)
        margins = {}
        for col in ("L33", "L34"):
            losses = frame[col].sort_values(kind="stable").to_numpy()
            margins[col] = (losses[1]-losses[0])/losses[0]
        changed = w["L33"] != w["L34"]
        qualified = changed and max(margins.values()) >= .01
        assert frame.practical_evaluation_winner_change_1pct.eq(int(qualified)).all(), key
        old, best = float(frame.loc[w["L33"], "L34"]), float(frame.loc[w["L34"], "L34"])
        refit_old, refit_best = float(frame.loc[w["L33"], "L44"]), float(frame.loc[w["L44"], "L44"])
        rows.append(dict(zip(KEY, key)) | {
            "winner_V3": w["L33"], "winner_V4_fixed": w["L34"], "winner_V4_refit": w["L44"],
            "changed": int(changed), "margin_qualified_change": int(qualified),
            "margin_V3": margins["L33"], "margin_V4": margins["L34"],
            "L33_old": float(frame.loc[w["L33"], "L33"]), "L34_old": old, "L34_best": best,
            "excess_abs": old-best, "excess_rel": old/best-1,
            "L44_old_family": refit_old, "L44_best": refit_best,
            "refit_excess_abs": refit_old-refit_best, "refit_excess_rel": refit_old/refit_best-1,
        })
    result = pd.DataFrame(rows)
    assert len(result) == 1280 and not result.duplicated(KEY).any()
    assert result.excess_rel.ge(0).all() and result.refit_excess_rel.ge(0).all()
    return result


def summaries(x: pd.DataFrame, choices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    grouping = ["region", "operator", "score", "model"]
    for groups in (grouping, grouping + ["severity_k"]):
        for key, frame in x.groupby(groups):
            r = dict(zip(groups, key)) | {"aggregation": "all_severities" if len(groups)==4 else "by_severity", "n_cells": len(frame)}
            for col in list(LOSSES) + ["d_eval", "d_train", "d_total"]:
                r[col] = float(frame[col].median())
                r[col+"_q25"] = float(frame[col].quantile(.25))
                r[col+"_q75"] = float(frame[col].quantile(.75))
            records.append(r)
    paths = pd.DataFrame(records)
    records = []
    for key, frame in choices.groupby(["region", "operator", "score"]):
        c = frame[frame.margin_qualified_change.eq(1)]
        records.append(dict(zip(["region", "operator", "score"], key)) | {
            "n_cells": len(frame), "n_changed": int(frame.changed.sum()), "n_qualified": len(c),
            "n_qualified_excess_below_1pct": int(c.excess_rel.lt(.01).sum()),
            "n_unique_fold_fault": len(c[["fold", "fault"]].drop_duplicates()) if key[0] != "CSTR_OUTPUT_Q1" else 0,
            "qualified_excess_min": c.excess_rel.min(), "qualified_excess_median": c.excess_rel.median(),
            "qualified_excess_max": c.excess_rel.max(),
        })
    return paths, pd.DataFrame(records)


def write_tables(paths: pd.DataFrame, choices: pd.DataFrame, selection: pd.DataFrame) -> None:
    pooled = paths[paths.aggregation.eq("all_severities") & paths.operator.eq("SWINGING_DOOR")]
    rows = []
    for region in REGIONS:
        for model in ["Persistence", "Ridge", "Tree"]:
            r = pooled[pooled.region.eq(region) & pooled.score.eq("MAE") & pooled.model.map(MODEL).eq(model)].iloc[0]
            rows.append(f"{LABEL[region]} & {model} & {int(r.n_cells)} & " + " & ".join(f"{r[c]:.4f}" for c in LOSSES))
    table("astra_clock_losses.tex", r"llr@{\hspace{1em}}r@{\hspace{1em}}r@{\hspace{1em}}r@{\hspace{1em}}r", r"Task & Model & $N$ & $L_{33}$ & $L_{34}$ & $L_{43}$ & $L_{44}$", rows)
    rows = []
    for region in REGIONS:
        for score in ("MAE", "RMSE"):
            r = pooled[pooled.region.eq(region) & pooled.score.eq(score) & pooled.model.map(MODEL).eq("Ridge")].iloc[0]
            entries = [f"{100*r[c]:+.2f} [{100*r[c+'_q25']:+.2f}, {100*r[c+'_q75']:+.2f}]" for c in ("d_eval", "d_train", "d_total")]
            rows.append(f"{LABEL[region]} & {score} & {int(r.n_cells)} & " + " & ".join(entries))
    table("astra_clock_contrasts.tex", "llrrrr", r"Task & Score & $N$ & $100d_E$ [IQR] & $100d_T$ [IQR] & $100d_D$ [IQR]", rows)
    rows = []
    for region in REGIONS:
        for score in ("MAE", "RMSE"):
            r = selection[selection.region.eq(region) & selection.score.eq(score) & selection.operator.eq("SWINGING_DOOR")].iloc[0]
            interval = "--" if r.n_qualified == 0 else f"{100*r.qualified_excess_min:.2f}--{100*r.qualified_excess_max:.2f}"
            rows.append(f"{LABEL[region]} & {score} & {int(r.n_cells)} & {int(r.n_changed)} & {int(r.n_qualified)} & {int(r.n_qualified_excess_below_1pct)} & {interval}")
    table("astra_selection_summary.tex", "llrrrrr", r"Task & Score & $N$ & Changed & Margin-qualified & $e<1\%$ & $100e$ range", rows)
    selected = choices[choices.region.isin(["PRESSURE_Q6", "PRESSURE_Q16"]) & choices.operator.eq("SWINGING_DOOR") & choices.score.eq("MAE") & choices.margin_qualified_change.eq(1)].sort_values(["horizon_minutes", "fold", "fault", "severity_k"])
    assert selected.groupby("region").size().to_dict() == {"PRESSURE_Q16": 9, "PRESSURE_Q6": 9}
    assert selected.winner_V3.eq("RIDGE_ARX").all() and selected.winner_V4_fixed.eq("SHALLOW_HGB").all()
    rows = [f"{int(r.horizon_minutes)} & {int(r.fold)} & {int(r.fault)} & {int(r.severity_k)} & {r.L33_old:.4f} & {r.L34_old:.4f} & {r.L34_best:.4f} & {r.excess_abs:.4f} & {100*r.excess_rel:.2f} & {100*r.margin_V3:.2f} & {100*r.margin_V4:.2f}" for r in selected.itertuples()]
    table("astra_pressure_switches.tex", "rrrrrrrrrrr", r"$h$ & Fold & Fault & $k$ & $L_{33}$(Ridge) & $L_{34}$(Ridge) & $L_{34}$(Tree) & Excess & $100e$ & Margin V3 & Margin V4", rows)


def scope_and_channel_tables() -> None:
    x = pd.read_csv(PAPER / "artifacts/stage1_strengthening/feature_scope_summary.csv")
    scopes = ["target current", "target history", "all-channel current", "full lag window"]
    rows = []
    for region in REGIONS[1:]:
        for scope in scopes:
            s = x[x.region.eq(region) & x.feature_scope.eq(scope)].set_index("severity_k")
            rows.append(f"{LABEL[region]} & {scope.capitalize()} & " + " & ".join([
                f"{100*s.loc[2,'median_strict_failure_share']:.0f}/{100*s.loc[8,'median_strict_failure_share']:.0f}",
                f"{100*s.loc[2,'median_active_failure_share']:.1f}/{100*s.loc[8,'median_active_failure_share']:.1f}",
                f"{s.loc[2,'median_strict_erosion_minutes']:.1f}/{s.loc[8,'median_strict_erosion_minutes']:.1f}",
            ]))
    table("astra_feature_scopes.tex", "llrrr", r"Task & Scope & Strict failure (\%) & Active failure (\%) & Strict erosion (min)", rows)
    x = pd.read_csv(SOURCE / "geometry_differentiation.csv")
    x = x[x.operator.eq("SWINGING_DOOR") & x.region.isin(["PRESSURE_Q6", "PRESSURE_Q16"])]
    counts = x.groupby(["top_strict_availability_channel", "region"]).size().unstack(fill_value=0)
    counts = counts.sort_values("PRESSURE_Q6", ascending=False, kind="stable")
    rows = [f"{channel} & {int(r.PRESSURE_Q6)}/72 & {int(r.PRESSURE_Q16)}/72" for channel,r in counts.iterrows()]
    table("astra_limiting_channels.tex", "lrr", "Most frequent limiting channel & 6-min geometry & 16-min geometry", rows)
    metadata = json.loads((PAPER / "artifacts/stage1_strengthening/worked_example_metadata.json").read_text())
    table("astra_worked_values.tex", "lr", "At origin 1498 & XMEAS-7 value", [
        f"Event-time replay (V3) & {metadata['event_time_value']:.4f}",
        f"Availability-aware replay (V4) & {metadata['commit_time_value']:.4f}",
        f"Absolute difference & {metadata['absolute_value_difference']:.4f}",
    ])


def plot_pressure_path(x: pd.DataFrame, choices: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family":"serif", "font.serif":["DejaVu Serif"], "font.size":9, "axes.labelsize":9, "xtick.labelsize":8.4, "ytick.labelsize":8.1, "pdf.fonttype":42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), gridspec_kw={"width_ratios":[1, 1.22]})
    styles = [("#193f59", "o", "-"), ("#a34625", "s", "--"), ("#487452", "^", "-."), ("#6b527d", "D", ":")]
    for row, region in enumerate(["PRESSURE_Q6", "PRESSURE_Q16"]):
        ax, bx = axes[row]
        s = x[x.region.eq(region) & x.operator.eq("SWINGING_DOOR") & x.score.eq("MAE") & x.model.eq("RIDGE_ARX")]
        for k, (color, marker, style) in zip([2,3,5,8], styles):
            g = s[s.severity_k.eq(k)]
            ax.plot([0,1,2], g[["normalized_L33","normalized_L34","normalized_L44"]].median(), color=color, marker=marker, linestyle=style, lw=1.4, ms=4, label=f"$k={k}$")
        ax.axhline(100, color=".65", lw=.6)
        ax.set_xticks([0,1,2], ["Original\nV3/V3", "Correct eval.\nV3/V4", "Retrain\nV4/V4"])
        ax.set_ylabel("Ridge MAE\n(each cell's original score = 100)")
        ax.set_xlim(-.12,2.15)
        ax.set_ylim(99.5,108)
        ax.set_title(f"({'ac'[row]}) Pressure, {6 if row==0 else 16} min", loc="left", fontsize=9.5, pad=8)
        c = choices[choices.region.eq(region) & choices.operator.eq("SWINGING_DOOR") & choices.score.eq("MAE") & choices.margin_qualified_change.eq(1)].sort_values(["fault","fold","severity_k"])
        if row == 0:
            positions = np.flatnonzero(c.fault.to_numpy() == 14)
            if positions.size:
                bx.axhspan(positions.min()-.45, positions.max()+.45, color=".93", zorder=0)
        for i, r in enumerate(c.itertuples()):
            color, marker, _ = styles[[2,3,5,8].index(int(r.severity_k))]
            bx.plot([0,100*r.excess_rel],[i,i],color=".80",lw=.8,zorder=1)
            bx.scatter(100*r.excess_rel, i, c=color, marker=marker, s=28, zorder=2)
            if row == 0 and r.excess_rel < .01:
                bx.annotate(f"{100*r.excess_rel:.2f}%", (100*r.excess_rel, i),
                            xytext=(7, 3), textcoords="offset points", fontsize=8.1,
                            bbox={"facecolor":"white", "edgecolor":"none", "pad":.5})
        bx.set_yticks(range(len(c)), [f"Fault {r.fault}, fold {r.fold}, k={r.severity_k}" for r in c.itertuples()])
        bx.invert_yaxis()
        bx.set_ylim(len(c)-.4, -1.3)
        bx.axvline(1,color=".35",ls="--",lw=.8)
        bx.set_xlim(-1,50) if row==0 else bx.set_xlim(-.06,3.1)
        bx.set_xlabel("Excess MAE from keeping Ridge (%)\n" +
                      ("0–50% scale; shaded rows: fault 14" if row == 0 else "0–3% detail scale; dashed: 1%"))
        bx.set_title(f"({'bd'[row]}) Model choice: 9/72 changes",loc="left",fontsize=9.5,pad=8)
        bx.text(.98,.98,"V3-trained models held fixed",transform=bx.transAxes,
                ha="right", va="top", fontsize=8.0)
        for a in (ax,bx):
            a.spines[["top","right"]].set_visible(False)
            a.grid(axis="x" if a is bx else "y", color=".92", lw=.5)
            a.set_axisbelow(True)
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.tight_layout(rect=(0,0,1,.96),h_pad=2.1,w_pad=2)
    for extension in ("pdf","png"):
        fig.savefig(FIGURES/f"astra_pressure_decisions.{extension}",dpi=320,bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path in (OUT,TABLES,FIGURES): path.mkdir(parents=True,exist_ok=True)
    x=load_cells()
    choices=selection_cells(x)
    paths, selections=summaries(x,choices)
    for frame,name in [(x,"clock_path_cells"),(paths,"clock_path_summary"),(choices,"selection_consequence_cells"),(selections,"selection_consequence_summary")]:
        frame.to_csv(OUT/f"{name}.csv",index=False)
    write_tables(paths,choices,selections)
    scope_and_channel_tables()
    plot_pressure_path(x,choices)
    checks={"source":str((SOURCE/"view_comparison_cells.csv").relative_to(PAPER)),"comparison":"EVENT_TIME_VS_EVENT_COMMIT","n_model_score_cells":len(x),"n_selection_score_cells":len(choices),"scores":["MAE","RMSE"],"four_losses_positive_finite":True,"unique_keys":True,"stored_contrasts_match":True,"cell_path_identity":True,"stored_winners_and_margin_flags_match":True,"periodic_zero_clock_contrast":True,"pressure_MAE_qualified_changes_each":9,"model_refits":0,"claim_boundary":"Post-selection descriptive comparisons on the same held-out cells; no validated selection policy."}
    (OUT/"checks.json").write_text(json.dumps(checks,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(checks,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
