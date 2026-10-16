#!/usr/bin/env python
"""Audit runs of the observed-availability testbed (spec §8, BRIEF_audit.md).

Usage:
  run_audit.py runs/<run_id> [runs/<run_id> ...] [--out DIR]
  run_audit.py --all runs/ [--out DIR] [--jobs N]

Per run, DIR/<run_id>/ gets summary.json, summary.md, ledger.csv, violations.csv, mismatches_primary.csv (the two
comparisons expected to match, with full row sets) and mismatches_all.csv.gz (every comparison, differing rows).
DIR/pooled/ gets the same summaries across runs. Runs without a DONE marker are skipped and listed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obsaudit.io_run import Issues, RunFormatError, load_ledger_inputs, load_truth_inputs, run_is_complete  # noqa: E402
from obsaudit.ledger import build_ledger, commit_lag_summary, retention_summary  # noqa: E402
from obsaudit.overlap import sdt_overlap  # noqa: E402
from obsaudit.reconstruct import LAGS, MIN_ORIGIN, TOL, WINDOW  # noqa: E402
from obsaudit.truth import BEFORE_FIRST_FLUSH, UNDETERMINED, build_truth  # noqa: E402
from obsaudit.views import COMPARISONS, FULL_ROWS, negative_check_insert, negative_check_shift, run_comparisons  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "results"
META_FIELDS = ("run_id", "fault", "batch", "n_ticks", "channels", "deadband", "compdev", "flush_every",
               "iotdb_version", "retention_level", "target_retention")


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def audit_run(run_dir: Path, out_root: Path) -> dict:
    run_dir = Path(run_dir)
    started = time.time()
    issues = Issues()
    summary: dict = {"run_dir": str(run_dir), "run_id": run_dir.name}
    if not run_is_complete(run_dir):
        summary["status"] = "incomplete"
        return summary
    try:
        ledger_inp = load_ledger_inputs(run_dir, issues)  # never opens snapshots.csv
        truth_inp = load_truth_inputs(run_dir, issues)    # never opens samples.csv or membership files
    except RunFormatError as exc:
        summary.update(status="unreadable", error=str(exc), validation=issues.items)
        return summary
    meta = ledger_inp.meta
    ledger = build_ledger(ledger_inp, issues)
    truth = build_truth(truth_inp, issues)
    comp = run_comparisons(ledger, truth, meta)
    neg_a = negative_check_shift(comp, truth, meta)
    neg_b = negative_check_insert(comp, truth, meta)
    n_ticks = int(meta["n_ticks"])
    origins = range(MIN_ORIGIN, n_ticks)
    first_flush = ledger_inp.schedule.regular_ticks[0] if ledger_inp.schedule.regular_ticks else None
    statuses = [truth.cells[(o, ch)].status for o in origins for ch in meta["channels"]]
    final_rows = ledger.rows.loc[ledger.rows["final_retained"]]
    summary.update({
        "meta": {k: meta.get(k) for k in META_FIELDS if k in meta},
        "definitions": {"window": WINDOW, "lags": list(LAGS), "min_origin": MIN_ORIGIN, "tolerance": TOL,
                        "final_flush_tick": n_ticks},
        "validation": {"n_errors": issues.n_errors, "n_warnings": len(issues.items) - issues.n_errors,
                       "issues": issues.items, "n_memberships": ledger.n_memberships,
                       "n_digests_checked": ledger.n_digests_checked},
        "origins": {"first": MIN_ORIGIN, "last": n_ticks - 1, "n": len(origins), "first_flush_tick": first_flush,
                    "n_before_first_flush": sum(1 for o in origins if first_flush is None or o < first_flush),
                    "n_cells_before_first_flush": statuses.count(BEFORE_FIRST_FLUSH),
                    "n_cells_truth_undetermined": statuses.count(UNDETERMINED),
                    "n_commit_anchor_rows": truth.n_commit_anchor_rows},
        "ledger": {"retention": retention_summary(ledger),
                   "mean_achieved_retention": float(ledger.rows.groupby("channel")["final_retained"].mean().mean()),
                   "commit_lag": commit_lag_summary(ledger),
                   "n_violations": len(ledger.violations), "n_anomalies": len(ledger.anomalies),
                   "violations": ledger.violations[:50], "anomalies": ledger.anomalies[:50]},
        "truth": {"n_violations": len(truth.violations), "violations": truth.violations[:50]},
        "comparisons": {name: comp.tallies[name].summary() for name, _, _ in COMPARISONS},
        "v_server_vs_v3": {"n_cells_equal": comp.v_server_equals_v3_cells,
                           "n_cells_different": comp.v_server_differs_v3_cells,
                           "equal_everywhere": comp.v_server_differs_v3_cells == 0,
                           "n_final_rows_server_tick_ne_tick": int((final_rows["server_tick"] != final_rows["tick"]).sum())},
        "negative_checks": {"a_shift_commit_of_used_endpoint": neg_a, "b_insert_row": neg_b},
        "duplicates": {"ledger_membership": ledger.duplicates, "snapshot_rows": truth.duplicates,
                       "testbed_reported": meta.get("duplicate_rows_observed"),
                       "window_multiplicity_ledger_vs_snapshot": comp.multiplicity},
        "sdt_overlap": sdt_overlap(ledger.rows, meta),
        "audit_seconds": round(time.time() - started, 2),
    })
    summary = _clean(summary)
    summary.update(decide(summary))
    out_dir = Path(out_root) / meta["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = pd.DataFrame(comp.mismatches, columns=_MM_COLS)
    mm.loc[mm["comparison"].isin(FULL_ROWS)].to_csv(out_dir / "mismatches_primary.csv", index=False)
    mm.to_csv(out_dir / "mismatches_all.csv.gz", index=False, compression="gzip")
    ledger.rows.to_csv(out_dir / "ledger.csv", index=False)
    viol = [{"source": "ledger", **v} for v in ledger.violations + ledger.anomalies] + \
           [{"source": "truth", **v} for v in truth.violations]
    pd.DataFrame(viol).to_csv(out_dir / "violations.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "summary.md").write_text(run_markdown(summary), encoding="utf-8")
    return summary


def _result(c: dict) -> dict:
    if c["n_cells_evaluated"] == 0:
        result = "not_evaluable"
    else:
        result = "mismatch" if c["n_cells_any_mismatch"] else "match"
    return {"result": result, "n_cells_evaluated": c["n_cells_evaluated"],
            "n_cells_mismatch": c["n_cells_any_mismatch"], "n_cells_rowset_mismatch": c["n_cells_rowset_mismatch"],
            "n_feature_mismatch": c["n_feature_mismatch"], "n_features": c["n_features"]}


def decide(s: dict) -> dict:
    """Run gate (spec §6/§8 and code audit finding 1): validity is separate from the scientific result.

    Invalid (implementation or testbed defect; the run cannot test the claim): validation errors (format, digest,
    unscheduled flush, same-time different-value duplicates), committed rows removed or changed (membership or
    snapshot), unexplained exclusions (truth-undetermined cells), and applicable negative checks that failed.
    Pre-first-flush origins are a declared exclusion. A primary or visible mismatch in a valid run is a result.
    """
    reasons = []
    if s["validation"]["n_errors"]:
        checks = sorted({i["check"] for i in s["validation"]["issues"] if i["level"] == "error"})
        reasons.append(f"validation errors: {', '.join(checks)}")
    if s["ledger"]["n_violations"]:
        kinds = sorted({v["type"] for v in s["ledger"]["violations"]})
        reasons.append(f"{s['ledger']['n_violations']} ledger violations ({', '.join(kinds)})")
    if s["truth"]["n_violations"]:
        reasons.append(f"{s['truth']['n_violations']} snapshot persistence violations")
    if s["origins"]["n_cells_truth_undetermined"]:
        reasons.append(f"{s['origins']['n_cells_truth_undetermined']} truth-undetermined cells")
    neg = s["negative_checks"]
    failed = [name for name, r in (("a", neg["a_shift_commit_of_used_endpoint"]),
                                   ("b_future_time", neg["b_insert_row"]["future_time"]),
                                   ("b_future_commit", neg["b_insert_row"]["future_commit"]))
              if r["status"] == "NOT_DETECTED"]
    if failed:
        reasons.append(f"negative checks not detected: {', '.join(failed)}")
    primary = _result(s["comparisons"]["Replay_commit_vs_S_commit"])
    visible = _result(s["comparisons"]["Replay_visible_vs_S_all"])
    if reasons:
        status = "invalid"
    elif primary["result"] == "not_evaluable":
        status, reasons = "invalid", ["primary comparison has no evaluable cells"]
    else:
        status = "mismatch" if "mismatch" in (primary["result"], visible["result"]) else "ok"
    return {"status": status, "status_valid": "invalid" if status == "invalid" else "valid",
            "invalid_reasons": reasons, "primary_result": primary, "visible_result": visible}


def status_line(s: dict) -> str:
    if "primary_result" not in s:
        return f"{s['status']}" + (f" ({s['error']})" if s.get("error") else "")
    p, v = s["primary_result"], s["visible_result"]
    text = (f"{s['status']} [{s['status_valid']}; primary {p['result']} {p['n_cells_mismatch']}/{p['n_cells_evaluated']}"
            f" cells; visible {v['result']} {v['n_cells_mismatch']}/{v['n_cells_evaluated']} cells]")
    return text + (f" invalid: {'; '.join(s['invalid_reasons'])}" if s["invalid_reasons"] else "")


_MM_COLS = ["run_id", "comparison", "origin", "channel", "t_f", "rowset_mismatch", "n_feature_mismatch",
            "mismatched_lags", "features_view", "features_truth", "max_abs_diff", "rows_only_view",
            "rows_only_truth", "rows_view", "rows_truth"]


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.2f}%"


def _neg_status(s: dict) -> str:
    a = s["negative_checks"]["a_shift_commit_of_used_endpoint"]["status"]
    b = s["negative_checks"]["b_insert_row"]
    return f"{a} / {b['future_time']['status']} / {b['future_commit']['status']}"


def run_markdown(s: dict) -> str:
    lines = [f"# Audit summary: {s['run_id']}", "",
             f"Status: **{status_line(s)}**", "",
             f"Validation errors {s['validation']['n_errors']}, warnings {s['validation']['n_warnings']}; digests "
             f"checked {s['validation']['n_digests_checked']}.", "",
             f"Origins {s['origins']['first']}..{s['origins']['last']} ({s['origins']['n']}); first flush tick "
             f"{s['origins']['first_flush_tick']}; origins before first flush {s['origins']['n_before_first_flush']}; "
             f"truth-undetermined cells {s['origins']['n_cells_truth_undetermined']}.", "",
             "| comparison | cells | excluded (pre-flush / undet.) | row-set mm | feature mm cells | cell mm rate | "
             "max abs diff | longest run |", "|---|---|---|---|---|---|---|---|"]
    for name, c in s["comparisons"].items():
        lines.append(f"| {name} | {c['n_cells_evaluated']} | {c['n_cells_excluded_before_first_flush']} / "
                     f"{c['n_cells_excluded_truth_undetermined']} | {c['n_cells_rowset_mismatch']} | "
                     f"{c['n_cells_feature_mismatch']} | {_pct(c['cell_mismatch_rate'])} | "
                     f"{c['abs_diff_both_present'].get('max', '–')} | {c['mismatch_run_lengths_origins'].get('max', '–')} |")
    lag = s["ledger"]["commit_lag"]
    lines += ["", f"Achieved retention (mean over channels): {s['ledger']['mean_achieved_retention']:.4f}.",
              f"Commit lag c_obs − τ: p50 {lag.get('p50', '–')}, p90 {lag.get('p90', '–')}, max {lag.get('max', '–')} "
              f"(n {lag['n_committed_regular']}; committed only at final flush {lag['n_committed_at_final_flush']}).",
              f"Ledger violations {s['ledger']['n_violations']}, anomalies {s['ledger']['n_anomalies']}; "
              f"snapshot violations {s['truth']['n_violations']}.",
              f"V_server equals V3 everywhere: {s['v_server_vs_v3']['equal_everywhere']}.",
              f"Duplicate rows (identical copies, collapsed for R): membership rows stored more than once "
              f"{s['duplicates']['ledger_membership']['n_rows_stored_more_than_once']}; window multiplicity agreement "
              f"{s['duplicates']['window_multiplicity_ledger_vs_snapshot']['n_cells_agree']}/"
              f"{s['duplicates']['window_multiplicity_ledger_vs_snapshot']['n_cells']} cells.",
              f"Negative checks (a shift / b future-time / b future-commit): {_neg_status(s)}."]
    ov = {ch: o.get("jaccard") for ch, o in s["sdt_overlap"]["channels"].items() if "jaccard" in o}
    if ov:
        lines.append("SDT overlap (Jaccard, IoTDB vs whole-series local rule; IoTDB resets SDT at every FLUSH): " +
                     ", ".join(f"{ch} {v:.3f}" if v is not None else f"{ch} –" for ch, v in ov.items()) + ".")
    return "\n".join(lines) + "\n"


def pooled(summaries: list[dict], out_root: Path) -> None:
    out = Path(out_root) / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    done = [s for s in summaries if "primary_result" in s]
    groups: dict[str, dict] = {}
    for s in done:
        key = f"deadband={s['meta'].get('deadband')},F={s['meta'].get('flush_every')}"
        g = groups.setdefault(key, {"runs": [], **{name: {"cells": 0, "any": 0, "features": 0, "feature_mm": 0}
                                                    for name, _, _ in COMPARISONS}})
        g["runs"].append(s["run_id"])
        for name, _, _ in COMPARISONS:
            c = s["comparisons"][name]
            g[name]["cells"] += c["n_cells_evaluated"]
            g[name]["any"] += c["n_cells_any_mismatch"]
            g[name]["features"] += c["n_features"]
            g[name]["feature_mm"] += c["n_feature_mismatch"]
    for g in groups.values():
        for name, _, _ in COMPARISONS:
            c = g[name]
            c["cell_mismatch_rate"] = c["any"] / c["cells"] if c["cells"] else None
            c["feature_mismatch_rate"] = c["feature_mm"] / c["features"] if c["features"] else None
    pooled_summary = {
        "n_runs_audited": len(done),
        "n_runs_ok": sum(s["status"] == "ok" for s in done),
        "n_runs_invalid": sum(s["status"] == "invalid" for s in done),
        "runs_invalid": {s["run_id"]: s["invalid_reasons"] for s in done if s["status"] == "invalid"},
        "runs_primary_mismatch": [s["run_id"] for s in done if s["primary_result"]["result"] == "mismatch"],
        "runs_visible_mismatch": [s["run_id"] for s in done if s["visible_result"]["result"] == "mismatch"],
        "runs_not_audited": [{"run_id": s["run_id"], "status": s["status"], "error": s.get("error")}
                             for s in summaries if s not in done],
        "primary_mismatch_cells_total": sum(s["comparisons"]["Replay_commit_vs_S_commit"]["n_cells_any_mismatch"]
                                            for s in done),
        "visible_mismatch_cells_total": sum(s["comparisons"]["Replay_visible_vs_S_all"]["n_cells_any_mismatch"]
                                            for s in done),
        "by_deadband_and_F": groups,
        "runs": [{"run_id": s["run_id"], "status": s["status"], "status_valid": s["status_valid"],
                  "invalid_reasons": s["invalid_reasons"], "primary_result": s["primary_result"],
                  "visible_result": s["visible_result"], "meta": s["meta"], "origins": s["origins"],
                  "mean_achieved_retention": s["ledger"]["mean_achieved_retention"],
                  "commit_lag": {k: v for k, v in s["ledger"]["commit_lag"].items() if k != "histogram"},
                  "violations": s["ledger"]["n_violations"] + s["truth"]["n_violations"],
                  "comparisons": {n: {k: c[k] for k in ("n_cells_evaluated", "n_cells_any_mismatch",
                                                         "cell_mismatch_rate", "n_feature_mismatch")}
                                  for n, c in s["comparisons"].items()},
                  "v_server_equals_v3": s["v_server_vs_v3"]["equal_everywhere"],
                  "negative_checks": _neg_status(s)} for s in done],
    }
    (out / "summary.json").write_text(json.dumps(_clean(pooled_summary), indent=1), encoding="utf-8")
    head = ("| run | status | valid | deadband | F | retention | pre-flush origins | undet. cells | primary mm | visible mm | "
            "V3 | V_server | V1 | S_all≠S_commit | lag p50/max | violations | neg a/b1/b2 |")
    rows = [head, "|" + "---|" * (head.count("|") - 1)]
    for s in done:
        c = s["comparisons"]
        lag = s["ledger"]["commit_lag"]
        rows.append(
            f"| {s['run_id']} | {s['status']} | {s['status_valid']} | {s['meta'].get('deadband')} | {s['meta'].get('flush_every')} | "
            f"{s['ledger']['mean_achieved_retention']:.3f} | {s['origins']['n_before_first_flush']} | "
            f"{s['origins']['n_cells_truth_undetermined']} | "
            f"{c['Replay_commit_vs_S_commit']['n_cells_any_mismatch']} | "
            f"{c['Replay_visible_vs_S_all']['n_cells_any_mismatch']} | "
            f"{_pct(c['V3_vs_S_commit']['cell_mismatch_rate'])} | {_pct(c['V_server_vs_S_commit']['cell_mismatch_rate'])} | "
            f"{_pct(c['V1_vs_S_commit']['cell_mismatch_rate'])} | {_pct(c['S_all_vs_S_commit']['cell_mismatch_rate'])} | "
            f"{lag.get('p50', '–')}/{lag.get('max', '–')} | {s['ledger']['n_violations'] + s['truth']['n_violations']} | "
            f"{_neg_status(s)} |")
    skipped = pooled_summary["runs_not_audited"]
    note = ["", "Mismatch columns: primary/visible = mismatching (origin, channel) cells; other columns = share of "
                "evaluated cells with a row-set or feature mismatch."]
    note.append("Status: ok = valid and primary/visible match; mismatch = valid run whose primary or visible "
                "comparison differs (a result, reported in full); invalid = the run cannot test the claim.")
    for s in done:
        if s["status"] == "invalid":
            note.append(f"- {s['run_id']} invalid: {'; '.join(s['invalid_reasons'])}")
    if skipped:
        note.append("Not audited: " + ", ".join(f"{r['run_id']} ({r['status']})" for r in skipped) + ".")
    (out / "summary.md").write_text("# Pooled audit summary\n\n" + "\n".join(rows + note) + "\n", encoding="utf-8")
    frames = []
    for s in done:
        path = Path(out_root) / s["meta"]["run_id"] / "mismatches_primary.csv"
        if path.is_file() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    (pd.concat(frames) if frames else pd.DataFrame(columns=_MM_COLS)).to_csv(out / "mismatches_primary.csv", index=False)


def _safe_audit(run_dir: Path, out_root: Path) -> dict:
    try:
        return audit_run(run_dir, out_root)
    except Exception as exc:  # report and continue with the other runs
        return {"run_dir": str(run_dir), "run_id": Path(run_dir).name, "status": "crashed",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--all", type=Path, metavar="RUNS_DIR", help="audit every run directory under RUNS_DIR")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)
    run_dirs = list(args.runs)
    if args.all:
        run_dirs += sorted(p for p in args.all.iterdir() if p.is_dir() and (p / "run_meta.json").is_file())
    if not run_dirs:
        ap.error("no run directories given")
    summaries = []
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(_safe_audit, d, args.out): d for d in run_dirs}
            for k, fut in enumerate(as_completed(futures), 1):
                s = fut.result()
                summaries.append(s)
                print(f"[{k}/{len(run_dirs)}] {s['run_id']}: {status_line(s)}", flush=True)
    else:
        for k, d in enumerate(run_dirs, 1):
            s = _safe_audit(d, args.out)
            summaries.append(s)
            print(f"[{k}/{len(run_dirs)}] {s['run_id']}: {status_line(s)}", flush=True)
    summaries.sort(key=lambda s: s["run_id"])
    for s in summaries:
        if s["status"] == "crashed":
            print(s["traceback"], file=sys.stderr)
    if args.all or len(run_dirs) > 1:
        pooled(summaries, args.out)
    # A valid mismatch is a result and exits 0; anything that cannot test the claim exits nonzero.
    bad = [s for s in summaries if s["status"] not in ("ok", "mismatch")]
    mism = [s["run_id"] for s in summaries if s["status"] == "mismatch"]
    print(f"runs: {len(summaries)}; ok {sum(s['status'] == 'ok' for s in summaries)}; valid with mismatch "
          f"{len(mism)}{' ' + str(mism) if mism else ''}; not valid {len(bad)}"
          f"{' ' + str([s['run_id'] for s in bad]) if bad else ''}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
