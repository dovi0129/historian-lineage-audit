"""Tests against hand-made fixture runs with known answers (see make_fixtures.py for the scenarios).

Values: A(t) = 1 + 0.1 t^2, B(t) = 5 - 0.3 t + 0.25 [t % 3 == 0]. Feature times s = o - {0, 2, 4, 6, 8}, o >= 8.
"""

from __future__ import annotations

import filecmp
import json
import math
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(AUDIT / "tests"))

import make_fixtures  # noqa: E402
import run_audit  # noqa: E402
from obsaudit.io_run import Issues, load_ledger_inputs, load_truth_inputs  # noqa: E402
from obsaudit.ledger import build_ledger  # noqa: E402
from obsaudit.overlap import sdt_overlap  # noqa: E402
from obsaudit.reconstruct import reconstruct  # noqa: E402
from obsaudit.truth import build_truth  # noqa: E402
from obsaudit.views import negative_check_insert, negative_check_shift, run_comparisons, view_rows  # noqa: E402

FIX = make_fixtures.FIXTURES


def A(t):
    return 1.0 + 0.1 * t * t


@lru_cache(maxsize=None)
def analyse(name: str):
    issues = Issues()
    ledger = build_ledger(load_ledger_inputs(FIX / name, issues), issues)
    truth_inp = load_truth_inputs(FIX / name, issues)
    truth = build_truth(truth_inp, issues)
    comp = run_comparisons(ledger, truth, truth_inp.meta)
    return issues, ledger, truth, comp, truth_inp.meta


def bad_cells(comp, name):
    return sorted((m["origin"], m["channel"]) for m in comp.mismatches if m["comparison"] == name)


def lrow(ledger, ch, t):
    rows = ledger.rows
    return rows.loc[(rows["channel"] == ch) & (rows["tick"] == t)].iloc[0]


# ------------------------------------------------------------------ rule R

def test_reconstruction_rule_hand_cases():
    rows = ((2, 1.0), (6, 3.0))
    assert math.isnan(reconstruct(rows, 1))          # no row at or before s: missing
    assert reconstruct(rows, 2) == 1.0               # exact row
    assert reconstruct(rows, 4) == 2.0               # linear between nearest rows
    assert reconstruct(rows, 6) == 3.0
    assert reconstruct(rows, 9) == 3.0               # no later row: hold
    assert math.isnan(reconstruct((), 3))
    assert math.isnan(reconstruct(((5, 2.0),), 3))   # a later row alone is never back-filled


def test_fixtures_match_generator(tmp_path):
    names = make_fixtures.build(tmp_path)
    assert names == sorted(p.name for p in FIX.iterdir() if p.is_dir())
    for name in names:
        cmp = filecmp.dircmp(tmp_path / name, FIX / name)
        assert not (cmp.diff_files or cmp.left_only or cmp.right_only), (name, cmp.diff_files)


# ------------------------------------------------------------------ independence of the two paths

def test_paths_are_independent(tmp_path):
    ledger_only = tmp_path / "provisional_discard"
    shutil.copytree(FIX / "provisional_discard", ledger_only)
    (ledger_only / "snapshots.csv").unlink()
    issues = Issues()
    ledger = build_ledger(load_ledger_inputs(ledger_only, issues), issues)
    assert issues.n_errors == 0 and len(ledger.rows) == 40

    truth_only = tmp_path / "truth" / "provisional_discard"
    shutil.copytree(FIX / "provisional_discard", truth_only)
    for name in ("samples.csv", "membership_delta.csv", "membership_digest.csv"):
        (truth_only / name).unlink()
    issues = Issues()
    truth = build_truth(load_truth_inputs(truth_only, issues), issues)
    assert issues.n_errors == 0 and (12, "A") in truth.cells


# ------------------------------------------------------------------ scenarios

def test_nodb_f1_all_views_equal():
    issues, ledger, truth, comp, meta = analyse("nodb_f1")
    assert issues.n_errors == 0
    assert all(t.n_any == 0 and t.n_cells == 24 for t in comp.tallies.values())  # origins 8..19, 2 channels
    assert comp.v_server_differs_v3_cells == 0
    assert (ledger.rows["c_obs"] == ledger.rows["tick"]).all()


def test_lag_f5_commit_lag():
    issues, ledger, truth, comp, meta = analyse("lag_f5")
    assert issues.n_errors == 0
    assert bad_cells(comp, "Replay_commit_vs_S_commit") == []
    assert bad_cells(comp, "Replay_visible_vs_S_all") == []
    between_flushes = [8, 10, 11, 12, 13, 15, 16, 17, 18]  # flushes at 4, 9, 14, 19
    expected = sorted((o, c) for o in between_flushes for c in "AB")
    for name in ("V3_vs_S_commit", "V_server_vs_S_commit", "V1_vs_S_commit", "S_all_vs_S_commit"):
        assert bad_cells(comp, name) == expected, name
    lag = ledger.rows["c_obs"] - ledger.rows["tick"]
    assert lag.value_counts().to_dict() == {0: 8, 1: 8, 2: 8, 3: 8, 4: 8}
    # Right-endpoint hold: at o = 12 the last committed row is 9, so s = 12 and s = 10 hold A(9).
    cell = truth.cells[(12, "A")]
    assert [t for t, _ in cell.s_commit] == [3, 4, 5, 6, 7, 8, 9]  # anchor 3, then window rows <= t_f = 9
    replay = view_rows(comp.arrays["Replay_commit"]["A"], 12)
    assert reconstruct(replay, 12) == reconstruct(cell.s_commit, 12) == A(9)
    assert reconstruct(replay, 10) == A(9)


def test_provisional_row_later_discarded():
    issues, ledger, truth, comp, meta = analyse("provisional_discard")
    assert issues.n_errors == 0
    r = lrow(ledger, "A", 11)
    assert r.write_ack_tick == 11 and math.isnan(r.c_obs) and r.discard_tick == 14 and not r.final_retained
    r = lrow(ledger, "A", 10)
    assert r.c_obs == 14 and r.final_retained
    cell = truth.cells[(12, "A")]
    assert [t for t, _ in cell.s_all] == [0, 4, 5, 9, 10, 11, 12]  # anchor 0 (rows 1..3 discarded)
    assert [t for t, _ in cell.s_commit] == [0, 4, 5, 9]
    assert (12, "A") in bad_cells(comp, "S_all_vs_S_commit")
    assert bad_cells(comp, "Replay_commit_vs_S_commit") == []
    assert bad_cells(comp, "Replay_visible_vs_S_all") == []
    retention = ledger.rows.groupby("channel")["final_retained"].mean().to_dict()
    assert retention == {"A": 0.4, "B": 1.0}
    # B's server stamp lags one tick, so V_server differs from V3 on B only.
    assert comp.v_server_differs_v3_cells > 0
    v3 = set(bad_cells(comp, "V3_vs_S_commit"))
    vs = set(bad_cells(comp, "V_server_vs_S_commit"))
    assert {c for _, c in vs ^ v3} == {"B"}


def test_origin_before_first_flush_and_commit_anchor():
    issues, ledger, truth, comp, meta = analyse("late_first_flush")
    assert issues.n_errors == 0
    tally = comp.tallies["Replay_commit_vs_S_commit"]
    assert tally.n_excluded_before_first_flush == 6        # origins 8, 9, 10 x 2 channels (first flush at 11)
    assert tally.n_excluded_undetermined == 0
    assert tally.n_cells == 30 and tally.n_any == 0          # origins 11..25
    assert truth.cells[(21, "A")].s_commit == ((8, A(8)),)  # A discarded 9..11; anchor 12 is provisional
    assert truth.cells[(21, "B")].s_commit[0][0] == 11
    # Without commit_anchor rows the same cells are undetermined, counted, and excluded.
    issues, ledger, truth, comp, meta = analyse("late_first_flush_noca")
    tally = comp.tallies["Replay_commit_vs_S_commit"]
    assert tally.n_excluded_undetermined == 4                 # origins 21, 22 x 2 channels
    assert tally.n_cells == 26 and tally.n_any == 0


def test_committed_row_removed_or_changed_is_flagged():
    issues, ledger, truth, comp, meta = analyse("violation")
    assert issues.n_errors == 0
    viol = {(v["type"], v["channel"], v["row_time"]): v for v in ledger.violations}
    removed = viol[("committed_row_removed", "A", 12)]
    assert removed["c_obs"] == 14 and removed["removed_at_tick"] == 19
    changed = viol[("committed_row_changed", "B", 7)]
    assert changed["c_obs"] == 9 and changed["changed_at_tick"] == 19 and changed["new_value"] == 99.5
    assert len(viol) == 2
    assert sorted(v["origin"] for v in truth.violations) == [19, 20, 21]
    assert bad_cells(comp, "Replay_commit_vs_S_commit") == [(19, "A"), (20, "A"), (21, "A")]
    assert bad_cells(comp, "Replay_visible_vs_S_all") == []


def test_late_commit_after_absence():
    issues, ledger, truth, comp, meta = analyse("late_commit")
    r = lrow(ledger, "A", 6)
    assert r.discard_tick == 9 and r.c_obs == 14 and r.final_retained
    assert [a["type"] for a in ledger.anomalies] == ["committed_after_discard"]
    assert bad_cells(comp, "Replay_commit_vs_S_commit") == []
    assert bad_cells(comp, "Replay_visible_vs_S_all") == []
    assert (9, "A") in bad_cells(comp, "V3_vs_S_commit")      # V3 assumes row 6 available at tau = 6


def test_one_row_segment_duplicates_do_not_mismatch():
    issues, ledger, truth, comp, meta = analyse("dup_f1")
    assert issues.n_errors == 0                                 # digests verified over the multiset
    assert all(t.n_any == 0 for t in comp.tallies.values())
    assert ledger.duplicates["by_channel"] == {"A": 20, "B": 0} and ledger.duplicates["max_copies"] == 2
    # Snapshot window copies for A: origin o shows min(o + 1, 9) doubled rows -> 45 + 9 * 11 = 144; anchors one copy.
    assert truth.duplicates == {"A": {"window": 144}}
    m = comp.multiplicity
    assert m["n_cells"] == 24 and m["n_cells_disagree"] == 0
    assert m["n_cells_with_duplicates_ledger"] == m["n_cells_with_duplicates_snapshot"] == 12


def test_same_time_different_values_raise(tmp_path):
    s = run_audit.audit_run(FIX / "dup_conflict", tmp_path)
    checks = {i["check"] for i in s["validation"]["issues"] if i["level"] == "error"}
    assert s["status"] == "invalid"
    assert {"delta:same_time_values", "snap:same_time_values"} <= checks
    snap = next(i for i in s["validation"]["issues"] if i["check"] == "snap:same_time_values")
    assert {r["origin"] for r in snap["rows"]} == set(range(9, 15))  # A row 6 is in the window at o = 9..14


# ------------------------------------------------------------------ negative checks

def test_negative_checks_detect():
    issues, ledger, truth, comp, meta = analyse("provisional_discard")
    a = negative_check_shift(comp, truth, meta)
    assert a["status"] == "detected" and a["mode"] == "interpolation"
    assert a["rowset_detected"] and a["feature_detected"]
    b = negative_check_insert(comp, truth, meta)
    assert b["future_time"]["status"] == "detected" and b["future_time"]["unused"]
    assert not b["future_time"]["features_changed"]
    assert b["future_commit"]["status"] == "detected"

    issues, ledger, truth, comp, meta = analyse("nodb_f1")
    a = negative_check_shift(comp, truth, meta)
    assert a["status"] == "detected" and a["mode"] == "exact"   # no interpolation exists without a deadband
    b = negative_check_insert(comp, truth, meta)
    assert b["future_time"]["status"] == "detected"
    assert b["future_commit"]["status"] == "not_applicable"


# ------------------------------------------------------------------ run gate (code audit findings 1 and 4)

def test_violation_fixture_is_not_ok(tmp_path):
    s = run_audit.audit_run(FIX / "violation", tmp_path)
    assert s["status"] == "invalid" and s["status_valid"] == "invalid"
    assert any("ledger violations" in r for r in s["invalid_reasons"])
    assert any("snapshot persistence" in r for r in s["invalid_reasons"])
    assert s["primary_result"]["result"] == "mismatch" and s["primary_result"]["n_cells_mismatch"] == 3
    assert s["visible_result"]["result"] == "match"


def test_unscheduled_flush_invalidates(tmp_path):
    s = run_audit.audit_run(FIX / "unscheduled_flush", tmp_path)
    errors = {i["check"]: i for i in s["validation"]["issues"] if i["level"] == "error"}
    assert s["status"] == "invalid" and set(errors) == {"flush:unscheduled"}
    assert [(r["tick"], r["tsfiles_before"], r["previous_tsfiles_after"]) for r in errors["flush:unscheduled"]["rows"]] \
        == [(9, 2, 1)]


def test_flush_sealing_two_files_invalidates(tmp_path):
    run = _copy(tmp_path, "lag_f5")
    path = run / "flushes.csv"
    lines = path.read_text().splitlines()
    for k in range(1, len(lines)):  # first FLUSH seals 2 files; later counts shift by one, so no gap
        cells = lines[k].split(",")
        cells[-1] = str(int(cells[-1]) + 1)
        if k > 1:
            cells[-2] = str(int(cells[-2]) + 1)
        lines[k] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")
    s = run_audit.audit_run(run, tmp_path / "out")
    errors = {i["check"] for i in s["validation"]["issues"] if i["level"] == "error"}
    assert s["status"] == "invalid" and errors == {"flush:sealed_count"}


def test_valid_run_with_primary_mismatch_is_a_result(tmp_path):
    # Change the value IoTDB "returned" for A row 19 in the last snapshot only: no persistence check can see it,
    # so the run stays valid and the primary and visible comparisons report one mismatching cell each.
    run = _copy(tmp_path, "lag_f5")
    path = run / "snapshots.csv"
    lines = path.read_text().splitlines()
    k = next(i for i, line in enumerate(lines) if line.startswith("lag_f5,19,A,19,"))
    cells = lines[k].split(",")
    cells[4] = repr(float(cells[4]) + 0.5)
    lines[k] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n")
    s = run_audit.audit_run(run, tmp_path / "out")
    assert s["status"] == "mismatch" and s["status_valid"] == "valid" and s["invalid_reasons"] == []
    assert s["primary_result"]["n_cells_mismatch"] == 1 and s["visible_result"]["n_cells_mismatch"] == 1
    assert run_audit.main([str(run), "--out", str(tmp_path / "cli")]) == 0


# ------------------------------------------------------------------ validator and CLI

def _copy(tmp_path, name):
    dst = tmp_path / name
    shutil.copytree(FIX / name, dst)
    return dst


def test_validator_flags_bad_digest_and_n_ticks(tmp_path):
    run = _copy(tmp_path, "lag_f5")
    path = run / "membership_digest.csv"
    lines = path.read_text().splitlines()
    head, first = lines[1].rsplit(",", 1)
    lines[1] = head + "," + ("0" * 64 if first != "0" * 64 else "1" * 64)
    path.write_text("\n".join(lines) + "\n")
    s = run_audit.audit_run(run, tmp_path / "out")
    assert s["status"] == "invalid"
    assert any(i["check"] == "digest:mismatch" for i in s["validation"]["issues"])

    run2 = _copy(tmp_path / "b", "lag_f5")
    meta = json.loads((run2 / "run_meta.json").read_text())
    meta["n_ticks"] = 21
    (run2 / "run_meta.json").write_text(json.dumps(meta))
    s = run_audit.audit_run(run2, tmp_path / "out2")
    checks = {i["check"] for i in s["validation"]["issues"]}
    assert s["status"] == "invalid" and {"samples:n_ticks", "flush:final"} <= checks

    run3 = _copy(tmp_path / "c", "lag_f5")
    (run3 / "DONE").unlink()
    assert run_audit.audit_run(run3, tmp_path / "out3")["status"] == "incomplete"


def test_cli_all_writes_outputs(tmp_path):
    rc = run_audit.main(["--all", str(FIX), "--out", str(tmp_path)])
    assert rc == 1  # four fixtures are invalid by design
    for name in ("nodb_f1", "violation"):
        for f in ("summary.json", "summary.md", "ledger.csv", "violations.csv", "mismatches_primary.csv",
                  "mismatches_all.csv.gz"):
            assert (tmp_path / name / f).is_file()
    pooled = json.loads((tmp_path / "pooled" / "summary.json").read_text())
    assert pooled["n_runs_audited"] == 10 and pooled["primary_mismatch_cells_total"] == 3
    # late_first_flush_noca: 4 truth-undetermined cells are an unexplained exclusion, so the run is invalid.
    assert set(pooled["runs_invalid"]) == {"dup_conflict", "late_first_flush_noca", "unscheduled_flush", "violation"}
    assert pooled["runs_invalid"]["late_first_flush_noca"] == ["4 truth-undetermined cells"]
    assert pooled["runs_primary_mismatch"] == ["violation"]
    assert pooled["n_runs_ok"] == 6
    assert (tmp_path / "pooled" / "summary.md").read_text().count("| detected") >= 7


def test_sdt_overlap_hand_case():
    import pandas as pd
    ticks = list(range(10))
    rows = pd.DataFrame({"channel": "A", "tick": ticks, "value_sent": [0.5 * t for t in ticks],
                         "final_retained": [t in (0, 5) for t in ticks]})
    out = sdt_overlap(rows, {"compdev": {"A": 0.1}})["channels"]["A"]
    # A straight line keeps only its endpoints under the local rule: {0, 9} vs IoTDB {0, 5}.
    assert out["n_local_rule"] == 2 and out["n_both"] == 1 and out["jaccard"] == pytest.approx(1 / 3)
    assert sdt_overlap(rows, {"compdev": {"A": None}})["channels"]["A"] == {"status": "no_deadband"}
