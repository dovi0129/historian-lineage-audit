#!/usr/bin/env python3
"""Record the exact public TEP source files used by the manuscript.

The script hashes the 60 retained Mode-1 fault files, checks the fault-by-batch
grid, and writes a CSV plus a supplementary LaTeX table.  With
``--check-upstream`` it also compares every local SHA-256 with the Git LFS
pointer stored at the cited upstream commit, and with ``--check-distinct`` it
confirms that no two replicate trajectories of a fault coincide.  It never
transforms the process values.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
WORKSPACE = PAPER
SOURCE_DIR = PAPER / "data/tep"
EXPORT_DIR = PAPER / "artifacts/manuscript_export"
TABLE_PATH = PAPER / "tables/generated/tep_source_inventory.tex"

UPSTREAM_REPO = "Blound04/tennessee-eastman-dataset"
UPSTREAM_COMMIT = "309b944f35ac440ff0c70616947ffe723c766e14"
UPSTREAM_DIR = "simulations/mode_1/faults"
FAULTS = (1, 4, 5, 11, 14, 15)
BATCHES = tuple(range(1, 11))
NAME_RE = re.compile(r"mode1_(\d+)_(\d+)\.xlsx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE_DIR.glob("mode1_*.xlsx")):
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected file name: {path.name}")
        fault, batch = map(int, match.groups())
        rows.append(
            {
                "fault": fault,
                "batch": batch,
                "file": path.name,
                "upstream_path": f"{UPSTREAM_DIR}/{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["fault", "batch"], ignore_index=True)
    expected = {(f, b) for f in FAULTS for b in BATCHES}
    observed = set(zip(frame["fault"], frame["batch"]))
    if observed != expected:
        raise ValueError(
            f"fault-batch grid mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return frame


def upstream_oid(file_name: str) -> tuple[str, int]:
    url = (
        f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/"
        f"{UPSTREAM_DIR}/{file_name}"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        pointer = response.read(512).decode("ascii", errors="replace")
    oid = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    size = re.search(r"size (\d+)", pointer)
    if oid is None or size is None:
        raise ValueError(f"{file_name}: upstream blob is not a Git LFS pointer")
    return oid.group(1), int(size.group(1))


def check_upstream(frame: pd.DataFrame) -> dict[str, object]:
    mismatches = []
    for row in frame.itertuples(index=False):
        oid, size = upstream_oid(row.file)
        if oid != row.sha256 or size != row.bytes:
            mismatches.append(row.file)
    return {
        "repository": UPSTREAM_REPO,
        "commit": UPSTREAM_COMMIT,
        "files_checked": int(len(frame)),
        "mismatches": mismatches,
    }


def check_distinct(frame: pd.DataFrame) -> dict[str, object]:
    """Confirm that no two replicate trajectories of a fault coincide."""
    identical = []
    latest_first_difference = 0
    for fault, group in frame.groupby("fault", sort=True):
        values = {}
        for row in group.itertuples(index=False):
            table = pd.read_excel(SOURCE_DIR / row.file)
            columns = [c for c in table.columns if str(c).upper().startswith(("XMEAS", "XMV"))]
            values[row.batch] = table[columns].to_numpy(float)
        for left, right in itertools.combinations(sorted(values), 2):
            differs = np.flatnonzero(np.abs(values[left] - values[right]).max(axis=1) > 0)
            if differs.size == 0:
                identical.append([int(fault), left, right])
            else:
                latest_first_difference = max(latest_first_difference, int(differs[0]))
    return {
        "identical_pairs": identical,
        "latest_first_differing_row": latest_first_difference,
    }


def write_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{@{}rrl@{}}",
        r"\toprule",
        r"Fault & Batch & SHA-256 of \texttt{mode1\_<fault>\_<batch>.xlsx} \\",
        r"\midrule",
    ]
    for fault, group in frame.groupby("fault", sort=True):
        for row in group.itertuples(index=False):
            lines.append(f"{row.fault} & {row.batch} & \\texttt{{{row.sha256}}} \\\\")
        if fault != FAULTS[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-upstream", action="store_true")
    parser.add_argument("--check-distinct", action="store_true")
    args = parser.parse_args()

    frame = build_inventory()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXPORT_DIR / "tep_source_inventory.csv", index=False)
    write_table(frame, TABLE_PATH)

    checks: dict[str, object] = {
        "files": int(len(frame)),
        "total_bytes": int(frame["bytes"].sum()),
        "grid": "faults 1,4,5,11,14,15 x batches 1-10",
    }
    if args.check_distinct:
        checks["distinct"] = check_distinct(frame)
        if checks["distinct"]["identical_pairs"]:
            raise SystemExit(f"identical trajectories: {checks['distinct']['identical_pairs']}")
    if args.check_upstream:
        checks["upstream"] = check_upstream(frame)
        if checks["upstream"]["mismatches"]:
            raise SystemExit(f"upstream mismatch: {checks['upstream']['mismatches']}")
    (EXPORT_DIR / "tep_source_inventory_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
