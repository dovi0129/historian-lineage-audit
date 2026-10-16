"""Load one public TEP mode-1 trajectory after checking its sha256 against a checksum table.

Locations (environment variables, defaults relative to the repository that holds `observed_testbed/testbed/`):
- TEP_DATA_DIR: directory with `mode1_<fault>_<batch>.xlsx`; default `<repo root>/data/tep`.
- TEP_SHA256_CSV: checksum table with columns `file` and `sha256` (the public repo's `config/tep_sha256.csv`:
  file,bytes,sha256; the paper's source inventory: fault,batch,file,upstream_path,bytes,sha256);
  default `<repo root>/config/tep_sha256.csv`.
<repo root> is the parent of `observed_testbed/`, i.e. Path(__file__).resolve().parents[2].
"""
from __future__ import annotations

import csv
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("TEP_DATA_DIR", REPO_ROOT / "data" / "tep"))
INVENTORY = Path(os.environ.get("TEP_SHA256_CSV", REPO_ROOT / "config" / "tep_sha256.csv"))
N_ROWS = 3001


def file_name(fault: int, batch: int) -> str:
    return f"mode1_{fault}_{batch}.xlsx"


def expected_sha256(fault: int, batch: int) -> str:
    name = file_name(fault, batch)
    with INVENTORY.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["file"] == name:
                return row["sha256"]
    raise KeyError(f"{name} not in {INVENTORY}")


@lru_cache(maxsize=None)
def load_tep(fault: int, batch: int) -> dict[str, list[float]]:
    path = DATA / file_name(fault, batch)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    want = expected_sha256(fault, batch)
    if digest != want:
        raise RuntimeError(f"{path.name}: sha256 {digest} != inventory {want}")
    df = pd.read_excel(path)
    if len(df) != N_ROWS:
        raise RuntimeError(f"{path.name}: {len(df)} rows, expected {N_ROWS}")
    return {c: [float(v) for v in df[c].to_numpy()] for c in df.columns if c.startswith(("XMEAS-", "XMV-"))}


def file_sha256(fault: int, batch: int) -> str:
    return expected_sha256(fault, batch)
