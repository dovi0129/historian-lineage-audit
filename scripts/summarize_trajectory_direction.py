#!/usr/bin/env python3
"""Check the Ridge matched-clock direction on individual TEP test trajectories.

No model is refitted.  The one-fallback sensitivity run saved per-trajectory
losses; its ``hold`` fallback is the original zero-order-hold replay, whose cell
losses that run already matched to the original scores.  Here each cell's two
trajectory MAEs are re-averaged to confirm that pairing, and the sign of
L34 - L33 is counted per trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
FALLBACK_DIR = PAPER / "artifacts/core_revision/fallback"
EXPORT_DIR = PAPER / "artifacts/manuscript_export"
KEYS = ["region", "fold", "fault", "operator", "severity_k"]


def main() -> None:
    cells = pd.read_csv(FALLBACK_DIR / "cells.csv")
    traj = pd.read_csv(FALLBACK_DIR / "trajectory_losses.csv")
    cells = cells[cells.fallback.eq("hold") & cells.score.eq("MAE")]
    traj = traj[traj.fallback.eq("hold") & traj.score.eq("MAE")].copy()

    # Cell MAE is the row-weighted mean of its two trajectory MAEs.
    traj["w33"] = traj["L33"] * traj["n_test"]
    traj["w34"] = traj["L34"] * traj["n_test"]
    pooled = traj.groupby(KEYS)[["w33", "w34", "n_test"]].sum()
    pooled["L33"] = pooled["w33"] / pooled["n_test"]
    pooled["L34"] = pooled["w34"] / pooled["n_test"]
    joined = cells.set_index(KEYS)[["L33", "L34"]].join(pooled[["L33", "L34"]], rsuffix="_traj")
    for loss in ("L33", "L34"):
        np.testing.assert_allclose(joined[loss], joined[f"{loss}_traj"], rtol=1e-10)

    traj["delta"] = traj["L34"] - traj["L33"]
    rows = []
    for (region, operator), group in traj.groupby(["region", "operator"], sort=True):
        cell_sign = group.groupby(KEYS)["delta"].agg(
            both_positive=lambda d: bool((d > 0).all()),
            mixed=lambda d: bool((d > 0).any() and (d <= 0).any()),
        )
        rows.append(
            {
                "region": region,
                "operator": operator,
                "trajectories": int(len(group)),
                "positive_trajectories": int((group["delta"] > 0).sum()),
                "zero_trajectories": int((group["delta"] == 0).sum()),
                "cells": int(len(cell_sign)),
                "cells_both_positive": int(cell_sign["both_positive"].sum()),
                "cells_mixed": int(cell_sign["mixed"].sum()),
                "median_trajectory_delta": float(group["delta"].median()),
                "min_trajectory_delta": float(group["delta"].min()),
                "median_trajectory_pct": float(100 * (group["L34"] / group["L33"] - 1).median()),
            }
        )
    summary = pd.DataFrame(rows)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EXPORT_DIR / "trajectory_direction_summary.csv", index=False)
    (EXPORT_DIR / "trajectory_direction_checks.json").write_text(
        json.dumps({"cell_mae_reconstructed_from_trajectories": True, "cells_checked": int(len(joined))}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
