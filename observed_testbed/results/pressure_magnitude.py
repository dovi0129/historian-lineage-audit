"""Pressure (XMEAS-7) feature differences between the V3 proxy and the committed snapshot, by deadband and F.

Reads the audit outputs only. The difference is reported per mismatching feature (origin, lag) in kPa and relative
to the standard deviation of the run's sent pressure values.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for summ in sorted((ROOT / "results_full").glob("f*/summary.json")):
    s = json.loads(summ.read_text())
    meta = s["meta"]
    if meta["run_id"].endswith("_rep1"):
        continue
    mm = pd.read_csv(summ.parent / "mismatches_all.csv.gz")
    mm = mm[(mm.comparison == "V3_vs_S_commit") & (mm.channel == "XMEAS-7")]
    diffs = []
    for fv, ft in zip(mm.features_view, mm.features_truth):
        v = dict(x.split(":") for x in str(fv).split(";"))
        t = dict(x.split(":") for x in str(ft).split(";"))
        diffs += [abs(float(v[k]) - float(t[k])) for k in v if k in t and float(v[k]) != float(t[k])]
    samples = pd.read_csv(ROOT / "runs" / meta["run_id"] / "samples.csv")
    sd = samples.loc[samples.channel == "XMEAS-7", "value_sent"].astype(float).std(ddof=1)
    rows.append(dict(run=meta["run_id"], deadband=meta["deadband"], F=meta["flush_every"],
                     n=len(diffs), sd=sd, diffs=np.array(diffs)))
out = []
for (db, F), g in pd.DataFrame(rows).groupby(["deadband", "F"]):
    d = np.concatenate(g.diffs.values) if g.n.sum() else np.array([])
    rel = np.concatenate([r.diffs / r.sd for r in g.itertuples()]) if g.n.sum() else np.array([])
    out.append(dict(deadband=db, F=F, runs=len(g), n_feature_diffs=int(d.size),
                    median_kPa=float(np.median(d)) if d.size else 0.0,
                    p90_kPa=float(np.quantile(d, 0.9)) if d.size else 0.0,
                    max_kPa=float(d.max()) if d.size else 0.0,
                    median_rel_sd=float(np.median(rel)) if rel.size else 0.0))
res = pd.DataFrame(out)
res.to_csv(ROOT / "results_full" / "pressure_magnitude.csv", index=False)
print(res.to_string(index=False))
