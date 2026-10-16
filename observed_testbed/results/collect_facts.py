"""Collect per-policy facts from the audit outputs into facts.json (no recomputation of the audit)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
groups = defaultdict(lambda: defaultdict(list))
for summ in sorted((ROOT / "results_full").glob("f*/summary.json")):
    s = json.loads(summ.read_text())
    m = s["meta"]
    key = f"{m['deadband']}|{m.get('retention_level', 'none')}|F={m['flush_every']}"
    g = groups[key]
    g["runs"].append(m["run_id"])
    g["is_repeat"].append(m["run_id"].endswith("_rep1"))
    g["valid"].append(s["status_valid"])
    g["status"].append(s["status"])
    led = s["ledger"]
    g["retention"].append(led["mean_achieved_retention"])
    g["discarded_share"].append(np.mean([c["n_discarded"] / c["n_samples"] for c in led["retention"].values()]))
    g["committed_changed"].append(s["truth"]["n_violations"])
    lag = led["commit_lag"]
    g["lag_p50"].append(lag.get("p50")); g["lag_max"].append(lag.get("max"))
    g["vserver_eq_v3"].append(s["v_server_vs_v3"]["equal_everywhere"])
    g["dups"].append(s["duplicates"]["ledger_membership"]["n_rows_stored_more_than_once"])
    ov = s.get("sdt_overlap") or {}
    if ov.get("channels"):
        js = [c["jaccard"] for c in ov["channels"].values() if "jaccard" in c]
        if js:
            g["jaccard"].append(np.mean(js))
    for comp, v in s["comparisons"].items():
        g[f"{comp}:cell"].append(v["cell_mismatch_rate"]); g[f"{comp}:feat"].append(v["feature_mismatch_rate"])
    g["neg"].append(json.dumps(s["negative_checks"], sort_keys=True)[:0] or
                    ",".join(f"{k}={v.get('result', v) if isinstance(v, dict) else v}" for k, v in s["negative_checks"].items()))
out = {}
for key, g in sorted(groups.items()):
    rec = {"n_runs": len(g["runs"]), "all_valid": all(v == "valid" for v in g["valid"]), "statuses": sorted(set(g["status"]))}
    for k, v in g.items():
        if k in ("runs", "valid", "status", "neg"):
            continue
        vals = [x for x in v if x is not None]
        if vals and isinstance(vals[0], bool):
            rec[k] = all(vals)
        elif vals:
            rec[k] = [round(float(min(vals)), 4), round(float(max(vals)), 4)]
    rec["negative_checks_distinct"] = sorted(set(g["neg"]))
    out[key] = rec
(ROOT / "results_full" / "facts.json").write_text(json.dumps(out, indent=1))
for k, r in out.items():
    print(k, {x: r[x] for x in ("n_runs", "all_valid", "retention", "discarded_share", "lag_p50", "lag_max",
                                  "committed_changed", "vserver_eq_v3", "dups") if x in r}, "jacc", r.get("jaccard"))
    print("   ", {x.split(":")[0]: r[x] for x in r if x.endswith(":feat")})
    print("   neg:", r["negative_checks_distinct"])
