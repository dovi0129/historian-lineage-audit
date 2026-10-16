"""Render the raw-output appendix of BOUNDARY_SMOKE.md from boundary_smoke/raw.json and raw_probe.json."""
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "boundary_smoke"
raw = json.loads((D / "raw.json").read_text())
probe = json.loads((D / "raw_probe.json").read_text())
out = []
for c in raw["cells"]:
    out.append(f"### {c['cell']} (n={c['n']}, F={c['F']}, SDT={'on, CompDev=' + repr(c['compdev']) if c['sdt'] else 'off'})")
    out.append(f"DDL: `{c['ddl'][-1]}`  ")
    out.append(f"SHOW TIMESERIES: `{c['show'][0].replace(chr(9) + chr(9), ' | ')}`")
    out.append("```")
    small = c["n"] <= 20
    for e in c["log"]:
        if e["event"] == "write":
            vis = [a for a, _ in e["visible"]]
            out.append(f"write t={e['tick']:>2} value={e['value']:<22} query -> ticks {vis}")
        else:
            rows = e["visible_after"]
            body = rows if (small or True) else rows
            out.append(f"FLUSH t={e['tick']:>2} returned in {e['flush_wall_s']} s; query -> "
                       + ", ".join(f"({a},{b})" for a, b in body))
            out.append(f"         same result 1 s later: {e['same_after_1s']}; tsfile+resource files: "
                       f"{len(e['tsfiles'])} -> {len(e['tsfiles_after_1s'])}")
    out.append("```")
out.append("### Follow-up probe: line 0.1·t, n = 16, CompDev 0.01, SDT on (`boundary_probe.py`)")
out.append("```")
for r in probe:
    out.append(f"F={r['F']} flush ticks {r['flush_ticks']}")
    out.append(f"  SELECT x            -> {r['raw_result_lines']}")
    out.append(f"  SELECT count(x)     -> {r['count']}")
    out.append(f"  window 13..15       -> {r['window_13_15']}")
    out.append(f"  latest row <= 15    -> {r['latest_le_15']}")
out.append("```")
print("\n".join(out))
