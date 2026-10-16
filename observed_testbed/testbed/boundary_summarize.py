"""Summarise boundary_smoke/raw.json (no IoTDB access)."""
import json
from pathlib import Path

raw = json.loads((Path(__file__).resolve().parent / "boundary_smoke" / "raw.json").read_text())
for c in raw["cells"]:
    flushes = [e for e in c["log"] if e["event"] == "flush"]
    writes = [e for e in c["log"] if e["event"] == "write"]
    prev_after: set = set()
    seg_start = 0
    lines = []
    pre_all_visible = all([w["tick"], w["value"]] in w["visible"] for w in writes)
    pre_raw_only = True  # before flush, rows of the open segment are all visible
    for f in flushes:
        t = f["tick"]
        seg_end = min(t, c["n"] - 1)
        after = {a for a, _ in f["visible_after"]}
        seg = list(range(seg_start, seg_end + 1)) if seg_end >= seg_start else []
        kept = [x for x in seg if x in after]
        lost_prev = sorted(prev_after - after)
        lines.append(f"  flush@{t}: segment {seg_start}..{seg_end} kept {kept}; earlier committed lost {lost_prev}; "
                     f"stable_1s={f['same_after_1s']}; files={len(f['tsfiles'])}->{len(f['tsfiles_after_1s'])}; "
                     f"{f['flush_wall_s']}s")
        prev_after = after
        seg_start = seg_end + 1
    final = [a for a, _ in flushes[-1]["visible_after"]]
    print(f"{c['cell']}: n={c['n']} compdev={c['compdev']} retained={len(final)}/{c['n']} final={final}")
    print(f"  every written row visible right after its write (pre-flush): {pre_all_visible}")
    for l in lines[:6] + (["  ..."] if len(lines) > 8 else []) + lines[-2:] if len(lines) > 8 else lines:
        print(l)
