#!/usr/bin/env python3
"""Time one complete TEP operator/severity unit without writing paper outputs."""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
p=ROOT/'scripts/101_historian_vintage_replay_kill_test.py'
spec=importlib.util.spec_from_file_location('historian_s101_unit',p)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
started=time.monotonic()
segments=m.load_tep_segments(False)
loaded=time.monotonic()
calibration=m.load_tep_calibration()
m.init_tep_worker(segments,calibration,False)
result=m.run_tep_config(('SWINGING_DOOR',2,'selected'))
finished=time.monotonic()
payload={'unit':'parent TEP Swinging Door k=2 selected panel','files':len(segments),'load_seconds':loaded-started,'compute_seconds':finished-loaded,'elapsed_seconds':finished-started,'rows':{k:len(v) for k,v in result.items()}}
print(json.dumps(payload,indent=2),flush=True)
(ROOT/'artifacts/full_unit_timing.json').write_text(json.dumps(payload,indent=2)+'\n')
