#!/usr/bin/env python3
"""Audit a small retained-source fixture without raw process measurements."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER/'src'))
import historian_vintage as hv


def audit(records, origin=1498, horizon=6, lags=(0,2,4,6,8)):
    if any(r.get('availability_time') in (None,'') for r in records):
        return {'status':'UNDETERMINED_AVAILABILITY',
                'reason':'Source clocks absent; no zero-latency assumption substituted.'}
    events=np.array([int(r['event_time']) for r in records])
    commits=np.array([int(r['availability_time']) for r in records])
    values=np.array([float(r['value']) for r in records])
    start=int(events[0]); n=int(events[-1]-start+1)
    trace=hv.RetentionTrace(events-start,commits-start)
    trace.validate(n)
    # Nonretained entries are placeholders: final_linear/replay read retained values only.
    x=np.zeros(n); x[events-start]=values
    o=np.array([origin-start]); times=np.array([origin-start-l for l in lags])
    if times.min()<0 or origin+horizon>events[-1]:
        raise ValueError('fixture does not cover requested window/target time')
    r=hv.final_linear(x,trace)
    replay,left,right,c=hv._origin_vintage_cells(x,trace,r,times,np.repeat(o,len(times)))
    witnesses=[]
    for j,t in enumerate(times):
        sources=sorted(set([int(r.left_source[t]+start),int(r.right_source[t]+start)]))
        witnesses.append(dict(feature_time=int(t+start), source_event_times=sources,
            source_ids=[records[int(np.flatnonzero(events==event)[0])]['source_id'] for event in sources],
            source_availability_times=[int(commits[np.flatnonzero(events==event)[0]]) for event in sources],
            availability_time=int(r.availability[t]+start),feasible=bool(r.availability[t]<=o[0]),
            final_value=float(r.values[t]), replay_value=float(replay[j]),
            replay_source_event_times=sorted(set([int(left[j]+start),int(right[j]+start)])),
            replay_availability_time=int(c[j]+start)))
    ax=max(w['availability_time'] for w in witnesses)
    return dict(status='AUDITED',clock_provenance='reconstructed local Swinging Door decision',
        channel='XMEAS-7',origin=origin,nominal_horizon=horizon,target_vintage='native target assumed; no scoring in this example',
        input_availability=ax,effective_horizon=origin+horizon-max(origin,ax),
        feasible=ax<=origin,witnesses=witnesses)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,default=PAPER/'examples/retained_sources.csv')
    parser.add_argument('--output',type=Path,default=PAPER/'artifacts/core_revision/example_audit.json')
    parser.add_argument('--without-clock',action='store_true')
    args=parser.parse_args()
    with args.records.open() as stream:
        records=list(csv.DictReader(stream))
    if args.without_clock:
        records=[{**r,'availability_time':None} for r in records]
    result=audit(records)
    if result['status']=='AUDITED':
        assert result['input_availability']==1499 and result['effective_horizon']==5
        current=result['witnesses'][0]
        np.testing.assert_allclose(current['replay_value'],2799.328221565883,atol=1e-9,rtol=0)
        assert current['source_event_times']==[1498] and current['replay_source_event_times']==[1485]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
