#!/usr/bin/env python3
"""Recompute the frozen Swinging Door tolerances from public TEP batches 1-4."""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import historian_inferential_instability as hii

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('s101_calibration',ROOT/'scripts/101_historian_vintage_replay_kill_test.py')
s101=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s101;spec.loader.exec_module(s101)

def calibrate(segments, channels):
    support=[s for s in segments if s.batch in (1,2,3,4)]
    rows=[]
    for k in s101.LEVELS:
        target=1/k
        for j in channels:
            name=s101.TEP_CHANNELS[j]
            values=[s.values[:,j] for s in support]
            span=float(np.ptp(np.concatenate(values)))
            if span<=1e-15:
                epsilon,achieved,feasible,status=0.,1.,False,'STRUCTURAL_CONSTANT'
            else:
                def retention(epsilon):
                    kept=sum(len(hii.swinging_door_indices(x,epsilon)) for x in values)
                    return kept/sum(map(len,values))
                low,high=0.,span
                candidates=[(low,retention(low)),(high,retention(high))]
                for _ in range(28):
                    mid=.5*(low+high);ratio=retention(mid);candidates.append((mid,ratio))
                    if ratio>target:low=mid
                    else:high=mid
                epsilon,achieved=min(candidates,key=lambda x:(abs(x[1]-target),x[0]))
                feasible=True;status='CALIBRATED'
            obs={2:'SWINGING_DOOR_50',5:'SWINGING_DOOR_20'}.get(k,f'SWINGING_DOOR_K{k}')
            rows.append(dict(operator='SWINGING_DOOR',observation=obs,severity_k=k,nominal_retention=target,
                channel_index=j,channel=name,epsilon=epsilon,achieved_calibration_retention=achieved,
                calibration_feasible=int(feasible),calibration_status=status,calibration_batches='1|2|3|4'))
    return pd.DataFrame(rows)

_SEGMENTS = None

def init_worker(segments):
    global _SEGMENTS
    _SEGMENTS=segments

def channel_job(j):
    return calibrate(_SEGMENTS,(j,))

def main():
    p=argparse.ArgumentParser();p.add_argument('--one-channel',type=int);args=p.parse_args()
    segments=s101.load_tep_segments(False)
    channels=range(34) if args.one_channel is None else (args.one_channel,)
    if args.one_channel is None:
        workers=min(int(os.environ.get('REPRO_JOBS','16')),len(channels),16)
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(segments,)) as pool:
            frames=list(pool.map(channel_job,channels))
        frame=pd.concat(frames,ignore_index=True).sort_values(['severity_k','channel_index']).reset_index(drop=True)
    else:
        frame=calibrate(segments,channels)
    expected=pd.read_csv(ROOT/'config/operator_calibration.csv')
    keys=['severity_k','channel_index']
    joined=frame.merge(expected,on=keys,validate='one_to_one',suffixes=('_new','_frozen'))
    if len(joined)!=len(frame):raise AssertionError('missing frozen calibration rows')
    for c in frame.columns:
        if c in keys:continue
        a,b=joined[f'{c}_new'],joined[f'{c}_frozen']
        if pd.api.types.is_numeric_dtype(a):
            np.testing.assert_allclose(a,b,rtol=1e-8,atol=0)
        elif not a.equals(b):raise AssertionError(f'{c} differs')
    out=ROOT/'artifacts/calibration_recomputed.csv';out.parent.mkdir(exist_ok=True)
    frame.to_csv(out,index=False)
    print(f'verified {len(frame)} calibration rows against frozen public table')
if __name__=='__main__':main()
