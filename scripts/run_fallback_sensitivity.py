#!/usr/bin/env python3
"""Run the single approved fallback sensitivity, preserving original losses."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
import time
import json
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER
OSC = PAPER
sys.path[:0] = [str(PAPER/'src'), str(OSC/'src')]
import historian_vintage as hv
from replay_fallback import replay_cells
spec = importlib.util.spec_from_file_location('frozen_vintage_runner', OSC/'scripts/101_historian_vintage_replay_kill_test.py')
s101 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s101
spec.loader.exec_module(s101)
OUT = PAPER/'artifacts/core_revision/fallback'
CACHE = OUT/'tep_panel.npz'


def self_check():
    x = np.array([0., 1., 2., 9., 8., 7., 6., 5., 4., 3.])
    trace = hv.RetentionTrace(np.array([0,2,9]), np.array([0,3,9]))
    r = hv.final_linear(x, trace)
    s = np.array([1,2,2,4,7,1,9])
    o = np.array([1,2,3,4,7,3,9])
    np.testing.assert_allclose(replay_cells(x,trace,r,s,o), hv._origin_vintage_cells(x,trace,r,s,o)[0])
    np.testing.assert_allclose(replay_cells(x,trace,r,s,o,'capped_slope'), [0,0,2,4,4,1,3])
    missing = hv.RetentionTrace(np.array([0,2,9]),np.array([1,3,9]))
    try:
        replay_cells(x,missing,hv.final_linear(x,missing),np.array([0]),np.array([0]))
    except ValueError as error:
        assert str(error) == 'no eligible past source'
    else:
        raise AssertionError('missing source did not fail')
    return {'small_replay_known_answers': True, 'legacy_hold_parity': True, 'missing_source_rejected': True}


def read_segment(key):
    fault,batch = key
    frame = pd.read_excel(s101.TEP_DATA/f'mode1_{fault}_{batch}.xlsx')
    values = frame.loc[:,list(s101.TEP_CHANNELS)].to_numpy(dtype=float)
    assert values.shape == (3001,34) and np.isfinite(values).all()
    np.testing.assert_allclose(frame['Time (h)'].to_numpy()*60,np.arange(3001),atol=1e-4)
    return fault,batch,values


def prepare_cache():
    if CACHE.exists():
        return
    keys = [(f,b) for f in s101.TEP_FAULTS for b in s101.TEP_BATCHES]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("REPRO_JOBS", "8"))) as pool:
        loaded = list(pool.map(read_segment, keys))
    np.savez(CACHE, values=np.stack([r[2] for r in loaded]),
             faults=np.array([r[0] for r in loaded]), batches=np.array([r[1] for r in loaded]))
    (OUT/'cache_source.json').write_text(json.dumps(dict(data_path="data/tep",
        upstream_revision='309b944f35ac440ff0c70616947ffe723c766e14',
        channels=list(s101.TEP_CHANNELS), shape=[60,3001,34]),indent=2)+'\n')


def run_job(task):
    operator,k = task
    started=time.monotonic()
    cache=np.load(CACHE)
    panels=cache['values']
    calibration=s101.load_tep_calibration()
    parameters=[calibration[k,j] for j in range(34)]
    origins=np.arange(8,3001-6)
    # Origins for the 16-minute tasks are sliced from this shared feature cache.
    parts={(fallback,view):[] for fallback in ('hold','capped_slope') for view in (3,4)}
    max_parity=0.
    for panel in panels:
        traces=s101.make_traces(panel,operator=operator,k=k,sda_parameters=parameters)
        matrices={key:np.empty((len(origins),170),dtype=np.float32) for key in parts}
        for j,trace in enumerate(traces):
            for view,t in [(3,hv.event_time_only_trace(trace,len(panel))),(4,trace)]:
                reconstruction=hv.final_linear(panel[:,j],t)
                for li,lag in enumerate(s101.TEP_LAGS):
                    s=origins-lag
                    for fallback in ('hold','capped_slope'):
                        z=replay_cells(panel[:,j],t,reconstruction,s,origins,fallback)
                        matrices[fallback,view][:,j*5+li]=z
                        if fallback == 'hold':
                            legacy=hv._origin_vintage_cells(panel[:,j],t,reconstruction,s,origins)[0]
                            max_parity=max(max_parity,float(np.max(np.abs(z-legacy))))
        for key in parts:
            parts[key].append(matrices[key])
    features={key:np.stack(value) for key,value in parts.items()}
    del parts
    if operator == 'PERIODIC':
        for fallback in ('hold','capped_slope'):
            np.testing.assert_array_equal(features[fallback,3],features[fallback,4])
    rows, trajectory_rows=[],[]
    for region,target_name,h in s101.TEP_FULL_PANEL_TARGETS:
        n=3001-h-8
        target=panels[:,8+h:3001,s101.TEP_CHANNEL_INDEX[target_name]].reshape(-1)
        faults=np.repeat(cache['faults'],n)
        batches=np.repeat(cache['batches'],n)
        for fallback in ('hold','capped_slope'):
            x3=features[fallback,3][:,:n].reshape(-1,170)
            x4=features[fallback,4][:,:n].reshape(-1,170)
            for fold,train_batches,test_batches in s101.TEP_FOLDS:
                train=s101.even_subsample(np.flatnonzero(np.isin(batches,train_batches)))
                scaler=StandardScaler().fit(np.asarray(x3[train],dtype=float))
                model=Ridge(alpha=1.).fit(scaler.transform(np.asarray(x3[train],dtype=float)),target[train])
                for fault in s101.TEP_FAULTS:
                    test=np.flatnonzero(np.isin(batches,test_batches)&(faults==fault))
                    p3=model.predict(scaler.transform(np.asarray(x3[test],dtype=float)))
                    p4=model.predict(scaler.transform(np.asarray(x4[test],dtype=float)))
                    for score in s101.SCORES:
                        l3=s101.point_score(p3-target[test],score)
                        l4=s101.point_score(p4-target[test],score)
                        row=dict(region=region,horizon_minutes=h,fold=fold,fault=fault,
                                 operator=operator,severity_k=k,fallback=fallback,score=score,
                                 n_train=len(train),n_test=len(test),L33=l3,L34=l4,d_eval=l4/l3-1)
                        rows.append(row)
                        for batch in test_batches:
                            mask=batches[test]==batch
                            b3=s101.point_score((p3-target[test])[mask],score)
                            b4=s101.point_score((p4-target[test])[mask],score)
                            trajectory_rows.append({**row,'batch':batch,'n_test':int(mask.sum()),
                                                    'L33':b3,'L34':b4,'d_eval':b4/b3-1})
    pd.DataFrame(rows).to_csv(OUT/f'{operator}_{k}_cells.csv',index=False)
    pd.DataFrame(trajectory_rows).to_csv(OUT/f'{operator}_{k}_trajectories.csv',index=False)
    return dict(operator=operator,k=k,seconds=time.monotonic()-started,legacy_feature_max_difference=max_parity)


def summarize(checks):
    cells=pd.concat([pd.read_csv(OUT/f'{op}_{k}_cells.csv') for op in s101.OPERATORS for k in s101.LEVELS],ignore_index=True)
    traj=pd.concat([pd.read_csv(OUT/f'{op}_{k}_trajectories.csv') for op in s101.OPERATORS for k in s101.LEVELS],ignore_index=True)
    keys=['region','fold','fault','operator','severity_k','score']
    assert len(cells)==2304 and not cells.duplicated(keys+['fallback']).any()
    assert np.isfinite(cells[['L33','L34','d_eval']]).all().all()
    assert set(cells.n_test)=={2*(3001-6-8),2*(3001-16-8)}
    np.testing.assert_allclose(cells[cells.operator.eq('PERIODIC')].d_eval,0,atol=1e-12)
    original=pd.read_csv(PAPER/'artifacts/astra_revision/clock_path_cells.csv').query("process == 'TEP' and model == 'RIDGE_ARX'")
    paired=cells[cells.fallback.eq('hold')].merge(original[keys+['L33','L34']],on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==1152
    for loss in ('L33','L34'):
        np.testing.assert_allclose(paired[loss+'_new'],paired[loss+'_old'],rtol=1e-8,atol=1e-10)
    checks['original_zoh_losses_recovered']=True
    checks['max_absolute_zoh_loss_difference']=float(max((paired[l+'_new']-paired[l+'_old']).abs().max() for l in ('L33','L34')))
    cells.to_csv(OUT/'cells.csv',index=False)
    traj.to_csv(OUT/'trajectory_losses.csv',index=False)
    records=[]
    for (region,op,fallback,score),g in cells.groupby(['region','operator','fallback','score']):
        records.append(dict(region=region,operator=op,fallback=fallback,score=score,n=len(g),
                            median_pct=100*g.d_eval.median(),q25_pct=100*g.d_eval.quantile(.25),
                            q75_pct=100*g.d_eval.quantile(.75),positive=int((g.d_eval>0).sum())))
    summary=pd.DataFrame(records)
    summary.to_csv(OUT/'summary.csv',index=False)
    cells.groupby(['region','operator','fallback','score','severity_k']).d_eval.agg(['median','min','max','count']).to_csv(OUT/'by_level.csv')
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(summary.query("operator == 'SWINGING_DOOR' and score == 'MAE'").to_string(index=False),flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    checks=self_check()
    print('Small replay checks passed; preparing one durable TEP cache.',flush=True)
    prepare_cache()
    print('Panel ready; starting eight operator/severity jobs.',flush=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("REPRO_JOBS", "8"))) as pool:
        futures=[pool.submit(run_job,(op,k)) for op in s101.OPERATORS for k in s101.LEVELS]
        checks['jobs']=[]
        for future in as_completed(futures):
            result=future.result()
            checks['jobs'].append(result)
            print(json.dumps(result),flush=True)
    summarize(checks)
