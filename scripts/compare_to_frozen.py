#!/usr/bin/env python3
"""Compare regenerated public data cells and displayed tables to frozen references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
KEY_NUM={'seed','fold','fault','batch','severity_k','horizon_minutes','channel_index','excluded_fault','n_lags','n_channels','n_steps','index','feature_index','source_event_time','source_availability_time','origin'}

def compare_csv(reference: Path, actual: Path) -> list[str]:
    a=pd.read_csv(reference); b=pd.read_csv(actual)
    if set(a.columns)!=set(b.columns): raise AssertionError(f'{actual}: columns differ: {set(a.columns)^set(b.columns)}')
    if len(a)!=len(b): raise AssertionError(f'{actual}: rows {len(b)} != {len(a)}')
    columns=list(a.columns); b=b[columns]
    keys=[c for c in columns if not pd.api.types.is_numeric_dtype(a[c]) or c in KEY_NUM]
    if keys:
        a=a.sort_values(keys,kind='stable',na_position='first').reset_index(drop=True)
        b=b.sort_values(keys,kind='stable',na_position='first').reset_index(drop=True)
    compared=[]
    for c in columns:
        x,y=a[c],b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            exact=c in KEY_NUM or pd.api.types.is_integer_dtype(x)
            ok=np.array_equal(x.to_numpy(),y.to_numpy(),equal_nan=True) if exact else np.allclose(x.to_numpy(float),y.to_numpy(float),rtol=1e-8,atol=0,equal_nan=True)
        else:
            ok=x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str))
        if not ok: raise AssertionError(f'{actual}: mismatch column {c}')
        compared.append(c)
    return compared

def compare_json(reference: Path, actual: Path) -> list[str]:
    expected=json.loads(reference.read_text()); observed=json.loads(actual.read_text())
    paths=[]
    def walk(a,b,path):
        if isinstance(a,dict):
            if not isinstance(b,dict): raise AssertionError(f'{actual}: type mismatch {path}')
            for key,value in a.items():
                if key not in b: raise AssertionError(f'{actual}: missing {path}.{key}')
                walk(value,b[key],f'{path}.{key}')
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b): raise AssertionError(f'{actual}: list mismatch {path}')
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{path}[{i}]')
        elif isinstance(a,float):
            if not isinstance(b,(int,float)) or not np.isclose(a,b,rtol=1e-8,atol=0,equal_nan=True):
                raise AssertionError(f'{actual}: numeric mismatch {path}: {a} != {b}')
            paths.append(path)
        else:
            if a!=b:raise AssertionError(f'{actual}: label/count mismatch {path}: {a} != {b}')
            paths.append(path)
    walk(expected,observed,'summary')
    return paths

def compare_tables(reference: Path, actual: Path) -> None:
    a=reference.read_text().splitlines(); b=actual.read_text().splitlines()
    if a!=b: raise AssertionError(f'{actual}: displayed table differs')

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--reference',type=Path,default=ROOT/'reference');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:
        for name in ('historian_vintage_replay_kill_test_smoke','historian_reconstruction_lineage_differentiation_smoke'):
            path=ROOT/'artifacts'/name/'summary.json'
            data=json.loads(path.read_text())
            if data.get('run_mode')!='SMOKE' and 'smoke' not in str(data).lower():
                raise AssertionError(f'{path}: not a smoke artifact')
            print(f'validated smoke summary: {path.relative_to(ROOT)}')
        return
    compared=0
    for reference in sorted(args.reference.rglob('*.csv')):
        actual=ROOT/'artifacts'/reference.relative_to(args.reference)
        if not actual.is_file(): raise FileNotFoundError(actual)
        columns=compare_csv(reference,actual)
        print(f'PASS {reference.relative_to(args.reference)}: {len(columns)} columns: {", ".join(columns)}',flush=True)
        compared+=1
    for reference in sorted(args.reference.rglob('summary.json')):
        actual=ROOT/'artifacts'/reference.relative_to(args.reference)
        if not actual.is_file(): raise FileNotFoundError(actual)
        fields=compare_json(reference,actual)
        print(f'PASS {reference.relative_to(args.reference)}: {len(fields)} fields',flush=True)
        compared+=1
    for reference in sorted((args.reference/'tables').glob('*.tex')):
        actual=ROOT/'tables/generated'/reference.name
        if not actual.is_file(): raise FileNotFoundError(actual)
        compare_tables(reference,actual)
        print(f'PASS table {reference.name}',flush=True)
        compared+=1
    if compared==0: raise RuntimeError('empty comparison')
    print(f'PASS {compared} public references compared')
if __name__=='__main__': main()
