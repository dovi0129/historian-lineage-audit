from __future__ import annotations
import csv
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import historian_vintage as hv

ROOT=Path(__file__).resolve().parents[1]

def load_fetch():
    spec=importlib.util.spec_from_file_location('fetch_tep',ROOT/'scripts/fetch_tep.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_availability_and_origin_replay_known_answer():
    values=np.array([0.,1.,2.,3.,4.])
    trace=hv.RetentionTrace(np.array([0,2,4]),np.array([0,3,4]))
    final=hv.final_linear(values,trace)
    np.testing.assert_array_equal(final.availability,[0,3,3,4,4])
    result,left,right,commit=hv._origin_vintage_cells(values,trace,final,np.array([2,2,1]),np.array([2,3,1]))
    np.testing.assert_allclose(result,[0.,2.,0.])
    np.testing.assert_array_equal(left,[0,2,0])
    np.testing.assert_array_equal(right,[0,2,0])
    np.testing.assert_array_equal(commit,[0,3,0])

def test_manifest_and_bad_hash(tmp_path):
    fetch=load_fetch()
    with (ROOT/'config/tep_sha256.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==60
    assert len({r['file'] for r in rows})==60
    p=tmp_path/rows[0]['file'];p.write_bytes(b'invalid')
    with pytest.raises(ValueError,match='expected'):
        fetch.verify(p,rows[0])
    assert not p.exists()

def test_no_private_references():
    forbidden=['PM'+'3','data_'+'role_audit','Raw'+'Data','/home/'+'minu']
    for folder in ('src','scripts','examples','config','reference','tables','tests'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py','.md','.csv','.json','.tex','.txt','.sh'}:
                data=p.read_text(encoding='utf-8')
                assert not any(word in data for word in forbidden),p

def test_constant_calibration_identity():
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('calibrate_tep',ROOT/'scripts/calibrate_tep.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    segments=[SimpleNamespace(batch=b,values=np.zeros((5,34))) for b in range(1,5)]
    frame=module.calibrate(segments,(0,))
    assert len(frame)==4
    assert frame.epsilon.eq(0).all()
    assert frame.calibration_feasible.eq(0).all()
    assert frame.achieved_calibration_retention.eq(1).all()

def test_comparison_relative_tolerance_and_exact_counts(tmp_path):
    spec=importlib.util.spec_from_file_location('compare_to_frozen',ROOT/'scripts/compare_to_frozen.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import pandas as pd
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    pd.DataFrame({'region':['A','B'],'count':[2,3],'loss':[100.0,1.0]}).to_csv(a,index=False)
    pd.DataFrame({'region':['B','A'],'count':[3,2],'loss':[1.0,100.0000001]}).to_csv(b,index=False)
    module.compare_csv(a,b)
    pd.DataFrame({'region':['B','A'],'count':[3,2],'loss':[1.0,100.0001]}).to_csv(b,index=False)
    with pytest.raises(AssertionError,match='loss'):module.compare_csv(a,b)
    pd.DataFrame({'region':['B','A'],'count':[3,4],'loss':[1.0,100.0]}).to_csv(b,index=False)
    with pytest.raises(AssertionError,match='count'):module.compare_csv(a,b)

def test_network_fetch_branch_with_verified_bytes(tmp_path,monkeypatch):
    import hashlib
    import io
    fetch=load_fetch()
    name='mode1_1_1.xlsx'
    data=b'known public-file fixture'
    repo=tmp_path/'repo';(repo/'config').mkdir(parents=True)
    with (repo/'config/tep_sha256.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','bytes','sha256']);writer.writeheader()
        for i in range(60):
            writer.writerow({'file':name if i==0 else f'placeholder_{i}.xlsx',
                             'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(fetch,'ROOT',repo)
    seen=[]
    def response(url,timeout):
        seen.append(url)
        return io.BytesIO(data)
    monkeypatch.setattr(fetch.urllib.request,'urlopen',response)
    assert fetch.fetch(None,name,repo/'data/tep')==1
    assert seen==[f'{fetch.URL}/{name}']
    assert (repo/'data/tep'/name).read_bytes()==data
