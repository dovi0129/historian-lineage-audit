#!/usr/bin/env python3
"""Fetch or copy the frozen public TEP subset, validating every byte."""
from __future__ import annotations
import argparse
import csv
import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '309b944f35ac440ff0c70616947ffe723c766e14'
URL = f'https://media.githubusercontent.com/media/Blound04/tennessee-eastman-dataset/{COMMIT}/simulations/mode_1/faults'

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()

def verify(path: Path, row: dict[str,str]) -> None:
    size=path.stat().st_size
    actual=digest(path)
    if size != int(row['bytes']) or actual != row['sha256']:
        path.unlink(missing_ok=True)
        raise ValueError(f'{row["file"]}: expected {row["bytes"]} bytes/{row["sha256"]}, got {size}/{actual}')

def fetch(source: Path | None, only: str | None=None, destination: Path | None=None, smoke: bool=False) -> int:
    dest=destination or ROOT/'data/tep'
    dest.mkdir(parents=True,exist_ok=True)
    with (ROOT/'config/tep_sha256.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=60 or len({r['file'] for r in rows})!=60:
        raise ValueError('hash manifest must contain 60 distinct files')
    if smoke:
        rows=[r for r in rows if r['file'] in {f"mode1_1_{batch}.xlsx" for batch in range(1,7)}]
    if only:
        rows=[r for r in rows if r['file']==only]
        if not rows: raise ValueError(f'not in manifest: {only}')
    for row in rows:
        name=row['file']; target=dest/name
        if target.exists():
            verify(target,row)
        else:
            temporary=dest/(name+'.part')
            try:
                if source is None:
                    with urllib.request.urlopen(f'{URL}/{name}', timeout=90) as response, temporary.open('wb') as out:
                        shutil.copyfileobj(response,out)
                else:
                    shutil.copyfile(source/name,temporary)
                verify(temporary,row)
                temporary.rename(target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f'verified {name}',flush=True)
    return len(rows)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--from-local',type=Path)
    p.add_argument('--smoke',action='store_true',help='fetch only six files for the bounded smoke run')
    p.add_argument('--one',help='validate one named file; useful for network check')
    p.add_argument('--destination',type=Path,help='optional destination for one-file test')
    args=p.parse_args()
    print(f'{fetch(args.from_local,args.one,args.destination,args.smoke)} files verified')
if __name__=='__main__': main()
