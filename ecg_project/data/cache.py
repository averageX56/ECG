"""Content-addressed resumable shards; manifests are published by the parent."""
import json
from pathlib import Path
import numpy as np
from ecg_project.data.catalog import file_hash
from ecg_project.data.io import header

CANONICAL_DELINEATOR='artifacts/cluster/delineator_qt.pt'


def require_checkpoint(path):
    if not Path(path).is_file():raise FileNotFoundError(f'Delineator checkpoint missing: {path}. Run GPU delineation first and pass --checkpoint {CANONICAL_DELINEATOR}')


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,sort_keys=True,indent=2),encoding='utf-8');temp.replace(path)


def begin_cache(root,identity):
    root=Path(root);p=root/'cache_identity.json'
    normalized=json.loads(json.dumps(identity,sort_keys=True))
    if p.exists():
        if json.loads(p.read_text())!=normalized:raise ValueError(f'Stale cache configuration/preprocessing: {root}; choose a new output')
    elif root.exists() and any(root.iterdir()):raise ValueError(f'Unversioned existing cache: {root}; choose a new output')
    else:atomic_json(p,normalized)
    return root


def source_hashes(path,extra=()):
    p=Path(path);paths=[p,*map(Path,extra)]
    if p.suffix=='.hea':paths += [p.parent/c[0] for c in header(p)['channels']]
    return {str(x).replace('\\','/'):file_hash(x) for x in sorted(set(paths))}


def shard(root,key,identity,compute):
    root=Path(root);p=root/(key+'.npz');meta=root/(key+'.json')
    normalized=json.loads(json.dumps(identity,sort_keys=True))
    if meta.exists():
        old=json.loads(meta.read_text())
        if old['identity']!=normalized or not p.exists() or old['sha256']!=file_hash(p):raise ValueError(f'Stale/modified cache shard {key}')
        return p
    arrays=compute();tmp=p.with_suffix('.partial')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(p);atomic_json(meta,dict(identity=normalized,sha256=file_hash(p)))
    return p


def publish(root,provenance,files):
    root=Path(root)
    atomic_json(root/'provenance.json',dict(**provenance,sha256={str(Path(p).relative_to(root)).replace('\\','/'):file_hash(p) for p in files}))


def validate_cache(root,preprocessing=None):
    root=Path(root);p=root/'provenance.json'
    if not p.is_file():raise FileNotFoundError(f'Missing completed cache {p}; run the CPU preparation command first')
    meta=json.loads(p.read_text())
    if preprocessing and meta['preprocessing']!=preprocessing:raise ValueError('Cache preprocessing version mismatch')
    for name,digest in meta['sha256'].items():
        if not (root/name).is_file() or file_hash(root/name)!=digest:raise ValueError(f'Cache provenance/hash mismatch: {name}')
    return meta
