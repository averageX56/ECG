"""Identity-only holdout registry for exclusion, never for tuning or targets."""
from pathlib import Path
from functools import partial,lru_cache
import hashlib
import json
import numpy as np
from ecg_project.data.io import load_record
from ecg_project.data.catalog import ludb_split
from ecg_project.data.policy import build_record_manifest
from ecg_project.data.delineation_v2 import qt_rows
from ecg_project.processing.parallel import ordered_map
from ecg_project.data.cache import begin_cache, source_hashes, shard


def signal_hash(rec):
    h=hashlib.sha256()
    h.update(str(float(rec.fs)).encode());h.update(';'.join(rec.leads).encode())
    h.update(np.ascontiguousarray(rec.signal,dtype='<f4').tobytes())
    return h.hexdigest()


def _hashed(row, root):
    key=hashlib.sha256((row['source']+':'+row['record_id']).encode()).hexdigest()
    path=shard(root,key,dict(row=row,source_sha256=source_hashes(row['path'])),
               lambda:dict(signal_hash=signal_hash(load_record(row['path']))))
    with np.load(path) as z:return dict(**row,signal_hash=str(z['signal_hash']))


def protected_rows(catalog='artifacts/catalog.csv',ludb_root='LUDB',qt_root='data/qtdb_external'):
    frame=build_record_manifest(catalog)
    rows=[dict(path=r.path,source=r.source_key,record_id=str(r.record_id),patient_id=r.patient_id,split=r.split) for r in frame.itertuples() if r.split!='train' and r.readable]
    paths=list(Path(ludb_root).glob('*.hea'));splits=ludb_split([p.stem for p in paths])
    if not paths:raise FileNotFoundError('LUDB is required for the protection registry')
    rows += [dict(path=str(p),source='LUDB',record_id=p.stem,patient_id='LUDB:'+p.stem,split=splits[p.stem]) for p in paths if splits[p.stem]!='train']
    # Registry is mandatory even for LUDB-only ablations; protects all QT controls.
    manual_valid=qt_rows(qt_root)
    rows += [{k:r[k] for k in ('path','source','record_id','patient_id','split')} for r in manual_valid if r['split']=='valid']
    for r in json.loads((Path(qt_root)/'split.json').read_text())['records']:
        if r['split']!='adapt_train':rows.append(dict(path=str(Path(qt_root)/(r['record_id']+'.hea')),source='QTDB',record_id=r['record_id'],patient_id='QT:'+r['record_id'],split=r['split']))
    return rows


def build_registry(catalog='artifacts/catalog.csv',workers=None,output='artifacts/protected_identities_v2'):
    rows=protected_rows(catalog)
    root=begin_cache(output,dict(version='protected_signal_identity_v2',rows=rows))
    return list(ordered_map(partial(_hashed,root=str(root)),rows,workers))


def registry_index(registry):
    if isinstance(registry,(str,Path)):
        path=Path(registry);stat=path.stat()
        return _read_index(str(path),stat.st_mtime_ns,stat.st_size)
    if isinstance(registry,dict):return registry
    return dict(patients={r['patient_id'] for r in registry if r.get('patient_id')},
                records={(r['source'],str(r['record_id'])) for r in registry},
                hashes={r['signal_hash'] for r in registry if r.get('signal_hash')})


@lru_cache(maxsize=4)
def _read_index(path,mtime,size):
    return registry_index(json.loads(Path(path).read_text()))


def assert_unprotected(row,registry):
    index=registry_index(registry)
    if row.get('patient_id') in index['patients'] or (row['source'],str(row['record_id'])) in index['records']:
        raise ValueError('Protected patient/record in pseudo/SSL train pool')
    if row.get('signal_hash') in index['hashes']:
        raise ValueError('Exact signal duplicate belongs to a protected split')
