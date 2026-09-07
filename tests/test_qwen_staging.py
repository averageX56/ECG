import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from ecg_project.data.cache import publish,validate_cache
from ecg_project.data.catalog import file_hash
from pipelines.gpu.qwen_staging import copy_snapshot,sync_completed_cache,EXTENDED_QWEN_SOURCES,stage_qwen_raw


def test_qwen_sources_exactly_without_ningbo(monkeypatch,tmp_path):
    from pipelines.gpu import colab,experiments
    assert EXTENDED_QWEN_SOURCES==colab.EXTENDED_QWEN_SOURCES==('CPSC_EXTRA','PTBXL','CPSC','CHAPMAN')
    drive=tmp_path/'drive';raw=tmp_path/'scratch';root=tmp_path/'repo'
    dirs=['LUDB','data/qtdb_external','data/Training_2','data/Training_WFDB','data/WFDB_PTB-XL','data/WFDB_ChapmanShaoxing','data/WFDB_GEORGIA']
    for name in dirs:
        (drive/name).mkdir(parents=True);(drive/name/'signal.dat').write_bytes(b'signal')
    (drive/'data/WFDB_Ningbo').mkdir();(drive/'data/WFDB_Ningbo/never.dat').write_bytes(b'never')
    from ecg_project.data import policy
    monkeypatch.setattr(policy,'build_record_manifest',lambda *a:pd.DataFrame([dict(path='data/WFDB_GEORGIA/g.hea',split='external',readable=True)]))
    stage_qwen_raw(root,drive,raw)
    assert all((raw/name/'signal.dat').is_file() for name in dirs)
    assert not (raw/'data/WFDB_Ningbo').exists()
    assert 'NINGBO' not in __import__('inspect').getsource(experiments.assert_qwen_inputs)


def test_staging_preserves_unrelated_and_reuses_snapshot(tmp_path,monkeypatch):
    import shutil
    src=tmp_path/'source';dst=tmp_path/'local';src.mkdir();dst.mkdir()
    (src/'a.dat').write_bytes(b'a');(dst/'user.txt').write_bytes(b'keep')
    with pytest.raises(FileExistsError,match='Nonempty unrelated'):copy_snapshot(src,dst,tmp_path/'stage.json')
    assert (dst/'user.txt').read_bytes()==b'keep'
    dst=tmp_path/'owned';copy_snapshot(src,dst,tmp_path/'stage.json')
    monkeypatch.setattr(shutil,'copy2',lambda *a:pytest.fail('Complete snapshot copied again'))
    copy_snapshot(src,dst,tmp_path/'stage.json')
    (src/'a.dat').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Drive source changed'):copy_snapshot(src,dst,tmp_path/'stage.json',verify_sources=True)


def test_sync_publishes_only_complete_validated_cache(tmp_path,monkeypatch):
    from pipelines.gpu import qwen_staging as staging
    local=tmp_path/'local';drive=tmp_path/'drive/cache';local.mkdir()
    np.savez(local/'r.npz',x=np.ones(5))
    with pytest.raises(FileNotFoundError):sync_completed_cache(local,drive)
    assert not drive.exists()
    publish(local,dict(preprocessing='test'),[local/'r.npz'])
    original=staging.shutil.copy2
    monkeypatch.setattr(staging.shutil,'copy2',lambda *a:(_ for _ in ()).throw(InterruptedError('sync interrupted')))
    with pytest.raises(InterruptedError):sync_completed_cache(local,drive)
    assert not drive.exists()
    monkeypatch.setattr(staging.shutil,'copy2',original)
    sync_completed_cache(local,drive)
    assert validate_cache(drive)==validate_cache(local)
    sync_completed_cache(local,drive)  # Repeating successful sync is safe.


def test_resume_skips_raw_reads_but_rejects_teacher_and_source_changes(tmp_path,monkeypatch):
    from ecg_project.data import qwen_pseudo as worker
    raw=tmp_path/'raw.dat';raw.write_bytes(b'raw')
    monkeypatch.setattr(worker,'source_hashes',lambda _:{str(raw):file_hash(raw)})
    monkeypatch.setattr(worker,'load_record',lambda _:SimpleNamespace(signal=np.ones((2500,1),np.float32),fs=250,leads=['II']))
    row=dict(source='CPSC',record_id='r',patient_id='p',path=str(raw),split='train')
    item,_=worker._input(row,tmp_path,'teacher',.95,[])
    worker._result(item,tmp_path,np.zeros((4,2500),np.float32))
    for name in ('source_hashes','load_record','signal_hash','preprocess','resample'):
        monkeypatch.setattr(worker,name,lambda *a:pytest.fail('Expensive raw work during resume'))
    cached,_=worker._input(row,tmp_path,'teacher',.95,[]);assert cached['cached']
    with pytest.raises(ValueError,match='Stale cache identity'):worker._input(row,tmp_path,'new_teacher',.95,[])
    with pytest.raises(ValueError,match='Protected'):worker._input(row,tmp_path,'teacher',.95,[dict(source='CPSC',record_id='r',patient_id='p')])
    raw.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='Stale source'):worker._input(row,tmp_path,'teacher',.95,[])


def test_protection_registry_resume_skips_waveform_decode(tmp_path,monkeypatch):
    from ecg_project.data import protection
    raw=tmp_path/'raw.dat';raw.write_bytes(b'raw')
    monkeypatch.setattr(protection,'source_hashes',lambda _:{str(raw):file_hash(raw)})
    monkeypatch.setattr(protection,'load_record',lambda _:SimpleNamespace(signal=np.ones((100,1)),fs=250,leads=['II']))
    row=dict(source='LUDB',record_id='1',patient_id='p',path=str(raw),split='test')
    first=protection._hashed(row,tmp_path)
    monkeypatch.setattr(protection,'source_hashes',lambda _:pytest.fail('Raw files hashed again'))
    monkeypatch.setattr(protection,'load_record',lambda _:pytest.fail('Raw ECG decoded again'))
    assert protection._hashed(row,tmp_path)==first


def test_selected_lead_preprocessing_preserves_original_input(tmp_path,monkeypatch):
    from ecg_project.data import qwen_pseudo as worker
    raw=tmp_path/'raw.dat';raw.write_bytes(b'raw')
    rec=SimpleNamespace(signal=np.random.default_rng(6).normal(size=(6500,12)),fs=500,
                        leads=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'])
    monkeypatch.setattr(worker,'source_hashes',lambda _:{str(raw):file_hash(raw)})
    monkeypatch.setattr(worker,'load_record',lambda _:rec)
    row=dict(source='CPSC',record_id='r',patient_id='p',path=str(raw),split='train')
    expected=worker.normalize(worker.resample(worker.preprocess(rec.signal,rec.fs),rec.fs)[:2500,1:2]).T
    item,_=worker._input(row,tmp_path,'teacher',.95,[])
    np.testing.assert_allclose(item['x'],expected,rtol=0,atol=1e-7)


def test_strict_source_verification_detects_preserved_stat_changes(tmp_path):
    import os
    from ecg_project.data.cache import checked_shard,shard,source_stats
    raw=tmp_path/'raw.dat';raw.write_bytes(b'abc');before=raw.stat()
    hashes={str(raw):file_hash(raw)}
    shard(tmp_path,'record',dict(source_sha256=hashes,source_stats=source_stats(hashes)),lambda:dict(x=np.ones(4)))
    raw.write_bytes(b'xyz');os.utime(raw,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='Stale source hashes'):checked_shard(tmp_path,'record',{},strict_sources=True)


def test_legacy_pseudo_metadata_upgrades_without_decoding(tmp_path,monkeypatch):
    from ecg_project.data import qwen_pseudo as worker
    raw=tmp_path/'raw.dat';raw.write_bytes(b'raw')
    monkeypatch.setattr(worker,'source_hashes',lambda _:{str(raw):file_hash(raw)})
    monkeypatch.setattr(worker,'load_record',lambda _:SimpleNamespace(signal=np.ones((2500,1)),fs=250,leads=['II']))
    row=dict(source='CPSC',record_id='r',patient_id='p',path=str(raw),split='train')
    item,_=worker._input(row,tmp_path,'teacher',.95,[])
    worker._result(item,tmp_path,np.zeros((4,2500),np.float32))
    meta=tmp_path/'CPSC_r.json';old=json.loads(meta.read_text())
    for k in ('row','preprocessing','source_stats'):old['identity'].pop(k)
    meta.write_text(json.dumps(old))
    (tmp_path/'cache_identity.json').write_text(json.dumps(dict(preprocessing=worker.VERSION,teacher_sha256='teacher',tau=.95,rows=[row])))
    monkeypatch.setattr(worker,'load_record',lambda _:pytest.fail('Legacy resume decoded WFDB'))
    assert worker._input(row,tmp_path,'teacher',.95,[])[0]['cached']
    assert 'source_stats' in json.loads(meta.read_text())['identity']
