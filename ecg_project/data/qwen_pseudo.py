"""Resumable teacher logits: CPU fallback or batched CUDA inference in Colab."""
from pathlib import Path
from functools import partial,lru_cache
import json
import time
from itertools import islice
import numpy as np
import torch
from ecg_project.data.io import load_record
from ecg_project.data.catalog import file_hash
from ecg_project.data.policy import build_record_manifest
from ecg_project.data.protection import build_registry,assert_unprotected,signal_hash,registry_index
from ecg_project.data.cache import CANONICAL_DELINEATOR,require_checkpoint,begin_cache,source_hashes,shard,publish,atomic_json,source_stats,checked_shard
from ecg_project.models.segmentation import normalize
from ecg_project.processing.signal import preprocess,resample
from ecg_project.processing.parallel import ordered_map,cached_predictor

VERSION='qwen_pseudo_teacher_unet_250hz_logits_fixed10s_v2'


@lru_cache(maxsize=4)
def _legacy_owner(path,mtime_ns):
    identity=json.loads(Path(path).read_text())
    return identity,{(r['source'],str(r['record_id'])):r for r in identity['rows']}


def _upgrade_legacy_shard(root,key,row,teacher_hash,tau):
    """Old caches get one content verification, never another WFDB decode."""
    path=Path(root)/(key+'.json');old=json.loads(path.read_text())
    if 'row' in old['identity']:return
    owner=Path(root)/'cache_identity.json'
    if not owner.is_file():raise ValueError('Legacy pseudo cache needs its original cache_identity.json')
    identity,rows=_legacy_owner(str(owner),owner.stat().st_mtime_ns)
    if (identity.get('preprocessing')!=VERSION or identity.get('teacher_sha256')!=teacher_hash or
        identity.get('tau')!=tau or rows.get((row['source'],str(row['record_id'])))!=row):
        raise ValueError('Stale legacy pseudo cache identity')
    checked=checked_shard(root,key,dict(teacher_sha256=teacher_hash,tau=tau),strict_sources=True)
    old['identity']=dict(checked,row=row,preprocessing=VERSION,source_stats=source_stats(checked['source_sha256']))
    atomic_json(path,old)


def teacher_training_manifest(checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    identity=saved.get('identity',{})
    if identity.get('config',{}).get('architecture')!='unet' or 'cache' not in identity:
        raise ValueError('Teacher lacks auditable v2 training provenance; train the v2 U-Net before pseudo preparation')
    path=Path(identity['config']['input_root'])/'manifest.json'
    expected=identity['cache'].get('sha256',{}).get('manifest.json')
    if not path.is_file():raise FileNotFoundError(f'Teacher training manifest required locally: {path}')
    if not expected or file_hash(path)!=expected:raise ValueError('Teacher training manifest hash mismatch')
    rows=json.loads(path.read_text())
    training=[r for r in rows if r['split'] in ('train','unlabeled_train')]
    if not training or any(not r.get('signal_hash') for r in training):raise ValueError('Teacher training signal identities are missing')
    return training,expected


def _input(row,root,teacher_hash,tau,registry,strict_sources=False):
    key=row['source']+'_'+str(row['record_id'])
    expected=dict(teacher_sha256=teacher_hash,tau=tau,row=row,preprocessing=VERSION)
    if (Path(root)/(key+'.json')).exists():
        _upgrade_legacy_shard(root,key,row,teacher_hash,tau)
        identity=checked_shard(root,key,expected,strict_sources)
        cached_row=dict(row,signal_hash=identity['signal_hash'])
        assert_unprotected(cached_row,registry)
        return dict(row=cached_row,key=key,identity=identity,cached=True),None
    raw_row=dict(row)
    identity=source_hashes(row['path']);rec=load_record(row['path']);row=dict(row,signal_hash=signal_hash(rec))
    shard(root,key+'_source',dict(source_sha256=identity,signal_hash=row['signal_hash']),lambda:dict(checked=True))
    if not np.isfinite(rec.signal).all():return None,dict(source=row['source'],record_id=row['record_id'],reason='nonfinite_signal')
    try:assert_unprotected(row,registry)
    except ValueError as e:return None,dict(source=row['source'],record_id=row['record_id'],reason=str(e))
    item=dict(row=row,key=key,identity=dict(source_sha256=identity,source_stats=source_stats(identity),
        teacher_sha256=teacher_hash,signal_hash=row['signal_hash'],tau=tau,row=raw_row,preprocessing=VERSION))
    ch=rec.leads.index('II') if 'II' in rec.leads else 0
    # Channel-independent filters: preserve full temporal context, process only the used lead.
    x=resample(preprocess(rec.signal[:,ch:ch+1],rec.fs),rec.fs)
    if len(x)<2500:raise ValueError('Pseudo candidate shorter than 10 seconds')
    item.update(x=normalize(x[:2500]).T,lead=rec.leads[ch],cached=False)
    return item,None


def _result(item,root,logits=None,tau=.95):
    key=item['key'];row=item['row'];identity=item['identity']
    def compute():
        if logits is None or not np.isfinite(logits).all():raise FloatingPointError('Non-finite/missing teacher logits')
        probability=torch.from_numpy(logits).softmax(0).numpy();confidence=probability.max(0)
        valid=np.isfinite(item['x'][0]);valid[:125]=False;valid[-125:]=False
        target=probability.argmax(0).astype(np.int64);target[(confidence<tau)|~valid]=-100
        return dict(x=item['x'],logits=logits.astype(np.float32),confidence=confidence,target=target,valid=valid,
            lead=item['lead'],window_start=0.,window_end=10.)
    path=shard(root,key,identity,compute)
    with np.load(path) as z:
        known=z['target']>=0;counts=np.bincount(z['target'][known],minlength=4)
        result=dict(row,key=key,split='train',source_sha256=identity['source_sha256'],teacher_sha256=identity['teacher_sha256'],
            preprocessing=VERSION,window_start=0.,window_end=10.,lead=str(z['lead']),samples=len(known),accepted=int(known.sum()),
            confidence_sum=float(z['confidence'].sum()),class_counts=counts.tolist())
    return result,None


def _one(row,root,checkpoint,tau,registry,strict_sources=False):
    item,reason=_input(row,root,file_hash(checkpoint),tau,registry,strict_sources)
    if reason:return None,reason
    logits=None
    if not item['cached']:
        teacher=cached_predictor(checkpoint,'cpu').model
        with torch.inference_mode():logits=teacher(torch.from_numpy(item['x'][None])).float()[0].numpy()
    return _result(item,root,logits,tau)


def batched_results(inputs,root,model,device,batch_size,tau):
    """All CUDA work stays in the parent; spawned workers only load/filter signals."""
    iterator=iter(inputs)
    while chunk:=list(islice(iterator,batch_size)):
        fresh=[item for item,reason in chunk if reason is None and not item['cached']]
        logits=iter(())
        if fresh:
            x=torch.from_numpy(np.stack([item['x'] for item in fresh])).to(device)
            with torch.inference_mode():pred=model(x).float().cpu().numpy()
            logits=iter(pred)
        for item,reason in chunk:
            if reason:yield None,reason
            else:yield _result(item,root,None if item['cached'] else next(logits),tau)


def prepare(output='artifacts/qwen_pseudo_inputs',sources=('CPSC_EXTRA',),checkpoint=CANONICAL_DELINEATOR,
            catalog='artifacts/catalog.csv',workers=None,tau=.95,device='cpu',batch_size=64,
            registry_output='artifacts/protected_identities_v2',strict_sources=False,report_path='reports/qwen_pseudo_dataset_report.json'):
    require_checkpoint(checkpoint)
    if device not in ('cpu','cuda') or batch_size<1:raise ValueError('Invalid inference device/batch size')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: start the Jupyter kernel on an allocated GPU node and check the CUDA PyTorch installation')
    teacher_rows,teacher_manifest_hash=teacher_training_manifest(checkpoint)
    if not 0<tau<=1:raise ValueError('Confidence threshold must be in (0,1]')
    frame=build_record_manifest(catalog,sources,include_holdout=False)
    frame=frame[(frame.split=='train')&(frame.readable==True)&(frame.duration_s>=10)]
    if frame.empty:raise ValueError('No eligible pseudo training records')
    registry=build_registry(catalog,workers,output=registry_output,strict_sources=strict_sources)
    protected=registry_index(registry)
    for row in teacher_rows:assert_unprotected(row,protected)
    rows=[dict(path=r.path,source=r.source_key,record_id=str(r.record_id),patient_id=r.patient_id,split='train') for r in frame.itertuples()]
    for r in rows:assert_unprotected(r,protected)
    root=begin_cache(output,dict(preprocessing=VERSION,teacher_sha256=file_hash(checkpoint),teacher_training_manifest_sha256=teacher_manifest_hash,tau=tau,rows=rows,registry=registry,inference_device=device))
    atomic_json(root/'protected_registry.json',registry)
    results=[];excluded=[]
    start=time.monotonic();status='complete';old_tf32=torch.backends.cudnn.allow_tf32
    old_matmul=torch.backends.cuda.matmul.allow_tf32
    try:
        if device=='cuda':
            from ecg_project.models.segmentation import Predictor
            torch.cuda.reset_peak_memory_stats();torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            teacher=Predictor(checkpoint,'cuda').model
            from tqdm.auto import tqdm
            inputs=tqdm(ordered_map(partial(_input,root=str(root),teacher_hash=file_hash(checkpoint),tau=tau,
                registry=str(root/'protected_registry.json'),strict_sources=strict_sources),rows,workers),
                total=len(rows),desc='Pseudo input preprocessing',unit='record',mininterval=.2)
            iterator=batched_results(inputs,root,teacher,device,batch_size,tau)
        else:iterator=ordered_map(partial(_one,root=str(root),checkpoint=checkpoint,tau=tau,
            registry=str(root/'protected_registry.json'),strict_sources=strict_sources),rows,workers)
        from tqdm.auto import tqdm
        for row,reason in tqdm(iterator,total=len(rows),desc='GPU inference/write' if device=='cuda' else 'CPU inference/write',unit='record'):
            if row:results.append(row)
            else:excluded.append(reason)
    except BaseException:status='interrupted';raise
    finally:
        atomic_json(root/'preparation_run.json',dict(status=status,device=device,batch_size=batch_size if device=='cuda' else 1,
            records_completed=len(results),records_per_second=len(results)/max(time.monotonic()-start,1e-9),
            max_memory_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            max_memory_reserved=torch.cuda.max_memory_reserved() if device=='cuda' else 0))
        torch.backends.cudnn.allow_tf32=old_tf32;torch.backends.cuda.matmul.allow_tf32=old_matmul
    # Exact train duplicates do not receive additional sampling weight.
    seen=set();unique=[]
    for r in results:
        if r['signal_hash'] in seen:excluded.append(dict(source=r['source'],record_id=r['record_id'],reason='duplicate train signal'))
        else:seen.add(r['signal_hash']);unique.append(r)
    if not unique:raise ValueError('No pseudo samples remain after protected-cohort exclusion')
    atomic_json(root/'manifest.json',unique);atomic_json(root/'excluded.json',excluded);atomic_json(root/'protected_registry.json',registry)
    total=sum(r['samples'] for r in unique);accepted=sum(r['accepted'] for r in unique)
    stats=dict(total_records=len(unique),total_samples=total,accepted_pseudo_fraction=accepted/total,ignored_fraction=1-accepted/total,
        mean_teacher_confidence=sum(r['confidence_sum'] for r in unique)/total,
        class_counts=np.sum([r['class_counts'] for r in unique],axis=0).tolist(),
        by_source={s:dict(records=sum(r['source']==s for r in unique),accepted=sum(r['accepted'] for r in unique if r['source']==s),
            samples=sum(r['samples'] for r in unique if r['source']==s),
            mean_teacher_confidence=sum(r['confidence_sum'] for r in unique if r['source']==s)/sum(r['samples'] for r in unique if r['source']==s),
            class_counts=np.sum([r['class_counts'] for r in unique if r['source']==s],axis=0).tolist()) for s in sorted({r['source'] for r in unique})})
    publish(root,dict(preprocessing=VERSION,teacher_sha256=file_hash(checkpoint),source_datasets=list(sources),confidence_threshold=tau,
        inference_device=device,inference_precision='float32',
        teacher_training_manifest_sha256=teacher_manifest_hash,teacher_training_overlap_checked=True,
        temperature_scaled=False,stored='unscaled_logits',class_names=['background','P','QRS','T'],
        excluded_cohorts='LUDB valid/test; QTDB manual valid and external; PTB fold9/10; Georgia; StP',**stats),
        [root/'manifest.json',root/'excluded.json',root/'protected_registry.json',*[root/(r['key']+'.npz') for r in unique]])
    atomic_json(report_path,stats)
    return root
