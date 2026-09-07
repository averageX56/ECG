"""CPU-only U-Net teacher inference, resumable logits and confidence masks."""
from pathlib import Path
from functools import partial
import json
import numpy as np
import torch
from ecg_project.data.io import load_record
from ecg_project.data.catalog import file_hash
from ecg_project.data.policy import build_record_manifest
from ecg_project.data.protection import build_registry,assert_unprotected,signal_hash,registry_index
from ecg_project.data.cache import CANONICAL_DELINEATOR,require_checkpoint,begin_cache,source_hashes,shard,publish,atomic_json
from ecg_project.models.segmentation import normalize
from ecg_project.processing.signal import preprocess,resample
from ecg_project.processing.parallel import ordered_map,cached_predictor

VERSION='qwen_pseudo_teacher_unet_250hz_logits_fixed10s_v2'


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


def _one(row,root,checkpoint,tau,registry):
    rec=load_record(row['path']);identity=source_hashes(row['path']);row=dict(row,signal_hash=signal_hash(rec))
    key=row['source']+'_'+str(row['record_id'])
    shard(root,key+'_source',dict(source_sha256=identity,signal_hash=row['signal_hash']),lambda:dict(checked=True))
    if not np.isfinite(rec.signal).all():return None,dict(source=row['source'],record_id=row['record_id'],reason='nonfinite_signal')
    try:assert_unprotected(row,registry)
    except ValueError as e:return None,dict(source=row['source'],record_id=row['record_id'],reason=str(e))
    def compute():
        x=resample(preprocess(rec.signal,rec.fs),rec.fs)
        if len(x)<2500:raise ValueError('Pseudo candidate shorter than 10 seconds')
        # Lead selection deterministic, no label-aware best-lead search.
        ch=rec.leads.index('II') if 'II' in rec.leads else 0
        signal=normalize(x[:2500,ch:ch+1]).T[None]
        teacher=cached_predictor(checkpoint,'cpu').model
        with torch.no_grad():logits=teacher(torch.from_numpy(signal)).float()[0].numpy()
        if not np.isfinite(logits).all():raise FloatingPointError('Non-finite teacher logits')
        probability=torch.from_numpy(logits).softmax(0).numpy();confidence=probability.max(0)
        valid=np.isfinite(signal[0,0]);valid[:125]=False;valid[-125:]=False
        target=probability.argmax(0).astype(np.int64);target[(confidence<tau)|~valid]=-100
        return dict(x=signal[0],logits=logits.astype(np.float32),confidence=confidence,target=target,valid=valid,
            lead=rec.leads[ch],window_start=0.,window_end=10.)
    path=shard(root,key,dict(source_sha256=identity,teacher_sha256=file_hash(checkpoint),signal_hash=row['signal_hash'],tau=tau),compute)
    with np.load(path) as z:
        known=z['target']>=0;counts=np.bincount(z['target'][known],minlength=4)
        result=dict(row,key=key,split='train',source_sha256=identity,teacher_sha256=file_hash(checkpoint),
            preprocessing=VERSION,window_start=0.,window_end=10.,lead=str(z['lead']),samples=len(known),accepted=int(known.sum()),
            confidence_sum=float(z['confidence'].sum()),class_counts=counts.tolist())
    return result,None


def prepare(output='artifacts/qwen_pseudo_inputs',sources=('CPSC_EXTRA',),checkpoint=CANONICAL_DELINEATOR,
            catalog='artifacts/catalog.csv',workers=None,tau=.95):
    require_checkpoint(checkpoint)
    teacher_rows,teacher_manifest_hash=teacher_training_manifest(checkpoint)
    if not 0<tau<=1:raise ValueError('Confidence threshold must be in (0,1]')
    frame=build_record_manifest(catalog,sources,include_holdout=False)
    frame=frame[(frame.split=='train')&(frame.readable==True)&(frame.duration_s>=10)]
    if frame.empty:raise ValueError('No eligible pseudo training records')
    registry=build_registry(catalog,workers)
    protected=registry_index(registry)
    for row in teacher_rows:assert_unprotected(row,protected)
    rows=[dict(path=r.path,source=r.source_key,record_id=str(r.record_id),patient_id=r.patient_id,split='train') for r in frame.itertuples()]
    for r in rows:assert_unprotected(r,protected)
    root=begin_cache(output,dict(preprocessing=VERSION,teacher_sha256=file_hash(checkpoint),teacher_training_manifest_sha256=teacher_manifest_hash,tau=tau,rows=rows,registry=registry))
    atomic_json(root/'protected_registry.json',registry)
    results=[];excluded=[]
    for row,reason in ordered_map(partial(_one,root=str(root),checkpoint=checkpoint,tau=tau,registry=str(root/'protected_registry.json')),rows,workers):
        if row:results.append(row)
        else:excluded.append(reason)
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
        teacher_training_manifest_sha256=teacher_manifest_hash,teacher_training_overlap_checked=True,
        temperature_scaled=False,stored='unscaled_logits',class_names=['background','P','QRS','T'],
        excluded_cohorts='LUDB valid/test; QTDB manual valid and external; PTB fold9/10; Georgia; StP',**stats),
        [root/'manifest.json',root/'excluded.json',root/'protected_registry.json',*[root/(r['key']+'.npz') for r in unique]])
    atomic_json('reports/qwen_pseudo_dataset_report.json',stats)
    return root
