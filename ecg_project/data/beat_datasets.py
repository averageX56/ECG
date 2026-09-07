"""Source adapters for supervised AAMI beats and strictly train-only SSL."""
from pathlib import Path
from functools import partial
from collections import Counter
import json
import numpy as np
import pandas as pd
import wfdb
from ecg_project.data.io import load_record,mit_annotations
from ecg_project.data.catalog import file_hash
from ecg_project.data.cache import CANONICAL_DELINEATOR,require_checkpoint,source_hashes,begin_cache,shard,publish,atomic_json
from ecg_project.data.policy import build_record_manifest,source_name
from ecg_project.processing.parallel import ordered_map,cached_predictor
from ecg_project.processing.signal import preprocess,rpeaks
from ecg_project.processing.features import beat_features
from ecg_project.evaluation.metrics import match_events
from ecg_project.training.beats import load_mit,split_for,PACED,CLASSES

VERSION='beat_features_aami_sources_v2'
AAMI={**dict.fromkeys(['N','L','R','e','j'],'N'),**dict.fromkeys(['A','a','J','S'],'S'),
      **dict.fromkeys(['V','E'],'V'),'F':'F',**dict.fromkeys(['/','f','Q'],'Q')}


def supervised_rows(sources=('MIT',),data_root='data'):
    roots={'MIT':'mit-bih','SVDB':'svdb','INCART':'incart'};rows=[]
    if 'MIT' not in sources:raise ValueError('Include MIT for the historical train/valid/test comparison')
    for source in sources:
        if source not in roots:raise ValueError(f'Unknown beat source {source}')
        root=Path(data_root)/roots[source];paths=sorted(root.glob('*.csv' if source=='MIT' else '*.hea'))
        if not paths:raise FileNotFoundError(f'Missing {source} in {root}; download dataset explicitly, never substitute a different source')
        mapping={}
        if (root/'patients.csv').exists():mapping=pd.read_csv(root/'patients.csv',dtype=str).set_index('record_id').patient_id.to_dict()
        for p in paths:
            if source=='MIT' and int(p.stem) in PACED:continue
            patient=('201_202' if p.stem in ('201','202') else p.stem) if source=='MIT' else mapping.get(p.stem,'')
            rows.append(dict(path=str(p),record_id=p.stem,source=source,record=source+'_'+p.stem,
                patient=source+':'+patient if patient else source+':unknown_entire_train',
                patient_id=source+':'+patient if patient else '',split=split_for(p.stem) if source=='MIT' else 'train',
                identity_limitation='' if patient else 'Unknown patient: entire source train; no fragment splits',supervised=True))
    return rows


def _prepare(row,root,checkpoint,registry=None):
    p=Path(row['path']);is_mit=row['source']=='MIT';supervised=row['supervised']
    annotation=p.with_name(p.stem+'annotations.txt') if is_mit else p.with_suffix('.atr')
    extra=[annotation] if supervised else []
    if is_mit:extra.append(Path('artifacts/mit_headers')/(p.stem+'.hea'))
    if registry is not None:
        from ecg_project.data.protection import signal_hash,assert_unprotected
        row=dict(row,signal_hash=signal_hash(load_record(p)))
        assert_unprotected(row,registry)
    identity=dict(source_sha256=source_hashes(p,extra),checkpoint_sha256=file_hash(checkpoint),metadata=row)
    metadata={}
    def compute():
        rec=load_mit(p) if is_mit else load_record(p)
        lead='MLII' if is_mit else 'II' if 'II' in rec.leads else rec.leads[0]
        x=rec.signal[:,rec.leads.index(lead)];predictor=cached_predictor(checkpoint,'cpu')
        waves=predictor.predict(x,rec.fs)[0];peaks,_=rpeaks(preprocess(x,rec.fs),rec.fs)
        crops,f,ids=beat_features(x,peaks,waves,rec.fs)
        refs=[]
        if supervised:
            if is_mit:refs=mit_annotations(annotation)
            else:
                ann=wfdb.rdann(str(p.with_suffix('')),'atr');refs=[dict(sample=int(s),symbol=c) for s,c in zip(ann.sample,ann.symbol)]
            refs=[a for a in refs if a['symbol'] in AAMI]
        pairs=match_events([a['sample'] for a in refs],peaks,.15*rec.fs)
        original={j:refs[i]['symbol'] for i,j in pairs}
        symbols=np.array([original.get(int(i),'') for i in ids],dtype=str)
        labels=np.array([CLASSES.index(AAMI[s]) if s else -1 for s in symbols])
        metadata.update(lead=lead,references=len(refs),detected=len(peaks),cropped=len(ids),
            reference_classes=dict(Counter(AAMI[a['symbol']] for a in refs)))
        return dict(waveform=crops,interval=f,labels=labels,peaks=peaks[ids],original_symbols=symbols,
            model_hash=identity['checkpoint_sha256'],source=row['source'],summary=json.dumps(metadata))
    path=shard(root,row['record'],identity,compute)
    with np.load(path) as z:metadata=json.loads(str(z['summary']))
    return dict(**row,**metadata,source_sha256=identity['source_sha256'],checkpoint_sha256=identity['checkpoint_sha256'])


def prepare(output='artifacts/beat_features_v2',sources=('MIT',),checkpoint=CANONICAL_DELINEATOR,data_root='data',workers=None,
            catalog='artifacts/catalog.csv',unlabeled=False):
    require_checkpoint(checkpoint)  # Before ProcessPool and before scanning raw records.
    registry=None
    if unlabeled:
        frame=build_record_manifest(catalog,sources,include_holdout=False)
        frame=frame[(frame.split=='train')&(frame.readable==True)]
        if frame.empty:raise ValueError('No eligible train-only SSL records')
        rows=[dict(path=r.path,record_id=str(r.record_id),record='SSL_'+r.source_key+'_'+str(r.record_id),source=r.source_key,
            patient_id=r.patient_id,patient=r.patient_id or r.source_key+':unknown_entire_train',split='train',supervised=False) for r in frame.itertuples()]
        from ecg_project.data.protection import build_registry,registry_index,assert_unprotected
        registry=build_registry(catalog,workers)
        protected=registry_index(registry)
        for row in rows:assert_unprotected(row,protected)
    else:rows=supervised_rows(sources,data_root)
    root=begin_cache(output,dict(preprocessing=VERSION,sources=list(sources),checkpoint_sha256=file_hash(checkpoint),rows=rows,unlabeled=unlabeled,protected_registry=registry))
    if registry is not None:
        atomic_json(root/'protected_registry.json',registry);registry=str(root/'protected_registry.json')
    manifest=list(ordered_map(partial(_prepare,root=str(root),checkpoint=checkpoint,registry=registry),rows,workers))
    atomic_json(root/'manifest.json',manifest)
    support={split:{c:sum(r['reference_classes'].get(c,0) for r in manifest if r['split']==split) for c in CLASSES} for split in ('train','valid','test')}
    publish(root,dict(preprocessing=VERSION,checkpoint_sha256=file_hash(checkpoint),class_support=support,
        classes_excluded_from_selection=['Q'],q_policy='retained explicitly; train with or without Q via ClusterConfig.include_q',
        source_policy='MIT historical split; SVDB/INCART entire train; no random fragment splitting',
        limitation='INCART and St Petersburg may share source records; St Petersburg is not independent for an INCART-trained beat model',
        unlabeled=unlabeled),[root/'manifest.json',*[root/(r['record']+'.npz') for r in manifest]])
    return root
