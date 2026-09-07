"""Manual, partial-label segmentation caches shared by U-Net and Qwen."""
from pathlib import Path
from functools import partial
import json
import numpy as np
import pandas as pd
from ecg_project.data.catalog import ludb_split
from ecg_project.data.io import load_record,annotations
from ecg_project.data.cache import begin_cache,source_hashes,shard,publish
from ecg_project.data.policy import build_record_manifest
from ecg_project.models.segmentation import normalize
from ecg_project.processing.signal import preprocess,resample
from ecg_project.processing.parallel import ordered_map
from ecg_project.training.qtdb import manual

VERSION='delineation_250hz_manual_partial_q1c_v2'


def partial_mask(waves,n,fs=250):
    y=np.full(n,-100,np.int64)
    for w in waves:
        cls={'P':1,'QRS':2,'T':3}.get(w['wave'])
        if cls is None:continue
        a=round((w['onset'] if w['onset'] is not None else w['peak'])*250/fs)
        b=round(w['offset']*250/fs)
        y[max(0,a):min(n,b+1)]=cls
        if w['onset'] is not None:y[max(0,a-3):max(0,a)]=0
        y[min(n,b+1):min(n,b+4)]=0
    return y


def qt_rows(root):
    root=Path(root);p=root/'split.json'
    if not p.is_file():raise FileNotFoundError(f'Missing QT manual split {p}; download QTDB explicitly')
    # External_valid remains untouched. A fixed 25% of the old adaptation pool
    # supplies manual validation, never wave/sample random splitting.
    eligible=sorted([r for r in json.loads(p.read_text())['records'] if r['split']=='adapt_train'],key=lambda r:r['record_id'])
    if len(eligible)<2:raise ValueError('Need >=2 independent QT adaptation records')
    nv=max(1,round(len(eligible)*.25))
    result=[]
    for i,r in enumerate(eligible):
        path=root/(r['record_id']+'.hea')
        if not path.with_suffix('.q1c').is_file():raise FileNotFoundError(f'Manual q1c required: {path}; automatic pu annotations are forbidden')
        result.append(dict(path=str(path),record_id=r['record_id'],patient_id='QT:'+r['record_id'],source='QTDB',
            origin=r['origin'],split='valid' if i>=len(eligible)-nv else 'train',annotation='q1c'))
    return result


def _one(row,root,registry=None):
    p=Path(row['path']);source=row['source'];rec=load_record(p)
    if row['split']=='unlabeled_train' and registry is not None:
        from ecg_project.data.protection import signal_hash,assert_unprotected
        assert_unprotected(dict(**row,signal_hash=signal_hash(rec)),registry)
    extra=[p.with_suffix('.q1c')] if source=='QTDB' else [p.with_suffix('.'+l.lower()) for l in rec.leads] if source=='LUDB' else []
    identity=source_hashes(p,extra);key=source+'_'+row['record_id']
    def compute():
        x=resample(preprocess(rec.signal,rec.fs),rec.fs);xs=[];ys=[];starts=[];leads=[]
        channels=range(x.shape[1]) if source=='LUDB' else range(min(2,x.shape[1]))
        for ch in channels:
            waves=annotations(p,rec.leads[ch]) if source=='LUDB' else manual(p) if source=='QTDB' else []
            if source=='LUDB':
                if not waves:continue
                y=np.full(len(x),-100,np.int64)
                lo=round(min(w['onset'] for w in waves)*250/rec.fs);hi=round(max(w['offset'] for w in waves)*250/rec.fs)
                y[lo:hi+1]=0
                for w in waves:y[round(w['onset']*250/rec.fs):round(w['offset']*250/rec.fs)+1]={'P':1,'QRS':2,'T':3}[w['wave']]
                positions=[0]
            elif source=='QTDB':
                y=partial_mask(waves,len(x),rec.fs)
                positions=sorted(set(max(0,min(len(x)-2500,round(w['peak']*250/rec.fs)//1250*1250-1250)) for w in waves if w['wave']=='QRS'))
            else:y=np.full(len(x),-100,np.int64);positions=[0]
            for start in positions:
                if len(x)-start<2500:continue
                if source in ('LUDB','QTDB') and not (y[start:start+2500]>=0).any():continue
                xs.append(normalize(x[start:start+2500,ch:ch+1]).T);ys.append(y[start:start+2500]);starts.append(start);leads.append(rec.leads[ch])
        return dict(x=np.asarray(xs,dtype=np.float32).reshape(-1,1,2500),y=np.asarray(ys,dtype=np.int64).reshape(-1,2500),starts=starts,leads=np.asarray(leads,dtype=str))
    path=shard(root,key,dict(source_sha256=identity,metadata=row),compute)
    with np.load(path) as z:
        from ecg_project.data.protection import signal_hash
        digest=signal_hash(rec)
        entries=[dict(**row,key=key,index=i,window_start=int(s)/250,lead=str(l),source_sha256=identity,signal_hash=digest) for i,(s,l) in enumerate(zip(z['starts'],z['leads']))]
    return entries


def prepare(output='artifacts/qwen_delineation_inputs_v2',datasets=('LUDB','QTDB'),ludb_root='LUDB',qt_root='data/qtdb_external',
            catalog='artifacts/catalog.csv',unlabeled_sources=(),workers=None,validation_datasets=None):
    validation_datasets=list(validation_datasets or datasets)
    if not set(validation_datasets)<= {'LUDB','QTDB'} or 'LUDB' not in validation_datasets:raise ValueError('Invalid validation datasets')
    if not set(datasets)<= {'LUDB','QTDB'} or 'LUDB' not in datasets:raise ValueError('Supported ablations: LUDB or LUDB+QTDB')
    paths=sorted(Path(ludb_root).glob('*.hea'));mapping=ludb_split([p.stem for p in paths])
    if not paths:raise FileNotFoundError(f'Missing LUDB: {ludb_root}')
    rows=[dict(path=str(p),record_id=p.stem,patient_id='LUDB:'+p.stem,source='LUDB',split=mapping[p.stem],annotation='manual_per_lead') for p in paths if mapping[p.stem]!='test']
    if 'QTDB' in set(datasets)|set(validation_datasets):
        rows += [r for r in qt_rows(qt_root) if (r['split']=='train' and 'QTDB' in datasets) or (r['split']=='valid' and 'QTDB' in validation_datasets)]
    if unlabeled_sources:
        frame=build_record_manifest(catalog,unlabeled_sources,include_holdout=False)
        frame=frame[(frame.split=='train')&(frame.readable==True)&(frame.duration_s>=10)]
        rows += [dict(path=r.path,record_id=str(r.record_id),patient_id=r.patient_id,source=r.source_key,split='unlabeled_train',annotation='none') for r in frame.itertuples()]
    registry=None
    if unlabeled_sources:
        from ecg_project.data.protection import build_registry
        registry=build_registry(catalog,workers)
    root=begin_cache(output,dict(preprocessing=VERSION,datasets=list(datasets),validation_datasets=validation_datasets,rows=rows,unlabeled_sources=list(unlabeled_sources),protected_registry=registry))
    from ecg_project.data.cache import atomic_json
    if registry is not None:
        atomic_json(root/'protected_registry.json',registry);registry=str(root/'protected_registry.json')
    entries=[]
    for result in ordered_map(partial(_one,root=str(root),registry=registry),rows,workers):entries.extend(result)
    from ecg_project.data.catalog import assert_disjoint
    assert_disjoint(pd.DataFrame(entries))
    atomic_json(root/'manifest.json',entries)
    files=[root/'manifest.json']
    for split in ('train','valid','unlabeled_train'):
        selected=[r for r in entries if r['split']==split]
        if not selected:continue
        for field,shape,dtype in [('x',(1,2500),'float32'),('y',(2500,),'int64')]:
            tmp=root/f'{split}_{field}.partial.npy';arr=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=(len(selected),*shape))
            for i,r in enumerate(selected):
                with np.load(root/(r['key']+'.npz')) as z:arr[i]=z[field][r['index']]
            arr.flush();del arr;final=root/f'{split}_{field}.npy';tmp.replace(final);files.append(final)
    publish(root,dict(preprocessing=VERSION,fs=250,datasets=list(datasets),validation_datasets=validation_datasets,test_accessed=False,
        patients={s:sorted({r['patient_id'] for r in entries if r['split']==s and r['patient_id']}) for s in ('train','valid','unlabeled_train')},
        qt_protocol='manual q1c; both leads; unknown samples -100; external_valid excluded'),files)
    return root
