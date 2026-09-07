"""Shared multi-source, deterministic ten-second neural input preparation."""
from pathlib import Path
from functools import partial
import json
import numpy as np
from scipy.signal import resample_poly,iirnotch,butter,filtfilt,medfilt
from fractions import Fraction
from ecg_project.data.io import load_record
from ecg_project.data.policy import build_record_manifest,POLICY_VERSION
from ecg_project.data.cache import begin_cache,source_hashes,shard,publish,atomic_json
from ecg_project.processing.parallel import ordered_map
from ecg_project.processing.hubert import VERSION_V2,LEADS,select_window,prepare_signal_v2

FOUNDER_VERSION='founder_I_500_notch50_bp067_40_median201_zscore_fixed10s_v2'


def founder_array(rec):
    selected,_=select_window(rec);x=selected.signal[:,selected.leads.index('I')].astype(float)
    ratio=Fraction(500/float(selected.fs)).limit_denominator(10000)
    x=resample_poly(x,ratio.numerator,ratio.denominator)[:5000]
    b,a=iirnotch(50,30,500);x=filtfilt(b,a,x)
    b,a=butter(4,[.67,40],btype='bandpass',fs=500);x=filtfilt(b,a,x)
    x=x-medfilt(x,201)
    return ((x-x.mean())/(x.std()+1e-8)).astype(np.float32)


def _prepare(row,kind,root,train_windows):
    identity=source_hashes(row['path']);rec=load_record(row['path'])
    shard(root,row['source_key']+'_'+str(row['record_id'])+'_source',dict(source_sha256=identity),lambda:dict(checked=True))
    duration=len(rec.signal)/rec.fs;reason=''
    if duration<10:reason='shorter_than_10_seconds'
    elif kind=='hubert' and (len(rec.leads)!=12 or set(rec.leads)!=set(LEADS)):reason='missing_canonical_12_leads'
    elif kind=='founder' and 'I' not in rec.leads:reason='missing_lead_I'
    elif not np.isfinite(rec.signal).all():reason='nonfinite_signal'
    if reason:return [],[dict(source=row['source'],record_id=row['record_id'],original_duration=duration,exclusion_reason=reason,source_sha256=identity)]
    starts=[0.] if row['split']!='train' or train_windows==1 else np.linspace(0,duration-10,train_windows).tolist()
    results=[]
    for i,start in enumerate(sorted(set(starts))):
        key=row['source_key']+'_'+str(row['record_id'])+f'_w{i}'
        selected,window=select_window(rec,start)
        def compute():return dict(signal=prepare_signal_v2(selected)[0] if kind=='hubert' else founder_array(selected))
        p=shard(root,key,dict(source_sha256=identity,start=start,kind=kind),compute)
        results.append(dict(row,**window,lead_availability=';'.join(rec.leads),exclusion_reason='',cache_key=key,
            source_sha256=json.dumps(identity,sort_keys=True),shard=str(p).replace('\\','/')))
    return results,[]


def prepare(kind,output=None,catalog='artifacts/catalog.csv',sources=None,workers=None,train_windows=1):
    if kind not in ('hubert','founder') or train_windows<1:raise ValueError('Invalid input configuration')
    version=VERSION_V2 if kind=='hubert' else FOUNDER_VERSION
    frame=build_record_manifest(catalog,sources,include_holdout=False)
    frame=frame[(frame.readable==True)&(frame.has_labels==True)].reset_index(drop=True)
    if frame.empty:raise ValueError('No supervised train/valid records')
    root=begin_cache(output or f'artifacts/{kind}_inputs_v2',dict(preprocessing=version,policy=POLICY_VERSION,
        rows=frame.to_json(orient='records'),train_windows=train_windows))
    rows=[];excluded=[]
    for records,fail in ordered_map(partial(_prepare,kind=kind,root=str(root),train_windows=train_windows),frame.to_dict('records'),workers):
        rows.extend(records);excluded.extend(fail)
    if not rows:raise ValueError('All records excluded')
    import pandas as pd
    manifest=pd.DataFrame(rows);manifest.to_csv(root/'manifest.csv',index=False)
    atomic_json(root/'exclusions.json',excluded)
    shape=(2,6000) if kind=='hubert' else (5000,)
    tmp=root/'signals.partial.npy';signals=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(len(rows),*shape))
    for i,row in enumerate(rows):
        with np.load(row['shard']) as z:signals[i]=z['signal']
    signals.flush();del signals;tmp.replace(root/'signals.npy')
    from ecg_project.data.catalog import file_hash
    publish(root,dict(preprocessing=version,policy=POLICY_VERSION,records=len(rows),test_accessed=False,
        signals_sha256=file_hash(root/'signals.npy'),
        evaluation_status='pretraining_source_overlap; not independent external validation' if kind=='hubert' else 'source holdouts remain excluded'),
        [root/'signals.npy',root/'manifest.csv',root/'exclusions.json'])
    return root
