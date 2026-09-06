"""QTDB manual partial labels, source-disjoint adaptation and external validation."""
from pathlib import Path
from collections import defaultdict
import json
import time
import numpy as np
import torch
import wfdb
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,ConcatDataset
from ecg_project.data.io import load_record
from ecg_project.models.segmentation import Predictor,Delineator,normalize,prepare_ludb
from ecg_project.processing.signal import preprocess
from ecg_project.evaluation.metrics import match_events
from ecg_project.utils import save_json,seed_all
from ecg_project.data.catalog import file_hash
from tqdm.auto import tqdm

def manual(path):
    """T onset is frequently unannotated and is retained as None, never invented."""
    a=wfdb.rdann(str(Path(path).with_suffix('')),'q1c');waves=[];onset=None;peak=None;wave=None
    for s,symbol in zip(a.sample,a.symbol):
        if symbol=='(':onset=int(s);peak=None;wave=None
        elif symbol in ['p','N','t','u']:
            peak=int(s);wave={'p':'P','N':'QRS','t':'T','u':'U'}[symbol]
        elif symbol==')':
            if peak is not None and wave!='U':waves.append(dict(wave=wave,onset=onset,peak=peak,offset=int(s)))
            onset=None;peak=None;wave=None
    return waves

def entries(root,split):
    return [r for r in json.loads((Path(root)/'split.json').read_text())['records'] if r['split']==split]

def evaluate(root='data/qtdb_external',checkpoint='artifacts/delineator.pt',output='reports/qtdb_baseline.json'):
    seed_all();predictor=Predictor(checkpoint);detail=[]
    rows = entries(root, "external_valid")

    for row in tqdm(
        rows,
        desc="QT external validation",
        unit="record",
    ):
        p=Path(root)/(row['record_id']+'.hea');ref=manual(p)
        lo=max(0,min(w['peak'] for w in ref)-1250);hi=max(w['offset'] for w in ref)+1250
        rec=load_record(p,start=lo,stop=hi);pred=predictor.predict(rec.signal,rec.fs)
        for channel,q in enumerate(pred):
            # Both channels are reported against the same multilead manual reference. No oracle best-lead selection.
            for wave in ['P','QRS','T']:
                r=[w for w in ref if w['wave']==wave];v=[{**w,**{k:w[k]+lo for k in ['onset','peak','offset']}} for w in q if w['wave']==wave]
                pairs=match_events([w['peak'] for w in r],[w['peak'] for w in v],.15*rec.fs)
                errors={k:[(v[j][k]-r[i][k])*1000/rec.fs for i,j in pairs if r[i][k] is not None] for k in ['onset','peak','offset']}
                widths=[((v[j]['offset']-v[j]['onset'])-(r[i]['offset']-r[i]['onset']))*1000/rec.fs for i,j in pairs if r[i]['onset'] is not None]
                detail.append(dict(record=row['record_id'],channel=channel,wave=wave,reference_n=len(r),matched_n=len(pairs),
                                   errors_ms=errors,width_errors_ms=widths))
        print('QT evaluated',row['record_id'],flush=True)
    summary={}
    for channel in [0,1]:
        for wave in ['P','QRS','T']:
            d=[r for r in detail if r['channel']==channel and r['wave']==wave];n=sum(r['reference_n'] for r in d);m=sum(r['matched_n'] for r in d)
            metrics=dict(reference_n=n,matched_n=m,recall=m/n if n else None,precision=None)
            for key in ['onset','peak','offset']:
                e=np.array([v for r in d for v in r['errors_ms'][key]])
                metrics[key]=dict(n=len(e),mae_ms=np.mean(abs(e)) if len(e) else None,bias_ms=np.mean(e) if len(e) else None)
            e=np.array([v for r in d for v in r['width_errors_ms']]);metrics['width_mae_ms']=np.mean(abs(e)) if len(e) else None
            summary[f'channel{channel}/{wave}']=metrics
    report=dict(checkpoint=checkpoint,split='external_valid',n_records=len(entries(root,'external_valid')),summary=summary,details=detail,
       protocol='Manual q1c; both leads against shared reference; no best-lead selection. Partial annotation: precision/whole-record F1 not identifiable. T-onset only where annotated. Selected mostly normal beats; not an arrhythmia classification benchmark.')
    save_json(output,report);return report

def qt_training(root):
    xs=[];ys=[];provenance=[]
    for row in tqdm(entries(root,'adapt_train'), desc="QT training", unit="record"):
        p=Path(root)/(row['record_id']+'.hea');r=load_record(p);waves=manual(p)
        x=preprocess(r.signal,r.fs);y=np.full(len(x),-100,np.int64)
        for w in waves:
            cls={'P':1,'QRS':2,'T':3}[w['wave']]
            a=w['onset'] if w['onset'] is not None else w['peak']
            y[a:w['offset']+1]=cls
            # Only local boundary-adjacent samples get background labels. The unannotated T onset remains ignored.
            if w['onset'] is not None:y[max(0,a-3):a]=0
            y[w['offset']+1:w['offset']+4]=0
        qs=[w['peak'] for w in waves if w['wave']=='QRS']
        starts=sorted(set(max(0,min(len(x)-2500,(p//1250)*1250-1250)) for p in qs))
        for start in starts:
            z=normalize(x[start:start+2500,0:1]);xs.append(z.T);ys.append(y[start:start+2500])
        provenance.append(dict(record=row['record_id'],origin=row['origin'],signal_hash=file_hash(p.with_suffix('.dat')),manual_hash=file_hash(p.with_suffix('.q1c'))))
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys)),provenance

def train(root='data/qtdb_external',initial='artifacts/delineator_transfer.pt',output='artifacts/delineator_qt.pt',epochs=10,minutes=15):
    seed_all();start=time.monotonic();device='cuda' if torch.cuda.is_available() else 'cpu'
    qx,qy,provenance=qt_training(root);lx,ly=prepare_ludb('LUDB','train');vx,vy=prepare_ludb('LUDB','valid')
    labeled=DataLoader(TensorDataset(lx,ly),batch_size=16,shuffle=True)
    external=DataLoader(TensorDataset(qx,qy),batch_size=8,shuffle=True)
    valid=DataLoader(TensorDataset(vx,vy),batch_size=16)
    model=Delineator().to(device);model.load_state_dict(torch.load(initial,map_location=device,weights_only=True)['state_dict'])
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);lossfn=nn.CrossEntropyLoss(ignore_index=-100)
    best=-1;history=[];qi=iter(external)
    for epoch in tqdm(range(epochs), desc="QT training"):
        model.train();losses=[]
        for x,y in tqdm(labeled, desc="QT training batch", unit="batch"):
            if time.monotonic()-start>minutes*60:break
            try:ux,uy=next(qi)
            except StopIteration:qi=iter(external);ux,uy=next(qi)
            loss=lossfn(model(x.to(device)),y.to(device))+.25*lossfn(model(ux.to(device)),uy.to(device))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval();cm=np.zeros((4,4),np.int64)
        with torch.no_grad():
            for x,y in tqdm(valid, desc="QT validation", unit="record"):
                pp=model(x.to(device)).argmax(1).cpu().numpy();yy=y.numpy();ok=yy>=0
                cm+=np.bincount(yy[ok]*4+pp[ok],minlength=16).reshape(4,4)
        dice=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
        history.append(dict(epoch=epoch+1,loss=np.mean(losses),valid_dice=dice.tolist(),seconds=time.monotonic()-start));print(history[-1],flush=True)
        # New external_valid is not used to select epochs. It stays a post-training external check.
        if dice[1:].mean()>best:
            best=dice[1:].mean();torch.save(dict(state_dict=model.cpu().state_dict(),fs=250,architecture='Delineator',
                method='LUDB+QT_partial_supervision',epoch=epoch+1,valid_dice=dice.tolist()),output);model.to(device)
        if time.monotonic()-start>minutes*60:break
    save_json(Path(output).with_suffix('.history.json'),history)
    save_json(Path(output).with_suffix('.provenance.json'),dict(initial_hash=file_hash(initial),qt_train=provenance,
         selection='LUDB validation Dice only; external QT validation never enters training/early stopping',
         missing_T_onset='ignored',channel='fixed channel0, manual reference derived from both leads'))
    save_json(Path(output).with_suffix('.budget.json'),dict(seconds=time.monotonic()-start,device=device,
         max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
