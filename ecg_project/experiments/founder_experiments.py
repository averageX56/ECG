"""ECGFounder transfer: frozen embeddings and final-stage fine-tuning on PTB.

Uses only existing PTB-XL train and validation pilot records, lead I, 10 seconds.
Does not read the test split. Official net1d.py is downloaded separately/reviewed.
"""
from pathlib import Path
import importlib.util
import time
import json
import copy
import joblib
import numpy as np
import pandas as pd
from scipy.signal import iirnotch,butter,filtfilt,medfilt,resample_poly
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from ecg_project.data.io import load_record
from ecg_project.data.catalog import TARGETS,file_hash
from ecg_project.evaluation.metrics import multilabel_metrics,thresholds_on_validation
from ecg_project.utils import seed_all,save_json
from tqdm.auto import tqdm


def founder_signal(path):
    rec=load_record(str(path).replace('\\','/'));x=rec.signal[:,rec.leads.index('I')].astype(float)
    if rec.fs!=500:
        from fractions import Fraction
        ratio=Fraction(500/int(rec.fs));x=resample_poly(x,ratio.numerator,ratio.denominator)
    if len(x)!=5000:raise ValueError('Expected exactly ten seconds')
    b,a=iirnotch(50,30,500);x=filtfilt(b,a,x)
    b,a=butter(4,[.67,40],btype='bandpass',fs=500);x=filtfilt(b,a,x)
    x=x-medfilt(x,kernel_size=201);return ((x-x.mean())/(x.std()+1e-8)).astype(np.float32)


def prepare_full_ptb(output='artifacts/founder_full_inputs'):
    """Entire PTB training folds 1-8 and validation fold 9, never fold 10."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv('artifacts/catalog.csv')
    frame=frame[(frame.source=='WFDB_PTB-XL') & frame.fold.between(1,9) & (frame.readable==True)].copy().reset_index(drop=True)
    frame['split']=np.where(frame.fold==9,'valid','train')
    if not len(frame):raise ValueError('Run audit with official PTB patient metadata first')
    signals=np.lib.format.open_memmap(out/'signals.npy',mode='w+',dtype='float32',shape=(len(frame),5000))
    from ecg_project.processing.parallel import ordered_map
    for i,signal in enumerate(tqdm(ordered_map(founder_signal,frame.path.tolist()),total=len(frame),desc="Processing PTB signals")):
        signals[i]=signal
    signals.flush();frame.to_csv(out/'manifest.csv',index=False)
    return out


def prepare(output='artifacts/founder_experiments'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv('artifacts/record_features_qt/manifest.csv')
    frame=frame[(frame.source=='WFDB_PTB-XL') & frame.split.isin(['train','valid'])].reset_index(drop=True)
    if set(frame.loc[frame.split=='train','patient_id']) & set(frame.loc[frame.split=='valid','patient_id']):
        raise ValueError('Patient overlap')
    signals=[];features=[]
    for i,row in tqdm(frame.iterrows(),total=len(frame),desc="Processing founder inputs"):
        rec=load_record(row.path);x=rec.signal[:,rec.leads.index('I')].astype(float)
        if rec.fs!=500:
            from fractions import Fraction
            ratio=Fraction(500/int(rec.fs));x=resample_poly(x,ratio.numerator,ratio.denominator)
        if len(x)!=5000:raise ValueError('Expected exactly ten seconds')
        # Exact official single-lead filtering: notch, .67-40Hz, median baseline.
        b,a=iirnotch(50,30,500);x=filtfilt(b,a,x)
        b,a=butter(4,[.67,40],btype='bandpass',fs=500);x=filtfilt(b,a,x)
        x=x-medfilt(x,kernel_size=201);x=(x-x.mean())/(x.std()+1e-8)
        signals.append(x.astype(np.float32))
        with np.load(str(row.cache).replace('\\','/')) as z:
            mask=np.array([str(n).startswith('I/') for n in z['interval_names']]);features.append(z['interval'][mask])
        if (i+1)%200==0:print('Founder inputs',i+1,flush=True)
    np.savez_compressed(out/'inputs.npz',signal=np.stack(signals),interval=np.stack(features))
    frame.to_csv(out/'manifest.csv',index=False)


def load_founder():
    source=Path('artifacts/ecgfounder/net1d.py')
    spec=importlib.util.spec_from_file_location('official_ecgfounder_net',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.Net1D(in_channels=1,base_filters=64,ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16,stride=2,groups_width=16,n_classes=150,use_bn=False,use_do=False)
    # Restrict unpickling to tensor checkpoint types; never use weights_only=False.
    with torch.serialization.safe_globals([(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,np.dtypes.Float64DType]):
        ckpt=torch.load('artifacts/ecgfounder/1_lead_ECGFounder.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    return model


def run(output='artifacts/founder_experiments',minutes=25):
    seed_all();out=Path(output);start=time.monotonic()
    if not (out/'inputs.npz').exists():prepare(output)
    from ecg_project.training.local_budget import read_ledger
    ledger=read_ledger()
    # Reserve both current jobs if the representation experiment has not entered ledger yet.
    reserve=35*60 if 'artifacts/representation_experiments/budget.json' not in ledger['runs'] else 0
    if ledger['total_seconds']+reserve+minutes*60>10800:raise RuntimeError('Training budget would be exceeded; ask user for approval.')
    device='cuda' if torch.cuda.is_available() else 'cpu';results={};history=[]
    def guard():
        if time.monotonic()-start>minutes*60:raise TimeoutError('Founder experiment reservation reached')
    try:
        frame=pd.read_csv(out/'manifest.csv');tr=frame.split.to_numpy()=='train';va=frame.split.to_numpy()=='valid'
        with np.load(out/'inputs.npz') as z:signals=z['signal'];fi=z['interval']
        classes=[c for c in TARGETS if frame.loc[tr,c].sum()>=20 and (1-frame.loc[tr,c]).sum()>=20]
        y=frame[classes].to_numpy(dtype=np.float32)
        def evaluate(name,prob):
            thresholds=thresholds_on_validation(y[va],prob[va])
            result=multilabel_metrics(y[va],prob[va],classes,thresholds)
            results[name]=result;save_json(out/'metrics.json',results)
            np.savez_compressed(out/(name+'_valid.npz'),truth=y[va],probability=prob[va],classes=np.array(classes))
            print(name,'validation AUROC',result['macro_auroc'],flush=True)
            return result['macro_auroc']
        def hgb(name,x):
            models=[];prob=np.zeros_like(y)
            for j,c in tqdm(enumerate(classes),total=len(classes),desc=f"Training {name}"):
                guard();m=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,early_stopping=False,class_weight='balanced',l2_regularization=3,random_state=42)
                m.fit(x[tr],y[tr,j]);prob[:,j]=m.predict_proba(x)[:,1];models.append(m)
            joblib.dump(dict(models=models,classes=classes),out/(name+'.joblib'));evaluate(name,prob)
        hgb('lead_I_intervals',fi)
        model=load_founder().to(device).eval()
        for p in model.parameters():p.requires_grad=False
        # Cache activations before last stage. Frozen trunk saves memory and compute.
        activation=[]
        with torch.no_grad():
            for i in tqdm(range(0,len(signals),4),desc="Caching founder activations"):
                x=torch.from_numpy(signals[i:i+4,None]).to(device)
                z=model.first_activation(model.first_conv(x))
                for stage in model.stage_list[:-1]:z=stage(z)
                activation.append(z.cpu())
                if i%100==0:print('Founder frozen trunk',i,'/',len(signals),flush=True)
        activation=torch.cat(activation);torch.save(activation,out/'trunk_activations.pt')
        last=model.stage_list[-1]
        with torch.no_grad():emb=torch.cat([last(a.to(device)).mean(-1).cpu() for a in activation.split(16)]).numpy()
        np.save(out/'frozen_embeddings.npy',emb)
        hgb('founder_frozen',emb);hgb('founder_frozen_intervals',np.concatenate([emb,fi],1))
        head=nn.Linear(1024,len(classes)).to(device)
        for p in last.parameters():p.requires_grad=True
        optimizer=torch.optim.AdamW([{'params':last.parameters(),'lr':1e-5},{'params':head.parameters(),'lr':1e-3}],weight_decay=.01)
        pos=torch.tensor(np.sqrt((1-y[tr]).sum(0)/np.maximum(y[tr].sum(0),1)),device=device)
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos)
        loader=DataLoader(TensorDataset(activation[tr],torch.from_numpy(y[tr])),batch_size=16,shuffle=True)
        best=-1;beststate=None
        for epoch in tqdm(range(12),desc="Training founder"):
            last.train();head.train();losses=[]
            for a,label in tqdm(loader,desc="Training batches"):
                guard();logits=head(last(a.to(device)).mean(-1));loss=criterion(logits,label.to(device))
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(last.parameters())+list(head.parameters()),1);optimizer.step();losses.append(loss.item())
            last.eval();head.eval()
            with torch.no_grad():prob=torch.cat([head(last(a.to(device)).mean(-1)).sigmoid().cpu() for a in activation.split(16)]).numpy()
            metric=multilabel_metrics(y[va],prob[va],classes,np.full(len(classes),.5))['macro_auroc']
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),valid_auroc=metric,seconds=time.monotonic()-start));print(history[-1],flush=True)
            if metric>best:best=metric;beststate=dict(last=copy.deepcopy(last.state_dict()),head=copy.deepcopy(head.state_dict()))
        last.load_state_dict(beststate['last']);head.load_state_dict(beststate['head'])
        with torch.no_grad():prob=torch.cat([head(last(a.to(device)).mean(-1)).sigmoid().cpu() for a in activation.split(16)]).numpy()
        evaluate('founder_finetuned_last_stage',prob);torch.save(beststate,out/'finetuned_last_stage.pt')
        save_json(out/'protocol.json',dict(classes=classes,train=int(tr.sum()),valid=int(va.sum()),lead='I',test_accessed=False,
            pretrained_parameters=sum(p.numel() for p in model.parameters()),trained_parameters=sum(p.numel() for p in last.parameters())+sum(p.numel() for p in head.parameters()),
            checkpoint_sha256=file_hash('artifacts/ecgfounder/1_lead_ECGFounder.pth'),
            limitation='PTB pilot only; HEEDB source provenance per authors, no individual cross-source identity map. Scores not calibrated.'))
    finally:
        save_json(out/'history.json',history);save_json(out/'budget.json',dict(seconds=time.monotonic()-start,reservation_minutes=minutes,device=device,
            max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))


if __name__=='__main__':
    import sys
    if '--prepare' in sys.argv:prepare()
    else:run()
