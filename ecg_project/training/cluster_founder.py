"""Full-backbone ECGFounder fine-tuning from a Jupyter cell on one GPU."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ecg_project.experiments.founder_experiments import load_founder
from ecg_project.training.cluster_training import _atomic_save
from ecg_project.data.catalog import TARGETS,file_hash
from ecg_project.evaluation.metrics import multilabel_metrics
from ecg_project.utils import seed_all,save_json


def train_founder_cluster(input_root='artifacts/founder_experiments',output='artifacts/cluster/founder_full',
                          epochs=100,batch_size=32,accumulation=1,workers=0,patience=15,max_batches=None):
    seed_all();start=time.monotonic();root=Path(input_root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    frame=pd.read_csv(root/'manifest.csv');tr=frame.split.to_numpy()=='train';va=frame.split.to_numpy()=='valid'
    if not (tr|va).all():raise ValueError('Only train/valid inputs accepted')
    if set(frame.loc[tr,'patient_id']) & set(frame.loc[va,'patient_id']):raise ValueError('Patient leakage')
    if (root/'signals.npy').exists():signals=np.load(root/'signals.npy',mmap_mode='r')
    else:
        with np.load(root/'inputs.npz') as z:signals=z['signal']
    classes=[c for c in TARGETS if frame.loc[tr,c].sum()>=20 and (1-frame.loc[tr,c]).sum()>=20]
    if not classes:raise ValueError('Insufficient training support')
    y=frame[classes].to_numpy(dtype=np.float32)
    model=load_founder();model.dense=nn.Linear(model.dense.in_features,len(classes));model.to(device)
    optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if not n.startswith('dense.')],'lr':1e-5},
        {'params':model.dense.parameters(),'lr':1e-3}],weight_decay=.01)
    amp=device=='cuda' and torch.cuda.is_bf16_supported()
    pos=torch.tensor(np.sqrt((1-y[tr]).sum(0)/np.maximum(y[tr].sum(0),1)),device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos)
    # TensorDataset copies only the selected cohort; complete PTB fits comfortably in host RAM.
    loader=DataLoader(TensorDataset(torch.from_numpy(np.array(signals[tr,None])),torch.from_numpy(y[tr])),
        batch_size=batch_size,shuffle=True,num_workers=workers,pin_memory=device=='cuda',persistent_workers=workers>0)
    latest=out/'latest.pt';begin=0;best=-1.;stale=0;history=[]
    config=dict(input_root=str(root),classes=classes,batch_size=batch_size,accumulation=accumulation,
        manifest_sha256=file_hash(root/'manifest.csv'),max_batches=max_batches)
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=True)
        if saved['config']!=config:raise ValueError('Resume input/config mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        begin=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in saved['cuda_rng']])
    save_json(out/'config.json',dict(**config,epochs=epochs,train=int(tr.sum()),valid=int(va.sum()),unfrozen='all'))
    try:
        for epoch in range(begin,epochs):
            model.train();optimizer.zero_grad();losses=[];n=min(len(loader),max_batches) if max_batches else len(loader)
            for k,(x,label) in enumerate(loader):
                if k>=n:break
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    logits=model(x.to(device,non_blocking=True));loss=criterion(logits,label.to(device,non_blocking=True))
                    group=min(accumulation,n-(k//accumulation)*accumulation)
                (loss/group).backward();losses.append(float(loss.detach()))
                if (k+1)%accumulation==0 or k+1==n:
                    nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();optimizer.zero_grad()
            model.eval();pp=[]
            with torch.no_grad():
                for x in torch.from_numpy(np.array(signals[va,None])).split(batch_size):
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):p=model(x.to(device)).sigmoid()
                    pp.append(p.float().cpu().numpy())
            prob=np.concatenate(pp);report=multilabel_metrics(y[va],prob,classes);metric=report['macro_auroc']
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),valid_auroc=metric,seconds=time.monotonic()-start);history.append(row);print(row,flush=True)
            if metric>best:
                best=metric;stale=0;_atomic_save(model.state_dict(),out/'best.pt');save_json(out/'best_metrics.json',report)
                np.savez_compressed(out/'valid_predictions.npz',truth=y[va],probability=prob,classes=np.array(classes),records=frame.loc[va,'record_id'].astype(str).to_numpy())
            else:stale+=1
            _atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,stale=stale,
                history=history,config=config,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),latest)
            save_json(out/'history.json',history)
            if stale>=patience:break
    finally:
        save_json(out/'run.json',dict(seconds_this_invocation=time.monotonic()-start,device=device,smoke=bool(max_batches),
            max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,test_accessed=False))
    return out
