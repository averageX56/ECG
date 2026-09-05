"""Mean Teacher consistency transfer, with real LUDB supervision retained."""
from pathlib import Path
from copy import deepcopy
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from ecg_project.models.segmentation import Delineator,prepare_ludb,normalize
from ecg_project.processing.signal import resample,preprocess
from ecg_project.data.io import load_record
from ecg_project.data.catalog import file_hash
from ecg_project.utils import seed_all,save_json

def train(catalog='artifacts/catalog.csv',initial='artifacts/delineator.pt',output='artifacts/delineator_transfer.pt',
          epochs=12,minutes=20,unlabeled_records=200):
    seed_all();start=time.monotonic();device='cuda' if torch.cuda.is_available() else 'cpu'
    frame=pd.read_csv(catalog).fillna('')
    # Only CPSC-extra is an adaptation-training source. No PTB validation/test, Georgia or MIT test.
    source=frame[(frame.source=='Training_2') & (frame.readable==True) & (frame.has_labels==False)]
    source=source.sample(n=min(len(source),unlabeled_records),random_state=42)
    if len(source)==0:raise ValueError('No eligible unlabeled training records')
    unlabeled=[];provenance=[]
    for _,r in source.iterrows():
        rec=load_record(r.path,stop=min(int(r.n_samples),round(10*float(r.fs))))
        x=normalize(resample(preprocess(rec.signal,rec.fs),rec.fs))
        if len(x)<2500:continue
        # Three distinct lead views per record contain different morphology without inflating patient count.
        for i in [0,1,6]:unlabeled.append(x[:2500,i][None])
        provenance.append(dict(path=r.path,record_id=r.record_id,source=r.source,header_hash=file_hash(r.path),
             data_hash=file_hash(Path(r.path).with_suffix('.dat')),role='unlabeled_adaptation_train'))
    u=DataLoader(TensorDataset(torch.from_numpy(np.stack(unlabeled))),batch_size=16,shuffle=True)
    tr=prepare_ludb('LUDB','train');va=prepare_ludb('LUDB','valid')
    dl=DataLoader(TensorDataset(*tr),batch_size=16,shuffle=True);vl=DataLoader(TensorDataset(*va),batch_size=16)
    model=Delineator().to(device);checkpoint=torch.load(initial,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);teacher=deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
    counts=torch.bincount(tr[1][tr[1]>=0],minlength=4).float();weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-100)
    best=-1;history=[];ui=iter(u)
    for epoch in range(epochs):
        model.train();losses=[];coverage=[]
        for x,y in dl:
            if time.monotonic()-start>minutes*60:break
            try:(ux,)=next(ui)
            except StopIteration:ui=iter(u);(ux,)=next(ui)
            ux=ux.to(device);x=x.to(device);y=y.to(device)
            with torch.no_grad():
                soft=teacher(ux).softmax(1);confidence,pseudo=soft.max(1)
                # Per-wave mask plus a small background sample prevents trivial all-background consistency.
                mask=(confidence>.95) & ((pseudo>0) | (torch.rand_like(confidence)<.15))
                mask[:,:125]=False;mask[:,-125:]=False
            phase=torch.rand(len(ux),1,1,device=device)*6.28
            t=torch.arange(ux.shape[-1],device=device)[None,None]/250
            drift=.08*torch.sin(2*torch.pi*.3*t+phase)
            strong=ux*(.75+.5*torch.rand(len(ux),1,1,device=device))+torch.randn_like(ux)*.035+drift
            unlabeled_logits=model(strong)
            consistency=nn.functional.cross_entropy(unlabeled_logits,pseudo,reduction='none')
            consistency=(consistency*mask).sum()/mask.sum().clamp_min(1)
            supervised=loss_fn(model(x),y)
            loss=supervised+.1*consistency
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            with torch.no_grad():
                for a,b in zip(teacher.parameters(),model.parameters()):a.mul_(.99).add_(b,alpha=.01)
            losses.append(loss.item());coverage.append(mask.float().mean().item())
        model.eval();cm=np.zeros((4,4),dtype=np.int64)
        with torch.no_grad():
            for x,y in vl:
                p=model(x.to(device)).argmax(1).cpu().numpy(); yy=y.numpy();ok=yy>=0
                cm+=np.bincount(yy[ok]*4+p[ok],minlength=16).reshape(4,4)
        dice=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1);score=dice[1:].mean()
        row=dict(epoch=epoch+1,loss=np.mean(losses),pseudo_label_coverage=np.mean(coverage),valid_dice=dice.tolist(),seconds=time.monotonic()-start)
        history.append(row);print(row,flush=True)
        if score>best:
            best=score;torch.save(dict(state_dict=model.cpu().state_dict(),fs=250,preprocess='morphology',
               split_seed=42,epoch=epoch+1,valid_dice=dice.tolist(),architecture='Delineator',method='mean_teacher_consistency'),output);model.to(device)
        save_json(Path(output).with_suffix('.history.json'),history)
        if time.monotonic()-start>minutes*60:break
    save_json(Path(output).with_suffix('.provenance.json'),dict(initial_hash=file_hash(initial),records=provenance,
        confidence_threshold=.95,consistency_weight=.1,ema=.99,
        excluded_sources=['LUDB validation/test','PTB-XL validation/test','Georgia','MIT-BIH','StPetersburg'],
        caveat='Unlabeled consistency does not establish boundary accuracy in the target domain.'))
    save_json(Path(output).with_suffix('.budget.json'),dict(seconds=time.monotonic()-start,device=device,
         max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
