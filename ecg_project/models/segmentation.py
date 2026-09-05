"""Small lead-independent U-Net, trained only on LUDB training patients."""
from pathlib import Path
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ecg_project.data.io import load_record,annotations
from ecg_project.processing.signal import preprocess,resample
from ecg_project.data.catalog import ludb_split
from ecg_project.utils import save_json,seed_all
from ecg_project.evaluation.metrics import event_metrics,summarize_events,match_events

def block(a,b):
    return nn.Sequential(nn.Conv1d(a,b,7,padding=3),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv1d(b,b,7,padding=3),nn.GroupNorm(4,b),nn.SiLU())

class Delineator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down=nn.ModuleList([block(1,12),block(12,24),block(24,48),block(48,96)])
        self.up=nn.ModuleList([block(144,48),block(72,24),block(36,12)])
        self.head=nn.Conv1d(12,4,1)
    def forward(self,x):
        skips=[]
        for i,b in enumerate(self.down):
            x=b(x);skips.append(x)
            if i<3:x=nn.functional.avg_pool1d(x,4)
        for b,s in zip(self.up,reversed(skips[:-1])):
            x=nn.functional.interpolate(x,size=s.shape[-1],mode='linear',align_corners=False)
            x=b(torch.cat([x,s],1))
        return self.head(x)

def normalize(x):
    x=x-np.median(x,axis=0)
    return (x/np.maximum(np.std(x,axis=0),1e-5)).clip(-15,15).astype(np.float32)

def prepare_ludb(root,split):
    paths=sorted(Path(root).glob('*.hea'),key=lambda p:int(p.stem)); mapping=ludb_split([p.stem for p in paths])
    xs=[];ys=[]
    for p in paths:
        if mapping[p.stem]!=split:continue
        rec=load_record(p);x=normalize(resample(preprocess(rec.signal,rec.fs),rec.fs))
        for i,lead in enumerate(rec.leads):
            waves=annotations(p,lead)
            if not waves:continue
            y=np.full(len(x),-100,dtype=np.int64)
            lo=round(min(w['onset'] for w in waves)*250/rec.fs)
            hi=round(max(w['offset'] for w in waves)*250/rec.fs)
            y[lo:hi+1]=0
            for w in waves:
                a=round(w['onset']*250/rec.fs);b=round(w['offset']*250/rec.fs)
                y[a:b+1]={'P':1,'QRS':2,'T':3}[w['wave']]
            xs.append(x[:,i][None]);ys.append(y)
    return torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))

def train(root='LUDB',output='artifacts/delineator.pt',epochs=35,minutes=35,device='auto'):
    seed_all();start=time.monotonic()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    tr=prepare_ludb(root,'train');va=prepare_ludb(root,'valid')
    train_loader=DataLoader(TensorDataset(*tr),batch_size=16,shuffle=True)
    valid_loader=DataLoader(TensorDataset(*va),batch_size=16)
    model=Delineator().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    # Modest inverse sqrt weights; no validation/test distribution enters the loss.
    counts=torch.bincount(tr[1][tr[1]>=0],minlength=4).float()
    weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-100)
    best=-1;history=[];Path(output).parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train();losses=[]
        for x,y in train_loader:
            if time.monotonic()-start>minutes*60:break
            x=x.to(device);y=y.to(device)
            # Lead sign/gain augmentation, without changing temporal intervals.
            gain=torch.rand(len(x),1,1,device=device)*.5+.75
            signs=torch.where(torch.rand(len(x),1,1,device=device)<.3,-1.,1.)
            logits=model(x*gain*signs+torch.randn_like(x)*.015)
            loss=loss_fn(logits,y);opt.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval();cm=np.zeros((4,4),dtype=np.int64)
        with torch.no_grad():
            for x,y in valid_loader:
                p=model(x.to(device)).argmax(1).cpu().numpy(); yy=y.numpy();ok=yy>=0
                cm+=np.bincount(yy[ok]*4+p[ok],minlength=16).reshape(4,4)
        dice=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1);score=dice[1:].mean()
        row=dict(epoch=epoch+1,train_loss=np.mean(losses),valid_dice=dice.tolist(),seconds=time.monotonic()-start)
        history.append(row);print(row,flush=True)
        if score>best:
            best=score;torch.save(dict(state_dict=model.cpu().state_dict(),fs=250,preprocess='morphology',
                split_seed=42,epoch=epoch+1,valid_dice=dice.tolist(),architecture='Delineator'),output);model.to(device)
        save_json(Path(output).with_suffix('.history.json'),history)
        if time.monotonic()-start>minutes*60:break
    save_json(Path(output).with_suffix('.budget.json'),dict(seconds=time.monotonic()-start,device=device,
        max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        parameters=sum(p.numel() for p in model.parameters()),epochs=len(history)))

class Predictor:
    def __init__(self,checkpoint='artifacts/delineator.pt',device='cpu'):
        self.device=device;self.checkpoint=str(checkpoint);self.model=Delineator().to(device)
        self.model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state_dict']);self.model.eval()
    def predict(self,signal,fs):
        if signal.ndim==1:signal=signal[:,None]
        x=resample(preprocess(signal,fs),fs)
        # Long signals use overlap-add windows; prevents arbitrarily large GPU allocations.
        n=len(x);probs=np.zeros((n,signal.shape[1],4),np.float32);count=np.zeros(n,np.float32)
        window=2500;hop=2000
        starts=sorted(set(list(range(0,max(n-window,0)+1,hop))+[max(0,n-window)]))
        with torch.no_grad():
            for start in starts:
                v=x[start:start+window];valid=len(v)
                z=normalize(v)
                if len(z)<256:z=np.pad(z,((0,256-len(z)),(0,0)),mode='edge')
                batch=torch.from_numpy(z.T[:,None]).to(self.device)
                pr=self.model(batch).softmax(1).cpu().numpy().transpose(2,0,1)[:valid]
                weight=np.minimum(np.arange(valid)+1,np.arange(valid,0,-1)).clip(1,125).astype(np.float32)
                probs[start:start+valid]+=pr*weight[:,None,None];count[start:start+valid]+=weight
        probs/=count[:,None,None]
        out=[]
        for ch in range(signal.shape[1]):
            if np.ptp(signal[:,ch])<1e-8:
                out.append([]);continue
            baseline=np.median(x[:,ch])
            mask=probs[:,ch].argmax(-1);waves=[]
            for cls,wave in [(1,'P'),(2,'QRS'),(3,'T')]:
                active=(mask==cls);changes=np.diff(np.r_[False,active,False].astype(int))
                for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                    if b-a<5:continue
                    # Segmentation predicts extent; peak is maximum absolute deviation in the extent.
                    segment=x[a:b,ch];k=a+int(np.argmax(abs(segment-baseline)))
                    onset=min(round(a*fs/250),len(signal)-1);offset=min(round((b-1)*fs/250),len(signal)-1)
                    peak=min(max(round(k*fs/250),onset),offset)
                    waves.append(dict(wave=wave,onset=onset,peak=peak,offset=offset,
                                      confidence=float(probs[a:b,ch,cls].mean()),source='predicted'))
            out.append(sorted(waves,key=lambda w:w['onset']))
        return out

def evaluate(root='LUDB',checkpoint='artifacts/delineator.pt',split='test',output='reports/segmentation_test.json'):
    predictor=Predictor(checkpoint);results=defaultdict(list);width_errors=[];width_labels=[];detail=[]
    paths=sorted(Path(root).glob('*.hea'));mapping=ludb_split([p.stem for p in paths])
    for p in paths:
        if mapping[p.stem]!=split:continue
        rec=load_record(p);pred=predictor.predict(rec.signal,rec.fs)
        for lead,waves in zip(rec.leads,pred):
            ref=annotations(p,lead)
            if not ref:continue
            lo=min(w['onset'] for w in ref);hi=max(w['offset'] for w in ref)
            for wave in ('P','QRS','T'):
                r=[w for w in ref if w['wave']==wave];q=[w for w in waves if w['wave']==wave and lo<=w['peak']<=hi]
                m=event_metrics(r,q,rec.fs);results[wave].append(m);detail.append(dict(record=p.stem,lead=lead,wave=wave,**m))
                if wave=='QRS':
                    for i,j in match_events([w['peak'] for w in r],[w['peak'] for w in q],.15*rec.fs):
                        rw=(r[i]['offset']-r[i]['onset'])/rec.fs*1000;pw=(q[j]['offset']-q[j]['onset'])/rec.fs*1000
                        width_errors.append(pw-rw);width_labels.append((rw>=120,pw>=120))
    widths=np.array(width_labels)
    report=dict(split=split,summary={k:summarize_events(v) for k,v in results.items()},details=detail,
       qrs_width=dict(mae_ms=np.mean(abs(np.array(width_errors))),bias_ms=np.mean(width_errors),
       threshold_120_agreement=np.mean(widths[:,0]==widths[:,1]),matched_n=len(width_errors)),
       per_lead={f'{lead}/{wave}':summarize_events([x for x in detail if x['lead']==lead and x['wave']==wave])
          for lead in sorted({x['lead'] for x in detail}) for wave in ('P','QRS','T')},
       protocol='Patient split seed42; evaluation limited to annotated envelope; 150ms one-to-one peak matching; widths on matched QRS only.')
    save_json(output,report);print(report['summary'],report['qrs_width'],flush=True)
    return report
