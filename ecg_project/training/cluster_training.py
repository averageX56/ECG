"""Jupyter-friendly scalable beat Transformer training with resumable epochs."""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import time
import random
import hashlib
import os
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ecg_project.experiments.representation_experiments import BeatBERT,Sequences,load_records,fit_tokens,score
from ecg_project.utils import seed_all,save_json
from itertools import islice
from tqdm.auto import tqdm
from ecg_project.training.telemetry import tracked


@dataclass
class ClusterConfig:
    data_root: str='artifacts/beat_features_qt'
    output: str='artifacts/cluster/beat_bert_large'
    dim: int=512
    layers: int=8
    radius: int=32
    batch_size: int=256
    accumulation: int=1
    pretrain_epochs: int=30
    finetune_epochs: int=100
    learning_rate: float=0.0001
    patience: int=15
    workers: int=0
    amp: bool=True
    seed: int=42
    resume: bool=True
    max_batches: int|None=None  # Smoke only; never compare its metrics as full validation.
    gpu_resident: bool=True
    auto_batch: bool=True
    unlabeled_root: str|None=None
    unlabeled_roots: list[str]|None=None
    include_q: bool=True
    min_selection_support: int=1


def _atomic_save(obj,path):
    path=Path(path);temp=path.with_suffix('.partial');torch.save(obj,temp);temp.replace(path)


class ResidentBatches:
    """Gather windows on GPU; clipping bounds prevent crossing patients."""
    def __init__(self,records,radius,batch_size,labeled,device):
        xs=[];ys=[];low=[];high=[];offset=0
        for r in records:
            if r['split']!='train':continue
            n=len(r['x']);xs.append(r['x']);ys.append(r['y'])
            low.extend([offset]*n);high.extend([offset+n-1]*n);offset+=n
        self.x=torch.as_tensor(np.concatenate(xs),device=device)
        self.y=torch.as_tensor(np.concatenate(ys),device=device)
        self.low=torch.tensor(low,device=device);self.high=torch.tensor(high,device=device)
        self.ids=torch.nonzero(self.y>=0).flatten() if labeled else torch.arange(offset,device=device)
        self.shifts=torch.arange(-radius,radius+1,device=device);self.batch_size=batch_size
    def __len__(self):return (len(self.ids)+self.batch_size-1)//self.batch_size
    def __iter__(self):
        order=self.ids[torch.randperm(len(self.ids),device=self.ids.device)]
        for ids in order.split(self.batch_size):
            indices=ids[:,None]+self.shifts
            indices=torch.maximum(torch.minimum(indices,self.high[ids,None]),self.low[ids,None])
            yield self.x[indices],self.y[ids]


def tune_batch(model,dim,length,device,dtype,initial=32,maximum=1024):
    """Probe forward/backward, retaining VRAM headroom for Adam and validation."""
    if device!='cuda':return initial
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();best=None
    total=torch.cuda.get_device_properties(0).total_memory
    try:
        for batch in [2**k for k in range(0,11)]:
            if batch>maximum:break
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                x=torch.zeros(batch,length,dim,device=device)
                with torch.autocast('cuda',dtype=dtype):
                    logits,rec,_=model(x);loss=logits.square().mean()+rec.square().mean()
                loss.backward();torch.cuda.synchronize()
                used=torch.cuda.max_memory_allocated();model.zero_grad(set_to_none=True)
                del x,logits,rec,loss
                if used>.7*total:break
                best=batch
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True);torch.cuda.empty_cache();break
    finally:
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if best is None:raise RuntimeError('Even microbatch 1 exceeds the memory allowance; reduce model size.')
    return best


def infer(model,records,device,radius,batch_size=128):
    model.eval();prob=[];latent=[]
    with torch.no_grad():
        for r in tqdm(
            records,
            desc="Inference",
            unit="record",
            leave=False,
        ):
            n=len(r['x']);pp=[];ee=[]
            for start in range(0,n,batch_size):
                ids=np.clip(np.arange(start,min(n,start+batch_size))[:,None]+np.arange(-radius,radius+1),0,n-1)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and torch.cuda.is_bf16_supported()):
                    logits,_,z=model(torch.from_numpy(r['x'][ids]).to(device))
                pp.append(logits.float().softmax(1).cpu().numpy());ee.append(z.float().cpu().numpy())
            prob.append(np.concatenate(pp));latent.append(np.concatenate(ee))
    return prob,latent


@tracked
def train_cluster(config:ClusterConfig):
    """Run from a notebook cell; interrupt safely, then rerun to resume last epoch.

    Cluster runs intentionally have no wall-clock cap. Local smoke should use
    tiny dim/layers, one epoch, max_batches=2. No test files are loaded.
    """
    cfg=config;seed_all(cfg.seed);out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    if cfg.dim%4 or cfg.radius<0 or cfg.batch_size<1 or cfg.accumulation<1:raise ValueError('Invalid model/batch configuration')
    device='cuda' if torch.cuda.is_available() else 'cpu';start=time.monotonic()
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=True
    torch.set_num_threads(min(16,os.cpu_count() or 4))
    records=load_records(cfg.data_root)
    roots=[Path(cfg.data_root)]
    extra_roots=list(cfg.unlabeled_roots or [])
    if cfg.unlabeled_root and cfg.unlabeled_root not in extra_roots:extra_roots.append(cfg.unlabeled_root)
    if extra_roots and cfg.pretrain_epochs<=0:raise ValueError('Extended SSL roots require pretrain_epochs > 0')
    from ecg_project.data.cache import validate_cache
    for cache_root in [cfg.data_root,*extra_roots]:
        if (Path(cache_root)/'provenance.json').exists():validate_cache(cache_root)
    for extra_root in extra_roots:
        raw=json.loads((Path(extra_root)/'manifest.json').read_text())
        if any(r['split']!='train' for r in raw):raise ValueError('SSL manifest contains validation/test/external records')
        extra=load_records(extra_root)
        if any(np.any(r['y']>=0) or r['split']!='train' for r in extra):raise ValueError('Expected train-only unlabelled cache')
        records+=extra;roots.append(Path(extra_root))
    if len({r['record'] for r in records})!=len(records):raise ValueError('Duplicate records across SSL/supervised roots')
    patients={}
    for r in records:
        patient=r.get('patient_id') or r.get('patient')
        if patient:
            if patient in patients and patients[patient]!=r['split']:raise ValueError('Patient leakage across BERT inputs')
            patients[patient]=r['split']
    raw_support=np.bincount(np.concatenate([r['y'][r['y']>=0] for r in records if r['split']=='train']),minlength=5)
    if not cfg.include_q:
        for r in records:
            if r['split']=='train':r['y']=np.where(r['y']==4,-1,r['y'])
    # Every feature file is hashed: data drift must not silently resume a run.
    digest=hashlib.sha256()
    for r in records:
        digest.update(r['record'].encode());digest.update(r['split'].encode())
        path=next(p/(r['record']+'.npz') for p in roots if (p/(r['record']+'.npz')).exists())
        digest.update(path.read_bytes())
    data_hash=digest.hexdigest()
    token_path=out/'tokens.joblib'
    latest=out/'latest.pt'
    if latest.exists() and not cfg.resume:raise FileExistsError('Use a new output directory or resume=True')
    # Validate checkpoint identity before mutating any files in an existing run.
    saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        identity_keys=['dim','layers','radius','seed','batch_size','accumulation','learning_rate','pretrain_epochs','max_batches','amp','gpu_resident','include_q','min_selection_support']
        saved['config'].setdefault('include_q',True);saved['config'].setdefault('min_selection_support',1)
        if saved['data_hash']!=data_hash or any(saved['config'].get(k)!=asdict(cfg)[k] for k in identity_keys):
            raise ValueError('Resume config or data differs; choose a fresh output directory')
    tokens=fit_tokens(records)
    if saved is None:joblib.dump(tokens,token_path)
    model=BeatBERT(54,cfg.dim,cfg.layers,2*cfg.radius+1).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=.01)
    amp=cfg.amp and device=='cuda'
    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    state=dict(stage='pretrain' if cfg.pretrain_epochs else 'finetune',epoch=0,best=-1.,stale=0)
    history=[]
    actual_batch=cfg.batch_size
    if cfg.auto_batch and device=='cuda' and not latest.exists():
        actual_batch=tune_batch(model,54,2*cfg.radius+1,device,dtype)
        print('Selected microbatch',actual_batch,flush=True)
    if saved is not None:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        state=saved['state'];history=saved['history'];torch.set_rng_state(saved['torch_rng'].cpu())
        actual_batch=saved['actual_batch']
        random.setstate(saved['python_rng'])
        if device=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    save_json(out/'config.json',asdict(cfg))
    valid=[r for r in records if r['split']=='valid']
    train_y=np.concatenate([r['y'][r['y']>=0] for r in records if r['split']=='train'])
    count=np.bincount(train_y,minlength=5);weight=np.sqrt(count.sum()/np.maximum(count,1));weight/=weight.mean()
    selection_classes=[c for i,c in enumerate(['N','S','V','F']) if count[i]>=cfg.min_selection_support]
    if not selection_classes:raise ValueError('No supported classes for selection')
    support_report=dict(train_support=dict(zip(['N','S','V','F','Q'],map(int,count))),include_q=cfg.include_q,
        original_train_support=dict(zip(['N','S','V','F','Q'],map(int,raw_support))),
        selection_classes=selection_classes,classes_excluded_from_selection=[c for c in ['N','S','V','F','Q'] if c not in selection_classes])
    save_json(out/'class_support.json',support_report)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=device))
    def checkpoint():
        _atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            state=state.copy(),history=history,config=asdict(cfg),actual_batch=actual_batch,data_hash=data_hash,torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),latest)
    if saved is None:checkpoint()  # Resume is also possible after interrupting the first epoch.
    while state['stage']!='complete':
        pretrain=state['stage']=='pretrain';epochs=cfg.pretrain_epochs if pretrain else cfg.finetune_epochs
        if state['epoch']>=epochs:
            if pretrain:
                _atomic_save(model.state_dict(),out/'pretrained.pt');state.update(stage='finetune',epoch=0)
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=.01)
            else:state['stage']='complete'
            checkpoint();continue
        ds=Sequences(records,'train',labeled=not pretrain,radius=cfg.radius)
        loader=ResidentBatches(records,cfg.radius,actual_batch,not pretrain,device) if cfg.gpu_resident else DataLoader(ds,batch_size=actual_batch,shuffle=True,num_workers=cfg.workers,pin_memory=device=='cuda')
        model.train();optimizer.zero_grad();losses=[];examples=0
        n_batches = (
            min(len(loader), cfg.max_batches)
            if cfg.max_batches
            else len(loader)
        )
        batches = tqdm(
            islice(loader, n_batches),
            total=n_batches,
            desc=f"{state['stage']} {state['epoch'] + 1}/{epochs}",
            unit="batch",
            leave=False,
        )

        for k,(x,y) in enumerate(batches):
            if k>=n_batches:break
            x=x.to(device);y=y.to(device)
            examples+=len(x)
            with torch.autocast(device_type=device,dtype=dtype,enabled=amp):
                if pretrain:
                    mask=torch.rand(x.shape[:2],device=device)<.25;mask[:,x.shape[1]//2]=True
                    _,rec,_=model(x,mask);loss=((rec-x)**2)[mask].mean()
                else:logits,_,_=model(x);loss=criterion(logits,y)
                group=min(cfg.accumulation,n_batches-(k//cfg.accumulation)*cfg.accumulation)
                scaled=loss/group
            scaler.scale(scaled).backward();losses.append(float(loss.detach()))
            if (k+1)%cfg.accumulation==0 or k+1==n_batches:
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad()
        state['epoch']+=1
        row=dict(stage=state['stage'],epoch=state['epoch'],loss=float(np.mean(losses)),seconds=time.monotonic()-start,examples=examples)
        if not pretrain:
            pp,_=infer(model,valid,device,cfg.radius,actual_batch);report=score(valid,pp,selection_classes);row['valid_score']=report['selection_score']
            if report['selection_score']>state['best']:
                state['best']=report['selection_score'];state['stale']=0
                state['best_epoch']=state['epoch']
                _atomic_save(model.state_dict(),out/'best.pt');save_json(out/'best_metrics.json',dict(**report,**support_report,best_epoch=state['epoch']))
            else:state['stale']+=1
            if state['stale']>=cfg.patience:state['stage']='complete'
        history.append(row);save_json(out/'history.json',history);checkpoint();print(row,flush=True)
    if (out/'best.pt').exists():model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    pp,emb=infer(model,valid,device,cfg.radius,actual_batch)
    for r,p,e in zip(valid,pp,emb):np.savez_compressed(out/('valid_'+r['record']+'.npz'),probability=p,embedding=e,truth=r['y'])
    save_json(out/'run.json',dict(seconds_this_invocation=time.monotonic()-start,device=device,
        parameters=sum(p.numel() for p in model.parameters()),max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        test_accessed=False,smoke=bool(cfg.max_batches),data_sha256=data_hash,status=state['stage'],actual_batch=actual_batch,
        effective_batch=actual_batch*cfg.accumulation,best_epoch=state.get('best_epoch'),epochs_completed=len(history),
        max_memory_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        max_memory_reserved=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
        examples_per_second=sum(r.get('examples',0) for r in history)/max(time.monotonic()-start,1e-9)))
    return out
