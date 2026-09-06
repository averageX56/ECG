"""Patient-separated LUDB preparation and resumable Qwen QLoRA delineation."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ecg_project.data.catalog import file_hash,ludb_split
from ecg_project.models.segmentation import prepare_ludb
from ecg_project.models.qwen_delineator import QwenDelineator,load_qwen
from ecg_project.models.lora import adapter_state,load_adapter
from ecg_project.training.cluster_training import _atomic_save
from ecg_project.utils import save_json,seed_all


def prepare(root='LUDB',output='artifacts/qwen_delineation_inputs'):
    out=Path(output)
    if out.exists():raise FileExistsError('Use a fresh cache directory')
    out.mkdir(parents=True)
    paths=sorted(Path(root).glob('*.hea'),key=lambda p:int(p.stem))
    splits=ludb_split([p.stem for p in paths])
    for split in ('train','valid'):
        x,y=prepare_ludb(root,split)
        np.save(out/f'{split}_x.npy',x.numpy());np.save(out/f'{split}_y.npy',y.numpy())
    save_json(out/'provenance.json',dict(fs=250,preprocessing='segmentation.prepare_ludb.v1',
        patients={s:[p.stem for p in paths if splits[p.stem]==s] for s in ('train','valid')},
        sha256={p.name:file_hash(p) for p in out.glob('*.npy')},test_accessed=False,
        ignored_label=-100,classes=['background','P','QRS','T']))
    return out


@dataclass
class QwenConfig:
    input_root:str='artifacts/qwen_delineation_inputs'
    model_root:str='artifacts/qwen3_4b'
    output:str='artifacts/cluster/qwen3_4b_delineation'
    quantization:str='nf4'
    patch_size:int=20
    rank:int=16
    alpha:int=32
    epochs:int=40
    batch_size:int=4
    accumulation:int=4
    learning_rate:float=2e-4
    patience:int=8
    max_batches:int=0
    valid_limit:int=0
    local_minutes:float|None=None


def train(cfg:QwenConfig):
    seed_all();start=time.monotonic();out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    if min(cfg.epochs,cfg.batch_size,cfg.accumulation,cfg.patience)<1:raise ValueError('Invalid training configuration')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.local_minutes is None and (device=='cpu' or torch.cuda.get_device_properties(0).total_memory<35*2**30):
        raise ValueError('Set local_minutes for a local smoke; unlimited mode is for the cluster')
    if cfg.local_minutes is not None:
        from ecg_project.training.local_budget import check_reservation
        check_reservation(cfg.local_minutes)
    root=Path(cfg.input_root);meta=json.loads((root/'provenance.json').read_text())
    if set(meta['patients']['train']) & set(meta['patients']['valid']):raise ValueError('Patient leakage')
    for name,digest in meta['sha256'].items():
        if file_hash(root/name)!=digest:raise ValueError('Modified delineation cache')
    train_x=torch.from_numpy(np.load(root/'train_x.npy'));train_y=torch.from_numpy(np.load(root/'train_y.npy'))
    vx=torch.from_numpy(np.load(root/'valid_x.npy'));vy=torch.from_numpy(np.load(root/'valid_y.npy'))
    if cfg.valid_limit:vx=vx[:cfg.valid_limit];vy=vy[:cfg.valid_limit]
    identity=dict(config={k:v for k,v in asdict(cfg).items() if k not in ('epochs','output','local_minutes')},
        cache=meta,base=json.loads((Path(cfg.model_root)/'provenance.json').read_text()))
    model=QwenDelineator(load_qwen(cfg.model_root,cfg.quantization),cfg.patch_size,cfg.rank,cfg.alpha).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate)
    counts=torch.bincount(train_y[train_y>=0],minlength=4).float().clamp_min(1)
    weights=(counts.sum()/counts).sqrt();weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=weights.to(device),ignore_index=-100)
    loader=DataLoader(TensorDataset(train_x,train_y),batch_size=cfg.batch_size,shuffle=True,pin_memory=device=='cuda')
    valid=DataLoader(TensorDataset(vx,vy),batch_size=cfg.batch_size,pin_memory=device=='cuda')
    amp=device=='cuda' and torch.cuda.is_bf16_supported();torch.backends.cuda.matmul.allow_tf32=True
    begin=0;best=-1.;stale=0;history=[];latest=out/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Changed run identity; use a new output')
        load_adapter(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        begin=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history']
        torch.set_rng_state(saved['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    save_json(out/'config.json',asdict(cfg))
    def checkpoint(epoch):
        _atomic_save(dict(adapter=adapter_state(model),optimizer=optimizer.state_dict(),identity=identity,
            epoch=epoch,best=best,stale=stale,history=history,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),latest)
    if not latest.exists():checkpoint(0)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    status='complete'
    def deadline():
        if cfg.local_minutes is not None and time.monotonic()-start>cfg.local_minutes*60:raise TimeoutError('Local budget reached')
    try:
        for epoch in range(begin,cfg.epochs):
            if stale>=cfg.patience:break
            model.train();optimizer.zero_grad();losses=[]
            n=min(len(loader),cfg.max_batches) if cfg.max_batches else len(loader)
            for k,(x,y) in enumerate(loader):
                if k>=n:break
                deadline();x=x.to(device);y=y.to(device)
                gain=.75+.5*torch.rand(len(x),1,1,device=device)
                sign=torch.where(torch.rand(len(x),1,1,device=device)<.3,-1.,1.)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    loss=criterion(model(x*gain*sign+torch.randn_like(x)*.015),y)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                group=min(cfg.accumulation,n-(k//cfg.accumulation)*cfg.accumulation)
                (loss/group).backward();losses.append(float(loss.detach()))
                if (k+1)%cfg.accumulation==0 or k+1==n:
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
                    optimizer.step();optimizer.zero_grad()
            model.eval();cm=np.zeros((4,4),dtype=np.int64)
            with torch.no_grad():
                for x,y in valid:
                    deadline()
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):p=model(x.to(device)).argmax(1).cpu()
                    ok=y>=0;cm+=np.bincount((y[ok]*4+p[ok]).numpy(),minlength=16).reshape(4,4)
            dice=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1);metric=float(dice[1:].mean())
            row=dict(epoch=epoch+1,loss=float(np.mean(losses)),valid_dice=dice.tolist(),seconds=time.monotonic()-start)
            history.append(row)
            if metric>best:
                best=metric;stale=0;_atomic_save(dict(adapter=adapter_state(model),identity=identity),out/'best.pt')
                save_json(out/'best_metrics.json',dict(valid_macro_wave_dice=metric,valid_dice=dice.tolist(),confusion=cm.tolist(),
                    smoke=bool(cfg.max_batches),test_accessed=False))
            else:stale+=1
            checkpoint(epoch+1);save_json(out/'history.json',history);print(row,flush=True)
    except BaseException:status='interrupted';raise
    finally:
        from ecg_project.training.local_budget import record_invocation
        elapsed=time.monotonic()-start
        record_invocation(out,elapsed,cfg.local_minutes is not None)
        save_json(out/'run.json',dict(seconds_this_invocation=elapsed,status=status,device=device,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,smoke=bool(cfg.max_batches),test_accessed=False))
    return out
