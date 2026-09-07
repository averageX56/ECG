"""Common record classification pipeline for frozen and LoRA HuBERT-ECG."""
from pathlib import Path
from dataclasses import dataclass,asdict
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ecg_project.data.io import load_record
from ecg_project.data.catalog import TARGETS,file_hash
from ecg_project.processing.hubert import prepare_signal,VERSION
from ecg_project.models.hubert import load_backbone,HubertClassifier
from ecg_project.models.lora import adapter_state,load_adapter
from ecg_project.training.cluster_training import _atomic_save
from ecg_project.evaluation.metrics import multilabel_metrics
from ecg_project.utils import seed_all,save_json
from ecg_project.training.telemetry import tracked


def _prepare_hubert_path(path):
    return prepare_signal(load_record(str(path).replace('\\','/')))


def prepare_inputs(output='artifacts/hubert_inputs',full=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.csv').exists():raise FileExistsError('Inputs already exist; reuse or choose a fresh output')
    if full:
        frame=pd.read_csv('artifacts/catalog.csv')
        frame=frame[(frame.source=='WFDB_PTB-XL') & frame.fold.between(1,9) & (frame.readable==True)].copy()
        frame['split']=np.where(frame.fold==9,'valid','train')
    else:
        frame=pd.read_csv('artifacts/record_features_qt/manifest.csv')
        frame=frame[(frame.source=='WFDB_PTB-XL') & frame.split.isin(['train','valid'])].copy()
    frame=frame.reset_index(drop=True)
    if frame.empty:raise ValueError('No PTB records selected')
    if frame.patient_id.isna().any():raise ValueError('PTB patient identities required')
    if set(frame.loc[frame.split=='train','patient_id']) & set(frame.loc[frame.split=='valid','patient_id']):raise ValueError('Patient overlap')
    signals=np.lib.format.open_memmap(out/'signals.npy',mode='w+',dtype='float32',shape=(len(frame),2,6000))
    from ecg_project.processing.parallel import ordered_map
    for i,signal in enumerate(ordered_map(_prepare_hubert_path,frame.path.tolist())):
        signals[i]=signal
        if (i+1)%200==0:print('HuBERT inputs',i+1,'/',len(frame),flush=True)
    signals.flush();frame.to_csv(out/'manifest.csv',index=False)
    save_json(out/'provenance.json',dict(preprocessing=VERSION,full_cohort=full,records=len(frame),
        signals_sha256=file_hash(out/'signals.npy'),test_accessed=False,
        evaluation_status='pretraining_source_overlap; not independent external validation'))
    return out


@dataclass
class LoRAConfig:
    input_root:str='artifacts/hubert_inputs'
    model_root:str='artifacts/hubert_large'
    output:str='artifacts/cluster/hubert_lora_r16'
    rank:int=16
    alpha:int=32
    targets:tuple=('q_proj','v_proj')
    epochs:int=30
    batch_size:int=8
    accumulation:int=4
    learning_rate:float=2e-4
    patience:int=8
    checkpointing:bool=True
    workers:int=0
    train_limit:int=0
    valid_limit:int=0
    max_batches:int=0
    local_minutes:float|None=None
    quantization:str='none'


@tracked
def run(cfg:LoRAConfig):
    seed_all();start=time.monotonic();root=Path(cfg.input_root);out=Path(cfg.output);out.mkdir(parents=True,exist_ok=True)
    if cfg.rank<0 or min(cfg.batch_size,cfg.accumulation,cfg.epochs,cfg.patience)<1:raise ValueError('Invalid configuration')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    if cfg.local_minutes is not None:
        from ecg_project.training.local_budget import check_reservation
        check_reservation(cfg.local_minutes)
    elif device=='cuda' and torch.cuda.get_device_properties(0).total_memory<35*2**30:
        raise ValueError('On a local small GPU set local_minutes explicitly; unlimited mode is for the cluster')
    frame=pd.read_csv(root/'manifest.csv');provenance=json.loads((root/'provenance.json').read_text())
    from ecg_project.processing.hubert import VERSION_V2
    if provenance['preprocessing'] not in (VERSION,VERSION_V2):raise ValueError('Preprocessing mismatch')
    if provenance['preprocessing']==VERSION_V2:
        from ecg_project.data.cache import validate_cache
        validate_cache(root,VERSION_V2)
    if not frame.split.isin(['train','valid']).all():raise ValueError('Invalid splits')
    from ecg_project.data.catalog import assert_disjoint
    assert_disjoint(frame)
    tr=np.flatnonzero(frame.split.to_numpy()=='train');va=np.flatnonzero(frame.split.to_numpy()=='valid')
    classes=[c for c in TARGETS if frame.iloc[tr][c].sum()>=20 and (1-frame.iloc[tr][c]).sum()>=20]
    if not classes:raise ValueError('Insufficient training label support')
    rng=np.random.default_rng(42)
    if cfg.train_limit:tr=np.sort(rng.choice(tr,min(len(tr),cfg.train_limit),replace=False))
    if cfg.valid_limit:va=np.sort(rng.choice(va,min(len(va),cfg.valid_limit),replace=False))
    y=frame[classes].to_numpy(np.float32);signals=np.load(root/'signals.npy',mmap_mode='r')
    identity=dict(config={k:v for k,v in asdict(cfg).items() if k not in ['output','epochs','local_minutes']},
        model_sha256=file_hash(Path(cfg.model_root)/'model.safetensors'),manifest_sha256=file_hash(root/'manifest.csv'),
        signals_sha256=file_hash(root/'signals.npy'),classes=classes,train_indices=tr.tolist(),valid_indices=va.tolist())
    if provenance['preprocessing']==VERSION_V2:identity['preprocessing']=VERSION_V2
    if identity['signals_sha256']!=provenance['signals_sha256']:raise ValueError('Signal cache modified')
    if cfg.quantization not in ('none','nf4'):raise ValueError('Unknown quantization mode')
    if cfg.quantization=='nf4' and not cfg.rank:raise ValueError('QLoRA requires positive adapter rank')
    model=HubertClassifier(load_backbone(cfg.model_root,cfg.quantization),len(classes),cfg.rank,cfg.alpha,cfg.targets,cfg.checkpointing)
    total=sum(p.numel() for p in model.parameters())  # Before NF4 packing changes tensor shapes.
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate,weight_decay=.01)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    pos=torch.tensor(np.sqrt((1-y[tr]).sum(0)/np.maximum(y[tr].sum(0),1)),device=device)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos)
    loader=DataLoader(TensorDataset(torch.from_numpy(np.array(signals[tr])),torch.from_numpy(y[tr])),
        batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.workers,pin_memory=device=='cuda',persistent_workers=cfg.workers>0)
    amp=device=='cuda' and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32=True
    latest=out/'latest.pt';begin=0;best=-1.;stale=0;history=[]
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        saved['identity']['config'].setdefault('quantization','none')
        if saved['identity']!=identity:raise ValueError('Checkpoint identity differs; choose new output')
        load_adapter(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        begin=saved['epoch'];best=saved['best'];stale=saved['stale'];history=saved['history']
        torch.set_rng_state(saved['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def checkpoint(epoch):
        _atomic_save(dict(adapter=adapter_state(model),optimizer=optimizer.state_dict(),identity=identity,epoch=epoch,
            best=best,stale=stale,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),latest)
    save_json(out/'config.json',dict(**asdict(cfg),classes=classes,trainable_parameters=trainable,total_parameters=total,
        adapted_modules=model.adapted_modules,evaluation_status=provenance['evaluation_status']))
    if not latest.exists():checkpoint(0)
    status='complete'
    print('HuBERT parameters',total,'trainable',trainable,flush=True)
    try:
        for epoch in range(begin,cfg.epochs):
            if stale>=cfg.patience:break
            # Frozen trunk dropout is disabled in both comparisons; LoRA dropout stays active.
            model.train();model.backbone.train(bool(cfg.rank and cfg.checkpointing))
            for module in model.backbone.modules():
                if isinstance(module,nn.Dropout):module.eval()
            for module in model.backbone.modules():
                from ecg_project.models.lora import LoRALinear
                if isinstance(module,LoRALinear):module.train()
            optimizer.zero_grad();losses=[];examples=0;n=min(len(loader),cfg.max_batches) if cfg.max_batches else len(loader)
            for k,(x,label) in enumerate(loader):
                if k>=n:break
                examples+=len(x)
                if cfg.local_minutes is not None and time.monotonic()-start>cfg.local_minutes*60:raise TimeoutError('Local run budget reached')
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    logits=model(x.to(device));loss=criterion(logits,label.to(device))
                group=min(cfg.accumulation,n-(k//cfg.accumulation)*cfg.accumulation)
                (loss/group).backward();losses.append(float(loss.detach()))
                if (k+1)%cfg.accumulation==0 or k+1==n:
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step();optimizer.zero_grad()
            model.eval();pred=[]
            with torch.no_grad():
                for ids in np.array_split(va,max(1,int(np.ceil(len(va)/cfg.batch_size)))):
                    if cfg.local_minutes is not None and time.monotonic()-start>cfg.local_minutes*60:raise TimeoutError('Local validation budget reached')
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                        p=model(torch.from_numpy(np.array(signals[ids])).to(device)).sigmoid()
                    pred.append(p.float().cpu().numpy())
            prob=np.concatenate(pred);report=multilabel_metrics(y[va],prob,classes)
            metric=report['macro_auroc'];metric=-1. if metric is None else float(metric)
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),valid_auroc=metric,seconds=time.monotonic()-start,examples=examples))
            if metric>best or not (out/'best.pt').exists():
                best=metric;stale=0;_atomic_save(dict(adapter=adapter_state(model),identity=identity),out/'best.pt')
                save_json(out/'best_metrics.json',dict(**report,evaluation_status=provenance['evaluation_status'],test_accessed=False,best_epoch=epoch+1))
                np.savez_compressed(out/'valid_predictions.npz',truth=y[va],probability=prob,classes=np.array(classes),records=frame.iloc[va].record_id.astype(str).to_numpy(dtype=str))
            else:stale+=1
            checkpoint(epoch+1);save_json(out/'history.json',history);print(history[-1],flush=True)
    except BaseException:
        status='interrupted';raise
    finally:
        from ecg_project.training.local_budget import record_invocation
        elapsed=time.monotonic()-start;record_invocation(out,elapsed,cfg.local_minutes is not None)
        save_json(out/'run.json',dict(seconds_this_invocation=elapsed,device=device,status=status,
            trainable_parameters=trainable,total_parameters=total,rank=cfg.rank,max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            train_records=len(tr),valid_records=len(va),test_accessed=False,evaluation_status=provenance['evaluation_status'],
            quantization=cfg.quantization,smoke=bool(cfg.max_batches)))
    return out


def predict_record(path,run_root,model_root='artifacts/hubert_large',device='cpu'):
    from ecg_project.processing.hubert import VERSION_V2
    root=Path(run_root);config=json.loads((root/'config.json').read_text())
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    if file_hash(Path(model_root)/'model.safetensors')!=saved['identity']['model_sha256']:raise ValueError('Base model mismatch')
    quantization=config.get('quantization','none')
    if quantization=='nf4' and not str(device).startswith('cuda'):raise ValueError('QLoRA inference requires device=cuda')
    model=HubertClassifier(load_backbone(model_root,quantization),len(config['classes']),config['rank'],config['alpha'],tuple(config['targets']),False)
    load_adapter(model,saved['adapter']);model.to(device).eval()
    rec=path if hasattr(path,'signal') else load_record(path)
    if saved['identity'].get('preprocessing')==VERSION_V2:
        from ecg_project.processing.hubert import prepare_signal_v2
        x,_=prepare_signal_v2(rec)
    else:x=prepare_signal(rec)
    with torch.no_grad():prob=model(torch.from_numpy(x[None]).to(device)).sigmoid()[0].cpu().numpy()
    return dict(scores=dict(zip(config['classes'],map(float,prob))),model='HuBERT-ECG Large',rank=config['rank'],
        evaluation_status=config['evaluation_status'],calibrated=False)
