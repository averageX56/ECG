"""One partial-supervision trainer for U-Net and Qwen, optional EMA consistency."""
from dataclasses import dataclass,asdict
from pathlib import Path
from copy import deepcopy
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from ecg_project.data.cache import validate_cache,CANONICAL_DELINEATOR
from ecg_project.data.delineation_v2 import VERSION
from ecg_project.models.segmentation import Delineator
from ecg_project.models.qwen_delineator import load_qwen,QwenDelineator
from ecg_project.models.lora import adapter_state,load_adapter
from ecg_project.training.cluster_training import _atomic_save
from ecg_project.utils import save_json,seed_all


@dataclass
class DelineationConfig:
    input_root:str='artifacts/qwen_delineation_inputs_v2'
    output:str='artifacts/cluster/delineator'
    architecture:str='unet'
    final_checkpoint:str=CANONICAL_DELINEATOR
    model_root:str='artifacts/qwen3_1.7b'
    quantization:str='nf4'
    patch_size:int=20
    rank:int=16
    alpha:int=32
    epochs:int=100
    patience:int=9
    batch_size:int=32
    accumulation:int=1
    learning_rate:float=2e-4
    qt_weight:float=.5
    consistency_weight:float=.1
    max_batches:int=0
    valid_limit:int=0
    device:str='cuda'
    pseudo_root:str|None=None
    lambda_supervised:float=1.
    lambda_pseudo:float=.5
    lambda_kd:float=0.
    pseudo_confidence_threshold:float=.95
    kd_confidence_threshold:float=0.
    distillation_temperature:float=2.
    pseudo_per_manual:int=1
    pseudo_ramp_epochs:int=5
    warm_start:str|None=None  # A different run's best.pt; optimizer intentionally starts fresh.


def masked_ce(logits,y):
    known=y>=0
    if not known.any():return logits.sum()*0
    loss=nn.functional.cross_entropy(logits,y,ignore_index=-100,reduction='none')
    return loss[known].mean()


def train(cfg:DelineationConfig):
    seed_all();start=time.monotonic();root=Path(cfg.input_root);out=Path(cfg.output)
    meta=validate_cache(root,VERSION)
    if cfg.architecture not in ('unet','qwen') or min(cfg.batch_size,cfg.accumulation,cfg.patience,cfg.epochs)<1:raise ValueError('Invalid delineation configuration')
    manifest=json.loads((root/'manifest.json').read_text());device=cfg.device
    if set(meta['patients']['train']) & set(meta['patients']['valid']):raise ValueError('Patient leakage')
    config=asdict(cfg);identity=dict(config={k:v for k,v in config.items() if k not in ('epochs','output')},cache=meta)
    initial=None
    if cfg.warm_start:
        from ecg_project.data.catalog import file_hash
        if not Path(cfg.warm_start).is_file():raise FileNotFoundError(f'Missing warm-start checkpoint: {cfg.warm_start}')
        identity['warm_start_sha256']=file_hash(cfg.warm_start)
        initial=torch.load(cfg.warm_start,map_location='cpu',weights_only=True)
    pseudo_loader=None;pseudo_meta=None
    if cfg.lambda_supervised<=0 or min(cfg.lambda_pseudo,cfg.lambda_kd)<0 or cfg.pseudo_per_manual<1:raise ValueError('Invalid mixed-supervision weights/ratio')
    if cfg.distillation_temperature<=0 or cfg.pseudo_ramp_epochs<0:raise ValueError('Invalid distillation temperature/ramp')
    if cfg.lambda_kd and not cfg.pseudo_root:raise ValueError('Soft KD requires a pseudo cache')
    if cfg.pseudo_root:
        from ecg_project.training.distillation import PseudoDataset
        pseudo=PseudoDataset(cfg.pseudo_root,cfg.pseudo_confidence_threshold,cfg.kd_confidence_threshold)
        pseudo_meta=pseudo.meta;identity['pseudo']=pseudo_meta
        pseudo_loader=DataLoader(pseudo,batch_size=cfg.batch_size,shuffle=True)
        from ecg_project.data.protection import registry_index,assert_unprotected
        protected=registry_index([r for r in manifest if r['split']=='valid'])
        for row in pseudo.rows:assert_unprotected(row,protected)
    regime='supervised_only' if pseudo_loader is None else 'teacher_student_hard_soft' if cfg.lambda_kd else 'teacher_student_hard'
    if cfg.architecture=='qwen':identity['base']=json.loads((Path(cfg.model_root)/'provenance.json').read_text())
    if initial:
        previous=initial['identity']
        keys=('architecture','patch_size','rank','alpha','quantization')
        if any(previous['config'].get(k)!=config[k] for k in keys) or previous.get('base')!=identity.get('base'):
            raise ValueError('Warm-start architecture/base mismatch')
    out.mkdir(parents=True,exist_ok=True);latest=out/'latest.pt';saved=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('Resume config or cache mismatch; choose a new run')
    elif cfg.architecture=='unet' and cfg.final_checkpoint and Path(cfg.final_checkpoint).exists():
        raise FileExistsError('Final checkpoint already exists; use an explicit new final_checkpoint, never overwrite a different run')
    model=Delineator() if cfg.architecture=='unet' else QwenDelineator(load_qwen(cfg.model_root,cfg.quantization),cfg.patch_size,cfg.rank,cfg.alpha)
    if initial is not None:
        if cfg.architecture=='unet':model.load_state_dict(initial['state_dict'])
        else:load_adapter(model,initial['adapter'])
    model.to(device);teacher=None
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    if (root/'unlabeled_train_x.npy').exists() and cfg.consistency_weight:
        if cfg.architecture=='qwen':raise ValueError('Qwen EMA doubles the base; set consistency_weight=0 for Qwen')
        teacher=deepcopy(model).eval().requires_grad_(False)
    def arrays(split):return tuple(torch.from_numpy(np.load(root/f'{split}_{s}.npy')) for s in ('x','y'))
    x,y=arrays('train');vx,vy=arrays('valid')
    train_sources=np.array([r['source'] for r in manifest if r['split']=='train'])
    valid_sources=np.array([r['source'] for r in manifest if r['split']=='valid'])
    source=torch.tensor((train_sources=='QTDB').astype(np.int64))
    if cfg.valid_limit:vx=vx[:cfg.valid_limit];vy=vy[:cfg.valid_limit];valid_sources=valid_sources[:cfg.valid_limit]
    loader=DataLoader(TensorDataset(x,y,source),batch_size=cfg.batch_size,shuffle=True)
    uloader=None
    if teacher is not None:
        ux=torch.from_numpy(np.load(root/'unlabeled_train_x.npy'));uloader=DataLoader(TensorDataset(ux),batch_size=cfg.batch_size,shuffle=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate)
    begin=0;best=-1.;best_epoch=0;stale=0;history=[]
    if saved:
        if cfg.architecture=='unet':model.load_state_dict(saved['state_dict'])
        else:load_adapter(model,saved['adapter'])
        if teacher is not None:teacher.load_state_dict(saved['teacher'])
        optimizer.load_state_dict(saved['optimizer']);begin=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];stale=saved['stale'];history=saved['history']
        torch.set_rng_state(saved['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    amp=device=='cuda' and torch.cuda.is_bf16_supported();torch.backends.cuda.matmul.allow_tf32=True
    def weights():return {'state_dict':model.state_dict(),'fs':250,'architecture':'Delineator'} if cfg.architecture=='unet' else {'adapter':adapter_state(model)}
    def checkpoint(epoch):
        _atomic_save(dict(**weights(),identity=identity,optimizer=optimizer.state_dict(),teacher=teacher.state_dict() if teacher is not None else None,
            epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,history=history,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else []),latest)
    save_json(out/'config.json',dict(**config,training_regime=regime))
    if saved is None:checkpoint(0)
    processed=0;status='complete'
    try:
        for epoch in range(begin,cfg.epochs):
            if stale>=cfg.patience:break
            model.train();optimizer.zero_grad();losses=[];ui=iter(uloader) if uloader is not None else None
            pi=iter(pseudo_loader) if pseudo_loader is not None else None
            n=min(len(loader),cfg.max_batches) if cfg.max_batches else len(loader)
            for k,(bx,by,bs) in enumerate(loader):
                if k>=n:break
                bx=bx.to(device);by=by.to(device);bs=bs.to(device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                    logits=model(bx*(.75+.5*torch.rand(len(bx),1,1,device=device))+torch.randn_like(bx)*.015)
                    loss=cfg.lambda_supervised*(masked_ce(logits[bs==0],by[bs==0])+cfg.qt_weight*masked_ce(logits[bs==1],by[bs==1]))
                    if ui is not None:
                        try:(ux,)=next(ui)
                        except StopIteration:ui=iter(uloader);(ux,)=next(ui)
                        ux=ux.to(device)
                        with torch.no_grad():confidence,pseudo=teacher(ux).softmax(1).max(1)
                        pseudo=pseudo.masked_fill(confidence<.95,-100)
                        loss+=cfg.consistency_weight*masked_ce(model(ux+torch.randn_like(ux)*.035),pseudo)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite segmentation loss')
                group=min(cfg.accumulation,n-(k//cfg.accumulation)*cfg.accumulation)
                (loss/group).backward();losses.append(float(loss.detach()));processed+=len(bx)
                if pi is not None:
                    from ecg_project.training.distillation import distillation_loss,strong_signal
                    ramp=min(1.,(epoch+1)/cfg.pseudo_ramp_epochs) if cfg.pseudo_ramp_epochs else 1.
                    for _ in range(cfg.pseudo_per_manual):
                        try:px,py,teacher_logits,kd_mask=next(pi)
                        except StopIteration:pi=iter(pseudo_loader);px,py,teacher_logits,kd_mask=next(pi)
                        px=px.to(device);py=py.to(device);teacher_logits=teacher_logits.to(device);kd_mask=kd_mask.to(device)
                        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                            student=model(strong_signal(px))
                            extra=cfg.lambda_pseudo*masked_ce(student,py)+cfg.lambda_kd*distillation_loss(student.float(),teacher_logits,kd_mask,cfg.distillation_temperature)
                        if not torch.isfinite(extra):raise FloatingPointError('Non-finite pseudo/KD loss')
                        (ramp*extra/(group*cfg.pseudo_per_manual)).backward();processed+=len(px)
                if (k+1)%cfg.accumulation==0 or k+1==n:
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step();optimizer.zero_grad()
                    if teacher is not None:
                        with torch.no_grad():
                            for t,s in zip(teacher.parameters(),model.parameters()):t.mul_(.99).add_(s,alpha=.01)
            model.eval();cms={str(s):np.zeros((4,4),np.int64) for s in set(valid_sources)};predictions=[]
            with torch.no_grad():
                for start_idx in range(0,len(vx),cfg.batch_size):
                    stop=start_idx+cfg.batch_size;truth=vy[start_idx:stop].numpy()
                    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):pred=model(vx[start_idx:stop].to(device)).argmax(1).cpu().numpy()
                    predictions.append(pred.astype(np.uint8))
                    for source_name in cms:
                        known=(truth>=0)&(valid_sources[start_idx:stop,None]==source_name)
                        cms[source_name]+=np.bincount(truth[known]*4+pred[known],minlength=16).reshape(4,4)
            reports={s:dict(dice=(2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)).tolist(),confusion=cm.tolist(),
                known_samples=int(cm.sum()),partial_manual=s=='QTDB') for s,cm in cms.items()}
            metric=float(np.mean([np.mean(r['dice'][1:]) for r in reports.values()]))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),valid_score=metric,by_source=reports))
            if metric>best:
                best=metric;best_epoch=epoch+1;stale=0
                best_payload=dict(**weights(),identity=identity,epoch=best_epoch)
                _atomic_save(best_payload,out/'best.pt')
                if cfg.architecture=='unet' and cfg.final_checkpoint:
                    Path(cfg.final_checkpoint).parent.mkdir(parents=True,exist_ok=True);_atomic_save(best_payload,cfg.final_checkpoint)
                save_json(out/'best_metrics.json',dict(valid_macro_wave_dice=best,best_epoch=best_epoch,by_source=reports,test_accessed=False,training_regime=regime))
                for s,r in reports.items():save_json(out/f'{s.lower()}_valid.json',r)
                np.savez_compressed(out/'valid_predictions.npz',prediction=np.concatenate(predictions),truth=vy.numpy(),source=valid_sources)
            else:stale+=1
            checkpoint(epoch+1);save_json(out/'history.json',history);print(history[-1],flush=True)
    except BaseException:status='interrupted';raise
    finally:
        elapsed=time.monotonic()-start
        save_json(out/'run.json',dict(status=status,seconds_this_invocation=elapsed,actual_batch=cfg.batch_size,effective_batch=cfg.batch_size*cfg.accumulation,
            max_memory_allocated=torch.cuda.max_memory_allocated() if device=='cuda' else 0,max_memory_reserved=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
            examples_per_second=processed/elapsed,epochs_completed=len(history),best_epoch=best_epoch,stale=stale,smoke=bool(cfg.max_batches),test_accessed=False,
            training_regime=regime,pseudo_statistics=pseudo_meta,manual_class_counts=torch.bincount(y[y>=0],minlength=4).tolist()))
    return out
