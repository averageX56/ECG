"""Experimental ECG patch adapter + shared forward/reverse Qwen3 + sample head."""
from pathlib import Path
import json
import torch
from torch import nn
from ecg_project.models.lora import inject_lora,load_adapter
from ecg_project.models.hubert import quantize_backbone
from ecg_project.data.catalog import file_hash


def load_qwen(root,quantization='nf4'):
    from transformers import Qwen3Model
    root=Path(root)
    provenance=json.loads((root/'provenance.json').read_text())
    for name,digest in provenance['sha256'].items():
        if name=='config.json' or name.endswith('.safetensors'):
            if file_hash(root/name)!=digest:raise ValueError('Qwen source hash mismatch: '+name)
    # Local official safetensors only. No LM head or text tokenizer is needed.
    model=Qwen3Model.from_pretrained(str(root),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='sdpa')
    model.embed_tokens=nn.Identity()  # inputs_embeds bypasses vocabulary embeddings.
    model.config.use_cache=False
    if quantization=='nf4':model=quantize_backbone(model)
    elif quantization!='none':raise ValueError('Unknown quantization mode')
    return model


class QwenDelineator(nn.Module):
    def __init__(self,backbone,patch_size=20,rank=16,alpha=32,checkpointing=True):
        super().__init__()
        if patch_size<1:raise ValueError('Positive patch size required')
        self.backbone=backbone;self.patch_size=patch_size
        hidden=backbone.config.hidden_size
        self.input_adapter=nn.Sequential(nn.Linear(patch_size,hidden),nn.LayerNorm(hidden))
        self.head=nn.Linear(hidden,4*patch_size)
        self.adapted_modules=inject_lora(backbone,rank,alpha,('q_proj','v_proj'))
        if checkpointing:backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def forward(self,x):
        # Causal decoder runs in both directions with shared weights, retaining its
        # pretrained attention behavior. This is offline, not streaming, delineation.
        length=x.shape[-1];pad=(-length)%self.patch_size
        x=nn.functional.pad(x,(0,pad),mode='replicate')
        tokens=self.input_adapter(x[:,0].unfold(-1,self.patch_size,self.patch_size))
        both=torch.cat([tokens,tokens.flip(1)],0)
        h=self.backbone(inputs_embeds=both,use_cache=False,return_dict=True).last_hidden_state
        forward,reverse=h.chunk(2,0);h=(forward+reverse.flip(1))*.5
        logits=self.head(h).reshape(len(x),-1,self.patch_size,4).reshape(len(x),-1,4)
        return logits.transpose(1,2)[...,:length]


def load_run(run_root,model_root=None,device='cuda'):
    root=Path(run_root);cfg=json.loads((root/'config.json').read_text())
    model_root=model_root or cfg['model_root']
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
    provenance=json.loads((Path(model_root)/'provenance.json').read_text())
    if provenance!=saved['identity']['base']:raise ValueError('Qwen base differs from training')
    model=QwenDelineator(load_qwen(model_root,cfg['quantization']),cfg['patch_size'],cfg['rank'],cfg['alpha'],False)
    load_adapter(model,saved['adapter']);return model.to(device).eval()


class Predictor:
    """Reuse the established overlap-add and interval decoding at 250 Hz."""
    def __init__(self,run_root,model_root=None,device='cuda'):
        from ecg_project.models.segmentation import Predictor as BasePredictor
        self.device=device;self.model=load_run(run_root,model_root,device)
        self.checkpoint=str(run_root)
        self._predict=BasePredictor.predict
    def predict(self,signal,fs):
        # One lead per call bounds memory independently of channel count.
        import numpy as np
        if signal.ndim==1:signal=signal[:,None]
        return [self._predict(self,signal[:,i],fs)[0] for i in range(signal.shape[1])]
