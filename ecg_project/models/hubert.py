"""HuBERT ECG encoder loaded from pinned, tensor-only official weights."""
from pathlib import Path
import json
import torch
from torch import nn
from ecg_project.models.lora import inject_lora


class HubertClassifier(nn.Module):
    def __init__(self,backbone,n_classes,rank=16,alpha=32,targets=('q_proj','v_proj'),checkpointing=True):
        super().__init__();self.backbone=backbone
        self.head=nn.Sequential(nn.LayerNorm(backbone.config.hidden_size),nn.Linear(backbone.config.hidden_size,n_classes))
        backbone.requires_grad_(False)
        backbone.feature_extractor._freeze_parameters()
        self.adapted_modules=inject_lora(backbone,rank,alpha,targets) if rank else []
        if rank and checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    def forward(self,x):
        # Each record contributes two non-overlapping five-second 12-lead views.
        b,v,t=x.shape
        hidden=self.backbone(input_values=x.reshape(b*v,t),return_dict=True).last_hidden_state
        pooled=hidden.mean(1).reshape(b,v,-1).mean(1)
        return self.head(pooled)


def quantize_backbone(model):
    """NF4 + double quantization of Linear layers; convolution/norm stay FP32.

    Convert on CPU, then move to CUDA to initialize bitsandbytes quantization.
    No downloaded Python or remote model loader is executed.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('QLoRA requires CUDA in this pipeline')
    import bitsandbytes as bnb
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    selected=[(name,m) for name,m in model.named_modules() if isinstance(m,nn.Linear)]
    for name,base in selected:
        # Replacement initialization must not change the subsequent adapter/head seed.
        with torch.random.fork_rng(devices=[]):
            layer=bnb.nn.Linear4bit(base.in_features,base.out_features,bias=base.bias is not None,
                compute_dtype=dtype,compress_statistics=True,quant_type='nf4')
        layer.load_state_dict(base.state_dict())
        layer.requires_grad_(False)
        path,_,leaf=name.rpartition('.')
        setattr(model.get_submodule(path) if path else model,leaf,layer)
    return model


def load_backbone(root='artifacts/hubert_large',quantization='none'):
    from safetensors.torch import load_file
    from transformers import HubertConfig,HubertModel
    root=Path(root);raw=json.loads((root/'config.json').read_text())
    config=HubertConfig(**{k:v for k,v in raw.items() if k not in ['model_type','auto_map','architectures']})
    config.apply_spec_augment=False;config.layerdrop=0.
    # Native Transformers implementation; downloaded custom Python is not executed.
    model=HubertModel(config)
    weights=load_file(str(root/'model.safetensors'))
    encoder={k:v for k,v in weights.items() if not k.startswith(('final_proj.','label_embedding.'))}
    model.load_state_dict(encoder,strict=True)
    if quantization=='nf4':model=quantize_backbone(model)
    elif quantization!='none':raise ValueError('Unknown quantization mode')
    return model
