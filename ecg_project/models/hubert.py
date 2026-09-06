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


def load_backbone(root='artifacts/hubert_large'):
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
    return model
