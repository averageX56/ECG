"""Low-rank linear adapters with a frozen base and zero initial update."""
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self,base:nn.Linear,rank=16,alpha=32,dropout=.05):
        super().__init__()
        if rank<1 or alpha<=0:raise ValueError('Positive rank and alpha required')
        self.base=base
        self.base.requires_grad_(False)
        self.scale=alpha/rank
        self.dropout=nn.Dropout(dropout)
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        update=torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(x),self.lora_A),self.lora_B)
        return self.base(x)+update*self.scale


def inject_lora(model,rank=16,alpha=32,targets=('q_proj','v_proj'),dropout=.05):
    model.requires_grad_(False)
    selected=[(name,module) for name,module in model.named_modules()
              if isinstance(module,nn.Linear) and name.rsplit('.',1)[-1] in targets]
    if not selected:raise ValueError('No matching attention projections found')
    for name,module in selected:
        path,_,leaf=name.rpartition('.')
        parent=model.get_submodule(path) if path else model
        setattr(parent,leaf,LoRALinear(module,rank,alpha,dropout))
    return [name for name,_ in selected]


def adapter_state(model):
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}


def load_adapter(model,state):
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(expected)!=set(state):raise ValueError('Adapter parameter names differ from model configuration')
    if any(expected[n].shape!=state[n].shape for n in expected):raise ValueError('Adapter parameter shape mismatch')
    with torch.no_grad():
        for name,p in expected.items():p.copy_(state[name])
