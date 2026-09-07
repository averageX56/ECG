"""Aligned, confidence-masked hard/soft losses for the Qwen student."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from ecg_project.data.cache import validate_cache
from ecg_project.data.qwen_pseudo import VERSION
from ecg_project.data.protection import assert_unprotected,registry_index


def distillation_loss(student,teacher,valid,temperature=2.):
    if temperature<=0:raise ValueError('Positive distillation temperature required')
    kl=F.kl_div(F.log_softmax(student/temperature,dim=1),F.softmax(teacher/temperature,dim=1),reduction='none').sum(1)*temperature**2
    return kl[valid].mean() if valid.any() else student.sum()*0


class PseudoDataset(Dataset):
    def __init__(self,root,tau=.95,kd_threshold=0.):
        if not 0<tau<=1 or not 0<=kd_threshold<=1:raise ValueError('Invalid confidence thresholds')
        self.root=Path(root);self.meta=validate_cache(root,VERSION)
        self.rows=json.loads((self.root/'manifest.json').read_text())
        registry=registry_index(json.loads((self.root/'protected_registry.json').read_text()))
        self.tau=tau;self.kd_threshold=kd_threshold
        for r in self.rows:
            if r['split']!='train':raise ValueError('Pseudo cache must contain train only')
            assert_unprotected(r,registry)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with np.load(self.root/(self.rows[i]['key']+'.npz')) as z:
            confidence=z['confidence'];valid=z['valid'];logits=z['logits']
            target=logits.argmax(0).astype(np.int64);target[(confidence<self.tau)|~valid]=-100
            return torch.from_numpy(z['x'].copy()),torch.from_numpy(target),torch.from_numpy(logits.copy()),torch.from_numpy(valid&(confidence>=self.kd_threshold))


def strong_signal(x):
    # Changes amplitude only; every target retains its original time coordinate.
    gain=.65+.7*torch.rand(len(x),1,1,device=x.device)
    sign=torch.where(torch.rand(len(x),1,1,device=x.device)<.3,-1.,1.)
    t=torch.arange(x.shape[-1],device=x.device)/250
    phase=torch.rand(len(x),1,1,device=x.device)*6.283185
    drift=.08*torch.sin(6.283185*.3*t[None,None]+phase)
    return x*gain*sign+torch.randn_like(x)*.035+drift
