"""Serializable per-record beat-sequence inference adapter."""
import numpy as np
import torch
from ecg_project.experiments.representation_experiments import BeatBERT


class SequencePredictor:
    classes_=np.arange(5)
    def __init__(self,state,tokens):
        self.state=state;self.tokens=tokens;self._net=None
    def __getstate__(self):return dict(state=self.state,tokens=self.tokens,_net=None)
    def predict_proba(self,x):
        if not len(x):return np.empty((0,5),np.float32)
        if self._net is None:
            self._net=BeatBERT(54);self._net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()});self._net.eval()
        f=x[:,:11];w=x[:,11:];t=self.tokens;missing=~np.isfinite(f)
        f=np.where(missing,t['median'],f)
        tokens=np.concatenate([((f-t['median'])/t['scale']).clip(-10,10),missing,t['pca'].transform(w).clip(-10,10)],1).astype(np.float32)
        result=[];n=len(tokens)
        with torch.no_grad():
            for start in range(0,n,256):
                ids=np.clip(np.arange(start,min(start+256,n))[:,None]+np.arange(-8,9)[None,:],0,n-1)
                logits,_,_=self._net(torch.from_numpy(tokens[ids]));result.append(logits.softmax(1).numpy())
        return np.concatenate(result)
    def predict(self,x):return self.predict_proba(x).argmax(1)
