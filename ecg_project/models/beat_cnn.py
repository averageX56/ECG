"""Small learned morphology + interval branch for MIT-BIH beat classification."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import classification_report,confusion_matrix
from ecg_project.training.beats import load_features,CLASSES,BEAT_FEATURE_NAMES
from ecg_project.utils import seed_all,save_json

class BeatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(1,16,9,padding=4),nn.GroupNorm(4,16),nn.SiLU(),nn.AvgPool1d(2),
            nn.Conv1d(16,32,7,padding=3),nn.GroupNorm(4,32),nn.SiLU(),nn.AvgPool1d(2),
            nn.Conv1d(32,64,5,padding=2),nn.GroupNorm(4,64),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(128+22,64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,5))
    def forward(self,wave,features):
        z=self.encoder(wave)
        return self.head(torch.cat([z.mean(-1),z.amax(-1),features],1))

class BeatPredictor:
    """Sklearn-compatible wrapper with saved train-only imputation/scaling."""
    classes_=np.arange(5)
    def __init__(self,state,median,scale):self.state=state;self.median=median;self.scale=scale;self._net=None
    def __getstate__(self):return dict(state=self.state,median=self.median,scale=self.scale,_net=None)
    def features(self,f):
        mask=~np.isfinite(f);x=np.where(mask,self.median,f)
        return np.concatenate([((x-self.median)/self.scale).clip(-10,10),mask.astype(float)],1).astype(np.float32)
    def predict_proba(self,x):
        if self._net is None:
            self._net=BeatNet();self._net.load_state_dict({k:torch.from_numpy(v) for k,v in self.state.items()});self._net.eval()
        f=self.features(x[:,:11]);w=x[:,11:].astype(np.float32);out=[]
        with torch.no_grad():
            for start in range(0,len(x),512):
                out.append(self._net(torch.from_numpy(w[start:start+512,None]),torch.from_numpy(f[start:start+512])).softmax(1).numpy())
        return np.concatenate(out)
    def predict(self,x):return self.predict_proba(x).argmax(1)

def train(root='artifacts/beat_features_qt',output='artifacts/beat_models',epochs=25,minutes=20):
    seed_all();start=time.monotonic();device='cuda' if torch.cuda.is_available() else 'cpu'
    w,f,y,splits,records,manifest=load_features(root);tr=splits=='train';va=splits=='valid';out=Path(output);out.mkdir(parents=True,exist_ok=True)
    median=np.nanmedian(f[tr],axis=0);median=np.nan_to_num(median)
    filled=np.where(np.isfinite(f[tr]),f[tr],median);scale=np.maximum(np.std(filled,axis=0),1e-5)
    wrapper=BeatPredictor({},median,scale);features=wrapper.features(f)
    dl=DataLoader(TensorDataset(torch.from_numpy(w[tr,None]),torch.from_numpy(features[tr]),torch.from_numpy(y[tr])),batch_size=256,shuffle=True)
    vl=DataLoader(TensorDataset(torch.from_numpy(w[va,None]),torch.from_numpy(features[va])),batch_size=512)
    model=BeatNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    count=np.bincount(y[tr],minlength=5);weights=np.sqrt(count.sum()/np.maximum(count,1));weights=np.minimum(weights,20);weights/=weights.mean()
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device));best=-1;history=[]
    beststate=None
    for epoch in range(epochs):
        model.train();losses=[]
        for wave,feat,label in dl:
            if time.monotonic()-start>minutes*60:break
            wave=wave.to(device);feat=feat.to(device);label=label.to(device)
            # Small amplitude/noise perturbations preserve interval durations and peak alignment.
            wave=wave*(.9+.2*torch.rand(len(wave),1,1,device=device))+torch.randn_like(wave)*.02
            loss=loss_fn(model(wave,feat),label);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval();pred=[]
        with torch.no_grad():
            for wave,feat in vl:pred.extend(model(wave.to(device),feat.to(device)).argmax(1).cpu().numpy())
        r=classification_report(y[va],pred,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
        score=np.mean([r[c]['f1-score'] for c in ['N','S','V','F']])
        history.append(dict(epoch=epoch+1,loss=np.mean(losses),valid_macro_f1_NSVF=score,seconds=time.monotonic()-start));print(history[-1],flush=True)
        if score>best:
            best=score;beststate={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        if time.monotonic()-start>minutes*60:break
    wrapper=BeatPredictor(beststate,median,scale)
    with np.load(Path(root)/(manifest[0]['record']+'.npz')) as z:delineator_hash=str(z['model_hash'])
    joblib.dump(dict(model=wrapper,mode='fusion',classes=CLASSES,feature_model_hash=delineator_hash,trained_lead='MLII',feature_names=BEAT_FEATURE_NAMES),out/'cnn_fusion.joblib')
    data=json.loads((out/'metrics.json').read_text()) if (out/'metrics.json').exists() else {};data['cnn_fusion']={}
    x=np.concatenate([f,w],1)
    for s in ['valid','test']:
        sel=splits==s;prob=wrapper.predict_proba(x[sel]);p=prob.argmax(1)
        r=classification_report(y[sel],p,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
        r['confusion_matrix']=confusion_matrix(y[sel],p,labels=list(range(5))).tolist();unknown_counts=np.zeros(5,dtype=int)
        for row in manifest:
            if row['split']!=s:continue
            with np.load(Path(root)/(row['record']+'.npz')) as z:
                unknown=z['labels']<0
                if unknown.any():unknown_counts+=np.bincount(wrapper.predict(np.concatenate([z['interval'][unknown],z['waveform'][unknown]],1)),minlength=5)
        for i,c in enumerate(CLASSES):
            nref=sum(row['reference_classes'].get(c,0) for row in manifest if row['split']==s);tp=int(((y[sel]==i)&(p==i)).sum());npred=int((p==i).sum())+unknown_counts[i]
            r[c].update(all_reference_support=nref,end_to_end_recall=tp/nref if nref else None,
                end_to_end_precision=tp/npred if npred else 0,end_to_end_f1=2*tp/(nref+npred) if nref+npred else None,unmatched_detection_predictions=int(unknown_counts[i]))
        r['detection']={k:sum(row[k] for row in manifest if row['split']==s) for k in ['references','detected','matched','cropped','labeled_crops']}
        data['cnn_fusion'][s]=r
        np.savez_compressed(out/('cnn_fusion_'+s+'_predictions.npz'),truth=y[sel],prediction=p,probability=prob,records=records[sel])
    save_json(out/'metrics.json',data)
    selected=max(data,key=lambda m:np.mean([data[m]['valid'][c]['f1-score'] for c in ['N','S','V','F']]))
    save_json(out/'selection.json',dict(selected=selected,criterion='validation macro F1 on N,S,V,F; Q excluded from selection'))
    save_json(out/'cnn_history.json',history)
    save_json(out/'cnn_budget.json',dict(seconds=time.monotonic()-start,device=device,max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))
