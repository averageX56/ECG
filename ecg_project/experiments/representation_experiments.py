"""Train-only latent representations and bidirectional masked-beat Transformer.

Validation is record-separated; test files are never opened. This is a small
BERT-style continuous-token model, not an imported text BERT checkpoint.
"""
from pathlib import Path
import json
import time
import copy
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from ecg_project.utils import seed_all,save_json
from ecg_project.training.beats import CLASSES


class BeatBERT(nn.Module):
    def __init__(self,input_dim,dim=96,layers=3,length=17):
        super().__init__()
        self.project=nn.Linear(input_dim,dim)
        self.position=nn.Parameter(torch.randn(1,length,dim)*.02)
        self.mask_token=nn.Parameter(torch.zeros(dim))
        block=nn.TransformerEncoderLayer(dim,4,dim_feedforward=dim*3,dropout=.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(dim)
        self.reconstruction=nn.Linear(dim,input_dim)
        self.head=nn.Linear(dim,5)

    def forward(self,x,mask=None):
        z=self.project(x)
        if mask is not None:z=torch.where(mask[:,:,None],self.mask_token,z)
        z=self.norm(self.encoder(z+self.position[:,:x.shape[1]]))
        return self.head(z[:,x.shape[1]//2]),self.reconstruction(z),z[:,x.shape[1]//2]


class Sequences(Dataset):
    def __init__(self,records,split,labeled=True,radius=8):
        self.records=records;self.radius=radius;self.index=[]
        for j,r in enumerate(records):
            if r['split']==split:
                self.index.extend((j,int(i)) for i in np.flatnonzero(r['y']>=0 if labeled else np.ones(len(r['y']),bool)))
    def __len__(self):return len(self.index)
    def __getitem__(self,k):
        j,i=self.index[k];r=self.records[j]
        ids=np.clip(np.arange(i-self.radius,i+self.radius+1),0,len(r['x'])-1)
        return torch.from_numpy(r['x'][ids]),int(r['y'][i])


def load_records(root):
    records=[]
    for row in json.loads((Path(root)/'manifest.json').read_text()):
        if row['split'] not in ['train','valid']:continue
        with np.load(Path(root)/(row['record']+'.npz')) as z:
            records.append(dict(**row,f=z['interval'],w=z['waveform'],y=z['labels']))
    return records


def fit_tokens(records):
    train=[r for r in records if r['split']=='train']
    f=np.concatenate([r['f'] for r in train]);w=np.concatenate([r['w'] for r in train])
    med=np.nan_to_num(np.nanmedian(f,axis=0));scale=np.maximum(np.std(np.where(np.isfinite(f),f,med),axis=0),1e-3)
    pca=PCA(n_components=32,whiten=True,random_state=42).fit(w)
    for r in records:
        missing=~np.isfinite(r['f']);f=np.where(missing,med,r['f'])
        r['x']=np.concatenate([((f-med)/scale).clip(-10,10),missing.astype(float),pca.transform(r['w']).clip(-10,10)],1).astype(np.float32)
    return dict(pca=pca,median=med,scale=scale)


def score(records,probabilities):
    truth=[];pred=[];unmatched=np.zeros(5,int);per_record={}
    for r,p in zip(records,probabilities):
        if r['split']!='valid':continue
        yp=p.argmax(1);known=r['y']>=0
        truth.extend(r['y'][known]);pred.extend(yp[known]);unmatched+=np.bincount(yp[~known],minlength=5)
        per_record[r['record']]=classification_report(r['y'][known],yp[known],labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
    truth=np.asarray(truth);pred=np.asarray(pred)
    report=classification_report(truth,pred,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
    for i,c in enumerate(CLASSES):
        ref=sum(r['reference_classes'].get(c,0) for r in records if r['split']=='valid')
        tp=int(((truth==i)&(pred==i)).sum());n=int((pred==i).sum()+unmatched[i])
        report[c].update(end_to_end_f1=2*tp/(ref+n) if ref+n else 0,reference=ref)
    return dict(validation=report,per_record=per_record,
        selection_score=float(np.mean([report[c]['end_to_end_f1'] for c in ['N','S','V','F']])))


def encode(model,records,device):
    model.eval();probabilities=[];embeddings=[]
    with torch.no_grad():
        for r in records:
            pp=[];ee=[];n=len(r['x'])
            for start in range(0,n,256):
                ids=np.clip(np.arange(start,min(start+256,n))[:,None]+np.arange(-8,9)[None,:],0,n-1)
                logits,_,z=model(torch.from_numpy(r['x'][ids]).to(device))
                pp.append(logits.softmax(1).cpu().numpy());ee.append(z.cpu().numpy())
            probabilities.append(np.concatenate(pp));embeddings.append(np.concatenate(ee))
    return probabilities,embeddings


def run(root='artifacts/beat_features_qt',output='artifacts/representation_experiments',minutes=35):
    seed_all();start=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    from ecg_project.training.local_budget import read_ledger
    ledger=read_ledger()
    if ledger['total_seconds']+minutes*60>10800:raise RuntimeError('Requested reservation exceeds remaining 180-minute budget; ask user first.')
    device='cuda' if torch.cuda.is_available() else 'cpu';results={};history=[]
    def guard():
        if time.monotonic()-start>minutes*60:raise TimeoutError('Reserved training time reached')
    def save(name,result):
        results[name]=result;save_json(out/'metrics.json',results)
        print(name,'macro',round(result['selection_score'],4),'S',round(result['validation']['S']['end_to_end_f1'],4),'V',round(result['validation']['V']['end_to_end_f1'],4),flush=True)
    try:
        records=load_records(root);tokens=fit_tokens(records);joblib.dump(tokens,out/'tokens.joblib')
        def latent_hgb(name,embeddings):
            guard();xx=[np.concatenate([r['x'][:,:22],e],1) for r,e in zip(records,embeddings)]
            train=[i for i,r in enumerate(records) if r['split']=='train']
            x=np.concatenate([xx[i][records[i]['y']>=0] for i in train]);y=np.concatenate([records[i]['y'][records[i]['y']>=0] for i in train])
            counts=np.bincount(y,minlength=5);weights=np.sqrt(len(y)/np.maximum(counts[y],1));weights/=weights.mean()
            model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=42)
            model.fit(x,y,sample_weight=weights);joblib.dump(model,out/(name+'.joblib'))
            pp=[model.predict_proba(a) for a in xx];save(name,score(records,pp))
        latent_hgb('pca_latent_hgb',[r['x'][:,22:] for r in records])
        ds=Sequences(records,'train');ssl=Sequences(records,'train',labeled=False)
        dl=DataLoader(ds,batch_size=128,shuffle=True);sl=DataLoader(ssl,batch_size=128,shuffle=True)
        counts=np.bincount([records[j]['y'][i] for j,i in ds.index],minlength=5)
        weights=np.sqrt(counts.sum()/np.maximum(counts,1));weights/=weights.mean()
        criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
        initial=BeatBERT(records[0]['x'].shape[1]).to(device)
        initial_state=copy.deepcopy(initial.state_dict())
        for mode in ['scratch_bert','masked_bert']:
            model=BeatBERT(records[0]['x'].shape[1]).to(device);model.load_state_dict(initial_state)
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            if mode=='masked_bert':
                for epoch in range(8):
                    model.train();losses=[]
                    for x,_ in sl:
                        guard();x=x.to(device);mask=torch.rand(x.shape[:2],device=device)<.25
                        _,reconstructed,_=model(x,mask)
                        loss=((reconstructed-x)**2)[mask].mean()
                        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
                    print('SSL',epoch+1,round(float(np.mean(losses)),4),flush=True)
                torch.save(model.state_dict(),out/'masked_pretrained.pt')
                _,emb=encode(model,records,device);latent_hgb('masked_frozen_latent_hgb',emb)
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            best=-1;best_state=None
            for epoch in range(15):
                model.train();losses=[]
                for x,y in dl:
                    guard();x=x.to(device);y=y.to(device);logits,_,_=model(x)
                    loss=criterion(logits,y);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
                valid=[r for r in records if r['split']=='valid'];pp,_=encode(model,valid,device);s=score(valid,pp)
                history.append(dict(mode=mode,epoch=epoch+1,loss=float(np.mean(losses)),score=s['selection_score'],elapsed=time.monotonic()-start))
                print(mode,epoch+1,round(s['selection_score'],4),flush=True)
                if s['selection_score']>best:best=s['selection_score'];best_state=copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state);torch.save(model.state_dict(),out/(mode+'.pt'))
            pp,emb=encode(model,records,device);save(mode,score(records,pp));latent_hgb(mode+'_latent_hgb',emb)
            for r,p,e in zip(records,pp,emb):
                np.savez_compressed(out/(mode+'_'+r['record']+'.npz'),probability=p,embedding=e)
        save_json(out/'protocol.json',dict(test_accessed=False,input_dim=54,sequence_length=17,
            parameters=sum(p.numel() for p in initial.parameters()),selection='validation end-to-end macro F1 N,S,V,F',
            note='Continuous masked-beat reconstruction; not text BERT or an exact HeartBERT reproduction.'))
    finally:
        save_json(out/'history.json',history)
        save_json(out/'budget.json',dict(seconds=time.monotonic()-start,reservation_minutes=minutes,device=device,
            max_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0))


if __name__=='__main__':run()
