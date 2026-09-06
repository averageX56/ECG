from pathlib import Path
import time
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from ecg_project.data.catalog import TARGETS,assert_disjoint,file_hash
from ecg_project.data.io import load_record
from ecg_project.processing.features import extract_record
from ecg_project.models.segmentation import Predictor
from ecg_project.utils import seed_all,save_json
from ecg_project.evaluation.metrics import multilabel_metrics,thresholds_on_validation
from tqdm.auto import tqdm

def prepare(catalog='artifacts/catalog.csv',output='artifacts/record_features',limit=0,device='cpu',checkpoint='artifacts/delineator.pt'):
    seed_all();frame=pd.read_csv(catalog).fillna('');frame=frame[(frame.source!='LUDB') & (frame.readable==True) & (frame.has_labels==True)].copy()
    # Entire sources remain held out. Unknown identities never get random record splits.
    frame['split']='train'
    frame.loc[frame.source=='WFDB_GEORGIA','split']='external'
    frame.loc[frame.source=='Training_StPetersburg','split']='external_long'
    frame.loc[(frame.source=='WFDB_PTB-XL') & (frame.fold==9),'split']='valid'
    frame.loc[(frame.source=='WFDB_PTB-XL') & (frame.fold==10),'split']='test'
    if limit:
        # Explicit exploratory sample, chosen independently of diagnosis.
        sampled=frame.groupby(['source','split'],group_keys=False).sample(frac=1,random_state=42).groupby(['source','split'],group_keys=False).head(limit)
        # Retain all rare TRAINING positives, without consulting validation/test labels.
        rare=frame[(frame.split=='train') & (frame[['PVC','AVB2','AVB3','VT']].astype(int).max(axis=1)>0)]
        frame=pd.concat([sampled,rare]).drop_duplicates(['source','record_id'])
    assert_disjoint(frame)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    predictor=Predictor(checkpoint,device=device);rows=[];fail=[];start=time.monotonic();model_hash=file_hash(checkpoint)
    iterator = tqdm(
        frame.iterrows(),
        total=len(frame),
        desc="Record features",
        unit="record",
    )

    for k, (_, row) in enumerate(iterator):
        name=row.source+'_'+row.record_id;cache=out/(name+'.npz')
        try:
            p=Path(row.path);signal_path=p.with_suffix('.mat')
            digest=file_hash(signal_path)
            header_digest=file_hash(p)
            if cache.exists():
                with np.load(cache) as cached:
                    if str(cached['model_hash'])!=model_hash or str(cached['signal_hash'])!=digest or str(cached['header_hash'])!=header_digest:
                        raise ValueError('Stale cache: choose a fresh output directory after changing model/data.')
            else:
                # Long records: keep all windows in the analysis CLI; record benchmark uses first 30s and marks this.
                rec=load_record(p,stop=round(min(30,float(row.duration_s))*float(row.fs)))
                f,w,qc=extract_record(rec,predictor)
                np.savez_compressed(cache,interval=np.array(list(f.values()),np.float32),waveform=np.array(list(w.values()),np.float32),
                    interval_names=np.array(list(f)),waveform_names=np.array(list(w)),signal_hash=digest,header_hash=header_digest,model_hash=model_hash)
            item=row.to_dict();item.update(cache=str(cache),signal_hash=digest,analyzed_seconds=min(30,float(row.duration_s)))
            rows.append(item)
        except Exception as e:
            fail.append(dict(record=name,error=repr(e)));print('ERROR',name,repr(e),flush=True)
        if (k+1)%100==0:
            pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    result=pd.DataFrame(rows)
    # Exact duplicated payloads: keep the most protected split, drop training copies.
    rank={'external_long':0,'external':1,'test':2,'valid':3,'train':4}
    result['_rank']=result.split.map(rank)
    duplicates=result[result.duplicated('signal_hash',keep=False)].copy()
    result=result.sort_values('_rank').drop_duplicates('signal_hash').drop(columns='_rank')
    assert_disjoint(result)
    result.to_csv(out/'manifest.csv',index=False);duplicates.to_csv(out/'duplicates.csv',index=False)
    save_json(out/'extraction.json',dict(seconds=time.monotonic()-start,failures=fail,records=len(result),exploratory_limit_per_source_split=limit))
    return result

def load_matrix(manifest):
    frame=pd.read_csv(manifest).fillna('');fi=[];wf=[]
    assert_disjoint(frame)
    for p in frame.cache:
        with np.load(str(p).replace('\\','/')) as z:fi.append(z['interval']);wf.append(z['waveform'])
    return frame,np.stack(fi),np.stack(wf)

def train(manifest='artifacts/record_features/manifest.csv',output='artifacts/record_models',minutes=45):
    seed_all();start=time.monotonic();frame,fi,wf=load_matrix(manifest);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    # Training-support criterion never uses test labels.
    with np.load(frame.cache.iloc[0]) as z:feature_model_hash=str(z['model_hash'])
    allclasses=list(TARGETS);y=frame[allclasses].to_numpy(dtype=int);tr=frame.split=='train';va=frame.split=='valid'
    active=[i for i,c in enumerate(allclasses) if y[tr,i].sum()>=20 and (1-y[tr,i]).sum()>=20]
    classes=[allclasses[i] for i in active];y=y[:,active]
    if not tr.any() or not va.any():raise ValueError('Train and validation partitions required')
    yt=y[tr];yv=y[va]
    # Train-only co-occurrence report. A source-specific report reveals collection/labeling effects.
    co=(yt.T@yt).astype(float);den=yt.sum(0)
    save_json(out/'label_dependencies.json',dict(classes=classes,joint_counts=co,conditional=co/np.maximum(den[:,None],1),
       per_source={s:frame.loc[tr & (frame.source==s),classes].astype(int).corr().to_dict() for s in frame.loc[tr,'source'].unique()},
       interpretation='Associations of recorded labels, not causal disease relationships; fitted on training only.'))
    results={};budgets={}
    for mode, x in tqdm(
        [
            ("interval", fi),
            ("waveform", wf),
            ("fusion", np.concatenate([fi, wf], 1)),
        ],
        desc="Record models",
        unit="mode",
    ):
        estimators=[];prob=np.zeros((len(frame),len(classes)));t0=time.monotonic()
        for i, c in enumerate(
            tqdm(
                classes,
                desc=mode,
                unit="class",
                leave=False,
            )
        ):
            if time.monotonic()-start>minutes*60:raise TimeoutError('Training budget reached before complete model; rerun with smaller max_iter.')
            # Fixed iterations: sklearn internal random early-stop split would violate patient grouping.
            model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,l2_regularization=2,
                  class_weight='balanced',early_stopping=False,random_state=42)
            model.fit(x[tr],yt[:,i]);prob[:,i]=model.predict_proba(x)[:,1];estimators.append(model)
        thresholds=thresholds_on_validation(yv,prob[va]);budgets[mode]=time.monotonic()-t0
        bundle=dict(models=estimators,mode=mode,classes=classes,thresholds=thresholds,
           unsupported={c:int(frame.loc[tr,c].sum()) for c in allclasses if c not in classes},
           feature_model_hash=feature_model_hash,interval_dim=fi.shape[1],waveform_dim=wf.shape[1])
        joblib.dump(bundle,out/(mode+'.joblib'))
        results[mode]={s:multilabel_metrics(y[frame.split==s],prob[frame.split==s],classes,thresholds)
                       for s in ['valid','test','external','external_long'] if (frame.split==s).any()}
        np.savez_compressed(out/(mode+'_predictions.npz'),probability=prob,labels=y,split=frame.split.to_numpy(dtype=str),
             record_id=frame.record_id.to_numpy(dtype=str),source=frame.source.to_numpy(dtype=str),classes=np.array(classes))
        save_json(out/'metrics.json',results)
    # Select branch using validation only. External/test results never tune the selected model.
    selected=max(results,key=lambda m:results[m]['valid']['macro_auroc'] or 0)
    save_json(out/'selection.json',dict(selected=selected,criterion='validation macro AUROC',unsupported=bundle['unsupported']))
    save_json(out/'budget.json',dict(seconds=time.monotonic()-start,per_mode_seconds=budgets))
    return results

def predict(rec,predictor,model_path):
    bundle=joblib.load(model_path)
    if bundle['feature_model_hash']!=file_hash(predictor.checkpoint):raise ValueError('Delineator checkpoint differs from classifier training')
    f,w,qc=extract_record(rec,predictor);a=np.array(list(f.values()))[None];b=np.array(list(w.values()))[None]
    x={'interval':a,'waveform':b,'fusion':np.concatenate([a,b],1)}[bundle['mode']]
    return {c:dict(probability=float(m.predict_proba(x)[0,1]),threshold=float(t),flag=bool(m.predict_proba(x)[0,1]>=t))
              for c,m,t in zip(bundle['classes'],bundle['models'],bundle['thresholds'])},bundle['unsupported']
