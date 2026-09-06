from pathlib import Path
from collections import Counter
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from ecg_project.data.io import load_record,header,mit_annotations
from ecg_project.processing.signal import preprocess,rpeaks
from ecg_project.models.segmentation import Predictor
from ecg_project.processing.features import beat_features,BEAT_FEATURE_NAMES
from ecg_project.evaluation.metrics import match_events
from ecg_project.utils import seed_all,save_json
from ecg_project.data.catalog import file_hash

CLASSES=['N','S','V','F','Q']
SYMBOL_MAP={**{k:'N' for k in ['N','L','R','e','j']},**{k:'S' for k in ['A','a','J','S']},'V':'V','F':'F','E':'Q','/':'Q','f':'Q','Q':'Q'}
# De Chazal-style record separation, modified to put 201/202 in the same test partition.
DS1={101,106,108,109,112,114,115,116,118,119,122,124,203,205,207,208,209,215,220,223,230}
VALID={108,114,207,223}
PACED={102,104,107,217}

def split_for(record):
    n=int(record)
    return 'valid' if n in VALID else 'train' if n in DS1 else 'test'

def load_mit(path):
    p=Path(path);hp=Path('artifacts/mit_headers')/(p.stem+'.hea')
    if not hp.exists():raise ValueError('Download official MIT headers with scripts/download_metadata.py first')
    h=header(hp);rec=load_record(p,csv_fs=h['fs'])
    if len(rec.signal)!=h['n_samples'] or rec.leads!=[c[-1] for c in h['channels']]:raise ValueError('MIT CSV/header mismatch')
    for i,c in enumerate(h['channels']):
        if rec.signal[0,i]!=int(c[5]):raise ValueError('MIT initial sample mismatch')
        checksum=int(rec.signal[:,i].sum())%65536
        if checksum!=int(c[6])%65536:raise ValueError('MIT checksum mismatch')
        rec.signal[:,i]=(rec.signal[:,i]-float(c[4]))/float(c[2])
    rec.unit='mV';rec.metadata['calibration_status']='official_header_checksum_verified'
    return rec

def _prepare_one(p,output,checkpoint,device):
    from ecg_project.processing.parallel import cached_predictor
    predictor=cached_predictor(checkpoint,device);out=Path(output)
    start=time.monotonic();model_hash=file_hash(checkpoint)
    cache=out/(p.stem+'.npz');report_path=out/(p.stem+'.json')
    if cache.exists():
        with np.load(cache) as z:
            if str(z['model_hash'])!=model_hash:raise ValueError('Stale beat features')
        return json.loads(report_path.read_text())
    rec=load_mit(p)
    # MLII is not substituted for lead II in multilead models; this model is trained on MLII itself.
    if 'MLII' not in rec.leads:raise ValueError(f'MLII absent: {p}')
    ch=rec.leads.index('MLII');x=rec.signal[:,ch]
    waves=predictor.predict(x,rec.fs)[0]
    peaks,_=rpeaks(preprocess(x,rec.fs),rec.fs)
    ref=[a for a in mit_annotations(p.with_name(p.stem+'annotations.txt')) if a['symbol'] in SYMBOL_MAP]
    pairs=match_events([a['sample'] for a in ref],peaks,.15*rec.fs)
    label_by_det={j:CLASSES.index(SYMBOL_MAP[ref[i]['symbol']]) for i,j in pairs}
    crops,f,ids=beat_features(x,peaks,waves,rec.fs)
    y=np.array([label_by_det.get(int(i),-1) for i in ids])
    np.savez_compressed(cache,waveform=crops,interval=f,labels=y,peaks=peaks[ids],model_hash=model_hash)
    row=dict(record=p.stem,patient='201_202' if p.stem in ['201','202'] else p.stem,split=split_for(p.stem),
         references=len(ref),detected=len(peaks),matched=len(pairs),cropped=len(ids),labeled_crops=int((y>=0).sum()),
         reference_classes=dict(Counter(SYMBOL_MAP[a['symbol']] for a in ref)),
         seconds=time.monotonic()-start)
    save_json(report_path,row);return row


def prepare(root='data/mit-bih',output='artifacts/beat_features',device='cpu',checkpoint='artifacts/delineator.pt'):
    from functools import partial
    from ecg_project.processing.parallel import ordered_map
    seed_all();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    paths=[p for p in sorted(Path(root).glob('*.csv')) if int(p.stem) not in PACED]
    function=partial(_prepare_one,output=output,checkpoint=checkpoint,device=device)
    summary=list(ordered_map(function,paths,workers=None if device=='cpu' else 1))
    save_json(out/'manifest.json',summary)
    return summary


def load_features(root):
    root=Path(root)
    if not (root/'manifest.json').exists():raise ValueError('Complete prepare-beats first; the final manifest is not present.')
    manifest=json.loads((root/'manifest.json').read_text());xs=[];fs=[];ys=[];splits=[];records=[]
    for row in manifest:
        with np.load(root/(row['record']+'.npz')) as z:
            valid=z['labels']>=0
            xs.append(z['waveform'][valid]);fs.append(z['interval'][valid]);ys.append(z['labels'][valid])
            splits.extend([row['split']]*int(valid.sum()));records.extend([row['record']]*int(valid.sum()))
    return np.concatenate(xs),np.concatenate(fs),np.concatenate(ys),np.array(splits),np.array(records),manifest

def train(root='artifacts/beat_features',output='artifacts/beat_models',minutes=40):
    seed_all();start=time.monotonic();wf,fi,y,splits,records,manifest=load_features(root)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);tr=splits=='train';results={};times={}
    with np.load(Path(root)/(manifest[0]['record']+'.npz')) as z:feature_model_hash=str(z['model_hash'])
    # Use waveform samples at 250 Hz directly. Every sample's time relative to detected R is fixed.
    for mode,x in [('interval',fi),('waveform',wf),('fusion',np.concatenate([fi,wf],1))]:
        t0=time.monotonic()
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=3,
            learning_rate=.08,class_weight='balanced',early_stopping=False,random_state=42)
        model.fit(x[tr],y[tr]);times[mode]=time.monotonic()-t0
        bundle=dict(model=model,mode=mode,classes=CLASSES,feature_names=BEAT_FEATURE_NAMES,
                    feature_model_hash=feature_model_hash,trained_lead='MLII')
        joblib.dump(bundle,out/(mode+'.joblib'));results[mode]={}
        for s in ['valid','test']:
            sel=splits==s;pred=model.predict(x[sel]);probs=model.predict_proba(x[sel])
            report=classification_report(y[sel],pred,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
            report['confusion_matrix']=confusion_matrix(y[sel],pred,labels=list(range(5))).tolist()
            unmatched_predictions=Counter()
            for row in manifest:
                if row['split']!=s:continue
                with np.load(Path(root)/(row['record']+'.npz')) as z:
                    unknown=z['labels']<0
                    if unknown.any():
                        u={'interval':z['interval'][unknown],'waveform':z['waveform'][unknown],
                           'fusion':np.concatenate([z['interval'][unknown],z['waveform'][unknown]],1)}[mode]
                        unmatched_predictions.update(CLASSES[int(i)] for i in model.predict(u))
            # Detection-inclusive recall: matched-class TP / all reference-class beats, including missed detections and edge crops.
            for c in CLASSES:
                nref=sum(r['reference_classes'].get(c,0) for r in manifest if r['split']==s)
                idx=CLASSES.index(c);tp=int(((y[sel]==idx)&(pred==idx)).sum())
                report[c]['end_to_end_recall']=tp/nref if nref else None
                report[c]['all_reference_support']=nref
                n_pred=int((pred==idx).sum())+unmatched_predictions[c]
                report[c]['end_to_end_precision']=tp/n_pred if n_pred else 0
                report[c]['end_to_end_f1']=2*tp/(nref+n_pred) if nref+n_pred else None
                report[c]['unmatched_detection_predictions']=unmatched_predictions[c]
            report['detection']={k:sum(r[k] for r in manifest if r['split']==s) for k in ['references','detected','matched','cropped','labeled_crops']}
            results[mode][s]=report
            np.savez_compressed(out/(mode+'_'+s+'_predictions.npz'),truth=y[sel],prediction=pred,probability=probs,records=records[sel])
        print(mode,results[mode]['valid']['macro avg'],flush=True);save_json(out/'metrics.json',results)
        if time.monotonic()-start>minutes*60:raise TimeoutError('Beat training budget exhausted')
    best=max(results,key=lambda m:np.mean([results[m]['valid'][c]['f1-score'] for c in ['N','S','V','F']]))
    save_json(out/'selection.json',dict(selected=best,criterion='validation macro F1 on N,S,V,F; Q excluded from selection'))
    save_json(out/'budget.json',dict(seconds=time.monotonic()-start,per_mode_seconds=times))
    return results

def predict(signal,peaks,waves,fs,model_path,checkpoint='artifacts/delineator.pt'):
    bundle=joblib.load(model_path)
    if bundle['feature_model_hash']!=file_hash(checkpoint):raise ValueError('Delineator mismatch')
    crops,f,ids=beat_features(signal,peaks,waves,fs)
    if not len(ids):return []
    x={'interval':f,'waveform':crops,'fusion':np.concatenate([f,crops],1)}[bundle['mode']]
    prob=bundle['model'].predict_proba(x);out=[]
    for k,i in enumerate(ids):
        pp={CLASSES[int(c)]:float(prob[k,j]) for j,c in enumerate(bundle['model'].classes_)}
        cls=max(pp,key=pp.get)
        out.append(dict(peak=int(peaks[i]),beat_index=int(i),class_name=cls,probabilities=pp,
            qrs_ms=float(f[k,5]),qrs_ge_120=bool(f[k,5]>=120) if np.isfinite(f[k,5]) else None,
            rr_previous_relative=float(f[k,2]),compensatory_ratio=float(f[k,4])))
    return out
