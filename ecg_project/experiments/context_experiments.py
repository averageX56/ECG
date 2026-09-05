"""Validation-only rhythm-context ablations. Never loads test feature files."""
from pathlib import Path
import time
import json
import joblib
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from ecg_project.utils import seed_all, save_json
from ecg_project.training.beats import CLASSES


def context_features(f, w):
    """Unlabelled within-record context; call separately for every recording."""
    n = len(f)
    if not n:
        return np.empty((0, 50+2*len(range(0,w.shape[1],10))), dtype=np.float32)
    rr = f[:, 0].copy()
    rr = np.where(np.isfinite(rr), rr, np.nanmedian(rr))
    rr = np.nan_to_num(rr, nan=1.)
    columns = [f]
    for size in [11, 51, 201]:
        # Explicit padding also handles records shorter than the filter window.
        pad=size//2
        padded=np.pad(rr,(pad,pad),mode='edge')
        baseline = np.maximum(median_filter(padded, size=size, mode='nearest')[pad:pad+n], .1)
        mean = uniform_filter1d(padded, size=size, mode='nearest')[pad:pad+n]
        std = np.sqrt(np.maximum(uniform_filter1d(padded**2, size=size, mode='nearest')[pad:pad+n]-mean**2, 0))
        columns.append(np.column_stack([rr/baseline, f[:, 1]/baseline, std/baseline]))
    idx = np.arange(n)
    for shift in [-3, -2, -1, 1, 2, 3]:
        columns.append(f[np.clip(idx+shift, 0, n-1)][:, [0, 2, 5, 7, 8]])
    # Shape relative to this record's unlabelled median; no patient labels used.
    normalized = w / np.maximum(np.std(w, axis=1, keepdims=True), .01)
    template = np.median(normalized, axis=0)
    columns.extend([normalized[:, ::10], (normalized-template)[:, ::10]])
    return np.concatenate(columns, axis=1).astype(np.float32)


def run(root='artifacts/beat_features_qt', output='artifacts/context_experiments'):
    seed_all(); start=time.monotonic(); root=Path(root); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text()); rows=[]
    for row in manifest:
        if row['split'] not in ['train','valid']: continue
        with np.load(root/(row['record']+'.npz')) as z:
            f=z['interval']; w=z['waveform']; y=z['labels']
            rows.append((row, f, w, context_features(f,w), y))
    train=[r for r in rows if r[0]['split']=='train']
    results={}
    variants=[('context_balanced','balanced',True), ('context_sqrt', 'sqrt',True),
              ('context_natural',None,True), ('fusion_sqrt','sqrt',False)]
    for name,weight,context in variants:
        t=time.monotonic()
        def matrix(r): return r[3] if context else np.concatenate([r[1],r[2]],1)
        x=np.concatenate([matrix(r)[r[4]>=0] for r in train]); y=np.concatenate([r[4][r[4]>=0] for r in train])
        counts=np.bincount(y,minlength=5)
        weights=None if weight is None else (len(y)/np.maximum(counts[y],1))**(.5 if weight=='sqrt' else 1)
        if weights is not None: weights/=weights.mean()
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,
            learning_rate=.07,early_stopping=False,random_state=42)
        model.fit(x,y,sample_weight=weights)
        per_record={}; truths=[]; predictions=[]; extra=np.zeros(5,int)
        for r in rows:
            if r[0]['split']!='valid': continue
            p=model.predict(matrix(r)); known=r[4]>=0
            truths.extend(r[4][known]); predictions.extend(p[known]); extra+=np.bincount(p[~known],minlength=5)
            per_record[r[0]['record']]=classification_report(r[4][known],p[known],labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
        report=classification_report(truths,predictions,labels=list(range(5)),target_names=CLASSES,output_dict=True,zero_division=0)
        truths=np.asarray(truths); predictions=np.asarray(predictions)
        for i,c in enumerate(CLASSES):
            nref=sum(r[0]['reference_classes'].get(c,0) for r in rows if r[0]['split']=='valid')
            tp=int(((truths==i)&(predictions==i)).sum()); npred=int((predictions==i).sum()+extra[i])
            report[c]['end_to_end_f1']=2*tp/(nref+npred) if nref+npred else 0
        score=np.mean([report[c]['end_to_end_f1'] for c in ['N','S','V','F']])
        results[name]=dict(validation=report,per_record=per_record,selection_score=score,seconds=time.monotonic()-t)
        joblib.dump(dict(model=model,context=context),out/(name+'.joblib'))
        save_json(out/'metrics.json',results)
        print(name, 'macro',round(score,4),'S',round(report['S']['end_to_end_f1'],4),'V',round(report['V']['end_to_end_f1'],4),flush=True)
    save_json(out/'budget.json',dict(seconds=time.monotonic()-start))
    save_json(out/'selection.json',dict(selected=max(results,key=lambda k:results[k]['selection_score']),
        criterion='validation end-to-end macro F1 N,S,V,F',test_accessed=False,experimental=True))


if __name__=='__main__': run()
