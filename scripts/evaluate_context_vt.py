"""Fixed-rule VT validation ablation; no test annotations are opened."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from ecg_project.experiments.context_experiments import context_features
from ecg_project.evaluation.episodes import rhythm_intervals,intersection_seconds
from ecg_project.data.io import mit_annotations
from ecg_project.utils import save_json,seed_all

seed_all()
root=Path('artifacts/beat_features_qt')
manifest=json.loads((root/'manifest.json').read_text())
paths=[Path('artifacts/beat_models/fusion.joblib')]+list(Path('artifacts/context_experiments').glob('*.joblib'))
paths += [p for p in [Path('artifacts/representation_experiments/scratch_bert.joblib')] if p.exists()]
results={}
for path in paths:
    bundle=joblib.load(path); model=bundle['model']
    for threshold in [.5,.8]:
        rows=[]
        for row in manifest:
            if row['split']!='valid': continue
            with np.load(root/(row['record']+'.npz')) as z:
                f=z['interval'];w=z['waveform'];peaks=z['peaks']
                x=context_features(f,w) if bundle.get('context',False) else np.concatenate([f,w],1)
                prob=model.predict_proba(x); vi=list(model.classes_).index(2)
                active=(prob[:,vi]>=threshold)&(model.classes_[prob.argmax(1)]==2)
                groups=np.split(np.flatnonzero(active),np.flatnonzero(np.diff(np.flatnonzero(active))>1)+1)
                pred=[]
                for g in groups:
                    if len(g)>=3 and (peaks[g[-1]]-peaks[g[0]])/(len(g)-1)<=216:
                        pred.append((int(peaks[g[0]]),int(peaks[g[-1]])))
            ref=rhythm_intervals(mit_annotations(Path('data/mit-bih')/(row['record']+'annotations.txt')),650000)
            hits=sum(any(intersection_seconds(r,p,360)>0 for p in pred) for r in ref)
            false=sum(not any(intersection_seconds(r,p,360)>0 for r in ref) for p in pred)
            rows.append(dict(record=row['record'],reference=len(ref),hits=hits,predicted=len(pred),false=false))
        results[path.stem+'/'+str(threshold)]=dict(details=rows,reference=sum(r['reference'] for r in rows),
            hits=sum(r['hits'] for r in rows),false=sum(r['false'] for r in rows),predicted=sum(r['predicted'] for r in rows))
save_json('reports/context_vt_validation.json',results)
print({k:{n:v[n] for n in ['reference','hits','false','predicted']} for k,v in results.items()})
