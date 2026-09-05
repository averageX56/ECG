from pathlib import Path
import json
import joblib
import numpy as np
from ecg_project.training.beats import CLASSES
from ecg_project.workflows.analysis import ventricular_runs,selected_path
from ecg_project.data.io import mit_annotations
from ecg_project.utils import save_json,seed_all

def rhythm_intervals(annotations,n_samples,code='(VT'):
    changes=[a for a in annotations if a['symbol']=='+' and a['aux'].startswith('(')]
    return [(r['sample'],changes[i+1]['sample'] if i+1<len(changes) else n_samples)
            for i,r in enumerate(changes) if r['aux']==code]

def intersection_seconds(a,b,fs):
    return max(0,min(a[1],b[1])-max(a[0],b[0]))/fs

def evaluate(root='artifacts/beat_features_qt',output='reports/vt_episodes.json'):
    seed_all();manifest=json.loads((Path(root)/'manifest.json').read_text());model_path=selected_path('artifacts/beat_models')
    if model_path is None:raise ValueError('Train beat model first')
    bundle=joblib.load(model_path);model=bundle['model'];mode=bundle['mode'];rows=[]
    for row in manifest:
        if row['split'] not in ['valid','test']:continue
        with np.load(Path(root)/(row['record']+'.npz')) as z:
            f=z['interval'];w=z['waveform'];peaks=z['peaks'];x={'interval':f,'waveform':w,'fusion':np.concatenate([f,w],1)}[mode]
            prob=model.predict_proba(x);beats=[]
            for k,peak in enumerate(peaks):
                pp={CLASSES[int(c)]:float(prob[k,j]) for j,c in enumerate(model.classes_)}
                beats.append(dict(peak=int(peak),beat_index=k,class_name=max(pp,key=pp.get),probabilities=pp))
        pred=ventricular_runs(beats,360);ranges=[(e['start_sample'],e['end_sample']) for e in pred]
        ann=mit_annotations(Path('data/mit-bih')/(row['record']+'annotations.txt'));ref=rhythm_intervals(ann,650000)
        # Any temporal overlap is a lenient candidate-retrieval criterion, not a boundary-accuracy claim.
        hits=sum(any(intersection_seconds(r,q,360)>0 for q in ranges) for r in ref)
        false=sum(not any(intersection_seconds(r,q,360)>0 for r in ref) for q in ranges)
        overlap=sum(intersection_seconds(r,q,360) for r in ref for q in ranges)
        duration=sum((b-a)/360 for a,b in ref)
        rows.append(dict(record=row['record'],split=row['split'],reference_episodes=len(ref),predicted_episodes=len(pred),
            retrieved_reference_episodes=hits,false_candidate_episodes=false,reference_duration_s=duration,overlap_s=overlap,
            predicted_candidates=pred))
    summary={}
    for s in ['valid','test']:
        rr=[r for r in rows if r['split']==s];n=sum(r['reference_episodes'] for r in rr);m=sum(r['retrieved_reference_episodes'] for r in rr)
        summary[s]=dict(n_records=len(rr),reference_episodes=n,candidate_recall=m/n if n else None,
             predicted_episodes=sum(r['predicted_episodes'] for r in rr),false_candidates=sum(r['false_candidate_episodes'] for r in rr),
             vt_reference_duration_s=sum(r['reference_duration_s'] for r in rr),
             overlap_s=sum(r['overlap_s'] for r in rr))
    report=dict(summary=summary,details=rows,protocol='Fixed >=3 V at >=100bpm and V score >=0.8; retrieval uses any temporal overlap; no thresholds tuned on these episode labels.',
         limitation='Very few VT episodes. Candidate retrieval is not a validated VT diagnostic classifier; rate/morphology cannot rule out aberrant SVT.')
    save_json(output,report);print(summary,flush=True);return report
