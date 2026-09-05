"""Prespecified synthetic perturbations; validation patients only."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from ecg_project.data.io import load_record,annotations
from ecg_project.data.catalog import ludb_split
from ecg_project.models.segmentation import Predictor
from ecg_project.evaluation.metrics import event_metrics,summarize_events
from ecg_project.utils import seed_all,save_json

def run(output='reports/robustness.json'):
    seed_all();paths=sorted(Path('LUDB').glob('*.hea'),key=lambda p:int(p.stem));split=ludb_split([p.stem for p in paths])
    paths=[p for p in paths if split[p.stem]=='valid'][:5]
    result={}
    for checkpoint in ['artifacts/delineator.pt','artifacts/delineator_transfer.pt','artifacts/delineator_qt.pt']:
        if not Path(checkpoint).exists():continue
        predictor=Predictor(checkpoint);metrics=defaultdict(list);rng=np.random.default_rng(92)
        for p in paths:
            rec=load_record(p);idx=[rec.leads.index(l) for l in ['II','aVR','V1']];x=rec.signal[:,idx]
            t=np.arange(len(x))/rec.fs
            perturbations={'clean':x,'noise_20db':x+rng.normal(size=x.shape)*np.std(x,axis=0)/10,
                'drift_0.3hz':x+.2*np.sin(2*np.pi*.3*t)[:,None],'powerline_50hz':x+.05*np.sin(2*np.pi*50*t)[:,None]}
            for name,signal in perturbations.items():
                pred=predictor.predict(signal,rec.fs)
                for j,i in enumerate(idx):
                    ref=annotations(p,rec.leads[i]);lo=min(w['onset'] for w in ref);hi=max(w['offset'] for w in ref)
                    for wave in ['P','QRS','T']:
                        r=[w for w in ref if w['wave']==wave];q=[w for w in pred[j] if w['wave']==wave and lo<=w['peak']<=hi]
                        metrics[name+'/'+wave].append(event_metrics(r,q,rec.fs))
        result[checkpoint]={k:summarize_events(v) for k,v in metrics.items()}
    save_json(output,dict(protocol='Five fixed LUDB validation patients, II/aVR/V1; synthetic 20dB noise, 0.2mV drift, 0.05mV 50Hz; not real-noise validation.',results=result))
    return result
