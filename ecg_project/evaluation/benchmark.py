from pathlib import Path
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from ecg_project.data.io import load_record,annotations
from ecg_project.processing.signal import delineate
from ecg_project.data.catalog import ludb_split
from ecg_project.evaluation.metrics import event_metrics,summarize_events
from ecg_project.utils import save_json

def benchmark_ludb(root='LUDB',output='reports/delineation_valid.json',split='valid',limit=0,leads=None,modes=('raw','morphology')):
    paths=sorted(Path(root).glob('*.hea'),key=lambda p:int(p.stem))
    splits=ludb_split([p.stem for p in paths]); paths=[p for p in paths if splits[p.stem]==split]
    if limit:paths=paths[:limit]
    results=defaultdict(list);details=[];start=time.monotonic()
    for p in paths:
        rec=load_record(p)
        for i,lead in enumerate(rec.leads):
            if leads and lead not in leads:continue
            ref=annotations(p,lead)
            # LUDB peripheral beats are unannotated. Scoring window is the envelope of annotated waves.
            if not ref:continue
            lo=min(w['onset'] for w in ref);hi=max(w['offset'] for w in ref)
            for mode in modes:
                pred,_,status=delineate(rec.signal[:,i],rec.fs,mode=mode)
                for wave in ('P','QRS','T'):
                    r=[w for w in ref if w['wave']==wave]
                    q=[w for w in pred if w['wave']==wave and lo<=w['peak']<=hi]
                    m=event_metrics(r,q,rec.fs)
                    results[mode+'/'+wave].append(m)
                    details.append(dict(record=p.stem,lead=lead,mode=mode,wave=wave,status=status['status'],**m))
        print(f'LUDB {p.stem} complete; {time.monotonic()-start:.1f}s',flush=True)
    summary={k:summarize_events(v) for k,v in results.items()}
    report=dict(split=split,n_records=len(paths),seconds=time.monotonic()-start,
       protocol='Patient split seed 42; annotated envelope; peak match <=150ms; boundary errors conditional on matched waves; failures count as FN.',
       summary=summary,details=details,
       per_lead={f'{mode}/{lead}/{wave}':summarize_events([x for x in details if x['lead']==lead and x['mode']==mode and x['wave']==wave])
          for mode in modes for lead in sorted({x['lead'] for x in details}) for wave in ('P','QRS','T')})
    save_json(output,report)
    print(pd.DataFrame(summary).T[['precision','recall','f1']].to_string(),flush=True)
    return report
