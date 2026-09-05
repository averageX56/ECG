"""Patient-cluster bootstrap for delineation; intervals are descriptive, not clinical calibration."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'reports/segmentation_qt_test.json').read_text())
details=r['details'];patients=sorted({d['record'] for d in details});rng=np.random.default_rng(42)
out={}
for wave in ['P','QRS','T']:
    grouped={p:[d for d in details if d['record']==p and d['wave']==wave] for p in patients}
    samples=[]
    for _ in range(500):
        chosen=rng.choice(patients,len(patients),replace=True);rows=[d for p in chosen for d in grouped[p]]
        tp=sum(d['tp'] for d in rows);fp=sum(d['fp'] for d in rows);fn=sum(d['fn'] for d in rows)
        onset=np.array([e for d in rows for e in d['onset_errors_ms']]);offset=np.array([e for d in rows for e in d['offset_errors_ms']])
        samples.append([2*tp/max(1,2*tp+fp+fn),float(np.mean(abs(onset))),float(np.mean(abs(offset))),float(np.mean(abs(offset-onset)))])
    a=np.array(samples)
    out[wave]={k:np.percentile(a[:,i],[2.5,97.5]).tolist() for i,k in enumerate(['f1_ci95','onset_mae_ci95_ms','offset_mae_ci95_ms','width_mae_ci95_ms'])}
(ROOT/'reports/delineation_uncertainty.json').write_text(json.dumps(dict(n_patients=len(patients),bootstrap_replicates=500,
    protocol='Resample patients, retaining all leads and waves; percentile intervals; fixed selected model, selection uncertainty not included.',results=out),indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
