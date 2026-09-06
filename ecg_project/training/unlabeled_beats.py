"""Build train-only unlabelled beat sequences from altered Training_2 records."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from ecg_project.data.io import load_record
from ecg_project.models.segmentation import Predictor
from ecg_project.processing.features import beat_features
from ecg_project.processing.signal import preprocess,rpeaks
from ecg_project.data.catalog import file_hash
from ecg_project.utils import save_json
import tqdm.auto as tqdm


def prepare(output='artifacts/unlabeled_beats',limit=0,checkpoint='artifacts/delineator_qt.pt',device='cuda'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv('artifacts/catalog.csv')
    frame=catalog[(catalog.source=='Training_2') & (catalog.readable==True) & (catalog.has_labels==False)].sort_values('record_id')
    if limit:frame=frame.head(limit)
    predictor=Predictor(checkpoint,device=device);digest=file_hash(checkpoint);rows=[];start=time.monotonic()
    for _,row in tqdm.tqdm(frame.iterrows(),total=len(frame)):
        name='unlabeled_Training2_'+str(row.record_id);path=out/(name+'.npz')
        if path.exists():
            with np.load(path) as z:
                if str(z['model_hash'])!=digest:raise ValueError('Stale delineator features')
                n=len(z['labels'])
        else:
            rec=load_record(row.path);ch=rec.leads.index('II') if 'II' in rec.leads else 0
            x=rec.signal[:,ch];waves=predictor.predict(x,rec.fs)[0];peaks,_=rpeaks(preprocess(x,rec.fs),rec.fs)
            w,f,ids=beat_features(x,peaks,waves,rec.fs);n=len(ids)
            if not n:continue
            np.savez_compressed(path,waveform=w,interval=f,labels=np.full(n,-1,dtype=int),peaks=peaks[ids],model_hash=digest)
        rows.append(dict(record=name,patient=name,split='train',reference_classes={},cropped=n,source='Training_2_unlabelled',lead='II_or_first'))
    save_json(out/'manifest.json',rows);save_json(out/'extraction.json',dict(records=len(rows),seconds=time.monotonic()-start,
        limitation='Lead II is not MLII; used for masked pretraining only. Unknown source patient IDs; source training-only.'))
    return out
