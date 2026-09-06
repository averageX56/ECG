from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
from ecg_project.data.io import header
from ecg_project.utils import save_json
from tqdm.auto import tqdm

TARGETS = {
    'AF': ['164889003','282825002','426749004','314208002'],
    'PVC': ['427172004','17338001','251182009'],
    'SVEB': ['284470004','63593006','251170000','251164006'],
    'VT': ['164895002','425856008'],
    'AVB1': ['270492004'],
    'AVB2': ['195042002','54016002'],
    'AVB3': ['27885002'],
    'AVB': ['233917008','270492004','195042002','54016002','27885002','204384007'],
    'IVCD': ['713427006','59118001','713426002','164909002','698252002','6374002','82226007','251120003'],
}

def audit(root='.', output='artifacts/catalog.csv'):
    directories = [root / "LUDB", *sorted((root / "data").iterdir())]

    for directory in tqdm(directories, desc="Datasets", unit="dataset"):
        if not directory.is_dir():
            continue

        paths = sorted(directory.glob("*.hea"))

        for p in tqdm(
            paths,
            desc=directory.name,
            unit="record",
            leave=False,
        ):
            h=header(p); codes=h['labels']; meta=h['metadata']
            data=p.parent/h['channels'][0][0]
            source=directory.name
            patient=''; fold=0; identity='unknown'
            if source=='LUDB':patient='LUDB:'+p.stem;identity='dataset_unique_subject'
            if source=='WFDB_PTB-XL' and ptb is not None:
                idx=int(p.stem.removeprefix('HR'))
                if idx in ptb.index:
                    m=ptb.loc[idx]
                    # Check mapping against independent age/sex metadata, fail closed.
                    age=float(meta.get('age','nan'))
                    sex=0 if meta.get('sex','').lower()=='male' else 1
                    if (np.isfinite(age) and age!=m.age) or sex!=int(m.sex):
                        raise ValueError(f'PTB-XL mapping mismatch: {p}')
                    patient='PTBXL:'+str(int(m.patient_id));fold=int(m.strat_fold);identity='official_patient_id'
            row=dict(record_id=p.stem,source=source,path=p.relative_to(root).as_posix(),fs=h['fs'],
                n_samples=h['n_samples'],n_leads=h['n_leads'],duration_s=h['n_samples']/h['fs'],
                labels=';'.join(codes),has_labels=bool(codes),patient_id=patient,identity=identity,fold=fold,
                readable=data.exists(),referenced_format=data.suffix,
                modified_header=(source!='LUDB' and data.suffix=='.dat'))
            row.update({k:int(bool(set(codes)&set(v))) if codes else -1 for k,v in TARGETS.items()})
            rows.append(row);counts['records']+=1;counts['labeled']+=bool(codes);counts['readable']+=data.exists();dx.update(codes)
        if counts:summaries[directory.name]={'counts':dict(counts),'diagnoses':dict(dx)}
    frame=pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
    save_json(root/'reports/dataset_audit.json',dict(datasets=summaries,targets=TARGETS,
        mit_bih_csv_count=len(list((root/'data/mit-bih').glob('*.csv'))),
        notes=['Unknown patient identities are not claimed to be patient-disjoint.',
               'Missing Dx is unlabeled, never a negative label.',
               'Existing LUDB beat export is not an independent dataset.']))
    print(frame.groupby('source')[['readable','has_labels']].agg(['count','sum']).to_string(),flush=True)
    return frame

def ludb_split(ids, seed=42):
    ids=np.array(sorted(set(map(str,ids)),key=int));np.random.default_rng(seed).shuffle(ids)
    n=len(ids); nt=round(.15*n);nv=round(.15*n)
    return {p:('test' if i<nt else 'valid' if i<nt+nv else 'train') for i,p in enumerate(ids)}

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def assert_disjoint(frame):
    for column in ('patient_id','signal_hash'):
        if column not in frame:continue
        known=frame[frame[column].notna() & (frame[column]!='')]
        if (known.groupby(column)['split'].nunique()>1).any():
            raise ValueError(f'Leakage across splits in {column}')
