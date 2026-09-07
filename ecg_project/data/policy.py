"""Shared record-source policy. Unknown subjects never receive random splits."""
import json
from pathlib import Path
import pandas as pd
from ecg_project.data.catalog import TARGETS,assert_disjoint

ALIASES={
    'WFDB_PTB-XL':'PTBXL','PTB-XL':'PTBXL','PTBXL':'PTBXL',
    'Training_WFDB':'CPSC','CPSC':'CPSC','CPSC2018':'CPSC',
    'Training_2':'CPSC_EXTRA','CPSC-extra':'CPSC_EXTRA','CPSC_EXTRA':'CPSC_EXTRA',
    'WFDB_PTB':'PTB','PTB':'PTB','Chapman-Shaoxing':'CHAPMAN','WFDB_ChapmanShaoxing':'CHAPMAN','CHAPMAN':'CHAPMAN',
    'Ningbo':'NINGBO','WFDB_Ningbo':'NINGBO','NINGBO':'NINGBO',
    'WFDB_GEORGIA':'GEORGIA','Georgia':'GEORGIA','GEORGIA':'GEORGIA',
    'Training_StPetersburg':'STP','St Petersburg':'STP','STP':'STP'}
TRAIN_SOURCES=('PTBXL','CPSC','CPSC_EXTRA','PTB','CHAPMAN','NINGBO')
EXTERNAL_SOURCES=('GEORGIA','STP')
POLICY_VERSION='record_source_policy_v2'


def source_name(source):return ALIASES.get(str(source),str(source))


def record_split(source,fold=0):
    source=source_name(source)
    if source=='PTBXL':
        if int(fold) not in range(1,11):raise ValueError('PTB-XL requires official patient folds 1..10')
        return 'valid' if int(fold)==9 else 'test' if int(fold)==10 else 'train'
    if source=='GEORGIA':return 'external'
    if source=='STP':return 'external_long'
    if source in TRAIN_SOURCES:return 'train'
    raise ValueError(f'No split policy for source {source}; configure an adapter explicitly')


def assign_splits(frame):
    frame=frame.copy();frame['source_key']=frame.source.map(source_name)
    frame['split']=[record_split(r.source,getattr(r,'fold',0)) for r in frame.itertuples()]
    frame['patient_id']=frame.patient_id.fillna('') if 'patient_id' in frame else ''
    ptb=frame.source_key=='PTBXL'
    if (frame.loc[ptb,'patient_id']=='').any():raise ValueError('PTB-XL official patient identity is missing')
    assert_disjoint(frame)
    return frame


def build_record_manifest(catalog='artifacts/catalog.csv',sources=None,include_holdout=True):
    if not Path(catalog).is_file():raise FileNotFoundError(f'Missing {catalog}; run python -m pipelines.cpu.run audit')
    frame=pd.read_csv(catalog).fillna('')
    frame=frame[frame.source.map(source_name).isin(TRAIN_SOURCES+EXTERNAL_SOURCES)]
    if sources:
        wanted={source_name(s) for s in sources};missing=wanted-set(frame.source.map(source_name))
        if missing:raise FileNotFoundError(f'Requested datasets absent from catalog: {sorted(missing)}; download/audit them explicitly')
        frame=frame[frame.source.map(source_name).isin(wanted)]
    frame=assign_splits(frame)
    if not include_holdout:frame=frame[frame.split.isin(['train','valid'])]
    # A missing code list means unknown labels, never an all-negative record.
    for c,codes in TARGETS.items():
        expected=frame.labels.map(lambda s:int(bool(set(str(s).split(';')) & set(codes))) if str(s) else -1)
        if c in frame and not (pd.to_numeric(frame[c])==expected).all():raise ValueError(f'Inconsistent SNOMED mapping for {c}; rerun audit')
        frame[c]=expected
    frame['identity_limitation']=frame.patient_id.map(lambda p:'' if p else 'unknown patient; entire-source assignment, no random split')
    return frame.reset_index(drop=True)


def ssl_train_rows(frame,sources):
    frame=assign_splits(frame)
    selected=frame[frame.source_key.isin([source_name(s) for s in sources]) & (frame.split=='train')].copy()
    if selected.source_key.isin(EXTERNAL_SOURCES).any():raise ValueError('External data in SSL')
    return selected
