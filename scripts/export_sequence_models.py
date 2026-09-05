from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import torch
from ecg_project.models.sequence_predictor import SequencePredictor
from ecg_project.data.catalog import file_hash
from ecg_project.training.beats import CLASSES,BEAT_FEATURE_NAMES

root=Path('artifacts/representation_experiments')
for name in ['scratch_bert','masked_bert']:
    state=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=True)
    predictor=SequencePredictor({k:v.numpy() for k,v in state.items()},joblib.load(root/'tokens.joblib'))
    joblib.dump(dict(model=predictor,mode='fusion',classes=CLASSES,feature_names=BEAT_FEATURE_NAMES,
        feature_model_hash=file_hash('artifacts/delineator_qt.pt'),trained_lead='MLII',experimental=True),root/(name+'.joblib'))
    print('Exported',name)
