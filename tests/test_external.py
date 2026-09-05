from pathlib import Path
import json
import numpy as np
import pytest
from ecg_project.training.qtdb import manual
from ecg_project.data.io import load_record
from ecg_project.models.segmentation import Predictor

@pytest.mark.skipif(not Path('data/qtdb_external/split.json').exists(),reason='External data optional')
def test_qtdb_source_disjoint_and_missing_T_onset():
    rows=json.loads(Path('data/qtdb_external/split.json').read_text())['records']
    tr={r['origin'] for r in rows if r['split']=='adapt_train'}
    va={r['origin'] for r in rows if r['split']=='external_valid'}
    assert not tr & va
    assert all(not (r['record_id'][3:].isdigit() and len(r['record_id'][3:])==3) for r in rows)
    waves=manual('data/qtdb_external/sel16265.hea')
    assert len([w for w in waves if w['wave']=='QRS'])==30
    assert any(w['wave']=='T' and w['onset'] is None for w in waves)
    assert load_record('data/qtdb_external/sel16265.hea').fs==250

@pytest.mark.skipif(not Path('artifacts/delineator.pt').exists(),reason='Trained artifact optional')
def test_neural_flat_lead_no_hallucinated_intervals():
    p=Predictor()
    assert p.predict(np.zeros((2500,2)),250)==[[],[]]
