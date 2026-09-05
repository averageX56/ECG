from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from ecg_project.data.io import header,load_record,annotations
from ecg_project.data.catalog import ludb_split,assert_disjoint,TARGETS
from ecg_project.evaluation.metrics import match_events,multilabel_metrics
from ecg_project.processing.signal import preprocess,resample,beat_windows,interval_features
from ecg_project.workflows.analysis import ventricular_runs
from ecg_project.training.beats import split_for,load_mit
from ecg_project.models.segmentation import Delineator

def test_missing_dat_does_not_read_mat(tmp_path):
    p=tmp_path/'x.hea';p.write_text('x 1 500 1000\nx.dat 16 1000/mV 16 0 0 0 0 II\n')
    savemat(tmp_path/'x.mat',{'val':np.zeros((1,1000),np.int16)})
    with pytest.raises(ValueError,match='missing DAT'):load_record(p)

def test_mat_gain_baseline_and_unlabeled(tmp_path):
    p=tmp_path/'x.hea';p.write_text('x 1 500 1000\nx.mat 16+24 200(100)/mV 16 0 0 0 0 II\n')
    savemat(tmp_path/'x.mat',{'val':np.full((1,1000),300,np.int16)})
    assert np.all(load_record(p).signal==1)
    assert header(p)['labels']==[]

def test_csv_requires_frequency(tmp_path):
    p=tmp_path/'x.csv';p.write_text('sample,II\n0,100\n1,101\n')
    with pytest.raises(ValueError,match='frequency'):load_record(p)
    assert load_record(p,csv_fs=360).unit!='mV'

def test_leakage_patient_and_hash():
    for column in ['patient_id','signal_hash']:
        f=pd.DataFrame({column:['same','same'],'split':['train','test']})
        with pytest.raises(ValueError,match='Leakage'):assert_disjoint(f)
    assert split_for('201')==split_for('202')=='test'

def test_ludb_split_all_leads_share_patient():
    s=ludb_split(range(1,201));assert pd.Series(s).value_counts().to_dict()=={'train':140,'test':30,'valid':30}
    assert s==ludb_split(range(1,201))

def test_event_matching_one_to_one_and_cardinality():
    assert len(match_events([100,110],[105],10))==1
    assert len(match_events([0,10],[8,19],10))==2
    assert match_events([], [1], 10)==[]

def test_width_units_and_missing():
    f=interval_features([dict(wave='QRS',onset=10,peak=40,offset=70)],np.array([40]),500)
    assert f['qrs_ms_median']==120
    assert f['qrs_ge_120_fraction']==1
    assert np.isnan(f['pr_ms_median'])

def test_filter_phase_and_resampling():
    x=np.zeros(2000);x[1000]=1
    assert np.argmax(preprocess(x,500))==1000
    assert len(resample(x,500))==1000

def test_crop_preserves_time_and_excludes_edges():
    x=np.arange(1000);crops,ids=beat_windows(x,np.array([1,500,999]),500)
    assert crops.shape==(1,450)
    assert ids.tolist()==[1]
    assert crops[0,175]==500

def test_flat_delineation_status():
    from ecg_project.processing.signal import delineate
    w,p,status=delineate(np.zeros(5000),500)
    assert not w and status['status']=='flat_signal'

def test_network_non_divisible_shape():
    import torch
    model=Delineator()
    assert model(torch.zeros(2,1,2500)).shape==(2,4,2500)

def test_vt_requires_consecutive_and_fast():
    def beat(i,p):return dict(beat_index=i,peak=p,class_name='V',probabilities={'V':.9})
    assert len(ventricular_runs([beat(0,0),beat(1,200),beat(2,400)],500))==1
    assert not ventricular_runs([beat(0,0),beat(1,500),beat(2,1000)],500)
    assert not ventricular_runs([beat(0,0),beat(2,200),beat(3,400)],500)

def test_auc_undefined_class_is_not_zero():
    m=multilabel_metrics(np.array([[0],[0]]),np.array([[.1],[.2]]),['VT'])
    assert m['per_class']['VT']['auroc'] is None

@pytest.mark.skipif(not Path('LUDB/1.hea').exists(),reason='Local data optional')
def test_real_ludb_annotation_order():
    r=load_record('LUDB/1.hea');waves=annotations('LUDB/1.hea','II')
    assert r.signal.shape==(5000,12)
    assert all(0<=w['onset']<=w['peak']<=w['offset']<5000 for w in waves)
    assert waves[0]==dict(wave='QRS',onset=644,peak=662,offset=682,lead='II',source='annotation')

@pytest.mark.skipif(not Path('artifacts/mit_headers/100.hea').exists(),reason='Metadata optional')
def test_real_mit_calibration():
    r=load_mit('data/mit-bih/100.csv')
    assert r.signal[0,0]==pytest.approx((995-1024)/200)
