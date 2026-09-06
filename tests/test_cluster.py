import numpy as np
import torch
import pytest
from ecg_project.training.cluster_training import ResidentBatches


def test_resident_windows_never_cross_record_boundaries():
    records=[dict(split='train',x=np.full((3,54),v,np.float32),y=np.array([0,-1,1])) for v in [1,9]]
    batches=ResidentBatches(records,radius=4,batch_size=2,labeled=True,device='cpu')
    seen=0
    for x,y in batches:
        assert x.shape[1:]==(9,54)
        assert torch.all(x==x[:,0:1,:])
        assert torch.all(y>=0)
        seen+=len(y)
    assert seen==4


def test_resume_rejects_data_changes_without_overwriting_tokens(tmp_path,monkeypatch):
    from ecg_project.training import cluster_training as training
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    data=tmp_path/'data';data.mkdir()
    records=[]
    for name,split in [('a','train'),('b','valid')]:
        (data/(name+'.npz')).write_bytes(b'original')
        records.append(dict(record=name,split=split,x=np.zeros((8,54),np.float32),
            y=np.array([0,1]*4),reference_classes={'N':4,'S':4}))
    monkeypatch.setattr(training,'load_records',lambda _:records)
    monkeypatch.setattr(training,'fit_tokens',lambda _:dict(test_only=True))
    cfg=training.ClusterConfig(data_root=str(data),output=str(tmp_path/'run'),dim=16,layers=1,radius=1,
        batch_size=4,pretrain_epochs=0,finetune_epochs=1,max_batches=1,amp=False,auto_batch=False)
    output=training.train_cluster(cfg)
    original=(output/'tokens.joblib').read_bytes()
    training.train_cluster(cfg)
    (data/'a.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Resume config or data differs'):
        training.train_cluster(cfg)
    assert (output/'tokens.joblib').read_bytes()==original
