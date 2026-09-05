import numpy as np
import torch
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
