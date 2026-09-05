import numpy as np
from ecg_project.experiments.context_experiments import context_features


def test_context_short_record_and_empty():
    f=np.ones((2,11),dtype=np.float32)
    w=np.ones((2,226),dtype=np.float32)
    out=context_features(f,w)
    assert out.shape==(2,96)
    assert np.isfinite(out).all()
    assert context_features(f[:0],w[:0]).shape==(0,96)
    np.testing.assert_array_equal(f,np.ones((2,11)))
