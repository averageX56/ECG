import joblib
import numpy as np

from ecg_project.models.beat_cnn import BeatNet, BeatPredictor
from ecg_project.evaluation.episodes import rhythm_intervals, intersection_seconds


def test_predictor_roundtrip_missing_features(tmp_path):
    net = BeatNet()
    state = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
    predictor = BeatPredictor(state, np.zeros(11), np.ones(11))
    x = np.zeros((3, 211), dtype=np.float32)
    x[0, 0] = np.nan
    x[1, 1] = np.inf
    before = predictor.predict_proba(x)
    assert before.shape == (3, 5)
    assert np.isfinite(before).all()
    np.testing.assert_allclose(before.sum(1), 1, atol=1e-6)
    path = tmp_path / 'model.joblib'
    joblib.dump(predictor, path)
    restored = joblib.load(path)
    assert restored._net is None
    np.testing.assert_allclose(restored.predict_proba(x), before)


def test_vt_reference_ends_at_next_rhythm_or_record_end():
    annotations = [dict(sample=10, symbol='+', aux='(VT'),
                   dict(sample=20, symbol='V', aux=''),
                   dict(sample=40, symbol='+', aux='(N'),
                   dict(sample=80, symbol='+', aux='(VT')]
    assert rhythm_intervals(annotations, 100) == [(10, 40), (80, 100)]
    assert intersection_seconds((10, 40), (40, 80), 10) == 0
    assert intersection_seconds((10, 40), (20, 50), 10) == 2
