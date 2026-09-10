import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from ecg_project.processing.trust import decode_probability_bands, feature_bands
from ecg_project.training.trust_intervals import TrustConfig, METHOD, record_features
from ecg_project.training import interval_classification as base
from ecg_project.data.classifier_labels import LABEL_VERSION
from ecg_project.data.catalog import file_hash


def test_contiguous_core_and_wider_foreground_without_background():
    probs = np.tile([.9, .03, .04, .03], (100, 1))
    probs[20:80] = [.25, .15, .45, .15]
    probs[35:65] = [.1, .05, .8, .05]
    probs[48] = [.25, .15, .45, .15]  # Internal dip does not split the wave.
    waves = decode_probability_bands(probs, np.arange(100), 250, 100, smoothing_ms=0)
    assert len(waves) == 1
    q = waves[0]
    assert (q['onset'], q['offset']) == (35, 64)
    assert (q['onset_lower'], q['offset_upper']) == (25, 74)  # 40 ms cap.
    assert q['onset_lower'] < q['onset_upper'] <= q['offset_lower'] < q['offset_upper']
    probs[45:55] = [.9, .03, .04, .03]
    separated = decode_probability_bands(probs, np.arange(100), 250, 100, smoothing_ms=0)
    assert len(separated) == 2
    assert separated[0]['offset_upper'] < 45 and separated[1]['onset_lower'] >= 55


def test_smoothing_removes_single_sample_piano_but_not_sustained_background():
    probs = np.tile([.9, .03, .04, .03], (100, 1))
    probs[20:80] = [.1, .05, .8, .05]
    probs[49] = [.9, .03, .04, .03]
    assert len(decode_probability_bands(probs, np.arange(100), 250, 100)) == 1
    probs[45:55] = [.9, .03, .04, .03]
    assert len(decode_probability_bands(probs, np.arange(100), 250, 100)) == 2


def test_uncertain_region_without_core_is_not_a_detection():
    probs = np.tile([.2, .1, .4, .3], (50, 1))
    assert decode_probability_bands(probs, np.arange(50), 500, 100) == []
    with pytest.raises(ValueError):
        TrustConfig(uncertainty_threshold=.6).validate()


def test_propagated_qrs_pr_qt_bounds_and_missing_p():
    def wave(name, a, b):
        return dict(wave=name, onset=a, onset_upper=a, onset_lower=a-2, offset=b,
                    offset_lower=b, offset_upper=b+3, peak=(a+b)//2, confidence=.8)
    waves = [wave('P', 50, 70), wave('QRS', 90, 110), wave('T', 140, 180)]
    point = np.ones((1, 13), np.float32)
    point[0, [5, 6, 10, 11, 12]] = [80, 160, 1, 360, 160]
    _, trust = feature_bands(point, [100], waves, 250)
    lower, upper = trust[:, 13:26], trust[:, 26:39]
    assert lower[0, 5] == 80 and upper[0, 5] == 100
    assert lower[0, 6] == 152 and upper[0, 6] == 168
    assert lower[0, 11] == 360 and upper[0, 11] == 380
    assert lower[0, 12] == 160 and upper[0, 12] == 180
    assert trust.shape == (1, 65)
    point[0, 7] = 0
    missing, _ = feature_bands(point, [100], waves[1:], 250)
    assert np.isnan(missing[0, 7])


def test_record_features_uses_one_probability_pass(monkeypatch):
    calls = []
    class Predictor:
        def predict(self, signal, fs, **kwargs):
            calls.append(kwargs)
            return [[]]
    monkeypatch.setattr(base, 'interval_features', lambda *args: (
        np.zeros((1, 226)), np.full((1, 13), np.nan), np.array([0])))
    _, point, trust, _ = record_features(np.zeros(500), 250, np.array([250]), Predictor(), TrustConfig())
    assert len(calls) == 1 and calls[0]['confidence_threshold'] == .5
    assert calls[0]['uncertainty_threshold'] == .3
    assert point.shape == (1, 13) and trust.shape == (1, 65)


@pytest.mark.parametrize('use_masks', [False, True])
def test_preparation_resumes_and_preserves_serving_identity(tmp_path, monkeypatch, use_masks):
    import ecg_project.training.trust_intervals as module
    from ecg_project.models import segmentation, qwen_delineator
    original = tmp_path/'original'
    original.mkdir()
    identity = dict(unet='unet-hash', qwen='qwen-hash')
    (original/'cache_identity.json').write_text(json.dumps(identity))
    frame = pd.DataFrame([dict(key='record', path='fake.hea', source='CPSC')])
    frame.to_csv(original/'manifest.csv', index=False)
    for variant in ('unet', 'qwen'):
        (original/variant).mkdir()
        np.savez(original/variant/'record.npz', wave=np.zeros((1, 226)), peaks=np.array([250]))
    calls = []
    monkeypatch.setattr(segmentation, 'Predictor', lambda *args: calls.append('unet'))
    monkeypatch.setattr(qwen_delineator, 'Predictor', lambda *args: calls.append('qwen'))
    monkeypatch.setattr(module, '_load_signal', lambda row: (np.zeros(500), 250))
    monkeypatch.setattr(base, 'rpeaks', lambda *args: (np.array([250]), None))
    monkeypatch.setattr(module, 'record_features', lambda *args, **kwargs: (
        np.zeros((1, 226)), np.ones((1, 13)), np.ones((1, 65)),
        dict(ids=np.array([0]), waves=[], confidence_mask=np.full((1, 4, 226), .25, np.float32))))
    cfg = base.ClassificationConfig(device='cpu')
    root, point = module.prepare(cfg, TrustConfig(), original, frame, tmp_path/'qwen', tmp_path/'out', use_masks)
    assert calls == ['unet', 'qwen']
    saved_identity = json.loads((root/'cache_identity.json').read_text())
    assert saved_identity['unet'] == 'unet-hash' and saved_identity['qwen'] == 'qwen-hash'
    if use_masks:
        with np.load(root/'qwen/record.npz') as z:
            assert z['confidence_mask'].shape == (1, 4, 226)
            np.testing.assert_allclose(z['confidence_mask'].sum(1), 1)
    with np.load(point/'qwen/record.npz') as z:
        assert z['interval'].shape == (1, 13)
    module.prepare(cfg, TrustConfig(), original, frame, tmp_path/'qwen', tmp_path/'out', use_masks)
    assert calls == ['unet', 'qwen']


@pytest.mark.parametrize('use_masks', [False, True])
def test_trust_classifier_checkpoint_and_serving(tmp_path, monkeypatch, use_masks):
    torch.set_num_threads(2)
    root = tmp_path/'inputs'
    (root/'unet').mkdir(parents=True)
    checkpoint = tmp_path/'unet.pt'
    checkpoint.write_bytes(b'fixture')
    (root/'cache_identity.json').write_text(json.dumps(dict(unet=file_hash(checkpoint))))
    rows = []
    rng = np.random.default_rng(42)
    for i in range(6):
        np.savez(root/'unet'/f'{i}.npz', wave=rng.normal(size=(2, 226)).astype('float32'),
                 confidence_mask=np.full((2, 4, 226), .25, np.float32),
                 interval=rng.normal(size=(2, 65)).astype('float32'))
        rows.append(dict(key=str(i), record_id=str(i), source='fake', AF=i % 2,
                         split='train' if i < 2 else 'valid' if i < 4 else 'test'))
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'manifest.csv', index=False)
    cfg = base.ClassificationConfig(output=str(tmp_path/'out'), device='cpu', epochs=1, batch_size=2)
    spec = dict(method=METHOD, config=asdict(TrustConfig()), mode='trust')
    if use_masks:
        spec.update(mode='masks', confidence_masks=True)
    base.train_branch(cfg, root, frame, 'unet', ['AF'], spec)
    saved_path = tmp_path/'out'/LABEL_VERSION/'unet/best.pt'
    saved = torch.load(saved_path, weights_only=True)
    assert len(saved['median']) == 65 and saved['feature_spec'] == spec
    assert saved['input_channels'] == (5 if use_masks else 1)
    from ecg_project.models import segmentation
    monkeypatch.setattr(segmentation, 'Predictor', lambda *args: object())
    import ecg_project.training.trust_intervals as module
    monkeypatch.setattr(base, 'rpeaks', lambda *args: (np.array([100, 300]), None))
    monkeypatch.setattr(module, 'record_features', lambda *args, **kwargs: (
        np.zeros((2, 226), np.float32), np.zeros((2, 13)), np.zeros((2, 65)),
        dict(confidence_mask=np.full((2, 4, 226), .25, np.float32))))
    predictor = base.ClassificationPredictor(saved_path, checkpoint, device='cpu')
    result = predictor.predict(SimpleNamespace(signal=np.zeros((500, 1)), fs=250, leads=['II']))
    assert result['status'] == 'ok' and 'AF' in result['predictions']
