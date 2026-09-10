import numpy as np
import torch

from ecg_project.models.interval_classifier import IntervalClassifier
from ecg_project.processing.trust import smooth_probabilities
from ecg_project.training.interval_classification import augment_wave, collate
from ecg_project.training.trust_intervals import record_features, TrustConfig


def test_soft_mask_alignment_and_single_pass_at_500hz(monkeypatch):
    from ecg_project.training import interval_classification as base
    calls = []
    p = np.zeros((500, 1, 4), np.float32)
    p[:, 0, 0] = .6
    p[:, 0, 2] = .4  # Keep non-winning, sub-0.5 probabilities.
    p[250, 0] = [.1, .2, .6, .1]
    class Predictor:
        def predict(self, signal, fs, **kwargs):
            calls.append(kwargs)
            return [[]], p
    monkeypatch.setattr(base, 'interval_features', lambda *args: (
        np.zeros((1, 226)), np.full((1, 13), np.nan), np.array([1])))
    # Edge peaks must be dropped identically; 500 native samples -> sample 250.
    _, _, _, meta = record_features(np.zeros(1000), 500, np.array([10, 500, 990]),
        Predictor(), TrustConfig(probability_smoothing_ms=0), include_confidence_masks=True)
    masks = meta['confidence_mask']
    assert masks.shape == (1, 4, 226) and len(calls) == 1
    np.testing.assert_allclose(masks[0, :, 88], p[250, 0])
    np.testing.assert_allclose(masks.sum(1), 1)
    assert masks[0, 2, 0] == np.float32(.4)
    assert calls[0]['return_probabilities'] is True


def test_soft_smoothing_preserves_simplex_and_ecg_only_augmentation():
    probabilities = np.random.default_rng(1).dirichlet(np.ones(4), size=100)
    smooth = smooth_probabilities(probabilities)
    np.testing.assert_allclose(smooth.sum(1), 1)
    assert np.all((smooth >= 0) & (smooth <= 1))
    wave = torch.rand(2, 3, 5, 226)
    augmented = augment_wave(wave)
    torch.testing.assert_close(augmented[:, :, 1:], wave[:, :, 1:])
    assert not torch.equal(augmented[:, :, 0], wave[:, :, 0])


def test_model_uses_soft_masks_and_ignores_padding():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = IntervalClassifier(2, 26, input_channels=5).eval()
    wave, feat = torch.randn(1, 2, 5, 226), torch.randn(1, 2, 26)
    with torch.no_grad():
        a = model(wave, feat, torch.ones(1, 2, dtype=torch.bool))
        changed = wave.clone()
        changed[:, :, 1:] = 0
        b = model(changed, feat, torch.ones(1, 2, dtype=torch.bool))
        padded = model(torch.cat([wave, torch.randn(1, 1, 5, 226)*100], 1),
                       torch.cat([feat, torch.randn(1, 1, 26)], 1), torch.tensor([[True, True, False]]))
    assert not torch.allclose(a, b)
    torch.testing.assert_close(a, padded)
    batch = collate([(wave[0].numpy(), feat[0].numpy(), np.ones(2)),
                     (wave[0, :1].numpy(), feat[0, :1].numpy(), np.zeros(2))])
    assert batch[0].shape == (2, 2, 5, 226)
    assert batch[2].tolist() == [[True, True], [True, False]]


def test_unet_and_qwen_probability_api():
    from ecg_project.models.segmentation import Predictor as Unet
    from ecg_project.models.qwen_delineator import Predictor as Qwen
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(len(x), 4, x.shape[-1])
    signal = np.sin(np.arange(600)*.1)
    unet = Unet.__new__(Unet)
    unet.model, unet.device = Model(), 'cpu'
    waves, p = unet.predict(signal, 500, return_probabilities=True)
    assert len(waves) == 1 and p.shape == (300, 1, 4)
    np.testing.assert_allclose(p.sum(-1), 1)
    qwen = Qwen.__new__(Qwen)
    qwen.model, qwen.device, qwen._predict = Model(), 'cpu', Unet.predict
    multi = np.stack([signal, signal], axis=1)
    waves, p = qwen.predict(multi, 500, return_probabilities=True)
    assert len(waves) == 2 and p.shape == (300, 2, 4)
    assert len(qwen.predict(multi, 500)) == 2
