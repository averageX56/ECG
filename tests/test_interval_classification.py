import json
import numpy as np
import pandas as pd
import pytest
import torch
from ecg_project.models.interval_classifier import IntervalClassifier
from ecg_project.training.interval_classification import (
    ClassificationConfig, selected_qwen, scale_features, metrics_by_cohort, train_branch,
)
from ecg_project.data.catalog import file_hash


def test_curriculum_uses_accepted_checkpoint_and_checks_hash(tmp_path):
    for name in ('cycle_01', 'cycle_02'):
        root = tmp_path/name
        root.mkdir()
        (root/'best.pt').write_bytes(name.encode())
        (root/'config.json').write_text('{}')
    journal = dict(best_checkpoint='/old/cluster/cycle_01/best.pt', cycles=[
        dict(accepted=True, best_checkpoint_sha256=file_hash(tmp_path/'cycle_01/best.pt')),
        dict(accepted=False, best_checkpoint_sha256=file_hash(tmp_path/'cycle_02/best.pt'))])
    (tmp_path/'selection.json').write_text(json.dumps(journal))
    assert selected_qwen(tmp_path) == tmp_path/'cycle_01'
    (tmp_path/'cycle_01/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        selected_qwen(tmp_path)


def test_padding_cannot_affect_record_prediction():
    torch.set_num_threads(2)
    model = IntervalClassifier(2, 26).eval()
    wave, feat = torch.randn(1, 2, 226), torch.randn(1, 2, 26)
    with torch.no_grad():
        a = model(wave, feat, torch.ones(1, 2, dtype=torch.bool))
        b = model(torch.cat([wave, torch.randn(1, 3, 226)*100], 1),
                  torch.cat([feat, torch.randn(1, 3, 26)*100], 1),
                  torch.tensor([[True, True, False, False, False]]))
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match='valid beat'):
        model(wave, feat, torch.zeros(1, 2, dtype=torch.bool))


def test_unknown_labels_and_train_are_excluded_from_holdout_metrics():
    frame = pd.DataFrame(dict(split=['train', 'test', 'test', 'external'], source=['A', 'A', 'A', 'B']))
    y = np.array([[1], [0], [-1], [1]])
    p = np.array([[.9], [.2], [.9], [.8]])
    result = metrics_by_cohort(frame, y, p, ['AF'], [.5])
    assert result['all_labeled_descriptive']['records'] == 3
    assert result['heldout_all_sources']['records'] == 2
    assert result['test']['records'] == 1
    assert result['heldout_all_sources']['macro_auroc'] == 1


def test_missing_features_have_explicit_mask():
    x = scale_features(np.array([[np.nan, np.inf, 7.]]), np.array([1., 2., 3.]), np.ones(3))
    np.testing.assert_array_equal(x, [[0, 0, 4, 1, 1, 0]])


def test_training_writes_reloadable_checkpoint_and_unknown_predictions(tmp_path):
    root = tmp_path/'inputs'
    (root/'unet').mkdir(parents=True)
    (root/'cache_identity.json').write_text('{}')
    rows = []
    rng = np.random.default_rng(4)
    for i in range(7):
        key = str(i)
        np.savez(root/'unet'/f'{key}.npz', wave=rng.normal(size=(2, 226)).astype('float32'),
                 interval=rng.normal(size=(2, 13)).astype('float32'))
        rows.append(dict(key=key, record_id=key, source='fake', AF=i % 2 if i < 6 else -1,
                         split='train' if i < 2 else 'valid' if i < 4 else 'test'))
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'manifest.csv', index=False)
    cfg = ClassificationConfig(output=str(tmp_path/'run'), device='cpu', epochs=1, batch_size=2, max_beats=2)
    result = train_branch(cfg, root, frame, 'waveform', ['AF'])
    assert result['test']['records'] == 2
    saved = torch.load(tmp_path/'run/waveform/best.pt', weights_only=True)
    model = IntervalClassifier(1, 26, False)
    model.load_state_dict(saved['state_dict'])
    assert len(saved['thresholds']) == 1
    with np.load(tmp_path/'run/waveform/predictions.npz') as z:
        assert len(z['probability']) == 7
        assert z['labels'][-1, 0] == -1
    assert train_branch(cfg, root, frame, 'waveform', ['AF']) == result


def test_preparation_pairs_crops_and_resumes_without_loading_models(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import ecg_project.training.interval_classification as module
    from ecg_project.models import segmentation, qwen_delineator
    from ecg_project.training import beats
    qwen = tmp_path/'qwen'
    qwen.mkdir(); (qwen/'best.pt').write_bytes(b'qwen')
    unet = tmp_path/'unet.pt'; unet.write_bytes(b'unet')
    catalog = tmp_path/'catalog.csv'
    row = dict(path='fake.hea', source='fake', record_id='1', patient_id='p1', readable=True,
               split='train', **{c: 0 for c in module.TARGETS})
    pd.DataFrame([row]).to_csv(catalog, index=False)
    mit = tmp_path/'data/mit-bih'
    mit.mkdir(parents=True)
    (mit/'100.csv').write_text('fixture')
    monkeypatch.setattr(module, 'build_record_manifest', lambda _: pd.DataFrame([row]))
    monkeypatch.setattr(module, 'source_hashes', lambda *args: {'fake': 'hash'})
    monkeypatch.setattr(module, 'load_record', lambda _: SimpleNamespace(signal=np.arange(1000.)[:, None], fs=250., leads=['II']))
    monkeypatch.setattr(beats, 'load_mit', lambda _: SimpleNamespace(signal=(np.arange(1000.)+1000)[:, None], fs=360., leads=['MLII']))
    monkeypatch.setattr(module, 'preprocess', lambda x, fs: x)
    monkeypatch.setattr(module, 'rpeaks', lambda x, fs: (np.array([250, 500]), None))
    monkeypatch.setattr(module, 'interval_features', lambda *args: (
        np.ones((2, 226), np.float32), np.ones((2, 13), np.float32), np.array([0, 1])))
    calls = []
    class FakePredictor:
        def __init__(self, *args):
            calls.append('loaded')
        def predict(self, x, fs):
            return [[]]
    monkeypatch.setattr(segmentation, 'Predictor', FakePredictor)
    monkeypatch.setattr(qwen_delineator, 'Predictor', FakePredictor)
    cfg = ClassificationConfig(catalog=str(catalog), output=str(tmp_path/'output'),
        unet_checkpoint=str(unet), device='cpu', evaluate_manual=False, data_root=str(tmp_path/'data'))
    root, frame = module.prepare(cfg, qwen)
    assert len(frame) == 2 and frame.beats.eq(2).all()
    assert frame.loc[frame.source == 'MIT', 'split'].iloc[0] == 'inference'
    assert frame.loc[frame.source == 'MIT', list(module.TARGETS)].eq(-1).all().all()
    assert len(calls) == 2
    report = json.loads((root/'interval_agreement.json').read_text())
    assert report['by_source']['fake']['mean_absolute_difference'] == [0.] * 13
    module.prepare(cfg, qwen)
    assert len(calls) == 2
