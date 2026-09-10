import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import train_classifiers as console
from ecg_project.training.interval_classification import ClassificationConfig, early_stop_state
from ecg_project.training.trust_intervals import TrustConfig, METHOD
from ecg_project.data.catalog import file_hash
from ecg_project.data.cache import atomic_json
from ecg_project.data.classifier_labels import LABEL_VERSION, TARGETS


def test_early_stop_reconstructs_small_gains_and_cumulative_improvement():
    history = [dict(valid_macro_auroc=s) for s in [.8, .8004, .8008, .8012, .8013]]
    reference, stale = early_stop_state(history, .001)
    assert reference == .8012 and stale == 1
    assert early_stop_state(history, 0.)[1] == 0
    assert early_stop_state([dict(valid_macro_auroc=s) for s in [.8, .79, .78]], .001)[1] == 2


def qwen_args(tmp_path):
    return SimpleNamespace(manual_root='manual', pseudo_root='pseudo', qwen_model_root=str(tmp_path/'base'),
        qwen_output=str(tmp_path/'fresh'), qwen_batch_size=16, qwen_accumulation=4,
        warmup_epochs=2, qwen_epochs=20, qwen_patience=5)


def test_fresh_17b_warmup_then_full_cache_without_4b_checkpoint(tmp_path, monkeypatch):
    from pipelines.gpu import experiments
    args = qwen_args(tmp_path)
    atomic_json(tmp_path/'base/provenance.json', {'model_id': 'Qwen/Qwen3-1.7B'})
    checks, runs = [], []
    monkeypatch.setattr(experiments, 'assert_qwen_inputs', checks.append)
    monkeypatch.setattr(console, 'run_qwen_stage', lambda cfg: runs.append(cfg) or cfg)
    full = console.train_fresh_qwen(args)
    warm = runs[0]
    assert len(runs) == len(checks) == 2
    assert warm.warm_start is None and warm.pseudo_root is None and warm.epochs == 2
    assert warm.quantization == full.quantization == 'nf4'
    assert full.warm_start == str(tmp_path/'fresh_manual_warmup/best.pt')
    assert full.pseudo_full_pass and full.lambda_kd > 0 and full.pseudo_root == 'pseudo'
    assert full.max_batches == full.valid_limit == 0


def test_reject_4b_base_before_training(tmp_path):
    args = qwen_args(tmp_path)
    _, cfg = console.qwen_configs(args)
    atomic_json(tmp_path/'base/provenance.json', {'model_id': 'Qwen/Qwen3-4B'})
    with pytest.raises(ValueError, match='1.7B'):
        console.run_qwen_stage(cfg)


def test_job_selection_reuses_existing_unet_and_excludes_old_qwen(tmp_path):
    base, trust, masks = [tmp_path/name for name in ('base', 'trust', 'masks')]
    for directory in (base/'inputs', trust/'inputs', trust/'point_inputs', masks/'inputs', masks/'point_inputs'):
        identity = {} if directory == base/'inputs' else dict(method=METHOD, config=asdict(TrustConfig()))
        if directory == masks/'inputs':
            identity['confidence_masks'] = 'softmax_250hz_background_P_QRS_T_v1'
        atomic_json(directory/'cache_identity.json', identity)
        (directory/'manifest.csv').write_text('fixture')
        for variant in ('unet', 'qwen'):
            (directory/variant).mkdir()
    jobs, missing = console.make_jobs(base, trust, masks,
        ['waveform', 'intervals', 'point', 'trust', 'masks'], variants=('unet',))
    assert not missing and len(jobs) == 5
    assert all(j.variant != 'qwen' for j in jobs)
    assert next(j for j in jobs if j.stage == 'point').root == trust/'point_inputs'
    assert next(j for j in jobs if j.stage == 'masks').feature_spec['confidence_masks']


def test_fresh_qwen_refresh_is_one_pass_and_resumable(tmp_path, monkeypatch):
    from ecg_project.training import classifier_refresh as refresh
    from ecg_project.training import interval_classification as base
    from ecg_project.models import qwen_delineator
    original, new_qwen = tmp_path/'original', tmp_path/'qwen17'
    new_qwen.mkdir()
    (new_qwen/'best.pt').write_bytes(b'new-17-adapter')
    (new_qwen/'config.json').write_text('{}')
    unet = tmp_path/'unet.pt'
    unet.write_bytes(b'unchanged-unet')
    raw = tmp_path/'raw.hea'
    raw.write_bytes(b'unchanged-source')
    identity = dict(qwen='old-4b-hash', unet=file_hash(unet))
    atomic_json(original/'cache_identity.json', identity)
    (original/'qwen').mkdir()
    frame = pd.DataFrame([dict(key='record', path=str(raw), source='CPSC', beats=1)])
    frame.to_csv(original/'manifest.csv', index=False)
    path = original/'qwen/record.npz'
    np.savez(path, wave=np.ones((1, 226), np.float32), peaks=np.array([250]),
             signal_hash='signal', lead='II')
    atomic_json(path.with_suffix('.json'), dict(sha256=file_hash(path),
        identity=dict(models=identity, source_sha256={str(raw): file_hash(raw)})))
    old_digest = file_hash(path)
    loads, predictions = [], []
    class Predictor:
        def __init__(self, qwen, *args):
            assert qwen == new_qwen
            loads.append(qwen)
        def predict(self, *args, **kwargs):
            predictions.append(1)
            return [[]], np.full((1000, 1, 4), .25, np.float32)
    monkeypatch.setattr(qwen_delineator, 'Predictor', Predictor)
    monkeypatch.setattr(refresh, '_load_signal', lambda row: (np.sin(np.arange(1000)/20), 250))
    monkeypatch.setattr(base, 'rpeaks', lambda *args: (np.array([250]), None))
    monkeypatch.setattr(base, 'interval_features', lambda *args: (
        np.ones((1, 226), np.float32), np.ones((1, 13), np.float32), np.array([0])))
    cfg = ClassificationConfig(qwen_run=str(new_qwen), unet_checkpoint=str(unet), device='cpu')
    output = tmp_path/'new_features'
    root, bands = refresh.refresh_qwen(original, output, cfg, TrustConfig())
    assert len(loads) == len(predictions) == 1 and file_hash(path) == old_digest
    assert json.loads((root/'cache_identity.json').read_text())['qwen'] == file_hash(new_qwen/'best.pt')
    with np.load(bands/'inputs/qwen/record.npz') as z:
        assert z['confidence_mask'].shape == (1, 4, 226)
    monkeypatch.setattr(refresh, '_load_signal', lambda row: pytest.fail('Resume must not reread raw ECG'))
    refresh.refresh_qwen(original, output, cfg, TrustConfig())
    assert len(loads) == len(predictions) == 1


def test_console_reuses_completed_classifier_without_model_or_shard_reads(tmp_path, monkeypatch):
    base = tmp_path/'old_classifier'
    root = base/'inputs'
    atomic_json(root/'cache_identity.json', {})
    (root/'unet').mkdir()
    rows = [dict(key=f'key{i}', record_id=str(i), source='CPSC', split='train' if i < 42 else 'valid',
                 labels='164889003' if i % 2 else '426783006', beats=1, signal_hash=f'signal{i}') for i in range(44)]
    pd.DataFrame(rows).to_csv(root/'manifest.csv', index=False)
    stored = ClassificationConfig(output=str(base), device='cpu', batch_size=3)
    cfg_path = tmp_path/'config.json'
    cfg_path.write_text(json.dumps(asdict(ClassificationConfig(output=str(base), device='cpu'))))
    out = base/LABEL_VERSION/'waveform'
    atomic_json(out/'cache_identity.json', dict(config=asdict(stored), manifest=file_hash(root/'manifest.csv'),
        inputs={}, variant='waveform', classes=['AF'], label_version=LABEL_VERSION, diagnosis_codes={'AF': TARGETS['AF']}))
    atomic_json(out/'metrics.json', {'valid': {'macro_auroc': .9}})
    monkeypatch.setattr(console, 'validate_shards', lambda *args: pytest.fail('Completed run must not reread all shards'))
    monkeypatch.setattr(console, 'train_fresh_qwen', lambda *args: pytest.fail('Cache-only must not train Qwen'))
    report = tmp_path/'report.json'
    result = console.main(['--config', str(cfg_path), '--cache-only', '--stages', 'waveform', '--report', str(report)])
    assert result == 0
    assert json.loads(report.read_text())['jobs'][0]['action'] == 'completed'
