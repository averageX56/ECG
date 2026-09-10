import json
from dataclasses import replace

import torch
import pytest

from ecg_project.training.delineation_v2 import epoch_schedule


def test_full_schedule_covers_every_pseudo_batch_without_wrap():
    assert epoch_schedule(2, 11, 4, True) == [4, 4, 3]
    assert epoch_schedule(5, 2, 4, True) == [2, 0, 0, 0, 0]
    assert epoch_schedule(2, 11, 4, False) == [4, 4]
    assert epoch_schedule(2, 11, 4, True, max_batches=1) == [4]
    with pytest.raises(ValueError):
        epoch_schedule(2, 0, 4, True)


def test_full_training_visits_all_pseudo_windows_and_resumes(tmp_path, monkeypatch):
    from test_v2 import manual_cache, pseudo_cache
    from ecg_project.training import delineation_v2 as trainer
    from ecg_project.training import distillation
    original = distillation.PseudoDataset
    visited = []
    class Dataset(original):
        def __len__(self):
            return 11
        def __getitem__(self, index):
            visited.append(index)
            return super().__getitem__(0)
    monkeypatch.setattr(distillation, 'PseudoDataset', Dataset)
    monkeypatch.setattr(trainer, 'Delineator', lambda: torch.nn.Conv1d(1, 4, 1))
    cfg = trainer.DelineationConfig(input_root=str(manual_cache(tmp_path/'manual')),
        pseudo_root=str(pseudo_cache(tmp_path/'pseudo')), output=str(tmp_path/'out'),
        final_checkpoint='', device='cpu', epochs=1, batch_size=2, accumulation=2,
        lambda_kd=.5, consistency_weight=0., pseudo_full_pass=True, pseudo_per_manual=4)
    trainer.train(cfg)
    assert sorted(visited) == list(range(11))
    row = json.loads((tmp_path/'out/history.json').read_text())[0]
    assert row['pseudo_examples_seen'] == 11 and row['manual_examples_seen'] == 4
    visited.clear()
    trainer.train(replace(cfg, epochs=2))
    assert sorted(visited) == list(range(11))


def test_full_e_reuses_selected_config_without_preparation(tmp_path, monkeypatch):
    from dataclasses import asdict
    from pipelines.gpu import experiments
    cfg = experiments.qwen_experiment('E', '4B')
    (tmp_path/'best.pt').write_bytes(b'checkpoint')
    (tmp_path/'config.json').write_text(json.dumps(asdict(cfg)))
    checked = []
    monkeypatch.setattr(experiments, 'assert_qwen_inputs', checked.append)
    full = experiments.full_cached_e(tmp_path)
    assert full.pseudo_full_pass and full.lambda_kd == .5
    assert full.input_root == cfg.input_root and full.pseudo_root == cfg.pseudo_root
    assert full.model_root == cfg.model_root and full.warm_start == str(tmp_path/'best.pt')
    assert full.max_batches == full.valid_limit == 0 and checked == [full]
