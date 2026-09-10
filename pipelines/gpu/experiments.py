"""Reproducible v2 profiles, with optional cluster GPU pseudo-cache preparation."""
from dataclasses import replace,fields
from pathlib import Path
import json
from ecg_project.training.delineation_v2 import DelineationConfig


def require_cache(root,command,files=('provenance.json',)):
    missing=[str(Path(root)/f) for f in files if not (Path(root)/f).is_file()]
    if missing:raise FileNotFoundError('Missing cache: '+', '.join(missing)+'\nPreparation: '+command)


def require_train_sources(root,expected):
    import pandas as pd
    from ecg_project.data.policy import source_name
    root=Path(root)
    rows=pd.read_csv(root/'manifest.csv').fillna('').to_dict('records') if (root/'manifest.csv').exists() else json.loads((root/'manifest.json').read_text())
    actual={source_name(r.get('source_key',r['source'])) for r in rows if r['split']=='train'}
    if actual!=set(expected):raise ValueError(f'Cache train-source/profile mismatch at {root}: expected {sorted(expected)}, got {sorted(actual)}. Prepare the matching CPU cache in a new output directory.')


def qwen_experiment(name='B',size='1.7B',vram_gb=40):
    if name not in 'ABCDE' or len(name)!=1:raise ValueError('Qwen experiment must be A, B, C, D or E')
    if size not in ('1.7B','4B'):raise ValueError('Supported comparison sizes: 1.7B / 4B')
    batch=32 if vram_gb>60 else 16
    return DelineationConfig(architecture='qwen',
        input_root='artifacts/qwen_ludb_inputs_v2' if name=='A' else 'artifacts/qwen_delineation_inputs_v2',
        output=f'artifacts/cluster/qwen_{size.lower()}_{name}',final_checkpoint='',
        model_root=f'artifacts/qwen3_{size.lower()}',batch_size=batch,accumulation=64//batch,
        consistency_weight=0.,qt_weight=1.,epochs=100,patience=9,
        pseudo_root=None if name in 'AB' else 'artifacts/qwen_pseudo_training2' if name=='C' else 'artifacts/qwen_pseudo_extended',
        lambda_kd=.5 if name=='E' else 0.)


def assert_qwen_inputs(cfg):
    datasets='LUDB' if 'qwen_ludb_' in cfg.input_root else 'LUDB QTDB'
    require_cache(cfg.input_root,f'python -m pipelines.cpu.run prepare-qwen --datasets {datasets} --validation-datasets LUDB QTDB --output {cfg.input_root} --workers 4',
                  ('provenance.json','manifest.json','train_x.npy','train_y.npy','valid_x.npy','valid_y.npy'))
    require_train_sources(cfg.input_root,datasets.split())
    manual=json.loads((Path(cfg.input_root)/'manifest.json').read_text())
    if {r['source'] for r in manual if r['split']=='valid'}!={'LUDB','QTDB'}:raise ValueError('A–E require the same LUDB + manual QTDB validation protocol')
    if cfg.pseudo_root:
        sources='CPSC_EXTRA' if cfg.pseudo_root.endswith('training2') else 'CPSC_EXTRA PTBXL CPSC CHAPMAN'
        require_cache(cfg.pseudo_root,f'In Colab enable PREPARE_QWEN_PSEUDO_GPU=True before RUN_QWEN; sources={sources}, output={cfg.pseudo_root}',
                      ('provenance.json','manifest.json','protected_registry.json'))
        require_train_sources(cfg.pseudo_root,sources.split())


def run_qwen(cfg,run_smoke=False):
    from ecg_project.training.delineation_v2 import train
    assert_qwen_inputs(cfg)
    if run_smoke:
        train(replace(cfg,output=cfg.output+'_smoke',epochs=1,max_batches=2,valid_limit=8))
    return train(cfg)


def full_cached_e(run_root, output=None, pseudo_root=None, model_root=None):
    """Continue the selected E checkpoint using every existing pseudo window.

    No cache builders/downloads are called. Architecture and input paths come
    from the selected run so the warm-start adapter stays compatible.
    """
    from ecg_project.training.interval_classification import selected_qwen
    selected = selected_qwen(run_root)
    previous = json.loads((selected/'config.json').read_text())
    known = {f.name for f in fields(DelineationConfig)}
    cfg = DelineationConfig(**{k:v for k,v in previous.items() if k in known})
    if cfg.architecture != 'qwen' or cfg.lambda_kd <= 0:
        raise ValueError('Select an experiment E Qwen checkpoint with soft KD')
    cfg = replace(cfg, output=str(output or (str(run_root)+'_full_cache_E')),
                  warm_start=str(selected/'best.pt'), pseudo_root=pseudo_root or cfg.pseudo_root,
                  model_root=model_root or cfg.model_root, pseudo_full_pass=True,
                  pseudo_per_manual=4, epochs=20, patience=5, learning_rate=5e-5,
                  max_batches=0, valid_limit=0, final_checkpoint='')
    if Path(cfg.output).resolve() in (Path(run_root).resolve(), selected.resolve()):
        raise ValueError('Use a separate output for full-cache E')
    assert_qwen_inputs(cfg)
    return cfg
