"""Reproducible v2 profiles, with optional cluster GPU pseudo-cache preparation."""
from dataclasses import replace
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
