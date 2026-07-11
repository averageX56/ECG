"""
smoke_test.py
Самопроверка всех решений на синтетике (без реальных данных и GPU).

Генерирует несколько WFDB-записей (.hea+.mat) во временной папке и прогоняет:
таблицу записей, аналитику + оценку качества, делинеацию+диагноз (решение 1),
извлечение фич (extract_features), обучение CTN (решения 2 и 3, CPU, 1 эпоха).
Части, требующие тяжёлых зависимостей (torch, biosppy), пропускаются, если их нет.

  python smoke_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_PASS, _FAIL, _SKIP = [], [], []


class _Skip(Exception):
    pass


def check(name, fn):
    try:
        fn()
        print(f'  OK    {name}'); _PASS.append(name)
    except _Skip as e:
        print(f'  SKIP  {name}: {e}'); _SKIP.append(name)
    except Exception as e:
        print(f'  FAIL  {name}: {e}'); traceback.print_exc(); _FAIL.append(name)


LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DX = ['426783006', '427084000', '426627000', '164889003', '59118001,713427006']


def _write(root, rid, hr, dx, age, sex, noise=0.03):
    import scipy.io as sio
    try:
        import neurokit2 as nk
        sig = np.stack([nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=hr,
                                        noise=noise, random_state=i)[:5000] for i in range(12)])
    except Exception:
        t = np.linspace(0, 10, 5000)
        sig = np.stack([np.sin(2 * np.pi * hr / 60 * t) for _ in range(12)])
    root.mkdir(parents=True, exist_ok=True)
    sio.savemat(root / f'{rid}.mat', {'val': np.round(sig * 1000).astype(np.int16)})
    (root / f'{rid}.hea').write_text('\n'.join(
        [f'{rid} 12 500 5000'] + [f'{rid}.mat 16 1000/mV 16 0 0 0 0 {l}' for l in LEADS]
        + [f'#Age: {age}', f'#Sex: {sex}', f'#Dx: {dx}']) + '\n')


def main():
    import random
    random.seed(0)
    tmp = tempfile.mkdtemp(prefix='ecg_smoke_')
    root = Path(tmp) / 'data'
    for k in range(16):
        _write(root, f'REC{k:04d}', random.choice([50, 60, 75, 90, 110]),
               random.choice(DX), random.randint(30, 80), random.choice(['Male', 'Female']))
    print(f'Синтетика -> {root}')

    from ecg_data import build_record_table, read_record, make_split
    state = {}

    def t_data():
        df = build_record_table(root)
        assert len(df) == 16
        rec = read_record(df.iloc[0]['filepath'])
        assert rec.signal.shape[0] == 12 and rec.fs == 500
        state['df'] = df
    check('ecg_data: таблица записей + чтение', t_data)

    def t_analytics():
        import analytics as A
        df = state['df']
        assert A.class_distribution(df)['count'].sum() > 0
        assert A.dataset_summary(df)['n_records'] == 16
        q = A.quality_table(df.head(4))
        A.find_noisy_records(q, 2)
    check('analytics: классы + качество', t_analytics)

    def t_delineation():
        import delineation as D
        res = D.run_pipeline(state['df'].head(4))
        m = D.quality_metrics(res)
        assert (m['label'] == 'MICRO_AVERAGE').any()
    check('delineation: делинеация -> диагноз -> метрики', t_delineation)

    def t_extract():
        try:
            import biosppy  # noqa
        except Exception:
            raise _Skip('biosppy не установлен')
        import extract_features as E
        fdf = E.extract_dataset(state['df'].head(3), show_progress=False)
        assert fdf.shape[0] == 3 and fdf.shape[1] > 50
        out = os.path.join(tmp, 'features.csv')
        fdf.to_csv(out, index=False)
        feats, names = E.load_features(out)
        assert len(feats) == 3 and len(names) > 50
        state['features_csv'] = out
    check('extract_features: фичи -> CSV -> загрузка', t_extract)

    def t_train_dlonly():
        try:
            import torch  # noqa
        except Exception:
            raise _Skip('torch не установлен')
        import train_ctn as T
        ckpt = os.path.join(tmp, 'ckpt.pt')
        r = T.main(['--data-root', str(root), '--smoke', '--cache-dir', '',
                    '--checkpoint', ckpt, '--results-out', ''])
        assert r['pipeline'] == 2 and 'test_perclass_thr' in r['metrics']
        state['ckpt'] = ckpt
    check('train_ctn: решение 2 (DL-only, CPU, 1 эпоха)', t_train_dlonly)

    def t_eval_model():
        try:
            import torch  # noqa
        except Exception:
            raise _Skip('torch не установлен')
        if 'ckpt' not in state:
            raise _Skip('нет чекпоинта')
        import eval_model as EV
        tbl = EV.main(['--checkpoint', state['ckpt'], '--data-root', str(root), '--split', 'all',
                       '--window', '2500', '--nb-windows-eval', '2', '--cache-dir', ''])
        assert 'AUROC' in tbl.columns and (tbl['abbr'] == 'MACRO').any()
    check('eval_model: загрузка модели + метрики по классам', t_eval_model)

    def t_models():
        try:
            import torch  # noqa
        except Exception:
            raise _Skip('torch не установлен')
        import train_ctn as T
        for name in ('resnet1d', 'unet1d'):
            r = T.main(['--data-root', str(root), '--model', name, '--smoke',
                        '--cache-dir', '', '--checkpoint', '', '--results-out', ''])
            assert r['model'] == name and 'test_perclass_thr' in r['metrics']
    check('train_ctn: модели resnet1d + unet1d (CPU, 1 эпоха)', t_models)

    def t_fraction():
        try:
            import torch  # noqa
        except Exception:
            raise _Skip('torch не установлен')
        import train_ctn as T
        r = T.main(['--data-root', str(root), '--fraction', '0.5', '--smoke',
                    '--cache-dir', '', '--checkpoint', '', '--results-out', ''])
        assert r['fraction'] == 0.5
    check('train_ctn: обучение на части датасета (--fraction)', t_fraction)

    def t_feat_select():
        try:
            import biosppy  # noqa (нужен только для импорта extract_features)
        except Exception:
            raise _Skip('biosppy не установлен')
        import pandas as pd
        from extract_features import fit_selector, apply_selector
        from ecg_data import CLASSES
        rng = np.random.default_rng(0)
        n, d = 20, 60
        src = [f'f{i}' for i in range(d)]
        feats = {f'R{i}': rng.normal(size=d) for i in range(n)}
        Y = (rng.random((n, len(CLASSES))) > 0.8).astype(int)
        tdf = pd.DataFrame({'record_id': [f'R{i}' for i in range(n)]})
        for j, c in enumerate(CLASSES):
            tdf[c] = Y[:, j]
        for method in ('rf', 'variance', 'pca'):
            sel = fit_selector(feats, src, tdf, method=method, nb_feats=8)
            red = apply_selector(feats, sel)
            assert next(iter(red.values())).shape[0] == 8 and len(sel['names']) == 8, method
    check('feature select: rf/variance/pca -> top-N (fit+apply)', t_feat_select)

    def t_train_feats():
        try:
            import torch  # noqa
            import biosppy  # noqa
        except Exception:
            raise _Skip('torch/biosppy не установлены')
        if 'features_csv' not in state:
            raise _Skip('нет features.csv')
        import train_ctn as T
        r = T.main(['--data-root', str(root), '--features', state['features_csv'],
                    '--smoke', '--cache-dir', '', '--checkpoint', '', '--results-out', ''])
        assert r['pipeline'] == 3 and r['n_feats'] > 50
    check('train_ctn: решение 3 (+фичи, CPU, 1 эпоха)', t_train_feats)

    print('\n================ SMOKE SUMMARY ================')
    print(f'  passed : {len(_PASS)}')
    print(f'  skipped: {len(_SKIP)}  {_SKIP}')
    print(f'  failed : {len(_FAIL)}  {_FAIL}')
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 1 if _FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
