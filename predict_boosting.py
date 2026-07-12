"""
predict_boosting.py
Загрузка сохранённой бустинг-модели (train_boosting.py) и прогон на части
датасета -> вероятности по классам + метрики ПО КАЖДОМУ классу.

Аналог eval_model.py, но для решения 4 (бустинг на ручных фичах + интервальных
фичах neurokit). Ничего не обучает: берёт .joblib-модель и CSV ручных фич,
досчитывает интервальные фичи neurokit (как при обучении), выравнивает всё по
именам из модели, предсказывает, считает метрики и (опц.) сохраняет предсказания.

Примеры:
  python predict_boosting.py --model outputs/boosting/boosting_model.joblib \
      --features features.csv --data-root data
  # на 10% всех записей, с поклассовым порогом, с сохранением предсказаний
  python predict_boosting.py --model outputs/boosting/boosting_model.joblib \
      --features features.csv --split all --fraction 0.1 --tune \
      --out outputs/boosting/per_label_eval.csv --preds-out outputs/boosting/preds.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import joblib

from ecg_data import CLASSES, build_record_table, make_split
from train_boosting import build_feature_frame, predict_proba, per_label_metrics


def align_X(records_df, feat_df, feature_names):
    """X по фичам модели (в порядке feature_names; недостающие столбцы -> NaN), Y и ids."""
    ids = [r for r in records_df['record_id'].tolist() if r in feat_df.index]
    X = feat_df.reindex(index=ids, columns=feature_names).to_numpy(dtype=np.float64)
    Y = records_df.set_index('record_id').loc[ids][CLASSES].to_numpy(dtype=int)
    return X, Y, ids


def tune_thresholds_f1(probs, Y, grid=None):
    """Поклассовый порог, максимизирующий F1 (по этой же выборке)."""
    grid = np.arange(0.05, 0.95, 0.01) if grid is None else grid
    thr = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        y = Y[:, k]
        if y.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = probs[:, k] > t
            tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
            den = 2 * tp + fp + fn
            f1 = (2 * tp / den) if den else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Инференс бустинг-модели -> метрики по классам')
    p.add_argument('--model', required=True, help='boosting_model.joblib из train_boosting.py')
    p.add_argument('--features', required=True, help='CSV ручных фич (extract_features.py)')
    p.add_argument('--nk-cache', default='nk_features.csv',
                   help='Кэш интервальных фич neurokit. "" — считать без кэша')
    p.add_argument('--data-root', default='data', help='Датасет для меток/сплита (из .hea)')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train', 'all'])
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--tune', action='store_true', help='Поклассовый порог (max F1) вместо --threshold')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--val-fold', type=int, default=8)
    p.add_argument('--test-fold', type=int, default=9)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--out', default=None, help='CSV метрик по классам')
    p.add_argument('--preds-out', default=None, help='CSV предсказаний (record_id + вероятности классов)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bundle = joblib.load(args.model)
    models, base_rate = bundle['models'], np.asarray(bundle['base_rate'])
    fnames = bundle['feature_names']
    print(f"Модель: бустинг | отведения: {bundle.get('leads')} | перевзвешивание: {bundle.get('reweight')} "
          f"| фич: {len(fnames)}")

    df = build_record_table(args.data_root, limit=args.limit)
    if args.fraction < 1.0:
        df = df.sample(frac=args.fraction, random_state=42).reset_index(drop=True)
        print(f'Взята доля датасета: {args.fraction:.2f} -> {len(df)} записей')
    if args.split == 'all':
        sub = df
    else:
        tr, va, te = make_split(df, args.val_fold, args.test_fold)
        sub = {'train': tr, 'val': va, 'test': te}[args.split]
    print(f'Оценка на сплите "{args.split}": {len(sub)} записей')

    # ручные фичи + тот же отбор, что при обучении (hand_selector из модели)
    from extract_features import load_features, apply_selector
    hand_feats, hand_names = load_features(args.features)
    selector = bundle.get('hand_selector')
    if selector is not None:
        hand_feats = apply_selector(hand_feats, selector)
        hand_names = selector['names']
    # + интервальные фичи neurokit (как при обучении)
    nk_lead = int(bundle.get('nk_lead_index', 1))
    nk_needed = any(str(n).startswith('nk_') for n in fnames)
    if nk_needed:
        print('Досчитываю интервальные фичи neurokit...')
    feat_df, _ = build_feature_frame(hand_feats, hand_names, sub, (args.nk_cache or None),
                                     nk_lead, with_nk=nk_needed)

    X, Y, ids = align_X(sub, feat_df, fnames)
    if len(X) == 0:
        raise SystemExit('Нет записей с фичами в этой выборке — проверьте --features/--split/--fraction.')

    probs = predict_proba(models, base_rate, X)
    thresholds = tune_thresholds_f1(probs, Y) if args.tune else args.threshold
    table = per_label_metrics(probs, Y, thresholds)

    print(f"\n=== Метрики по классам (порог: {'поклассовый (max F1)' if args.tune else args.threshold}) ===")
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', lambda v: f'{v:.3f}' if v == v else 'NA'):
        print(table[['abbr', 'name', 'support', 'AUROC', 'AUPRC', 'precision', 'recall', 'f1']].to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f'\nМетрики по классам -> {args.out}')
    if args.preds_out:
        os.makedirs(os.path.dirname(args.preds_out) or '.', exist_ok=True)
        preds_df = pd.DataFrame(probs, columns=CLASSES)
        preds_df.insert(0, 'record_id', ids)
        preds_df.to_csv(args.preds_out, index=False)
        print(f'Предсказания (вероятности) -> {args.preds_out}')
    return table


if __name__ == '__main__':
    main()
