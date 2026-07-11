"""
eval_model.py
Загрузка сохранённой модели и прогон на части датасета -> метрики ПО КАЖДОМУ
классу отдельно (support, AUROC, AUPRC, precision, recall, F1).

Модель, отведения и статистики фич берутся из чекпоинта (сохранён train_ctn.py);
при необходимости переопределяются флагами. Ничего не обучает — только инференс.

Примеры:
  # метрики по классам на тесте
  python eval_model.py --checkpoint checkpoints/model.pt --data-root data
  # быстро на 10% всех записей, с поклассовым подбором порога
  python eval_model.py --checkpoint checkpoints/model.pt --data-root data \
      --split all --fraction 0.1 --tune --out outputs/per_label.csv
  # модель с ручными фичами (пайплайн 3): укажите тот же CSV
  python eval_model.py --checkpoint checkpoints/model.pt --data-root data --features features.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ecg_data import (CLASSES, class_names, class_abbrs, build_record_table,
                      make_split, ECGDataset)
from model import build_model
from train_ctn import get_probs, make_wide, tune_thresholds_f1, get_device  # переиспользуем


def _make_loader(dataset, batch_size):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def per_label_metrics(probs, lbls, thresholds) -> pd.DataFrame:
    """Метрики по каждому из 27 классов при заданных порогах (скаляр или вектор)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    names, abbrs = class_names(), class_abbrs()
    thr = np.asarray(thresholds) if np.ndim(thresholds) else np.full(len(CLASSES), thresholds)
    rows = []
    for k, c in enumerate(CLASSES):
        y, p = lbls[:, k].astype(int), probs[:, k]
        sup = int(y.sum())
        auroc = float(roc_auc_score(y, p)) if 0 < sup < len(y) else np.nan
        auprc = float(average_precision_score(y, p)) if sup > 0 else np.nan
        pred = p > thr[k]
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else (0.0 if sup else np.nan)
        rows.append({'code': c, 'abbr': abbrs.get(c, c), 'name': names.get(c, c),
                     'support': sup, 'threshold': round(float(thr[k]), 3),
                     'AUROC': auroc, 'AUPRC': auprc, 'precision': prec, 'recall': rec, 'f1': f1})
    df = pd.DataFrame(rows).sort_values('support', ascending=False).reset_index(drop=True)
    macro = {'code': 'MACRO', 'abbr': 'MACRO', 'name': 'macro-average', 'support': int(df['support'].sum()),
             'threshold': np.nan}
    for col in ('AUROC', 'AUPRC', 'precision', 'recall', 'f1'):
        macro[col] = float(np.nanmean(df[col].values))
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Загрузка модели -> метрики по каждому классу')
    p.add_argument('--checkpoint', required=True, help='Чекпоинт из train_ctn.py')
    p.add_argument('--data-root', default='data')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train', 'all'],
                   help='На каком сплите считать метрики (all = все записи выборки)')
    p.add_argument('--fraction', type=float, default=1.0, help='Доля датасета (для быстрого прогона)')
    p.add_argument('--features', default=None, help='CSV фич (если модель обучалась с фичами)')
    p.add_argument('--model', default=None, choices=[None, 'ctn', 'resnet1d', 'unet1d'],
                   help='Переопределить модель (иначе из чекпоинта)')
    p.add_argument('--leads', default=None, choices=[None, 'all', 'lead1'])
    p.add_argument('--window', type=int, default=7500)
    p.add_argument('--nb-windows-eval', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--tune', action='store_true', help='Поклассовый порог (макс. F1) вместо 0.5')
    p.add_argument('--threshold', type=float, default=0.5, help='Порог, если не --tune')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--val-fold', type=int, default=8)
    p.add_argument('--test-fold', type=int, default=9)
    p.add_argument('--cache-dir', default='cache/signals', help='Кэш сигнала ("" — отключить)')
    p.add_argument('--out', default=None, help='Куда сохранить CSV метрик по классам')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)

    # weights_only=False: чекпоинт свой, содержит метаданные (не только веса)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = args.model or ckpt.get('model', 'ctn')
    leads = args.leads or ckpt.get('leads', 'all')
    n_feats = int(ckpt.get('n_feats', 0))
    age_mean = float(ckpt.get('age_mean', 60.0))
    age_std = float(ckpt.get('age_std', 15.0))
    print(f'Модель: {model_name} | отведения: {leads} | ручных фич: {n_feats}')

    model = build_model(model_name, CLASSES, in_channels=(12 if leads == 'all' else 1),
                        nb_feats=n_feats, nb_demo=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # выбор записей
    df = build_record_table(args.data_root)
    if args.fraction < 1.0:
        df = df.sample(frac=args.fraction, random_state=42).reset_index(drop=True)
        print(f'Взята доля датасета: {args.fraction:.2f} -> {len(df)} записей')
    if args.split == 'all':
        sub = df
    else:
        train_df, val_df, test_df = make_split(df, args.val_fold, args.test_fold)
        sub = {'train': train_df, 'val': val_df, 'test': test_df}[args.split]
    print(f'Оценка на сплите "{args.split}": {len(sub)} записей')
    if len(sub) == 0:
        raise SystemExit('Пустая выборка — увеличьте --fraction или смените --split.')

    # ручные фичи (пайплайн 3)
    features, feat_means, feat_stds = None, None, None
    if n_feats > 0:
        if not args.features:
            raise SystemExit('Модель обучена с фичами — укажите --features features.csv')
        from extract_features import load_features, apply_selector
        features, _ = load_features(args.features)
        features = apply_selector(features, ckpt.get('selector'))   # тот же отбор, что при обучении
        if ckpt.get('feat_means') is not None:
            feat_means = np.asarray(ckpt['feat_means'], dtype=np.float64)
            feat_stds = np.asarray(ckpt['feat_stds'], dtype=np.float64)
        else:                                   # старый чекпоинт без статистик -> считаем по выборке
            X = np.vstack([features[i] for i in sub['record_id'] if i in features]).astype(np.float64)
            X[np.isinf(X)] = np.nan
            feat_means = np.nan_to_num(np.nanmean(X, axis=0))
            feat_stds = np.nanstd(X, axis=0); feat_stds[~np.isfinite(feat_stds) | (feat_stds == 0)] = 1.0

    ds = ECGDataset(sub, args.window, args.nb_windows_eval, leads=leads, features=features,
                    feat_means=feat_means, feat_stds=feat_stds,
                    cache_dir=(args.cache_dir or None), augment=False)
    loader = _make_loader(ds, args.batch_size)

    probs, lbls = get_probs(model, loader, age_mean, age_std, device, desc='инференс')
    thresholds = tune_thresholds_f1(probs, lbls) if args.tune else args.threshold
    table = per_label_metrics(probs, lbls, thresholds)

    print(f"\n=== Метрики по классам (порог: {'поклассовый (max F1)' if args.tune else args.threshold}) ===")
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', lambda v: f'{v:.3f}' if v == v else 'NA'):
        print(table[['abbr', 'name', 'support', 'AUROC', 'AUPRC', 'precision', 'recall', 'f1']].to_string(index=False))

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f'\nМетрики по классам -> {args.out}')
    return table


if __name__ == '__main__':
    main()
