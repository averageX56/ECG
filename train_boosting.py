"""
train_boosting.py
Решение 4: градиентный бустинг деревьев на РУЧНЫХ ФИЧАХ + ИНТЕРВАЛЬНЫХ ФИЧАХ
NEUROKIT.

Обучается бустинг (HistGradientBoostingClassifier, по одной модели на каждый из
27 классов) на объединённом наборе признаков:
  - ручные фичи Prna из CSV (extract_features.py), и
  - интервальные фичи neurokit (rr/qrs/qt/pq + девиации, qtc, n_beats), которые
    считаются делинеацией (delineation.interval_feature_table) и кэшируются.
Модель сохраняется, метрики по каждому классу пишутся на диск сразу.

Отбор лучших фич (--nb-feats/--feat-select) применяется ТОЛЬКО к ручным фичам
(rf/variance/mutual_info/pca, важность считается по train); интервальные фичи
neurokit сохраняются всегда и добавляются после отбора.

Гиперпараметры бустинга заданы ЯВНО в этом файле (BOOSTING_PARAMS), а не флагами.
CLI-флаги: --leads (число отведений), --features (CSV ручных фич), --out (папка),
--reweight/--no-reweight (перевзвешивать ли редкие классы), --nb-feats/--feat-select
(отбор ручных фич), --nk-cache (кэш интервальных фич neurokit). Метки/фолды — из .hea.

Пример:
  python extract_features.py --data-root data --out features.csv
  python train_boosting.py --features features.csv --leads all --out outputs/boosting
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from ecg_data import (CLASSES, NORMAL_CLASS, class_names, class_abbrs,
                      build_record_table, make_split)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_HERE, 'eval', 'weights.csv')

# ══════════════════════════════════════════════════════════════════════════
# ГИПЕРПАРАМЕТРЫ БУСТИНГА — правьте здесь (НЕ через CLI)
# ══════════════════════════════════════════════════════════════════════════
BOOSTING_PARAMS = dict(
    max_iter=400,               # число деревьев (бустинг-итераций)
    learning_rate=0.05,
    max_leaf_nodes=31,
    max_depth=None,             # None = ограничение только по max_leaf_nodes
    min_samples_leaf=20,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=True,        # ранняя остановка по внутренней валидации
    validation_fraction=0.1,
    n_iter_no_change=25,
    random_state=42,
)

# Порог бинаризации вероятностей для precision/recall/F1
DECISION_THRESHOLD = 0.5

# Отведение для интервальных фич neurokit (0 = I, 1 = II, ...)
NEUROKIT_LEAD_INDEX = 1


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------
def load_feature_csv(path):
    """CSV фич -> (DataFrame, индексированный record_id; список имён фич)."""
    df = pd.read_csv(path)
    df['record_id'] = df['record_id'].astype(str)
    names = [c for c in df.columns if c != 'record_id']
    df[names] = df[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df.set_index('record_id'), names


def load_or_compute_nk(records_df, cache_path, lead_index=NEUROKIT_LEAD_INDEX):
    """Интервальные фичи neurokit для записей records_df -> DataFrame (indexed).
    Дорого (делинеация на запись) — поэтому кэшируется в cache_path; недостающие
    записи досчитываются и дописываются."""
    from delineation import interval_feature_table
    existing = None
    if cache_path and os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        existing['record_id'] = existing['record_id'].astype(str)
    have = set(existing['record_id']) if existing is not None else set()
    need = records_df[~records_df['record_id'].isin(have)]
    if len(need):
        new = interval_feature_table(need, lead_index=lead_index)
        allnk = pd.concat([existing, new], ignore_index=True) if existing is not None else new
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            allnk.to_csv(cache_path, index=False)
    else:
        allnk = existing
    return allnk.set_index('record_id')


def build_feature_frame(hand_feats: dict, hand_names, records_df, nk_cache,
                        lead_index=NEUROKIT_LEAD_INDEX, with_nk=True):
    """Собирает матрицу признаков: (уже отобранные) ручные фичи + neurokit.

    hand_feats : {record_id -> вектор} ручных фич (после отбора, если он есть).
    Возвращает (DataFrame, индексированный record_id; список имён = ручные + nk).
    """
    hand_df = pd.DataFrame.from_dict(hand_feats, orient='index', columns=list(hand_names))
    if not with_nk:
        return hand_df, list(hand_names)
    nk_df = load_or_compute_nk(records_df, nk_cache, lead_index)
    return hand_df.join(nk_df, how='outer'), list(hand_names) + list(nk_df.columns)


def build_xy(records_df, feat_df, names):
    """Матрицы X (фичи, NaN сохраняются — HistGBM их умеет) и Y (27 меток)."""
    ids = [r for r in records_df['record_id'].tolist() if r in feat_df.index]
    X = feat_df.loc[ids, names].to_numpy(dtype=np.float64)
    Y = records_df.set_index('record_id').loc[ids][CLASSES].to_numpy(dtype=int)
    return X, Y, ids


# --------------------------------------------------------------------------
# Бустинг: по одной модели на класс
# --------------------------------------------------------------------------
def train_boosting(X, Y, reweight=True, params=None):
    """27 бинарных бустинг-моделей (или None для классов без вариации).
    reweight=True -> class_weight='balanced' (перевзвешивание редких классов)."""
    params = params or BOOSTING_PARAMS
    models, base_rate = [], []
    try:
        from tqdm.auto import tqdm
        rng = tqdm(range(len(CLASSES)), desc='Бустинг по классам')
    except Exception:
        rng = range(len(CLASSES))
    for k in rng:
        y = Y[:, k]
        pos = int(y.sum())
        base_rate.append(pos / len(y) if len(y) else 0.0)
        if pos < 2 or pos > len(y) - 2:        # почти нет вариации -> без модели
            models.append(None)
            continue
        try:
            clf = HistGradientBoostingClassifier(
                class_weight=('balanced' if reweight else None), **params)
            clf.fit(X, y)
            models.append(clf)
        except Exception:                       # напр. слишком мало для early stopping
            models.append(None)
    return models, np.asarray(base_rate)


def predict_proba(models, base_rate, X):
    """Вероятности [N, 27]; для классов без модели — базовая частота из train."""
    P = np.zeros((X.shape[0], len(CLASSES)))
    for k, clf in enumerate(models):
        P[:, k] = base_rate[k] if clf is None else clf.predict_proba(X)[:, 1]
    return P


# --------------------------------------------------------------------------
# Метрики по каждому классу
# --------------------------------------------------------------------------
def per_label_metrics(probs, Y, thresholds=DECISION_THRESHOLD) -> pd.DataFrame:
    """Метрики по каждому классу. thresholds — скаляр или вектор длины 27."""
    names, abbrs = class_names(), class_abbrs()
    thr = np.asarray(thresholds) if np.ndim(thresholds) else np.full(len(CLASSES), thresholds)
    rows = []
    for k, c in enumerate(CLASSES):
        y, p = Y[:, k], probs[:, k]
        sup = int(y.sum())
        auroc = float(roc_auc_score(y, p)) if 0 < sup < len(y) else np.nan
        auprc = float(average_precision_score(y, p)) if sup > 0 else np.nan
        pred = p > thr[k]
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else (0.0 if sup else np.nan)
        rows.append({'code': c, 'abbr': abbrs.get(c, c), 'name': names.get(c, c), 'support': sup,
                     'threshold': round(float(thr[k]), 3),
                     'AUROC': auroc, 'AUPRC': auprc, 'precision': prec, 'recall': rec, 'f1': f1})
    df = pd.DataFrame(rows).sort_values('support', ascending=False).reset_index(drop=True)
    macro = {'code': 'MACRO', 'abbr': 'MACRO', 'name': 'macro-average', 'support': int(df['support'].sum())}
    for col in ('AUROC', 'AUPRC', 'precision', 'recall', 'f1'):
        macro[col] = float(np.nanmean(df[col].values))
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Решение 4: бустинг деревьев на ручных фичах')
    p.add_argument('--features', required=True, help='CSV ручных фич (extract_features.py)')
    p.add_argument('--nb-feats', type=int, default=0,
                   help='Отобрать топ-N ручных фич (0 = все). neurokit-фичи сохраняются всегда.')
    p.add_argument('--feat-select', default='rf', choices=['rf', 'variance', 'mutual_info', 'pca'],
                   help='Метод отбора ручных фич при --nb-feats')
    p.add_argument('--leads', default='all', choices=['all', 'lead1'],
                   help='Число отведений (пишется в метаданные; фичи берутся из --features)')
    p.add_argument('--out', default='outputs/boosting', help='Папка вывода (модель + метрики)')
    p.add_argument('--reweight', dest='reweight', action='store_true', default=True,
                   help='Перевзвешивать редкие классы (по умолчанию да)')
    p.add_argument('--no-reweight', dest='reweight', action='store_false',
                   help='Не перевзвешивать редкие классы')
    p.add_argument('--nk-cache', default='nk_features.csv',
                   help='Кэш интервальных фич neurokit (считаются один раз). "" — не кэшировать')
    p.add_argument('--data-root', default='data', help='Датасет для меток/фолдов (из .hea)')
    p.add_argument('--val-fold', type=int, default=8)
    p.add_argument('--test-fold', type=int, default=9)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    df = build_record_table(args.data_root, limit=args.limit)
    train_df, val_df, test_df = make_split(df, args.val_fold, args.test_fold)
    if len(val_df) == 0 or len(test_df) == 0:      # мало данных -> случайный сплит 80/10/10
        idx = np.random.RandomState(42).permutation(len(df))
        n_t = max(1, int(0.1 * len(df)))
        test_df = df.iloc[idx[:n_t]].reset_index(drop=True)
        val_df = df.iloc[idx[n_t:2 * n_t]].reset_index(drop=True)
        train_df = df.iloc[idx[2 * n_t:]].reset_index(drop=True)
        print('fold-сплит дал пустой val/test -> случайный сплит 80/10/10')
    # ручные фичи из CSV + опциональный отбор лучших (только по ручным)
    from extract_features import load_features, fit_selector, apply_selector
    hand_feats, hand_names = load_features(args.features)
    selector = None
    if args.nb_feats and 0 < args.nb_feats < len(hand_names):
        selector = fit_selector(hand_feats, hand_names, train_df, args.feat_select, args.nb_feats)
        hand_feats = apply_selector(hand_feats, selector)
        hand_names = selector['names']
        print(f'Отбор ручных фич: метод={args.feat_select}, {len(hand_names)} из {len(selector["src_names"])}')

    # + интервальные фичи neurokit (считаются/кэшируются), объединение
    lead_idx = 0 if args.leads == 'lead1' else NEUROKIT_LEAD_INDEX
    print('Считаю/загружаю интервальные фичи neurokit...')
    feat_df, names = build_feature_frame(hand_feats, hand_names, df, (args.nk_cache or None), lead_idx)
    hand_n = len([n for n in names if not n.startswith('nk_')])
    # бустинг обучаем на train+val (у него своя внутренняя валидация для early stopping)
    Xtr, Ytr, _ = build_xy(pd.concat([train_df, val_df], ignore_index=True), feat_df, names)
    Xte, Yte, _ = build_xy(test_df, feat_df, names)
    print(f'Отведения: {args.leads} | фич: {len(names)} (ручных {hand_n} + neurokit {len(names)-hand_n}) | '
          f'train={len(Xtr)} test={len(Xte)} | перевзвешивание: {args.reweight}')
    if len(Xtr) == 0 or len(Xte) == 0:
        raise SystemExit('Пустой train/test — проверьте --features и --data-root.')

    models, base_rate = train_boosting(Xtr, Ytr, reweight=args.reweight)

    # сохраняем модель
    model_path = os.path.join(args.out, 'boosting_model.joblib')
    joblib.dump({'models': models, 'base_rate': base_rate, 'classes': CLASSES,
                 'feature_names': names, 'hand_selector': selector,
                 'leads': args.leads, 'reweight': args.reweight,
                 'nk_lead_index': lead_idx, 'params': BOOSTING_PARAMS}, model_path)
    print(f'Модель -> {model_path}')

    # метрики по классам (сразу на диск)
    probs = predict_proba(models, base_rate, Xte)
    table = per_label_metrics(probs, Yte, DECISION_THRESHOLD)
    metrics_path = os.path.join(args.out, 'per_label_metrics.csv')
    table.to_csv(metrics_path, index=False)

    from eval.evaluate_12ECG_score import compute_challenge_metric, compute_f_measure, load_weights
    weights = load_weights(_WEIGHTS, CLASSES)
    preds = (probs > DECISION_THRESHOLD).astype(int)
    macro = table[table['abbr'] == 'MACRO'].iloc[0]
    summary = {
        'model': 'boosting', 'leads': args.leads, 'reweight': args.reweight,
        'n_features': len(names), 'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
        'threshold': DECISION_THRESHOLD,
        'macro_AUROC': float(macro['AUROC']), 'macro_AUPRC': float(macro['AUPRC']),
        'macro_F1': float(compute_f_measure(Yte, preds)),
        'challenge_metric': float(compute_challenge_metric(weights, Yte, preds, CLASSES, NORMAL_CLASS)),
        'params': BOOSTING_PARAMS,
    }
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n=== Метрики по классам (test) ===')
    with pd.option_context('display.max_rows', None, 'display.width', 200,
                           'display.float_format', lambda v: f'{v:.3f}' if v == v else 'NA'):
        print(table[['abbr', 'name', 'support', 'AUROC', 'AUPRC', 'precision', 'recall', 'f1']].to_string(index=False))
    print(f"\nИтог: macro-F1={summary['macro_F1']:.3f} challenge={summary['challenge_metric']:.3f}")
    print(f'Метрики -> {metrics_path}, {os.path.join(args.out, "results.json")}')
    return table, summary


if __name__ == '__main__':
    main()
