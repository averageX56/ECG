"""
train_ctn.py
Обучение гибридной модели CTN (CNN-энкодер + Transformer) — пайплайны 2 и 3.

  Пайплайн 2 (DL-only):  только сигнал + демография (age/sex).
  Пайплайн 3 (+ фичи):   добавляет ручные фичи из CSV (--features features.csv),
                          предварительно посчитанные extract_features.py.

Локальный запуск на CUDA (или CPU). Метрика — официальная метрика контеста
(eval/). Слабый F1_macro лечится перевзвешиванием классов (pos_weight) и
поклассовым перебором порогов; отчёт печатает метрики при трёх режимах порогов.
Простой код: обычный torch DataLoader, без ручного мультипроцессинга.

Примеры:
  # Пайплайн 2:
  python train_ctn.py --data-root data --epochs 20
  # Пайплайн 3 (сначала extract_features.py --out features.csv):
  python train_ctn.py --data-root data --features features.csv --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ecg_data import (CLASSES, NORMAL_CLASS, build_record_table, make_split,
                      ECGDataset, FS_TARGET)
from model import build_model
from eval.evaluate_12ECG_score import (compute_auc, compute_beta_measures,
                                       compute_f_measure, compute_challenge_metric,
                                       load_weights)

_HERE = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_HERE, 'eval', 'weights.csv')


def make_model(name, leads, n_feats, device):
    """Строит модель ('ctn'|'resnet1d'|'unet1d') с нужным числом отведений."""
    in_ch = 12 if leads == 'all' else 1
    m = build_model(name, CLASSES, in_channels=in_ch, nb_feats=n_feats, nb_demo=2).to(device)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return m


def compute_pos_weight(df, cap=50.0, device=None):
    pos = np.array([df[c].sum() if c in df.columns else 0 for c in CLASSES], dtype=np.float64)
    neg = len(df) - pos
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.clip(np.where(pos > 0, neg / pos, 1.0), 1.0, cap)
    t = torch.tensor(w, dtype=torch.float32)
    return t.to(device) if device is not None else t


def make_wide(feats_t, age_t, sex_t, age_mean, age_std, device):
    age = ((age_t.float() - age_mean) / (age_std + 1e-8)).view(-1, 1)
    sex = sex_t.float().view(-1, 1)
    parts = [age.to(device), sex.to(device)]
    if feats_t is not None and feats_t.numel() > 0 and feats_t.shape[-1] > 0:
        parts.append(feats_t.float().to(device))
    return torch.cat(parts, dim=1)


def _bar(loader, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return loader


def train_one_epoch(model, loader, optimizer, age_mean, age_std, device, pos_weight, grad_clip, desc):
    model.train()
    losses = []
    bar = _bar(loader, desc)
    for segs, feats_t, lbl_t, age_t, sex_t in bar:
        inp = segs[:, 0].to(device, non_blocking=True)
        lbl_t = lbl_t.to(device, non_blocking=True)
        wide = make_wide(feats_t, age_t, sex_t, age_mean, age_std, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(inp, wide)
        loss = F.binary_cross_entropy_with_logits(out, lbl_t, pos_weight=pos_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        if hasattr(bar, 'set_postfix'):
            bar.set_postfix(loss=f'{np.mean(losses[-50:]):.4f}')
    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def get_probs(model, loader, age_mean, age_std, device, desc='eval'):
    model.eval()
    probs, lbls = [], []
    for segs, feats_t, lbl_t, age_t, sex_t in _bar(loader, desc):
        b, nw, c, l = segs.shape
        inp = segs.reshape(b * nw, c, l).to(device, non_blocking=True)
        wide = make_wide(feats_t, age_t, sex_t, age_mean, age_std, device)
        out = model(inp, wide.repeat_interleave(nw, dim=0)).view(b, nw, -1).mean(1)
        probs.append(out.sigmoid().cpu().numpy())
        lbls.append(lbl_t.numpy())
    return np.concatenate(probs), np.concatenate(lbls)


def tune_thresholds_f1(probs, lbls):
    """Поклассовый порог, максимизирующий F1 класса (лечит слабый F1_macro)."""
    thr = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        y = lbls[:, k]
        if y.sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            pred = probs[:, k] > t
            tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
            den = 2 * tp + fp + fn
            f1 = (2 * tp / den) if den else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[k] = best_t
    return thr


def find_global_threshold(probs, lbls, weights):
    best_s, best_t = -1e9, 0.5
    for t in np.arange(0.0, 1.0, 0.02):
        s = compute_challenge_metric(weights, lbls, (probs > t).astype(int), CLASSES, NORMAL_CLASS)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def evaluate(probs, lbls, thr, weights, beta=2):
    preds = (probs > np.asarray(thr)).astype(int)
    auroc, auprc = compute_auc(lbls, probs)
    fbeta, gbeta = compute_beta_measures(lbls, preds, beta)
    return {'AUROC': float(auroc), 'AUPRC': float(auprc),
            'macro_F1': float(compute_f_measure(lbls, preds)),
            'Fbeta': float(fbeta), 'Gbeta': float(gbeta),
            'challenge_metric': float(compute_challenge_metric(weights, lbls, preds, CLASSES, NORMAL_CLASS))}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Обучение модели (ctn/resnet1d/unet1d): пайплайн 2 или 3')
    p.add_argument('--data-root', default='data')
    p.add_argument('--model', default='ctn', choices=['ctn', 'resnet1d', 'unet1d'])
    p.add_argument('--features', default=None, help='CSV фич (extract_features.py) -> пайплайн 3')
    p.add_argument('--nb-feats', type=int, default=0,
                   help='Отобрать топ-N важных фич (0 = все). Важность считается по train.')
    p.add_argument('--feat-select', default='rf', choices=['rf', 'variance', 'mutual_info', 'pca'],
                   help='Метод отбора/снижения при --nb-feats')
    p.add_argument('--leads', default='all', choices=['all', 'lead1'])
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--fraction', type=float, default=1.0,
                   help='Доля датасета (0<f<=1) для быстрых прогонов на части данных')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4, help='Learning rate (Adam)')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--window', type=int, default=7500)
    p.add_argument('--nb-windows-eval', type=int, default=20)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--no-reweight', action='store_true')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--val-fold', type=int, default=8)
    p.add_argument('--test-fold', type=int, default=9)
    p.add_argument('--cache-dir', default='cache/signals',
                   help='Дисковый кэш сигнала (ускоряет эпохи). "" — отключить.')
    p.add_argument('--checkpoint', default='checkpoints/model.pt')
    p.add_argument('--save-epochs', default='', help=(
        'Список отсечек через запятую (напр. "40,50,60"). На каждой сохраняется '
        'best-so-far чекпоинт (лучшие веса среди эпох 0..N по val_AUROC). '
        'Имя = {checkpoint без .pt}_e{N}.pt. Пусто = выключено.'))
    p.add_argument('--results-out', default='outputs/results.json')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args(argv)


def _feature_stats(features, names, train_ids):
    X = np.vstack([features[i] for i in train_ids if i in features]).astype(np.float64)
    X[np.isinf(X)] = np.nan
    means = np.nan_to_num(np.nanmean(X, axis=0))
    stds = np.nanstd(X, axis=0)
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    return means, stds


def get_device(prefer):
    if prefer == 'cpu':
        return torch.device('cpu')
    if prefer in ('cuda', 'auto') and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(42); np.random.seed(42)
    device = get_device('cpu' if args.smoke else args.device)

    df = build_record_table(args.data_root, limit=args.limit)
    if args.fraction < 1.0:                       # обучение на части датасета
        df = df.sample(frac=args.fraction, random_state=42).reset_index(drop=True)
        print(f'Взята доля датасета: {args.fraction:.2f} -> {len(df)} записей')
    train_df, val_df, test_df = make_split(df, args.val_fold, args.test_fold)
    if len(val_df) == 0 or len(test_df) == 0:      # мало данных -> случайный сплит 80/10/10
        idx = np.random.RandomState(42).permutation(len(df))
        n_t = max(1, int(0.1 * len(df)))
        test_df = df.iloc[idx[:n_t]].reset_index(drop=True)
        val_df = df.iloc[idx[n_t:2 * n_t]].reset_index(drop=True)
        train_df = df.iloc[idx[2 * n_t:]].reset_index(drop=True)
        print('fold-сплит дал пустой val/test -> случайный сплит 80/10/10')
    print(f'Модель: {args.model} | train={len(train_df)} val={len(val_df)} test={len(test_df)}')

    # ручные фичи (пайплайн 3)
    features, feat_means, feat_stds, n_feats, selector = None, None, None, 0, None
    if args.features:
        from extract_features import load_features, fit_selector, apply_selector
        features, names = load_features(args.features)
        if args.nb_feats and 0 < args.nb_feats < len(names):
            selector = fit_selector(features, names, train_df, args.feat_select, args.nb_feats)
            features = apply_selector(features, selector)
            names = selector['names']
            print(f"Отбор фич: метод={args.feat_select}, {len(names)} из {len(selector['src_names'])}")
        feat_means, feat_stds = _feature_stats(features, names, train_df['record_id'].tolist())
        n_feats = len(names)
        print(f'Пайплайн 3: ручных фич = {n_feats} (из {args.features})')
    else:
        print('Пайплайн 2: DL-only (без ручных фич)')

    window, nb_eval, epochs = args.window, args.nb_windows_eval, args.epochs
    cache_dir = args.cache_dir or None
    if args.smoke:
        window, nb_eval, epochs = 2500, 2, 1

    def ds(d, nb, aug):
        return ECGDataset(d, window, nb, leads=args.leads, features=features,
                          feat_means=feat_means, feat_stds=feat_stds,
                          cache_dir=cache_dir, augment=aug)

    def dl(dataset, shuffle):
        # Однопроцессная загрузка (num_workers=0) — всё «обычно», без доп. процессов.
        # Скорость эпох обеспечивает дисковый кэш сигнала (--cache-dir).
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=torch.cuda.is_available())

    trn = dl(ds(train_df, 1, True), True)
    val = dl(ds(val_df, nb_eval, False), False)
    tst = dl(ds(test_df, nb_eval, False), False)

    age_mean = float(np.nanmean(train_df['age'])) if train_df['age'].notna().any() else 60.0
    age_std = float(np.nanstd(train_df['age'])) if train_df['age'].notna().any() else 15.0

    model = make_model(args.model, args.leads, n_feats, device)
    print(f'Устройство={device.type} | параметров={sum(p.numel() for p in model.parameters())/1e6:.1f}M')
    pos_weight = None if args.no_reweight else compute_pos_weight(train_df, device=device)
    weights = load_weights(_WEIGHTS, CLASSES)

    # Обычный Adam + ReduceLROnPlateau по val-AUROC (стабильно, без Noam-warmup,
    # который перехлёстывал LR и ронял обучение после первых эпох).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    # Единый формат чекпоинта: снапшоты best-so-far и финальный файл пишутся
    # одинаково, чтобы eval_model.py читал их без различий.
    def save_checkpoint(state_dict, path, auroc):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'model_state_dict': state_dict,
                    'model': args.model, 'leads': args.leads, 'n_feats': n_feats,
                    'best_auroc': float(auroc), 'age_mean': age_mean, 'age_std': age_std,
                    'feat_means': (feat_means.tolist() if feat_means is not None else None),
                    'feat_stds': (feat_stds.tolist() if feat_stds is not None else None),
                    'selector': selector},   # None или отбор фич (для воспроизведения в eval)
                   path)

    # Отсечки для best-so-far снапшотов: "40,50,60" -> {40, 50, 60}.
    save_epochs = sorted({int(x) for x in args.save_epochs.split(',') if x.strip()})
    ckpt_base = args.checkpoint[:-3] if args.checkpoint.endswith('.pt') else args.checkpoint

    best_auroc, best_state, patience = -1.0, None, 0
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, trn, optimizer, age_mean, age_std, device,
                               pos_weight, 1.0, desc=f'эпоха {epoch}')
        vp, vl = get_probs(model, val, age_mean, age_std, device, desc='val')
        vauroc, _ = compute_auc(vl, vp)
        scheduler.step(vauroc)
        lr_now = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'loss': loss, 'val_auroc': float(vauroc), 'lr': lr_now})
        print(f'эпоха {epoch:3d}: loss={loss:.4f} val_AUROC={vauroc:.4f} lr={lr_now:.1e} ({time.time()-t0:.1f}s)')
        if vauroc > best_auroc:
            best_auroc, patience = vauroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f'ранняя остановка на эпохе {epoch}')
                break
        # Отсечка = число пройденных эпох (epoch 0-индексный -> epoch+1).
        # Сохраняем best-so-far: лучшие веса среди эпох 0..(epoch+1).
        if (epoch + 1) in save_epochs and best_state is not None:
            snap = f'{ckpt_base}_e{epoch + 1}.pt'
            save_checkpoint(best_state, snap, best_auroc)
            print(f'  снапшот best-so-far ({epoch + 1} эпох, val_AUROC={best_auroc:.4f}) -> {snap}')
    if best_state is not None:
        model.load_state_dict(best_state)

    if args.checkpoint:
        save_checkpoint(model.state_dict(), args.checkpoint, best_auroc)
        print(f'чекпоинт -> {args.checkpoint}')

    # пороги: 0.5 vs глобальный vs поклассовый (на валидации)
    vp, vl = get_probs(model, val, age_mean, age_std, device, desc='val')
    thr_f1 = tune_thresholds_f1(vp, vl)
    thr_glob = find_global_threshold(vp, vl, weights)
    tp, tl = get_probs(model, tst, age_mean, age_std, device, desc='test')
    metrics = {
        'test_default@0.5':   evaluate(tp, tl, 0.5, weights),
        'test_global_thr':    evaluate(tp, tl, thr_glob, weights),
        'test_perclass_thr':  evaluate(tp, tl, thr_f1, weights),
    }
    print('\n=== метрики (test) ===')
    for name, m in metrics.items():
        print(f'  {name:20s} macroF1={m["macro_F1"]:.3f} challenge={m["challenge_metric"]:.3f} AUROC={m["AUROC"]:.3f}')

    results = {'model': args.model, 'pipeline': 3 if args.features else 2, 'leads': args.leads,
               'fraction': args.fraction, 'n_feats': n_feats, 'history': history,
               'metrics': metrics, 'thresholds_perclass': thr_f1.tolist(), 'threshold_global': thr_glob}
    if args.results_out:
        os.makedirs(os.path.dirname(args.results_out) or '.', exist_ok=True)
        with open(args.results_out, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'результаты -> {args.results_out}')
    return results


if __name__ == '__main__':
    main()
