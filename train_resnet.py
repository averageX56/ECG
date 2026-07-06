"""
Train a plain 1D ResNet (resnet1d.py) directly on raw ECG windows, dropping
the whole hand-crafted HRV/template feature pipeline. Only a tiny side-input
(normalized age + sex) is kept, matching model.py's nb_demo=2 slot.

Reuses from ecg_pipeline.py (must be importable, e.g. same project dir):
  - apply_compat_patches, load_records_table, build_splits
  - load_challenge_data, apply_filter, normalize
  - CLASSES, NORMAL_CLASS, FS, FILTER_BANDWIDTH
  - find_best_thresholds, evaluate_full (via eval.evaluate_12ECG_score)

Usage:
    python train_resnet.py --data-root /path/to/physionet2020 \
        --csv records_stratified_10_folds_v2.csv --tst-fold 3

Trains and evaluates BOTH variants in one run: leads='lead1' and leads='all'.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import ecg_pipeline as ep          # reused: splits, IO, filtering, metrics
from resnet1d import build_resnet1d


# --------------------------------------------------------------------------
# Dataset: raw signal + age/sex only (NO hand-crafted feature computation)
# --------------------------------------------------------------------------
class ECGRawDataset(Dataset):
    def __init__(self, df, window, nb_windows, leads):
        self.df = df.reset_index(drop=True)
        self.window = window
        self.nb_windows = nb_windows
        self.leads = leads  # 'all' | 'lead1'

    def __len__(self):
        return len(self.df)

    def _windows(self, data):
        seq_len = data.shape[-1]
        pad = max(self.window - seq_len, 0)
        if pad > 0:
            data = np.pad(data, ((0, 0), (0, pad + 1)))
            seq_len = data.shape[-1]
        starts = np.random.randint(0, seq_len - self.window + 1, size=self.nb_windows)
        return np.stack([data[:, s:s + self.window] for s in starts])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec, hdr = ep.load_challenge_data(row.hea_path)

        sig = rec[0:1, :] if self.leads == 'lead1' else rec
        sig = ep.apply_filter(sig, ep.FILTER_BANDWIDTH)
        sig = ep.normalize(sig)
        segs = self._windows(sig).astype(np.float32)

        lbl = row[ep.CLASSES].values.astype(np.float32)
        age = ep.get_age(hdr)
        sex = ep.get_sex(hdr)
        return segs, lbl, age, sex


# --------------------------------------------------------------------------
# Train / eval loops (mirrors ecg_pipeline.py but without feats_t)
# --------------------------------------------------------------------------
def make_wide_feats(age_t, sex_t, age_mean, age_std, device):
    age_t = ((age_t.float() - age_mean) / (age_std + 1e-8))[:, None]
    sex_t = sex_t.float()[:, None]
    return torch.cat([age_t, sex_t], dim=1).to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, age_mean, age_std, device):
    model.train()
    losses = []
    for segs, lbl_t, age_t, sex_t in loader:
        inp = segs[:, 0].to(device, non_blocking=True)   # nb_windows=1 during training
        lbl_t = lbl_t.to(device, non_blocking=True)
        wide = make_wide_feats(age_t, sex_t, age_mean, age_std, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(inp, wide)
        loss = F.binary_cross_entropy_with_logits(out, lbl_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def get_probs(model, loader, age_mean, age_std, device):
    model.eval()
    probs_all, lbls_all = [], []
    with torch.no_grad():
        for segs, lbl_t, age_t, sex_t in loader:
            b, nw, c, l = segs.shape
            inp = segs.reshape(b * nw, c, l).to(device, non_blocking=True)
            wide = make_wide_feats(age_t, sex_t, age_mean, age_std, device)
            wide_rep = wide.repeat_interleave(nw, dim=0)
            out = model(inp, wide_rep)
            out = out.view(b, nw, -1).mean(dim=1)
            probs_all.append(out.sigmoid().cpu().numpy())
            lbls_all.append(lbl_t.numpy())
    return np.concatenate(probs_all), np.concatenate(lbls_all)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=num_workers > 0,
                       prefetch_factor=4 if num_workers > 0 else None)


def run_resnet_pipeline(leads, tag, trn_df, val_df, tst_df, window,
                         weights_matrix, config, device):
    trn_ds = ECGRawDataset(trn_df, window, config['NB_WINDOWS_TRAIN'], leads)
    val_ds = ECGRawDataset(val_df, window, config['NB_WINDOWS_EVAL'], leads)
    tst_ds = ECGRawDataset(tst_df, window, config['NB_WINDOWS_EVAL'], leads)

    trn_loader = make_loader(trn_ds, config['BATCH_SIZE'], True, config['NUM_WORKERS'])
    val_loader = make_loader(val_ds, config['BATCH_SIZE'], False, config['NUM_WORKERS'])
    tst_loader = make_loader(tst_ds, config['BATCH_SIZE'], False, config['NUM_WORKERS'])

    age_mean, age_std = float(trn_df.Age.mean()), float(trn_df.Age.std())

    model = build_resnet1d(
        ep.CLASSES, leads=leads,
        layers=config['RESNET_LAYERS'],
        base_planes=config['BASE_PLANES'],
        deepfeat_sz=config['DEEPFEAT_SZ'],
        dropout_rate=config['DROPOUT_RATE'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_auroc, best_state, patience_count = 0., None, 0
    history = []
    for epoch in range(config['N_EPOCHS']):
        trn_loss = train_one_epoch(model, trn_loader, optimizer, age_mean, age_std, device)
        val_probs, val_lbls = get_probs(model, val_loader, age_mean, age_std, device)
        from eval.evaluate_12ECG_score import compute_auc
        val_auroc, _ = compute_auc(val_lbls, val_probs)
        scheduler.step(val_auroc)
        history.append({'epoch': epoch, 'trn_loss': trn_loss, 'val_auroc': val_auroc})
        print(f'[{tag}] epoch {epoch}: trn_loss={trn_loss:.4f} val_auroc={val_auroc:.4f}')

        if val_auroc > best_auroc:
            best_auroc, patience_count = val_auroc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if patience_count >= config['PATIENCE']:
            print(f'[{tag}] early stopping at epoch {epoch}')
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs, val_lbls = get_probs(model, val_loader, age_mean, age_std, device)
    thrs = ep.find_best_thresholds(val_probs, val_lbls, weights_matrix)
    val_metrics = ep.evaluate_full(val_probs, val_lbls, thrs, weights_matrix)

    tst_probs, tst_lbls = get_probs(model, tst_loader, age_mean, age_std, device)
    tst_metrics = ep.evaluate_full(tst_probs, tst_lbls, thrs, weights_matrix)

    return {
        'tag': tag, 'leads': leads, 'thrs': thrs.tolist(),
        'history': history, 'val_metrics': val_metrics, 'tst_metrics': tst_metrics,
    }, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--csv', default='records_stratified_10_folds_v2.csv')
    parser.add_argument('--tst-fold', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--window-seconds', type=int, default=15)
    parser.add_argument('--output-dir', default='resnet_results')
    args = parser.parse_args()

    ep.apply_compat_patches()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    data_df, n_total = ep.load_records_table(args.csv, args.data_root)
    print(f'used {len(data_df)} / {n_total} records')
    trn_df, val_df, tst_df, val_fold, trn_fold = ep.build_splits(data_df, args.tst_fold)
    print('trn:', len(trn_df), 'val:', len(val_df), 'tst:', len(tst_df))

    from eval.evaluate_12ECG_score import load_weights
    weights_matrix = load_weights('eval/weights.csv', ep.CLASSES)

    window = args.window_seconds * ep.FS
    config = {
        'N_EPOCHS': args.n_epochs,
        'PATIENCE': args.patience,
        'BATCH_SIZE': args.batch_size,
        'NUM_WORKERS': args.num_workers,
        'NB_WINDOWS_TRAIN': 1,
        'NB_WINDOWS_EVAL': 20,
        'LR': 1e-3,
        'RESNET_LAYERS': (2, 2, 2, 2),   # ResNet18-style depth
        'BASE_PLANES': 64,
        'DEEPFEAT_SZ': 128,
        'DROPOUT_RATE': 0.3,
    }

    results = {}
    for leads, tag in [('lead1', 'ResNet1d (Lead I)'), ('all', 'ResNet1d (12 leads)')]:
        res, model = run_resnet_pipeline(leads, tag, trn_df, val_df, tst_df,
                                          window, weights_matrix, config, device)
        results[leads] = res
        torch.save({'model_state_dict': model.state_dict(), 'config': config},
                   os.path.join(args.output_dir, f'resnet1d_{leads}.tar'))
        print(f"[{tag}] test metrics: {res['tst_metrics']}")

    rows = []
    for leads, res in results.items():
        row = {'variant': res['tag']}
        row.update({f'val_{k}': v for k, v in res['val_metrics'].items()})
        row.update({f'test_{k}': v for k, v in res['tst_metrics'].items()})
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
    print(comparison)


if __name__ == '__main__':
    main()
