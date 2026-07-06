import os
import re
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

FS = 500
FILTER_BANDWIDTH = [3, 45]
CH_IDX = 1

CLASSES = sorted([
    '270492004', '164889003', '164890007', '426627000', '713427006',
    '713426002', '445118002', '39732003', '164909002', '251146004',
    '698252002', '10370003', '284470004', '427172004', '164947007',
    '111975006', '164917005', '47665007', '59118001', '427393009',
    '426177001', '426783006', '427084000', '63593006', '164934002',
    '59931005', '17338001',
])
NORMAL_CLASS = '426783006'

CHAR2DIR = {'Q': 'Training_2', 'A': 'Training_WFDB', 'E': 'WFDB',
            'S': 'WFDB', 'H': 'WFDB', 'I': 'WFDB'}

PYEEG_SOURCE = '''
import numpy as np

def embed_seq(X, Tau, D):
    X = np.asarray(X)
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

def pfd(X, D=None):
    X = np.asarray(X, dtype=np.float64)
    D = np.diff(X) if D is None else np.asarray(D)
    N_delta = np.sum(D[1:] * D[:-1] < 0)
    n = len(X)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

def ap_entropy(X, M, R):
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if N <= M + 1:
        return np.nan
    Em = embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)
    InRange = np.max(D, axis=2) <= R
    Cm = InRange.mean(axis=0)
    Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)
    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)
    with np.errstate(divide="ignore"):
        Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))
    return (Phi_m - Phi_mp) / (N - M)
'''.lstrip()


def apply_compat_patches():
    for name, type_ in [('int', int), ('float', float), ('bool', bool)]:
        if not hasattr(np, name):
            setattr(np, name, type_)
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    if not hasattr(pd.DataFrame, 'append'):
        def _append(self, other, ignore_index=False, **kwargs):
            if isinstance(other, pd.Series):
                other = other.to_frame().T
            return pd.concat([self, other], ignore_index=ignore_index)
        pd.DataFrame.append = _append


def ensure_pyeeg(path='feats/pyeeg.py'):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(PYEEG_SOURCE)


def check_top_feats(path='top_feats.npy'):
    try:
        tf = np.load(path, allow_pickle=True)
        return True, tf
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Предвычисленный кэш фич (feats/<Dataset>/all_feats_ch{N}.zip)
#
# Раньше этот кэш вообще не использовался: utils.py искал файлы по паттерну
# '*/*all_feats_ch_{ch_idx}.zip' (с подчёркиванием перед номером канала), а по факту
# файлы называются 'all_feats_ch1.zip' (без подчёркивания) -> glob ничего не находил.
# Из-за этого весь конвейер (compute_feature_table/ECGBenchDataset) пересчитывал фичи
# заново из сырого сигнала через compute_features(), которая на части записей падает
# и молча возвращает None -> фичи записи схлопываются в NaN -> заполняются feat_means.
# ---------------------------------------------------------------------------

def find_feats_cache_files(feats_dir='feats', ch_idx=CH_IDX):
    """Найти все файлы предвычисленных фич под feats_dir, независимо от того,
    называются они 'all_feats_ch{N}.zip' или (устаревший вариант) 'all_feats_ch_{N}.zip'."""
    feats_dir = Path(feats_dir)
    files = sorted(set(
        list(feats_dir.rglob(f'*all_feats_ch{ch_idx}.zip')) +
        list(feats_dir.rglob(f'*all_feats_ch_{ch_idx}.zip'))
    ))
    return files


def load_feats_cache(feats_dir='feats', ch_idx=CH_IDX):
    """Прочитать и объединить все предвычисленные фичи. Возвращает (DataFrame, id_col).

    Бросает FileNotFoundError, если ничего не нашлось (лучше упасть явно здесь,
    чем молча уйти в ненадёжный пересчёт на лету).
    """
    files = find_feats_cache_files(feats_dir, ch_idx)
    if not files:
        raise FileNotFoundError(
            f"Не найдено ни одного файла '*all_feats_ch{ch_idx}.zip' под '{Path(feats_dir).resolve()}'. "
            "Проверьте путь и реальные имена файлов внутри подпапок датасетов."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df['__source_zip'] = str(f)
        dfs.append(df)
    all_feats = pd.concat(dfs, ignore_index=True)

    id_col_candidates = [c for c in all_feats.columns if c.lower() in ('filename', 'patient', 'record')]
    if not id_col_candidates:
        raise ValueError(
            "В закэшированной таблице фич не нашлось колонки-идентификатора записи "
            "(ожидалось одно из: filename/Patient/record). Колонки: "
            f"{list(all_feats.columns)}"
        )
    return all_feats, id_col_candidates[0]


def build_feats_lookup(all_feats, id_col, feature_names):
    """Построить словарь Patient -> np.ndarray нормализуемых фич (без нормализации)."""
    missing = [f for f in feature_names if f not in all_feats.columns]
    if missing:
        print(f'[build_feats_lookup] ВНИМАНИЕ: этих фич нет в кэше и они будут пропущены: {missing}')
    present = [f for f in feature_names if f in all_feats.columns]

    indexed = all_feats.drop_duplicates(subset=id_col).set_index(id_col)
    lookup = {
        pid: indexed.loc[pid, present].values.astype(np.float64)
        for pid in indexed.index
    }
    return lookup, present


def feat_stats_from_cache(all_feats, feature_names):
    """feat_means/feat_stds, посчитанные по реальному предвычисленному кэшу, а не по
    выборке из compute_feature_table (часть которой могла провалиться в NaN)."""
    feats_only = all_feats[feature_names].replace([np.inf, -np.inf], np.nan)
    feat_means = feats_only.mean().values.copy()
    feat_stds = feats_only.std().values.copy()
    feat_stds[feat_stds == 0] = 1.0
    return feat_means, feat_stds


def record_dir(patient_id):
    return CHAR2DIR.get(patient_id[0], None)


def build_hea_path(patient_id, data_root):
    d = record_dir(patient_id)
    if d is None:
        return None
    return os.path.join(data_root, d, patient_id + '.hea')


def load_records_table(csv_path, data_root, data_fraction=1.0, max_records=None, seed=42):
    data_df = pd.read_csv(csv_path, index_col=0)
    data_df['hea_path'] = data_df.Patient.apply(lambda p: build_hea_path(p, data_root))
    found = data_df.hea_path.apply(lambda p: p is not None and os.path.exists(p))
    data_df = data_df[found].reset_index(drop=True)

    n_total = len(data_df)
    n_keep = n_total
    if max_records is not None:
        n_keep = min(n_keep, max_records)
    if data_fraction is not None:
        n_keep = min(n_keep, int(round(n_total * data_fraction)))
    if n_keep < n_total:
        data_df = data_df.sample(n=n_keep, random_state=seed).reset_index(drop=True)
    return data_df, n_total


def build_splits(data_df, tst_fold):
    val_fold = (tst_fold - 1) % 10
    trn_fold = [f for f in range(10) if f not in (val_fold, tst_fold)]
    trn_df = data_df[data_df.fold.isin(trn_fold)].reset_index(drop=True)
    val_df = data_df[data_df.fold == val_fold].reset_index(drop=True)
    tst_df = data_df[data_df.fold == tst_fold].reset_index(drop=True)
    return trn_df, val_df, tst_df, val_fold, trn_fold


def load_challenge_data(header_file):
    from scipy.io import loadmat
    from scipy.signal import resample

    with open(header_file, 'r') as f:
        header = f.readlines()
    sampling_rate = int(header[0].split()[2])
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    if sampling_rate != FS:
        n_target = int(round(recording.shape[-1] * (FS / sampling_rate)))
        recording = resample(recording, n_target, axis=1)
    return recording, header


def apply_filter(signal, band=None, fs=FS):
    from biosppy.signals.tools import filter_signal
    band = FILTER_BANDWIDTH if band is None else band
    order = int(0.3 * fs)
    filtered, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                    order=order, frequency=band, sampling_rate=fs)
    return filtered


def normalize(seq, smooth=1e-8):
    mn = np.min(seq, axis=1)[:, None]
    mx = np.max(seq, axis=1)[:, None]
    return 2 * (seq - mn) / (mx - mn + smooth) - 1


# Счётчики/примеры провалов compute_features(), чтобы проблема была видна, а не тонула
# в тихом `except Exception: return None` (именно так фичи молча превращались в NaN
# для части записей, если в кэше их не было).
FEATURE_EXTRACTION_STATS = {'ok': 0, 'failed': 0, 'first_error': None, 'first_error_record': None}


def compute_features(recording, channel=CH_IDX, raise_errors=False, record_id=None):
    import warnings
    import traceback
    from feats.features import Features
    channel = min(channel, recording.shape[0] - 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ef = Features(data=recording[channel].copy(), fs=FS,
                          feature_groups=['full_waveform_statistics',
                                          'heart_rate_variability_statistics',
                                          'template_statistics'])
            ef.calculate_features(filter_bandwidth=FILTER_BANDWIDTH, show=False,
                                   channel=channel, normalize=True, polarity_check=True,
                                   template_before=0.25, template_after=0.4)
            feats = ef.get_features()
        FEATURE_EXTRACTION_STATS['ok'] += 1
        return feats.iloc[0] if len(feats) else None
    except Exception:
        FEATURE_EXTRACTION_STATS['failed'] += 1
        if FEATURE_EXTRACTION_STATS['first_error'] is None:
            FEATURE_EXTRACTION_STATS['first_error'] = traceback.format_exc()
            FEATURE_EXTRACTION_STATS['first_error_record'] = record_id
        if raise_errors:
            raise
        return None


def get_age(hdr_lines):
    for line in hdr_lines:
        if line.startswith('#Age'):
            m = re.search(r'(\d+)', line)
            if m:
                return float(m.group(1))
    return 60.0


def get_sex(hdr_lines):
    for line in hdr_lines:
        if line.startswith('#Sex'):
            return 1.0 if 'Female' in line else 0.0
    return 0.0


def _cache_key(hea_file):
    return hashlib.md5(hea_file.encode()).hexdigest() + '.npz'


def load_record_cached(hea_file, cache_dir, skip_feats=False):
    """skip_feats=True пропускает вызов compute_features() (сигнал/возраст/пол всё равно
    считаются). Используется ECGBenchDataset, когда фичи для записи уже есть в
    предвычисленном кэше (feats_lookup) — тогда пересчитывать их на лету не нужно и
    не стоит, т.к. compute_features на части записей ненадёжен/падает."""
    os.makedirs(cache_dir, exist_ok=True)
    cpath = os.path.join(cache_dir, _cache_key(hea_file))
    if os.path.exists(cpath):
        with np.load(cpath, allow_pickle=True) as d:
            return d['sig'], d['feats'].item(), float(d['age']), float(d['sex'])

    rec, hdr = load_challenge_data(hea_file)
    if skip_feats:
        feats_dict = {}
    else:
        feats = compute_features(rec, record_id=hea_file)
        feats_dict = feats.to_dict() if feats is not None else {}
    filt = apply_filter(rec)
    sig = normalize(filt).astype(np.float32)
    age = get_age(hdr)
    sex = get_sex(hdr)

    # Если фичи не считали (skip_feats), НЕ пишем результат на диск как окончательный —
    # иначе при последующем обращении без feats_lookup запись навсегда останется без фич.
    if not skip_feats:
        tmp_path = os.path.join(cache_dir, f'.tmp{os.getpid()}_{_cache_key(hea_file)}')
        np.savez(tmp_path, sig=sig, feats=feats_dict, age=age, sex=sex)
        os.replace(tmp_path, cpath)
    return sig, feats_dict, age, sex


def compute_feature_table(df, cache_dir, sample_size, seed=42, debug=False):
    import traceback
    sub = df.sample(n=min(sample_size, len(df)), random_state=seed)
    rows, labels = [], []
    n_ok, n_empty_feats, n_exceptions = 0, 0, 0
    first_error_path, first_error_tb = None, None

    for _, row in sub.iterrows():
        try:
            _, feats_dict, _, _ = load_record_cached(row.hea_path, cache_dir)
        except Exception:
            n_exceptions += 1
            if first_error_tb is None:
                first_error_path = row.hea_path
                first_error_tb = traceback.format_exc()
            if debug and n_exceptions == 1:
                raise
            continue
        if not feats_dict:
            n_empty_feats += 1
            continue
        n_ok += 1
        rows.append(feats_dict)
        labels.append(row[CLASSES].values.astype(int))

    print(f'compute_feature_table: ok={n_ok} empty_feats={n_empty_feats} '
          f'exceptions={n_exceptions} out of {len(sub)}')

    if n_ok == 0:
        msg = (f'0 записей дали фичи: {n_exceptions} упали с исключением, '
               f'{n_empty_feats} вернули пустые фичи.\n'
               f'Первая ошибка на {first_error_path}:\n{first_error_tb}')
        raise RuntimeError(msg)

    X = pd.DataFrame(rows).reset_index(drop=True)
    y = np.array(labels)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    return X, y, num_cols


def select_top_features(X, y, num_cols, nb_feats, n_top_labels=5, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, max_depth=8)
    top_label_idx = np.argsort(y.sum(axis=0))[::-1][:n_top_labels]
    importances = np.zeros(len(num_cols))
    n_used = 0
    for li in top_label_idx:
        yy = y[:, li]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        rf.fit(X, yy)
        importances += rf.feature_importances_
        n_used += 1
    importances /= max(n_used, 1)
    feat_importance = pd.Series(importances, index=num_cols).sort_values(ascending=False)
    return list(feat_importance.head(nb_feats).index)


class ECGBenchDataset(Dataset):
    def __init__(self, df, window, nb_windows, leads, top_feat_names, feat_means, feat_stds, cache_dir,
                 feats_lookup=None):
        """
        feats_lookup: optional dict {Patient -> np.ndarray фич в порядке top_feat_names}, полученный
            из ecg_pipeline.load_feats_cache()/build_feats_lookup(). Если для записи есть значение
            в feats_lookup, оно используется вместо пересчёта из сырого сигнала (который на части
            записей ненадёжен, см. compute_features/FEATURE_EXTRACTION_STATS). Если запись отсутствует
            в кэше (например, это новая запись при реальном инференсе), фичи считаются на лету, как раньше.
        """
        self.df = df.reset_index(drop=True)
        self.window = window
        self.nb_windows = nb_windows
        self.leads = leads
        self.top_feat_names = top_feat_names
        self.feat_means = feat_means
        self.feat_stds = feat_stds
        self.cache_dir = cache_dir
        self.feats_lookup = feats_lookup or {}
        self.n_from_cache = 0
        self.n_recomputed = 0

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

        cached_vals = self.feats_lookup.get(row.Patient)
        if cached_vals is not None:
            # Берём предвычисленные фичи из feats/<Dataset>/all_feats_ch{N}.zip — надёжнее и
            # быстрее пересчёта на лету. compute_features() для этой записи не вызывается вовсе.
            sig, _feats_dict, age, sex = load_record_cached(row.hea_path, self.cache_dir, skip_feats=True)
            vals = cached_vals.astype(np.float64).copy()
            self.n_from_cache += 1
        else:
            sig, feats_dict, age, sex = load_record_cached(row.hea_path, self.cache_dir)
            vals = np.array([feats_dict.get(name, np.nan) for name in self.top_feat_names], dtype=np.float64)
            self.n_recomputed += 1

        vals[np.isinf(vals)] = np.nan
        mask = np.isnan(vals)
        vals[mask] = self.feat_means[mask]
        feats_norm = (vals - self.feat_means) / self.feat_stds

        sig = sig[0:1] if self.leads == 'lead1' else sig
        segs = self._windows(sig).astype(np.float32)

        lbl = row[CLASSES].values.astype(np.float32)
        return segs, feats_norm.astype(np.float32), lbl, age, sex


def build_model(leads, top_feat_names, device, d_model=256, nhead=8, d_ff=2048,
                 num_layers=8, dropout=0.2, deepfeat_sz=64, nb_demo=2, seed=None):
    from model import CTN

    class CTN1Lead(CTN):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            old = self.encoder[0]
            self.encoder[0] = nn.Conv1d(1, old.out_channels, kernel_size=old.kernel_size,
                                         stride=old.stride, padding=old.padding, bias=False)

    if seed is not None:
        torch.manual_seed(seed)

    cls = CTN if leads == 'all' else CTN1Lead
    m = cls(d_model, nhead, d_ff, num_layers, dropout, deepfeat_sz,
            len(top_feat_names), nb_demo, CLASSES).to(device)
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return m


def load_pretrained(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt.get('best_auroc')


class NoamScheduler:
    def __init__(self, optimizer, model_size, factor=1, warmup=4000):
        self.optimizer = optimizer
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self._step = 0

    def rate(self, step=None):
        step = self._step if step is None else step
        step = max(step, 1)
        return self.factor * (self.model_size ** (-0.5) *
                               min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def step_lr(self):
        self._step += 1
        lr = self.rate()
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


def make_wide_feats(feats_t, age_t, sex_t, age_mean, age_std, device):
    age_t = ((age_t.float() - age_mean) / (age_std + 1e-8))[:, None]
    sex_t = sex_t.float()[:, None]
    return torch.cat([age_t, sex_t, feats_t.float()], dim=1).to(device, non_blocking=True)


def train_one_epoch(model, loader, scheduler, age_mean, age_std, device, scaler):
    model.train()
    losses = []
    for segs, feats_t, lbl_t, age_t, sex_t in loader:
        inp = segs[:, 0].to(device, non_blocking=True)
        lbl_t = lbl_t.to(device, non_blocking=True)
        wide = make_wide_feats(feats_t, age_t, sex_t, age_mean, age_std, device)

        scheduler.step_lr()
        scheduler.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            out = model(inp, wide)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(scheduler.optimizer)
            scaler.update()
        else:
            loss.backward()
            scheduler.optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses))


def get_probs(model, loader, age_mean, age_std, device):
    model.eval()
    probs_all, lbls_all = [], []
    with torch.no_grad():
        for segs, feats_t, lbl_t, age_t, sex_t in loader:
            b, nw, c, l = segs.shape
            inp = segs.reshape(b * nw, c, l).to(device, non_blocking=True)
            wide = make_wide_feats(feats_t, age_t, sex_t, age_mean, age_std, device)
            wide_rep = wide.repeat_interleave(nw, dim=0)
            out = model(inp, wide_rep)
            out = out.view(b, nw, -1).mean(dim=1)
            probs_all.append(out.sigmoid().cpu().numpy())
            lbls_all.append(lbl_t.numpy())
    return np.concatenate(probs_all), np.concatenate(lbls_all)


def find_best_thresholds(probs, lbls, weights_matrix):
    from eval.evaluate_12ECG_score import compute_challenge_metric
    step = 0.02
    scores = [compute_challenge_metric(weights_matrix, lbls, (probs > thr).astype(int),
                                        CLASSES, NORMAL_CLASS)
              for thr in np.arange(0., 1., step)]
    return np.array([np.argmax(scores) * step])


def evaluate_full(probs, lbls, thrs, weights_matrix, beta=2):
    from eval.evaluate_12ECG_score import compute_auc, compute_beta_measures, compute_challenge_metric
    preds = (probs > thrs).astype(int)
    auroc, auprc = compute_auc(lbls, probs)
    f_beta, g_beta = compute_beta_measures(lbls, preds, beta)
    return {
        'AUROC': auroc, 'AUPRC': auprc, 'Fbeta': f_beta, 'Gbeta': g_beta,
        'geometric_mean': float(np.sqrt(f_beta * g_beta)),
        'challenge_metric': compute_challenge_metric(weights_matrix, lbls, preds, CLASSES, NORMAL_CLASS),
    }


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def run_pipeline(leads, tag, trn_df, val_df, tst_df, top_feat_names, feat_means, feat_stds,
                  cache_dir, window, weights_matrix, config, device, checkpoint_path=None,
                  feats_lookup=None):
    torch.backends.cudnn.benchmark = True

    trn_ds = ECGBenchDataset(trn_df, window, config['NB_WINDOWS_TRAIN'], leads,
                              top_feat_names, feat_means, feat_stds, cache_dir, feats_lookup=feats_lookup)
    val_ds = ECGBenchDataset(val_df, window, config['NB_WINDOWS_EVAL'], leads,
                              top_feat_names, feat_means, feat_stds, cache_dir, feats_lookup=feats_lookup)
    tst_ds = ECGBenchDataset(tst_df, window, config['NB_WINDOWS_EVAL'], leads,
                              top_feat_names, feat_means, feat_stds, cache_dir, feats_lookup=feats_lookup)

    if feats_lookup is not None:
        for name, ds in (('train', trn_ds), ('val', val_ds), ('test', tst_ds)):
            cov = ds.df.Patient.isin(feats_lookup).mean() * 100
            print(f'[{tag}] покрытие кэшем фич, {name}: {cov:.1f}% записей')

    trn_loader = make_loader(trn_ds, config['BATCH_SIZE'], True, config['NUM_WORKERS'])
    val_loader = make_loader(val_ds, config['BATCH_SIZE'], False, config['NUM_WORKERS'])
    tst_loader = make_loader(tst_ds, config['BATCH_SIZE'], False, config['NUM_WORKERS'])

    age_mean, age_std = float(trn_df.Age.mean()), float(trn_df.Age.std())
    model = build_model(leads, top_feat_names, device, seed=config.get('SEED'))

    history = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        model, _ = load_pretrained(model, checkpoint_path, device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, config['D_MODEL'], factor=1, warmup=4000)
        scaler = torch.amp.GradScaler(device.type) if config['USE_AMP'] and device.type == 'cuda' else None

        best_auroc, best_state, patience_count = 0., None, 0
        history = []
        for epoch in range(config['N_EPOCHS']):
            trn_loss = train_one_epoch(model, trn_loader, scheduler, age_mean, age_std, device, scaler)
            val_probs, val_lbls = get_probs(model, val_loader, age_mean, age_std, device)
            from eval.evaluate_12ECG_score import compute_auc
            val_auroc, _ = compute_auc(val_lbls, val_probs)
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
        history = pd.DataFrame(history)

    val_probs, val_lbls = get_probs(model, val_loader, age_mean, age_std, device)
    thrs = find_best_thresholds(val_probs, val_lbls, weights_matrix)
    val_metrics = evaluate_full(val_probs, val_lbls, thrs, weights_matrix)
    tst_probs, tst_lbls = get_probs(model, tst_loader, age_mean, age_std, device)
    tst_metrics = evaluate_full(tst_probs, tst_lbls, thrs, weights_matrix)

    return {
        'tag': tag, 'leads': leads, 'thrs': thrs.tolist(),
        'history': history.to_dict('records') if history is not None else None,
        'val_metrics': val_metrics, 'tst_metrics': tst_metrics,
    }, model
