"""
ecg_data.py
Данные ЭКГ: классы, чтение записей и таблица записей + простой torch-Dataset.

Всё, что нужно, извлекается из самого датасета (папка data/):
  - сигнал        — из пары .hea + .mat (формат PhysioNet/CinC 2020);
  - метки         — из строки #Dx: заголовка .hea;
  - возраст/пол   — из #Age / #Sex;
  - фолды         — детерминированный сплит по хэшу record_id (воспроизводим).

Единственное «невыводимое» — соответствие SNOMED-кодов именам и веса метрики
контеста — лежит в eval/ и здесь только читается (имена классов).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
FS_TARGET = 500
STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# 27 scored-классов PhysioNet/CinC Challenge 2020 (отсортированы — как в метрике)
CLASSES: List[str] = sorted([
    '270492004', '164889003', '164890007', '426627000', '713427006',
    '713426002', '445118002', '39732003', '164909002', '251146004',
    '698252002', '10370003', '284470004', '427172004', '164947007',
    '111975006', '164917005', '47665007', '59118001', '427393009',
    '426177001', '426783006', '427084000', '63593006', '164934002',
    '59931005', '17338001',
])
N_CLASSES = len(CLASSES)
NORMAL_CLASS = '426783006'
EQUIVALENT_CLASSES = [['713427006', '59118001'], ['284470004', '63593006'],
                      ['427172004', '17338001']]


# --------------------------------------------------------------------------
# Имена классов (из eval/dx_mapping_scored.csv)
# --------------------------------------------------------------------------
def _dx_mapping() -> pd.DataFrame:
    df = pd.read_csv(_HERE / 'eval' / 'dx_mapping_scored.csv', dtype={'SNOMED CT Code': str})
    return df.set_index('SNOMED CT Code')


def class_names() -> Dict[str, str]:
    m = _dx_mapping()
    return {c: (str(m.loc[c, 'Dx']) if c in m.index else c) for c in CLASSES}


def class_abbrs() -> Dict[str, str]:
    m = _dx_mapping()
    return {c: (str(m.loc[c, 'Abbreviation']) if c in m.index else c) for c in CLASSES}


# --------------------------------------------------------------------------
# Чтение .hea / .mat
# --------------------------------------------------------------------------
@dataclass
class RawRecord:
    record_id: str
    signal: np.ndarray          # [n_leads, n_samples] float32, мВ, @fs
    fs: float
    sig_names: List[str]
    age: Optional[float]
    sex: Optional[str]
    dx_codes: List[str]
    filepath: str


def parse_header(hea_path):
    """Разбирает .hea: параметры сигнала + #Age/#Sex/#Dx."""
    hea_path = Path(hea_path)
    lines = hea_path.read_text(encoding='utf-8', errors='replace').splitlines()
    sig_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith('#')]
    rec = sig_lines[0].split()
    record_id, n_sig, fs = rec[0], int(rec[1]), float(rec[2])

    sig_names, gains, baselines = [], [], []
    for i in range(1, n_sig + 1):
        parts = sig_lines[i].split()
        gain = parts[2].split('/')[0].split('(')[0]
        gains.append(float(gain) if gain and float(gain) != 0 else 1.0)
        baselines.append(float(parts[4]) if len(parts) > 4 else 0.0)
        sig_names.append(parts[-1])

    age, sex, dx = None, None, []
    for ln in lines:
        s, low = ln.strip(), ln.strip().lower()
        if low.startswith('#age'):
            m = re.search(r'(\d+)', s)
            age = float(m.group(1)) if m else None
        elif low.startswith('#sex'):
            sex = s.split(':', 1)[1].strip() if ':' in s else None
        elif low.startswith('#dx'):
            raw = s.split(':', 1)[1] if ':' in s else ''
            dx = [c.strip() for c in raw.split(',') if c.strip()]
    return dict(record_id=record_id, n_sig=n_sig, fs=fs, sig_names=sig_names,
                gains=gains, baselines=baselines, age=age, sex=sex, dx_codes=dx)


def _resample(sig, fs_in, fs_out):
    if int(fs_in) == int(fs_out):
        return sig.astype(np.float32)
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(int(fs_out), int(round(fs_in))).limit_denominator(1000)
    return resample_poly(sig.astype(np.float64), fr.numerator, fr.denominator, axis=-1).astype(np.float32)


def read_record(hea_path, fs_out: int = FS_TARGET) -> RawRecord:
    """Читает пару .hea+.mat -> RawRecord (сигнал в мВ, ресэмпл до fs_out)."""
    import scipy.io as sio
    hea_path = Path(hea_path)
    h = parse_header(hea_path)
    mat = sio.loadmat(str(hea_path.with_suffix('.mat')))
    raw = np.asarray(mat['val'], dtype=np.float64)
    gains = np.asarray(h['gains']).reshape(-1, 1)
    baselines = np.asarray(h['baselines']).reshape(-1, 1)
    sig = (raw - baselines) / gains
    fs = h['fs']
    if int(fs) != int(fs_out):
        sig = _resample(sig, fs, fs_out)
        fs = float(fs_out)
    sig = np.nan_to_num(sig).astype(np.float32)
    return RawRecord(h['record_id'], sig, fs, h['sig_names'], h['age'], h['sex'],
                     h['dx_codes'], str(hea_path))


def lead_index(sig_names, name):
    low = [s.lower() for s in sig_names]
    return low.index(name.lower()) if name.lower() in low else 0


# --------------------------------------------------------------------------
# Таблица записей (метки/возраст/пол — из заголовков, фолды — из хэша id)
# --------------------------------------------------------------------------
def find_records(data_root) -> List[Path]:
    data_root = Path(data_root)
    return [h for h in sorted(data_root.rglob('*.hea')) if h.with_suffix('.mat').exists()]


def _fold_of(record_id: str, n_folds: int = 10) -> int:
    """Детерминированный фолд 0..n_folds-1 по хэшу id (воспроизводимый сплит)."""
    return int(hashlib.md5(record_id.encode()).hexdigest(), 16) % n_folds


def build_record_table(data_root, limit: Optional[int] = None) -> pd.DataFrame:
    """Сканирует data_root -> DataFrame: record_id, filepath, dataset, age, sex,
    dx_codes, fold, + 27 one-hot столбцов CLASSES."""
    heas = find_records(data_root)
    if limit:
        heas = heas[:limit]
    if not heas:
        raise FileNotFoundError(f'Под {Path(data_root).resolve()} нет пар .hea+.mat')
    rows = []
    for hea in heas:
        h = parse_header(hea)
        rid = h['record_id']
        onehot = {c: (1 if c in h['dx_codes'] else 0) for c in CLASSES}
        row = {
            'record_id': rid, 'filepath': str(hea), 'dataset': hea.parent.name,
            'age': h['age'] if h['age'] is not None else np.nan,
            'sex': h['sex'] if h['sex'] else 'Unknown',
            'dx_codes': h['dx_codes'], 'fold': _fold_of(rid),
        }
        row.update(onehot)
        rows.append(row)
    return pd.DataFrame(rows)


def make_split(df: pd.DataFrame, val_fold: int = 8, test_fold: int = 9):
    """Делит таблицу по столбцу fold (0..9): test=test_fold, val=val_fold, остальное train."""
    test_df = df[df['fold'] == test_fold].reset_index(drop=True)
    val_df = df[df['fold'] == val_fold].reset_index(drop=True)
    train_df = df[~df['fold'].isin([val_fold, test_fold])].reset_index(drop=True)
    return train_df, val_df, test_df


# --------------------------------------------------------------------------
# Предобработка сигнала
# --------------------------------------------------------------------------
def bandpass(sig, fs=500.0, band=(3.0, 45.0)):
    """FIR-полосовой фильтр [n_leads, N] (как в исходном решении)."""
    from scipy.signal import firwin, filtfilt
    order = int(0.3 * fs)
    if order % 2 == 0:
        order += 1
    taps = firwin(order, [band[0], band[1]], pass_zero=False, fs=fs)
    return filtfilt(taps, [1.0], sig, axis=-1)


def normalize_pm1(seq, eps=1e-8):
    """Каждое отведение -> диапазон [-1, 1]."""
    mn = seq.min(axis=1, keepdims=True)
    mx = seq.max(axis=1, keepdims=True)
    return 2 * (seq - mn) / (mx - mn + eps) - 1


# --------------------------------------------------------------------------
# torch-Dataset (простой; опциональный дисковый кэш сигнала; без мультипроцессинга)
# --------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import Dataset as _Dataset
    _TORCH = True
except Exception:
    _TORCH = False
    _Dataset = object


class ECGDataset(_Dataset):
    """Отдаёт (windows[nb_windows, n_leads, window], feats, label[27], age, sex).

    features : dict {record_id -> np.ndarray} нормализуемых ручных фич (пайплайн 3)
        или None (пайплайн 2, DL-only).
    feat_means/feat_stds : нормировка ручных фич (обязательны, если features задан).
    cache_dir : если указан, отфильтрованный+нормированный сигнал [12, L]
        кэшируется на диск (.npy) — первая эпоха заполняет, дальше читает готовое.
    """

    def __init__(self, df, window, nb_windows, leads='all',
                 features: Optional[dict] = None,
                 feat_means=None, feat_stds=None, cache_dir: Optional[str] = None,
                 band=(3.0, 45.0), augment=True, fs_out=FS_TARGET, seed=42):
        if not _TORCH:
            raise RuntimeError('Нужен PyTorch.')
        self.df = df.reset_index(drop=True)
        self.window = window
        self.nb_windows = nb_windows
        self.leads = leads
        self.features = features or {}
        self.feat_means = feat_means
        self.feat_stds = feat_stds
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.band = band
        self.augment = augment
        self.fs_out = fs_out
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _load_signal(self, row):
        """Возвращает предобработанный [12, L] float32 (из кэша или считает)."""
        rid = row['record_id']
        if self.cache_dir:
            p = os.path.join(self.cache_dir, f'{rid}.npy')
            if os.path.exists(p):
                try:
                    return np.load(p).astype(np.float32)
                except Exception:
                    pass
        rec = read_record(row['filepath'], fs_out=self.fs_out)
        sig = rec.signal
        if self.band is not None:
            sig = bandpass(sig, rec.fs, self.band)
        sig = normalize_pm1(sig).astype(np.float32)
        if self.cache_dir:
            try:
                p = os.path.join(self.cache_dir, f'{rid}.npy')
                tmp = p + f'.tmp{os.getpid()}'
                np.save(tmp, sig.astype(np.float16))
                os.replace(tmp, p)
            except Exception:
                pass
        return sig

    def _select_leads(self, sig):
        if self.leads == 'all':
            return sig
        if self.leads == 'lead1':
            return sig[0:1]
        if isinstance(self.leads, int):
            return sig[self.leads:self.leads + 1]
        return sig[lead_index(STANDARD_LEADS, str(self.leads)):][0:1]

    def _windows(self, sig):
        n = sig.shape[-1]
        if n < self.window:
            sig = np.pad(sig, ((0, 0), (0, self.window - n + 1)))
            n = sig.shape[-1]
        hi = n - self.window + 1
        if self.augment:
            starts = self._rng.integers(0, hi, size=self.nb_windows)
        else:
            starts = np.linspace(0, hi - 1, self.nb_windows).astype(int)
        return np.stack([sig[:, s:s + self.window] for s in starts]).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = self._select_leads(self._load_signal(row))
        segs = self._windows(sig)

        if self.feat_means is not None:          # режим с ручными фичами (пайплайн 3)
            raw = self.features.get(row['record_id'])
            f = (np.asarray(raw, dtype=np.float64).copy() if raw is not None
                 else self.feat_means.copy())    # нет записи в CSV -> берём средние
            f[np.isinf(f)] = np.nan
            m = np.isnan(f)
            f[m] = self.feat_means[m]
            feats = ((f - self.feat_means) / self.feat_stds).astype(np.float32)
        else:                                    # DL-only (пайплайн 2)
            feats = np.zeros(0, dtype=np.float32)

        label = np.array([float(row.get(c, 0)) for c in CLASSES], dtype=np.float32)
        age = float(row['age']) if pd.notna(row['age']) else 60.0
        sex = 1.0 if str(row['sex']).lower().startswith('f') else 0.0
        return (torch.from_numpy(segs), torch.from_numpy(feats),
                torch.from_numpy(label), torch.tensor(age, dtype=torch.float32),
                torch.tensor(sex, dtype=torch.float32))
