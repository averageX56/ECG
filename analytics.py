"""
analytics.py
Аналитика датасета: распределение по классам и поиск шумных записей.

  class_distribution / dataset_summary — по таблице build_record_table.
  assess_quality / quality_table / find_noisy_records — оценка качества сигнала
    через neurokit2.ecg_quality (zhao2018 + averageQRS).
Функции plot_* возвращают matplotlib.Figure (для ноутбука).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ecg_data import CLASSES, class_names, class_abbrs, read_record, lead_index


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    names, abbrs = class_names(), class_abbrs()
    n = len(df)
    rows = [{'code': c, 'abbr': abbrs.get(c, c), 'name': names.get(c, c),
             'count': int(df[c].sum()) if c in df.columns else 0,
             'prevalence': (int(df[c].sum()) / n if (n and c in df.columns) else 0.0)}
            for c in CLASSES]
    return pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)


def labels_per_record(df: pd.DataFrame) -> pd.Series:
    present = [c for c in CLASSES if c in df.columns]
    return df[present].sum(axis=1).astype(int)


def dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    lpr = labels_per_record(df)
    normal = '426783006'
    return {
        'n_records': len(df),
        'n_datasets': int(df['dataset'].nunique()) if 'dataset' in df else 1,
        'age_mean': float(np.nanmean(df['age'])) if 'age' in df else None,
        'sex_counts': df['sex'].value_counts().to_dict() if 'sex' in df else {},
        'labels_per_record_mean': float(lpr.mean()) if len(lpr) else 0.0,
        'multilabel_fraction': float((lpr > 1).mean()) if len(lpr) else 0.0,
        'no_label_fraction': float((lpr == 0).mean()) if len(lpr) else 0.0,
        'normal_fraction': float(df[normal].mean()) if normal in df.columns else None,
    }


def assess_quality(hea_path, lead='II', fs_out=500) -> Dict[str, object]:
    """Качество записи через neurokit2: метка zhao2018 + средний averageQRS-скор."""
    import neurokit2 as nk
    out = {'record_id': None, 'quality_label': None, 'quality_score': np.nan, 'error': None}
    try:
        rec = read_record(hea_path, fs_out=fs_out)
        out['record_id'] = rec.record_id
        sig = rec.signal[lead_index(rec.sig_names, lead)].astype(float)
        cleaned = nk.ecg_clean(sig, sampling_rate=int(rec.fs))
        try:
            out['quality_label'] = nk.ecg_quality(cleaned, sampling_rate=int(rec.fs), method='zhao2018')
        except Exception:
            pass
        try:
            q = nk.ecg_quality(cleaned, sampling_rate=int(rec.fs), method='averageQRS')
            out['quality_score'] = float(np.nanmean(np.asarray(q, dtype=float)))
        except Exception:
            pass
    except Exception as e:
        out['error'] = repr(e)
    return out


def quality_table(df: pd.DataFrame, lead='II', limit: Optional[int] = None) -> pd.DataFrame:
    sub = df if limit is None else df.head(limit)
    try:
        from tqdm.auto import tqdm
        it = tqdm(sub.itertuples(index=False), total=len(sub), desc='Оценка качества')
    except Exception:
        it = sub.itertuples(index=False)
    rows = []
    for r in it:
        q = assess_quality(r.filepath, lead=lead)
        q['dataset'] = getattr(r, 'dataset', None)
        rows.append(q)
    return pd.DataFrame(rows)


def find_noisy_records(qdf: pd.DataFrame, n=5) -> pd.DataFrame:
    q = qdf.copy()
    order = {'Unacceptable': 0, 'Barely acceptable': 1, 'Excellent': 2}
    q['_r'] = q['quality_label'].map(order).fillna(1.5)
    q = q.sort_values(['_r', 'quality_score'], na_position='first')
    return q.drop(columns='_r').head(n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Графики
# --------------------------------------------------------------------------
def plot_class_distribution(dist: pd.DataFrame, top: Optional[int] = None):
    import matplotlib.pyplot as plt
    d = dist if top is None else dist.head(top)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(d))))
    ax.barh(d['abbr'][::-1], d['count'][::-1], color='#3b7dd8')
    ax.set_xlabel('Число записей')
    ax.set_title('Распределение по классам (SNOMED, scored)')
    for i, (cnt, prev) in enumerate(zip(d['count'][::-1], d['prevalence'][::-1])):
        ax.text(cnt, i, f' {cnt} ({prev*100:.1f}%)', va='center', fontsize=8)
    fig.tight_layout()
    return fig


def plot_labels_per_record(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    vc = labels_per_record(df).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vc.index, vc.values, color='#3b7dd8')
    ax.set_xlabel('Меток на запись'); ax.set_ylabel('Число записей')
    ax.set_title('Мультиметочность записей')
    fig.tight_layout()
    return fig
