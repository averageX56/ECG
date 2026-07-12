"""
delineation.py
Пайплайн 1: делинеация ЭКГ (neurokit2) -> rule-based диагноз по интервалам ->
метрики совпадения с реальными метками + визуализация разметки.

Метрика качества — precision/recall/F1 диагноза, поставленного ТОЛЬКО по
геометрии интервалов (rr/pq/qrs/qt), против реальных SNOMED-меток записи.
Всё считается обычным последовательным проходом (без мультипроцессинга).

CLI:
  python delineation.py --data-root data --delineate-method dwt --plot-dir outputs/plots
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import neurokit2 as nk

from ecg_data import read_record


# --------------------------------------------------------------------------
# 1. Настраиваемый пайплайн делинеации
# --------------------------------------------------------------------------
@dataclass
class DelineationPipeline:
    """Методы 3 шагов neurokit (можно менять по имени или своей функцией).
      clean_method:     'neurokit' | 'biosppy' | 'pantompkins1985' | ...
      peaks_method:     'neurokit' | 'pantompkins1985' | 'hamilton2002' | ...
      delineate_method: 'dwt' | 'peak' | 'cwt'
    """
    clean_method: str = 'neurokit'
    peaks_method: str = 'neurokit'
    delineate_method: str = 'dwt'
    clean_fn: Optional[Callable] = None
    peaks_fn: Optional[Callable] = None
    delineate_fn: Optional[Callable] = None


DEFAULT_PIPELINE = DelineationPipeline()


def _run_steps(signal, fs, pipe: DelineationPipeline):
    cleaned = (pipe.clean_fn(signal, fs) if pipe.clean_fn
               else nk.ecg_clean(signal, sampling_rate=fs, method=pipe.clean_method))
    cleaned = np.asarray(cleaned, dtype=float)
    if pipe.peaks_fn:
        r_peaks = np.asarray(pipe.peaks_fn(cleaned, fs), dtype=int)
    else:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=pipe.peaks_method)
        r_peaks = np.asarray(info.get('ECG_R_Peaks', []), dtype=int)
    if len(r_peaks) < 3:
        return cleaned, r_peaks, {}
    if pipe.delineate_fn:
        waves = pipe.delineate_fn(cleaned, r_peaks, fs)
    else:
        _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs, method=pipe.delineate_method)
    return cleaned, r_peaks, waves


def delineate_full(rec: dict, pipeline: Optional[DelineationPipeline] = None,
                   lead_index: int = 0, fs_out: int = 500) -> dict:
    """Делинеация одной записи (rec — строка build_record_table, нужен filepath)."""
    pipe = pipeline or DEFAULT_PIPELINE
    out = {'record_id': rec.get('record_id'), 'dataset': rec.get('dataset'),
           'signal': None, 'fs': None, 'lead_name': None,
           'r_peaks': [], 'p_peaks': [], 'q_peaks': [], 's_peaks': [], 't_peaks': [],
           'p_onsets': [], 'qrs_onsets': [], 'qrs_offsets': [], 't_offsets': [],
           'quality_label': None, 'error': None}
    try:
        record = read_record(rec['filepath'], fs_out=fs_out)
        fs = int(record.fs)
        idx = lead_index if lead_index < len(record.sig_names) else 0
        signal = record.signal[idx].astype(float)
        if len(signal) < fs * 2:
            out['error'] = 'too_short'
            return out
        cleaned, r_peaks, waves = _run_steps(signal, fs, pipe)
        r_peaks = [int(x) for x in r_peaks]
        if len(r_peaks) < 3:
            out['error'] = 'not_enough_rpeaks'
            return out

        def ints(key):
            return [int(x) for x in waves.get(key, []) if not (isinstance(x, float) and np.isnan(x))]

        try:
            ql = nk.ecg_quality(cleaned, sampling_rate=fs, method='zhao2018')
        except Exception:
            ql = None
        out.update({'signal': cleaned, 'fs': fs, 'lead_name': record.sig_names[idx],
                    'r_peaks': r_peaks, 'p_peaks': ints('ECG_P_Peaks'), 'q_peaks': ints('ECG_Q_Peaks'),
                    's_peaks': ints('ECG_S_Peaks'), 't_peaks': ints('ECG_T_Peaks'),
                    'p_onsets': ints('ECG_P_Onsets'), 'qrs_onsets': ints('ECG_R_Onsets'),
                    'qrs_offsets': ints('ECG_R_Offsets'), 't_offsets': ints('ECG_T_Offsets'),
                    'quality_label': ql})
    except Exception as e:
        out['error'] = repr(e)
    return out


# --------------------------------------------------------------------------
# 2. Интервалы и пороговые правила
# --------------------------------------------------------------------------
DX_GROUPS = {
    'sinus_tachycardia': ['427084000'], 'sinus_bradycardia': ['427393003', '426177001'],
    'bradycardia_general': ['426627000'], 'sinus_arrhythmia': ['713422000', '427393009'],
    'av_block_1st_degree': ['270492004'], 'short_pr': ['49578007'], 'prolonged_pr': ['164947007'],
    'complete_rbbb': ['713427006'], 'incomplete_rbbb': ['713426002'], 'lbbb': ['164909002'],
    'incomplete_lbbb': ['251120003'], 'bundle_branch_block_general': ['6374002'],
    'nonspecific_ivcd': ['428750005'], 'diffuse_ivcd': ['82226007'],
    'prolonged_qt': ['111975006'], 'short_qt': ['77867006'],
}
INTERVAL_LABELS = list(DX_GROUPS)


@dataclass
class Thresholds:
    rr_tachy_max_ms: float = 600.0
    rr_brady_min_ms: float = 1000.0
    rr_arrhythmia_abs_ms: float = 120.0
    rr_arrhythmia_rel: float = 0.10
    pq_long_min_ms: float = 200.0
    pq_short_max_ms: float = 120.0
    qrs_incomplete_min_ms: float = 110.0
    qrs_incomplete_max_ms: float = 120.0
    qrs_complete_min_ms: float = 120.0
    qrs_nonspecific_min_ms: float = 110.0
    qtc_long_male_ms: float = 450.0
    qtc_long_female_ms: float = 460.0
    qtc_short_min_ms: float = 330.0


THRESH = Thresholds()


def bazett_qtc(qt_ms, rr_ms):
    if qt_ms is None or rr_ms is None or rr_ms <= 0:
        return None
    return qt_ms / math.sqrt(rr_ms / 1000.0)


def _cycle_intervals_ms(delin: dict) -> dict:
    """Интервалы по каждому циклу (для медиан и межцикловых девиаций)."""
    fs, r = delin.get('fs'), delin.get('r_peaks') or []
    if not fs or len(r) < 2:
        return {'rr': np.array([]), 'qrs': np.array([]), 'qt': np.array([]), 'pq': np.array([])}
    r = np.asarray(r, dtype=float)
    rr = np.diff(r) / fs * 1000.0

    def in_range(points, lo, hi):
        p = np.asarray(points, dtype=float)
        return p[(p >= lo) & (p < hi)]

    p_on, qrs_on = delin.get('p_onsets') or [], delin.get('qrs_onsets') or []
    qrs_off, t_off = delin.get('qrs_offsets') or [], delin.get('t_offsets') or []
    qrs_l, qt_l, pq_l = [], [], []
    for i in range(len(r) - 1):
        lo, hi = r[i] - 0.5 * fs, r[i + 1] + 0.1 * fs
        won, woff = in_range(qrs_on, lo, hi), in_range(qrs_off, lo, hi)
        ton, pon = in_range(t_off, lo, hi), in_range(p_on, lo, hi)
        if len(won) and len(woff):
            onset = won[np.argmin(np.abs(won - r[i]))]
            offs = woff[woff > onset]
            if len(offs):
                qrs_l.append((offs[0] - onset) / fs * 1000.0)
        if len(won) and len(ton):
            onset = won[np.argmin(np.abs(won - r[i]))]
            offs = ton[ton > onset]
            if len(offs):
                qt_l.append((offs[0] - onset) / fs * 1000.0)
        if len(pon) and len(won):
            oq = won[np.argmin(np.abs(won - r[i]))]
            pons = pon[pon < oq]
            if len(pons):
                pq_l.append((oq - pons[-1]) / fs * 1000.0)
    return {'rr': rr, 'qrs': np.asarray(qrs_l), 'qt': np.asarray(qt_l), 'pq': np.asarray(pq_l)}


def _dev(arr):
    arr = arr[~np.isnan(arr)] if len(arr) else arr
    if len(arr) == 0:
        return None, None
    med = float(np.median(arr))
    return med, float(np.median(np.abs(arr - med)))


def compute_interval_features(rec: dict, pipeline=None, lead_index=0) -> dict:
    delin = delineate_full(rec, pipeline=pipeline, lead_index=lead_index)
    f = {'record_id': rec.get('record_id'), 'dataset': rec.get('dataset'),
         'sex': rec.get('sex'), 'age': rec.get('age'),
         'quality_label': delin.get('quality_label'),
         'n_beats': len(delin.get('r_peaks') or []),
         'delineation_error': delin.get('error'),
         'rr_ms': None, 'rr_mad_ms': None, 'qrs_ms': None, 'qrs_mad_ms': None,
         'qt_ms': None, 'qt_mad_ms': None, 'pq_ms': None, 'pq_mad_ms': None, 'qtc_ms': None}
    if delin.get('error') is None:
        cyc = _cycle_intervals_ms(delin)
        f['rr_ms'], f['rr_mad_ms'] = _dev(cyc['rr'])
        f['qrs_ms'], f['qrs_mad_ms'] = _dev(cyc['qrs'])
        f['qt_ms'], f['qt_mad_ms'] = _dev(cyc['qt'])
        f['pq_ms'], f['pq_mad_ms'] = _dev(cyc['pq'])
    f['qtc_ms'] = bazett_qtc(f['qt_ms'], f['rr_ms'])
    return f


def diagnose_from_intervals(f: dict, thr: Thresholds = THRESH) -> list:
    labels = []
    is_female = str(f.get('sex') or '').strip().lower().startswith(('f', 'ж'))
    rr = f.get('rr_ms')
    if rr is not None:
        if rr < thr.rr_tachy_max_ms:
            labels.append('sinus_tachycardia')
        if rr > thr.rr_brady_min_ms:
            labels += ['sinus_bradycardia', 'bradycardia_general']
    if f.get('rr_mad_ms') is not None and rr:
        if f['rr_mad_ms'] > thr.rr_arrhythmia_abs_ms or f['rr_mad_ms'] / rr > thr.rr_arrhythmia_rel:
            labels.append('sinus_arrhythmia')
    pq = f.get('pq_ms')
    if pq is not None:
        if pq > thr.pq_long_min_ms:
            labels += ['av_block_1st_degree', 'prolonged_pr']
        if pq < thr.pq_short_max_ms:
            labels.append('short_pr')
    qrs = f.get('qrs_ms')
    if qrs is not None:
        if thr.qrs_incomplete_min_ms <= qrs < thr.qrs_incomplete_max_ms:
            labels += ['incomplete_rbbb', 'incomplete_lbbb']
        if qrs >= thr.qrs_complete_min_ms:
            labels += ['complete_rbbb', 'lbbb', 'bundle_branch_block_general']
        if qrs > thr.qrs_nonspecific_min_ms:
            labels.append('nonspecific_ivcd')
        if qrs >= thr.qrs_complete_min_ms and f.get('qrs_mad_ms') and f['qrs_mad_ms'] > 20:
            labels.append('diffuse_ivcd')
    qtc = f.get('qtc_ms')
    if qtc is not None:
        if qtc > (thr.qtc_long_female_ms if is_female else thr.qtc_long_male_ms):
            labels.append('prolonged_qt')
        if qtc < thr.qtc_short_min_ms:
            labels.append('short_qt')
    return sorted(set(labels))


def ground_truth_labels(dx_codes) -> set:
    dx = set(dx_codes or [])
    return {label for label, codes in DX_GROUPS.items() if dx & set(codes)}


def run_pipeline(df: pd.DataFrame, pipeline=None, lead_index=0, limit=None) -> pd.DataFrame:
    """Обычный последовательный проход по записям -> DataFrame результатов."""
    records = df.to_dict('records')
    if limit:
        records = records[:limit]
    try:
        from tqdm.auto import tqdm
        records = tqdm(records, desc='Делинеация + диагноз')
    except Exception:
        pass
    rows = []
    for rec in records:
        f = compute_interval_features(rec, pipeline=pipeline, lead_index=lead_index)
        pred = set(diagnose_from_intervals(f))
        gt = ground_truth_labels(rec.get('dx_codes', []))
        rows.append({**{k: f[k] for k in ('record_id', 'dataset', 'sex', 'age', 'quality_label',
                                          'n_beats', 'rr_ms', 'rr_mad_ms', 'qrs_ms', 'qrs_mad_ms',
                                          'qt_ms', 'qt_mad_ms', 'pq_ms', 'pq_mad_ms', 'qtc_ms',
                                          'delineation_error')},
                     'predicted_labels': sorted(pred), 'ground_truth_labels': sorted(gt),
                     'true_positive': sorted(pred & gt), 'false_positive': sorted(pred - gt),
                     'false_negative': sorted(gt - pred)})
    return pd.DataFrame(rows)


def interval_feature_table(df: pd.DataFrame, pipeline=None, lead_index=0, limit=None) -> pd.DataFrame:
    """Таблица интервальных фич neurokit по записям: record_id + nk_* столбцы
    (rr/qrs/qt/pq медианы и MAD-девиации, qtc, число ударов). Используется
    бустингом (решение 4) в дополнение к ручным фичам. Обычный последовательный
    проход (делинеация каждой записи через neurokit)."""
    records = df.to_dict('records')
    if limit:
        records = records[:limit]
    try:
        from tqdm.auto import tqdm
        records = tqdm(records, desc='Интервальные фичи (neurokit)')
    except Exception:
        pass
    rows = []
    for rec in records:
        f = compute_interval_features(rec, pipeline=pipeline, lead_index=lead_index)
        rows.append({'record_id': f['record_id'],
                     'nk_rr_ms': f['rr_ms'], 'nk_rr_mad_ms': f['rr_mad_ms'],
                     'nk_qrs_ms': f['qrs_ms'], 'nk_qrs_mad_ms': f['qrs_mad_ms'],
                     'nk_qt_ms': f['qt_ms'], 'nk_qt_mad_ms': f['qt_mad_ms'],
                     'nk_pq_ms': f['pq_ms'], 'nk_pq_mad_ms': f['pq_mad_ms'],
                     'nk_qtc_ms': f['qtc_ms'], 'nk_n_beats': f['n_beats']})
    return pd.DataFrame(rows)


def quality_metrics(results: pd.DataFrame, labels=None) -> pd.DataFrame:
    """precision/recall/F1 по каждой метке + micro-average."""
    labels = labels or INTERVAL_LABELS
    rows = []
    TP = FP = FN = 0
    for label in labels:
        tp = sum(label in r for r in results['true_positive'])
        fp = sum(label in r for r in results['false_positive'])
        fn = sum(label in r for r in results['false_negative'])
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        rows.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn, 'precision': prec,
                     'recall': rec, 'f1': f1, 'support': sum(label in r for r in results['ground_truth_labels'])})
        TP += tp; FP += fp; FN += fn
    mp = TP / (TP + FP) if (TP + FP) else None
    mr = TP / (TP + FN) if (TP + FN) else None
    rows.append({'label': 'MICRO_AVERAGE', 'tp': TP, 'fp': FP, 'fn': FN, 'precision': mp, 'recall': mr,
                 'f1': (2 * mp * mr / (mp + mr)) if (mp and mr) else None,
                 'support': sum(r['support'] for r in rows)})
    return pd.DataFrame(rows)


def delineation_failure_rate(results: pd.DataFrame) -> dict:
    n = len(results)
    nf = int(results['delineation_error'].notna().sum())
    return {'n_records': n, 'n_failed': nf, 'failure_rate': (nf / n if n else None)}


# --------------------------------------------------------------------------
# 3. Визуализация
# --------------------------------------------------------------------------
_WAVES = {'p_peaks': ('#2ca02c', 'o', 'P'), 'q_peaks': ('#9467bd', 'v', 'Q'),
          'r_peaks': ('#d62728', '^', 'R'), 's_peaks': ('#8c564b', 'v', 'S'),
          't_peaks': ('#1f77b4', 'o', 'T')}
_BOUNDS = {'p_onsets': ('#2ca02c', '--', 'P onset'), 'qrs_onsets': ('#d62728', '--', 'QRS onset'),
           'qrs_offsets': ('#d62728', ':', 'QRS offset'), 't_offsets': ('#1f77b4', ':', 'T offset')}


def plot_delineation(delin: dict, seconds: Optional[float] = 5.0, t_start=0.0, show_bounds=True):
    import matplotlib.pyplot as plt
    if delin.get('signal') is None:
        raise ValueError(f"Нет сигнала (error={delin.get('error')})")
    sig = np.asarray(delin['signal'], dtype=float)
    fs = int(delin['fs'])
    i0 = int(t_start * fs)
    i1 = len(sig) if seconds is None else min(len(sig), i0 + int(seconds * fs))
    t = np.arange(i0, i1) / fs
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(t, sig[i0:i1], color='#333', lw=0.8, zorder=1)

    def sel(idxs):
        a = np.asarray(idxs, dtype=int)
        return a[(a >= i0) & (a < i1)]

    for key, (color, marker, lbl) in _WAVES.items():
        pts = sel(delin.get(key, []))
        if len(pts):
            ax.scatter(pts / fs, sig[pts], s=40, color=color, marker=marker, label=lbl, zorder=3)
    if show_bounds:
        seen = set()
        for key, (color, ls, lbl) in _BOUNDS.items():
            for x in sel(delin.get(key, [])):
                ax.axvline(x / fs, color=color, ls=ls, lw=0.8, alpha=0.6, zorder=2,
                           label=lbl if lbl not in seen else None)
                seen.add(lbl)
    q = delin.get('quality_label')
    ax.set_title(f"{delin.get('record_id','?')} — отведение {delin.get('lead_name','?')}"
                 + (f"  |  качество: {q}" if q else ''), fontsize=11)
    ax.set_xlabel('Время, с'); ax.set_ylabel('мВ')
    ax.legend(loc='upper right', ncol=5, fontsize=8, framealpha=0.9)
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig


def plot_record(rec: dict, pipeline=None, lead_index=0, seconds=5.0):
    return plot_delineation(delineate_full(rec, pipeline=pipeline, lead_index=lead_index), seconds=seconds)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Пайплайн 1: делинеация + диагноз по интервалам')
    p.add_argument('--data-root', default='data')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--lead-index', type=int, default=0)
    p.add_argument('--clean-method', default='neurokit')
    p.add_argument('--peaks-method', default='neurokit')
    p.add_argument('--delineate-method', default='dwt', choices=['dwt', 'peak', 'cwt'])
    p.add_argument('--out', default=None, help='CSV per-record результатов')
    p.add_argument('--metrics-out', default=None, help='CSV метрик по меткам')
    p.add_argument('--plot-dir', default=None, help='Куда сохранить примеры разметки (PNG)')
    p.add_argument('--n-plots', type=int, default=6)
    return p.parse_args(argv)


def main(argv=None):
    from ecg_data import build_record_table
    args = parse_args(argv)
    pipe = DelineationPipeline(clean_method=args.clean_method, peaks_method=args.peaks_method,
                               delineate_method=args.delineate_method)
    df = build_record_table(args.data_root, limit=args.limit)
    print(f'Записей: {len(df)}')
    results = run_pipeline(df, pipeline=pipe, lead_index=args.lead_index)

    fail = delineation_failure_rate(results)
    print(f"\nНадёжность делинеации: упало {fail['n_failed']}/{fail['n_records']} "
          f"({(fail['failure_rate'] or 0)*100:.1f}%)")
    metrics = quality_metrics(results)
    print('\n=== Качество (диагноз по интервалам vs реальные метки) ===')
    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print(metrics.to_string(index=False))

    if args.out:
        results.to_csv(args.out, index=False); print(f'Подробности -> {args.out}')
    if args.metrics_out:
        metrics.to_csv(args.metrics_out, index=False); print(f'Метрики -> {args.metrics_out}')
    if args.plot_dir:
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(args.plot_dir, exist_ok=True)
        for _, row in df.head(args.n_plots).iterrows():
            try:
                fig = plot_record(row.to_dict(), pipeline=pipe, lead_index=args.lead_index)
                fig.savefig(os.path.join(args.plot_dir, f"{row['record_id']}.png"), dpi=110)
                plt.close(fig)
            except Exception as e:
                print(f"  [{row['record_id']}] не нарисовать: {e}")
        print(f'Примеры разметки -> {args.plot_dir}')
    return results, metrics


if __name__ == '__main__':
    main()
