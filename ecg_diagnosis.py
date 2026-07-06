# ecg_diagnosis.py
"""
Пайплайн: интервалы (rr/qrs/qt/pq) -> правило-based диагноз по длинам и
девиациям интервалов -> сверка с реальными dx_codes из датасета ->
метрики качества метода delineate (precision/recall/F1 на уровне диагнозов,
+ опционально MAE интервалов против эталонных, если такие есть).

Зависит ТОЛЬКО от:
  - ecg_delineate_full.delineate_full(rec) -> dict с массивами точек разметки
    по каждому циклу (P/Q/R/S/T-пики, onset/offset). И медианы интервалов, и
    их межцикловые девиации считаются здесь, в _cycle_intervals_ms, из ОДНОГО
    и того же набора точек — на одном и том же прочитанном сигнале.

Раньше медианы (rr_ms/qrs_ms/...) считались отдельно через
ecg_worker.process_record(), который читал сигнал через wfdb.rdrecord() БЕЗ
.mat-фоллбека. Для .mat-записей (Georgia/CPSC_Extra/PTB-XL и т.п.) это давало
неверно прочитанный сигнал (gain/baseline не применялись, либо чтение падало
тихо иначе), в то время как делинеация (через delineate_full) читала тот же
файл правильно. В результате медианы и девиации интервалов считались по
РАЗНЫМ версиям сигнала и "расходились" — это и была причина бага. Теперь
ecg_worker.py больше не используется нигде в пайплайне.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ecg_delineate_full import delineate_full, DelineationPipeline, DEFAULT_PIPELINE


# --------------------------------------------------------------------------
# 1. Группы SNOMED-кодов (как в описании, через "/")
# --------------------------------------------------------------------------

# Каждой "правило-метке" сопоставлена ГРУППА реальных кодов из датасета,
# которые засчитываются как совпадение (см. ответ пользователя: по группе).
DX_GROUPS: dict[str, list[str]] = {
    'sinus_tachycardia':            ['427084000'],
    'sinus_bradycardia':            ['427393003', '426177001'],
    'bradycardia_general':          ['426627000'],
    'sinus_arrhythmia':             ['713422000', '427393009'],

    'av_block_1st_degree':          ['270492004'],
    'short_pr':                     ['49578007'],
    'prolonged_pr':                 ['164947007'],  # дублирует av_block_1st_degree по смыслу

    'complete_rbbb':                ['713427006'],
    'incomplete_rbbb':              ['713426002'],
    'lbbb':                         ['164909002'],
    'incomplete_lbbb':              ['251120003'],
    'bundle_branch_block_general':  ['6374002'],
    'nonspecific_ivcd':             ['428750005'],
    'diffuse_ivcd':                 ['82226007'],
    'low_qrs_voltage':              ['251146004'],  # амплитудный критерий, по интервалам НЕ диагностируется

    'prolonged_qt':                 ['111975006'],
    'short_qt':                     ['77867006'],
}

# Какие правило-метки в принципе диагностируемы по длине/девиации интервалов.
# low_qrs_voltage оставлен в DX_GROUPS для полноты словаря кодов, но исключён
# из набора диагностируемых меток, т.к. это амплитудный критерий, не интервальный.
INTERVAL_DIAGNOSABLE = [k for k in DX_GROUPS if k != 'low_qrs_voltage']


# --------------------------------------------------------------------------
# 2. Пороговые правила
# --------------------------------------------------------------------------

@dataclass
class Thresholds:
    # RR (мс)
    rr_tachy_max_ms: float = 600.0       # RR < 600 мс -> ЧСС > 100
    rr_brady_min_ms: float = 1000.0      # RR > 1000 мс -> ЧСС < 60
    # синусовая аритмия: вариабельность RR между соседними циклами
    rr_arrhythmia_abs_ms: float = 120.0  # > 120 мс
    rr_arrhythmia_rel: float = 0.10      # или > 10% от среднего RR

    # PQ/PR (мс)
    pq_long_min_ms: float = 200.0        # PQ > 200 -> AV block I / prolonged PR
    pq_short_max_ms: float = 120.0       # PQ < 120 -> shortened PR

    # QRS (мс)
    qrs_incomplete_min_ms: float = 110.0
    qrs_incomplete_max_ms: float = 120.0
    qrs_complete_min_ms: float = 120.0
    qrs_nonspecific_min_ms: float = 110.0  # > 110 без типичной морфологии блокады

    # QTc (мс), формула Bazett: QTc = QT / sqrt(RR_seconds)
    qtc_long_male_ms: float = 450.0
    qtc_long_female_ms: float = 460.0    # верхняя граница диапазона 460-470 берём консервативно
    qtc_short_min_ms: float = 330.0      # нижняя граница диапазона 330-360, консервативно


THRESH = Thresholds()


def bazett_qtc(qt_ms: Optional[float], rr_ms: Optional[float]) -> Optional[float]:
    """QTc по Bazett: QTc[ms] = QT[ms] / sqrt(RR[s])."""
    if qt_ms is None or rr_ms is None or rr_ms <= 0:
        return None
    rr_s = rr_ms / 1000.0
    return qt_ms / math.sqrt(rr_s)


# --------------------------------------------------------------------------
# 3. Девиации интервалов (cycle-to-cycle), на основе точек delineate_full
# --------------------------------------------------------------------------

def _cycle_intervals_ms(delin: dict) -> dict[str, np.ndarray]:
    """Считает интервал ПО КАЖДОМУ циклу (не медиану), сопоставляя точки
    внутри одного R-R окна, чтобы корректно посчитать девиацию между циклами.

    В отличие от ecg_worker.process_record (который просто берёт arr[:n] по
    позиционному индексу), здесь точки P/QRS/T сопоставляются с конкретным
    R-зубцом по принадлежности интервалу (ближайшая точка справа от R[i] и
    слева от R[i+1]), что даёт более честную оценку межцикловой девиации.
    """
    fs = delin.get('fs')
    r_peaks = delin.get('r_peaks') or []
    if not fs or len(r_peaks) < 2:
        return {'rr': np.array([]), 'qrs': np.array([]), 'qt': np.array([]), 'pq': np.array([])}

    r = np.asarray(r_peaks, dtype=float)
    rr_ms = np.diff(r) / fs * 1000.0

    def nearest_in_range(points, lo, hi):
        pts = np.asarray(points, dtype=float)
        pts = pts[(pts >= lo) & (pts < hi)]
        return pts

    p_on = delin.get('p_onsets') or []
    qrs_on = delin.get('qrs_onsets') or []
    qrs_off = delin.get('qrs_offsets') or []
    t_off = delin.get('t_offsets') or []

    qrs_list, qt_list, pq_list = [], [], []
    for i in range(len(r) - 1):
        lo, hi = r[i] - 0.5 * fs, r[i + 1] + 0.1 * fs  # окно вокруг цикла с запасом
        won = nearest_in_range(qrs_on, lo, hi)
        woff = nearest_in_range(qrs_off, lo, hi)
        ton = nearest_in_range(t_off, lo, hi)
        pon = nearest_in_range(p_on, lo, hi)

        if len(won) and len(woff):
            # берём ближайшую пару onset/offset с offset > onset
            onset = won[np.argmin(np.abs(won - r[i]))]
            offs = woff[woff > onset]
            if len(offs):
                qrs_list.append((offs[0] - onset) / fs * 1000.0)

        if len(won) and len(ton):
            onset = won[np.argmin(np.abs(won - r[i]))]
            offs = ton[ton > onset]
            if len(offs):
                qt_list.append((offs[0] - onset) / fs * 1000.0)

        if len(pon) and len(won):
            onset_qrs = won[np.argmin(np.abs(won - r[i]))]
            pons = pon[pon < onset_qrs]
            if len(pons):
                pq_list.append((onset_qrs - pons[-1]) / fs * 1000.0)

    return {
        'rr': rr_ms,
        'qrs': np.asarray(qrs_list),
        'qt': np.asarray(qt_list),
        'pq': np.asarray(pq_list),
    }


def _dev_stats(arr: np.ndarray) -> dict:
    """Медиана, MAD (median absolute deviation) и относительная девиация."""
    arr = arr[~np.isnan(arr)] if len(arr) else arr
    if len(arr) == 0:
        return {'median_ms': None, 'mad_ms': None, 'rel_dev': None, 'n': 0}
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    rel = (mad / med) if med else None
    return {'median_ms': med, 'mad_ms': mad, 'rel_dev': rel, 'n': int(len(arr))}


@dataclass
class IntervalFeatures:
    record_id: str
    dataset: Optional[str]
    age: Optional[float] = None
    sex: Optional[str] = None

    rr_ms: Optional[float] = None
    qrs_ms: Optional[float] = None
    qt_ms: Optional[float] = None
    pq_ms: Optional[float] = None
    qtc_ms: Optional[float] = None

    rr_mad_ms: Optional[float] = None
    qrs_mad_ms: Optional[float] = None
    qt_mad_ms: Optional[float] = None
    pq_mad_ms: Optional[float] = None

    n_beats: int = 0
    delineation_error: Optional[str] = None
    worker_error: Optional[str] = None


def compute_interval_features(rec: dict, n_samples: Optional[int] = None,
                               pipeline: Optional[DelineationPipeline] = None,
                               lead_index: Optional[int] = 0) -> IntervalFeatures:
    """Считает интервалы (медианы) И девиации (MAD) из ОДНОГО вызова
    delineate_full — оба считаются на одном и том же прочитанном сигнале и
    одном и том же наборе точек разметки (_cycle_intervals_ms), что устраняет
    рассинхрон, который раньше возникал из-за параллельного использования
    ecg_worker.process_record с другим (некорректным для .mat) чтением сигнала.

    rec: словарь записи из df (record_id, dataset, filepath, age, sex, ...).
    n_samples: если задано, ограничивает число используемых сердечных циклов
        (берутся первые n_samples R-R циклов, а не первые n_samples сэмплов
        сигнала) — например, n_samples=10 значит "первые ~10 ударов".
    pipeline: опциональная DelineationPipeline — позволяет подменить
        clean/peaks/delineate шаги neurokit по отдельности (см.
        ecg_delineate_full.DelineationPipeline), не трогая остальной код.
    lead_index: индекс отведения (по умолчанию 0 — первое отведение записи).
        Передайте None, чтобы вернуться к выбору по списку предпочтительных
        имён (II/MLII/I/...) через lead_pref в delineate_full.
    """
    delin = delineate_full(rec, pipeline=pipeline or DEFAULT_PIPELINE, lead_index=lead_index)

    feats = IntervalFeatures(
        record_id=rec['record_id'],
        dataset=rec.get('dataset'),
        age=rec.get('age'),
        sex=rec.get('sex'),
        n_beats=len(delin.get('r_peaks') or []),
        delineation_error=delin.get('error'),
    )

    if delin.get('error') is None:
        if n_samples is not None and delin.get('r_peaks'):
            delin = dict(delin)
            delin['r_peaks'] = delin['r_peaks'][: n_samples + 1]
            feats.n_beats = len(delin['r_peaks'])

        cyc = _cycle_intervals_ms(delin)

        rr = _dev_stats(cyc['rr']);   feats.rr_ms,  feats.rr_mad_ms  = rr['median_ms'],  rr['mad_ms']
        qrs = _dev_stats(cyc['qrs']); feats.qrs_ms, feats.qrs_mad_ms = qrs['median_ms'], qrs['mad_ms']
        qt = _dev_stats(cyc['qt']);   feats.qt_ms,  feats.qt_mad_ms  = qt['median_ms'],  qt['mad_ms']
        pq = _dev_stats(cyc['pq']);   feats.pq_ms,  feats.pq_mad_ms  = pq['median_ms'],  pq['mad_ms']

    feats.qtc_ms = bazett_qtc(feats.qt_ms, feats.rr_ms)
    return feats


# --------------------------------------------------------------------------
# 4. Диагностика по правилам
# --------------------------------------------------------------------------

def diagnose_from_intervals(f: IntervalFeatures, thr: Thresholds = THRESH) -> list[str]:
    """Возвращает список сработавших правило-меток (ключи DX_GROUPS, кроме
    low_qrs_voltage). Несколько меток могут сработать одновременно (например,
    sinus_tachycardia и prolonged_pr).
    """
    labels: list[str] = []
    sex = (f.sex or '').strip().lower()
    is_female = sex.startswith('f') or sex.startswith('ж')

    # --- RR ---
    if f.rr_ms is not None:
        if f.rr_ms < thr.rr_tachy_max_ms:
            labels.append('sinus_tachycardia')
        if f.rr_ms > thr.rr_brady_min_ms:
            labels.append('sinus_bradycardia')
            labels.append('bradycardia_general')

    if f.rr_mad_ms is not None and f.rr_ms:
        rel = f.rr_mad_ms / f.rr_ms
        if f.rr_mad_ms > thr.rr_arrhythmia_abs_ms or rel > thr.rr_arrhythmia_rel:
            labels.append('sinus_arrhythmia')

    # --- PQ ---
    if f.pq_ms is not None:
        if f.pq_ms > thr.pq_long_min_ms:
            labels.append('av_block_1st_degree')
            labels.append('prolonged_pr')
        if f.pq_ms < thr.pq_short_max_ms:
            labels.append('short_pr')

    # --- QRS ---
    if f.qrs_ms is not None:
        if thr.qrs_incomplete_min_ms <= f.qrs_ms < thr.qrs_incomplete_max_ms:
            labels.append('incomplete_rbbb')
            labels.append('incomplete_lbbb')
        if f.qrs_ms >= thr.qrs_complete_min_ms:
            labels.append('complete_rbbb')
            labels.append('lbbb')
            labels.append('bundle_branch_block_general')
        if f.qrs_ms > thr.qrs_nonspecific_min_ms:
            labels.append('nonspecific_ivcd')
        # диффузное уширение - эвристика: явно широкий QRS с высокой межцикловой девиацией
        if f.qrs_ms >= thr.qrs_complete_min_ms and f.qrs_mad_ms and f.qrs_mad_ms > 20:
            labels.append('diffuse_ivcd')

    # --- QTc (Bazett) ---
    if f.qtc_ms is not None:
        long_thr = thr.qtc_long_female_ms if is_female else thr.qtc_long_male_ms
        if f.qtc_ms > long_thr:
            labels.append('prolonged_qt')
        if f.qtc_ms < thr.qtc_short_min_ms:
            labels.append('short_qt')

    return sorted(set(labels))


# --------------------------------------------------------------------------
# 5. Сверка с реальными dx_codes (ground truth) — по группам кодов
# --------------------------------------------------------------------------

def ground_truth_labels(dx_codes: list[str]) -> set[str]:
    """Переводит реальные dx_codes записи в набор правило-меток, с которыми
    они сопоставляются (по группе, см. DX_GROUPS)."""
    dx_set = set(dx_codes or [])
    hits = set()
    for label, codes in DX_GROUPS.items():
        if label == 'low_qrs_voltage':
            continue
        if dx_set & set(codes):
            hits.add(label)
    return hits


def evaluate_against_dataset(rec: dict, predicted_labels: list[str]) -> dict:
    """Сравнивает предсказанные по интервалам метки с ground truth dx_codes
    этой же записи. Возвращает TP/FP/FN метки и булевы признаки по каждой
    диагностируемой метке."""
    gt = ground_truth_labels(rec.get('dx_codes', []))
    pred = set(predicted_labels)

    tp = pred & gt
    fp = pred - gt
    fn = gt - pred

    return {
        'record_id': rec['record_id'],
        'predicted': sorted(pred),
        'ground_truth': sorted(gt),
        'true_positive': sorted(tp),
        'false_positive': sorted(fp),
        'false_negative': sorted(fn),
    }


# --------------------------------------------------------------------------
# 6. Сводный пайплайн по DataFrame записей
# --------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, n_samples: Optional[int] = None, limit: Optional[int] = None,
                  pipeline: Optional[DelineationPipeline] = None,
                  lead_index: Optional[int] = 0) -> pd.DataFrame:
    """Прогоняет весь пайплайн (1-3) по df записей (как в ноутбуке: df с
    record_id/dataset/filepath/age/sex/dx_codes).

    pipeline: опциональная DelineationPipeline для подмены отдельных шагов
    neurokit (clean/peaks/delineate) без изменения остального кода — см.
    ecg_delineate_full.DelineationPipeline.
    lead_index: индекс отведения (по умолчанию 0 — первое отведение записи).

    Возвращает DataFrame с одной строкой на запись: интервалы, девиации,
    предсказанные метки, ground truth, TP/FP/FN.
    """
    records = df.to_dict('records')
    if limit:
        records = records[:limit]

    try:
        from tqdm.auto import tqdm
        records_iter = tqdm(records, desc='ECG diagnosis pipeline')
    except ImportError:
        records_iter = records

    rows = []
    for rec in records_iter:
        feats = compute_interval_features(rec, n_samples=n_samples, pipeline=pipeline, lead_index=lead_index)
        labels = diagnose_from_intervals(feats)
        match = evaluate_against_dataset(rec, labels)

        rows.append({
            'record_id': feats.record_id,
            'dataset': feats.dataset,
            'n_beats': feats.n_beats,
            'rr_ms': feats.rr_ms, 'rr_mad_ms': feats.rr_mad_ms,
            'qrs_ms': feats.qrs_ms, 'qrs_mad_ms': feats.qrs_mad_ms,
            'qt_ms': feats.qt_ms, 'qt_mad_ms': feats.qt_mad_ms,
            'pq_ms': feats.pq_ms, 'pq_mad_ms': feats.pq_mad_ms,
            'qtc_ms': feats.qtc_ms,
            'predicted_labels': labels,
            'ground_truth_labels': match['ground_truth'],
            'true_positive': match['true_positive'],
            'false_positive': match['false_positive'],
            'false_negative': match['false_negative'],
            'worker_error': feats.worker_error,
            'delineation_error': feats.delineation_error,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 7. Метрики качества метода delineate (на уровне диагностических меток)
# --------------------------------------------------------------------------

def quality_metrics(results_df: pd.DataFrame, labels: Optional[list[str]] = None) -> pd.DataFrame:
    """Считает precision/recall/F1 ПО КАЖДОЙ диагностируемой метке (а не
    усреднённо), плюс micro-average по всем меткам.

    Это и есть оценка качества пайплайна delineate -> диагноз: насколько
    хорошо диагноз, поставленный ИСКЛЮЧИТЕЛЬНО по геометрии точек делинеации
    (длины/девиации интервалов), согласуется с реальной разметкой SNOMED.
    """
    labels = labels or INTERVAL_DIAGNOSABLE
    rows = []
    tp_total = fp_total = fn_total = 0

    for label in labels:
        tp = sum(label in r for r in results_df['true_positive'])
        fp = sum(label in r for r in results_df['false_positive'])
        fn = sum(label in r for r in results_df['false_negative'])

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and (precision + recall) > 0 else None)

        n_support = sum(label in r for r in results_df['ground_truth_labels'])

        rows.append({
            'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'support': n_support,
        })
        tp_total += tp; fp_total += fp; fn_total += fn

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else None
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else None
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if micro_p is not None and micro_r is not None and (micro_p + micro_r) > 0 else None)

    rows.append({
        'label': 'MICRO_AVERAGE', 'tp': tp_total, 'fp': fp_total, 'fn': fn_total,
        'precision': micro_p, 'recall': micro_r, 'f1': micro_f1,
        'support': sum(r['support'] for r in rows),
    })

    return pd.DataFrame(rows)


def delineation_failure_rate(results_df: pd.DataFrame) -> dict:
    """Доля записей, на которых сама делинеация (не диагностика) упала —
    это отдельная, более базовая метрика качества метода delineate."""
    n = len(results_df)
    n_failed = int((results_df['delineation_error'].notna()).sum())
    return {
        'n_records': n,
        'n_delineation_failed': n_failed,
        'delineation_failure_rate': n_failed / n if n else None,
        'n_worker_failed': int((results_df['worker_error'].notna()).sum()),
    }