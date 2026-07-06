# ecg_delineate_full.py
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import wfdb
import neurokit2 as nk

try:
    import scipy.io as sio
except ImportError:
    sio = None


def pick_lead_index(sig_names, preferred=('II', 'ii', 'MLII', 'I', 'i')):
    for p in preferred:
        if p in sig_names:
            return sig_names.index(p)
    return 0


def _parse_hea_header(hea_path):
    """Минимальный парсер .hea-заголовка WFDB — нужен, когда сигнал лежит не в
    .dat, а в .mat (формат PhysioNet/CinC Challenge 2020: Georgia, CPSC,
    PTB-XL и т.п.), который wfdb.rdrecord не умеет читать напрямую.
    """
    with open(hea_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

    record_line = lines[0].split()
    n_sig = int(record_line[1])
    fs = float(record_line[2])

    sig_names = []
    gains = []
    baselines = []
    sig_file = None
    for i in range(1, n_sig + 1):
        parts = lines[i].split()
        if sig_file is None:
            sig_file = parts[0]
        # формат: filename format gain(/units) adc_res adc_zero baseline checksum block sig_name
        gain_field = parts[2]
        gain = float(gain_field.split('/')[0]) if '/' in gain_field else float(gain_field)
        gains.append(gain)
        baseline = float(parts[4]) if len(parts) > 4 else 0.0
        baselines.append(baseline)
        sig_names.append(parts[-1])

    return {
        'n_sig': n_sig, 'fs': fs, 'sig_names': sig_names,
        'gains': gains, 'baselines': baselines, 'sig_file': sig_file,
    }


def _load_mat_signal(base_path):
    """Читает сигнал из пары .hea + .mat (формат PhysioNet/CinC Challenge).
    Возвращает (p_signal[n_samples, n_leads] в мВ, fs, sig_names).
    """
    if sio is None:
        raise RuntimeError('Для чтения .mat-сигналов нужен scipy: pip install scipy')

    header = _parse_hea_header(base_path + '.hea')

    mat = sio.loadmat(base_path + '.mat')
    if 'val' not in mat:
        raise KeyError(f"В {base_path}.mat не найден ожидаемый ключ 'val'; ключи: {list(mat.keys())}")

    raw = mat['val'].astype(np.float64)  # shape (n_leads, n_samples), raw ADC units
    gains = np.array(header['gains']).reshape(-1, 1)
    baselines = np.array(header['baselines']).reshape(-1, 1)
    gains[gains == 0] = 1.0  # защита от деления на ноль при кривом заголовке

    physical = (raw - baselines) / gains  # -> мВ
    p_signal = physical.T  # (n_samples, n_leads)

    return p_signal, header['fs'], header['sig_names']


def _read_record(base_path):
    """Читает запись либо через wfdb (.dat), либо через .mat-фоллбек
    (PhysioNet2020-style), автоопределяя формат по наличию файла на диске.
    Возвращает объект с атрибутами .p_signal, .fs, .sig_name — как у wfdb.Record.

    Это ЕДИНСТВЕННОЕ место в пайплайне, которое должно читать сигнал с диска.
    Раньше ecg_worker.py читал тот же сигнал напрямую через wfdb.rdrecord без
    .mat-фоллбека, из-за чего на .mat-записях (Georgia/CPSC_Extra/PTB-XL) gain/
    baseline применялись неверно (или чтение падало) — отсюда расхождение
    между медианами интервалов (из ecg_worker) и девиациями (из delineate_full).
    """
    if os.path.exists(base_path + '.mat'):
        p_signal, fs, sig_names = _load_mat_signal(base_path)

        class _Rec:
            pass
        r = _Rec()
        r.p_signal = p_signal
        r.fs = fs
        r.sig_name = sig_names
        return r

    return wfdb.rdrecord(base_path)


# --------------------------------------------------------------------------
# Конфигурация пайплайна: clean -> peaks -> delineate как 3 независимых шага
# --------------------------------------------------------------------------

@dataclass
class DelineationPipeline:
    """Описывает, какими методами/функциями выполняются 3 шага neurokit:
    ecg_clean -> ecg_peaks -> ecg_delineate.

    Каждый шаг можно переопределить ЛИБО просто сменив метод по имени
    (как в nk.ecg_clean(method=...) / nk.ecg_peaks(method=...) /
    nk.ecg_delineate(method=...)), ЛИБО подсунув свою функцию (clean_fn /
    peaks_fn / delineate_fn) — например, кастомный фильтр или альтернативный
    R-peak детектор, не относящийся к neurokit вовсе.

    Сигнатуры кастомных функций:
      clean_fn(signal, sampling_rate) -> cleaned_signal (np.ndarray)
      peaks_fn(cleaned, sampling_rate) -> r_peaks (list[int] | np.ndarray)
      delineate_fn(cleaned, r_peaks, sampling_rate) -> waves (dict),
          с теми же ключами, что у nk.ecg_delineate: ECG_P_Peaks, ECG_Q_Peaks,
          ECG_S_Peaks, ECG_T_Peaks, ECG_P_Onsets, ECG_R_Onsets, ECG_R_Offsets,
          ECG_T_Offsets (списки индексов сэмплов, может содержать NaN).
    """
    clean_method: str = 'neurokit'
    peaks_method: str = 'neurokit'
    delineate_method: str = 'dwt'

    clean_fn: Optional[Callable] = None
    peaks_fn: Optional[Callable] = None
    delineate_fn: Optional[Callable] = None


DEFAULT_PIPELINE = DelineationPipeline()


def _run_pipeline(signal, fs, pipeline: DelineationPipeline):
    """Прогоняет 3 шага (clean/peaks/delineate) по отдельности — вместо
    единого nk.ecg_process — чтобы любой шаг можно было заменить независимо
    от остальных."""

    if pipeline.clean_fn is not None:
        cleaned = pipeline.clean_fn(signal, fs)
    else:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method=pipeline.clean_method)
    cleaned = np.asarray(cleaned, dtype=float)

    if pipeline.peaks_fn is not None:
        r_peaks = np.asarray(pipeline.peaks_fn(cleaned, fs), dtype=int)
    else:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method=pipeline.peaks_method)
        r_peaks = np.asarray(info.get('ECG_R_Peaks', []), dtype=int)

    if len(r_peaks) < 3:
        return cleaned, r_peaks, {}

    if pipeline.delineate_fn is not None:
        waves = pipeline.delineate_fn(cleaned, r_peaks, fs)
    else:
        _, waves = nk.ecg_delineate(cleaned, r_peaks, sampling_rate=fs, method=pipeline.delineate_method)

    return cleaned, r_peaks, waves


def delineate_full(rec: dict, lead_pref=None, lead_index: Optional[int] = 0,
                    pipeline: Optional[DelineationPipeline] = None) -> dict:
    """Возвращает сигнал + все найденные точки разметки (пики и onset/offset) +
    оценку качества (zhao2018) для визуализации.

    Выбор отведения:
      - lead_index (по умолчанию 0) — берём отведение по фиксированному индексу
        в файле, т.е. ПЕРВОЕ отведение записи, как оно лежит в .hea, без поиска
        по имени. Это и есть текущее поведение пайплайна по умолчанию.
      - lead_pref — старый режим: список предпочтительных имён отведений
        ('II','MLII','I',...), из которых берётся первое найденное. Включается,
        если явно передать lead_pref (и НЕ передавать lead_index, либо передать
        lead_index=None).

    pipeline: опциональная DelineationPipeline для подмены clean/peaks/delineate
    шагов по отдельности (по умолчанию — DEFAULT_PIPELINE, эквивалент старого
    nk.ecg_process(method='neurokit')).
    """
    pipeline = pipeline or DEFAULT_PIPELINE

    out = {
        'record_id': rec['record_id'], 'dataset': rec.get('dataset'),
        'signal': None, 'fs': None, 'lead_name': None,
        'r_peaks': [], 'p_peaks': [], 'q_peaks': [], 's_peaks': [], 't_peaks': [],
        'p_onsets': [], 'qrs_onsets': [], 'qrs_offsets': [], 't_offsets': [],
        'quality_label': None,  # 'Excellent' | 'Barely acceptable' | 'Unacceptable'
        'error': None,
    }
    base = rec['filepath'][:-4] if rec['filepath'].endswith('.hea') else rec['filepath']
    try:
        record = _read_record(base)
        fs = record.fs

        if lead_pref is not None and lead_index is None:
            idx = pick_lead_index(record.sig_name, lead_pref)
        else:
            idx = lead_index if lead_index is not None else 0
            if idx >= len(record.sig_name):
                idx = 0  # защита: если в записи меньше отведений, чем ожидалось

        signal = record.p_signal[:, idx]

        if signal is None or len(signal) < fs * 2:
            out['error'] = 'too_short'
            return out

        cleaned, r_peaks, waves = _run_pipeline(signal, fs, pipeline)

        r_peaks = [int(x) for x in r_peaks]
        if len(r_peaks) < 3:
            out['error'] = 'not_enough_rpeaks'
            return out

        def clean_ints(key):
            return [int(x) for x in waves.get(key, []) if not (isinstance(x, float) and np.isnan(x))]

        # --- оценка качества всей записи (zhao2018: Excellent / Barely acceptable / Unacceptable) ---
        try:
            quality_label = nk.ecg_quality(cleaned, sampling_rate=fs, method='zhao2018')
        except Exception:
            quality_label = None

        out.update({
            'signal': cleaned, 'fs': fs, 'lead_name': record.sig_name[idx],
            'r_peaks': r_peaks,
            'p_peaks': clean_ints('ECG_P_Peaks'),
            'q_peaks': clean_ints('ECG_Q_Peaks'),
            's_peaks': clean_ints('ECG_S_Peaks'),
            't_peaks': clean_ints('ECG_T_Peaks'),
            'p_onsets': clean_ints('ECG_P_Onsets'),
            'qrs_onsets': clean_ints('ECG_R_Onsets'),
            'qrs_offsets': clean_ints('ECG_R_Offsets'),
            't_offsets': clean_ints('ECG_T_Offsets'),
            'quality_label': quality_label,
        })
    except Exception as e:
        out['error'] = repr(e)

    return out