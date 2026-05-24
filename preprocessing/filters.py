"""
preprocessing/filters.py

Фильтры сигнала ЭКГ:
  - FIR Kaiser low-pass  (cutoff=40 Гц, transition=5 Гц)
  - IIR Notch             (50 Гц или 60 Гц, Q=30)

Все функции работают с numpy-массивами формы [..., T],
где T — временная ось (последняя). Совместимо с [12, 5000],
[2, 300] и пакетными тензорами [N, C, T].

Пример:
    sig = apply_bandpass(sig, fs=500)          # low-pass 40 Гц
    sig = apply_notch(sig, fs=500, freq=50)    # notch 50 Гц
"""

from __future__ import annotations

import numpy as np
from scipy.signal import (
    firwin,
    iirnotch,
    kaiserord,
    lfilter,
    sosfilt,
    sosfiltfilt,
    iirnotch,
    butter,
    filtfilt,
)
from typing import Literal


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _apply_fir(b: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    Применяет FIR-фильтр b вдоль последней оси массива.
    Использует lfilter (каузальный) + ручная фазовая коррекция сдвигом.

    Мы намеренно используем lfilter, а не filtfilt, чтобы сохранить
    причинно-следственную структуру при стриминговом применении.
    Сдвиг на (N-1)//2 компенсирует групповую задержку FIR.
    """
    n_taps = len(b)
    delay = (n_taps - 1) // 2

    # применяем вдоль последней оси через np.apply_along_axis
    def _filt1d(x: np.ndarray) -> np.ndarray:
        y = lfilter(b, [1.0], x)
        # компенсируем задержку циклическим сдвигом
        y = np.roll(y, -delay)
        # артефакт в хвосте после roll — обнуляем последние `delay` сэмплов
        if delay > 0:
            y[-delay:] = y[-(delay + 1)]
        return y

    return np.apply_along_axis(_filt1d, axis=-1, arr=signal)


def _apply_iir_notch(b: np.ndarray, a: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """
    Применяет IIR-фильтр (b, a) вдоль последней оси с нулевой фазой (filtfilt).
    Нулевая фаза критична для точного измерения интервалов.
    """
    def _filt1d(x: np.ndarray) -> np.ndarray:
        # filtfilt требует длину сигнала > 3 * max(len(a), len(b))
        min_len = 3 * max(len(a), len(b))
        if len(x) < min_len:
            return x.copy()
        return filtfilt(b, a, x)

    return np.apply_along_axis(_filt1d, axis=-1, arr=signal)


# ──────────────────────────────────────────────────────────────────────────────
# FIR Kaiser Low-pass
# ──────────────────────────────────────────────────────────────────────────────

def design_lowpass_kaiser(
    fs: float,
    cutoff: float = 40.0,
    transition_width: float = 5.0,
    ripple_db: float = 60.0,
) -> np.ndarray:
    """
    Проектирует FIR low-pass фильтр методом Kaiser window.

    Параметры
    ----------
    fs              : частота дискретизации, Гц
    cutoff          : частота среза, Гц (по умолчанию 40)
    transition_width: ширина переходной полосы, Гц (по умолчанию 5)
    ripple_db       : затухание в полосе задержки, дБ (по умолчанию 60)

    Возвращает
    ----------
    b : np.ndarray — коэффициенты FIR фильтра
    """
    nyq = fs / 2.0
    if cutoff >= nyq:
        raise ValueError(
            f"cutoff ({cutoff} Гц) должен быть < Найквист ({nyq} Гц)"
        )

    # kaiserord возвращает (num_taps, beta)
    num_taps, beta = kaiserord(ripple_db, transition_width / nyq)

    # нечётное число тапов для линейной фазы type I
    if num_taps % 2 == 0:
        num_taps += 1

    b = firwin(
        num_taps,
        cutoff / nyq,
        window=("kaiser", beta),
        pass_zero=True,   # low-pass
    )
    return b


def apply_lowpass(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 40.0,
    transition_width: float = 5.0,
    ripple_db: float = 60.0,
) -> np.ndarray:
    """
    Применяет FIR Kaiser low-pass фильтр к ЭКГ-сигналу.

    Параметры
    ----------
    signal : np.ndarray формы [..., T]
    fs     : частота дискретизации входного сигнала, Гц

    Возвращает
    ----------
    np.ndarray той же формы
    """
    b = design_lowpass_kaiser(fs, cutoff, transition_width, ripple_db)
    return _apply_fir(b, signal.astype(np.float64))


# ──────────────────────────────────────────────────────────────────────────────
# Notch-фильтр (сетевая помеха)
# ──────────────────────────────────────────────────────────────────────────────

def apply_notch(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    Q: float = 30.0,
) -> np.ndarray:
    """
    Применяет IIR notch-фильтр для подавления сетевой помехи.

    Параметры
    ----------
    signal : np.ndarray формы [..., T]
    fs     : частота дискретизации, Гц
    freq   : частота помехи (50 Гц — Европа/Азия, 60 Гц — Америка)
    Q      : добротность (ширина провала = freq/Q)

    Возвращает
    ----------
    np.ndarray той же формы
    """
    nyq = fs / 2.0
    if freq >= nyq:
        # частота помехи выше Найквиста — фильтровать нечего
        return signal.copy()

    b, a = iirnotch(freq / nyq, Q)
    return _apply_iir_notch(b, a, signal.astype(np.float64))


def apply_notch_harmonics(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    harmonics: int = 2,
    Q: float = 30.0,
) -> np.ndarray:
    """
    Подавляет основную частоту помехи и её гармоники.

    Например, для 50 Гц + 2 гармоники → фильтруем 50, 100, 150 Гц.

    Параметры
    ----------
    harmonics : количество гармоник сверх основной (0 = только основная)
    """
    out = signal.astype(np.float64)
    nyq = fs / 2.0
    for k in range(1, harmonics + 2):  # k=1 → основная, k=2,3 → гармоники
        f = freq * k
        if f >= nyq:
            break
        out = apply_notch(out, fs, freq=f, Q=Q)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Комбинированный пайплайн
# ──────────────────────────────────────────────────────────────────────────────

PowerlineFreq = Literal[50, 60]


def apply_ecg_filters(
    signal: np.ndarray,
    fs: float,
    powerline_freq: PowerlineFreq = 50,
    lowpass_cutoff: float = 40.0,
    lowpass_transition: float = 5.0,
    lowpass_ripple_db: float = 60.0,
    notch_Q: float = 30.0,
    notch_harmonics: int = 0,
) -> np.ndarray:
    """
    Полный фильтрационный пайплайн для потока [A] и [B].

    Порядок применения:
      1. Notch (сначала, чтобы не «размазать» помеху при low-pass)
      2. FIR low-pass 40 Гц

    Параметры
    ----------
    signal           : np.ndarray формы [..., T]
    fs               : частота дискретизации, Гц
    powerline_freq   : 50 или 60 Гц
    notch_harmonics  : 0 = только основная, 1 = + 1 гармоника, и т.д.

    Возвращает
    ----------
    np.ndarray той же формы, dtype float64
    """
    out = signal.astype(np.float64, copy=True)

    # 1. Notch
    out = apply_notch_harmonics(out, fs, freq=float(powerline_freq),
                                harmonics=notch_harmonics, Q=notch_Q)

    # 2. FIR low-pass
    out = apply_lowpass(out, fs,
                        cutoff=lowpass_cutoff,
                        transition_width=lowpass_transition,
                        ripple_db=lowpass_ripple_db)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Утилита: определить частоту сети по стране/датасету
# ──────────────────────────────────────────────────────────────────────────────

DATASET_POWERLINE: dict[str, PowerlineFreq] = {
    # Европа / Азия — 50 Гц
    "ptbxl": 50,
    "cpsc2018": 50,
    "mitbih": 60,          # США
    "physionet2020": 50,   # смешанный — используем 50 как default
    "physionet2021": 50,
}


def get_powerline_freq(dataset_name: str) -> PowerlineFreq:
    """
    Возвращает типичную частоту сетевой помехи для датасета.
    Если неизвестно — возвращает 50.
    """
    return DATASET_POWERLINE.get(dataset_name.lower(), 50)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (запускается напрямую)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    fs = 500.0
    t = np.arange(5000) / fs

    # синтетический ЭКГ: синус 1 Гц + помеха 50 Гц + шум
    ecg_clean = np.sin(2 * np.pi * 1.2 * t)
    noise_50 = 0.3 * np.sin(2 * np.pi * 50.0 * t)
    noise_hf  = 0.1 * np.sin(2 * np.pi * 80.0 * t)  # выше 40 Гц
    white     = 0.05 * rng.standard_normal(5000)
    signal    = ecg_clean + noise_50 + noise_hf + white

    # форматируем как [12, 5000]
    signal_12ch = np.tile(signal, (12, 1))

    filtered = apply_ecg_filters(signal_12ch, fs=fs, powerline_freq=50)

    print(f"Вход  : shape={signal_12ch.shape}  std={signal_12ch.std():.4f}")
    print(f"Выход : shape={filtered.shape}     std={filtered.std():.4f}")

    # быстрая проверка: FFT до и после
    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(5000, 1 / fs)
    sp_in  = np.abs(rfft(signal_12ch[0]))
    sp_out = np.abs(rfft(filtered[0]))

    idx_50 = np.argmin(np.abs(freqs - 50))
    idx_80 = np.argmin(np.abs(freqs - 80))
    print(f"Амплитуда @ 50 Гц: до={sp_in[idx_50]:.3f}  после={sp_out[idx_50]:.3f}")
    print(f"Амплитуда @ 80 Гц: до={sp_in[idx_80]:.3f}  после={sp_out[idx_80]:.3f}")

    attn_50 = 20 * np.log10(sp_out[idx_50] / (sp_in[idx_50] + 1e-12))
    attn_80 = 20 * np.log10(sp_out[idx_80] / (sp_in[idx_80] + 1e-12))
    print(f"Ослабление @ 50 Гц: {attn_50:.1f} дБ  (ожидание < -20 дБ)")
    print(f"Ослабление @ 80 Гц: {attn_80:.1f} дБ  (ожидание < -60 дБ)")

    assert attn_50 < -20, "Notch не работает!"
    assert attn_80 < -40, "Low-pass не работает!"
    print("✓ Все проверки пройдены")
