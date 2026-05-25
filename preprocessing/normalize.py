"""
preprocessing/normalize.py
Нормализация ЭКГ-сигналов: ресэмплинг, обрезка/дополнение, z-нормировка.

Поток [A]: [n_leads, n_samples] → [12, 5000] float64
Поток [B]: [2, beat_len] → [2, 300] float64
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


# ---------------------------------------------------------------------------
# Ресэмплинг
# ---------------------------------------------------------------------------

def resample_signal(
    sig: np.ndarray,
    fs_in: int,
    fs_out: int,
) -> np.ndarray:
    """
    Ресэмплирует сигнал [n_leads, n_samples] с fs_in на fs_out.

    Использует scipy.signal.resample_poly для точного полифазного ресэмплинга.
    Если fs_in == fs_out, возвращает копию.

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    fs_in : int
        Исходная частота дискретизации.
    fs_out : int
        Целевая частота дискретизации.

    Returns
    -------
    np.ndarray
        Ресэмплированный сигнал [n_leads, n_samples_new].
    """
    if fs_in == fs_out:
        return sig.copy()

    x = sig.astype(np.float64)
    # Вычисляем up/down через дробь для минимизации ошибки
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    up = frac.numerator
    down = frac.denominator
    out = resample_poly(x, up, down, axis=-1)
    return out


# ---------------------------------------------------------------------------
# Crop / Pad
# ---------------------------------------------------------------------------

def crop_or_pad(
    sig: np.ndarray,
    target_len: int,
    offset: Optional[int] = None,
) -> np.ndarray:
    """
    Обрезает или дополняет нулями сигнал до target_len по последней оси.

    Обрезка: от начала (offset=0) или с заданного смещения.
    Дополнение: нулями справа.

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    target_len : int
        Целевая длина.
    offset : int | None
        Начальная позиция при обрезке. None → 0 (с начала).

    Returns
    -------
    np.ndarray
        Сигнал [n_leads, target_len].
    """
    n_samples = sig.shape[-1]
    if n_samples == target_len:
        return sig.copy()

    if n_samples > target_len:
        start = 0 if offset is None else int(offset)
        start = min(start, n_samples - target_len)
        return sig[..., start : start + target_len].copy()

    # Дополнение нулями
    pad_width = [(0, 0)] * (sig.ndim - 1) + [(0, target_len - n_samples)]
    return np.pad(sig, pad_width, mode="constant", constant_values=0.0)


def crop_center_or_pad(
    sig: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Центрированная обрезка или дополнение нулями до target_len.

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    target_len : int
        Целевая длина.

    Returns
    -------
    np.ndarray
        Сигнал [n_leads, target_len].
    """
    n_samples = sig.shape[-1]
    if n_samples == target_len:
        return sig.copy()

    if n_samples > target_len:
        start = (n_samples - target_len) // 2
        return sig[..., start : start + target_len].copy()

    # Симметричное дополнение нулями
    pad_left = (target_len - n_samples) // 2
    pad_right = target_len - n_samples - pad_left
    pad_width = [(0, 0)] * (sig.ndim - 1) + [(pad_left, pad_right)]
    return np.pad(sig, pad_width, mode="constant", constant_values=0.0)


# ---------------------------------------------------------------------------
# Z-нормировка
# ---------------------------------------------------------------------------

def znorm(
    sig: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-нормирует сигнал вдоль заданной оси (mean=0, std=1).

    Константный сигнал (std≈0) возвращается как нулевой вектор (без NaN).

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал.
    axis : int
        Ось нормировки.
    eps : float
        Защита от деления на ноль.

    Returns
    -------
    np.ndarray
        Нормированный сигнал той же формы, dtype=float64.
    """
    x = sig.astype(np.float64)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x - mean) / (std + eps)


# ---------------------------------------------------------------------------
# Высокоуровневые функции для потоков A и B
# ---------------------------------------------------------------------------

def normalize_record_stream_a(
    sig: np.ndarray,
    fs_in: int,
    fs_out: int = 500,
    target_len: int = 5000,
    clip_mv: float = 10.0,
) -> np.ndarray:
    """
    Полная нормализация одной записи для потока [A].

    Шаги:
      1. Ресэмплинг → fs_out
      2. Clip амплитуды до [-clip_mv, +clip_mv]
      3. Crop/pad → target_len

    Parameters
    ----------
    sig : np.ndarray
        Исходный сигнал [n_leads, n_samples].
    fs_in : int
        Исходная частота дискретизации.
    fs_out : int
        Целевая частота дискретизации (по умолчанию 500 Гц).
    target_len : int
        Целевое число сэмплов (по умолчанию 5000 = 10 с при 500 Гц).
    clip_mv : float
        Граница клипирования амплитуды в мВ.

    Returns
    -------
    np.ndarray
        Нормализованный сигнал [n_leads, target_len], dtype=float64.
    """
    x = resample_signal(sig, fs_in=fs_in, fs_out=fs_out)
    x = np.clip(x, -clip_mv, clip_mv)
    x = crop_or_pad(x, target_len=target_len)
    return x.astype(np.float64)


def normalize_beat_stream_b(
    beat: np.ndarray,
    target_len: int = 300,
) -> np.ndarray:
    """
    Полная нормализация одного бита для потока [B].

    Шаги:
      1. Crop/pad → target_len
      2. Z-нормировка по каждому каналу (axis=-1)

    Parameters
    ----------
    beat : np.ndarray
        Бит [n_leads, beat_len].
    target_len : int
        Целевая длина бита (по умолчанию 300).

    Returns
    -------
    np.ndarray
        Нормализованный бит [n_leads, target_len], dtype=float64.
    """
    x = crop_or_pad(beat, target_len=target_len)
    x = znorm(x.astype(np.float64), axis=-1)
    return x

# ---------------------------------------------------------------------------
# Дополнительные утилиты (stub-реализации для совместимости)
# ---------------------------------------------------------------------------

def znorm_per_channel(
    sig: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-нормировка по каждому каналу независимо (alias для znorm axis=-1)."""
    return znorm(sig, axis=-1, eps=eps)


def align_by_rpeak(
    sig: np.ndarray,
    fs: float = 500.0,
    lead_idx: int = 1,
    target_rpeak: Optional[int] = None,
) -> np.ndarray:
    """
    Выравнивает сигнал по первому R-пику в указанном отведении.
    Если neurokit2 не установлен — возвращает сигнал без изменений.
    """
    try:
        import neurokit2 as nk
        lead = sig[lead_idx]
        _, info = nk.ecg_peaks(lead, sampling_rate=int(fs), method="pantompkins1985")
        r_peaks = info.get("ECG_R_Peaks", [])
        if len(r_peaks) == 0:
            return sig
        first_r = int(r_peaks[0])
        offset = first_r if target_rpeak is None else first_r - target_rpeak
        if offset <= 0:
            return sig
        return crop_or_pad(sig, target_len=sig.shape[-1], offset=offset)
    except Exception:
        return sig


def get_transform_info(
    sig: np.ndarray,
    fs_in: int,
    fs_out: int = 500,
    target_len: int = 5000,
) -> dict:
    """Возвращает информацию о трансформации (для отладки)."""
    from fractions import Fraction
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    n_resampled = round(sig.shape[-1] * fs_out / fs_in)
    return {
        "fs_in":       fs_in,
        "fs_out":      fs_out,
        "up":          frac.numerator,
        "down":        frac.denominator,
        "n_in":        sig.shape[-1],
        "n_resampled": n_resampled,
        "n_out":       target_len,
        "action":      "crop" if n_resampled > target_len else "pad",
    }
