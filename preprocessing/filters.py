"""
preprocessing/filters.py
FIR Kaiser low-pass + notch фильтры для ЭКГ-сигналов.

Форма входного сигнала: [n_leads, n_samples]
Все функции возвращают float64.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Таблица частот сетевой помехи по датасетам
# ---------------------------------------------------------------------------
DATASET_POWERLINE: dict[str, int] = {
    "ptbxl": 50,
    "cpsc2018": 50,
    "cpsc_extra": 50,
    "stpetersburg": 50,
    "ptb": 50,
    "georgia": 60,
    "mitbih": 60,
}


def get_powerline_freq(dataset_name: str) -> int:
    """Возвращает частоту сетевой помехи (50 или 60 Гц) для датасета."""
    return DATASET_POWERLINE.get(dataset_name.lower(), 50)


# ---------------------------------------------------------------------------
# Проектирование FIR-фильтра (Kaiser)
# ---------------------------------------------------------------------------

def design_lowpass_kaiser(
    fs: float,
    cutoff: float = 40.0,
    transition_width: float = 5.0,
    attenuation_db: float = 60.0,
) -> np.ndarray:
    """
    Проектирует FIR-фильтр низких частот по методу окна Кайзера.

    Parameters
    ----------
    fs : float
        Частота дискретизации, Гц.
    cutoff : float
        Частота среза, Гц.
    transition_width : float
        Ширина переходной полосы, Гц.
    attenuation_db : float
        Желаемое ослабление в полосе задерживания, дБ.

    Returns
    -------
    np.ndarray
        Коэффициенты FIR-фильтра нечётной длины (тип I).

    Raises
    ------
    ValueError
        Если cutoff превышает частоту Найквиста.
    """
    nyq = fs / 2.0
    if cutoff >= nyq:
        raise ValueError(
            f"cutoff={cutoff} Гц >= частота Найквиста {nyq} Гц (fs={fs})"
        )

    # Параметры окна Кайзера
    beta = sp_signal.kaiser_beta(attenuation_db)
    # Нормированные частоты
    width_norm = transition_width / nyq
    # Минимальный порядок фильтра (всегда нечётный → тип I)
    n_taps, _ = sp_signal.kaiserord(attenuation_db, width_norm)
    if n_taps % 2 == 0:
        n_taps += 1  # гарантируем нечётное число тапов

    cutoff_norm = cutoff / nyq
    b = sp_signal.firwin(n_taps, cutoff_norm, window=("kaiser", beta))
    return b.astype(np.float64)


# ---------------------------------------------------------------------------
# Применение LP-фильтра
# ---------------------------------------------------------------------------

def apply_lowpass(
    sig: np.ndarray,
    fs: float = 500.0,
    cutoff: float = 40.0,
    transition_width: float = 5.0,
) -> np.ndarray:
    """
    Применяет FIR LP-фильтр (Kaiser) к сигналу [n_leads, n_samples].

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    fs : float
        Частота дискретизации.
    cutoff : float
        Частота среза.
    transition_width : float
        Ширина переходной полосы.

    Returns
    -------
    np.ndarray
        Отфильтрованный сигнал той же формы, dtype=float64.
    """
    x = sig.astype(np.float64)
    b = design_lowpass_kaiser(fs=fs, cutoff=cutoff, transition_width=transition_width)
    out = sp_signal.filtfilt(b, [1.0], x, axis=-1)
    return out


# ---------------------------------------------------------------------------
# Notch-фильтр (IIR Notch)
# ---------------------------------------------------------------------------

def apply_notch(
    sig: np.ndarray,
    fs: float = 500.0,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Применяет notch-фильтр для подавления одной частоты.

    Если freq >= fs/2, сигнал возвращается без изменений (NOP).

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    fs : float
        Частота дискретизации.
    freq : float
        Подавляемая частота, Гц.
    quality : float
        Добротность фильтра.

    Returns
    -------
    np.ndarray
        Отфильтрованный сигнал той же формы.
    """
    nyq = fs / 2.0
    if freq >= nyq:
        # выше Найквиста — NOP, возвращаем как есть
        return sig

    x = sig.astype(np.float64)
    b, a = sp_signal.iirnotch(freq / nyq, quality)
    out = sp_signal.filtfilt(b, a, x, axis=-1)
    return out


def apply_notch_harmonics(
    sig: np.ndarray,
    fs: float = 500.0,
    freq: float = 50.0,
    harmonics: int = 1,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Применяет notch-фильтр для основной частоты и её гармоник.

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    fs : float
        Частота дискретизации.
    freq : float
        Основная частота (напр., 50 Гц).
    harmonics : int
        Число дополнительных гармоник (1 → фильтруем freq и 2*freq).
    quality : float
        Добротность фильтра.

    Returns
    -------
    np.ndarray
        Отфильтрованный сигнал.
    """
    out = sig.astype(np.float64)
    for k in range(1, harmonics + 2):
        target = freq * k
        out = apply_notch(out, fs=fs, freq=target, quality=quality)
    return out


# ---------------------------------------------------------------------------
# Полный пайплайн фильтрации ЭКГ
# ---------------------------------------------------------------------------

def apply_ecg_filters(
    sig: np.ndarray,
    fs: float = 500.0,
    powerline_freq: int = 50,
    lp_cutoff: float = 40.0,
    lp_transition: float = 5.0,
    notch_quality: float = 30.0,
) -> np.ndarray:
    """
    Полный пайплайн предобработки ЭКГ:
      1. Notch-фильтр на частоте сети (50 или 60 Гц)
      2. FIR Kaiser LP 40 Гц

    Parameters
    ----------
    sig : np.ndarray
        Входной сигнал [n_leads, n_samples].
    fs : float
        Частота дискретизации.
    powerline_freq : int
        Частота сетевой помехи (50 или 60).
    lp_cutoff : float
        Частота среза LP-фильтра.
    lp_transition : float
        Ширина переходной полосы LP.
    notch_quality : float
        Добротность notch-фильтра.

    Returns
    -------
    np.ndarray
        Очищенный сигнал той же формы, dtype=float64.
    """
    out = sig.astype(np.float64)
    # Шаг 1: notch
    out = apply_notch(out, fs=fs, freq=float(powerline_freq), quality=notch_quality)
    # Шаг 2: LP
    out = apply_lowpass(out, fs=fs, cutoff=lp_cutoff, transition_width=lp_transition)
    return out