"""
preprocessing/normalize.py

Нормализация временной оси ЭКГ-сигналов:
  - Ресэмплинг → целевую fs (по умолчанию 500 Гц) через resample_poly
  - Crop / pad → ровно target_len сэмплов (по умолчанию 5000)
  - Z-нормализация (для потока [B] MIT-BIH)
  - Детекция и выравнивание по R-пику через rlign (опционально, ablation-флаг)

Все функции работают с массивами [..., T] (T — последняя ось),
совместимы с [12, 5000], [2, 300], [N, 12, 5000].
"""

from __future__ import annotations

import logging
import warnings
from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Ресэмплинг
# ──────────────────────────────────────────────────────────────────────────────

def resample_signal(
    signal: np.ndarray,
    fs_in: float,
    fs_out: float = 500.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Ресэмплирует сигнал с fs_in → fs_out, используя resample_poly.

    resample_poly использует рациональный коэффициент up/down,
    внутри применяет anti-aliasing FIR — подходит для медицинских сигналов.

    Параметры
    ----------
    signal  : np.ndarray, любой формы
    fs_in   : исходная частота дискретизации, Гц
    fs_out  : целевая частота дискретизации, Гц
    axis    : ось времени

    Возвращает
    ----------
    np.ndarray с той же формой, кроме оси времени
    """
    if abs(fs_in - fs_out) < 1e-6:
        return signal.astype(np.float64, copy=False)

    fs_in  = float(fs_in)
    fs_out = float(fs_out)

    # рациональное приближение: up/down через НОД
    # масштабируем до целых с достаточной точностью (до 0.1 Гц)
    scale = 10
    up   = int(round(fs_out * scale))
    down = int(round(fs_in  * scale))
    g    = gcd(up, down)
    up   //= g
    down //= g

    return resample_poly(signal.astype(np.float64), up, down, axis=axis)


# ──────────────────────────────────────────────────────────────────────────────
# Crop / Pad → фиксированная длина
# ──────────────────────────────────────────────────────────────────────────────

def crop_or_pad(
    signal: np.ndarray,
    target_len: int = 5000,
    axis: int = -1,
    pad_mode: str = "edge",
    crop_start: int = 0,
) -> np.ndarray:
    """
    Обрезает или дополняет сигнал до target_len вдоль оси axis.

    Параметры
    ----------
    signal      : np.ndarray любой формы
    target_len  : целевая длина (5000 для потока [A], 300 для потока [B])
    axis        : временная ось (по умолчанию последняя)
    pad_mode    : режим np.pad — 'edge' (по умолчанию), 'constant', 'reflect'
    crop_start  : откуда начинать кроп при обрезке (по умолчанию 0)

    Возвращает
    ----------
    np.ndarray той же формы, кроме оси axis = target_len
    """
    current_len = signal.shape[axis]

    if current_len == target_len:
        return signal

    if current_len > target_len:
        # обрезаем
        slices = [slice(None)] * signal.ndim
        slices[axis] = slice(crop_start, crop_start + target_len)
        return signal[tuple(slices)]

    # дополняем
    pad_width = [(0, 0)] * signal.ndim
    pad_needed = target_len - current_len
    pad_width[axis] = (0, pad_needed)
    return np.pad(signal, pad_width, mode=pad_mode)


def crop_center_or_pad(
    signal: np.ndarray,
    target_len: int = 5000,
    axis: int = -1,
    pad_mode: str = "edge",
) -> np.ndarray:
    """
    Кроп по центру: оставляет средние target_len сэмплов.
    Предпочтительно для CPSC 2018 (переменная длина, центрируем кардиограмму).
    """
    current_len = signal.shape[axis]
    if current_len <= target_len:
        return crop_or_pad(signal, target_len, axis, pad_mode)
    start = (current_len - target_len) // 2
    return crop_or_pad(signal, target_len, axis, pad_mode, crop_start=start)


# ──────────────────────────────────────────────────────────────────────────────
# Z-нормализация (поток [B] — MIT-BIH)
# ──────────────────────────────────────────────────────────────────────────────

def znorm(
    signal: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-нормализация вдоль указанной оси.

    Параметры
    ----------
    signal : np.ndarray — например, [2, 300] (один бит) или [N, 2, 300]
    axis   : ось, вдоль которой считаем mean/std (временная)
    eps    : защита от деления на 0

    Возвращает
    ----------
    np.ndarray той же формы
    """
    mean = signal.mean(axis=axis, keepdims=True)
    std  = signal.std(axis=axis, keepdims=True)
    return (signal - mean) / (std + eps)


def znorm_per_channel(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-нормализация по каждому каналу отдельно (вдоль оси -1).
    Форма входа: [..., C, T] — нормируем каждый канал по своей временной оси.
    """
    return znorm(signal, axis=-1, eps=eps)


# ──────────────────────────────────────────────────────────────────────────────
# Полный пайплайн нормализации для потока [A]
# ──────────────────────────────────────────────────────────────────────────────

def normalize_record_stream_a(
    signal: np.ndarray,
    fs_in: float,
    fs_out: float = 500.0,
    target_len: int = 5000,
    center_crop: bool = True,
) -> np.ndarray:
    """
    Нормализует запись для потока [A] (запись-уровень, backbone ResNet).

    Шаги:
      1. float64
      2. Ресэмплинг → fs_out
      3. Crop/pad → target_len

    Параметры
    ----------
    signal      : [C, T] — уже отфильтрованный сигнал (filters.py)
    fs_in       : исходная fs
    fs_out      : целевая fs (500 Гц)
    target_len  : длина в сэмплах (5000 = 10 с при 500 Гц)
    center_crop : True → кроп по центру, False → кроп с начала

    Возвращает
    ----------
    np.ndarray [C, target_len], dtype float64
    """
    sig = signal.astype(np.float64, copy=False)

    # ресэмплинг
    if abs(fs_in - fs_out) > 1e-6:
        sig = resample_signal(sig, fs_in, fs_out, axis=-1)

    # crop / pad
    fn_crop = crop_center_or_pad if center_crop else crop_or_pad
    sig = fn_crop(sig, target_len, axis=-1)

    return sig


# ──────────────────────────────────────────────────────────────────────────────
# Полный пайплайн нормализации для потока [B]
# ──────────────────────────────────────────────────────────────────────────────

def normalize_beat_stream_b(
    beat: np.ndarray,
    target_len: int = 300,
) -> np.ndarray:
    """
    Нормализует один бит для потока [B] (MIT-BIH, beat-уровень).

    Предполагается, что бит уже вырезан вокруг R-пика (±150 сэмплов).
    Фильтрация выполнена в filters.py до вызова этой функции.

    Шаги:
      1. Crop/pad → [2, target_len]
      2. Z-нормализация по каждому каналу

    Параметры
    ----------
    beat       : [2, T] — два отведения (MLII, V1/V5), т.е. [C, T]
    target_len : 300 сэмплов = ±150 при 360 Гц

    Возвращает
    ----------
    np.ndarray [2, target_len], dtype float64, z-нормировано
    """
    sig = beat.astype(np.float64, copy=False)
    sig = crop_or_pad(sig, target_len, axis=-1)
    sig = znorm_per_channel(sig)
    return sig


# ──────────────────────────────────────────────────────────────────────────────
# Выравнивание по R-пику через rlign (ablation-флаг)
# ──────────────────────────────────────────────────────────────────────────────

_rlign_available: Optional[bool] = None


def _check_rlign() -> bool:
    """Проверяет наличие библиотеки rlign (lazy import)."""
    global _rlign_available
    if _rlign_available is None:
        try:
            import rlign  # noqa: F401
            _rlign_available = True
        except ImportError:
            _rlign_available = False
            logger.warning(
                "rlign не установлен — выравнивание по R-пику недоступно. "
                "Установите: pip install rlign"
            )
    return _rlign_available


def align_by_rpeak(
    signal: np.ndarray,
    fs: float = 500.0,
    lead_idx: int = 1,
) -> np.ndarray:
    """
    Выравнивает ЭКГ по R-пику через библиотеку rlign.
    Используется только как ablation-флаг для PTB-XL.

    Параметры
    ----------
    signal   : [C, T] — форма потока [A]
    fs       : частота дискретизации (после ресэмплинга = 500)
    lead_idx : отведение для детекции R-пика (1 = отведение II в PTB-XL)

    Возвращает
    ----------
    np.ndarray [C, T] — сигнал с выравниванием (или оригинал если rlign недоступен)
    """
    if not _check_rlign():
        return signal

    try:
        import rlign
        # rlign ожидает [T, C] → транспонируем туда и обратно
        sig_tc = signal.T  # [T, C]
        aligned_tc = rlign.align(sig_tc, fs=fs, lead=lead_idx)
        return aligned_tc.T  # [C, T]
    except Exception as exc:
        logger.warning(f"rlign.align упал с ошибкой: {exc}. Возвращаем оригинал.")
        return signal


# ──────────────────────────────────────────────────────────────────────────────
# Утилита: сводная информация о трансформации
# ──────────────────────────────────────────────────────────────────────────────

def get_transform_info(
    fs_in: float,
    fs_out: float = 500.0,
    n_samples_in: int = 5000,
    target_len: int = 5000,
) -> dict:
    """
    Возвращает словарь с ожидаемыми параметрами трансформации.
    Полезно для логирования и отладки.
    """
    n_samples_after_resample = int(round(n_samples_in * fs_out / fs_in))
    duration_in_sec = n_samples_in / fs_in
    return {
        "fs_in": fs_in,
        "fs_out": fs_out,
        "n_samples_in": n_samples_in,
        "duration_sec": round(duration_in_sec, 3),
        "n_samples_after_resample": n_samples_after_resample,
        "target_len": target_len,
        "action": "crop" if n_samples_after_resample > target_len else "pad",
        "diff_samples": n_samples_after_resample - target_len,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ── Поток [A]: 12-канальная запись PTB-XL 100 Гц → 500 Гц, 1000 → 5000 ──
    sig_100hz = rng.standard_normal((12, 1000)).astype(np.float32)
    out_a = normalize_record_stream_a(sig_100hz, fs_in=100.0)
    assert out_a.shape == (12, 5000), f"Ожидали (12, 5000), получили {out_a.shape}"
    assert out_a.dtype == np.float64
    print(f"[A] 100 Гц 1000 сэмп → {out_a.shape}  ✓")

    # ── Поток [A]: CPSC 2018 — переменная длина (6500) при 500 Гц ──
    sig_cpsc = rng.standard_normal((12, 6500))
    out_cpsc = normalize_record_stream_a(sig_cpsc, fs_in=500.0, center_crop=True)
    assert out_cpsc.shape == (12, 5000), f"Ожидали (12, 5000), получили {out_cpsc.shape}"
    print(f"[A] CPSC 500 Гц 6500 сэмп → {out_cpsc.shape}  ✓")

    # ── Поток [A]: уже 500 Гц, короткая запись (3000) → паддинг ──
    sig_short = rng.standard_normal((12, 3000))
    out_short = normalize_record_stream_a(sig_short, fs_in=500.0)
    assert out_short.shape == (12, 5000)
    print(f"[A] 500 Гц 3000 сэмп → {out_short.shape}  ✓")

    # ── Поток [B]: бит MIT-BIH 2 канала ──
    beat = rng.standard_normal((2, 310))   # немного длиннее
    out_b = normalize_beat_stream_b(beat, target_len=300)
    assert out_b.shape == (2, 300)
    # z-норм: среднее ≈ 0, std ≈ 1
    assert abs(out_b[0].mean()) < 1e-10, "Z-норм не работает"
    assert abs(out_b[0].std() - 1.0) < 1e-6, "Z-норм std != 1"
    print(f"[B] бит 310 → {out_b.shape}, mean={out_b[0].mean():.2e}  ✓")

    # ── Ресэмплинг точность ──
    t_in = np.arange(5000) / 500.0
    sine_500 = np.sin(2 * np.pi * 10 * t_in)[np.newaxis, :]  # [1, 5000]
    down_100 = resample_signal(sine_500, fs_in=500, fs_out=100)
    assert down_100.shape == (1, 1000), f"Ресэмплинг: {down_100.shape}"
    print(f"Ресэмплинг 500→100: {sine_500.shape} → {down_100.shape}  ✓")

    # ── get_transform_info ──
    info = get_transform_info(fs_in=100, fs_out=500, n_samples_in=1000)
    print(f"Transform info: {info}")
    assert info["n_samples_after_resample"] == 5000
    assert info["action"] == "pad"   # 5000 == 5000, но из-за ≥ edge case проверяем
    print("\n✓ Все проверки normalize.py пройдены")
