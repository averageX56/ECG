"""
preprocessing/snomed_map.py
Словарь 27 scored SNOMED CT классов PhysioNet Challenge 2020.

Источник: dx_mapping_scored.csv
  https://github.com/physionetchallenges/physionet-challenge-2020/
  blob/master/evaluation-2020/dx_mapping_scored.csv
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 27 scored SNOMED CT классов (порядок важен — определяет индекс)
# ---------------------------------------------------------------------------
# (abbreviation, snomed_code, full_name)
_SCORED: list[tuple[str, int, str]] = [
    ("AF",    164889003, "Atrial fibrillation"),
    ("AFL",   164890007, "Atrial flutter"),
    ("Brady", 426627000, "Bradycardia"),
    ("CRBBB", 713427006, "Complete right bundle branch block"),
    ("IAVB",  270492004, "First-degree atrioventricular block"),
    ("IBBB",  733534002, "Incomplete bundle branch block"),
    ("NSR",   426783006, "Normal sinus rhythm"),
    ("LBBB",  164909002, "Left bundle branch block"),
    ("LAnFB", 445118002, "Left anterior fascicular block"),
    ("LAD",   39732003,  "Left axis deviation"),
    ("LQRSV", 251146004, "Low QRS voltages"),
    ("NICD",  698252002, "Nonspecific intraventricular conduction disorder"),
    ("PR",    10370003,  "Pacing rhythm"),
    ("PAC",   284470004, "Premature atrial contraction"),
    ("PVC",   427172004, "Premature ventricular contractions"),
    ("LPR",   164947007, "Prolonged PR interval"),
    ("LQT",   111975006, "Prolonged QT interval"),
    ("QAb",   164917005, "Q wave abnormal"),
    ("RAD",   47665007,  "Right axis deviation"),
    ("SA",    427393009, "Sinus arrhythmia"),
    ("SB",    426177001, "Sinus bradycardia"),
    ("STach", 427084000, "Sinus tachycardia"),
    ("SVPB",  63593006,  "Supraventricular premature beats"),
    ("TAb",   164934002, "T wave abnormal"),
    ("TInv",  59931005,  "T wave inversion"),
    ("VPB",   17338001,  "Ventricular premature beats"),
    ("VTach", 164896001, "Ventricular tachycardia"),
]

assert len(_SCORED) == 27, f"Ожидали 27 scored классов, нашли {len(_SCORED)}"

# Публичные константы
N_CLASSES: int = len(_SCORED)

SCORED_SNOMED_CLASSES: list[int] = [entry[1] for entry in _SCORED]
"""Список SNOMED CT кодов 27 scored классов (порядок = индекс)."""

SNOMED_TO_INDEX: dict[int, int] = {
    code: idx for idx, code in enumerate(SCORED_SNOMED_CLASSES)
}
"""Маппинг SNOMED CT код → индекс в label-векторе."""

SNOMED_TO_ABBR: dict[int, str] = {
    entry[1]: entry[0] for entry in _SCORED
}

ABBR_TO_SNOMED: dict[str, int] = {
    entry[0]: entry[1] for entry in _SCORED
}

# ---------------------------------------------------------------------------
# Маппинг PTB-XL SCP-кодов → SNOMED CT кодов
# ---------------------------------------------------------------------------
SCP_TO_SNOMED: dict[str, int] = {
    # Ритмы
    "NORM":   426783006,  # Normal sinus rhythm
    "AFIB":   164889003,  # Atrial fibrillation
    "AFLT":   164890007,  # Atrial flutter
    "STACH":  427084000,  # Sinus tachycardia
    "SBRAD":  426177001,  # Sinus bradycardia
    "SARRH":  427393009,  # Sinus arrhythmia
    "PSVT":   63593006,   # Supraventricular premature beats (приближение)
    "SVT":    63593006,   # Supraventricular tachycardia → SVPB группа
    "PACE":   10370003,   # Pacing rhythm
    "BIGU":   427172004,  # Bigeminy → PVC группа
    "TRIGU":  427172004,  # Trigeminy → PVC группа
    # Морфология / блокады
    "LBBB":   164909002,  # Left bundle branch block
    "RBBB":   59118001,   # Right bundle branch block (без уточнения)
    "CRBBB":  713427006,  # Complete RBBB
    "IRBBB":  713426002,  # Incomplete RBBB (может не входить в 27)
    "LAFB":   445118002,  # Left anterior fascicular block
    "LPFB":   445211001,  # Left posterior fascicular block (вне 27, ignored)
    # АВ-блокады
    "1AVB":   270492004,  # First-degree AV block
    "2AVB":   195042002,  # Second-degree AV block (вне 27, ignored)
    "3AVB":   27885002,   # Complete heart block (вне 27, ignored)
    # Экстрасистолы
    "PVC":    427172004,  # Premature ventricular contractions
    "PAC":    284470004,  # Premature atrial contraction
    "SVPB":   63593006,   # Supraventricular premature beats
    # Изменения ST-T
    "STTC":   164934002,  # ST-T changes → T wave abnormal (приближение)
    "NDT":    164934002,  # Non-diagnostic T changes
    "DIG":    164934002,  # Digitalis effect
    "LNGQT":  111975006,  # Prolonged QT
    # Ось
    "LAD":    39732003,   # Left axis deviation
    "RAD":    47665007,   # Right axis deviation
    # Прочее
    "LVH":    89792004,   # Left ventricular hypertrophy (вне 27, ignored)
    "RVH":    55827005,   # Right ventricular hypertrophy (вне 27, ignored)
    "LOWV":   251146004,  # Low voltage
    "WPW":    74390002,   # WPW (вне 27, ignored)
    "AMI":    57054005,   # Acute MI (вне 27, ignored)
    "IMI":    57054005,   # Inferior MI (вне 27, ignored)
    "ASMI":   57054005,   # Anteroseptal MI (вне 27, ignored)
    "ILMI":   57054005,   # Inferolateral MI (вне 27, ignored)
    "ALMI":   57054005,   # Anterolateral MI (вне 27, ignored)
    "ISCAL":  413444003,  # Ischemia (вне 27, ignored)
    "ISCAN":  413444003,
    "ISCIN":  413444003,
    "ISCLA":  413444003,
}


# ---------------------------------------------------------------------------
# Функции кодирования / декодирования
# ---------------------------------------------------------------------------

def encode_snomed_labels(
    codes: list[int],
) -> np.ndarray:
    """
    Кодирует список SNOMED CT кодов в multi-hot вектор [N_CLASSES].

    Неизвестные коды пропускаются с предупреждением.

    Parameters
    ----------
    codes : list[int]
        Список SNOMED CT кодов.

    Returns
    -------
    np.ndarray
        Multi-hot вектор [N_CLASSES], dtype=float32.
    """
    vec = np.zeros(N_CLASSES, dtype=np.float32)
    for code in codes:
        idx = SNOMED_TO_INDEX.get(code)
        if idx is not None:
            vec[idx] = 1.0
        else:
            logger.debug("Неизвестный SNOMED код %d — пропущен", code)
    return vec


def encode_ptbxl_labels(
    scp_codes: dict[str, float],
    min_likelihood: float = 0.0,
) -> np.ndarray:
    """
    Кодирует PTB-XL scp_codes → multi-hot вектор [N_CLASSES].

    Parameters
    ----------
    scp_codes : dict[str, float]
        Словарь {scp_abbr: likelihood_percent} из PTB-XL.
    min_likelihood : float
        Минимальная достоверность (0–100) для включения метки.

    Returns
    -------
    np.ndarray
        Multi-hot вектор [N_CLASSES], dtype=float32.
    """
    vec = np.zeros(N_CLASSES, dtype=np.float32)
    for abbr, likelihood in scp_codes.items():
        if likelihood < min_likelihood:
            continue
        snomed_code = SCP_TO_SNOMED.get(abbr.upper())
        if snomed_code is None:
            logger.debug("PTB-XL SCP-код %r не найден в маппинге — пропущен", abbr)
            continue
        idx = SNOMED_TO_INDEX.get(snomed_code)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def decode_label_vector(vec: np.ndarray) -> list[int]:
    """
    Преобразует multi-hot вектор обратно в список SNOMED CT кодов.

    Parameters
    ----------
    vec : np.ndarray
        Multi-hot вектор [N_CLASSES].

    Returns
    -------
    list[int]
        Список SNOMED CT кодов с ненулевыми позициями.
    """
    indices = np.where(vec > 0.5)[0]
    return [SCORED_SNOMED_CLASSES[i] for i in indices]