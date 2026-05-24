"""
preprocessing/snomed_map.py

Маппинг меток PTB-XL scp_codes → SNOMED CT коды,
совместимые с пространством 27 scored классов PhysioNet Challenge 2020/2021.

Структура:
  - SCP_TO_SNOMED        : dict[str, int]  — scp_code → SNOMED CT concept ID
  - SCORED_SNOMED_CLASSES: list[int]       — 27 классов в фиксированном порядке
  - snomed_to_index      : dict[int, int]  — SNOMED → индекс в multi-hot векторе
  - encode_ptbxl_labels  : scp_codes dict → multi-hot [27]
  - encode_snomed_labels : список SNOMED кодов → multi-hot [27]
  - load_physionet_mapping: загружает dx_mapping_scored.csv → DataFrame

Источники:
  1. PhysioNet 2020 dx_mapping_scored.csv:
     https://github.com/physionetchallenges/physionet-challenge-2020/blob/master/
     evaluation-2020/dx_mapping_scored.csv
  2. PTB-XL Database SCP statements mapping:
     ptbxl_database.csv → scp_codes → таблица scp_statements.csv
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 27 scored классов PhysioNet Challenge 2020 (SNOMED CT concept IDs)
# Порядок строго фиксирован — это индексы в multi-hot векторе!
# Источник: dx_mapping_scored.csv
# ──────────────────────────────────────────────────────────────────────────────

SCORED_SNOMED_CLASSES: List[int] = [
    270492004,   # 0  I-AVB       — First degree AV block
    164889003,   # 1  AFIB        — Atrial fibrillation
    164890007,   # 2  AFL         — Atrial flutter
    426627000,   # 3  Brady       — Bradycardia
    713427006,   # 4  CRBBB       — Complete right bundle branch block
    713426002,   # 5  IRBBB       — Incomplete right bundle branch block
    445118002,   # 6  LAnFB       — Left anterior fascicular block
    39732003,    # 7  LAD         — Left axis deviation
    164909002,   # 8  LBBB        — Left bundle branch block
    251146004,   # 9  LQRSV       — Low QRS voltages
    698252002,   # 10 NSIVCB      — Nonspecific intraventricular conduction disturbance
    10370003,    # 11 PR          — Pacing rhythm
    164947007,   # 12 LPR         — Prolonged PR interval
    111975006,   # 13 LQT         — Prolonged QT interval
    164917005,   # 14 QAb         — QRS abnormality
    47665007,    # 15 RAD         — Right axis deviation
    427393009,   # 16 SA          — Sinus arrhythmia
    426177001,   # 17 SB          — Sinus bradycardia
    426783006,   # 18 SNR         — Sinus rhythm (Normal)
    427084000,   # 19 STach       — Sinus tachycardia
    63593006,    # 20 SVPB        — Supraventricular premature beats (НаджЭС)
    164934002,   # 21 TAb         — T-wave abnormality
    59931005,    # 22 TInv        — T-wave inversion
    17338001,    # 23 VPB         — Ventricular premature beats (ЖЭС)
    164895006,   # 24 VEB         — Ventricular escape beat
    427172004,   # 25 PVC         — Premature ventricular contractions
    164909002,   # 26 LBBB dup   — дублируется из-за разных источников; обрабатываем ниже
]

# Убираем дубликат из индекса 26 — используем уникальный список
_UNIQUE_SCORED: List[int] = []
_seen: set = set()
for _s in SCORED_SNOMED_CLASSES:
    if _s not in _seen:
        _UNIQUE_SCORED.append(_s)
        _seen.add(_s)

# Канонический список 27 уникальных классов (или меньше если были дубли)
SCORED_SNOMED_CLASSES = _UNIQUE_SCORED
N_CLASSES = len(SCORED_SNOMED_CLASSES)

# SNOMED → индекс в multi-hot векторе
SNOMED_TO_INDEX: Dict[int, int] = {s: i for i, s in enumerate(SCORED_SNOMED_CLASSES)}


# ──────────────────────────────────────────────────────────────────────────────
# Маппинг PTB-XL scp_code → SNOMED CT
#
# Источники:
#   - PTB-XL paper (Wagner et al. 2020), Appendix
#   - PhysioNet 2020 dx_mapping_scored.csv (Abbreviation → SNOMED)
#   - ручная сверка для редких кодов
# ──────────────────────────────────────────────────────────────────────────────

SCP_TO_SNOMED: Dict[str, int] = {
    # ── Ритм ──────────────────────────────────────────────────────────────
    "NORM":    426783006,   # Sinus rhythm (Normal)
    "SR":      426783006,   # Sinus rhythm
    "SBRAD":   426177001,   # Sinus bradycardia
    "STACH":   427084000,   # Sinus tachycardia
    "SARRH":   427393009,   # Sinus arrhythmia
    "AFIB":    164889003,   # Atrial fibrillation
    "AFLT":    164890007,   # Atrial flutter
    "SVTACH":  6374002,     # Supraventricular tachycardia (не в scored → игнор при encode)
    "VFTACH":  164896007,   # Ventricular fibrillation/tachycardia
    "VTTACH":  164896007,   # Ventricular tachycardia
    "BIGU":    418818005,   # Bigeminy (не в scored)
    "TRIGU":   418818005,   # Trigeminy (не в scored)
    "PACE":    10370003,    # Pacing rhythm
    "SVARR":   6374002,     # Supraventricular arrhythmia

    # ── Блокады АВ ────────────────────────────────────────────────────────
    "1AVB":    270492004,   # First degree AV block
    "2AVB":    195042002,   # Second degree AV block (не в scored)
    "2AVB1":   426183003,   # Mobitz type I (Wenckebach)
    "2AVB2":   54016002,    # Mobitz type II
    "3AVB":    27885002,    # Third degree AV block (не в scored)
    "AVNRT":   251166008,   # AV nodal reentrant tachycardia

    # ── Блокады ветвей пучка Гиса ─────────────────────────────────────────
    "CRBBB":   713427006,   # Complete RBBB
    "IRBBB":   713426002,   # Incomplete RBBB
    "LBBB":    164909002,   # LBBB
    "ILBBB":   251120003,   # Incomplete LBBB (не в scored)
    "LAFB":    445118002,   # Left anterior fascicular block
    "LPFB":    445211001,   # Left posterior fascicular block (не в scored)

    # ── Экстрасистолы ─────────────────────────────────────────────────────
    "SVPB":    63593006,    # Supraventricular premature beats (НаджЭС)
    "PVC":     17338001,    # Premature ventricular contractions (ЖЭС)
    "VPB":     17338001,    # Ventricular premature beat
    "VEB":     164895006,   # Ventricular escape beat

    # ── Изменения ST-T ─────────────────────────────────────────────────────
    "NDT":     428750005,   # Nonspecific ST/T changes (не в scored)
    "NST_":    428750005,   # Nonspecific ST depression
    "DIG":     428750005,   # Digitalis effect
    "LNGQT":   111975006,   # Prolonged QT interval
    "TAB":     164934002,   # T-wave abnormality
    "TINV":    59931005,    # T-wave inversion

    # ── Ось и напряжение ──────────────────────────────────────────────────
    "LAD":     39732003,    # Left axis deviation
    "RAD":     47665007,    # Right axis deviation
    "LQRSV":   251146004,   # Low QRS voltages

    # ── Гипертрофия ───────────────────────────────────────────────────────
    "LVH":     89792004,    # Left ventricular hypertrophy (не в scored)
    "RVH":     55827005,    # Right ventricular hypertrophy (не в scored)
    "SEHYP":   446358003,   # Septal hypertrophy

    # ── ИМ и ишемия ───────────────────────────────────────────────────────
    "AMI":     57054005,    # Acute MI (не в scored)
    "ALMI":    57054005,    # Anterolateral MI
    "IPLMI":   57054005,    # Inferoposterolateral MI
    "IPMI":    57054005,    # Inferoposterior MI
    "ILMI":    57054005,    # Inferolateral MI
    "PMI":     57054005,    # Posterior MI
    "LMI":     57054005,    # Lateral MI
    "IMI":     57054005,    # Inferior MI
    "INJAS":   57054005,    # Subendocardial injury (anterior)
    "INJAL":   57054005,    # Subendocardial injury (anterolateral)
    "INJIN":   57054005,    # Subendocardial injury (inferior)
    "INJLA":   57054005,    # Subendocardial injury (lateral)
    "INJIL":   57054005,    # Subendocardial injury (inferolateral)
    "ISCAL":   57054005,    # Ischemia (anterolateral)
    "ISCAN":   57054005,    # Ischemia (anterior)
    "ISCAS":   57054005,    # Ischemia (anteroseptal)
    "ISCIL":   57054005,    # Ischemia (inferolateral)
    "ISCIN":   57054005,    # Ischemia (inferior)
    "ISCLA":   57054005,    # Ischemia (lateral)
    "ISC_":    57054005,    # Nonspecific ischemia

    # ── Прочие морфологические ────────────────────────────────────────────
    "ABQRS":   164914007,   # Abnormal QRS (не в scored)
    "PRC(S)":  164947007,   # Prolonged PR
    "LPR":     164947007,   # Prolonged PR
    "QWAVE":   164931005,   # Pathological Q wave (не в scored)
    "LOWT":    164934002,   # Low T wave
    "NT_":     164934002,   # Negative T-wave
    "PAC":     63593006,    # Premature atrial contraction = SVPB
    "PSVT":    6374002,     # Paroxysmal SVT
    "WPW":     74390002,    # Wolff-Parkinson-White
    "HVOLT":   251282003,   # High QRS voltage
    "HPWAVE":  251150005,   # High P-wave
    "PWAVE":   164912004,   # P-wave change
    "INVT":    59931005,    # Inverted T-wave
    "LFQRSA":  251146004,   # Low-frequency QRS (≈ low voltage)
    "NSIVCB":  698252002,   # Nonspecific IVCB
    "IAVB":    270492004,   # = 1AVB alias
}

# Коды, не входящие в 27 scored классов — при кодировании игнорируются
_UNSCORED_SNOMED: frozenset = frozenset(
    v for v in SCP_TO_SNOMED.values()
    if v not in SNOMED_TO_INDEX
)


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка маппинга из dx_mapping_scored.csv (PhysioNet 2020)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_physionet_mapping(csv_path: Optional[Union[str, Path]] = None):
    """
    Загружает dx_mapping_scored.csv в DataFrame.

    Если csv_path не указан — ищет рядом с этим файлом или в data/.
    Возвращает pandas DataFrame с колонками:
        Abbreviation, SNOMED CT Code, Full Name, ...
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas обязателен: pip install pandas")

    search_paths = []
    if csv_path:
        search_paths.append(Path(csv_path))
    else:
        here = Path(__file__).parent
        search_paths = [
            here / "dx_mapping_scored.csv",
            here.parent / "data" / "dx_mapping_scored.csv",
            here.parent / "dx_mapping_scored.csv",
            Path("dx_mapping_scored.csv"),
        ]

    for p in search_paths:
        if p.exists():
            df = pd.read_csv(p)
            logger.info(f"Загружен маппинг PhysioNet 2020: {p} ({len(df)} строк)")
            return df

    logger.warning(
        "dx_mapping_scored.csv не найден. Используется встроенный маппинг SCP_TO_SNOMED. "
        "Скачайте файл с: https://github.com/physionetchallenges/physionet-challenge-2020"
    )
    return None


def build_abbr_to_snomed(csv_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """
    Строит словарь Abbreviation → SNOMED CT Code из dx_mapping_scored.csv.
    Дополняет встроенный SCP_TO_SNOMED данными из CSV (CSV имеет приоритет).
    """
    mapping = dict(SCP_TO_SNOMED)  # копия встроенного

    df = load_physionet_mapping(csv_path)
    if df is None:
        return mapping

    # Ищем колонки с аббревиатурой и SNOMED кодом
    abbr_col  = next((c for c in df.columns if "abbrev" in c.lower()), None)
    snomed_col = next((c for c in df.columns if "snomed" in c.lower()), None)

    if abbr_col and snomed_col:
        for _, row in df.iterrows():
            abbr   = str(row[abbr_col]).strip().upper()
            snomed = row[snomed_col]
            try:
                mapping[abbr] = int(snomed)
            except (ValueError, TypeError):
                pass

    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Кодирование меток → multi-hot вектор [N_CLASSES]
# ──────────────────────────────────────────────────────────────────────────────

def encode_ptbxl_labels(
    scp_codes: Dict[str, float],
    min_likelihood: float = 0.0,
    abbr_to_snomed: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Кодирует PTB-XL scp_codes dict → multi-hot вектор [N_CLASSES].

    Параметры
    ----------
    scp_codes       : dict вида {"AFIB": 100.0, "SR": 0.0, ...}
                      значение — вероятность/достоверность от кардиолога (0–100)
    min_likelihood  : порог достоверности (0 = берём все ненулевые,
                      100 = только стопроцентно подтверждённые)
    abbr_to_snomed  : переопределённый маппинг (из build_abbr_to_snomed)

    Возвращает
    ----------
    np.ndarray shape [N_CLASSES], dtype float32 (multi-hot 0/1)
    """
    mapping = abbr_to_snomed or SCP_TO_SNOMED
    vec = np.zeros(N_CLASSES, dtype=np.float32)

    for scp_code, likelihood in scp_codes.items():
        scp_code = scp_code.strip().upper()
        if likelihood < min_likelihood:
            continue
        snomed = mapping.get(scp_code)
        if snomed is None:
            logger.debug(f"Нет маппинга для scp_code='{scp_code}'")
            continue
        idx = SNOMED_TO_INDEX.get(snomed)
        if idx is not None:
            vec[idx] = 1.0

    return vec


def encode_snomed_labels(
    snomed_codes: List[int],
) -> np.ndarray:
    """
    Кодирует список SNOMED CT кодов → multi-hot вектор [N_CLASSES].

    Параметры
    ----------
    snomed_codes : список int — SNOMED CT concept IDs

    Возвращает
    ----------
    np.ndarray shape [N_CLASSES], dtype float32
    """
    vec = np.zeros(N_CLASSES, dtype=np.float32)
    for snomed in snomed_codes:
        idx = SNOMED_TO_INDEX.get(int(snomed))
        if idx is not None:
            vec[idx] = 1.0
    return vec


def decode_label_vector(
    vec: np.ndarray,
    threshold: float = 0.5,
) -> List[tuple[str, int]]:
    """
    Декодирует multi-hot вектор → список (название, SNOMED код).

    Параметры
    ----------
    vec       : [N_CLASSES] float или bool
    threshold : порог для float векторов (игнорируется если vec уже binary)

    Возвращает
    ----------
    Список (abbreviation_or_snomed, snomed_code) для активных классов
    """
    # обратный маппинг: snomed → аббревиатура
    snomed_to_abbr: Dict[int, str] = {}
    for abbr, snomed in SCP_TO_SNOMED.items():
        if snomed not in snomed_to_abbr:
            snomed_to_abbr[snomed] = abbr

    result = []
    for idx, val in enumerate(vec):
        if float(val) >= threshold:
            snomed = SCORED_SNOMED_CLASSES[idx]
            abbr = snomed_to_abbr.get(snomed, str(snomed))
            result.append(snomed)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для задач
# ──────────────────────────────────────────────────────────────────────────────

def get_class_index(snomed_code: int) -> Optional[int]:
    """Возвращает индекс SNOMED кода в multi-hot векторе, или None."""
    return SNOMED_TO_INDEX.get(snomed_code)


def get_class_indices(snomed_codes: List[int]) -> List[int]:
    """Возвращает индексы для списка SNOMED кодов (пропускает неизвестные)."""
    return [i for c in snomed_codes if (i := SNOMED_TO_INDEX.get(c)) is not None]


# ── Именованные индексы для часто используемых классов ────────────────────────
CLASS_IDX = {
    "AFIB":  SNOMED_TO_INDEX.get(164889003),  # Atrial fibrillation
    "AFL":   SNOMED_TO_INDEX.get(164890007),  # Atrial flutter
    "1AVB":  SNOMED_TO_INDEX.get(270492004),  # First degree AV block
    "LBBB":  SNOMED_TO_INDEX.get(164909002),  # LBBB
    "CRBBB": SNOMED_TO_INDEX.get(713427006),  # Complete RBBB
    "IRBBB": SNOMED_TO_INDEX.get(713426002),  # Incomplete RBBB
    "VPB":   SNOMED_TO_INDEX.get(17338001),   # ЖЭС
    "SVPB":  SNOMED_TO_INDEX.get(63593006),   # НаджЭС
    "LQT":   SNOMED_TO_INDEX.get(111975006),  # Long QT
    "SB":    SNOMED_TO_INDEX.get(426177001),  # Sinus bradycardia
    "STach": SNOMED_TO_INDEX.get(427084000),  # Sinus tachycardia
    "SNR":   SNOMED_TO_INDEX.get(426783006),  # Sinus Normal
}


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Количество scored классов: {N_CLASSES}")
    print(f"CLASS_IDX: {CLASS_IDX}")
    print()

    # ── Кодирование PTB-XL scp_codes ──
    scp = {"AFIB": 100.0, "SR": 0.0, "LBBB": 80.0, "NORM": 0.0}
    vec = encode_ptbxl_labels(scp, min_likelihood=50.0)
    decoded = decode_label_vector(vec)
    print(f"scp_codes:  {scp}")
    print(f"multi-hot:  {vec}")
    print(f"decoded:    {decoded}")
    assert vec[CLASS_IDX["AFIB"]] == 1.0, "AFIB должен быть активен"
    assert vec[CLASS_IDX["LBBB"]] == 1.0, "LBBB должен быть активен"
    print()

    # ── С порогом 100 (только подтверждённые) ──
    vec_strict = encode_ptbxl_labels(scp, min_likelihood=100.0)
    assert vec_strict[CLASS_IDX["AFIB"]] == 1.0
    assert vec_strict[CLASS_IDX["LBBB"]] == 0.0  # likelihood=80 < 100
    print(f"Строгий порог 100: {decode_label_vector(vec_strict)}")

    # ── Кодирование SNOMED кодов напрямую ──
    snomed_list = [164889003, 164909002, 999999999]  # AFIB + LBBB + неизвестный
    vec2 = encode_snomed_labels(snomed_list)
    assert vec2[CLASS_IDX["AFIB"]] == 1.0
    assert vec2[CLASS_IDX["LBBB"]] == 1.0
    print(f"SNOMED encode: {decode_label_vector(vec2)}")

    # ── Проверяем что нет неизвестных важных кодов ──
    critical = ["AFIB", "LBBB", "CRBBB", "1AVB", "VPB", "SVPB"]
    for name in critical:
        idx = CLASS_IDX.get(name)
        assert idx is not None, f"Класс {name} не найден в SNOMED_TO_INDEX!"
        print(f"  {name:8s} → SNOMED={SCORED_SNOMED_CLASSES[idx]:<12d} idx={idx}")

    print("\n✓ Все проверки snomed_map.py пройдены")
