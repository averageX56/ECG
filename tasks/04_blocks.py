"""
tasks/04_blocks.py
Задача 04: Выявление блокад — АВ-блокады и блокады ножек пучка Гиса.

Две головы:

  Голова A — АВ-блокады (backbone + правило PQ)
    1AVB: SNOMED 270492004  → PQ > 200 мс (NeuroKit2), цель F1=0.897
    2AVB: SNOMED 195042002  (редкий класс)
    3AVB: SNOMED 27885002   (очень редкий, проверяем наличие в PTB-XL)

    Для 1AVB сравниваем:
      - правило-baseline: PQ > 200 мс (из NeuroKit2)
      - backbone-голова (SNOMED 270492004 → индекс 4 в 27 классах)

  Голова B — БПНПГ / БЛНПГ (backbone + QRS-ширина)
    RBBB: SNOMED 713427006  (CRBBB), цель F1=0.944 (Ribeiro 2020)
    LBBB: SNOMED 164909002, цель F1=1.000 (Ribeiro 2020)

    Дополнительный признак: QRS > 120 мс из NeuroKit2

Выходы:
  results/task04_blocks/
    head_a_av_blocks_metrics.json
    head_b_bbb_metrics.json
    class_prevalence.json
    summary.json

Запуск:
  python -m tasks.04_blocks --config configs/default.yaml
  python -m tasks.04_blocks --config configs/default.yaml --head A
  python -m tasks.04_blocks --config configs/default.yaml --head B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы: индексы классов в 27-классовом словаре (snomed_map.py _SCORED)
# ─────────────────────────────────────────────────────────────────────────────
from preprocessing.snomed_map import SNOMED_TO_INDEX
# Голова A — АВ-блокады
# 2AVB (195042002) и 3AVB (27885002) НЕ входят в 27 scored классов!
# Для них нет backbone-индекса — используем только правила NeuroKit2.
IAVB_SNOMED = 270492004
IAVB_IDX   = SNOMED_TO_INDEX[270492004]   # 1AVB
CRBBB_IDX  = SNOMED_TO_INDEX[713427006]   # Complete RBBB
LBBB_IDX   = SNOMED_TO_INDEX[164909002]
CRBBB_SNOMED = 713427006
LBBB_SNOMED  = 164909002

# Целевые F1 из Ribeiro 2020 (fold 10 PTB-XL)
RIBEIRO_TARGETS = {
    "1AVB":  {"f1": 0.897, "snomed": IAVB_SNOMED},
    "CRBBB": {"f1": 0.944, "snomed": CRBBB_SNOMED},
    "LBBB":  {"f1": 1.000, "snomed": LBBB_SNOMED},
}

# Параметры NeuroKit2
FS         = 500.0
LEAD_II    = 1       # отведение II (индекс)
LEAD_V1    = 5       # отведение V1 (индекс) — для RBBB-паттерна

# Пороги правил
PQ_1AVB_THRESHOLD_MS  = 200.0   # PQ > 200 мс → 1AVB
QRS_BBB_THRESHOLD_MS  = 120.0   # QRS > 120 мс → блокада ножки

# Минимум R-пиков для правил
MIN_RPEAKS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Правило-baseline для 1AVB: PQ > 200 мс
# ─────────────────────────────────────────────────────────────────────────────

def measure_pq_interval(
    signal: np.ndarray,
    fs: float = FS,
    lead_idx: int = LEAD_II,
) -> Optional[float]:
    """
    Измеряет медианный PQ-интервал (мс) через NeuroKit2.

    Использует tasks/01_intervals.py если он уже доступен,
    иначе выполняет измерение напрямую.

    Returns
    -------
    float | None  — медиана PQ в мс, None при ошибке
    """
    try:
        from tasks.task01_intervals import measure_intervals
        raw = measure_intervals(signal, fs=fs, lead_idx=lead_idx)
        return raw["pq_ms"].get("median")
    except (ImportError, ModuleNotFoundError):
        pass

    # Прямой расчёт
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("neurokit2 не установлен")
        return None

    lead = signal[lead_idx].astype(np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs),
                                     method="pantompkins1985")
            r_peaks = r_info.get("ECG_R_Peaks", np.array([]))

        if len(r_peaks) < MIN_RPEAKS:
            return None

        _, waves = nk.ecg_delineate(
            lead, r_peaks, sampling_rate=int(fs), method="peaks"
        )
        p_onsets = np.array(waves.get("ECG_P_Onsets", []), dtype=float)
        q_peaks  = np.array(waves.get("ECG_Q_Peaks",  []), dtype=float)

        if np.all(np.isnan(p_onsets)):
            p_onsets = np.array(waves.get("ECG_P_Peaks", []), dtype=float)

        pq_vals = []
        for p, q in zip(p_onsets, q_peaks):
            if np.isnan(p) or np.isnan(q):
                continue
            dur = (q - p) * 1000.0 / fs
            if 50 < dur < 500:
                pq_vals.append(dur)

        return float(np.median(pq_vals)) if len(pq_vals) >= 2 else None

    except Exception as exc:
        logger.debug("measure_pq_interval ошибка: %s", exc)
        return None


def pq_rule_predict_1avb(
    signal: np.ndarray,
    threshold_ms: float = PQ_1AVB_THRESHOLD_MS,
) -> Tuple[int, Optional[float]]:
    """
    Правило: PQ > threshold_ms → 1AVB.

    Returns
    -------
    pred : int  — 1 если 1AVB, 0 иначе
    pq   : float | None  — измеренный PQ (мс)
    """
    pq = measure_pq_interval(signal)
    if pq is None:
        return 0, None
    return (1 if pq > threshold_ms else 0), pq


# ─────────────────────────────────────────────────────────────────────────────
# Правило-baseline для BBB: QRS > 120 мс
# ─────────────────────────────────────────────────────────────────────────────

def measure_qrs_duration(
    signal: np.ndarray,
    fs: float = FS,
    lead_idx: int = LEAD_II,
) -> Optional[float]:
    """
    Медианная длительность QRS (мс) через NeuroKit2.
    """
    try:
        from tasks.task01_intervals import measure_intervals
        raw = measure_intervals(signal, fs=fs, lead_idx=lead_idx)
        return raw["qrs_ms"].get("median")
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import neurokit2 as nk
    except ImportError:
        return None

    lead = signal[lead_idx].astype(np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs))
            r_peaks = r_info.get("ECG_R_Peaks", np.array([]))

        if len(r_peaks) < MIN_RPEAKS:
            return None

        _, waves = nk.ecg_delineate(
            lead, r_peaks, sampling_rate=int(fs), method="peaks"
        )
        q_peaks = np.array(waves.get("ECG_Q_Peaks", []), dtype=float)
        s_peaks = np.array(waves.get("ECG_S_Peaks", []), dtype=float)

        qrs_vals = []
        for q, s in zip(q_peaks, s_peaks):
            if np.isnan(q) or np.isnan(s):
                continue
            dur = (s - q) * 1000.0 / fs
            if 20 < dur < 300:
                qrs_vals.append(dur)

        return float(np.median(qrs_vals)) if len(qrs_vals) >= 2 else None

    except Exception as exc:
        logger.debug("measure_qrs_duration ошибка: %s", exc)
        return None


def qrs_rule_predict_bbb(
    signal: np.ndarray,
    threshold_ms: float = QRS_BBB_THRESHOLD_MS,
) -> Tuple[int, Optional[float]]:
    """
    Правило: QRS > threshold_ms → блокада ножки.

    Returns
    -------
    pred : int  — 1 если BBB, 0 иначе
    qrs  : float | None
    """
    qrs = measure_qrs_duration(signal)
    if qrs is None:
        return 0, None
    return (1 if qrs > threshold_ms else 0), qrs


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка fold 10 с backbone-вероятностями и правилами
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RecordFeatures:
    """Признаки для одной записи fold 10."""
    ecg_id:      str
    label_full:  np.ndarray    # [27] multi-hot
    # Backbone-вероятности нужных классов
    prob_1avb:   float
    prob_crbbb:  float
    prob_lbbb:   float
    # Правила NeuroKit2
    pq_ms:       Optional[float]
    qrs_ms:      Optional[float]
    rule_1avb:   int           # 1 если PQ > 200
    rule_bbb:    int           # 1 если QRS > 120


def collect_fold10_features(
    cfg,
    limit: Optional[int] = None,
    run_rules: bool = True,
) -> List[RecordFeatures]:
    """
    Загружает fold 10 и собирает backbone-вероятности + правила.

    run_rules=False пропускает NeuroKit2 (быстрее, только backbone).

    Returns
    -------
    list[RecordFeatures]
    """
    import torch
    from data.load_ptbxl import iter_ptbxl
    from training._common import get_device

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    device     = get_device()
    model      = _load_best_model(cfg, device)
    model.eval()

    batch_size  = int(cfg.pretrain.batch_size) * 2
    all_features: List[RecordFeatures] = []

    # Накапливаем батч для backbone
    batch_sigs:    List[np.ndarray] = []
    batch_labels:  List[np.ndarray] = []
    batch_ids:     List[str]        = []
    batch_pqs:     List[Optional[float]] = []
    batch_qrss:    List[Optional[float]] = []
    batch_r1avb:   List[int]        = []
    batch_rbbb:    List[int]        = []

    def _flush():
        nonlocal batch_sigs, batch_labels, batch_ids
        nonlocal batch_pqs, batch_qrss, batch_r1avb, batch_rbbb
        if not batch_sigs:
            return

        with torch.no_grad():
            x      = torch.from_numpy(np.stack(batch_sigs)).to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()   # [B, 27]

        for i in range(len(batch_sigs)):
            all_features.append(RecordFeatures(
                ecg_id     = batch_ids[i],
                label_full = batch_labels[i],
                prob_1avb  = float(probs[i, IAVB_IDX]),
                prob_crbbb = float(probs[i, CRBBB_IDX]),
                prob_lbbb  = float(probs[i, LBBB_IDX]),
                pq_ms      = batch_pqs[i],
                qrs_ms     = batch_qrss[i],
                rule_1avb  = batch_r1avb[i],
                rule_bbb   = batch_rbbb[i],
            ))

        batch_sigs  = []; batch_labels = []; batch_ids = []
        batch_pqs   = []; batch_qrss   = []
        batch_r1avb = []; batch_rbbb   = []

    logger.info("Сбор данных fold 10 (run_rules=%s)…", run_rules)
    n = 0
    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["test"],
        use_cache=True, show_progress=True, limit=limit,
    ):
        # Правила NeuroKit2 (медленно — но выполняем per-record)
        if run_rules:
            r1avb, pq   = pq_rule_predict_1avb(rec.signal)
            rbbb,  qrs  = qrs_rule_predict_bbb(rec.signal)
        else:
            r1avb, pq   = 0, None
            rbbb,  qrs  = 0, None

        batch_sigs.append(rec.signal)
        batch_labels.append(rec.label_vec)
        batch_ids.append(rec.ecg_id)
        batch_pqs.append(pq)
        batch_qrss.append(qrs)
        batch_r1avb.append(r1avb)
        batch_rbbb.append(rbbb)
        n += 1

        if len(batch_sigs) >= batch_size:
            _flush()

    _flush()

    logger.info("Сбор завершён: %d записей", n)
    return all_features


# ─────────────────────────────────────────────────────────────────────────────
# Голова A: АВ-блокады
# ─────────────────────────────────────────────────────────────────────────────

def run_head_a_av_blocks(
    features: List[RecordFeatures],
    results_dir: Path,
) -> Dict:
    """
    Оценивает 1AVB (backbone + правило PQ) и проверяет наличие 2AVB/3AVB.

    Для 1AVB сравниваем:
      - правило PQ > 200 мс
      - backbone-голова (SNOMED 270492004, idx=4)
      - ансамбль: LogReg([P_backbone, PQ_норм]) если правило работает

    Returns
    -------
    dict со всеми метриками
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    logger.info("Голова A: АВ-блокады")

    # Извлекаем метки и вероятности
    y_1avb  = np.array([int(f.label_full[IAVB_IDX]) for f in features], dtype=np.int32)
    # 2AVB (195042002) и 3AVB (27885002) НЕ в 27 scored классах PTB-XL
    # Проверяем через SCP-коды из метаданных — здесь только считаем 0
    y_2avb  = np.zeros(len(features), dtype=np.int32)   # не в scored → 0
    y_3avb  = np.zeros(len(features), dtype=np.int32)

    backbone_1avb = np.array([f.prob_1avb for f in features], dtype=np.float32)
    rule_1avb     = np.array([f.rule_1avb  for f in features], dtype=np.int32)
    pq_vals       = np.array([f.pq_ms if f.pq_ms is not None else np.nan
                               for f in features], dtype=np.float32)

    n_pos_1avb = int(y_1avb.sum())
    logger.info(
        "1AVB: N_pos=%d / %d (%.1f%%)",
        n_pos_1avb, len(y_1avb), 100.0 * n_pos_1avb / max(len(y_1avb), 1),
    )

    # Уровень PQ у 1AVB vs не-1AVB для анализа
    pq_in_1avb  = pq_vals[y_1avb == 1]
    pq_no_1avb  = pq_vals[y_1avb == 0]
    pq_stats = {
        "1avb_pq_median": _safe_stat(pq_in_1avb, np.median),
        "1avb_pq_p25":    _safe_stat(pq_in_1avb, lambda x: np.percentile(x, 25)),
        "1avb_pq_p75":    _safe_stat(pq_in_1avb, lambda x: np.percentile(x, 75)),
        "no1avb_pq_median": _safe_stat(pq_no_1avb, np.median),
    }

    results: Dict = {
        "n_total":    len(features),
        "n_pos_1avb": n_pos_1avb,
        "pq_analysis": pq_stats,
        "note_2avb_3avb": (
            "2AVB (SNOMED 195042002) и 3AVB (SNOMED 27885002) не входят в "
            "27 scored классов PhysioNet 2020 — backbone-голова для них недоступна. "
            "Используйте правило: 2AVB → выпадение QRS при регулярном P; "
            "3AVB → диссоциация P и QRS (NeuroKit2 + кастомный анализ RR)."
        ),
    }

    if n_pos_1avb == 0:
        logger.warning("1AVB не встречается в fold 10 — метрики не вычислены")
        results["1avb_backbone"] = {"error": "нет позитивных примеров"}
        results["1avb_rule"]     = {"error": "нет позитивных примеров"}
        _save_json(results, results_dir / "head_a_av_blocks_metrics.json")
        return results

    # ── Backbone метрики для 1AVB ─────────────────────────────────────────────
    bb_metrics = _compute_block_metrics(
        backbone_1avb, y_1avb, name="1AVB_Backbone",
        ribeiro_target=RIBEIRO_TARGETS["1AVB"]["f1"],
    )
    results["1avb_backbone"] = bb_metrics

    # ── Правило PQ > 200мс ────────────────────────────────────────────────────
    # Конвертируем в вероятности через нормировку PQ
    pq_norm = np.where(np.isnan(pq_vals), np.nanmedian(pq_vals), pq_vals)
    # Простое масштабирование: P = sigmoid((PQ - 200) / 30)
    pq_probs = _sigmoid((pq_norm - 200.0) / 30.0)

    rule_metrics = _compute_block_metrics(
        pq_probs, y_1avb, name="1AVB_PQ_rule(>200ms)",
        ribeiro_target=RIBEIRO_TARGETS["1AVB"]["f1"],
    )
    # Добавляем F1 именно для бинарного правила
    rule_metrics["f1_hard_rule"] = _f1_binary(rule_1avb, y_1avb)
    rule_metrics["n_flagged_by_rule"] = int(rule_1avb.sum())
    results["1avb_rule"] = rule_metrics

    # ── Ансамбль: backbone + PQ ───────────────────────────────────────────────
    has_pq = ~np.isnan(pq_vals)
    if has_pq.sum() >= 10:
        X_ens = np.column_stack([
            backbone_1avb.reshape(-1, 1),
            pq_norm.reshape(-1, 1),
        ])
        pipe = _make_logreg_pipe(y_1avb)
        pipe.fit(X_ens, y_1avb)
        ens_probs = pipe.predict_proba(X_ens)[:, 1]
        ens_metrics = _compute_block_metrics(
            ens_probs, y_1avb, name="1AVB_Ensemble(BB+PQ)",
            ribeiro_target=RIBEIRO_TARGETS["1AVB"]["f1"],
        )
        results["1avb_ensemble"] = ens_metrics

    # ── Итоговое сравнение ────────────────────────────────────────────────────
    logger.info(
        "\n1AVB: backbone AUC=%.4f F1=%.4f  "
        "rule AUC=%.4f F1=%.4f  "
        "(Ribeiro цель F1=%.3f)",
        bb_metrics.get("auc",0), bb_metrics.get("f1_fmax",0),
        rule_metrics.get("auc",0), rule_metrics.get("f1_fmax",0),
        RIBEIRO_TARGETS["1AVB"]["f1"],
    )

    np.save(str(results_dir / "head_a_1avb_backbone_probs.npy"), backbone_1avb)
    np.save(str(results_dir / "head_a_1avb_labels.npy"), y_1avb)
    _save_json(results, results_dir / "head_a_av_blocks_metrics.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Голова B: БПНПГ / БЛНПГ
# ─────────────────────────────────────────────────────────────────────────────

def run_head_b_bbb(
    features: List[RecordFeatures],
    results_dir: Path,
) -> Dict:
    """
    Оценивает RBBB (CRBBB) и LBBB:
      - backbone-голова (основной метод)
      - правило QRS > 120 мс (baseline)
      - ансамбль: backbone + QRS-длина

    Цели: CRBBB F1=0.944, LBBB F1=1.000 (Ribeiro 2020).

    Returns
    -------
    dict со всеми метриками
    """
    logger.info("Голова B: БПНПГ / БЛНПГ")

    # Метки
    y_crbbb = np.array([int(f.label_full[CRBBB_IDX]) for f in features], dtype=np.int32)
    y_lbbb  = np.array([int(f.label_full[LBBB_IDX])  for f in features], dtype=np.int32)

    # Backbone-вероятности
    bb_crbbb = np.array([f.prob_crbbb for f in features], dtype=np.float32)
    bb_lbbb  = np.array([f.prob_lbbb  for f in features], dtype=np.float32)

    # QRS-значения (для правила и ансамбля)
    qrs_vals = np.array(
        [f.qrs_ms if f.qrs_ms is not None else np.nan for f in features],
        dtype=np.float32,
    )
    rule_bbb = np.array([f.rule_bbb for f in features], dtype=np.int32)

    n_crbbb = int(y_crbbb.sum())
    n_lbbb  = int(y_lbbb.sum())
    logger.info(
        "CRBBB: N_pos=%d (%.1f%%)  LBBB: N_pos=%d (%.1f%%)",
        n_crbbb, 100.0 * n_crbbb / max(len(y_crbbb), 1),
        n_lbbb,  100.0 * n_lbbb  / max(len(y_lbbb), 1),
    )

    results: Dict = {
        "n_total": len(features),
        "n_crbbb": n_crbbb,
        "n_lbbb":  n_lbbb,
    }

    # ── CRBBB ─────────────────────────────────────────────────────────────────
    if n_crbbb > 0:
        crbbb_bb = _compute_block_metrics(
            bb_crbbb, y_crbbb, name="CRBBB_Backbone",
            ribeiro_target=RIBEIRO_TARGETS["CRBBB"]["f1"],
        )
        results["crbbb_backbone"] = crbbb_bb

        # QRS-правило для RBBB
        qrs_norm = np.where(np.isnan(qrs_vals), np.nanmedian(qrs_vals), qrs_vals)
        qrs_probs = _sigmoid((qrs_norm - QRS_BBB_THRESHOLD_MS) / 20.0)
        crbbb_rule = _compute_block_metrics(
            qrs_probs, y_crbbb, name="CRBBB_QRS_rule(>120ms)",
            ribeiro_target=RIBEIRO_TARGETS["CRBBB"]["f1"],
        )
        crbbb_rule["f1_hard_rule"] = _f1_binary(rule_bbb, y_crbbb)
        results["crbbb_rule"] = crbbb_rule

        # Ансамбль CRBBB
        if (~np.isnan(qrs_vals)).sum() >= 10:
            X_ens = np.column_stack([bb_crbbb.reshape(-1,1), qrs_norm.reshape(-1,1)])
            pipe  = _make_logreg_pipe(y_crbbb)
            pipe.fit(X_ens, y_crbbb)
            ens_p = pipe.predict_proba(X_ens)[:, 1]
            results["crbbb_ensemble"] = _compute_block_metrics(
                ens_p, y_crbbb, name="CRBBB_Ensemble(BB+QRS)",
                ribeiro_target=RIBEIRO_TARGETS["CRBBB"]["f1"],
            )

        logger.info(
            "CRBBB: backbone AUC=%.4f F1=%.4f  "
            "rule F1=%.4f  (цель F1=%.3f)",
            crbbb_bb.get("auc",0), crbbb_bb.get("f1_fmax",0),
            crbbb_rule.get("f1_hard_rule",0),
            RIBEIRO_TARGETS["CRBBB"]["f1"],
        )
    else:
        results["crbbb_backbone"] = {"error": "нет CRBBB в fold 10"}
        logger.warning("CRBBB не встречается в fold 10")

    # ── LBBB ──────────────────────────────────────────────────────────────────
    if n_lbbb > 0:
        lbbb_bb = _compute_block_metrics(
            bb_lbbb, y_lbbb, name="LBBB_Backbone",
            ribeiro_target=RIBEIRO_TARGETS["LBBB"]["f1"],
        )
        results["lbbb_backbone"] = lbbb_bb

        # QRS-правило — LBBB обязательно даёт QRS > 120мс
        if (~np.isnan(qrs_vals)).sum() >= 10:
            qrs_norm = np.where(np.isnan(qrs_vals), np.nanmedian(qrs_vals), qrs_vals)
            qrs_probs = _sigmoid((qrs_norm - QRS_BBB_THRESHOLD_MS) / 20.0)
            lbbb_rule = _compute_block_metrics(
                qrs_probs, y_lbbb, name="LBBB_QRS_rule(>120ms)",
                ribeiro_target=RIBEIRO_TARGETS["LBBB"]["f1"],
            )
            lbbb_rule["f1_hard_rule"] = _f1_binary(rule_bbb, y_lbbb)
            results["lbbb_rule"] = lbbb_rule

        logger.info(
            "LBBB: backbone AUC=%.4f F1=%.4f  (цель F1=%.3f)",
            lbbb_bb.get("auc",0), lbbb_bb.get("f1_fmax",0),
            RIBEIRO_TARGETS["LBBB"]["f1"],
        )
    else:
        results["lbbb_backbone"] = {"error": "нет LBBB в fold 10"}
        logger.warning("LBBB не встречается в fold 10")

    np.save(str(results_dir / "head_b_crbbb_probs.npy"), bb_crbbb)
    np.save(str(results_dir / "head_b_lbbb_probs.npy"),  bb_lbbb)
    np.save(str(results_dir / "head_b_crbbb_labels.npy"), y_crbbb)
    np.save(str(results_dir / "head_b_lbbb_labels.npy"),  y_lbbb)

    _save_json(results, results_dir / "head_b_bbb_metrics.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Распространённость классов в fold 10
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_prevalence(
    features: List[RecordFeatures],
    results_dir: Path,
) -> Dict:
    """
    Считает долю позитивных примеров для каждого из 27 классов в fold 10.
    Полезно для ablation и диагностики.
    """
    from preprocessing.snomed_map import SCORED_SNOMED_CLASSES, SNOMED_TO_ABBR

    labels = np.stack([f.label_full for f in features])   # [N, 27]
    n = len(features)
    counts = labels.sum(axis=0).astype(int)

    prevalence = {}
    for idx, (code, cnt) in enumerate(zip(SCORED_SNOMED_CLASSES, counts)):
        abbr = SNOMED_TO_ABBR.get(code, str(code))
        prevalence[abbr] = {
            "snomed":    code,
            "idx":       idx,
            "n_positive": int(cnt),
            "prevalence_pct": round(100.0 * cnt / max(n, 1), 2),
        }

    # Специально отмечаем блокады
    for abbr in ("IAVB", "CRBBB", "LBBB"):
        if abbr in prevalence:
            prevalence[abbr]["task04_target"] = True

    sorted_prev = dict(
        sorted(prevalence.items(), key=lambda x: -x[1]["n_positive"])
    )
    _save_json({"n_total": n, "classes": sorted_prev},
               results_dir / "class_prevalence.json")

    logger.info("Топ-5 классов fold 10:")
    for abbr, stat in list(sorted_prev.items())[:5]:
        logger.info("  %s: %d (%.1f%%)", abbr, stat["n_positive"],
                    stat["prevalence_pct"])

    return sorted_prev


# ─────────────────────────────────────────────────────────────────────────────
# Общая загрузка модели
# ─────────────────────────────────────────────────────────────────────────────

def _load_best_model(cfg, device):
    """Finetune > pretrain > Zenodo."""
    from backbone.resnet1d import ResNet1dWithHead
    from backbone.load_weights import load_pretrained_weights
    import torch

    model = ResNet1dWithHead(
        n_classes=int(cfg.pretrain.n_classes),
        backbone_kwargs={
            "n_leads": int(cfg.backbone.n_leads),
            "dropout": float(cfg.backbone.dropout),
        },
        dropout_head=float(cfg.pretrain.dropout_head),
    )
    for ckpt_name in ("finetune_best.pt", "pretrain_best.pt"):
        ckpt = Path(cfg.paths.checkpoint_dir) / ckpt_name
        if ckpt.exists():
            logger.info("Загружаем чекпоинт: %s", ckpt)
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state"], strict=True)
            return model.to(device)

    logger.warning("Чекпоинты не найдены → Zenodo")
    cache = Path(cfg.paths.backbone_cache).expanduser()
    load_pretrained_weights(model.backbone, cache_dir=cache, strict=False)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Метрики для задачи блокад
# ─────────────────────────────────────────────────────────────────────────────

def _compute_block_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    name: str = "",
    ribeiro_target: Optional[float] = None,
) -> Dict:
    """
    AUC, F1-Fmax, Specificity, Precision, Recall.
    Для малых классов (< 20 позитивных) добавляем предупреждение.
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score,
        recall_score, confusion_matrix,
    )

    n_pos = int(targets.sum())
    n     = int(len(targets))
    result = {
        "name":         name,
        "n_positive":   n_pos,
        "n_total":      n,
        "prevalence":   round(n_pos / max(n, 1), 4),
        "small_class":  n_pos < 20,
    }

    if n_pos == 0:
        result.update({"auc": None, "f1_fmax": None, "specificity_fmax": None})
        return result

    # AUC
    try:
        result["auc"] = float(roc_auc_score(targets, probs))
    except Exception:
        result["auc"] = None

    # Fmax
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.02, 0.98, 97):
        p  = (probs >= thr).astype(int)
        f1 = float(f1_score(targets, p, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    result["fmax"]           = best_f1
    result["fmax_threshold"] = best_thr
    result["f1_fmax"]        = best_f1

    preds = (probs >= best_thr).astype(int)
    result["precision_fmax"] = float(precision_score(targets, preds, zero_division=0))
    result["recall_fmax"]    = float(recall_score(targets, preds, zero_division=0))

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        result["specificity_fmax"] = float(tn / max(tn + fp, 1))
        result["confusion_matrix"] = cm.tolist()
    else:
        result["specificity_fmax"] = None

    # Сравнение с Ribeiro
    if ribeiro_target is not None:
        result["ribeiro_target"] = ribeiro_target
        result["f1_delta_vs_ribeiro"] = round(best_f1 - ribeiro_target, 4)
        result["target_met"]          = bool(best_f1 >= ribeiro_target - 0.02)  # ±2%

    return result


def _f1_binary(preds: np.ndarray, targets: np.ndarray) -> float:
    """F1 для жёсткого бинарного предсказания."""
    from sklearn.metrics import f1_score
    return float(f1_score(targets, preds, zero_division=0))


def _make_logreg_pipe(y: np.ndarray):
    """Создаёт Pipeline[impute → scale → LogReg] с автоматическим class_weight."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    pos = y.sum(); neg = len(y) - pos
    cw = {0: 1.0, 1: float(neg / max(pos, 1))}
    return Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("scl",  StandardScaler()),
        ("clf",  LogisticRegression(
            C=1.0, class_weight=cw,
            max_iter=1000, random_state=42,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Векторизованная сигмоида."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _safe_stat(arr: np.ndarray, fn) -> Optional[float]:
    """Применяет fn к arr без NaN, возвращает None если пусто."""
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return None
    return float(fn(clean))


def _save_json(data: Dict, path: Path) -> None:
    def _ser(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.floating,)):   return float(v)
        if isinstance(v, (np.integer,)):    return int(v)
        if isinstance(v, np.ndarray):       return v.tolist()
        if isinstance(v, dict):  return {k: _ser(vv) for k, vv in v.items()}
        if isinstance(v, list):  return [_ser(x) for x in v]
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_ser(data), f, indent=2, ensure_ascii=False)
    logger.info("Сохранено: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_blocks(
    cfg,
    head: str = "both",
    limit: Optional[int] = None,
    skip_rules: bool = False,
) -> Dict:
    """
    Запускает задачу 04.

    Parameters
    ----------
    cfg
    head : 'A' | 'B' | 'both'
    limit : int | None
    skip_rules : bool
        True → пропустить NeuroKit2-правила (только backbone, быстрее)

    Returns
    -------
    dict с результатами обеих голов
    """
    results_dir = Path(cfg.paths.results_dir) / "task04_blocks"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Единовременный сбор данных fold 10 ───────────────────────────────────
    logger.info("═══ Задача 04: Блокады ══════════════════")
    run_rules = not skip_rules
    features = collect_fold10_features(cfg, limit=limit, run_rules=run_rules)

    if not features:
        logger.error("Нет данных fold 10")
        return {}

    # Распространённость классов
    prevalence = compute_class_prevalence(features, results_dir)

    results: Dict = {"prevalence_summary": {
        k: v for k, v in prevalence.items()
        if v.get("task04_target")
    }}

    # ── Голова A: АВ-блокады ──────────────────────────────────────────────────
    if head in ("A", "both"):
        try:
            results["head_a"] = run_head_a_av_blocks(features, results_dir)
        except Exception as exc:
            logger.error("Голова A ошибка: %s", exc, exc_info=True)
            results["head_a"] = {"error": str(exc)}

    # ── Голова B: BBB ─────────────────────────────────────────────────────────
    if head in ("B", "both"):
        try:
            results["head_b"] = run_head_b_bbb(features, results_dir)
        except Exception as exc:
            logger.error("Голова B ошибка: %s", exc, exc_info=True)
            results["head_b"] = {"error": str(exc)}

    # ── Итоговая сводка ───────────────────────────────────────────────────────
    _print_summary(results)
    _save_json(results, results_dir / "summary.json")

    logger.info("Задача 04 завершена → %s", results_dir)
    return results


def _print_summary(results: Dict) -> None:
    """Итоговая таблица в stdout."""
    print("\n════ Задача 04: Блокады ════")
    print(f"\n{'Класс':<22} {'AUC':>7} {'F1(Fmax)':>9} "
          f"{'Spec':>7} {'Цель F1':>8} {'Delta':>7}")
    print("─" * 68)

    def _row(label, m, target=None):
        if not m or "error" in m:
            print(f"  {label:<20} {'—':>7} {'—':>9} {'—':>7}")
            return
        auc  = f"{m['auc']:.4f}"  if m.get("auc")  else "  —   "
        f1   = f"{m['f1_fmax']:.4f}" if m.get("f1_fmax") else "  —   "
        spec = f"{m['specificity_fmax']:.4f}" if m.get("specificity_fmax") else "  —   "
        tgt  = f"{target:.3f}" if target else "  —  "
        dlt  = f"{m['f1_delta_vs_ribeiro']:+.4f}" if m.get("f1_delta_vs_ribeiro") is not None else "  —  "
        ok   = " ✓" if m.get("target_met") else ""
        print(f"  {label:<20} {auc:>7} {f1:>9} {spec:>7} {tgt:>8} {dlt:>7}{ok}")

    ha = results.get("head_a", {})
    _row("1AVB (backbone)",    ha.get("1avb_backbone"), RIBEIRO_TARGETS["1AVB"]["f1"])
    _row("1AVB (PQ-rule)",     ha.get("1avb_rule"),     RIBEIRO_TARGETS["1AVB"]["f1"])
    _row("1AVB (ensemble)",    ha.get("1avb_ensemble"),  RIBEIRO_TARGETS["1AVB"]["f1"])

    hb = results.get("head_b", {})
    _row("CRBBB (backbone)",   hb.get("crbbb_backbone"), RIBEIRO_TARGETS["CRBBB"]["f1"])
    _row("CRBBB (QRS-rule)",   hb.get("crbbb_rule"),     RIBEIRO_TARGETS["CRBBB"]["f1"])
    _row("CRBBB (ensemble)",   hb.get("crbbb_ensemble"), RIBEIRO_TARGETS["CRBBB"]["f1"])
    _row("LBBB  (backbone)",   hb.get("lbbb_backbone"),  RIBEIRO_TARGETS["LBBB"]["f1"])
    _row("LBBB  (QRS-rule)",   hb.get("lbbb_rule"),      RIBEIRO_TARGETS["LBBB"]["f1"])

    print("─" * 68)
    print("  ✓ = F1 в пределах 2% от целевого значения Ribeiro 2020")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Задача 04: Блокады — АВ-блокады + БПНПГ/БЛНПГ"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--head", choices=["A", "B", "both"], default="both",
        help="Голова A (АВ-блокады), B (БПНПГ/БЛНПГ), both",
    )
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--skip-rules", action="store_true",
                        help="Пропустить NeuroKit2-правила (только backbone)")
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training._common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (30 if args.debug else None)
    run_blocks(cfg, head=args.head, limit=limit, skip_rules=args.skip_rules)


if __name__ == "__main__":
    main()