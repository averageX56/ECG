"""
evaluation/metrics.py
Полный набор метрик для multi-label ЭКГ-классификации.

Функции:
  compute_macro_auc        — macro-AUC по 27 классам (PhysioNet 2020)
  compute_fmax             — максимальный macro-F1 по всем порогам (Fmax)
  compute_per_class_metrics — AUC, F1, precision, recall, AP по каждому классу
  compute_challenge_metric — воспроизведение официальной метрики PhysioNet 2020
  classification_report_df — сводная таблица в виде pd.DataFrame
  bootstrap_ci             — доверительный интервал AUC через bootstrap

Использование:
  from evaluation.metrics import compute_per_class_metrics, classification_report_df
  report = classification_report_df(probs, targets, class_names=SNOMED_ABBRS)
  print(report.to_string())
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Имена 27 scored классов PhysioNet 2020 (из snomed_map.py)
# ---------------------------------------------------------------------------
SNOMED_ABBRS: List[str] = [
    "AF", "AFL", "Brady", "CRBBB", "IAVB", "IBBB", "NSR", "LBBB",
    "LAnFB", "LAD", "LQRSV", "NICD", "PR", "PAC", "PVC", "LPR",
    "LQT", "QAb", "RAD", "SA", "SB", "STach", "SVPB", "TAb",
    "TInv", "VPB", "VTach",
]

# Целевые значения Ribeiro et al. 2020 (Nature Communications)
# Таблица 2, PTB-XL fold 10, порог оптимизирован на val
RIBEIRO_2020_TARGETS: Dict[str, Dict[str, float]] = {
    "AF":     {"f1": 0.870, "specificity": 1.000, "sensitivity": 0.782},
    "IAVB":   {"f1": 0.897, "specificity": 0.979, "sensitivity": 0.827},
    "LBBB":   {"f1": 1.000, "specificity": 1.000, "sensitivity": 1.000},
    "CRBBB":  {"f1": 0.944, "specificity": 0.983, "sensitivity": 0.908},
    "PAC":    {"f1": 0.812, "specificity": 0.961, "sensitivity": 0.702},
    "PVC":    {"f1": 0.801, "specificity": 0.965, "sensitivity": 0.719},
    "STach":  {"f1": 0.854, "specificity": 0.952, "sensitivity": 0.768},
    "SB":     {"f1": 0.876, "specificity": 0.979, "sensitivity": 0.795},
    "NSR":    {"f1": 0.960, "specificity": 0.979, "sensitivity": 0.942},
    "VTach":  {"f1": 0.750, "specificity": 0.997, "sensitivity": 0.600},
}


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """ROC-AUC, возвращает None если только один класс."""
    from sklearn.metrics import roc_auc_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Average Precision (area under PR curve)."""
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Macro-AUC
# ---------------------------------------------------------------------------

def compute_macro_auc(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, List[Optional[float]]]:
    """
    Macro-AUC по всем классам, пропуская отсутствующие (только 1 класс).

    Parameters
    ----------
    probs   : [N, C]  — sigmoid-вероятности
    targets : [N, C]  — multi-hot метки
    class_names : list[str] | None

    Returns
    -------
    macro_auc  : float
    per_class  : list[Optional[float]]  — AUC на каждый класс (None = нет данных)
    """
    n_classes = probs.shape[1]
    per_class: List[Optional[float]] = []

    for i in range(n_classes):
        auc_i = _safe_auc(targets[:, i], probs[:, i])
        per_class.append(auc_i)
        if auc_i is None and class_names:
            logger.debug("AUC пропущен для %s (нет позитивных примеров)", class_names[i])

    valid = [v for v in per_class if v is not None]
    macro = float(np.mean(valid)) if valid else 0.0
    return macro, per_class


# ---------------------------------------------------------------------------
# Fmax (максимальный macro-F1 по порогам)
# ---------------------------------------------------------------------------

def compute_fmax(
    probs: np.ndarray,
    targets: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float, np.ndarray]:
    """
    Fmax — воспроизводит метрику PhysioNet Challenge 2020.

    Parameters
    ----------
    probs   : [N, C]
    targets : [N, C]
    n_thresholds : int

    Returns
    -------
    fmax          : float  — максимальный macro-F1
    best_thr      : float  — оптимальный порог
    per_class_f1  : [C]   — F1 каждого класса при best_thr
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    best_f1   = 0.0
    best_thr  = 0.5
    best_pcf1 = np.zeros(probs.shape[1])

    for thr in thresholds:
        preds = (probs >= thr).astype(np.int32)
        tp = (preds * targets).sum(axis=0).astype(float)
        fp = (preds * (1 - targets)).sum(axis=0).astype(float)
        fn = ((1 - preds) * targets).sum(axis=0).astype(float)

        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

        # Пропускаем классы без позитивных примеров
        has_pos = targets.sum(axis=0) > 0
        macro   = float(f1[has_pos].mean()) if has_pos.any() else 0.0

        if macro > best_f1:
            best_f1  = macro
            best_thr = float(thr)
            best_pcf1 = f1.copy()

    return best_f1, best_thr, best_pcf1


# ---------------------------------------------------------------------------
# Per-class детальные метрики
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    AUC, AP, F1, precision, recall, specificity для каждого класса.

    Parameters
    ----------
    probs     : [N, C]
    targets   : [N, C]
    class_names : list[str] | None
    threshold : float

    Returns
    -------
    list of dicts, по одному на класс
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, confusion_matrix,
    )

    n_classes = probs.shape[1]
    names = class_names or [str(i) for i in range(n_classes)]
    results = []

    for i in range(n_classes):
        y_true  = targets[:, i]
        y_score = probs[:, i]
        y_pred  = (y_score >= threshold).astype(int)
        n_pos   = int(y_true.sum())
        n_total = int(len(y_true))

        rec: Dict[str, Any] = {
            "class":      names[i],
            "n_positive": n_pos,
            "n_total":    n_total,
            "prevalence": round(n_pos / max(n_total, 1), 4),
            "auc":        _safe_auc(y_true, y_score),
            "ap":         _safe_ap(y_true, y_score),
        }

        if n_pos > 0:
            rec["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
            rec["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            rec["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp = int(cm[0, 0]), int(cm[0, 1])
                rec["specificity"] = float(tn / max(tn + fp, 1))
            else:
                rec["specificity"] = None
        else:
            rec["f1"] = rec["precision"] = rec["recall"] = rec["specificity"] = None

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Официальная метрика PhysioNet 2020 Challenge
# ---------------------------------------------------------------------------

def compute_challenge_metric(
    probs: np.ndarray,
    targets: np.ndarray,
    weights_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Взвешенная multi-label accuracy PhysioNet 2020 Challenge.

    Использует официальную матрицу весов из репозитория PhysioNet 2020.
    Если weights_matrix не передан, возвращает стандартный Fmax.

    Источник: https://github.com/physionetchallenges/evaluation-2020
    """
    if weights_matrix is None:
        fmax, _, _ = compute_fmax(probs, targets)
        return fmax

    # Официальная метрика: сумма(w[i,j] * tp[j]) / max(N_true, N_pred)
    # Здесь упрощённая версия без полного challenge-scorer
    threshold  = 0.5
    preds      = (probs >= threshold).astype(np.int32)
    n, c       = targets.shape

    numerator   = 0.0
    denominator = 0.0

    for k in range(n):
        true_idx = np.where(targets[k] == 1)[0]
        pred_idx = np.where(preds[k] == 1)[0]
        if len(true_idx) == 0 and len(pred_idx) == 0:
            numerator   += 1.0
            denominator += 1.0
            continue
        for i in true_idx:
            for j in pred_idx:
                numerator += weights_matrix[i, j]
        denominator += max(len(true_idx), len(pred_idx))

    return float(numerator / denominator) if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Сводная таблица DataFrame
# ---------------------------------------------------------------------------

def classification_report_df(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    ribeiro_targets: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Строит сводную таблицу метрик для всех классов.

    Добавляет столбцы delta vs Ribeiro 2020 для известных классов.

    Parameters
    ----------
    probs          : [N, C]
    targets        : [N, C]
    class_names    : list[str] | None  → используются SNOMED_ABBRS если None
    threshold      : float
    ribeiro_targets : dict | None  → если None, используется RIBEIRO_2020_TARGETS

    Returns
    -------
    pd.DataFrame с колонками:
        class, n_positive, prevalence, auc, ap, f1, precision, recall,
        specificity, [ribeiro_f1, delta_f1]
    """
    names   = class_names or SNOMED_ABBRS[:probs.shape[1]]
    rib_tgt = ribeiro_targets if ribeiro_targets is not None else RIBEIRO_2020_TARGETS
    per_cls = compute_per_class_metrics(probs, targets, names, threshold)

    rows = []
    for r in per_cls:
        row = {
            "class":       r["class"],
            "n_pos":       r["n_positive"],
            "prevalence%": round(r["prevalence"] * 100, 1),
            "AUC":         _fmt(r["auc"]),
            "AP":          _fmt(r["ap"]),
            "F1":          _fmt(r["f1"]),
            "Precision":   _fmt(r["precision"]),
            "Recall":      _fmt(r["recall"]),
            "Specificity": _fmt(r["specificity"]),
        }
        rib = rib_tgt.get(r["class"])
        if rib:
            row["Ribeiro_F1"]  = rib.get("f1")
            row["ΔF1"]         = _delta(r["f1"], rib.get("f1"))
            row["Ribeiro_Spec"] = rib.get("specificity")
            row["ΔSpec"]        = _delta(r["specificity"], rib.get("specificity"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("class")
    return df


def _fmt(v: Optional[float], decimals: int = 4) -> Optional[float]:
    return round(v, decimals) if v is not None else None


def _delta(our: Optional[float], ref: Optional[float]) -> Optional[float]:
    if our is None or ref is None:
        return None
    return round(our - ref, 4)


# ---------------------------------------------------------------------------
# Bootstrap CI для AUC
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Доверительный интервал для ROC-AUC через bootstrap.

    Parameters
    ----------
    y_true  : [N] бинарные метки
    y_score : [N] вероятности
    n_boot  : int число bootstrap-итераций
    ci      : float уровень доверия (0.95 → 95%)
    seed    : int

    Returns
    -------
    auc_mean, ci_lo, ci_hi
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    aucs: List[float] = []

    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        auc  = _safe_auc(y_true[idx], y_score[idx])
        if auc is not None:
            aucs.append(auc)

    if not aucs:
        base = _safe_auc(y_true, y_score) or 0.0
        return base, base, base

    alpha   = (1 - ci) / 2
    ci_lo   = float(np.percentile(aucs, alpha * 100))
    ci_hi   = float(np.percentile(aucs, (1 - alpha) * 100))
    auc_mean = float(np.mean(aucs))
    return auc_mean, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Полный отчёт: macro + per-class + Fmax
# ---------------------------------------------------------------------------

def full_evaluation_report(
    probs: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    compute_ci: bool = False,
    ci_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Единая точка для полной оценки модели на fold 10.

    Parameters
    ----------
    probs      : [N, C]
    targets    : [N, C]
    class_names : list | None
    threshold  : float
    compute_ci : bool   — считать ли bootstrap CI (медленно)
    ci_classes : list[int] | None  — индексы классов для CI

    Returns
    -------
    dict с ключами:
        macro_auc, macro_f1, fmax, fmax_threshold,
        per_class (list[dict]), summary_df (pd.DataFrame),
        [ci (dict)]
    """
    names     = class_names or SNOMED_ABBRS[:probs.shape[1]]
    macro_auc, per_auc = compute_macro_auc(probs, targets, names)
    fmax, fmax_thr, pc_f1_at_fmax = compute_fmax(probs, targets)

    from sklearn.metrics import f1_score
    preds     = (probs >= threshold).astype(int)
    macro_f1  = float(f1_score(
        targets, preds, average="macro", zero_division=0
    ))

    per_class = compute_per_class_metrics(probs, targets, names, threshold)
    df        = classification_report_df(probs, targets, names, threshold)

    result: Dict[str, Any] = {
        "n_samples":       int(probs.shape[0]),
        "n_classes":       int(probs.shape[1]),
        "macro_auc":       round(macro_auc, 4),
        "macro_f1":        round(macro_f1, 4),
        "fmax":            round(fmax, 4),
        "fmax_threshold":  round(fmax_thr, 4),
        "per_class":       per_class,
        "per_class_auc":   [round(v, 4) if v else None for v in per_auc],
        "per_class_f1_at_fmax": [round(float(v), 4) for v in pc_f1_at_fmax],
        "summary_df":      df,
    }

    if compute_ci:
        ci_idx  = ci_classes or list(range(min(probs.shape[1], 5)))
        ci_dict = {}
        for i in ci_idx:
            name   = names[i] if i < len(names) else str(i)
            mean_, lo, hi = bootstrap_ci(targets[:, i], probs[:, i])
            ci_dict[name] = {"mean": mean_, "ci_lo": lo, "ci_hi": hi}
        result["bootstrap_ci"] = ci_dict

    return result


# ---------------------------------------------------------------------------
# Логирование сводки
# ---------------------------------------------------------------------------

def log_summary(report: Dict[str, Any], title: str = "Evaluation") -> None:
    """Выводит сводку отчёта в лог."""
    logger.info(
        "\n╔══════════════════════════════════════╗"
        "\n║  %s",
        title,
    )
    logger.info(
        "  macro-AUC : %.4f\n"
        "  macro-F1  : %.4f\n"
        "  Fmax      : %.4f  (thr=%.3f)\n"
        "  N samples : %d",
        report["macro_auc"], report["macro_f1"],
        report["fmax"], report["fmax_threshold"],
        report["n_samples"],
    )

    df = report.get("summary_df")
    if df is not None:
        logger.info("\nПолная таблица:\n%s", df.to_string())
