"""
tasks/03_afib.py
Задача 03: Выявление фибрилляции предсердий (ФП).

Три метода сравниваются на PTB-XL fold 10:

  Метод 1 — RR-признаки → классический ML (LogReg / XGBoost)
    Признаки: CV, RMSSD, SDNN, SD1, SD2, pNN50, медиана RR, IQR RR
    Преимущество: интерпретируемость, работает без GPU

  Метод 2 — Backbone-голова (ResNet-1D)
    Класс: AF — SNOMED 164889003, индекс 0 в 27-классовом словаре
    Цель: F1=0.870, Specificity=1.000 (Ribeiro 2020)

  Метод 3 — Ансамбль: RR-признаки + backbone (логистическая регрессия
    над [P(AF|backbone), CV, RMSSD, SD1/SD2])

Выходы:
  results/task03_afib/
    method1_rr_ml_metrics.json
    method2_backbone_metrics.json
    method3_ensemble_metrics.json
    comparison_table.json
    fold10_probs_*.npy

Использование:
  python -m tasks.03_afib --config configs/default.yaml
  python -m tasks.03_afib --config configs/default.yaml --method 1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

AF_SNOMED_IDX    = 0          # AF — SNOMED 164889003 (индекс в 27-классовом словаре)
AF_SNOMED_CODE   = 164889003

FS_PTBXL         = 500.0
LEAD_II_IDX      = 1
MIN_RPEAKS       = 5

# Целевые значения Ribeiro 2020 для сравнения
RIBEIRO_AF_F1   = 0.870
RIBEIRO_AF_SPEC = 1.000

# Классификаторы ML
_ML_MODELS = ("logreg", "xgboost")


# ─────────────────────────────────────────────────────────────────────────────
# Извлечение RR-признаков
# ─────────────────────────────────────────────────────────────────────────────

def extract_rr_features(
    signal: np.ndarray,
    fs: float = FS_PTBXL,
    lead_idx: int = LEAD_II_IDX,
) -> Optional[np.ndarray]:
    """
    Извлекает вектор RR-признаков из записи для ML-классификатора ФП.

    Признаки (9 штук):
      0  median_rr     — медиана RR (мс)
      1  sdnn          — стандартное отклонение RR
      2  rmssd         — квадратный корень среднего квадрата разностей RR
      3  cv            — коэффициент вариации = sdnn / median_rr
      4  pnn50         — доля пар ΔRR > 50 мс
      5  sd1           — SD Пуанкаре (краткосрочная изменчивость)
      6  sd2           — SD Пуанкаре (долгосрочная изменчивость)
      7  iqr_rr        — межквартильный размах RR
      8  n_rpeaks      — число найденных R-пиков (нормировано на длину записи)

    Parameters
    ----------
    signal : np.ndarray [12, 5000]
    fs : float
    lead_idx : int

    Returns
    -------
    np.ndarray [9] float32 или None при ошибке
    """
    try:
        import neurokit2 as nk
    except ImportError:
        logger.error("neurokit2 не установлен: pip install neurokit2")
        return None

    lead = signal[lead_idx].astype(np.float64)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs),
                                     method="pantompkins1985")
        r_peaks = r_info.get("ECG_R_Peaks", np.array([]))
    except Exception as exc:
        logger.debug("ecg_peaks ошибка: %s", exc)
        return None

    if len(r_peaks) < MIN_RPEAKS:
        return None

    rr = np.diff(r_peaks).astype(float) * 1000.0 / fs   # мс
    rr = rr[(rr >= 200) & (rr <= 2000)]                  # физиологически допустимые

    if len(rr) < 3:
        return None

    rr_diff = np.diff(rr)
    sdnn    = float(np.std(rr))
    median  = float(np.median(rr))
    rmssd   = float(np.sqrt(np.mean(rr_diff ** 2))) if len(rr_diff) > 0 else 0.0
    cv      = sdnn / median if median > 0 else 0.0
    pnn50   = float(np.mean(np.abs(rr_diff) > 50)) if len(rr_diff) > 0 else 0.0

    # SD1/SD2 (Пуанкаре)
    sd1 = float(np.std(rr_diff) / np.sqrt(2)) if len(rr_diff) > 1 else 0.0
    sd2 = float(np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)))

    iqr  = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    n_pk = float(len(r_peaks)) / (signal.shape[-1] / fs)   # пиков/с → ЧСС-прокси

    return np.array([median, sdnn, rmssd, cv, pnn50, sd1, sd2, iqr, n_pk],
                    dtype=np.float32)


FEATURE_NAMES = [
    "median_rr", "sdnn", "rmssd", "cv", "pnn50",
    "sd1", "sd2", "iqr_rr", "n_rpeaks_per_sec",
]


# ─────────────────────────────────────────────────────────────────────────────
# Сбор данных fold 10 с признаками
# ─────────────────────────────────────────────────────────────────────────────

def collect_fold10_data(
    cfg,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Загружает PTB-XL fold 10 и извлекает:
      - сигналы для backbone (батчи)
      - RR-признаки для ML
      - метки AF (бинарный вектор)

    Returns
    -------
    X_rr      : [N, 9]  RR-признаки (NaN для записей без R-пиков)
    y_af      : [N]     бинарные метки AF (0/1)
    y_full    : [N, 27] полные метки (для всех классов)
    ecg_ids   : list[str]
    """
    from data.load_ptbxl import iter_ptbxl

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root

    X_rr_list: List[np.ndarray] = []
    y_af_list:  List[int]       = []
    y_full_list: List[np.ndarray] = []
    ecg_ids:    List[str]       = []
    n_no_peaks = 0

    logger.info("Сбор признаков fold 10…")
    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["test"],
        use_cache=True, show_progress=True, limit=limit,
    ):
        feats = extract_rr_features(rec.signal)
        if feats is None:
            feats = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
            n_no_peaks += 1

        X_rr_list.append(feats)
        y_af_list.append(int(rec.label_vec[AF_SNOMED_IDX]))
        y_full_list.append(rec.label_vec)
        ecg_ids.append(rec.ecg_id)

    logger.info(
        "fold 10: %d записей, AF=%d, без R-пиков=%d",
        len(y_af_list), sum(y_af_list), n_no_peaks,
    )

    return (
        np.stack(X_rr_list),
        np.array(y_af_list, dtype=np.int32),
        np.stack(y_full_list),
        ecg_ids,
    )


def collect_train_data(
    cfg,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает PTB-XL folds 1–8 для обучения ML-классификатора.

    Returns
    -------
    X_rr : [N, 9]
    y_af : [N]
    """
    from data.load_ptbxl import iter_ptbxl

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    X_list, y_list = [], []

    logger.info("Сбор признаков train (folds 1–8)…")
    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["train"],
        use_cache=True, show_progress=True, limit=limit,
    ):
        feats = extract_rr_features(rec.signal)
        if feats is None:
            feats = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
        X_list.append(feats)
        y_list.append(int(rec.label_vec[AF_SNOMED_IDX]))

    logger.info(
        "train: %d записей, AF=%d (%.1f%%)",
        len(y_list), sum(y_list),
        100.0 * sum(y_list) / max(len(y_list), 1),
    )
    return np.stack(X_list), np.array(y_list, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Метод 1: RR-признаки → ML
# ─────────────────────────────────────────────────────────────────────────────

def run_method1_rr_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path,
) -> Dict:
    """
    Обучает LogReg и XGBoost на RR-признаках, оценивает на fold 10.

    Использует SimpleImputer для NaN (записи без R-пиков).

    Returns
    -------
    dict с метриками обоих классификаторов
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    logger.info("Метод 1: RR-признаки → ML")

    # Общий препроцессинг: impute NaN → стандартизация
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)

    all_results = {}

    # ── LogReg ───────────────────────────────────────────────────────────────
    pos = y_train.sum()
    neg = len(y_train) - pos
    cw  = neg / max(pos, 1)

    logreg = LogisticRegression(
        C=1.0, class_weight={0: 1.0, 1: cw},
        max_iter=1000, random_state=42, solver="lbfgs",
    )
    logreg.fit(X_tr, y_train)
    lr_probs = logreg.predict_proba(X_te)[:, 1]
    lr_metrics = _compute_af_metrics(lr_probs, y_test, name="LogReg(RR)")
    all_results["logreg"] = lr_metrics

    logger.info(
        "LogReg: AUC=%.4f  F1=%.4f  Spec=%.4f",
        lr_metrics["auc"], lr_metrics["f1_fmax"], lr_metrics["specificity_fmax"],
    )

    # Важность признаков LogReg
    coefs = np.abs(logreg.coef_[0])
    feat_imp = {FEATURE_NAMES[i]: float(coefs[i]) for i in range(len(FEATURE_NAMES))}
    all_results["logreg"]["feature_importance"] = dict(
        sorted(feat_imp.items(), key=lambda x: -x[1])
    )
    np.save(str(results_dir / "method1_logreg_probs.npy"), lr_probs)

    # ── XGBoost (если установлен) ─────────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=float(cw),
            eval_metric="logloss", use_label_encoder=False,
            random_state=42, n_jobs=-1,
        )
        xgb.fit(X_tr, y_train, verbose=False)
        xgb_probs = xgb.predict_proba(X_te)[:, 1]
        xgb_metrics = _compute_af_metrics(xgb_probs, y_test, name="XGBoost(RR)")
        all_results["xgboost"] = xgb_metrics

        logger.info(
            "XGBoost: AUC=%.4f  F1=%.4f  Spec=%.4f",
            xgb_metrics["auc"], xgb_metrics["f1_fmax"], xgb_metrics["specificity_fmax"],
        )
        np.save(str(results_dir / "method1_xgb_probs.npy"), xgb_probs)

        # Важность признаков XGBoost
        fi = xgb.feature_importances_
        all_results["xgboost"]["feature_importance"] = dict(
            sorted(
                {FEATURE_NAMES[i]: float(fi[i]) for i in range(len(FEATURE_NAMES))}.items(),
                key=lambda x: -x[1],
            )
        )
    except ImportError:
        logger.info("XGBoost не установлен (pip install xgboost) — пропускаем")

    _save_json(all_results, results_dir / "method1_rr_ml_metrics.json")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Метод 2: Backbone-голова
# ─────────────────────────────────────────────────────────────────────────────

def run_method2_backbone(
    cfg,
    y_test: np.ndarray,
    y_full: np.ndarray,
    results_dir: Path,
    limit: Optional[int] = None,
) -> Tuple[Dict, np.ndarray]:
    """
    Инференс backbone на fold 10 → метрики AF-класса.

    Returns
    -------
    metrics : dict
    af_probs : np.ndarray [N]  вероятности AF для ансамбля
    """
    import torch
    from training._common import get_device, compute_metrics

    logger.info("Метод 2: Backbone-голова")
    device = get_device()

    model = _load_best_model(cfg, device)
    model.eval()

    from data.load_ptbxl import iter_ptbxl
    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root

    all_probs: List[np.ndarray] = []
    batch_signals: List[np.ndarray] = []
    batch_size = int(cfg.pretrain.batch_size) * 2
    n = 0

    def _flush():
        nonlocal batch_signals
        if not batch_signals:
            return
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch_signals)).to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(list(probs))
        batch_signals = []

    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["test"],
        use_cache=True, show_progress=True, limit=limit,
    ):
        batch_signals.append(rec.signal)
        n += 1
        if len(batch_signals) >= batch_size:
            _flush()
    _flush()

    if not all_probs:
        return {}, np.array([])

    probs_arr = np.stack(all_probs)    # [N, 27]
    af_probs  = probs_arr[:, AF_SNOMED_IDX]

    # Сохраняем все вероятности
    np.save(str(results_dir / "method2_backbone_probs_all.npy"), probs_arr)
    np.save(str(results_dir / "method2_backbone_probs_af.npy"), af_probs)

    # Метрики
    metrics = _compute_af_metrics(af_probs, y_test, name="Backbone")
    full_metrics = compute_metrics(probs_arr, y_full, n_classes=27)
    metrics["macro_auc_all"] = full_metrics["macro_auc"]

    # Сравнение с Ribeiro 2020
    metrics["ribeiro_f1_target"]   = RIBEIRO_AF_F1
    metrics["ribeiro_spec_target"] = RIBEIRO_AF_SPEC
    metrics["f1_delta_vs_ribeiro"] = round(
        metrics["f1_fmax"] - RIBEIRO_AF_F1, 4
    )

    logger.info(
        "Backbone: AUC=%.4f  F1=%.4f  Spec=%.4f  "
        "(Ribeiro2020: F1=%.3f Spec=%.3f  delta F1=%+.4f)",
        metrics["auc"], metrics["f1_fmax"], metrics["specificity_fmax"],
        RIBEIRO_AF_F1, RIBEIRO_AF_SPEC, metrics["f1_delta_vs_ribeiro"],
    )

    _save_json(metrics, results_dir / "method2_backbone_metrics.json")
    return metrics, af_probs


# ─────────────────────────────────────────────────────────────────────────────
# Метод 3: Ансамбль RR + backbone
# ─────────────────────────────────────────────────────────────────────────────

def run_method3_ensemble(
    X_rr_train: np.ndarray,
    y_train:    np.ndarray,
    X_rr_test:  np.ndarray,
    y_test:     np.ndarray,
    backbone_probs_train: np.ndarray,
    backbone_probs_test:  np.ndarray,
    results_dir: Path,
) -> Dict:
    """
    Ансамбль: LogReg над [P(AF|backbone), RR-признаки].

    backbone_probs_train/test : [N] вероятности AF от backbone на train/test.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    logger.info("Метод 3: Ансамбль Backbone + RR")

    # Конкатенируем backbone-вероятность с RR-признаками
    def _concat(bb_p: np.ndarray, rr: np.ndarray) -> np.ndarray:
        return np.column_stack([bb_p.reshape(-1, 1), rr])

    X_tr = _concat(backbone_probs_train, X_rr_train)
    X_te = _concat(backbone_probs_test,  X_rr_test)

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_tr = preprocessor.fit_transform(X_tr)
    X_te = preprocessor.transform(X_te)

    pos = y_train.sum()
    neg = len(y_train) - pos
    cw  = neg / max(pos, 1)

    model = LogisticRegression(
        C=1.0, class_weight={0: 1.0, 1: cw},
        max_iter=1000, random_state=42,
    )
    model.fit(X_tr, y_train)
    ens_probs = model.predict_proba(X_te)[:, 1]
    metrics   = _compute_af_metrics(ens_probs, y_test, name="Ensemble(BB+RR)")

    metrics["ribeiro_f1_target"]   = RIBEIRO_AF_F1
    metrics["f1_delta_vs_ribeiro"] = round(metrics["f1_fmax"] - RIBEIRO_AF_F1, 4)

    logger.info(
        "Ансамбль: AUC=%.4f  F1=%.4f  Spec=%.4f  delta_vs_Ribeiro=%+.4f",
        metrics["auc"], metrics["f1_fmax"],
        metrics["specificity_fmax"], metrics["f1_delta_vs_ribeiro"],
    )

    np.save(str(results_dir / "method3_ensemble_probs.npy"), ens_probs)
    _save_json(metrics, results_dir / "method3_ensemble_metrics.json")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Сбор backbone-вероятностей для train (нужно для ансамбля)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_backbone_train_probs(
    cfg,
    n_train: int,
    limit: Optional[int],
) -> np.ndarray:
    """
    Собирает backbone AF-вероятности для train fold (folds 1–8).
    Нужно для обучения мета-классификатора в методе 3.
    """
    import torch
    from data.load_ptbxl import iter_ptbxl
    from training._common import get_device

    device = get_device()
    model  = _load_best_model(cfg, device)
    model.eval()

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    batch_size  = int(cfg.pretrain.batch_size) * 2
    all_probs: List[float] = []
    batch_signals: List[np.ndarray] = []

    def _flush():
        nonlocal batch_signals
        if not batch_signals:
            return
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch_signals)).to(device)
            p = torch.sigmoid(model(x))[:, AF_SNOMED_IDX].cpu().numpy()
        all_probs.extend(p.tolist())
        batch_signals = []

    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["train"],
        use_cache=True, show_progress=False, limit=limit,
    ):
        batch_signals.append(rec.signal)
        if len(batch_signals) >= batch_size:
            _flush()
    _flush()

    return np.array(all_probs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная загрузка модели (дублируем из task02 для независимости)
# ─────────────────────────────────────────────────────────────────────────────

def _load_best_model(cfg, device):
    """Загружает лучший доступный чекпоинт (finetune > pretrain > Zenodo)."""
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
            logger.info("Загружаем %s", ckpt)
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state"], strict=True)
            return model.to(device)

    logger.warning("Чекпоинты не найдены → Zenodo backbone")
    cache = Path(cfg.paths.backbone_cache).expanduser()
    load_pretrained_weights(model.backbone, cache_dir=cache, strict=False)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Метрики для AF (бинарная задача)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_af_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    name: str = "",
) -> Dict:
    """
    Полный набор метрик для ФП-детекции.

    Дополнительно к AUC/F1 вычисляем:
      - specificity при пороге Fmax (как в Ribeiro 2020)
      - precision/recall кривые
      - AUPRC
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        average_precision_score, confusion_matrix,
    )

    n_pos = int(targets.sum())
    n     = int(len(targets))
    result: Dict = {
        "name": name, "n_positive": n_pos, "n_total": n,
        "prevalence": round(n_pos / max(n, 1), 4),
    }

    if n_pos == 0:
        logger.warning("%s: нет AF-записей в выборке", name)
        result.update({"auc": None, "auprc": None,
                       "f1_fmax": None, "specificity_fmax": None})
        return result

    # AUC и AUPRC
    try:
        result["auc"]   = float(roc_auc_score(targets, probs))
        result["auprc"] = float(average_precision_score(targets, probs))
    except Exception as e:
        logger.debug("AUC ошибка: %s", e)
        result["auc"] = result["auprc"] = None

    # Fmax: перебор порогов
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.02, 0.98, 97):
        p = (probs >= thr).astype(int)
        f1 = float(f1_score(targets, p, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    result["fmax"]           = best_f1
    result["fmax_threshold"] = best_thr

    # Метрики при пороге Fmax
    preds_fmax = (probs >= best_thr).astype(int)
    result["f1_fmax"]        = float(f1_score(targets, preds_fmax, zero_division=0))
    result["precision_fmax"] = float(precision_score(targets, preds_fmax, zero_division=0))
    result["recall_fmax"]    = float(recall_score(targets, preds_fmax, zero_division=0))

    # Specificity при пороге Fmax
    cm = confusion_matrix(targets, preds_fmax, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        result["specificity_fmax"] = float(tn / max(tn + fp, 1))
        result["confusion_matrix"] = cm.tolist()
    else:
        result["specificity_fmax"] = None

    # Метрики при стандартном пороге 0.5
    preds_05 = (probs >= 0.5).astype(int)
    result["f1_at_05"]        = float(f1_score(targets, preds_05, zero_division=0))
    result["precision_at_05"] = float(precision_score(targets, preds_05, zero_division=0))
    result["recall_at_05"]    = float(recall_score(targets, preds_05, zero_division=0))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Таблица сравнения методов
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison_table(
    method_results: Dict[str, Dict],
) -> List[Dict]:
    """
    Строит список строк для таблицы сравнения методов.
    Каждая строка: {method, auc, f1, specificity, source}
    """
    rows = []

    # Ribeiro 2020 reference
    rows.append({
        "method": "Ribeiro 2020 (ResNet + PTB-XL)",
        "auc":    None,
        "f1":     RIBEIRO_AF_F1,
        "specificity": RIBEIRO_AF_SPEC,
        "source": "published",
    })

    for key, r in method_results.items():
        if not r or "error" in r:
            continue

        # method1 может иметь logreg/xgboost
        if key == "method1":
            for clf, metrics in r.items():
                if not isinstance(metrics, dict) or "auc" not in metrics:
                    continue
                rows.append({
                    "method": metrics.get("name", clf),
                    "auc":    metrics.get("auc"),
                    "f1":     metrics.get("f1_fmax"),
                    "specificity": metrics.get("specificity_fmax"),
                    "source": "ours_rr_ml",
                })
        else:
            rows.append({
                "method":      r.get("name", key),
                "auc":         r.get("auc"),
                "f1":          r.get("f1_fmax"),
                "specificity": r.get("specificity_fmax"),
                "source":      "ours",
            })

    return rows


def _print_comparison_table(rows: List[Dict]) -> None:
    """Выводит таблицу сравнения в лог."""
    header = f"\n{'Метод':<40} {'AUC':>7} {'F1':>7} {'Spec':>7} {'Источник'}"
    sep    = "─" * 75
    logger.info("\n════ Задача 03: ФП — сравнение методов ════")
    print(header)
    print(sep)
    for row in rows:
        auc  = f"{row['auc']:.4f}"  if row.get("auc")  else "  —   "
        f1   = f"{row['f1']:.4f}"   if row.get("f1")   else "  —   "
        spec = f"{row['specificity']:.4f}" if row.get("specificity") else "  —   "
        print(f"{row['method']:<40} {auc:>7} {f1:>7} {spec:>7}  {row.get('source','')}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_afib(
    cfg,
    methods: Optional[List[int]] = None,
    limit: Optional[int] = None,
) -> Dict:
    """
    Запускает задачу 03.

    Parameters
    ----------
    cfg
    methods : list[int] | None
        Какие методы запустить: [1, 2, 3]. None → все три.
    limit : int | None

    Returns
    -------
    dict с результатами всех методов и таблицей сравнения
    """
    if methods is None:
        methods = [1, 2, 3]

    results_dir = Path(cfg.paths.results_dir) / "task03_afib"
    results_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {}

    # ── Общий сбор признаков fold 10 ─────────────────────────────────────────
    logger.info("Сбор данных fold 10…")
    X_rr_test, y_af_test, y_full_test, _ = collect_fold10_data(cfg, limit=limit)

    # ── Сбор train (нужен для методов 1 и 3) ─────────────────────────────────
    X_rr_train, y_af_train = None, None
    if 1 in methods or 3 in methods:
        X_rr_train, y_af_train = collect_train_data(cfg, limit=limit)

    # ── Метод 1 ───────────────────────────────────────────────────────────────
    if 1 in methods and X_rr_train is not None:
        try:
            results["method1"] = run_method1_rr_ml(
                X_rr_train, y_af_train,
                X_rr_test,  y_af_test,
                results_dir,
            )
        except Exception as exc:
            logger.error("Метод 1 ошибка: %s", exc, exc_info=True)
            results["method1"] = {"error": str(exc)}

    # ── Метод 2 ───────────────────────────────────────────────────────────────
    bb_probs_test = None
    if 2 in methods or 3 in methods:
        try:
            m2, bb_probs_test = run_method2_backbone(
                cfg, y_af_test, y_full_test, results_dir, limit=limit
            )
            results["method2"] = m2
        except Exception as exc:
            logger.error("Метод 2 ошибка: %s", exc, exc_info=True)
            results["method2"] = {"error": str(exc)}

    # ── Метод 3 (ансамбль) ────────────────────────────────────────────────────
    if 3 in methods and bb_probs_test is not None and X_rr_train is not None:
        try:
            logger.info("Сбор backbone-вероятностей для train (для ансамбля)…")
            bb_probs_train = _collect_backbone_train_probs(cfg, len(y_af_train), limit)
            results["method3"] = run_method3_ensemble(
                X_rr_train, y_af_train,
                X_rr_test,  y_af_test,
                bb_probs_train, bb_probs_test,
                results_dir,
            )
        except Exception as exc:
            logger.error("Метод 3 ошибка: %s", exc, exc_info=True)
            results["method3"] = {"error": str(exc)}

    # ── Таблица сравнения ─────────────────────────────────────────────────────
    comparison = _build_comparison_table(results)
    _print_comparison_table(comparison)
    results["comparison_table"] = comparison

    _save_json(results, results_dir / "comparison_table.json")
    logger.info("Задача 03 завершена → %s", results_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data: Dict, path: Path) -> None:
    def _ser(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _ser(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_ser(x) for x in v]
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_ser(data), f, indent=2, ensure_ascii=False)
    logger.info("Сохранено: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Задача 03: ФП — RR-признаки, backbone, ансамбль"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--method", type=int, nargs="+", choices=[1, 2, 3],
        default=None,
        help="Методы для запуска (1=RR-ML, 2=backbone, 3=ensemble). По умолчанию все.",
    )
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training._common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (50 if args.debug else None)
    run_afib(cfg, methods=args.method, limit=limit)


if __name__ == "__main__":
    main()