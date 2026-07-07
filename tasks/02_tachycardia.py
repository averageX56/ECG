"""
tasks/02_tachycardia.py
Задача 02: Выявление тахикардий.

Два потока:

  Голова A (backbone, поток [A]):
    Желудочковая тахикардия (ЖТ) — SNOMED 164896001 (VTach, индекс 26)
    Данные: PTB-XL fold 10 (тест)
    Цель: AUC ≥ 0.95

  Голова B (beat-level, поток [B]):
    ЖЭС (PVC) vs НаджЭС (PAC/SVE) из MIT-BIH
    Baseline: правило QRS > 120 мс (NeuroKit2) → ЖЭС
    Сравнение с BeatCNN-классификатором

  Оба результата пишутся в results/task02_tachycardia/

Использование:
  python -m tasks.02_tachycardia --config configs/default.yaml
  python -m tasks.02_tachycardia --config configs/default.yaml --stream A
  python -m tasks.02_tachycardia --config configs/default.yaml --stream B
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

# SNOMED-коды (индексы в 27-классовом словаре из snomed_map.py _SCORED)
VT_SNOMED_IDX   = 26   # VTach — SNOMED 164896001
AF_SNOMED_IDX   = 0    # AF    — SNOMED 164889003 (для контрольного теста)

# AAMI-классы из MIT-BIH
AAMI_V_IDX = 2   # Ventricular ectopic (PVC/ЖЭС)
AAMI_S_IDX = 1   # Supraventricular ectopic (PAC/НаджЭС)
AAMI_N_IDX = 0   # Normal

# Порог QRS для baseline-правила (мс)
QRS_THRESHOLD_MS = 120.0

# Частота дискретизации MIT-BIH
FS_MITBIH = 360.0


# ─────────────────────────────────────────────────────────────────────────────
# ── ПОТОК A: Backbone-голова для ЖТ ──────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run_stream_a_vt(
    cfg,
    results_dir: Path,
    limit: Optional[int] = None,
) -> Dict:
    """
    Оценивает backbone-голову для ЖТ на PTB-XL fold 10.

    Шаги:
      1. Загрузка лучшего чекпоинта (pretrain или finetune)
      2. Инференс на fold 10 → sigmoid-вероятности [N, 27]
      3. Метрики: AUC, F1, Fmax по классу VTach (индекс 26)

    Returns
    -------
    dict с ключами: vt_auc, vt_f1, vt_fmax, macro_auc, n_samples, n_positive
    """
    import torch
    from backbone.resnet1d import ResNet1dWithHead
    from data.load_ptbxl import iter_ptbxl
    from training._common import get_device, compute_metrics

    device = get_device()

    # ── Загрузка модели ───────────────────────────────────────────────────────
    model = _load_best_model(cfg, device)
    model.eval()

    # ── Данные fold 10 ────────────────────────────────────────────────────────
    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    logger.info("Поток A: загружаем PTB-XL fold 10 (test)…")

    all_probs:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    n_processed = 0

    batch_signals: List[np.ndarray] = []
    batch_labels:  List[np.ndarray] = []
    batch_size = int(cfg.pretrain.batch_size) * 2

    def _flush_batch() -> None:
        nonlocal batch_signals, batch_labels
        if not batch_signals:
            return
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch_signals)).to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(list(probs))
        all_targets.extend(batch_labels)
        batch_signals = []
        batch_labels  = []

    for rec in iter_ptbxl(
        root=ptbxl_root,
        splits=["test"],
        use_cache=True,
        show_progress=True,
        limit=limit,
    ):
        batch_signals.append(rec.signal)
        batch_labels.append(rec.label_vec)
        n_processed += 1
        if len(batch_signals) >= batch_size:
            _flush_batch()

    _flush_batch()

    if not all_probs:
        logger.error("Нет данных fold 10")
        return {}

    probs_arr   = np.stack(all_probs)    # [N, 27]
    targets_arr = np.stack(all_targets)  # [N, 27]
    n_classes   = probs_arr.shape[1]

    # ── Полные метрики ────────────────────────────────────────────────────────
    full_metrics = compute_metrics(probs_arr, targets_arr, n_classes=n_classes)

    # ── Метрики только VTach ──────────────────────────────────────────────────
    vt_probs   = probs_arr[:, VT_SNOMED_IDX]
    vt_targets = targets_arr[:, VT_SNOMED_IDX]
    n_positive = int(vt_targets.sum())

    vt_metrics = _compute_binary_metrics(vt_probs, vt_targets, name="VTach")

    logger.info(
        "Поток A | VTach: AUC=%.4f  F1=%.4f  Fmax=%.4f  "
        "N_pos=%d/%d  macro_AUC=%.4f",
        vt_metrics["auc"], vt_metrics["f1_at_05"], vt_metrics["fmax"],
        n_positive, n_processed, full_metrics["macro_auc"],
    )

    result = {
        "stream":      "A",
        "task":        "VT_detection",
        "n_samples":   n_processed,
        "n_positive":  n_positive,
        "vt_auc":      vt_metrics["auc"],
        "vt_f1":       vt_metrics["f1_at_05"],
        "vt_fmax":     vt_metrics["fmax"],
        "vt_fmax_thr": vt_metrics["fmax_threshold"],
        "vt_precision_at_05": vt_metrics["precision_at_05"],
        "vt_recall_at_05":    vt_metrics["recall_at_05"],
        "macro_auc":   full_metrics["macro_auc"],
        "macro_f1":    full_metrics["macro_f1"],
        "target_met":  bool(vt_metrics["auc"] >= 0.95),
    }

    # Сохраняем предсказания
    np.save(str(results_dir / "stream_a_probs.npy"),   probs_arr)
    np.save(str(results_dir / "stream_a_targets.npy"), targets_arr)

    _save_json(result, results_dir / "stream_a_vt_metrics.json")
    logger.info("Поток A: результаты сохранены → %s", results_dir)
    return result


def _load_best_model(cfg, device) -> "ResNet1dWithHead":
    """
    Загружает лучший доступный чекпоинт.
    Приоритет: finetune_best > pretrain_best > Zenodo-веса.
    """
    from backbone.resnet1d import ResNet1dWithHead
    from backbone.load_weights import load_pretrained_weights

    model = ResNet1dWithHead(
        n_classes       = int(cfg.pretrain.n_classes),
        backbone_kwargs = {
            "n_leads": int(cfg.backbone.n_leads),
            "dropout": float(cfg.backbone.dropout),
        },
        dropout_head = float(cfg.pretrain.dropout_head),
    )

    # Попытка загрузить finetune-чекпоинт
    ft_ckpt = Path(cfg.paths.checkpoint_dir) / "finetune_best.pt"
    pt_ckpt = Path(cfg.paths.checkpoint_dir) / "pretrain_best.pt"

    if ft_ckpt.exists():
        logger.info("Загружаем finetune чекпоинт: %s", ft_ckpt)
        state = __import__("torch").load(str(ft_ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
    elif pt_ckpt.exists():
        logger.info("Загружаем pretrain чекпоинт: %s", pt_ckpt)
        state = __import__("torch").load(str(pt_ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
    else:
        logger.warning("Чекпоинты не найдены, используем Zenodo-веса backbone")
        cache_dir = Path(cfg.paths.backbone_cache).expanduser()
        load_pretrained_weights(model.backbone, cache_dir=cache_dir, strict=False)

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# ── ПОТОК B: Beat-level ЖЭС vs НаджЭС ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def run_stream_b_ectopics(
    cfg,
    results_dir: Path,
    limit: Optional[int] = None,
) -> Dict:
    """
    Сравнивает два метода классификации экстрасистол на MIT-BIH DS2:
      1. Baseline: QRS > 120 мс → ЖЭС (через NeuroKit2)
      2. BeatCNN: обученный классификатор битов

    Returns
    -------
    dict с метриками обоих методов
    """
    from data.load_mitbih import DS2_RECORDS, load_mitbih_arrays, iter_mitbih_beats

    mitbih_root = str(cfg.paths.mitbih_root)
    logger.info("Поток B: загружаем MIT-BIH DS2…")

    # ── Загрузка битов с record_id для правила QRS ────────────────────────────
    beats_list: List[np.ndarray] = []
    classes_list: List[int]      = []
    rpeaks_list: List[int]       = []
    recids_list: List[str]       = []
    count = 0

    for rec_b in iter_mitbih_beats(
        root        = mitbih_root,
        record_ids  = list(DS2_RECORDS),
        znorm       = True,
        show_progress = True,
    ):
        beats_list.append(rec_b.beat)
        classes_list.append(rec_b.beat_class)
        rpeaks_list.append(rec_b.sample_pos)
        recids_list.append(rec_b.record_id)
        count += 1
        if limit and count >= limit:
            break

    if not beats_list:
        logger.error("Нет данных MIT-BIH DS2")
        return {}

    X = np.stack(beats_list)           # [N, 2, 300]
    y = np.array(classes_list, dtype=np.int64)

    logger.info(
        "DS2: %d битов, ЖЭС=%d, НаджЭС=%d, N=%d",
        len(y), int((y == AAMI_V_IDX).sum()),
        int((y == AAMI_S_IDX).sum()),
        int((y == AAMI_N_IDX).sum()),
    )

    # ── Метод 1: Baseline — правило QRS>120мс ─────────────────────────────────
    rule_preds = _qrs_rule_predict(X, fs=FS_MITBIH)
    rule_metrics = _evaluate_v_s_classification(y, rule_preds, name="QRS-rule")

    # ── Метод 2: BeatCNN ──────────────────────────────────────────────────────
    cnn_metrics: Optional[Dict] = None
    cnn_ckpt = Path(cfg.paths.checkpoint_dir) / "beat_clf" / "fold_0" / "beat_clf_best.pt"

    if cnn_ckpt.exists():
        cnn_preds = _cnn_predict(X, cfg, cnn_ckpt)
        cnn_metrics = _evaluate_v_s_classification(y, cnn_preds, name="BeatCNN")
    else:
        logger.warning(
            "BeatCNN чекпоинт не найден: %s\n"
            "Сначала запусти: python -m training.train_beat_clf --config …",
            cnn_ckpt,
        )

    # ── Итоговая сводка ───────────────────────────────────────────────────────
    result = {
        "stream":          "B",
        "task":            "PVC_vs_SVE_classification",
        "n_beats":         int(len(y)),
        "n_pvc":           int((y == AAMI_V_IDX).sum()),
        "n_sve":           int((y == AAMI_S_IDX).sum()),
        "n_normal":        int((y == AAMI_N_IDX).sum()),
        "qrs_rule":        rule_metrics,
        "beat_cnn":        cnn_metrics,
    }

    if cnn_metrics:
        _log_comparison(rule_metrics, cnn_metrics)

    np.save(str(results_dir / "stream_b_beats.npy"), X)
    np.save(str(results_dir / "stream_b_labels.npy"), y)
    np.save(str(results_dir / "stream_b_rule_preds.npy"), rule_preds)

    _save_json(result, results_dir / "stream_b_ectopic_metrics.json")
    logger.info("Поток B: результаты сохранены → %s", results_dir)
    return result


def _qrs_rule_predict(
    beats: np.ndarray,
    fs: float = FS_MITBIH,
) -> np.ndarray:
    """
    Baseline-правило: если длительность QRS-комплекса в бите > 120 мс,
    предсказываем ЖЭС (класс 2), иначе НаджЭС/N (класс по сигналу).

    Бит имеет форму [2, 300], отведение 0 используем для измерения.

    Parameters
    ----------
    beats : np.ndarray  [N, 2, 300]  z-нормированные биты
    fs : float  частота дискретизации MIT-BIH (360 Гц)

    Returns
    -------
    np.ndarray [N]  — предсказанные классы AAMI
    """
    try:
        import neurokit2 as nk
    except ImportError:
        logger.warning("NeuroKit2 не установлен, QRS-правило невозможно")
        return np.zeros(len(beats), dtype=np.int64)

    preds = np.zeros(len(beats), dtype=np.int64)   # default: N

    for i, beat in enumerate(beats):
        lead = beat[0].astype(np.float64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Нормализуем обратно для NeuroKit (z-нормировка убирает масштаб)
                # Ищем QRS-длительность непосредственно в бите ±150 сэмплов
                _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs))
                r_pks = r_info.get("ECG_R_Peaks", np.array([]))

                if len(r_pks) == 0:
                    # Берём пик в центре бита (R-пик ≈ позиция 150)
                    r_pks = np.array([150])

                _, waves = nk.ecg_delineate(
                    lead, r_pks,
                    sampling_rate=int(fs),
                    method="peaks",
                )
                q_pks = np.array(waves.get("ECG_Q_Peaks", []), dtype=float)
                s_pks = np.array(waves.get("ECG_S_Peaks", []), dtype=float)

                qrs_dur_ms: Optional[float] = None
                if len(q_pks) > 0 and len(s_pks) > 0:
                    q = q_pks[~np.isnan(q_pks)]
                    s = s_pks[~np.isnan(s_pks)]
                    if len(q) > 0 and len(s) > 0:
                        qrs_dur_ms = float((s[0] - q[0]) * 1000.0 / fs)

                # Если QRS не удалось определить — оцениваем по ширине
                # зоны вокруг пика через энергетический подход
                if qrs_dur_ms is None:
                    qrs_dur_ms = _estimate_qrs_width(lead, fs)

                if qrs_dur_ms is not None and qrs_dur_ms > QRS_THRESHOLD_MS:
                    preds[i] = AAMI_V_IDX   # ЖЭС
                # иначе остаётся 0 (N)

        except Exception:
            pass   # при ошибке оставляем N

    return preds


def _estimate_qrs_width(lead: np.ndarray, fs: float) -> Optional[float]:
    """
    Простая оценка ширины QRS по ширине центрального пика бита
    на уровне полуамплитуды.
    """
    center = len(lead) // 2  # ≈ 150 для бита 300 сэмплов
    window = int(0.15 * fs)  # ±150 мс вокруг R
    lo = max(0, center - window)
    hi = min(len(lead), center + window)
    segment = lead[lo:hi]

    if len(segment) < 10:
        return None

    peak_val = np.max(np.abs(segment))
    if peak_val < 1e-6:
        return None

    threshold = 0.5 * peak_val
    above = np.abs(segment) >= threshold
    if not np.any(above):
        return None

    idx = np.where(above)[0]
    width_samples = int(idx[-1] - idx[0]) + 1
    return float(width_samples * 1000.0 / fs)


def _cnn_predict(
    beats: np.ndarray,
    cfg,
    ckpt_path: Path,
) -> np.ndarray:
    """
    Предсказание через обученный BeatCNN.

    Parameters
    ----------
    beats : np.ndarray  [N, 2, 300]
    ckpt_path : Path    путь к чекпоинту beat_clf_best.pt

    Returns
    -------
    np.ndarray [N]  argmax-предсказания AAMI
    """
    import torch
    from training.beat_level_clf import BeatCNN

    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")

    model = BeatCNN(
        n_leads       = 2,
        conv_channels = list(cfg.beat_clf.conv_channels),
        kernel_size   = int(cfg.beat_clf.kernel_size),
        n_classes     = int(cfg.beat_clf.n_classes),
        dropout       = 0.0,   # inference без dropout
    ).to(device)

    state = __import__("torch").load(
        str(ckpt_path), map_location=str(device), weights_only=True
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    batch_size = 512
    all_preds: List[np.ndarray] = []

    with __import__("torch").no_grad():
        for start in range(0, len(beats), batch_size):
            batch = __import__("torch").from_numpy(
                beats[start : start + batch_size].astype(np.float32)
            ).to(device)
            logits = model(batch)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    return np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────

def _compute_binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    name: str = "",
    threshold: float = 0.5,
) -> Dict:
    """
    Бинарная задача: AUC, F1@0.5, Fmax, precision, recall.
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
    )

    result: Dict = {
        "name":         name,
        "n_positive":   int(targets.sum()),
        "n_total":      int(len(targets)),
        "auc":          None,
        "f1_at_05":     None,
        "fmax":         None,
        "fmax_threshold": None,
        "precision_at_05": None,
        "recall_at_05":    None,
    }

    if result["n_positive"] == 0:
        logger.warning("%s: нет позитивных примеров, метрики не вычислены", name)
        return result

    preds = (probs >= threshold).astype(int)

    try:
        result["auc"] = float(roc_auc_score(targets, probs))
    except Exception as e:
        logger.debug("AUC ошибка: %s", e)

    result["f1_at_05"]     = float(f1_score(targets, preds, zero_division=0))
    result["precision_at_05"] = float(precision_score(targets, preds, zero_division=0))
    result["recall_at_05"]    = float(recall_score(targets, preds, zero_division=0))

    # Fmax: перебор порогов
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        p = (probs >= thr).astype(int)
        f1 = float(f1_score(targets, p, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    result["fmax"]           = best_f1
    result["fmax_threshold"] = best_thr

    return result


def _evaluate_v_s_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "",
) -> Dict:
    """
    Метрики для задачи ЖЭС vs НаджЭС на MIT-BIH (multi-class → per-class F1).

    Считаем: per-class F1 для V (ЖЭС), S (НаджЭС), N.
    """
    from sklearn.metrics import f1_score, classification_report, confusion_matrix

    # Ограничиваемся битами классов V, S, N (остальные: F=3, Q=4 — игнорируем)
    mask = np.isin(y_true, [AAMI_N_IDX, AAMI_S_IDX, AAMI_V_IDX])
    yt = y_true[mask]
    yp = y_pred[mask]

    per_f1 = f1_score(yt, yp, labels=[AAMI_N_IDX, AAMI_S_IDX, AAMI_V_IDX],
                      average=None, zero_division=0)
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

    cm = confusion_matrix(yt, yp, labels=[AAMI_N_IDX, AAMI_S_IDX, AAMI_V_IDX])
    report = classification_report(
        yt, yp,
        labels=[AAMI_N_IDX, AAMI_S_IDX, AAMI_V_IDX],
        target_names=["N", "S(SVE)", "V(PVC)"],
        zero_division=0,
    )

    return {
        "name":         name,
        "n_evaluated":  int(mask.sum()),
        "macro_f1":     macro_f1,
        "f1_N":         float(per_f1[0]),
        "f1_S_SVE":     float(per_f1[1]),
        "f1_V_PVC":     float(per_f1[2]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def _log_comparison(rule: Dict, cnn: Dict) -> None:
    """Логирует сравнительную таблицу QRS-правило vs CNN."""
    logger.info(
        "\n╔══════════════════════════════════════════════════╗"
        "\n║   Поток B: ЖЭС vs НаджЭС — сравнение методов   ║"
        "\n╠══════════════════╦═══════════════╦══════════════╣"
        "\n║ Метрика          ║ QRS-правило   ║ BeatCNN      ║"
        "\n╠══════════════════╬═══════════════╬══════════════╣"
        "\n║ macro-F1         ║ %-13.4f ║ %-12.4f ║"
        "\n║ F1(PVC/ЖЭС)     ║ %-13.4f ║ %-12.4f ║"
        "\n║ F1(SVE/НаджЭС)  ║ %-13.4f ║ %-12.4f ║"
        "\n╚══════════════════╩═══════════════╩══════════════╝",
        rule.get("macro_f1", 0), cnn.get("macro_f1", 0),
        rule.get("f1_V_PVC", 0), cnn.get("f1_V_PVC", 0),
        rule.get("f1_S_SVE", 0), cnn.get("f1_S_SVE", 0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data: Dict, path: Path) -> None:
    def _ser(v):
        if isinstance(v, float) and np.isnan(v):
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
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_task02(
    cfg,
    stream: str = "both",
    limit: Optional[int] = None,
) -> Dict:
    """
    Запускает задачу 02.

    Parameters
    ----------
    cfg : _DotDict
    stream : str
        'A' — только backbone ЖТ
        'B' — только beat-level ЖЭС/НаджЭС
        'both' — оба потока
    limit : int | None

    Returns
    -------
    dict с результатами обоих потоков
    """
    results_dir = Path(cfg.paths.results_dir) / "task02_tachycardia"
    results_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {}

    if stream in ("A", "both"):
        logger.info("═══ Поток A: ЖТ-детекция через backbone ═══")
        try:
            results["stream_a"] = run_stream_a_vt(cfg, results_dir, limit=limit)
        except Exception as exc:
            logger.error("Поток A завершился с ошибкой: %s", exc, exc_info=True)
            results["stream_a"] = {"error": str(exc)}

    if stream in ("B", "both"):
        logger.info("═══ Поток B: ЖЭС/НаджЭС из MIT-BIH ═══")
        try:
            results["stream_b"] = run_stream_b_ectopics(cfg, results_dir, limit=limit)
        except Exception as exc:
            logger.error("Поток B завершился с ошибкой: %s", exc, exc_info=True)
            results["stream_b"] = {"error": str(exc)}

    # Итоговый отчёт
    _save_json(results, results_dir / "metrics_summary.json")
    _print_summary(results)
    return results


def _print_summary(results: Dict) -> None:
    """Выводит итоговую таблицу в лог."""
    print("\n════ Задача 02: Тахикардии ════")

    sa = results.get("stream_a", {})
    if sa and "error" not in sa:
        target_mark = "✓" if sa.get("target_met") else "✗"
        print(f"\nПоток A — ЖТ (backbone, PTB-XL fold 10):")
        print(f"  AUC      = {sa.get('vt_auc', 0):.4f}  {target_mark} цель ≥0.95")
        print(f"  F1@0.5   = {sa.get('vt_f1', 0):.4f}")
        print(f"  Fmax     = {sa.get('vt_fmax', 0):.4f}  "
              f"(thr={sa.get('vt_fmax_thr', 0):.2f})")
        print(f"  N_pos    = {sa.get('n_positive', 0)} из {sa.get('n_samples', 0)}")
        print(f"  macro-AUC всех 27 классов = {sa.get('macro_auc', 0):.4f}")

    sb = results.get("stream_b", {})
    if sb and "error" not in sb:
        rule = sb.get("qrs_rule", {})
        cnn  = sb.get("beat_cnn") or {}
        print(f"\nПоток B — ЖЭС vs НаджЭС (MIT-BIH DS2):")
        print(f"  Всего битов: {sb.get('n_beats', 0)}  "
              f"ЖЭС={sb.get('n_pvc', 0)}  НаджЭС={sb.get('n_sve', 0)}")
        print(f"  QRS-правило: macro-F1={rule.get('macro_f1', 0):.4f}  "
              f"F1(PVC)={rule.get('f1_V_PVC', 0):.4f}  "
              f"F1(SVE)={rule.get('f1_S_SVE', 0):.4f}")
        if cnn:
            print(f"  BeatCNN:     macro-F1={cnn.get('macro_f1', 0):.4f}  "
                  f"F1(PVC)={cnn.get('f1_V_PVC', 0):.4f}  "
                  f"F1(SVE)={cnn.get('f1_S_SVE', 0):.4f}")
        else:
            print("  BeatCNN: чекпоинт не найден (обучите через train_beat_clf.py)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Задача 02: Тахикардии — ЖТ (backbone) + ЖЭС/НаджЭС (beat clf)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--stream", choices=["A", "B", "both"], default="both",
        help="Какой поток запустить (default: both)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Ограничить число записей/битов (для отладки)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training._common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (30 if args.debug else None)
    run_task02(cfg, stream=args.stream, limit=limit)


if __name__ == "__main__":
    main()
