"""
evaluation/reproduce_ribeiro.py
Воспроизведение метрик Ribeiro et al. 2020 на PTB-XL fold 10.

Ribeiro et al. «Automatic diagnosis of the 12-lead ECG using a deep neural network»
Nature Communications 2020, Supplementary Table 2.

Сценарии:
  1. Baseline: загружаем предобученные Zenodo-веса → прогоняем fold 10
  2. После pretrain (шаг 1): загружаем checkpoints/pretrain_best.pt
  3. После finetune (шаг 2): загружаем checkpoints/finetune_best.pt

Выходы:
  results/reproduce_ribeiro/
    baseline_metrics.json
    pretrain_metrics.json
    finetune_metrics.json
    comparison_table.csv
    comparison_table.json

Использование:
  python -m evaluation.reproduce_ribeiro --config configs/default.yaml
  python -m evaluation.reproduce_ribeiro --config configs/default.yaml --stage baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# Классы, для которых Ribeiro публикует детальные метрики
RIBEIRO_CLASSES = ["AF", "IAVB", "LBBB", "CRBBB", "PAC", "PVC", "STach", "SB", "NSR", "VTach"]

# Ribeiro 2020 Supplementary Table 2
RIBEIRO_REFERENCE: Dict[str, Dict[str, float]] = {
    "AF":    {"f1": 0.870, "sensitivity": 0.782, "specificity": 1.000},
    "IAVB":  {"f1": 0.897, "sensitivity": 0.827, "specificity": 0.979},
    "LBBB":  {"f1": 1.000, "sensitivity": 1.000, "specificity": 1.000},
    "CRBBB": {"f1": 0.944, "sensitivity": 0.908, "specificity": 0.983},
    "PAC":   {"f1": 0.812, "sensitivity": 0.702, "specificity": 0.961},
    "PVC":   {"f1": 0.801, "sensitivity": 0.719, "specificity": 0.965},
    "STach": {"f1": 0.854, "sensitivity": 0.768, "specificity": 0.952},
    "SB":    {"f1": 0.876, "sensitivity": 0.795, "specificity": 0.979},
    "NSR":   {"f1": 0.960, "sensitivity": 0.942, "specificity": 0.979},
    "VTach": {"f1": 0.750, "sensitivity": 0.600, "specificity": 0.997},
}


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

def _load_model(cfg, stage: str, device):
    """
    Загружает модель для указанного этапа.

    stage: 'baseline' | 'pretrain' | 'finetune'
    """
    from backbone.resnet1d import ResNet1dWithHead
    from backbone.load_weights import load_pretrained_weights
    import torch

    model = ResNet1dWithHead(
        n_classes=int(cfg.pretrain.n_classes),
        backbone_kwargs={
            # "n_leads": int(cfg.backbone.n_leads),
            "dropout": float(cfg.backbone.dropout),
        },
        dropout_head=float(cfg.pretrain.dropout_head),
    )

    ckpt_dir = Path(cfg.paths.checkpoint_dir)

    if stage == "finetune":
        ckpt = ckpt_dir / "finetune_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Finetune чекпоинт не найден: {ckpt}\n"
                "Запустите: python -m training.finetune_cpcs --config ..."
            )
        logger.info("Загружаем finetune чекпоинт: %s", ckpt)
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)

    elif stage == "pretrain":
        ckpt = ckpt_dir / "pretrain_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Pretrain чекпоинт не найден: {ckpt}\n"
                "Запустите: python -m training.pretrain --config ..."
            )
        logger.info("Загружаем pretrain чекпоинт: %s", ckpt)
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)

    else:  # baseline: только Zenodo-веса backbone, голова случайная
        logger.info("Baseline: загружаем Zenodo-веса backbone")
        cache_dir = Path(cfg.paths.backbone_cache).expanduser()
        load_pretrained_weights(model.backbone, cache_dir=cache_dir, strict=False)

    return model.to(device)


# ---------------------------------------------------------------------------
# Инференс на fold 10
# ---------------------------------------------------------------------------

def run_inference_fold10(
    cfg,
    model,
    device,
    limit: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Прогоняет модель на PTB-XL fold 10.

    Returns
    -------
    probs   : [N, 27]
    targets : [N, 27]
    """
    import torch
    from data.load_ptbxl import iter_ptbxl

    ptbxl_root  = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    batch_size  = int(cfg.pretrain.batch_size) * 2

    all_probs:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    batch_sigs:  List[np.ndarray] = []
    batch_lbls:  List[np.ndarray] = []

    def _flush():
        nonlocal batch_sigs, batch_lbls
        if not batch_sigs:
            return
        with torch.no_grad():
            x      = torch.from_numpy(np.stack(batch_sigs)).to(device)
            logits = model(x)
            probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(list(probs))
        all_targets.extend(batch_lbls)
        batch_sigs = []
        batch_lbls = []

    model.eval()
    n = 0
    for rec in iter_ptbxl(
        root=ptbxl_root,
        splits=["test"],
        use_cache=True,
        show_progress=True,
        limit=limit,
    ):
        batch_sigs.append(rec.signal)
        batch_lbls.append(rec.label_vec)
        n += 1
        if len(batch_sigs) >= batch_size:
            _flush()
    _flush()

    logger.info("Fold 10: %d записей обработано", n)
    return np.stack(all_probs), np.stack(all_targets)


# ---------------------------------------------------------------------------
# Метрики по классам Ribeiro
# ---------------------------------------------------------------------------

def _compute_ribeiro_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Вычисляет F1-Fmax, sensitivity, specificity для каждого из
    10 классов Ribeiro на fold 10.
    """
    from preprocessing.snomed_map import SNOMED_TO_INDEX, ABBR_TO_SNOMED

    results = {}
    for abbr in RIBEIRO_CLASSES:
        snomed = ABBR_TO_SNOMED.get(abbr)
        if snomed is None:
            logger.warning("SNOMED код для %s не найден", abbr)
            continue
        idx = SNOMED_TO_INDEX.get(snomed)
        if idx is None:
            logger.warning("Индекс для %s (SNOMED %d) не найден", abbr, snomed)
            continue

        y_score = probs[:, idx]
        y_true  = targets[:, idx]
        n_pos   = int(y_true.sum())

        if n_pos == 0:
            logger.warning("%s: нет позитивных примеров в fold 10", abbr)
            results[abbr] = {
                "f1": None, "sensitivity": None, "specificity": None,
                "auc": None, "n_positive": 0,
            }
            continue

        from sklearn.metrics import (
            f1_score, confusion_matrix, roc_auc_score,
        )

        # Fmax: оптимальный порог по F1
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.02, 0.98, 97):
            preds = (y_score >= thr).astype(int)
            f1    = float(f1_score(y_true, preds, zero_division=0))
            if f1 > best_f1:
                best_f1  = f1
                best_thr = float(thr)

        preds_opt = (y_score >= best_thr).astype(int)
        cm        = confusion_matrix(y_true, preds_opt, labels=[0, 1])

        tn, fp  = int(cm[0, 0]), int(cm[0, 1])
        fn, tp  = int(cm[1, 0]), int(cm[1, 1])
        sens    = float(tp / max(tp + fn, 1))
        spec    = float(tn / max(tn + fp, 1))

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None

        results[abbr] = {
            "f1":          round(best_f1, 4),
            "fmax_threshold": round(best_thr, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "auc":         round(auc, 4) if auc else None,
            "n_positive":  n_pos,
        }

    return results


# ---------------------------------------------------------------------------
# Таблица сравнения
# ---------------------------------------------------------------------------

def _build_comparison_table(
    stage_results: Dict[str, Dict],
    stages: List[str],
) -> List[Dict]:
    """Строит список строк для сравнительной таблицы."""
    rows = []

    # Строка Ribeiro 2020
    for cls, ref in RIBEIRO_REFERENCE.items():
        rows.append({
            "class":  cls,
            "stage":  "Ribeiro 2020 (reference)",
            "f1":     ref["f1"],
            "sensitivity": ref["sensitivity"],
            "specificity": ref["specificity"],
            "auc":    None,
            "delta_f1": 0.0,
        })

    for stage in stages:
        res = stage_results.get(stage, {})
        for cls in RIBEIRO_CLASSES:
            cls_res = res.get(cls, {})
            ref_f1  = RIBEIRO_REFERENCE.get(cls, {}).get("f1")
            our_f1  = cls_res.get("f1")
            delta   = round(our_f1 - ref_f1, 4) if (our_f1 and ref_f1) else None
            rows.append({
                "class":  cls,
                "stage":  stage,
                "f1":     our_f1,
                "sensitivity": cls_res.get("sensitivity"),
                "specificity": cls_res.get("specificity"),
                "auc":    cls_res.get("auc"),
                "delta_f1": delta,
            })

    return rows


def _print_comparison(table: List[Dict], stages: List[str]) -> None:
    """Форматированный вывод таблицы сравнения."""
    print("\n" + "═" * 90)
    print("  Воспроизведение Ribeiro 2020 — PTB-XL fold 10")
    print("═" * 90)

    col_w = 16
    header = f"{'Класс':<8}"
    for s in ["Ribeiro 2020"] + stages:
        label = s.replace("_", " ")[:col_w]
        header += f"  {label:<{col_w}}"
    print(header)
    print("-" * 90)

    for cls in RIBEIRO_CLASSES:
        row_str = f"{cls:<8}"
        # Ribeiro reference
        ref = RIBEIRO_REFERENCE.get(cls, {})
        row_str += f"  {'F1=':>3}{ref.get('f1', '—'):<{col_w - 3}}"
        # Наши результаты
        for stage in stages:
            cls_data = _print_comparison.__dict__.get(f"_data_{stage}", {})
            # Найдём в table
            found = next(
                (r for r in table if r["class"] == cls and r["stage"] == stage), {}
            )
            f1    = found.get("f1")
            delta = found.get("delta_f1")
            if f1 is not None:
                sign = "+" if (delta or 0) >= 0 else ""
                cell = f"F1={f1:.3f}({sign}{delta:.3f})"
            else:
                cell = "—"
            row_str += f"  {cell:<{col_w}}"
        print(row_str)

    print("═" * 90)
    print("  δ = наш результат − Ribeiro 2020 (положительное = лучше)")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def run_reproduce(
    cfg,
    stages: Optional[List[str]] = None,
    limit: Optional[int] = None,
    save_preds: bool = True,
) -> Dict:
    """
    Запускает воспроизведение для всех указанных этапов.

    Parameters
    ----------
    cfg
    stages : list | None
        Список из 'baseline', 'pretrain', 'finetune'.
        По умолчанию запускает все доступные.
    limit : int | None
    save_preds : bool

    Returns
    -------
    dict с ключами: stage → per-class метрики + macro_auc + delta
    """
    import torch
    from training._common import get_device, compute_metrics

    device     = get_device()
    stages     = stages or ["baseline", "pretrain", "finetune"]
    results    = {}

    results_dir = Path(cfg.paths.results_dir) / "reproduce_ribeiro"
    results_dir.mkdir(parents=True, exist_ok=True)

    stage_results: Dict[str, Dict] = {}

    for stage in stages:
        logger.info("═══ Этап: %s ═══", stage)
        try:
            model = _load_model(cfg, stage, device)
        except FileNotFoundError as exc:
            logger.warning("Пропускаем %s: %s", stage, exc)
            results[stage] = {"error": str(exc)}
            continue

        probs, targets = run_inference_fold10(cfg, model, device, limit=limit)

        # Macro AUC / Fmax
        from evaluation.metrics import full_evaluation_report
        from preprocessing.snomed_map import SNOMED_TO_ABBR, SCORED_SNOMED_CLASSES
        class_names = [SNOMED_TO_ABBR.get(c, str(c)) for c in SCORED_SNOMED_CLASSES]

        full_report = full_evaluation_report(probs, targets, class_names)
        per_ribeiro = _compute_ribeiro_metrics(probs, targets)

        stage_res = {
            "macro_auc": full_report["macro_auc"],
            "macro_f1":  full_report["macro_f1"],
            "fmax":      full_report["fmax"],
            "per_ribeiro_class": per_ribeiro,
        }
        stage_results[stage] = per_ribeiro

        # Delta vs Ribeiro
        deltas = {}
        for cls, ref in RIBEIRO_REFERENCE.items():
            our = per_ribeiro.get(cls, {})
            our_f1 = our.get("f1")
            if our_f1 is not None:
                deltas[cls] = round(our_f1 - ref["f1"], 4)
        stage_res["delta_f1_vs_ribeiro"] = deltas
        stage_res["mean_delta_f1"] = round(
            float(np.mean([v for v in deltas.values() if v is not None])), 4
        ) if deltas else None

        results[stage] = stage_res

        # Сохраняем метрики
        _save_json(stage_res, results_dir / f"{stage}_metrics.json")

        if save_preds:
            np.save(str(results_dir / f"{stage}_probs.npy"), probs)
            np.save(str(results_dir / f"{stage}_targets.npy"), targets)

        logger.info(
            "%s: macro_AUC=%.4f  Fmax=%.4f  mean_ΔF1=%s",
            stage, stage_res["macro_auc"], stage_res["fmax"],
            stage_res.get("mean_delta_f1"),
        )

    # Таблица сравнения
    available = [s for s in stages if s in stage_results]
    if available:
        comp_table = _build_comparison_table(stage_results, available)
        _print_comparison(comp_table, available)
        import pandas as pd
        df = pd.DataFrame([r for r in comp_table if r["stage"] != "Ribeiro 2020 (reference)"])
        df.to_csv(str(results_dir / "comparison_table.csv"), index=False)
        _save_json(comp_table, results_dir / "comparison_table.json")

    return results


# ---------------------------------------------------------------------------
# Утилита сохранения JSON
# ---------------------------------------------------------------------------

def _save_json(data, path: Path) -> None:
    def _ser(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, np.ndarray):     return v.tolist()
        if isinstance(v, dict):   return {k: _ser(vv) for k, vv in v.items()}
        if isinstance(v, list):   return [_ser(x) for x in v]
        return v
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_ser(data), f, indent=2, ensure_ascii=False)
    logger.info("Сохранено: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Воспроизведение метрик Ribeiro 2020 на PTB-XL fold 10"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--stage", nargs="+",
        choices=["baseline", "pretrain", "finetune"],
        default=None,
        help="Этапы для воспроизведения (по умолчанию: все доступные)",
    )
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--no-save-preds", action="store_true")
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training._common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (20 if args.debug else None)
    run_reproduce(
        cfg,
        stages=args.stage,
        limit=limit,
        save_preds=not args.no_save_preds,
    )


if __name__ == "__main__":
    main()
