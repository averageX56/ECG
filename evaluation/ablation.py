"""
evaluation/ablation.py
Ablation study: сравнение конфигураций по подмножествам датасета.

Сценарии (из требований проекта):
  1. Baseline:    Zenodo-веса без дообучения
  2. PTB-XL only: только PTB-XL подмножество из PhysioNet 2020 (шаг 1)
  3. +Georgia+PTB+INCART: добавляем остальные подмножества
  4. +CPSC finetune: шаг 2, domain shift для AF/VT
  5. Weighted vs Unweighted: сравнение стратегий семплинга

Выходы:
  results/ablation/
    per_scenario_metrics.json
    ablation.csv   ← главная таблица для README
    delta_table.csv

Использование:
  python -m evaluation.ablation --config configs/default.yaml
  python -m evaluation.ablation --config configs/default.yaml --scenario 1 3 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# Классы особого интереса для ablation
FOCAL_CLASSES = [
    "AF", "VTach", "LBBB", "CRBBB", "IAVB", "SB", "STach", "NSR",
]

# Описания сценариев
SCENARIO_DESCRIPTIONS = {
    1: "Baseline: Zenodo-веса без дообучения",
    2: "Pretrain PTB-XL only (folds 1-8)",
    3: "Pretrain all subsets (PTB-XL + Georgia + PTB + INCART)",
    4: "Pretrain all + CPSC finetune (AF/VT)",
    5: "Weighted vs Unweighted sampling (delta AUC)",
}


# ---------------------------------------------------------------------------
# Загрузка предсказаний из файлов
# ---------------------------------------------------------------------------

def _load_preds(
    results_dir: Path,
    name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Загружает probs.npy и targets.npy по префиксу name."""
    probs_path   = results_dir / f"{name}_probs.npy"
    targets_path = results_dir / f"{name}_targets.npy"

    if not probs_path.exists() or not targets_path.exists():
        return None

    return np.load(str(probs_path)), np.load(str(targets_path))


# ---------------------------------------------------------------------------
# Сбор предсказаний для сценария
# ---------------------------------------------------------------------------

def _collect_preds_for_scenario(
    cfg,
    scenario: int,
    device,
    limit: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Запускает инференс для данного сценария.
    Возвращает (probs, targets) для PTB-XL fold 10.
    """
    import torch
    from backbone.resnet1d import ResNet1dWithHead
    from backbone.load_weights import load_pretrained_weights
    from data.load_ptbxl import iter_ptbxl

    ckpt_dir   = Path(cfg.paths.checkpoint_dir)
    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root

    # Определяем чекпоинт
    if scenario == 1:
        ckpt_path = None  # только Zenodo
    elif scenario == 2:
        ckpt_path = ckpt_dir / "ablation" / "pretrain_ptbxl_only_best.pt"
    elif scenario == 3:
        ckpt_path = ckpt_dir / "pretrain_best.pt"
    elif scenario == 4:
        ckpt_path = ckpt_dir / "finetune_best.pt"
    elif scenario == 5:
        # Два чекпоинта: weighted и unweighted
        # Возвращаем None, обрабатываем отдельно
        return None
    else:
        logger.warning("Неизвестный сценарий %d", scenario)
        return None

    # Создаём модель
    model = ResNet1dWithHead(
        n_classes=int(cfg.pretrain.n_classes),
        backbone_kwargs={
            "n_leads": int(cfg.backbone.n_leads),
            "dropout": 0.0,  # без dropout при инференсе
        },
        dropout_head=0.0,
    )

    if ckpt_path and ckpt_path.exists():
        logger.info("Загружаем %s", ckpt_path)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
    elif scenario == 1 or not ckpt_path:
        logger.info("Загружаем Zenodo-веса backbone")
        cache_dir = Path(cfg.paths.backbone_cache).expanduser()
        load_pretrained_weights(model.backbone, cache_dir=cache_dir, strict=False)
    else:
        logger.warning("Чекпоинт не найден: %s — пропускаем сценарий %d",
                       ckpt_path, scenario)
        return None

    model.to(device).eval()

    # Инференс
    batch_size  = int(cfg.pretrain.batch_size) * 2
    all_probs:  List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    batch_sigs: List[np.ndarray] = []
    batch_lbls: List[np.ndarray] = []

    def _flush():
        nonlocal batch_sigs, batch_lbls
        if not batch_sigs:
            return
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch_sigs)).to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.extend(list(p))
        all_targets.extend(batch_lbls)
        batch_sigs = []
        batch_lbls = []

    for rec in iter_ptbxl(
        root=ptbxl_root, splits=["test"],
        use_cache=True, show_progress=True, limit=limit,
    ):
        batch_sigs.append(rec.signal)
        batch_lbls.append(rec.label_vec)
        if len(batch_sigs) >= batch_size:
            _flush()
    _flush()

    if not all_probs:
        return None

    return np.stack(all_probs), np.stack(all_targets)


# ---------------------------------------------------------------------------
# Метрики для focal-классов
# ---------------------------------------------------------------------------

def _compute_focal_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    focal_classes: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """AUC + Fmax F1 для каждого focal-класса."""
    from preprocessing.snomed_map import SNOMED_TO_INDEX, ABBR_TO_SNOMED
    from sklearn.metrics import roc_auc_score, f1_score

    result = {}
    for abbr in focal_classes:
        snomed = ABBR_TO_SNOMED.get(abbr)
        idx    = SNOMED_TO_INDEX.get(snomed) if snomed else None
        if idx is None:
            continue

        y_true  = targets[:, idx]
        y_score = probs[:, idx]

        if y_true.sum() == 0:
            result[abbr] = {"auc": None, "f1_fmax": None, "n_pos": 0}
            continue

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = None

        best_f1 = 0.0
        for thr in np.linspace(0.02, 0.98, 97):
            f1 = float(f1_score(y_true, (y_score >= thr).astype(int), zero_division=0))
            if f1 > best_f1:
                best_f1 = f1

        result[abbr] = {
            "auc":     round(auc, 4) if auc else None,
            "f1_fmax": round(best_f1, 4),
            "n_pos":   int(y_true.sum()),
        }

    return result


# ---------------------------------------------------------------------------
# Сценарий 5: Weighted vs Unweighted
# ---------------------------------------------------------------------------

def run_scenario5_sampling(
    cfg,
    results_dir: Path,
    limit: Optional[int] = None,
) -> Dict:
    """
    Сравнивает weighted (inv_sqrt) vs uniform sampling
    по AUC на малых классах.

    Ожидает наличия двух чекпоинтов:
      checkpoints/ablation/pretrain_weighted_best.pt
      checkpoints/ablation/pretrain_unweighted_best.pt
    """
    from training.common import get_device
    device = get_device()

    ckpt_dir = Path(cfg.paths.checkpoint_dir) / "ablation"
    configs  = {
        "weighted":   ckpt_dir / "pretrain_weighted_best.pt",
        "unweighted": ckpt_dir / "pretrain_unweighted_best.pt",
    }

    scenario_result: Dict[str, Any] = {}

    for name, ckpt in configs.items():
        if not ckpt.exists():
            logger.warning("Чекпоинт %s не найден: %s", name, ckpt)
            scenario_result[name] = {"error": f"чекпоинт не найден: {ckpt}"}
            continue

        import torch
        from backbone.resnet1d import ResNet1dWithHead

        model = ResNet1dWithHead(
            n_classes=int(cfg.pretrain.n_classes),
            backbone_kwargs={
                "n_leads": int(cfg.backbone.n_leads),
                "dropout": 0.0,
            },
            dropout_head=0.0,
        )
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)

        # Собираем предсказания через reproduce_ribeiro helper
        from evaluation.reproduce_ribeiro import run_inference_fold10
        probs, targets = run_inference_fold10(cfg, model.to(device), device, limit=limit)

        from evaluation.metrics import full_evaluation_report
        from preprocessing.snomed_map import SNOMED_TO_ABBR, SCORED_SNOMED_CLASSES
        class_names = [SNOMED_TO_ABBR.get(c, str(c)) for c in SCORED_SNOMED_CLASSES]

        report = full_evaluation_report(probs, targets, class_names)
        focal  = _compute_focal_metrics(probs, targets, FOCAL_CLASSES)

        # Особый интерес: малые классы (< 1% prevalence)
        small_class_aucs = [
            m["auc"] for m in report["per_class"] 
            if m["prevalence"] < 0.01 and m["auc"] is not None
        ]

        scenario_result[name] = {
            "macro_auc":       report["macro_auc"],
            "fmax":            report["fmax"],
            "focal_classes":   focal,
            "small_class_auc": round(float(np.mean(small_class_aucs)), 4)
                               if small_class_aucs else None,
        }
        np.save(str(results_dir / f"sampling_{name}_probs.npy"), probs)

    # Delta: weighted - unweighted
    if "weighted" in scenario_result and "unweighted" in scenario_result:
        w  = scenario_result["weighted"]
        uw = scenario_result["unweighted"]
        if "macro_auc" in w and "macro_auc" in uw:
            delta_macro = round(w["macro_auc"] - uw["macro_auc"], 4)
            delta_small = None
            if w.get("small_class_auc") and uw.get("small_class_auc"):
                delta_small = round(w["small_class_auc"] - uw["small_class_auc"], 4)
            scenario_result["delta"] = {
                "macro_auc":       delta_macro,
                "small_class_auc": delta_small,
                "note":            "положительное = weighted лучше",
            }

    return scenario_result


# ---------------------------------------------------------------------------
# Основная функция ablation
# ---------------------------------------------------------------------------

def run_ablation(
    cfg,
    scenarios: Optional[List[int]] = None,
    limit: Optional[int] = None,
) -> Dict:
    """
    Запускает ablation study для указанных сценариев.

    Parameters
    ----------
    cfg
    scenarios : list[int] | None
        Сценарии 1–5. По умолчанию все.
    limit : int | None

    Returns
    -------
    dict с результатами по каждому сценарию
    """
    from training.common import get_device
    from evaluation.metrics import full_evaluation_report
    from preprocessing.snomed_map import SNOMED_TO_ABBR, SCORED_SNOMED_CLASSES

    device = get_device()
    class_names = [SNOMED_TO_ABBR.get(c, str(c)) for c in SCORED_SNOMED_CLASSES]

    scenarios    = scenarios or [1, 2, 3, 4, 5]
    results_dir  = Path(cfg.paths.results_dir) / "ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[int, Any] = {}

    for sc in scenarios:
        logger.info("═══ Ablation сценарий %d: %s ═══",
                    sc, SCENARIO_DESCRIPTIONS.get(sc, ""))

        if sc == 5:
            res = run_scenario5_sampling(cfg, results_dir, limit=limit)
            all_results[sc] = {"description": SCENARIO_DESCRIPTIONS[sc], **res}
            _save_json(all_results[sc], results_dir / f"scenario_{sc}.json")
            continue

        preds = _collect_preds_for_scenario(cfg, sc, device, limit=limit)
        if preds is None:
            logger.warning("Сценарий %d: нет данных/чекпоинта — пропускаем", sc)
            all_results[sc] = {
                "description": SCENARIO_DESCRIPTIONS.get(sc, ""),
                "error":       "чекпоинт недоступен",
            }
            continue

        probs, targets = preds

        # Сохраняем предсказания
        np.save(str(results_dir / f"scenario_{sc}_probs.npy"),   probs)
        np.save(str(results_dir / f"scenario_{sc}_targets.npy"), targets)

        # Метрики
        report = full_evaluation_report(probs, targets, class_names)
        focal  = _compute_focal_metrics(probs, targets, FOCAL_CLASSES)

        sc_result = {
            "description": SCENARIO_DESCRIPTIONS.get(sc, ""),
            "macro_auc":   report["macro_auc"],
            "macro_f1":    report["macro_f1"],
            "fmax":        report["fmax"],
            "focal_classes": focal,
        }
        all_results[sc] = sc_result
        _save_json(sc_result, results_dir / f"scenario_{sc}.json")

        logger.info(
            "Сценарий %d: macro_AUC=%.4f  Fmax=%.4f",
            sc, sc_result["macro_auc"], sc_result["fmax"],
        )

    # Итоговая CSV-таблица
    _build_ablation_csv(all_results, scenarios, results_dir)
    _save_json(all_results, results_dir / "per_scenario_metrics.json")
    _print_ablation_summary(all_results, scenarios)

    return all_results


# ---------------------------------------------------------------------------
# Итоговая CSV
# ---------------------------------------------------------------------------

def _build_ablation_csv(
    all_results: Dict,
    scenarios: List[int],
    results_dir: Path,
) -> None:
    """Строит ablation.csv для README."""
    import pandas as pd

    rows_summary = []
    rows_detail  = []

    for sc in scenarios:
        res = all_results.get(sc, {})
        if "error" in res:
            continue

        if sc == 5:
            for variant in ("weighted", "unweighted"):
                sub = res.get(variant, {})
                if not sub or "error" in sub:
                    continue
                rows_summary.append({
                    "Сценарий":   f"{sc}_{variant}",
                    "Описание":   f"{res.get('description', '')} [{variant}]",
                    "macro_AUC":  sub.get("macro_auc"),
                    "Fmax":       sub.get("fmax"),
                    "small_class_AUC": sub.get("small_class_auc"),
                })
                _add_focal_rows(rows_detail, sub.get("focal_classes", {}),
                                f"{sc}_{variant}", res.get("description", ""))
            continue

        rows_summary.append({
            "Сценарий":   sc,
            "Описание":   res.get("description", ""),
            "macro_AUC":  res.get("macro_auc"),
            "Fmax":       res.get("fmax"),
            "small_class_AUC": None,
        })
        _add_focal_rows(rows_detail, res.get("focal_classes", {}),
                        sc, res.get("description", ""))

    if rows_summary:
        df_s = pd.DataFrame(rows_summary)
        df_s.to_csv(str(results_dir / "ablation.csv"), index=False)
        logger.info("Ablation summary: %s", results_dir / "ablation.csv")

    if rows_detail:
        df_d = pd.DataFrame(rows_detail)
        df_d.to_csv(str(results_dir / "delta_table.csv"), index=False)
        logger.info("Delta table: %s", results_dir / "delta_table.csv")


def _add_focal_rows(
    rows: List,
    focal: Dict,
    scenario_id: Any,
    description: str,
) -> None:
    for cls, metrics in focal.items():
        rows.append({
            "Сценарий": scenario_id,
            "Описание": description,
            "Класс":    cls,
            "AUC":      metrics.get("auc"),
            "F1_Fmax":  metrics.get("f1_fmax"),
            "N_pos":    metrics.get("n_pos"),
        })


# ---------------------------------------------------------------------------
# Вывод сводки
# ---------------------------------------------------------------------------

def _print_ablation_summary(all_results: Dict, scenarios: List[int]) -> None:
    print("\n" + "═" * 80)
    print("  Ablation Study — PTB-XL fold 10")
    print("═" * 80)
    print(f"{'№':>3}  {'Описание':<45}  {'macro-AUC':>10}  {'Fmax':>7}")
    print("-" * 80)

    for sc in scenarios:
        res = all_results.get(sc, {})
        if "error" in res:
            print(f"{sc:>3}  {res.get('description', ''):<45}  {'—':>10}  {'—':>7}")
            continue
        if sc == 5:
            for variant in ("weighted", "unweighted"):
                sub = res.get(variant, {})
                if not sub or "error" in sub:
                    continue
                label = f"{res.get('description', '')} [{variant}]"
                auc   = sub.get("macro_auc")
                fmax  = sub.get("fmax")
                print(f"{sc}{variant[0]:>1}  {label:<45}  "
                      f"{auc if auc else '—':>10}  {fmax if fmax else '—':>7}")
            delta = res.get("delta", {})
            if delta:
                print(f"     → Δmacro_AUC(weighted-unweighted) = "
                      f"{delta.get('macro_auc_delta', '?')}  "
                      f"Δsmall_class = {delta.get('small_class_auc', '?')}")
            continue

        auc   = res.get("macro_auc")
        fmax  = res.get("fmax")
        desc  = res.get("description", "")[:45]
        auc_s = f"{auc:.4f}" if auc else "—"
        fmax_s = f"{fmax:.4f}" if fmax else "—"
        print(f"{sc:>3}  {desc:<45}  {auc_s:>10}  {fmax_s:>7}")

    print("═" * 80)

    # Сводка по focal-классам
    print("\nFocal-классы (макс. клинический интерес):")
    print(f"{'Класс':<10}", end="")
    for sc in scenarios:
        if sc == 5:
            continue
        print(f"  {'Сц.'+str(sc):>8}", end="")
    print()

    for cls in FOCAL_CLASSES:
        print(f"  {cls:<8}", end="")
        for sc in scenarios:
            if sc == 5:
                continue
            res = all_results.get(sc, {})
            auc = res.get("focal_classes", {}).get(cls, {}).get("auc")
            print(f"  {auc if auc else '—':>8.4f}" if auc else f"  {'—':>8}", end="")
        print()
    print()


# ---------------------------------------------------------------------------
# Утилита сохранения JSON
# ---------------------------------------------------------------------------

def _save_json(data, path: Path) -> None:
    def _ser(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, np.ndarray):     return v.tolist()
        if isinstance(v, dict):  return {k: _ser(vv) for k, vv in v.items()}
        if isinstance(v, list):  return [_ser(x) for x in v]
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
        description="Ablation study по подмножествам и стратегиям семплинга"
    )
    parser.add_argument("--config",   type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--scenario", nargs="+", type=int,
        choices=[1, 2, 3, 4, 5], default=None,
        help="Сценарии ablation (1-5). По умолчанию все.",
    )
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training.common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (30 if args.debug else None)
    run_ablation(cfg, scenarios=args.scenario, limit=limit)


if __name__ == "__main__":
    main()
