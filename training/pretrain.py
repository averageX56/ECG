"""
training/pretrain.py
Шаг 1: backbone заморожен, обучаются только task-specific головы.

Данные  : PhysioNet 2020 (все подмножества кроме CPSC) + PTB-XL fold-split
Train   : PTB-XL folds 1–8 + Georgia + StPetersburg + PTB (PhysioNet 2020)
Val     : PTB-XL fold 9
Test    : PTB-XL fold 10 (НИКОГДА не используется при обучении)

Запуск:
  python -m training.pretrain --config configs/default.yaml
  python -m training.pretrain --config configs/default.yaml --limit 500 --debug
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Гарантируем что корень проекта в PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training._common import (
    EarlyStopping,
    CheckpointManager,
    StepTimer,
    TrainingLogger,
    build_dataloader,
    collect_predictions,
    compute_metrics,
    compute_pos_weight,
    configure_root_logger,
    get_device,
    load_config,
    save_metrics,
    setup_logger,
    setup_seed,
    worker_init_fn,
)
from backbone.resnet1d import ResNet1dWithHead
from backbone.load_weights import load_pretrained_weights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных
# ─────────────────────────────────────────────────────────────────────────────

def _load_ptbxl_records(cfg, splits: List[str]) -> List:
    """
    Загружает PTB-XL записи для указанных сплитов.
    PTB-XL входит в PhysioNet 2020 как подмножество — грузим напрямую через
    iter_ptbxl, чтобы использовать fold-split.
    """
    from data.load_ptbxl import iter_ptbxl

    records = []
    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root

    logger.info("Загрузка PTB-XL (%s) из %s …", splits, ptbxl_root)
    for rec in iter_ptbxl(
        root=ptbxl_root,
        splits=splits,
        min_likelihood=float(cfg.data.ptbxl_min_likelihood),
        use_cache=True,
    ):
        records.append(rec)

    logger.info("PTB-XL (%s): %d записей", splits, len(records))
    return records


def _load_physionet2020_train_records(cfg, limit: Optional[int] = None) -> List:
    """
    Загружает все подмножества PhysioNet 2020 кроме PTB-XL и CPSC.
    (PTB-XL грузится отдельно через iter_ptbxl для fold-split)
    (CPSC используется только в шаге 2)
    """
    from data.load_physionet2020 import iter_physionet2020

    skip = set(cfg.data.skip_sources)  # {'PTBXL', 'CPSC2018', 'CPSC-Extra'}

    logger.info(
        "Загрузка PhysioNet 2020 (skip=%s) из %s …",
        skip, cfg.paths.data_root,
    )
    records = []
    for rec in iter_physionet2020(
        root=cfg.paths.data_root,
        skip_sources=skip,
        use_cache=True,
        limit=limit,
    ):
        # Все не-PTB-XL записи → train (нет fold-split)
        rec.split = "train"
        records.append(rec)

    logger.info("PhysioNet 2020 (не PTB-XL / CPSC): %d записей", len(records))
    return records


def load_all_records(
    cfg,
    limit: Optional[int] = None,
) -> Tuple[List, List]:
    """
    Собирает все записи для шага 1.

    Returns
    -------
    train_records : list[RecordA]
    val_records   : list[RecordA]
    """
    # PTB-XL train (folds 1-8) + val (fold 9)
    ptbxl_train = _load_ptbxl_records(cfg, splits=["train"])
    ptbxl_val   = _load_ptbxl_records(cfg, splits=["val"])

    # Остальные подмножества PhysioNet 2020 → только train
    pn2020_train = _load_physionet2020_train_records(cfg, limit=limit)

    train_records = ptbxl_train + pn2020_train
    val_records   = ptbxl_val

    logger.info(
        "Итого: train=%d val=%d",
        len(train_records), len(val_records),
    )
    return train_records, val_records


# ─────────────────────────────────────────────────────────────────────────────
# Построение датасетов и DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    train_records: List,
    val_records: List,
    cfg,
    seed: int = 42,
) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Создаёт train и val DataLoader.

    Train — WeightedRandomSampler (inv_sqrt по подмножеству).
    Val   — sequential (без shuffle).
    """
    from data.unified_dataset import build_unified_dataset

    train_ds, train_w = build_unified_dataset(
        records  = train_records,
        split    = "train",
        strategy = str(cfg.data.sampling_strategy),
        augment  = bool(cfg.data.augment_train),
    )
    val_ds, _ = build_unified_dataset(
        records  = val_records,
        split    = "val",
        strategy = "uniform",
        augment  = False,
    )

    # Логируем ожидаемое распределение подмножеств за эпоху
    train_ds.log_epoch_distribution(n_samples=int(cfg.pretrain.batch_size) * 100)

    train_sampler = train_ds.make_sampler()

    train_loader = build_dataloader(
        train_ds,
        batch_size  = int(cfg.pretrain.batch_size),
        sampler     = train_sampler,
        n_workers   = int(cfg.data.n_workers),
        pin_memory  = bool(cfg.data.pin_memory),
        prefetch_factor = int(cfg.data.prefetch_factor),
        seed        = seed,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size  = int(cfg.pretrain.batch_size) * 2,
        shuffle     = False,
        n_workers   = int(cfg.data.n_workers),
        pin_memory  = bool(cfg.data.pin_memory),
        prefetch_factor = int(cfg.data.prefetch_factor),
        seed        = seed,
    )

    logger.info(
        "DataLoader: train=%d batches, val=%d batches",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Построение модели
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg, device: torch.device) -> ResNet1dWithHead:
    """
    Создаёт ResNet1dWithHead, загружает предобученные веса, замораживает backbone.
    """
    model = ResNet1dWithHead(
        n_classes      = int(cfg.pretrain.n_classes),
        backbone_kwargs = {
            "n_leads": int(cfg.backbone.n_leads),
            "dropout": float(cfg.backbone.dropout),
        },
        dropout_head   = float(cfg.pretrain.dropout_head),
    )

    # Загрузка предобученных Zenodo-весов
    if cfg.backbone.load_pretrained:
        cache_dir = Path(cfg.paths.backbone_cache).expanduser()
        result = load_pretrained_weights(
            model.backbone,
            cache_dir = cache_dir,
            strict    = bool(cfg.backbone.pretrained_strict),
        )
        logger.info(
            "Веса backbone: загружено %d/%d ключей",
            result["loaded"], result["total"],
        )

    # Шаг 1: backbone заморожен, только головы обучаются
    model.backbone.freeze_all()
    logger.info(model.backbone.param_summary())

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Обучение одной эпохи
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:      ResNet1dWithHead,
    loader:     "torch.utils.data.DataLoader",
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    cfg,
    epoch:      int,
    tlog:       TrainingLogger,
    global_step: int,
) -> Tuple[float, int]:
    """
    Одна эпоха обучения.

    Returns
    -------
    avg_loss : float
    global_step : int  (обновлённый)
    """
    model.train()
    # Backbone заморожен, но BN-слои должны быть в eval режиме
    model.backbone.eval()

    total_loss = 0.0
    n_batches  = 0
    timer      = StepTimer()
    log_every  = int(cfg.logging.log_every_n_steps)
    grad_clip  = float(cfg.pretrain.grad_clip)

    for signals, labels in loader:
        timer.start()
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        timer.stop()

        total_loss += loss.item()
        n_batches  += 1
        global_step += 1

        if global_step % log_every == 0:
            avg = total_loss / n_batches
            tlog.log(
                {"train/loss": avg, "train/step_ms": timer.avg * 1000},
                step=global_step,
            )
            logger.info(
                "Эпоха %d  шаг %d  loss=%.4f  %.1f ms/step",
                epoch, global_step, avg, timer.avg * 1000,
            )

    return total_loss / max(n_batches, 1), global_step


# ─────────────────────────────────────────────────────────────────────────────
# Валидация
# ─────────────────────────────────────────────────────────────────────────────

def val_epoch(
    model:    ResNet1dWithHead,
    loader:   "torch.utils.data.DataLoader",
    device:   torch.device,
    cfg,
    n_classes: int,
) -> Dict:
    """
    Прогон на валидационном датасете.

    Returns
    -------
    dict  с ключами: macro_auc, macro_f1, fmax, per_class_auc, per_class_f1
    """
    probs, targets = collect_predictions(model, loader, device, n_classes)

    metrics = compute_metrics(
        probs=probs,
        targets=targets,
        n_classes=n_classes,
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция обучения
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain(cfg, limit: Optional[int] = None) -> Dict:
    """
    Полный цикл предобучения (шаг 1).

    Parameters
    ----------
    cfg : _DotDict
    limit : int | None  — ограничение числа записей (для отладки)

    Returns
    -------
    dict  — финальные метрики на val
    """
    seed = int(cfg.seed)
    setup_seed(seed)
    device = get_device()

    # ── Данные ────────────────────────────────────────────────────────────────
    train_records, val_records = load_all_records(cfg, limit=limit)
    train_loader, val_loader   = build_loaders(train_records, val_records, cfg, seed)

    # ── Модель ────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── pos_weight ────────────────────────────────────────────────────────────
    # Собираем метки из train
    train_labels = np.stack([r.label_vec for r in train_records])
    pos_weight   = compute_pos_weight(
        train_labels,
        n_classes = int(cfg.pretrain.n_classes),
        device    = device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Оптимизатор (только параметры с requires_grad=True → головы) ──────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Обучаемых параметров: %d (только головы, backbone заморожен)",
        sum(p.numel() for p in trainable),
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr           = float(cfg.pretrain.lr),
        weight_decay = float(cfg.pretrain.weight_decay),
    )

    # ── Логирование и чекпоинты ───────────────────────────────────────────────
    tlog = setup_logger(cfg, run_name="pretrain")
    tlog.watch(model)

    ckpt_mgr = CheckpointManager(
        checkpoint_dir = str(cfg.paths.checkpoint_dir),
        run_name       = "pretrain",
        save_last      = True,
    )

    early_stop = EarlyStopping(
        patience  = int(cfg.pretrain.patience),
        min_delta = float(cfg.pretrain.min_delta),
    )

    n_classes   = int(cfg.pretrain.n_classes)
    max_epochs  = int(cfg.pretrain.epochs)
    global_step = 0
    best_metrics: Dict = {}

    # ── Основной цикл ─────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        logger.info("── Эпоха %d/%d ──", epoch, max_epochs)

        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            device, cfg, epoch, tlog, global_step,
        )

        val_metrics = val_epoch(model, val_loader, device, cfg, n_classes)
        val_auc     = val_metrics["macro_auc"]

        logger.info(
            "Эпоха %d  train_loss=%.4f  val_macro_auc=%.4f  val_fmax=%.4f",
            epoch, train_loss, val_auc, val_metrics["fmax"],
        )
        tlog.log(
            {
                "train/epoch_loss": train_loss,
                "val/macro_auc":    val_auc,
                "val/macro_f1":     val_metrics["macro_f1"],
                "val/fmax":         val_metrics["fmax"],
                "epoch":            epoch,
            },
            step=global_step,
        )

        is_best = early_stop.improved
        early_stop(val_auc)

        ckpt_mgr.save(
            model     = model,
            optimizer = optimizer,
            epoch     = epoch,
            metric    = val_auc,
            config    = dict(cfg),
            is_best   = is_best,
        )

        if is_best:
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch

        if early_stop.should_stop:
            logger.info("Early stopping на эпохе %d", epoch)
            break

    # ── Финал ─────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir) / "pretrain"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(best_metrics, results_dir / "metrics.json")

    # Сохраняем предсказания на val (для ablation)
    if cfg.logging.save_preds:
        ckpt_mgr.load_best(model, device=device)
        probs, targets = collect_predictions(
            model, val_loader, device, n_classes
        )
        np.save(str(results_dir / "val_preds.npy"),   probs)
        np.save(str(results_dir / "val_targets.npy"), targets)
        logger.info("Val предсказания сохранены в %s", results_dir)

    tlog.finish()
    logger.info(
        "Pretrain завершён. Лучший val macro-AUC=%.4f (эпоха %d)",
        best_metrics.get("macro_auc", 0.0),
        best_metrics.get("epoch", -1),
    )
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Шаг 1: предобучение (backbone заморожен)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Путь к YAML-конфигу",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Максимум записей (для быстрого debug-прогона)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Режим отладки: limit=500, epochs=2, no wandb",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_root_logger(str(cfg.logging.level))

    limit = args.limit
    if args.debug:
        limit = limit or 500
        cfg["pretrain"]["epochs"] = 2
        cfg["pretrain"]["patience"] = 2
        cfg["logging"]["backend"] = "none"
        logger.info("DEBUG режим: limit=%d, epochs=2", limit)

    run_pretrain(cfg, limit=limit)


if __name__ == "__main__":
    main()