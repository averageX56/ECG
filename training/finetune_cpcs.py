"""
training/finetune_cpsc.py
Шаг 2: domain adaptation на CPSC 2018 + CPSC-Extra.

  - Размораживаем последние 2 residual-блока backbone (res_blocks.2, res_blocks.3)
  - Дифференциальный lr: lr_backbone=1e-5, lr_heads=1e-4
  - BCE loss маскирован на классы AF (idx=0) и VT (idx=26)
  - Тест проводится на PTB-XL fold 10 (неизменный)
  - Цель: измерить delta AUC на AF/VT vs шага 1

Запуск:
  python -m training.finetune_cpsc --config configs/default.yaml
  python -m training.finetune_cpsc --config configs/default.yaml --debug
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training._common import (
    EarlyStopping,
    CheckpointManager,
    StepTimer,
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
)
from backbone.resnet1d import ResNet1dWithHead
from backbone.load_weights import load_pretrained_weights

logger = logging.getLogger(__name__)

# Индексы AF и VT в 27-классовом словаре SNOMED (из snomed_map.py _SCORED)
_AF_IDX  = 0   # SNOMED 164889003
_VT_IDX  = 26  # SNOMED 164896001
_FINETUNE_CLASS_INDICES = [_AF_IDX, _VT_IDX]


# ─────────────────────────────────────────────────────────────────────────────
# Маскированный BCE-loss для AF/VT
# ─────────────────────────────────────────────────────────────────────────────

class MaskedBCELoss(nn.Module):
    """
    BCE loss только по указанным классам.

    Parameters
    ----------
    class_indices : list[int]
        Индексы классов, по которым считается loss.
    pos_weight : torch.Tensor | None
        Вес позитивного класса [n_target_classes].
    """

    def __init__(
        self,
        class_indices: List[int],
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_idx",
            torch.tensor(class_indices, dtype=torch.long),
        )
        self._criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight, reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, n_classes]
        targets : [B, n_classes]
        """
        idx = self.class_idx
        return self._criterion(logits[:, idx], targets[:, idx])


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных CPSC
# ─────────────────────────────────────────────────────────────────────────────

def _load_cpsc_records(
    cfg,
    split: str,
    limit: Optional[int] = None,
) -> List:
    """
    Загружает CPSC 2018 + CPSC-Extra для fine-tune.
    Случайный train/val сплит 90/10 (по умолчанию итератор отдаёт все в train).
    """
    from data.load_cpsc2018 import iter_cpsc2018
    from data.subset_registry import resolve_path

    records: List = []

    for subset_name, ds_name in (
        ("cpsc",       "cpsc2018"),
        ("cpsc_extra", "cpsc_extra"),
    ):
        try:
            root = resolve_path(subset_name, str(cfg.paths.data_root))
            count = 0
            for rec in iter_cpsc2018(
                root=root,
                dataset_name=ds_name,
                split=split,
                use_cache=True,
            ):
                rec.split = split
                records.append(rec)
                count += 1
                if limit and count >= limit:
                    break
            logger.info("%s (%s): %d записей", ds_name, split, count)
        except FileNotFoundError as exc:
            logger.warning("Пропускаем %s: %s", subset_name, exc)

    return records


def _random_split_records(
    records: List,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List]:
    """
    Случайный train/val сплит списка записей.

    CPSC не имеет официального fold-split → используем random 90/10.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_val = max(1, int(len(records) * val_fraction))
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    train = [records[i] for i in train_idx]
    val   = [records[i] for i in val_idx]

    for r in train:
        r.split = "train"
    for r in val:
        r.split = "val"

    logger.info("CPSC random split: train=%d val=%d", len(train), len(val))
    return train, val


def _load_ptbxl_test_records(cfg) -> List:
    """Загружает PTB-XL fold 10 (test) для финальной оценки."""
    from data.load_ptbxl import iter_ptbxl

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    records = []
    for rec in iter_ptbxl(
        root=ptbxl_root,
        splits=["test"],
        use_cache=True,
    ):
        records.append(rec)
    logger.info("PTB-XL fold 10 (test): %d записей", len(records))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Построение модели
# ─────────────────────────────────────────────────────────────────────────────

def build_finetuned_model(
    cfg,
    device: torch.device,
    pretrain_ckpt: Optional[Path] = None,
) -> ResNet1dWithHead:
    """
    Загружает чекпоинт шага 1, размораживает последние 2 блока.
    """
    model = ResNet1dWithHead(
        n_classes       = int(cfg.pretrain.n_classes),
        backbone_kwargs = {
            "n_leads": int(cfg.backbone.n_leads),
            "dropout": float(cfg.backbone.dropout),
        },
        dropout_head    = float(cfg.finetune.dropout_head),
    )

    ckpt_path = pretrain_ckpt or Path(cfg.finetune.pretrain_checkpoint)

    if ckpt_path.exists():
        logger.info("Загрузка pretrain чекпоинта: %s", ckpt_path)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
    else:
        logger.warning(
            "Pretrain чекпоинт не найден: %s → инициализация с Zenodo-весами",
            ckpt_path,
        )
        if cfg.backbone.load_pretrained:
            cache_dir = Path(cfg.paths.backbone_cache).expanduser()
            load_pretrained_weights(
                model.backbone, cache_dir=cache_dir, strict=False
            )

    # Размораживаем последние N блоков backbone
    n_unfreeze = int(cfg.finetune.unfreeze_last_n)
    model.backbone.freeze_all()
    model.backbone.unfreeze_last_n(n_unfreeze)
    logger.info(model.backbone.param_summary())

    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Оптимизатор с дифференциальным lr
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: ResNet1dWithHead,
    cfg,
) -> torch.optim.Optimizer:
    """
    Создаёт AdamW с двумя группами параметров:
      - backbone (разморожен): lr_backbone=1e-5
      - head:                  lr_heads=1e-4
    """
    lr_bb  = float(cfg.finetune.lr_backbone)
    lr_hd  = float(cfg.finetune.lr_heads)
    wd     = float(cfg.finetune.weight_decay)

    # Параметры backbone (только unfrozen)
    backbone_params = [
        p for p in model.backbone.parameters() if p.requires_grad
    ]
    # Параметры head
    head_params = list(model.head.parameters()) + list(model.dropout.parameters())

    param_groups = [
        {"params": backbone_params, "lr": lr_bb, "name": "backbone"},
        {"params": head_params,     "lr": lr_hd, "name": "head"},
    ]

    logger.info(
        "Оптимизатор: backbone lr=%.1e (%d param), head lr=%.1e (%d param)",
        lr_bb, sum(p.numel() for p in backbone_params),
        lr_hd, sum(p.numel() for p in head_params),
    )

    return torch.optim.AdamW(param_groups, weight_decay=wd)


# ─────────────────────────────────────────────────────────────────────────────
# Обучение одной эпохи
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:       ResNet1dWithHead,
    loader:      "torch.utils.data.DataLoader",
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    cfg,
    epoch:       int,
    tlog,
    global_step: int,
) -> Tuple[float, int]:
    model.train()
    # Замороженные BN-слои в eval (чтобы не обновлять статистику)
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            # Проверяем: параметры frozen → BN в eval
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()

    total_loss = 0.0
    n_batches  = 0
    timer      = StepTimer()
    log_every  = int(cfg.logging.log_every_n_steps)
    grad_clip  = float(cfg.finetune.grad_clip)

    for signals, labels in loader:
        timer.start()
        signals = signals.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()
        timer.stop()

        total_loss += loss.item()
        n_batches  += 1
        global_step += 1

        if global_step % log_every == 0:
            avg = total_loss / n_batches
            tlog.log(
                {"finetune/loss": avg, "finetune/step_ms": timer.avg * 1000},
                step=global_step,
            )

    avg_loss = total_loss / max(n_batches, 1)
    logger.info("Эпоха %d  finetune_loss=%.4f", epoch, avg_loss)
    return avg_loss, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(
    cfg,
    pretrain_ckpt: Optional[Path] = None,
    limit:         Optional[int]  = None,
) -> Dict:
    """
    Полный цикл fine-tune (шаг 2).

    Returns
    -------
    dict  — финальные метрики на PTB-XL fold 10
    """
    seed = int(cfg.seed)
    setup_seed(seed)
    device = get_device()

    # ── Данные ────────────────────────────────────────────────────────────────
    all_cpsc = _load_cpsc_records(cfg, split="train", limit=limit)
    train_records, val_records = _random_split_records(
        all_cpsc, val_fraction=0.1, seed=seed
    )
    test_records = _load_ptbxl_test_records(cfg)

    # Unified dataset для CPSC
    from data.unified_dataset import build_unified_dataset

    train_ds, train_w = build_unified_dataset(
        train_records, split="train", strategy="uniform", augment=True
    )
    val_ds, _ = build_unified_dataset(
        val_records, split="val", strategy="uniform", augment=False
    )
    # Test dataset (PTB-XL fold 10)
    test_ds, _ = build_unified_dataset(
        test_records, split="test", strategy="uniform", augment=False
    )

    train_sampler = train_ds.make_sampler()
    bs = int(cfg.finetune.batch_size)

    train_loader = build_dataloader(
        train_ds, batch_size=bs, sampler=train_sampler,
        n_workers=int(cfg.data.n_workers), seed=seed,
    )
    val_loader = build_dataloader(
        val_ds, batch_size=bs * 2, shuffle=False,
        n_workers=int(cfg.data.n_workers), seed=seed,
    )
    test_loader = build_dataloader(
        test_ds, batch_size=bs * 2, shuffle=False,
        n_workers=int(cfg.data.n_workers), seed=seed,
    )

    logger.info(
        "DataLoader: train=%d val=%d test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    # ── Модель ────────────────────────────────────────────────────────────────
    model = build_finetuned_model(cfg, device, pretrain_ckpt)

    # ── pos_weight только для AF/VT ───────────────────────────────────────────
    train_labels = np.stack([r.label_vec for r in train_records])
    # Берём только AF/VT колонки
    af_vt_labels = train_labels[:, _FINETUNE_CLASS_INDICES]
    n = len(af_vt_labels)
    pos_counts = af_vt_labels.sum(axis=0)
    neg_counts = n - pos_counts
    pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 50.0)
    pos_weight_tensor = torch.from_numpy(
        pos_weight_vals.astype(np.float32)
    ).to(device)

    criterion = MaskedBCELoss(
        class_indices=_FINETUNE_CLASS_INDICES,
        pos_weight=pos_weight_tensor,
    ).to(device)

    logger.info(
        "MaskedBCELoss на классы: AF(idx=%d) VT(idx=%d), pos_weight=%s",
        _AF_IDX, _VT_IDX, pos_weight_vals,
    )

    # ── Оптимизатор ───────────────────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)

    # ── Логирование и чекпоинты ───────────────────────────────────────────────
    tlog = setup_logger(cfg, run_name="finetune_cpsc")
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=str(cfg.paths.checkpoint_dir),
        run_name="finetune",
    )
    early_stop = EarlyStopping(
        patience  = int(cfg.finetune.patience),
        min_delta = float(cfg.pretrain.min_delta),
    )

    n_classes  = int(cfg.pretrain.n_classes)
    max_epochs = int(cfg.finetune.epochs)
    global_step = 0
    best_metrics: Dict = {}

    # ── Основной цикл ─────────────────────────────────────────────────────────
    for epoch in range(1, max_epochs + 1):
        logger.info("── Fine-tune эпоха %d/%d ──", epoch, max_epochs)

        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            device, cfg, epoch, tlog, global_step,
        )

        # Val: только AF/VT AUC (основная метрика fine-tune)
        probs_val, targets_val = collect_predictions(
            model, val_loader, device, n_classes
        )
        # Метрики только по AF/VT
        af_vt_metrics = compute_metrics(
            probs   = probs_val[:, _FINETUNE_CLASS_INDICES],
            targets = targets_val[:, _FINETUNE_CLASS_INDICES],
            n_classes = len(_FINETUNE_CLASS_INDICES),
            class_names = ["AF", "VT"],
        )
        val_auc = af_vt_metrics["macro_auc"]

        logger.info(
            "Эпоха %d  finetune_loss=%.4f  val_AF_VT_AUC=%.4f  val_F1=%.4f",
            epoch, train_loss, val_auc, af_vt_metrics["macro_f1"],
        )
        tlog.log(
            {
                "finetune/val_af_vt_auc": val_auc,
                "finetune/val_af_vt_f1":  af_vt_metrics["macro_f1"],
                "epoch": epoch,
            },
            step=global_step,
        )

        is_best = early_stop.improved
        early_stop(val_auc)

        ckpt_mgr.save(
            model, optimizer, epoch=epoch, metric=val_auc,
            config=dict(cfg), is_best=is_best,
        )

        if is_best:
            best_metrics = {
                "epoch": epoch,
                "val_af_vt": af_vt_metrics,
            }

        if early_stop.should_stop:
            logger.info("Early stopping на эпохе %d", epoch)
            break

    # ── Финальная оценка на PTB-XL fold 10 ───────────────────────────────────
    logger.info("Загрузка лучшего чекпоинта для теста на fold 10…")
    ckpt_mgr.load_best(model, device=device)

    probs_test, targets_test = collect_predictions(
        model, test_loader, device, n_classes
    )

    # Полные метрики на fold 10
    full_test_metrics = compute_metrics(
        probs     = probs_test,
        targets   = targets_test,
        n_classes = n_classes,
    )
    # AF/VT delta по сравнению с шагом 1
    af_vt_test_metrics = compute_metrics(
        probs     = probs_test[:, _FINETUNE_CLASS_INDICES],
        targets   = targets_test[:, _FINETUNE_CLASS_INDICES],
        n_classes = len(_FINETUNE_CLASS_INDICES),
        class_names = ["AF", "VT"],
    )

    results_dir = Path(cfg.paths.results_dir) / "finetune"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(full_test_metrics, results_dir / "metrics_fold10_full.json")
    save_metrics(af_vt_test_metrics, results_dir / "metrics_fold10_af_vt.json")
    save_metrics(best_metrics, results_dir / "metrics_val_best.json")

    if cfg.logging.save_preds:
        np.save(str(results_dir / "fold10_preds.npy"),   probs_test)
        np.save(str(results_dir / "fold10_targets.npy"), targets_test)

    logger.info(
        "Fine-tune fold10: macro_AUC=%.4f  AF_AUC=%.4f  VT_AUC=%.4f",
        full_test_metrics["macro_auc"],
        af_vt_test_metrics["per_class_auc"][0],
        af_vt_test_metrics["per_class_auc"][1],
    )

    tlog.finish()
    return {"full": full_test_metrics, "af_vt": af_vt_test_metrics}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Шаг 2: fine-tune backbone на CPSC (AF/VT)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--pretrain-ckpt", type=Path, default=None,
        help="Путь к чекпоинту шага 1 (default: из конфига)",
    )
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_root_logger(str(cfg.logging.level))

    if args.debug:
        cfg["finetune"]["epochs"]  = 2
        cfg["finetune"]["patience"] = 2
        cfg["logging"]["backend"]  = "none"
        logger.info("DEBUG режим: epochs=2")

    run_finetune(cfg, pretrain_ckpt=args.pretrain_ckpt, limit=args.limit)


if __name__ == "__main__":
    main()