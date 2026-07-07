"""
training/train_beat_clf.py
Поток [B]: CNN-классификатор битов на MIT-BIH Arrhythmia Database.

Архитектура:
  Input: [B, 2, 300]
  Conv1d(2→32, k=5) → BN → ReLU
  Conv1d(32→64, k=5) → BN → ReLU
  Conv1d(64→128, k=5) → BN → ReLU
  AdaptiveAvgPool1d(1) → [B, 128]
  Dropout(0.3)
  Linear(128, 5)

Стратегия: 5-кратная кросс-валидация по пациентам (DS1, 22 пациента).
Тест: DS2 (22 пациента, неизменный).
Метрики: per-class F1, macro-F1 (AAMI классы N/S/V/F/Q).

Запуск:
  python -m training.train_beat_clf --config configs/default.yaml
  python -m training.train_beat_clf --config configs/default.yaml --debug
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training._common import (
    CheckpointManager,
    EarlyStopping,
    StepTimer,
    configure_root_logger,
    get_device,
    load_config,
    save_metrics,
    setup_logger,
    setup_seed,
    worker_init_fn,
)

logger = logging.getLogger(__name__)

AAMI_NAMES = ["N", "S", "V", "F", "Q"]


# ─────────────────────────────────────────────────────────────────────────────
# Архитектура Beat CNN
# ─────────────────────────────────────────────────────────────────────────────

class BeatCNN(nn.Module):
    """
    Лёгкий 1D CNN для классификации ЭКГ-битов.

    Input : [B, n_leads, beat_len]  — [B, 2, 300]
    Output: [B, n_classes]          — логиты

    Parameters
    ----------
    n_leads : int
        Число входных каналов (2 для MIT-BIH).
    conv_channels : list[int]
        Число фильтров в каждом conv-блоке.
    kernel_size : int
        Размер ядра свёртки.
    n_classes : int
        Число AAMI классов (5: N/S/V/F/Q).
    dropout : float
    """

    def __init__(
        self,
        n_leads:       int       = 2,
        conv_channels: List[int] = None,
        kernel_size:   int       = 5,
        n_classes:     int       = 5,
        dropout:       float     = 0.3,
    ) -> None:
        super().__init__()
        conv_channels = conv_channels or [32, 64, 128]

        blocks: List[nn.Module] = []
        in_ch = n_leads
        for out_ch in conv_channels:
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        self.conv_net = nn.Sequential(*blocks)
        self.gap      = nn.AdaptiveAvgPool1d(1)
        self.dropout  = nn.Dropout(p=dropout)
        self.head     = nn.Linear(conv_channels[-1], n_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 2, 300]

        Returns
        -------
        logits : [B, n_classes]
        """
        out = self.conv_net(x)          # [B, 128, 300]
        out = self.gap(out).squeeze(-1) # [B, 128]
        out = self.dropout(out)
        return self.head(out)           # [B, 5]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset для битов
# ─────────────────────────────────────────────────────────────────────────────

class BeatDataset(Dataset):
    """
    Dataset для массивов битов [N, 2, 300] и меток [N].

    Parameters
    ----------
    X : np.ndarray  [N, 2, 300]
    y : np.ndarray  [N]  int64 AAMI классы
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        assert len(X) == len(y)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def class_weights(self) -> torch.Tensor:
        """
        Веса классов для WeightedRandomSampler (обратно пропорционально частоте).
        """
        counts = np.bincount(self.y.numpy(), minlength=5).astype(np.float64)
        # Заменяем нули единицами (класс отсутствует — вес = 0 → OK)
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        sample_weights = weights[self.y.numpy()]
        return torch.from_numpy(sample_weights.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка данных MIT-BIH
# ─────────────────────────────────────────────────────────────────────────────

def _load_split_arrays(
    root: str,
    record_ids: Set[str],
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает MIT-BIH биты для указанных записей.
    """
    from data.load_mitbih import load_mitbih_arrays

    X, y = load_mitbih_arrays(
        root          = root,
        record_ids    = list(record_ids),
        show_progress = True,
        znorm_beats   = True,
    )

    if limit and len(X) > limit:
        idx = np.random.default_rng(42).choice(len(X), limit, replace=False)
        X, y = X[idx], y[idx]

    logger.info(
        "Загружено %d битов (records=%d). Классы: %s",
        len(X), len(record_ids),
        {AAMI_NAMES[i]: int((y == i).sum()) for i in range(5)},
    )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Метрики для beat classifier
# ─────────────────────────────────────────────────────────────────────────────

def compute_beat_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = AAMI_NAMES,
) -> Dict:
    """
    Метрики для multi-class (не multi-label) задачи битов.

    Parameters
    ----------
    preds   : [N]  предсказанные классы (argmax)
    targets : [N]  истинные классы

    Returns
    -------
    dict с macro_f1, per_class_f1, accuracy
    """
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    macro_f1  = float(f1_score(targets, preds, average="macro", zero_division=0))
    per_f1    = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(5)))
    accuracy  = float(accuracy_score(targets, preds))
    report    = classification_report(
        targets, preds, target_names=class_names, zero_division=0
    )

    result = {
        "macro_f1":     macro_f1,
        "accuracy":     accuracy,
        "per_class_f1": {name: float(per_f1[i]) for i, name in enumerate(class_names)},
        "classification_report": report,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Одна эпоха обучения
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:       BeatCNN,
    loader:      DataLoader,
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    cfg,
    epoch:       int,
    tlog,
    global_step: int,
    fold:        int,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    timer      = StepTimer()
    log_every  = int(cfg.logging.log_every_n_steps)

    for beats, labels in loader:
        timer.start()
        beats  = beats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(beats)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        timer.stop()

        total_loss  += loss.item()
        n_batches   += 1
        global_step += 1

        if global_step % log_every == 0:
            tlog.log(
                {f"beat/fold{fold}_loss": total_loss / n_batches},
                step=global_step,
            )

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def eval_epoch(
    model:    BeatCNN,
    loader:   DataLoader,
    device:   torch.device,
    fold:     int,
) -> Tuple[float, Dict]:
    """Прогоняет loader, возвращает loss и метрики."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds:   List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    n_batches  = 0

    for beats, labels in loader:
        beats  = beats.to(device)
        labels = labels.to(device)

        logits = model(beats)
        loss   = criterion(logits, labels)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

        total_loss += loss.item()
        n_batches  += 1

    preds_arr   = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    metrics     = compute_beat_metrics(preds_arr, targets_arr)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Один fold обучения
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold:        int,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    cfg,
    device:      torch.device,
    seed:        int,
    tlog,
    ckpt_dir:    Path,
) -> Dict:
    """
    Обучает BeatCNN на одном fold, возвращает метрики val.

    Returns
    -------
    dict  — финальные метрики val для этого fold
    """
    logger.info(
        "Fold %d: train=%d val=%d",
        fold, len(X_train), len(X_val),
    )

    train_ds = BeatDataset(X_train, y_train)
    val_ds   = BeatDataset(X_val,   y_val)

    # Взвешенный семплинг для борьбы с дисбалансом классов
    sample_w  = train_ds.class_weights()
    sampler   = WeightedRandomSampler(sample_w, len(train_ds), replacement=True)

    g = torch.Generator()
    g.manual_seed(seed + fold)

    train_loader = DataLoader(
        train_ds,
        batch_size       = int(cfg.beat_clf.batch_size),
        sampler          = sampler,
        num_workers      = int(cfg.data.n_workers),
        worker_init_fn   = worker_init_fn,
        generator        = g,
        pin_memory        = torch.cuda.is_available(),
        persistent_workers = (int(cfg.data.n_workers) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = int(cfg.beat_clf.batch_size) * 2,
        shuffle          = False,
        num_workers      = int(cfg.data.n_workers),
        worker_init_fn   = worker_init_fn,
        pin_memory        = torch.cuda.is_available(),
        persistent_workers = (int(cfg.data.n_workers) > 0),
    )

    # Модель
    model = BeatCNN(
        n_leads       = 2,
        conv_channels = list(cfg.beat_clf.conv_channels),
        kernel_size   = int(cfg.beat_clf.kernel_size),
        n_classes     = int(cfg.beat_clf.n_classes),
        dropout       = float(cfg.beat_clf.dropout),
    ).to(device)

    logger.info("BeatCNN: %d параметров", model.param_count())

    # Класс-веса для CrossEntropyLoss
    class_counts = np.bincount(y_train, minlength=5).astype(np.float64)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weight = torch.from_numpy(
        (1.0 / class_counts).astype(np.float32)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(cfg.beat_clf.lr),
        weight_decay = float(cfg.beat_clf.weight_decay),
    )
    # OneCycleLR для быстрой сходимости
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = float(cfg.beat_clf.lr),
        steps_per_epoch = len(train_loader),
        epochs          = int(cfg.beat_clf.epochs),
        pct_start       = 0.3,
    )

    early_stop = EarlyStopping(
        patience  = int(cfg.beat_clf.patience),
        min_delta = 1e-4,
    )
    ckpt_mgr = CheckpointManager(
        ckpt_dir / f"fold_{fold}", run_name="beat_clf"
    )

    max_epochs  = int(cfg.beat_clf.epochs)
    global_step = 0
    best_metrics: Dict = {}

    for epoch in range(1, max_epochs + 1):
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer,
            device, cfg, epoch, tlog, global_step, fold,
        )
        scheduler.step()

        val_loss, val_metrics = eval_epoch(model, val_loader, device, fold)

        val_f1 = val_metrics["macro_f1"]
        logger.info(
            "Fold %d эпоха %d  train_loss=%.4f  val_loss=%.4f  val_macro_f1=%.4f",
            fold, epoch, train_loss, val_loss, val_f1,
        )
        tlog.log(
            {
                f"beat/fold{fold}_val_macro_f1": val_f1,
                f"beat/fold{fold}_val_loss":     val_loss,
            },
            step=global_step,
        )

        is_best = early_stop.improved
        early_stop(val_f1)

        ckpt_mgr.save(
            model, optimizer, epoch=epoch, metric=val_f1,
            config=dict(cfg), is_best=is_best,
        )

        if is_best:
            best_metrics = val_metrics.copy()
            best_metrics["epoch"] = epoch

        if early_stop.should_stop:
            logger.info("Fold %d early stop на эпохе %d", fold, epoch)
            break

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_beat_clf(
    cfg,
    limit: Optional[int] = None,
) -> Dict:
    """
    5-fold CV на DS1 + финальный тест на DS2.

    Returns
    -------
    dict  с 'cv_results', 'test_metrics', 'cv_mean_f1'
    """
    from data.load_mitbih import DS1_RECORDS, DS2_RECORDS, get_kfold_splits

    seed = int(cfg.seed)
    setup_seed(seed)
    device = get_device()
    tlog   = setup_logger(cfg, run_name="beat_clf")

    mitbih_root = str(cfg.paths.mitbih_root)
    ckpt_dir    = Path(cfg.paths.checkpoint_dir) / "beat_clf"

    # ── DS2: тестовый сет (загружаем один раз) ────────────────────────────────
    logger.info("Загрузка DS2 (test)…")
    X_test, y_test = _load_split_arrays(mitbih_root, DS2_RECORDS, limit=limit)

    # ── DS1: кросс-валидация ──────────────────────────────────────────────────
    logger.info("Загрузка DS1 (train/val fold splits)…")
    X_ds1, y_ds1, record_ids_ds1 = _load_ds1_with_ids(
        mitbih_root, DS1_RECORDS, limit=limit
    )

    kfold_splits = get_kfold_splits(k=int(cfg.beat_clf.k_folds), seed=seed)

    cv_results: List[Dict] = []
    all_val_f1s: List[float] = []

    for fold_idx, (train_ids, val_ids) in enumerate(kfold_splits, start=1):
        # Маска по record_id
        train_mask = np.isin(record_ids_ds1, list(train_ids))
        val_mask   = np.isin(record_ids_ds1, list(val_ids))

        X_train, y_train = X_ds1[train_mask], y_ds1[train_mask]
        X_val,   y_val   = X_ds1[val_mask],   y_ds1[val_mask]

        fold_metrics = train_fold(
            fold      = fold_idx,
            X_train   = X_train,
            y_train   = y_train,
            X_val     = X_val,
            y_val     = y_val,
            cfg       = cfg,
            device    = device,
            seed      = seed,
            tlog      = tlog,
            ckpt_dir  = ckpt_dir,
        )
        cv_results.append(fold_metrics)
        all_val_f1s.append(fold_metrics.get("macro_f1", 0.0))
        logger.info("Fold %d: macro_F1=%.4f", fold_idx, fold_metrics.get("macro_f1", 0))

    cv_mean_f1 = float(np.mean(all_val_f1s))
    cv_std_f1  = float(np.std(all_val_f1s))
    logger.info("CV macro-F1: %.4f ± %.4f", cv_mean_f1, cv_std_f1)

    # ── Финальная модель: train на всём DS1, eval на DS2 ─────────────────────
    logger.info("Обучение финальной модели на всём DS1 → тест на DS2…")

    final_metrics = _train_final_and_test(
        X_train=X_ds1, y_train=y_ds1,
        X_test=X_test, y_test=y_test,
        cfg=cfg, device=device, seed=seed, tlog=tlog,
        ckpt_dir=ckpt_dir,
    )

    # ── Сохранение результатов ────────────────────────────────────────────────
    results = {
        "cv_results":   cv_results,
        "cv_mean_f1":   cv_mean_f1,
        "cv_std_f1":    cv_std_f1,
        "test_metrics": final_metrics,
    }
    results_dir = Path(cfg.paths.results_dir) / "beat_clf"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(results, results_dir / "metrics.json")

    if cfg.logging.save_preds:
        # Сохраняем предсказания финальной модели на DS2
        _save_test_preds(
            X_test, y_test, cfg, device, ckpt_dir, results_dir
        )

    tlog.finish()

    logger.info(
        "Beat CLF завершён. DS2 macro-F1=%.4f  accuracy=%.4f",
        final_metrics.get("macro_f1", 0.0),
        final_metrics.get("accuracy", 0.0),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _load_ds1_with_ids(
    root: str,
    record_ids: Set[str],
    limit: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает биты DS1 вместе с record_id для fold-сплита.

    Returns
    -------
    X : [N, 2, 300]
    y : [N]
    record_ids_arr : [N] str
    """
    from data.load_mitbih import iter_mitbih_beats

    beats_list: List[np.ndarray] = []
    classes_list: List[int]      = []
    rec_ids_list: List[str]      = []
    count = 0

    for rec_b in iter_mitbih_beats(
        root        = root,
        record_ids  = list(record_ids),
        znorm       = True,
        show_progress = True,
    ):
        beats_list.append(rec_b.beat)
        classes_list.append(rec_b.beat_class)
        rec_ids_list.append(rec_b.record_id)
        count += 1
        if limit and count >= limit:
            break

    if not beats_list:
        return (
            np.empty((0, 2, 300), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=object),
        )

    X   = np.stack(beats_list)
    y   = np.array(classes_list, dtype=np.int64)
    ids = np.array(rec_ids_list, dtype=object)
    return X, y, ids


def _train_final_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    cfg,
    device:  torch.device,
    seed:    int,
    tlog,
    ckpt_dir: Path,
) -> Dict:
    """Обучает финальную модель на DS1 и тестирует на DS2."""
    fold_metrics = train_fold(
        fold     = 0,     # fold=0 означает "финальный"
        X_train  = X_train,
        y_train  = y_train,
        X_val    = X_test,   # используем DS2 как "val" для early stopping
        y_val    = y_test,
        cfg      = cfg,
        device   = device,
        seed     = seed,
        tlog     = tlog,
        ckpt_dir = ckpt_dir,
    )
    return fold_metrics


def _save_test_preds(
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg,
    device: torch.device,
    ckpt_dir: Path,
    results_dir: Path,
) -> None:
    """Загружает финальную модель и сохраняет предсказания на DS2."""
    ckpt_path = ckpt_dir / "fold_0" / "beat_clf_best.pt"
    if not ckpt_path.exists():
        logger.warning("Финальный чекпоинт не найден: %s", ckpt_path)
        return

    model = BeatCNN(
        n_leads       = 2,
        conv_channels = list(cfg.beat_clf.conv_channels),
        kernel_size   = int(cfg.beat_clf.kernel_size),
        n_classes     = int(cfg.beat_clf.n_classes),
        dropout       = float(cfg.beat_clf.dropout),
    ).to(device)

    state = torch.load(str(ckpt_path), map_location=str(device), weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()

    test_ds = BeatDataset(X_test, y_test)
    loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    all_preds:  List[np.ndarray] = []
    all_probs:  List[np.ndarray] = []

    with torch.no_grad():
        for beats, _ in loader:
            logits = model(beats.to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)

    np.save(str(results_dir / "test_preds.npy"),   np.concatenate(all_preds))
    np.save(str(results_dir / "test_probs.npy"),   np.concatenate(all_probs, axis=0))
    np.save(str(results_dir / "test_targets.npy"), y_test)
    logger.info("DS2 предсказания сохранены в %s", results_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Поток [B]: beat classifier на MIT-BIH"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
    )
    parser.add_argument("--limit",  type=int,   default=None,
                        help="Максимум битов (для debug)")
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_root_logger(str(cfg.logging.level))

    if args.debug:
        cfg["beat_clf"]["epochs"]  = 3
        cfg["beat_clf"]["patience"] = 2
        cfg["beat_clf"]["k_folds"] = 2
        cfg["logging"]["backend"]  = "none"
        logger.info("DEBUG режим: epochs=3, k_folds=2")

    limit = args.limit or (2000 if args.debug else None)
    run_beat_clf(cfg, limit=limit)


if __name__ == "__main__":
    main()