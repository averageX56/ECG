"""
training/_common.py
Общие утилиты для всех скриптов обучения.

Импорт:
  from training._common import (
      load_config, setup_seed, setup_logger, get_device,
      compute_pos_weight, compute_metrics, save_metrics,
      EarlyStopping, CheckpointManager, worker_init_fn,
  )
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Конфиг
# ─────────────────────────────────────────────────────────────────────────────

class _DotDict(dict):
    """dict с доступом через точку (рекурсивный)."""
    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(key)
        return _DotDict(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: Union[str, Path]) -> _DotDict:
    """
    Загружает YAML-конфиг, возвращает рекурсивный dot-dict.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    _DotDict
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = _DotDict(raw)
    logger.debug("Конфиг загружен из %s", path)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────────────────────────────────────

def setup_seed(seed: int = 42) -> None:
    """Устанавливает seed для random, numpy, torch и CUDA."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Детерминизм cudnn (замедляет, но нужен для воспроизводимости)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Seed=%d установлен", seed)


def worker_init_fn(worker_id: int) -> None:
    """
    Инициализирует seed в DataLoader-воркерах.
    Передаётся как worker_init_fn в DataLoader.
    """
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Устройство
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Возвращает CUDA если доступна, иначе CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Устройство: %s (%s)", dev, torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("Устройство: %s (Apple MPS)", dev)
    else:
        dev = torch.device("cpu")
        logger.info("Устройство: CPU")
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# Логирование: wandb / tensorboard / none
# ─────────────────────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Тонкая обёртка над wandb / tensorboard / none.

    Использование:
      tlog = setup_logger(cfg, run_name="pretrain")
      tlog.log({"loss": 0.3, "auc": 0.85}, step=100)
      tlog.finish()
    """

    def __init__(
        self,
        backend: str,
        run_name: str = "",
        config: Optional[dict] = None,
        log_dir: Optional[str] = None,
        project: str = "ecg-multidataset",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.backend = backend.lower()
        self._writer = None

        if self.backend == "wandb":
            self._init_wandb(run_name, config, project, entity, tags)
        elif self.backend == "tensorboard":
            self._init_tb(log_dir or "runs", run_name)
        elif self.backend != "none":
            logger.warning("Неизвестный бэкенд логирования '%s' → none", backend)
            self.backend = "none"

    def _init_wandb(
        self,
        run_name: str,
        config: Optional[dict],
        project: str,
        entity: Optional[str],
        tags: Optional[List[str]],
    ) -> None:
        try:
            import wandb
            self._writer = wandb.init(
                project=project,
                entity=entity,
                name=run_name or None,
                config=config,
                tags=tags or [],
                resume="allow",
            )
            logger.info("wandb запущен: %s/%s", project, run_name)
        except Exception as exc:
            logger.warning("wandb недоступен: %s → переключаюсь на none", exc)
            self.backend = "none"

    def _init_tb(self, log_dir: str, run_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = str(Path(log_dir) / run_name)
            self._writer = SummaryWriter(log_dir=tb_dir)
            logger.info("TensorBoard: %s", tb_dir)
        except Exception as exc:
            logger.warning("TensorBoard недоступен: %s → none", exc)
            self.backend = "none"

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Логирует скалярные метрики на шаге step."""
        if self.backend == "wandb" and self._writer is not None:
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "tensorboard" and self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, global_step=step)
        # none: ничего не делаем

    def watch(self, model: nn.Module) -> None:
        """wandb.watch(model) — для визуализации градиентов."""
        if self.backend == "wandb" and self._writer is not None:
            import wandb
            wandb.watch(model, log="gradients", log_freq=100)

    def finish(self) -> None:
        if self.backend == "wandb" and self._writer is not None:
            import wandb
            wandb.finish()
        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()


def setup_logger(
    cfg: _DotDict,
    run_name: str = "",
) -> TrainingLogger:
    """
    Создаёт TrainingLogger на основе конфига.

    Parameters
    ----------
    cfg : _DotDict
        Полный конфиг (используются cfg.logging.*)
    run_name : str
    """
    log_cfg = cfg.logging
    return TrainingLogger(
        backend  = log_cfg.backend,
        run_name = run_name,
        config   = dict(cfg),
        log_dir  = str(Path(cfg.paths.results_dir) / "tb_runs"),
        project  = log_cfg.wandb_project,
        entity   = log_cfg.wandb_entity,
        tags     = list(log_cfg.wandb_tags) if log_cfg.wandb_tags else [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# pos_weight для BCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_weight(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device,
    eps: float = 1.0,
    cap: float = 100.0,
) -> torch.Tensor:
    """
    Вычисляет pos_weight для BCEWithLogitsLoss из обучающих меток.

    pos_weight[i] = N_neg[i] / max(N_pos[i], eps)

    Parameters
    ----------
    labels : np.ndarray  [N, n_classes]
        Multi-hot матрица меток обучающего сплита.
    n_classes : int
    device : torch.device
    eps : float
        Защита от деления на ноль (мин. число позитивных примеров).
    cap : float
        Максимальный вес для предотвращения взрыва градиентов.

    Returns
    -------
    torch.Tensor  [n_classes]
    """
    n = len(labels)
    pos_counts = labels.sum(axis=0)          # [n_classes]
    neg_counts = n - pos_counts

    weights = neg_counts / np.maximum(pos_counts, eps)
    weights = np.clip(weights, 1.0, cap)

    logger.info(
        "pos_weight: min=%.2f max=%.2f mean=%.2f",
        weights.min(), weights.max(), weights.mean(),
    )
    return torch.from_numpy(weights.astype(np.float32)).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Метрики
# ─────────────────────────────────────────────────────────────────────────────

def _fmax_score(
    targets: np.ndarray,
    probs: np.ndarray,
    n_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Fmax — максимальный macro-F1 по всем порогам.

    Parameters
    ----------
    targets : [N, C]  multi-hot
    probs   : [N, C]  sigmoid-вероятности

    Returns
    -------
    fmax : float
    best_threshold : float
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds + 1)
    best_f1 = 0.0
    best_thr = 0.5

    for thr in thresholds:
        preds = (probs >= thr).astype(np.int32)
        # per-class precision / recall
        tp = (preds * targets).sum(axis=0)
        fp = (preds * (1 - targets)).sum(axis=0)
        fn = ((1 - preds) * targets).sum(axis=0)

        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

        macro_f1 = f1.mean()
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thr = float(thr)

    return float(best_f1), best_thr


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Вычисляет полный набор метрик для multi-label классификации.

    Parameters
    ----------
    probs    : np.ndarray  [N, n_classes]  — sigmoid-выходы
    targets  : np.ndarray  [N, n_classes]  — multi-hot метки
    n_classes : int
    threshold : float
    class_names : list[str] | None

    Returns
    -------
    dict с ключами:
      macro_auc, per_class_auc, macro_f1, per_class_f1,
      fmax, fmax_threshold, n_samples
    """
    from sklearn.metrics import roc_auc_score, f1_score

    assert probs.shape == targets.shape, (
        f"Форма probs={probs.shape} ≠ targets={targets.shape}"
    )

    n = probs.shape[0]
    preds = (probs >= threshold).astype(np.int32)

    # ── AUC ──────────────────────────────────────────────────────────────────
    per_auc: List[float] = []
    for i in range(n_classes):
        pos = targets[:, i].sum()
        if pos == 0 or pos == n:
            per_auc.append(float("nan"))
            continue
        try:
            auc_i = float(roc_auc_score(targets[:, i], probs[:, i]))
        except Exception:
            auc_i = float("nan")
        per_auc.append(auc_i)

    valid_auc = [v for v in per_auc if not np.isnan(v)]
    macro_auc = float(np.mean(valid_auc)) if valid_auc else 0.0

    # ── F1 ───────────────────────────────────────────────────────────────────
    per_f1: List[float] = []
    for i in range(n_classes):
        if targets[:, i].sum() == 0:
            per_f1.append(float("nan"))
            continue
        try:
            f1_i = float(f1_score(targets[:, i], preds[:, i], zero_division=0))
        except Exception:
            f1_i = float("nan")
        per_f1.append(f1_i)

    valid_f1 = [v for v in per_f1 if not np.isnan(v)]
    macro_f1 = float(np.mean(valid_f1)) if valid_f1 else 0.0

    # ── Fmax ─────────────────────────────────────────────────────────────────
    fmax, fmax_thr = _fmax_score(targets, probs)

    result: Dict[str, Any] = {
        "macro_auc":       macro_auc,
        "macro_f1":        macro_f1,
        "fmax":            fmax,
        "fmax_threshold":  fmax_thr,
        "n_samples":       n,
        "per_class_auc":   per_auc,
        "per_class_f1":    per_f1,
    }

    if class_names:
        result["class_names"] = class_names

    return result


def save_metrics(metrics: Dict[str, Any], path: Union[str, Path]) -> None:
    """Сохраняет metrics в JSON (non-serializable → str)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(v: Any) -> Any:
        if isinstance(v, float):
            return round(v, 6) if not np.isnan(v) else None
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list):
            return [_serialize(x) for x in v]
        if isinstance(v, dict):
            return {k: _serialize(vv) for k, vv in v.items()}
        return v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(metrics), f, indent=2, ensure_ascii=False)

    logger.info("Метрики сохранены: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Early stopping по метрике валидации (максимизация).

    Parameters
    ----------
    patience : int
        Число эпох без улучшения до остановки.
    min_delta : float
        Минимальное улучшение для сброса счётчика.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best     = -float("inf")
        self._counter  = 0
        self.should_stop = False
        self.improved    = False

    def __call__(self, val_metric: float) -> bool:
        """
        Обновляет состояние.

        Returns True если тренировку нужно остановить.
        """
        if val_metric > self._best + self.min_delta:
            self._best   = val_metric
            self._counter = 0
            self.improved = True
        else:
            self._counter += 1
            self.improved  = False
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: %d эпох без улучшения → стоп", self.patience
                )

        logger.debug(
            "EarlyStopping: val=%.5f best=%.5f counter=%d/%d",
            val_metric, self._best, self._counter, self.patience,
        )
        return self.should_stop

    @property
    def best(self) -> float:
        return self._best


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    epoch:        int
    best_metric:  float
    model_state:  Dict[str, Any]
    optim_state:  Dict[str, Any]
    config:       Dict[str, Any] = field(default_factory=dict)
    extra:        Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Сохраняет лучший чекпоинт и опционально последний.

    Parameters
    ----------
    checkpoint_dir : str | Path
    run_name : str
        Префикс файла.
    save_last : bool
        Сохранять ли также последний чекпоинт (для продолжения).
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        run_name: str = "model",
        save_last: bool = True,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.run_name  = run_name
        self.save_last = save_last
        self._best_path = self._dir / f"{run_name}_best.pt"
        self._last_path = self._dir / f"{run_name}_last.pt"

    def save(
        self,
        model:       nn.Module,
        optimizer:   torch.optim.Optimizer,
        epoch:       int,
        metric:      float,
        config:      Optional[dict] = None,
        extra:       Optional[dict] = None,
        is_best:     bool = False,
    ) -> None:
        """Сохраняет чекпоинт."""
        state = {
            "epoch":       epoch,
            "best_metric": metric,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "config":      config or {},
            "extra":       extra or {},
        }
        if self.save_last:
            torch.save(state, str(self._last_path))

        if is_best:
            torch.save(state, str(self._best_path))
            logger.info(
                "Чекпоинт сохранён [best]: эпоха=%d метрика=%.5f → %s",
                epoch, metric, self._best_path,
            )

    def load_best(
        self,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Загружает лучший чекпоинт в модель (и опционально оптимизатор)."""
        return self._load(self._best_path, model, optimizer, device)

    def load_last(
        self,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device:    Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        return self._load(self._last_path, model, optimizer, device)

    def _load(
        self,
        path:      Path,
        model:     nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device:    Optional[torch.device],
    ) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")

        map_loc = str(device) if device else "cpu"
        state = torch.load(str(path), map_location=map_loc, weights_only=True)
        model.load_state_dict(state["model_state"])
        if optimizer is not None and "optim_state" in state:
            optimizer.load_state_dict(state["optim_state"])

        logger.info(
            "Загружен чекпоинт: %s (эпоха=%d metric=%.5f)",
            path, state.get("epoch", -1), state.get("best_metric", float("nan")),
        )
        return state

    @property
    def best_path(self) -> Path:
        return self._best_path


# ─────────────────────────────────────────────────────────────────────────────
# Таймер шага (вывод в логи)
# ─────────────────────────────────────────────────────────────────────────────

class StepTimer:
    """Простой накопитель среднего времени шага."""

    def __init__(self, window: int = 50) -> None:
        self._times: List[float] = []
        self._window = window
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        if len(self._times) > self._window:
            self._times.pop(0)
        self._t0 = None
        return elapsed

    @property
    def avg(self) -> float:
        return float(np.mean(self._times)) if self._times else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    dataset: "torch.utils.data.Dataset",
    batch_size: int,
    sampler=None,
    shuffle: bool = False,
    n_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: int = 42,
) -> "torch.utils.data.DataLoader":
    """
    Строит DataLoader с фиксированным seed в воркерах.

    Parameters
    ----------
    dataset : Dataset
    batch_size : int
    sampler : Sampler | None
        Если задан, shuffle должен быть False.
    shuffle : bool
    n_workers : int
    pin_memory : bool
    prefetch_factor : int
    seed : int
    """
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)

    loader_kwargs: Dict[str, Any] = dict(
        batch_size       = batch_size,
        num_workers      = n_workers,
        pin_memory       = pin_memory and torch.cuda.is_available(),
        worker_init_fn   = worker_init_fn,
        generator        = g,
        persistent_workers = (n_workers > 0),
    )

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle

    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции для сбора предсказаний
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: "torch.utils.data.DataLoader",
    device: torch.device,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Прогоняет весь loader через model, собирает probs и targets.

    Parameters
    ----------
    model : nn.Module  (в режиме eval)
    loader : DataLoader  — возвращает (signal, label)
    device : torch.device
    n_classes : int

    Returns
    -------
    probs   : np.ndarray  [N, n_classes]
    targets : np.ndarray  [N, n_classes]
    """
    model.eval()
    probs_list:   List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    for batch in loader:
        signals, labels = batch[0].to(device), batch[1]
        logits = model(signals)
        prob   = torch.sigmoid(logits).cpu().numpy()
        probs_list.append(prob)
        targets_list.append(labels.numpy())

    if not probs_list:
        return (
            np.empty((0, n_classes), dtype=np.float32),
            np.empty((0, n_classes), dtype=np.float32),
        )

    return np.concatenate(probs_list, axis=0), np.concatenate(targets_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Настройка python-logging
# ─────────────────────────────────────────────────────────────────────────────

def configure_root_logger(level: str = "INFO") -> None:
    """Настраивает корневой python-логгер с отметкой времени."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    logging.basicConfig(level=lvl, format=fmt, datefmt="%H:%M:%S")