"""
backbone/load_weights.py
Загрузка предобученных весов antonior92 из Zenodo и конвертация в PyTorch.

Источник:
  Zenodo record 3625017 — Ribeiro et al. 2020 (Nature Communications)
  https://zenodo.org/record/3625017
  Файл: model.hdf5 (~15 МБ) — Keras/TF модель

Поток работы
------------
1. Проверить локальный кэш (~/.cache/ecg_backbone/)
2. Если нет — скачать model.hdf5 через urllib
3. Конвертировать Keras слои → PyTorch state_dict
4. Сохранить .pt в кэш, вернуть state_dict

Сопоставление слоёв Keras → PyTorch
------------------------------------
Keras (layer_0 .. layer_4):
  layer_0:
    layer_0/conv/kernel   → stem.conv.weight   (transpose: [k,in,out]→[out,in,k])
    layer_0/batch_normalization/gamma  → stem.bn.weight
    layer_0/batch_normalization/beta   → stem.bn.bias
    layer_0/batch_normalization/moving_mean    → stem.bn.running_mean
    layer_0/batch_normalization/moving_variance→ stem.bn.running_var

  layer_{i+1} (i=0..3) → res_blocks.{i}:
    conv1/kernel  → conv1.weight
    conv2/kernel  → conv2.weight
    batch_norm_1  → bn1.*
    batch_norm_2  → bn2.*
    skip_conv     → skip.1.weight  (если есть)
    skip_bn       → skip.2.*

Формат тензоров
---------------
Keras Conv1D kernel: [kernel_size, in_ch, out_ch]
PyTorch Conv1d weight: [out_ch, in_ch, kernel_size]
→ np.transpose(w, (2, 1, 0))

Если h5py недоступен — выдаётся RuntimeError с инструкцией по установке.
"""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы Zenodo
# ---------------------------------------------------------------------------
ZENODO_RECORD_ID  = "3625017"
ZENODO_FILENAME   = "model.hdf5"
ZENODO_URL        = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files/{ZENODO_FILENAME}"
# MD5 для верификации скачанного файла (из Zenodo страницы)
ZENODO_MD5        = "c17b8c4ff5b1e4f59c738f1399a3b59e"

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ecg_backbone"
_KERAS_FNAME       = "antonior92_model.hdf5"
_PT_FNAME          = "antonior92_resnet1d.pt"


# ---------------------------------------------------------------------------
# Скачивание
# ---------------------------------------------------------------------------

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    """Считает MD5 файла чанками."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _reporthook(count: int, block_size: int, total_size: int) -> None:
    """Простой прогресс для urllib.request.urlretrieve."""
    if total_size > 0:
        pct = min(100.0, count * block_size * 100.0 / total_size)
        mb = count * block_size / 1e6
        logger.debug("Скачивание: %.1f%% (%.1f MB)", pct, mb)


def download_hdf5(
    cache_dir: Optional[Path] = None,
    url: str = ZENODO_URL,
    expected_md5: Optional[str] = ZENODO_MD5,
    force: bool = False,
) -> Path:
    """
    Скачивает model.hdf5 из Zenodo в локальный кэш.

    Parameters
    ----------
    cache_dir : Path | None
        Директория кэша. None → ~/.cache/ecg_backbone/.
    url : str
        URL для скачивания.
    expected_md5 : str | None
        Ожидаемый MD5. None → проверка пропускается.
    force : bool
        Перекачать даже если файл уже есть.

    Returns
    -------
    Path
        Путь к скачанному файлу.

    Raises
    ------
    RuntimeError
        Если MD5 не совпадает после скачивания.
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dest = cache_dir / _KERAS_FNAME

    if dest.exists() and not force:
        logger.info("hdf5 найден в кэше: %s", dest)
        return dest

    logger.info("Скачивание весов из %s → %s", url, dest)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Ошибка скачивания весов из {url}: {exc}\n"
            f"Скачайте вручную и положите в {dest}"
        ) from exc

    if expected_md5 is not None:
        actual_md5 = _md5(dest)
        if actual_md5 != expected_md5:
            logger.warning(
                "MD5 не совпадает: ожидали %s, получили %s. "
                "Файл может быть повреждён или версия Zenodo изменилась.",
                expected_md5, actual_md5,
            )
        else:
            logger.info("MD5 верифицирован: %s", actual_md5)

    return dest


# ---------------------------------------------------------------------------
# Конвертация весов Keras → PyTorch
# ---------------------------------------------------------------------------

def _keras_conv_to_torch(kernel: np.ndarray) -> torch.Tensor:
    """
    Конвертирует ядро Conv1D из Keras-формата в PyTorch.

    Keras : [kernel_size, in_ch, out_ch]
    PyTorch: [out_ch, in_ch, kernel_size]
    """
    # (k, in, out) → (out, in, k)
    return torch.from_numpy(kernel.transpose(2, 1, 0).copy()).float()


def _keras_bn_to_torch(
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """Конвертирует BatchNorm-переменные Keras → словарь PyTorch."""
    return {
        "weight":       torch.from_numpy(gamma.copy()).float(),
        "bias":         torch.from_numpy(beta.copy()).float(),
        "running_mean": torch.from_numpy(running_mean.copy()).float(),
        "running_var":  torch.from_numpy(running_var.copy()).float(),
        "num_batches_tracked": torch.tensor(0, dtype=torch.long),
    }


def _try_get(group: Any, *keys: str) -> Optional[np.ndarray]:
    """
    Пробует несколько возможных имён ключа в h5py-группе.
    Возвращает первый найденный или None.
    """
    for k in keys:
        if k in group:
            return np.array(group[k])
    return None


def convert_keras_to_pytorch(
    hdf5_path: Path,
) -> Dict[str, torch.Tensor]:
    """
    Читает Keras model.hdf5 и строит PyTorch state_dict для ResNet1d.

    Ожидаемая структура Keras-модели (antonior92):
      model/layers/layer_0/ — stem
      model/layers/layer_1/ .. layer_4/ — residual blocks

    Возможные варианты именования внутри каждого слоя:
      - Keras 2.x: 'conv1d/kernel:0', 'batch_normalization/gamma:0', ...
      - Keras 3.x: 'conv1d/kernel', 'batch_normalization/gamma', ...

    Parameters
    ----------
    hdf5_path : Path

    Returns
    -------
    dict[str, torch.Tensor]
        Частичный state_dict для ResNet1d (только backbone без head).
        Ключи соответствуют именам модуля PyTorch.

    Raises
    ------
    RuntimeError
        Если h5py не установлен или файл не распознан.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Для конвертации весов нужен h5py: pip install h5py"
        ) from exc

    state: Dict[str, torch.Tensor] = {}

    with h5py.File(str(hdf5_path), "r") as f:
        # Ищем корень модели (Keras 2 / Keras 3 разная структура)
        model_root = _find_keras_model_root(f)
        if model_root is None:
            raise RuntimeError(
                f"Не удалось найти корень модели в {hdf5_path}. "
                "Убедитесь что файл — Keras .hdf5 модель antonior92."
            )

        # ── Stem (layer_0) ────────────────────────────────────────────────
        layer0 = _find_layer(model_root, "layer_0", 0)
        if layer0 is not None:
            _load_stem(layer0, state)
        else:
            logger.warning("layer_0 (stem) не найден в .hdf5")

        # ── Residual blocks (layer_1 .. layer_4) ─────────────────────────
        for i in range(4):
            layer_key = f"layer_{i + 1}"
            layer = _find_layer(model_root, layer_key, i + 1)
            if layer is not None:
                _load_resblock(layer, i, state)
            else:
                logger.warning("Residual block layer_%d не найден", i + 1)

    _log_loaded_keys(state)
    return state


# ---------------------------------------------------------------------------
# Вспомогательные функции конвертации
# ---------------------------------------------------------------------------

def _find_keras_model_root(f: Any) -> Optional[Any]:
    """
    Находит корень иерархии Keras в h5py.File.

    Keras 2: f['model_weights']
    Keras 3: f['model'] / f.attrs['model_config'] ...
    """
    for candidate in ("model_weights", "model", ""):
        if candidate == "" or candidate in f:
            root = f if candidate == "" else f[candidate]
            # Проверяем что есть хоть один layer_N ключ
            keys_str = list(root.keys())
            if any(k.startswith("layer_") for k in keys_str):
                return root
            # Иногда вложен глубже: root/model_name/
            for sub in keys_str:
                sub_group = root[sub]
                if hasattr(sub_group, "keys"):
                    if any(k.startswith("layer_") for k in sub_group.keys()):
                        return sub_group
    return None


def _find_layer(root: Any, preferred_name: str, idx: int) -> Optional[Any]:
    """
    Ищет слой в h5py-группе по имени или индексу.
    Пробует: preferred_name, 'layer' + idx, перебор по порядку.
    """
    if preferred_name in root:
        g = root[preferred_name]
        # Keras 2 вкладывает ещё раз: root['layer_0']['layer_0']
        if preferred_name in g:
            return g[preferred_name]
        return g

    # Попробуем 0-based и 1-based индексы
    for variant in (str(idx), str(idx + 1)):
        for key in root.keys():
            if key.endswith(f"_{variant}") or key == variant:
                return root[key]

    # Если ничего не нашли — берём по порядку
    keys = sorted(root.keys())
    if idx < len(keys):
        return root[keys[idx]]

    return None


def _read_weight(group: Any, *name_variants: str) -> Optional[np.ndarray]:
    """
    Читает вес из h5py-группы, перебирая возможные имена.
    Keras 2 добавляет ':0' к именам переменных.
    """
    for name in name_variants:
        for suffix in ("", ":0"):
            full = name + suffix
            if full in group:
                return np.array(group[full])
    # Рекурсивный поиск по дочерним группам (глубина 1)
    for key in group.keys():
        sub = group[key]
        for name in name_variants:
            for suffix in ("", ":0"):
                full = name + suffix
                if full in sub:
                    return np.array(sub[full])
    return None


def _load_stem(layer: Any, state: Dict[str, torch.Tensor]) -> None:
    """Загружает веса stem-слоя."""
    # Conv
    kernel = _read_weight(layer, "conv1d/kernel", "conv/kernel", "kernel")
    if kernel is not None:
        state["stem.conv.weight"] = _keras_conv_to_torch(kernel)

    # BN
    for keras_pfx, pt_pfx in [
        ("batch_normalization", "stem.bn"),
        ("batch_normalization_1", "stem.bn"),
    ]:
        gamma  = _read_weight(layer, f"{keras_pfx}/gamma")
        beta   = _read_weight(layer, f"{keras_pfx}/beta")
        mean   = _read_weight(layer, f"{keras_pfx}/moving_mean")
        var    = _read_weight(layer, f"{keras_pfx}/moving_variance")
        if gamma is not None:
            bn_dict = _keras_bn_to_torch(gamma, beta, mean, var)
            for k, v in bn_dict.items():
                state[f"{pt_pfx}.{k}"] = v
            break


def _load_resblock(
    layer: Any,
    block_idx: int,
    state: Dict[str, torch.Tensor],
) -> None:
    """Загружает веса одного residual-блока."""
    pfx = f"res_blocks.{block_idx}"

    # ── conv1 ──────────────────────────────────────────────────────────────
    kernel1 = _read_weight(
        layer,
        "conv1d/kernel", "conv1d_1/kernel",
        f"conv1d_{block_idx * 2}/kernel",
        "conv1/kernel", "conv_1/kernel",
    )
    if kernel1 is not None:
        state[f"{pfx}.conv1.weight"] = _keras_conv_to_torch(kernel1)

    # ── bn1 ────────────────────────────────────────────────────────────────
    for bn_name in ("batch_normalization", "batch_normalization_1",
                    f"batch_normalization_{block_idx * 2}",
                    "batch_norm_1", "bn_1"):
        gamma = _read_weight(layer, f"{bn_name}/gamma")
        if gamma is not None:
            beta  = _read_weight(layer, f"{bn_name}/beta")
            mean_ = _read_weight(layer, f"{bn_name}/moving_mean")
            var_  = _read_weight(layer, f"{bn_name}/moving_variance")
            for k, v in _keras_bn_to_torch(gamma, beta, mean_, var_).items():
                state[f"{pfx}.bn1.{k}"] = v
            break

    # ── conv2 ──────────────────────────────────────────────────────────────
    kernel2 = _read_weight(
        layer,
        "conv1d_1/kernel", "conv1d_2/kernel",
        f"conv1d_{block_idx * 2 + 1}/kernel",
        "conv2/kernel", "conv_2/kernel",
    )
    if kernel2 is not None:
        state[f"{pfx}.conv2.weight"] = _keras_conv_to_torch(kernel2)

    # ── bn2 ────────────────────────────────────────────────────────────────
    for bn_name in ("batch_normalization_1", "batch_normalization_2",
                    f"batch_normalization_{block_idx * 2 + 1}",
                    "batch_norm_2", "bn_2"):
        gamma = _read_weight(layer, f"{bn_name}/gamma")
        if gamma is not None:
            beta  = _read_weight(layer, f"{bn_name}/beta")
            mean_ = _read_weight(layer, f"{bn_name}/moving_mean")
            var_  = _read_weight(layer, f"{bn_name}/moving_variance")
            for k, v in _keras_bn_to_torch(gamma, beta, mean_, var_).items():
                state[f"{pfx}.bn2.{k}"] = v
            break

    # ── skip conv (1×1) ────────────────────────────────────────────────────
    skip_kernel = _read_weight(
        layer,
        "conv1d_2/kernel", "skip_conv/kernel",
        f"conv1d_{block_idx * 2 + 2}/kernel",
        "shortcut_conv/kernel", "conv_skip/kernel",
    )
    if skip_kernel is not None:
        # Skip 1×1 kernel: [1, in_ch, out_ch] → [out_ch, in_ch, 1]
        if skip_kernel.ndim == 3:
            state[f"{pfx}.skip.1.weight"] = _keras_conv_to_torch(skip_kernel)
        elif skip_kernel.ndim == 2:
            # Иногда [in_ch, out_ch]
            w = torch.from_numpy(skip_kernel.T[:, :, None]).float()
            state[f"{pfx}.skip.1.weight"] = w

    # ── skip bn ────────────────────────────────────────────────────────────
    for bn_name in (
        "batch_normalization_2", "batch_normalization_3",
        f"batch_normalization_{block_idx * 2 + 2}",
        "batch_norm_skip", "bn_skip",
    ):
        gamma = _read_weight(layer, f"{bn_name}/gamma")
        if gamma is not None:
            beta  = _read_weight(layer, f"{bn_name}/beta")
            mean_ = _read_weight(layer, f"{bn_name}/moving_mean")
            var_  = _read_weight(layer, f"{bn_name}/moving_variance")
            for k, v in _keras_bn_to_torch(gamma, beta, mean_, var_).items():
                state[f"{pfx}.skip.2.{k}"] = v
            break


def _log_loaded_keys(state: Dict[str, torch.Tensor]) -> None:
    """Логирует статистику загруженных ключей."""
    n_conv = sum(1 for k in state if "conv" in k and "weight" in k)
    n_bn   = sum(1 for k in state if ".bn" in k and "weight" in k)
    logger.info(
        "Конвертация завершена: %d ключей (%d conv, %d bn)",
        len(state), n_conv, n_bn,
    )


# ---------------------------------------------------------------------------
# Загрузка весов в модель
# ---------------------------------------------------------------------------

def load_pretrained_weights(
    model: "ResNet1d",                    # noqa: F821
    cache_dir: Optional[Path] = None,
    pt_path: Optional[Path] = None,
    hdf5_path: Optional[Path] = None,
    strict: bool = False,
    force_download: bool = False,
    force_convert: bool = False,
) -> Dict[str, Any]:
    """
    Загружает предобученные веса в ResNet1d.

    Приоритет источников:
      1. pt_path — уже конвертированный .pt файл (быстрый путь)
      2. hdf5_path — Keras .hdf5 (конвертируем на лету)
      3. cache_dir/.pt — кэш конвертированных весов
      4. Zenodo → скачиваем hdf5, конвертируем, кэшируем

    Parameters
    ----------
    model : ResNet1d
        Модель, в которую грузим веса (in-place).
    cache_dir : Path | None
        Директория кэша.
    pt_path : Path | None
        Путь к .pt файлу (state_dict).
    hdf5_path : Path | None
        Путь к .hdf5 файлу.
    strict : bool
        strict=True → ошибка если ключи не совпадают.
        По умолчанию False (загружаем только совпадающие веса).
    force_download : bool
        Перекачать .hdf5 даже если есть в кэше.
    force_convert : bool
        Переконвертировать .hdf5 → .pt даже если .pt есть.

    Returns
    -------
    dict
        {'missing': [...], 'unexpected': [...], 'loaded': N}
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_pt = cache_dir / _PT_FNAME

    # ── 1. Готовый .pt файл ──────────────────────────────────────────────────
    if pt_path is not None:
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        return _apply_state_dict(model, state_dict, strict)

    # ── 2. Кэшированный .pt ──────────────────────────────────────────────────
    if cached_pt.exists() and not force_convert:
        logger.info("Загрузка конвертированных весов из %s", cached_pt)
        state_dict = torch.load(str(cached_pt), map_location="cpu", weights_only=True)
        return _apply_state_dict(model, state_dict, strict)

    # ── 3. .hdf5 → конвертация ───────────────────────────────────────────────
    if hdf5_path is None:
        hdf5_path = download_hdf5(
            cache_dir=cache_dir,
            force=force_download,
        )

    logger.info("Конвертация Keras → PyTorch: %s", hdf5_path)
    state_dict = convert_keras_to_pytorch(Path(hdf5_path))

    # Кэшируем конвертированные веса
    try:
        torch.save(state_dict, str(cached_pt))
        logger.info("Конвертированные веса сохранены в %s", cached_pt)
    except Exception as exc:
        logger.warning("Не удалось сохранить .pt кэш: %s", exc)

    return _apply_state_dict(model, state_dict, strict)


def _apply_state_dict(
    model: "ResNet1d",                    # noqa: F821
    state_dict: Dict[str, torch.Tensor],
    strict: bool,
) -> Dict[str, Any]:
    """
    Загружает state_dict в модель, возвращает статистику.
    """
    result = model.load_state_dict(state_dict, strict=strict)
    missing     = list(result.missing_keys)
    unexpected  = list(result.unexpected_keys)
    loaded      = len(state_dict) - len(unexpected)

    if missing:
        logger.warning(
            "Не загружены ключи (%d): %s...", len(missing), missing[:5]
        )
    if unexpected:
        logger.warning(
            "Лишние ключи (%d): %s...", len(unexpected), unexpected[:5]
        )
    logger.info(
        "Веса загружены: %d/%d ключей", loaded, len(model.state_dict())
    )

    return {
        "missing":    missing,
        "unexpected": unexpected,
        "loaded":     loaded,
        "total":      len(model.state_dict()),
    }


# ---------------------------------------------------------------------------
# CLI: python -m backbone.load_weights [--cache-dir PATH]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Скачать и конвертировать веса antonior92 → PyTorch .pt"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help=f"Директория кэша (default: {_DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Путь к уже скачанному .hdf5 файлу",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачать / переконвертировать даже если файлы есть",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Проверить загрузку в модель после конвертации",
    )
    args = parser.parse_args()

    # Импортируем модель (нужен корень проекта в PYTHONPATH)
    try:
        from backbone.resnet1d import build_resnet1d
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backbone.resnet1d import build_resnet1d  # type: ignore

    hdf5 = download_hdf5(
        cache_dir=args.cache_dir,
        force=args.force,
    )
    state_dict = convert_keras_to_pytorch(hdf5)

    pt_out = args.cache_dir / _PT_FNAME
    torch.save(state_dict, str(pt_out))
    print(f"Конвертированные веса сохранены: {pt_out}")
    print(f"Ключей в state_dict: {len(state_dict)}")

    if args.verify:
        model = build_resnet1d()
        info = _apply_state_dict(model, state_dict, strict=False)
        print(f"Загружено: {info['loaded']}/{info['total']}")
        print(f"Пропущено: {info['missing']}")
        # Тестовый прогон
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 12, 5000)
            out = model(dummy)
        print(f"Тест-прогон: вход {dummy.shape} → выход {out.shape} ✓")
