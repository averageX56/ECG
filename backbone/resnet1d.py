"""
backbone/resnet1d.py
ResNet-1D backbone для 12-канальных ЭКГ-записей.

Архитектура: Ribeiro et al. 2020
  «Automatic diagnosis of the 12-lead ECG using a deep neural network»
  https://www.nature.com/articles/s41467-020-15432-4
  Реф-реализация: https://github.com/antonior92/automatic-ecg-diagnosis

Вход : [B, 12, 5000]   — 12 отведений, 10 с при 500 Гц
Выход: [B, 320]        — вектор признаков после Global Average Pool

Параллельно документированы оригинальные имена слоёв Keras, чтобы
load_weights.py мог точно сопоставить тензоры при конвертации.

Управление заморозкой для двухшагового обучения:
  model.backbone.freeze_all()           # шаг 1
  model.backbone.unfreeze_last_n(2)     # шаг 2
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Гиперпараметры архитектуры (должны совпадать с Zenodo-весами)
# ---------------------------------------------------------------------------
_N_LEADS        = 12
_INIT_FILTERS   = 64
_BLOCK_FILTERS  = [128, 192, 256, 320]   # каналы на выходе каждого блока
_KERNEL_SIZE    = 16
_BLOCK_STRIDES  = [4, 4, 2, 2]           # stride на каждый ResBlock
_DROPOUT        = 0.5                    # dropout keep_prob=0.5 как в оригинале
_FEAT_DIM       = 320                    # размерность признакового вектора

# Маппинг «наш слой» → «имя слоя в Keras» (для load_weights.py)
# Используется при конвертации весов из .hdf5
KERAS_LAYER_MAP: dict[str, str] = {
    # stem
    "stem.conv":   "layer_0/conv",
    "stem.bn":     "layer_0/batch_normalization",
    # residual blocks: "res_blocks.{i}.*" → "layer_{i+1}/..."
}


# ---------------------------------------------------------------------------
# Базовый строительный блок
# ---------------------------------------------------------------------------

class _ResBlock1d(nn.Module):
    """
    Один residual-блок для ResNet-1D (архитектура Ribeiro 2020).

    Main path
    ---------
    x → Conv1d → BN → ReLU → Dropout → Conv1d(stride) → BN → (+skip) → ReLU → Dropout

    Skip path
    ---------
    Если stride > 1 или in_channels != out_channels:
      x → MaxPool1d(stride) → Conv1d(1×1) → BN
    иначе:
      x (identity)

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
        Stride второй свёрточной операции (downsample).
    dropout : float
        Вероятность обнуления (PyTorch-соглашение: p=0.5 → half zeroed).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 16,
        stride: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Padding для «same» при нечётном ядре
        # kernel_size=16 (чётное) → padding=(k-1)//2 не даёт «same»,
        # поэтому используем явный pad перед каждой свёрткой.
        self.pad = (kernel_size - 1) // 2
        self.kernel_size = kernel_size

        # ── Main path ────────────────────────────────────────────────────────
        # conv_1: in_channels → out_channels, stride=1
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding=self.pad, bias=False,
        )
        self.bn1    = nn.BatchNorm1d(out_channels)
        self.drop1  = nn.Dropout(p=dropout)

        # conv_2: out_channels → out_channels, stride=stride
        # Padding корректируется для сохранения целочисленной длины
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=self.pad, bias=False,
        )
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(p=dropout)

        # ── Skip path ────────────────────────────────────────────────────────
        if stride != 1 or in_channels != out_channels:
            self.skip: nn.Module = nn.Sequential(
                # MaxPool1d: downsample пространственной оси
                nn.MaxPool1d(kernel_size=stride, stride=stride),
                # 1×1 conv для выравнивания числа каналов
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, in_channels, T]

        Returns
        -------
        torch.Tensor  [B, out_channels, T // stride]
        """
        skip_x = self.skip(x)

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Размеры могут отличаться на ±1 (out-of-phase padding)
        # выравниваем skip по main
        diff = out.size(-1) - skip_x.size(-1)
        if diff > 0:
            out = out[..., :skip_x.size(-1)]
        elif diff < 0:
            skip_x = skip_x[..., :out.size(-1)]

        out = out + skip_x
        out = self.relu(out)
        out = self.drop2(out)
        return out


# ---------------------------------------------------------------------------
# Stem-блок (первый свёрточный слой)
# ---------------------------------------------------------------------------

class _Stem1d(nn.Module):
    """
    Первый слой сети: Conv1d → BN → ReLU → Dropout.

    Keras-имена: layer_0/conv, layer_0/batch_normalization
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 64,
        kernel_size: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv    = nn.Conv1d(in_channels, out_channels,
                                 kernel_size=kernel_size, padding=pad, bias=False)
        self.bn      = nn.BatchNorm1d(out_channels)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


# ---------------------------------------------------------------------------
# ResNet-1D Backbone
# ---------------------------------------------------------------------------

class ResNet1d(nn.Module):
    """
    ResNet-1D backbone (Ribeiro et al. 2020 / antonior92).

    Architecture
    ============
    Input:  [B, 12, L]   (L = 5000 for 10 s at 500 Hz)

    Stem:   Conv1d(12→64, k=16) → BN → ReLU → Dropout

    Residual blocks (стридовый downsampling):
      Block 0: 64  → 128,  stride=4
      Block 1: 128 → 192,  stride=4
      Block 2: 192 → 256,  stride=2
      Block 3: 256 → 320,  stride=2

    Temporal sequence length после всех блоков:
      L=5000 → 1250 → 312 → 156 → 78

    Global Average Pool: [B, 320, 78] → [B, 320]

    Output: [B, 320]  ← признаковый вектор для task-specific heads

    Parameters
    ----------
    in_channels : int
        Число каналов входного сигнала (12 для 12-отведений ЭКГ).
    init_filters : int
        Число фильтров в stem-слое (64 в оригинале).
    block_filters : list[int]
        Число выходных каналов каждого residual-блока.
    kernel_size : int
        Размер ядра свёртки (16 в оригинале).
    block_strides : list[int]
        Stride для каждого блока.
    dropout : float
        Вероятность Dropout.
    feat_dim : int
        Размерность выходного вектора признаков (= последний элемент
        block_filters после GAP).
    """

    def __init__(
        self,
        in_channels:   int       = _N_LEADS,
        init_filters:  int       = _INIT_FILTERS,
        block_filters: List[int] = _BLOCK_FILTERS,
        kernel_size:   int       = _KERNEL_SIZE,
        block_strides: List[int] = _BLOCK_STRIDES,
        dropout:       float     = _DROPOUT,
    ) -> None:
        super().__init__()

        if len(block_filters) != len(block_strides):
            raise ValueError(
                f"block_filters и block_strides должны иметь одинаковую длину, "
                f"получено {len(block_filters)} и {len(block_strides)}"
            )

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = _Stem1d(
            in_channels=in_channels,
            out_channels=init_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # ── Residual blocks ───────────────────────────────────────────────────
        blocks: List[nn.Module] = []
        in_ch = init_filters
        for out_ch, stride in zip(block_filters, block_strides):
            blocks.append(
                _ResBlock1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.res_blocks = nn.ModuleList(blocks)

        # ── Global Average Pool ───────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        self.feat_dim: int = block_filters[-1]  # 320

        # Инициализация весов
        self._init_weights()

        logger.info(
            "ResNet1d: stem=%d, blocks=%s, strides=%s, feat_dim=%d",
            init_filters, block_filters, block_strides, self.feat_dim,
        )

    # ── Инициализация весов ───────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """He-инициализация для Conv1d, 1-нициализация для BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, 12, 5000]
            Нормализованные ЭКГ-записи.

        Returns
        -------
        torch.Tensor  [B, 320]
            Признаковый вектор.
        """
        out = self.stem(x)                      # [B, 64, L]
        for block in self.res_blocks:
            out = block(out)                    # [B, 320, T_out]
        out = self.gap(out)                     # [B, 320, 1]
        out = out.squeeze(-1)                   # [B, 320]
        return out

    # ── Управление заморозкой ─────────────────────────────────────────────────

    def freeze_all(self) -> None:
        """
        Заморозить весь backbone (шаг 1 обучения).

        Все параметры переводятся в requires_grad=False.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("ResNet1d: все веса заморожены")

    def unfreeze_all(self) -> None:
        """Разморозить все параметры backbone."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("ResNet1d: все веса разморожены")

    def unfreeze_last_n(self, n: int) -> None:
        """
        Разморозить последние n residual-блоков (шаг 2 обучения).

        Stem и первые (len(blocks) - n) блоков остаются замороженными.

        Parameters
        ----------
        n : int
            Число последних блоков для разморозки.
        """
        # Все сначала замораживаем
        self.freeze_all()

        n_blocks = len(self.res_blocks)
        n = min(n, n_blocks)
        start_idx = n_blocks - n

        for i in range(start_idx, n_blocks):
            for param in self.res_blocks[i].parameters():
                param.requires_grad = True

        # GAP параметров не имеет, но оставляем для явности
        thawed = [f"res_blocks.{i}" for i in range(start_idx, n_blocks)]
        logger.info(
            "ResNet1d: разморожены блоки %s (из %d)", thawed, n_blocks
        )

    def frozen_params(self) -> int:
        """Число замороженных параметров."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def trainable_params(self) -> int:
        """Число обучаемых параметров."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        """Однострочная сводка: всего / обучаемых / замороженных."""
        total   = sum(p.numel() for p in self.parameters())
        train   = self.trainable_params()
        frozen  = self.frozen_params()
        return (
            f"ResNet1d params: total={total:,} "
            f"trainable={train:,} frozen={frozen:,}"
        )

    # ── Промежуточные активации ───────────────────────────────────────────────

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Прямой проход с сохранением выходов каждого блока.

        Удобно для анализа и ablation.

        Parameters
        ----------
        x : torch.Tensor  [B, 12, L]

        Returns
        -------
        feat : torch.Tensor  [B, 320]
        intermediates : list[torch.Tensor]
            Выходы каждого residual-блока до GAP.
        """
        out = self.stem(x)
        intermediates: List[torch.Tensor] = []
        for block in self.res_blocks:
            out = block(out)
            intermediates.append(out)
        feat = self.gap(out).squeeze(-1)
        return feat, intermediates


# ---------------------------------------------------------------------------
# Полная модель: backbone + task head
# ---------------------------------------------------------------------------

class ResNet1dWithHead(nn.Module):
    """
    Вспомогательная обёртка: backbone + один task-specific head.

    Используется для быстрого запуска / тестирования без tasks/*.py.

    Parameters
    ----------
    n_classes : int
        Число классов (27 для PhysioNet 2020).
    backbone_kwargs : dict
        Параметры для ResNet1d.__init__.
    dropout_head : float
        Dropout перед классификационным слоем.
    """

    def __init__(
        self,
        n_classes: int = 27,
        backbone_kwargs: Optional[dict] = None,
        dropout_head: float = 0.5,
    ) -> None:
        super().__init__()
        kwargs = backbone_kwargs or {}
        self.backbone = ResNet1d(**kwargs)
        self.dropout  = nn.Dropout(p=dropout_head)
        self.head     = nn.Linear(self.backbone.feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [B, 12, 5000]

        Returns
        -------
        logits : torch.Tensor  [B, n_classes]
            Сырые логиты (без сигмоиды).
        """
        feat   = self.backbone(x)               # [B, 320]
        feat   = self.dropout(feat)
        logits = self.head(feat)                # [B, n_classes]
        return logits


# ---------------------------------------------------------------------------
# Фабричная функция
# ---------------------------------------------------------------------------

def build_resnet1d(
    n_leads: int    = 12,
    dropout: float  = 0.5,
    feat_dim: int   = 320,
) -> ResNet1d:
    """
    Создаёт ResNet1d с параметрами по умолчанию (antonior92).

    Parameters
    ----------
    n_leads : int
        Число ЭКГ-отведений.
    dropout : float
        Вероятность Dropout.
    feat_dim : int
        Ожидаемая выходная размерность (только для проверки).

    Returns
    -------
    ResNet1d
    """
    model = ResNet1d(
        in_channels   = n_leads,
        init_filters  = _INIT_FILTERS,
        block_filters = _BLOCK_FILTERS,
        kernel_size   = _KERNEL_SIZE,
        block_strides = _BLOCK_STRIDES,
        dropout       = dropout,
    )
    if model.feat_dim != feat_dim:
        raise ValueError(
            f"Ожидали feat_dim={feat_dim}, "
            f"архитектура даёт {model.feat_dim}"
        )
    return model
