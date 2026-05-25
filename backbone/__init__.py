"""
backbone/

ResNet-1D backbone для ecg-multidataset.

Публичный API:
  from backbone import ResNet1d, ResNet1dWithHead, build_resnet1d
  from backbone import load_pretrained_weights, download_hdf5
"""

from backbone.resnet1d import (
    ResNet1d,
    ResNet1dWithHead,
    build_resnet1d,
    _ResBlock1d,
    _Stem1d,
    _FEAT_DIM,
    _BLOCK_FILTERS,
    _BLOCK_STRIDES,
    _KERNEL_SIZE,
    _INIT_FILTERS,
    _N_LEADS,
    _DROPOUT,
)

from backbone.load_weights import (
    load_pretrained_weights,
    download_hdf5,
    convert_keras_to_pytorch,
    ZENODO_URL,
    ZENODO_RECORD_ID,
    _DEFAULT_CACHE_DIR,
)

__all__ = [
    # архитектура
    "ResNet1d",
    "ResNet1dWithHead",
    "build_resnet1d",
    "_ResBlock1d",
    "_Stem1d",
    # константы архитектуры
    "_FEAT_DIM",
    "_BLOCK_FILTERS",
    "_BLOCK_STRIDES",
    "_KERNEL_SIZE",
    "_INIT_FILTERS",
    "_N_LEADS",
    "_DROPOUT",
    # загрузка весов
    "load_pretrained_weights",
    "download_hdf5",
    "convert_keras_to_pytorch",
    "ZENODO_URL",
    "ZENODO_RECORD_ID",
    "_DEFAULT_CACHE_DIR",
]
