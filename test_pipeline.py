"""
tests/test_pipeline.py
Тесты загрузки и предобработки ЭКГ-датасетов.

Три уровня:
  1. UNIT  — работают всегда, используют синтетические данные (numpy/scipy)
  2. SMOKE — пропускаются если данных нет на диске; проверяют 5 записей
  3. INTEG — полная проверка пайплайна на реальных данных (медленно, --integ)

Запуск:
  pytest tests/test_pipeline.py -v                # unit + smoke (если есть данные)
  pytest tests/test_pipeline.py -v -m integ       # только интеграционные
  pytest tests/test_pipeline.py -v --data /data   # указать корень данных

Скачать данные перед smoke/integ:
  python -m data.download --dataset mitbih  --dest /data
  python -m data.download --dataset ptbxl   --dest /data
"""
from __future__ import annotations

import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Добавляем корень проекта в sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Пути к данным (конфигурируются через --data или переменные окружения)
# ---------------------------------------------------------------------------

def _data_root() -> Path:
    """Возвращает корень данных из pytest-аргумента, ENV или дефолта."""
    # conftest.py может передавать через pytestconfig, здесь fallback
    env = os.environ.get("ECG_DATA_ROOT")
    if env:
        return Path(env)
    # стандартный Kaggle путь
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        return kaggle
    return Path("data/raw")


DATA_ROOT = _data_root()


def _dataset_path(name: str) -> Path:
    """Возвращает путь к конкретному датасету."""
    return DATA_ROOT / name


def _skip_if_missing(*paths: str):
    """Декоратор: пропустить тест если хотя бы один маркер не найден."""
    markers = list(paths)
    def _skip(fn):
        for marker in markers:
            if not (DATA_ROOT / marker).exists():
                return pytest.mark.skip(
                    reason=f"Данные отсутствуют: {DATA_ROOT / marker}. "
                           f"Запустите: python -m data.download --dataset ... --dest {DATA_ROOT}"
                )(fn)
        return fn
    return _skip


# ===========================================================================
# 1. UNIT-тесты предобработки (синтетические данные, без диска)
# ===========================================================================

class TestFilters:
    """Тесты preprocessing/filters.py на синтетических ЭКГ."""

    @staticmethod
    def _synthetic_ecg(n_leads=12, n_samples=5000, fs=500.0) -> np.ndarray:
        """Синтетический ЭКГ: суперпозиция синусоид + шум."""
        t = np.linspace(0, n_samples / fs, n_samples)
        base = np.sin(2 * np.pi * 1.2 * t)          # ~72 уд/мин
        noise_50 = 0.3 * np.sin(2 * np.pi * 50 * t)  # помеха 50 Гц
        noise_60 = 0.3 * np.sin(2 * np.pi * 60 * t)  # помеха 60 Гц
        white = 0.05 * np.random.default_rng(42).standard_normal(n_samples)
        channel = base + noise_50 + noise_60 + white
        return np.tile(channel, (n_leads, 1)).astype(np.float64)

    def test_notch_50hz_attenuates(self):
        """Notch 50 Гц подавляет компоненту 50 Гц."""
        from preprocessing.filters import apply_notch
        sig = self._synthetic_ecg()
        fs = 500.0
        t = np.linspace(0, sig.shape[1] / fs, sig.shape[1])

        # Мощность на 50 Гц до и после
        sin50 = np.sin(2 * np.pi * 50 * t)
        power_before = float(np.abs(np.dot(sig[0], sin50)))

        filtered = apply_notch(sig, fs=fs, freq=50.0)
        power_after = float(np.abs(np.dot(filtered[0], sin50)))

        assert power_after < power_before * 0.1, (
            f"Notch 50 Гц слабо подавил помеху: "
            f"до={power_before:.1f}, после={power_after:.1f}"
        )

    def test_lowpass_attenuates_high_freq(self):
        """LP 40 Гц подавляет компоненту 60 Гц."""
        from preprocessing.filters import apply_lowpass
        sig = self._synthetic_ecg()
        fs = 500.0
        t = np.linspace(0, sig.shape[1] / fs, sig.shape[1])
        sin60 = np.sin(2 * np.pi * 60 * t)

        power_before = float(np.abs(np.dot(sig[0], sin60)))
        filtered = apply_lowpass(sig, fs=fs, cutoff=40.0)
        power_after = float(np.abs(np.dot(filtered[0], sin60)))

        assert power_after < power_before * 0.1, (
            f"LP 40 Гц слабо подавил 60 Гц: до={power_before:.1f}, после={power_after:.1f}"
        )

    def test_ecg_filters_preserves_shape(self):
        """apply_ecg_filters не меняет форму тензора."""
        from preprocessing.filters import apply_ecg_filters
        sig = self._synthetic_ecg(n_leads=12, n_samples=5000)
        out = apply_ecg_filters(sig, fs=500.0, powerline_freq=50)
        assert out.shape == sig.shape

    def test_notch_above_nyquist_is_noop(self):
        """Notch с freq > Nyquist возвращает сигнал без изменений."""
        from preprocessing.filters import apply_notch
        sig = self._synthetic_ecg()
        out = apply_notch(sig, fs=100.0, freq=60.0)  # 60 > 50 = Nyquist
        np.testing.assert_array_almost_equal(out, sig)

    def test_design_lowpass_kaiser_length(self):
        """Фильтр Кайзера имеет нечётное число тапов."""
        from preprocessing.filters import design_lowpass_kaiser
        b = design_lowpass_kaiser(fs=500.0, cutoff=40.0, transition_width=5.0)
        assert len(b) % 2 == 1, "Фильтр должен иметь нечётную длину (тип I)"
        assert b.dtype == np.float64

    def test_design_lowpass_kaiser_raises_above_nyquist(self):
        """ValueError если cutoff >= Nyquist."""
        from preprocessing.filters import design_lowpass_kaiser
        with pytest.raises(ValueError, match="Найквист"):
            design_lowpass_kaiser(fs=200.0, cutoff=150.0)


class TestNormalize:
    """Тесты preprocessing/normalize.py."""

    def test_resample_500_to_250(self):
        """Ресэмплинг 500→250 Гц вдвое уменьшает длину."""
        from preprocessing.normalize import resample_signal
        sig = np.random.randn(12, 5000).astype(np.float64)
        out = resample_signal(sig, fs_in=500, fs_out=250)
        assert out.shape == (12, 2500), f"Ожидали (12, 2500), получили {out.shape}"

    def test_resample_1000_to_500(self):
        """Ресэмплинг 1000→500 Гц (PTB-датасет)."""
        from preprocessing.normalize import resample_signal
        sig = np.random.randn(12, 10000).astype(np.float64)
        out = resample_signal(sig, fs_in=1000, fs_out=500)
        assert out.shape == (12, 5000)

    def test_resample_257_to_500(self):
        """Ресэмплинг 257→500 Гц (StPetersburg INCART)."""
        from preprocessing.normalize import resample_signal
        n_samples_in = 257 * 10  # 10 секунд при 257 Гц
        sig = np.random.randn(12, n_samples_in).astype(np.float64)
        out = resample_signal(sig, fs_in=257, fs_out=500)
        expected_len = round(n_samples_in * 500 / 257)
        assert abs(out.shape[1] - expected_len) <= 2, (
            f"Ожидали ~{expected_len} сэмплов, получили {out.shape[1]}"
        )

    def test_resample_same_fs_returns_copy(self):
        """Ресэмплинг с fs_in == fs_out возвращает копию."""
        from preprocessing.normalize import resample_signal
        sig = np.random.randn(12, 5000).astype(np.float64)
        out = resample_signal(sig, fs_in=500, fs_out=500)
        np.testing.assert_array_equal(out, sig)
        assert out is not sig  # копия, не тот же объект

    def test_crop_or_pad_pads_short(self):
        """Короткий сигнал дополняется нулями справа."""
        from preprocessing.normalize import crop_or_pad
        sig = np.ones((12, 3000))
        out = crop_or_pad(sig, target_len=5000)
        assert out.shape == (12, 5000)
        assert out[:, 3000:].sum() == 0.0, "Паддинг должен быть нулевым"

    def test_crop_or_pad_crops_long(self):
        """Длинный сигнал обрезается от начала."""
        from preprocessing.normalize import crop_or_pad
        sig = np.arange(12 * 6000).reshape(12, 6000).astype(float)
        out = crop_or_pad(sig, target_len=5000, offset=0)
        assert out.shape == (12, 5000)
        np.testing.assert_array_equal(out, sig[:, :5000])

    def test_crop_or_pad_with_offset(self):
        """Обрезка с заданным смещением."""
        from preprocessing.normalize import crop_or_pad
        sig = np.arange(6000).reshape(1, 6000).astype(float)
        out = crop_or_pad(sig, target_len=5000, offset=500)
        np.testing.assert_array_equal(out[0], sig[0, 500:5500])

    def test_znorm_zero_mean_unit_std(self):
        """Z-нормировка даёт mean≈0, std≈1."""
        from preprocessing.normalize import znorm
        sig = np.random.randn(12, 5000) * 10 + 5
        out = znorm(sig, axis=-1)
        means = out.mean(axis=-1)
        stds = out.std(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)
        np.testing.assert_allclose(stds, 1.0, atol=1e-6)

    def test_znorm_constant_signal_no_nan(self):
        """Z-нормировка константного сигнала не порождает NaN."""
        from preprocessing.normalize import znorm
        sig = np.ones((2, 300)) * 5.0
        out = znorm(sig)
        assert not np.isnan(out).any()

    def test_normalize_record_stream_a_shape(self):
        """normalize_record_stream_a → [12, 5000] из разных fs."""
        from preprocessing.normalize import normalize_record_stream_a
        for fs_in in (257, 500, 1000):
            n_samp = fs_in * 12  # ~12 секунд
            sig = np.random.randn(12, n_samp).astype(np.float64)
            out = normalize_record_stream_a(sig, fs_in=fs_in, fs_out=500, target_len=5000)
            assert out.shape == (12, 5000), (
                f"fs_in={fs_in}: ожидали (12, 5000), получили {out.shape}"
            )

    def test_normalize_record_stream_a_clips(self):
        """normalize_record_stream_a клипирует амплитуду."""
        from preprocessing.normalize import normalize_record_stream_a
        sig = np.ones((12, 5000)) * 100.0  # огромная амплитуда
        out = normalize_record_stream_a(sig, fs_in=500, fs_out=500, clip_mv=10.0)
        assert out.max() <= 10.0 + 1e-9
        assert out.min() >= -10.0 - 1e-9

    def test_normalize_beat_stream_b_shape(self):
        """normalize_beat_stream_b → [2, 300]."""
        from preprocessing.normalize import normalize_beat_stream_b
        beat = np.random.randn(2, 280)  # короче 300
        out = normalize_beat_stream_b(beat, target_len=300)
        assert out.shape == (2, 300)


class TestSnomedMap:
    """Тесты preprocessing/snomed_map.py."""

    def test_n_classes_is_27(self):
        from preprocessing.snomed_map import N_CLASSES
        assert N_CLASSES == 27

    def test_encode_known_code(self):
        """Известный SNOMED-код кодируется в ненулевой вектор."""
        from preprocessing.snomed_map import encode_snomed_labels, N_CLASSES
        # AF: 164889003 (индекс 0 в _SCORED)
        vec = encode_snomed_labels([164889003])
        assert vec.shape == (N_CLASSES,)
        assert vec.sum() == 1.0
        assert vec[0] == 1.0, "AF должен быть на индексе 0"

    def test_encode_unknown_code_returns_zeros(self):
        """Неизвестный SNOMED-код → нулевой вектор (без исключений)."""
        from preprocessing.snomed_map import encode_snomed_labels, N_CLASSES
        vec = encode_snomed_labels([999999999])
        assert vec.shape == (N_CLASSES,)
        assert vec.sum() == 0.0

    def test_encode_multihot(self):
        """Несколько кодов → корректный multi-hot вектор."""
        from preprocessing.snomed_map import encode_snomed_labels
        # AF (164889003) + NSR (426783006)
        vec = encode_snomed_labels([164889003, 426783006])
        assert vec.sum() == 2.0

    def test_encode_empty_list(self):
        """Пустой список → нулевой вектор."""
        from preprocessing.snomed_map import encode_snomed_labels, N_CLASSES
        vec = encode_snomed_labels([])
        assert vec.shape == (N_CLASSES,)
        assert vec.sum() == 0.0

    def test_snomed_to_index_has_27_entries(self):
        from preprocessing.snomed_map import SNOMED_TO_INDEX, N_CLASSES
        assert len(SNOMED_TO_INDEX) == N_CLASSES

    def test_decode_label_vector_roundtrip(self):
        """encode → decode возвращает те же коды."""
        from preprocessing.snomed_map import (
            encode_snomed_labels, decode_label_vector, SCORED_SNOMED_CLASSES
        )
        codes_in = [164889003, 426783006]
        vec = encode_snomed_labels(codes_in)
        codes_out = decode_label_vector(vec)
        assert set(codes_out) == set(codes_in)

    def test_encode_ptbxl_labels(self):
        """encode_ptbxl_labels работает с SCP-кодами."""
        from preprocessing.snomed_map import encode_ptbxl_labels, N_CLASSES
        scp = {"AFIB": 100.0, "NORM": 0.0, "UNKNWN_CODE": 50.0}
        vec = encode_ptbxl_labels(scp, min_likelihood=50.0)
        assert vec.shape == (N_CLASSES,)
        # AFIB → AF = индекс 0; NORM не прошёл порог (0 < 50)
        assert vec[0] == 1.0, "AFIB должен быть закодирован"


class TestSubsetRegistry:
    """Тесты data/subset_registry.py."""

    def test_all_subsets_present(self):
        from data.subset_registry import SUBSETS
        expected = {"ptbxl", "cpsc", "cpsc_extra", "stpetersburg", "ptb", "georgia"}
        assert set(SUBSETS.keys()) == expected

    def test_get_subset_weight_positive(self):
        from data.subset_registry import get_subset_weight
        for name in ["ptbxl", "cpsc", "georgia", "stpetersburg"]:
            w = get_subset_weight(name)
            assert w > 0.0, f"Вес {name} должен быть положительным"

    def test_stpetersburg_highest_weight(self):
        """INCART (75 записей) имеет самый высокий вес."""
        from data.subset_registry import get_subset_weight
        w_incart = get_subset_weight("stpetersburg")
        w_ptbxl = get_subset_weight("ptbxl")
        assert w_incart > w_ptbxl, "INCART должен весить больше PTB-XL"

    def test_resolve_path(self):
        from data.subset_registry import resolve_path
        path = resolve_path("ptbxl", "/data")
        assert "/WFDB_PTB-XL" in path
        assert "/data" in path

    def test_list_train_subsets_excludes_finetune(self):
        from data.subset_registry import list_train_subsets
        train = list_train_subsets(include_finetune_only=False)
        assert "cpsc" not in train
        assert "cpsc_extra" not in train

    def test_list_train_subsets_includes_finetune(self):
        from data.subset_registry import list_train_subsets
        train = list_train_subsets(include_finetune_only=True)
        assert "cpsc" in train

    def test_list_finetune_subsets(self):
        from data.subset_registry import list_finetune_subsets
        ft = list_finetune_subsets()
        assert set(ft) == {"cpsc", "cpsc_extra"}


class TestUnifiedDataset:
    """Тесты data/unified_dataset.py на синтетических RecordA."""

    @staticmethod
    def _make_records(n_per_dataset: dict[str, int]) -> list:
        """Создаёт синтетические RecordA для тестирования."""
        from data._base import RecordA
        from preprocessing.snomed_map import N_CLASSES
        records = []
        for dataset, n in n_per_dataset.items():
            for i in range(n):
                records.append(RecordA(
                    signal=np.zeros((12, 5000), dtype=np.float32),
                    label_vec=np.zeros(N_CLASSES, dtype=np.float32),
                    ecg_id=f"{dataset}_{i:04d}",
                    dataset=dataset,
                    split="train",
                ))
        return records

    def test_inv_sqrt_weights_shape(self):
        from data.unified_dataset import _compute_weights
        labels = ["ptbxl"] * 100 + ["incart"] * 10
        w = _compute_weights(labels, strategy="inv_sqrt")
        assert w.shape == (110,)
        assert w.dtype == np.float64

    def test_inv_sqrt_smaller_dataset_higher_weight(self):
        from data.unified_dataset import _compute_weights
        labels = ["big"] * 1000 + ["small"] * 10
        w = _compute_weights(labels, strategy="inv_sqrt")
        w_big = w[0]         # вес записи из большого датасета
        w_small = w[-1]      # вес записи из маленького
        assert w_small > w_big

    def test_uniform_weights_all_equal(self):
        from data.unified_dataset import _compute_weights
        labels = ["a"] * 50 + ["b"] * 200
        w = _compute_weights(labels, strategy="uniform")
        np.testing.assert_array_equal(w, np.ones(250))

    def test_unknown_strategy_raises(self):
        from data.unified_dataset import _compute_weights
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            _compute_weights(["a", "b"], strategy="bogus")

    def test_build_unified_dataset_filters_split(self):
        from data.unified_dataset import build_unified_dataset
        from data._base import RecordA
        from preprocessing.snomed_map import N_CLASSES

        # Создаём записи с разными split
        records = []
        for split in ("train", "val", "test"):
            for i in range(5):
                records.append(RecordA(
                    signal=np.zeros((12, 5000), dtype=np.float32),
                    label_vec=np.zeros(N_CLASSES, dtype=np.float32),
                    ecg_id=f"{split}_{i}",
                    dataset="ptbxl",
                    split=split,
                ))

        ds, w = build_unified_dataset(records, split="train")
        assert len(ds) == 5
        assert len(w) == 5

    def test_unified_dataset_getitem_shape(self):
        from data.unified_dataset import build_unified_dataset
        records = self._make_records({"ptbxl": 3, "cpsc": 2})
        ds, _ = build_unified_dataset(records, split="train")
        sig, label = ds[0]
        # Работает и с PyTorch, и без него
        assert sig.shape[-2:] == (12, 5000) or sig.shape == (12, 5000)


class TestDownloadModule:
    """Тесты data/download.py (без реального скачивания)."""

    def test_check_dataset_missing(self, tmp_path):
        from data.download import check_dataset
        info = check_dataset("mitbih", tmp_path)
        assert info["present"] is False
        assert info["path"] is None

    def test_check_dataset_present(self, tmp_path):
        from data.download import check_dataset
        # Создаём маркерный файл
        mitbih_dir = tmp_path / "mitbih"
        mitbih_dir.mkdir()
        (mitbih_dir / "100.hea").write_text("fake")
        info = check_dataset("mitbih", tmp_path)
        assert info["present"] is True
        assert info["path"] == mitbih_dir

    def test_unknown_dataset_raises(self, tmp_path):
        from data.download import download_dataset
        with pytest.raises(ValueError, match="Неизвестный датасет"):
            download_dataset("nonexistent_db", tmp_path)

    def test_download_via_wfdb_missing_package(self, monkeypatch):
        """Если wfdb не установлен — поднимается RuntimeError."""
        from data import download as dl_module
        original = dl_module._require_wfdb
        def fake_require():
            raise SystemExit("wfdb not installed")
        monkeypatch.setattr(dl_module, "_require_wfdb", fake_require)
        with pytest.raises(SystemExit):
            dl_module._require_wfdb()

    def test_fmt_size_empty_dir(self, tmp_path):
        from data.download import _fmt_size
        assert _fmt_size(tmp_path) == "0.0 B"


# ===========================================================================
# 2. SMOKE-тесты (пропускаются если данных нет)
# ===========================================================================

class TestMITBIHSmoke:
    """Smoke-тесты загрузчика MIT-BIH (нужны реальные данные)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        mitbih_path = DATA_ROOT / "mitbih"
        if not mitbih_path.exists() or not list(mitbih_path.glob("*.hea")):
            pytest.skip(
                f"MIT-BIH не найден в {mitbih_path}. "
                f"Запустите: python -m data.download --dataset mitbih --dest {DATA_ROOT}"
            )

    def test_iter_mitbih_yields_records(self):
        """Итератор по MIT-BIH выдаёт хотя бы 5 бит."""
        from data.load_mitbih import iter_mitbih_beats
        beats = []
        for rec in iter_mitbih_beats(
            root=DATA_ROOT / "mitbih",
            record_ids=["100"],   # одна короткая запись
            show_progress=False,
        ):
            beats.append(rec)
            if len(beats) >= 5:
                break
        assert len(beats) >= 5, "Ожидали ≥5 бит из записи 100"

    def test_beat_shape_is_2x300(self):
        """Каждый бит имеет форму [2, 300]."""
        from data.load_mitbih import iter_mitbih_beats
        for rec in iter_mitbih_beats(
            root=DATA_ROOT / "mitbih",
            record_ids=["100"],
            show_progress=False,
        ):
            assert rec.beat.shape == (2, 300), (
                f"Ожидали (2, 300), получили {rec.beat.shape}"
            )
            break

    def test_beat_class_is_valid_aami(self):
        """Класс бита входит в AAMI {0,1,2,3,4}."""
        from data.load_mitbih import iter_mitbih_beats
        valid_classes = {0, 1, 2, 3, 4}
        for rec in iter_mitbih_beats(
            root=DATA_ROOT / "mitbih",
            record_ids=["100"],
            show_progress=False,
        ):
            assert rec.beat_class in valid_classes, (
                f"Недопустимый AAMI-класс: {rec.beat_class}"
            )
            break

    def test_beat_no_nan_after_znorm(self):
        """После z-нормировки бит не содержит NaN."""
        from data.load_mitbih import iter_mitbih_beats
        for rec in iter_mitbih_beats(
            root=DATA_ROOT / "mitbih",
            record_ids=["100"],
            znorm=True,
            show_progress=False,
        ):
            assert not np.isnan(rec.beat).any(), "NaN в z-нормированном бите"
            break

    def test_load_mitbih_arrays(self):
        """load_mitbih_arrays возвращает массивы правильной формы."""
        from data.load_mitbih import load_mitbih_arrays
        X, y = load_mitbih_arrays(
            root=DATA_ROOT / "mitbih",
            record_ids=["100"],
            show_progress=False,
        )
        assert X.ndim == 3
        assert X.shape[1] == 2
        assert X.shape[2] == 300
        assert X.shape[0] == y.shape[0]
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_ds1_ds2_no_overlap(self):
        """DS1 и DS2 не пересекаются."""
        from data.load_mitbih import DS1_RECORDS, DS2_RECORDS
        overlap = DS1_RECORDS & DS2_RECORDS
        assert len(overlap) == 0, f"Найдено пересечение DS1∩DS2: {overlap}"

    def test_kfold_splits_coverage(self):
        """K-fold покрывает все записи DS1 ровно один раз."""
        from data.load_mitbih import get_kfold_splits, DS1_RECORDS
        k = 5
        splits = get_kfold_splits(k=k)
        assert len(splits) == k
        val_sets = [v for _, v in splits]
        # Каждая запись встречается в val ровно один раз
        all_val = set()
        for val in val_sets:
            overlap = all_val & val
            assert not overlap, f"Запись {overlap} в val нескольких фолдах"
            all_val |= val
        assert all_val == DS1_RECORDS


class TestPTBXLSmoke:
    """Smoke-тесты загрузчика PTB-XL (нужны реальные данные)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_data(self):
        ptbxl_path = DATA_ROOT / "ptbxl"
        csv_path = ptbxl_path / "ptbxl_database.csv"
        if not csv_path.exists():
            pytest.skip(
                f"PTB-XL не найден ({csv_path}). "
                f"Запустите: python -m data.download --dataset ptbxl --dest {DATA_ROOT}"
            )

    def test_manifest_loads(self):
        """ptbxl_database.csv загружается и содержит нужные колонки."""
        from data.load_ptbxl import load_ptbxl_manifest
        df = load_ptbxl_manifest(DATA_ROOT / "ptbxl")
        required_cols = {"ecg_id", "split", "scp_codes", "filename_hr", "strat_fold"}
        missing = required_cols - set(df.columns)
        assert not missing, f"Отсутствуют колонки: {missing}"

    def test_fold_split_distribution(self):
        """Fold 10 → test, fold 9 → val, fold 1-8 → train."""
        from data.load_ptbxl import load_ptbxl_manifest
        df = load_ptbxl_manifest(DATA_ROOT / "ptbxl")
        test_folds = df[df["split"] == "test"]["strat_fold"].unique()
        val_folds = df[df["split"] == "val"]["strat_fold"].unique()
        assert list(test_folds) == [10], f"Тест-фолд: {test_folds}"
        assert list(val_folds) == [9], f"Вал-фолд: {val_folds}"

    def test_iter_ptbxl_yields_records(self):
        """Итератор по PTB-XL выдаёт корректные RecordA."""
        from data.load_ptbxl import iter_ptbxl
        records = list(iter_ptbxl(
            root=DATA_ROOT / "ptbxl",
            splits="test",
            use_cache=False,
            limit=3,
            show_progress=False,
        ))
        assert len(records) == 3

    def test_ptbxl_record_signal_shape(self):
        """Сигнал из PTB-XL: [12, 5000] float32."""
        from data.load_ptbxl import iter_ptbxl
        for rec in iter_ptbxl(
            root=DATA_ROOT / "ptbxl",
            splits="test",
            use_cache=False,
            limit=1,
            show_progress=False,
        ):
            assert rec.signal.shape == (12, 5000), f"Форма: {rec.signal.shape}"
            assert rec.signal.dtype == np.float32
            assert not np.isnan(rec.signal).any()
            break

    def test_ptbxl_label_vec_shape(self):
        """Label-вектор из PTB-XL: [27] float32."""
        from data.load_ptbxl import iter_ptbxl
        from preprocessing.snomed_map import N_CLASSES
        for rec in iter_ptbxl(
            root=DATA_ROOT / "ptbxl",
            splits="test",
            use_cache=False,
            limit=1,
            show_progress=False,
        ):
            assert rec.label_vec.shape == (N_CLASSES,)
            assert rec.label_vec.dtype == np.float32
            assert rec.label_vec.min() >= 0.0
            assert rec.label_vec.max() <= 1.0
            break

    def test_ptbxl_amplitude_in_range(self):
        """Амплитуда сигнала после clip не превышает ±10 мВ."""
        from data.load_ptbxl import iter_ptbxl
        for rec in iter_ptbxl(
            root=DATA_ROOT / "ptbxl",
            splits="test",
            use_cache=False,
            limit=5,
            show_progress=False,
        ):
            assert rec.signal.max() <= 10.1, (
                f"Амплитуда {rec.signal.max():.2f} > 10 мВ"
            )
            assert rec.signal.min() >= -10.1
            break

    def test_get_ptbxl_ecg_ids(self):
        """get_ptbxl_ecg_ids возвращает непустое множество строк."""
        from data.load_ptbxl import get_ptbxl_ecg_ids
        ids = get_ptbxl_ecg_ids(root=DATA_ROOT / "ptbxl")
        assert len(ids) > 1000, f"Ожидали >1000 ecg_id, получили {len(ids)}"
        assert all(isinstance(i, str) for i in ids)


# ===========================================================================
# 3. ИНТЕГРАЦИОННЫЙ тест (только при -m integ)
# ===========================================================================

@pytest.mark.integ
class TestFullPipeline:
    """
    Сквозной тест всего пайплайна.
    Требует наличия данных MIT-BIH И PTB-XL.
    Запуск: pytest -m integ tests/test_pipeline.py -v
    """

    def test_unified_dataset_with_real_ptbxl(self):
        """
        Строит UnifiedStreamADataset из реальных PTB-XL записей
        и проверяет веса для WeightedRandomSampler.
        """
        ptbxl_path = DATA_ROOT / "ptbxl"
        if not (ptbxl_path / "ptbxl_database.csv").exists():
            pytest.skip("PTB-XL не найден")

        from data.load_ptbxl import iter_ptbxl
        from data.unified_dataset import build_unified_dataset

        records = list(iter_ptbxl(
            root=ptbxl_path,
            splits="val",
            use_cache=True,
            limit=20,
            show_progress=False,
        ))
        assert len(records) > 0

        ds, weights = build_unified_dataset(records, split="val", strategy="inv_sqrt")
        assert len(ds) == len(records)
        assert weights.shape[0] == len(records)
        assert (weights > 0).all()

        # Проверяем, что DataLoader работает
        try:
            import torch
            from torch.utils.data import DataLoader
            sampler = ds.make_sampler()
            loader = DataLoader(ds, batch_size=4, sampler=sampler)
            batch_sig, batch_label = next(iter(loader))
            assert batch_sig.shape == (4, 12, 5000)
            assert batch_label.shape[1] == 27
        except ImportError:
            pytest.skip("PyTorch не установлен")

    def test_mitbih_full_train_test_split(self):
        """
        Загружает все биты MIT-BIH, проверяет AAMI-распределение.
        """
        mitbih_path = DATA_ROOT / "mitbih"
        if not mitbih_path.exists():
            pytest.skip("MIT-BIH не найден")

        from data.load_mitbih import load_mitbih_arrays
        X, y = load_mitbih_arrays(root=mitbih_path, split="train", show_progress=False)

        assert X.ndim == 3
        assert X.shape[1:] == (2, 300)

        classes_present = set(np.unique(y).tolist())
        assert 0 in classes_present, "Класс N (0) должен присутствовать в train"
        assert 2 in classes_present, "Класс V (2) должен присутствовать в train"

        # Класс N доминирует (>70% по AAMI EC57)
        n_ratio = (y == 0).mean()
        assert n_ratio > 0.5, f"Доля класса N: {n_ratio:.2%} — подозрительно мало"


# ===========================================================================
# conftest-фрагмент (если tests/conftest.py не создан)
# ===========================================================================

def pytest_addoption(parser):
    """Добавляет опцию --data в pytest."""
    try:
        parser.addoption(
            "--data",
            action="store",
            default=None,
            help="Корень данных (переопределяет ECG_DATA_ROOT)",
        )
    except ValueError:
        pass  # уже добавлено


def pytest_configure(config):
    """Регистрирует маркер integ."""
    config.addinivalue_line(
        "markers",
        "integ: интеграционные тесты (медленно, требуют данных)",
    )


def pytest_collection_modifyitems(config, items):
    """Устанавливает корень данных из --data аргумента."""
    global DATA_ROOT
    opt = config.getoption("--data", default=None)
    if opt is not None:
        DATA_ROOT = Path(opt)


if __name__ == "__main__":
    # Быстрый запуск без pytest
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        check=False,
    )
    sys.exit(result.returncode)
