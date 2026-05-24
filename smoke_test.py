"""
tests/test_smoke.py

Smoke-тест загрузки данных и пайплайна предобработки.

Принципы:
  - Работает БЕЗ реальных датасетов (wfdb/scipy мокируются)
  - Реальные датасеты тестируются если найдены на диске (авто-skip иначе)
  - Никаких внешних зависимостей кроме pytest + numpy + scipy
  - Детерминировано: seed=42

Запуск:
    pytest tests/test_smoke.py -v
    pytest tests/test_smoke.py -v -k "filter"      # только тесты фильтров
    pytest tests/test_smoke.py -v --tb=short       # краткий traceback
    pytest tests/test_smoke.py -v --real-data      # включить тесты с реальными данными
    python tests/test_smoke.py                     # напрямую (все тесты)
"""

from __future__ import annotations

import logging
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Путь к корню проекта ─────────────────────────────────────────────────────
_TESTS_DIR  = Path(__file__).parent
_PROJECT    = _TESTS_DIR.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logging.basicConfig(level=logging.WARNING)

RNG = np.random.default_rng(42)


# ══════════════════════════════════════════════════════════════════════════════
# Вспомогательные фабрики синтетических данных
# ══════════════════════════════════════════════════════════════════════════════

def make_ecg_signal(
    n_leads: int = 12,
    n_samples: int = 5000,
    fs: float = 500.0,
    add_noise: bool = True,
    add_powerline: bool = True,
    powerline_freq: float = 50.0,
) -> np.ndarray:
    """
    Синтетический ЭКГ-сигнал [n_leads, n_samples].
    Синус QRS-подобной формы + опциональная помеха + белый шум.
    """
    t = np.arange(n_samples) / fs
    # базовая форма: синус ~1 Гц с гармониками (имитация QRS)
    base = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 2.4 * t)
        + 0.1 * np.sin(2 * np.pi * 4.8 * t)
    )
    if add_powerline:
        base = base + 0.5 * np.sin(2 * np.pi * powerline_freq * t)
    if add_noise:
        base = base + 0.05 * RNG.standard_normal(n_samples)
    # тиражируем на все отведения с небольшими вариациями
    leads = [base * (1.0 + 0.1 * RNG.standard_normal()) for _ in range(n_leads)]
    return np.stack(leads).astype(np.float32)


def make_beat_signal(n_leads: int = 2, beat_len: int = 300) -> np.ndarray:
    """Синтетический бит [n_leads, beat_len]."""
    t = np.linspace(-1, 1, beat_len)
    qrs = np.exp(-50 * t ** 2)          # острый QRS-подобный пик
    p   = 0.3 * np.exp(-20 * (t + 0.5) ** 2)  # P-волна
    tw  = 0.2 * np.exp(-8  * (t - 0.5) ** 2)  # T-волна
    base = qrs + p + tw
    leads = [base + 0.02 * RNG.standard_normal(beat_len) for _ in range(n_leads)]
    return np.stack(leads).astype(np.float32)


def make_wfdb_record_mock(
    n_leads: int = 12,
    n_samples: int = 5000,
    fs: float = 500.0,
    sig_names: Optional[List[str]] = None,
) -> MagicMock:
    """Мок wfdb-записи."""
    record = MagicMock()
    signal = make_ecg_signal(n_leads, n_samples, fs).T.astype(np.float32)
    record.p_signal = signal        # [T, C]
    record.fs = fs
    record.sig_name = sig_names or [f"lead_{i}" for i in range(n_leads)]
    return record


def make_wfdb_annotation_mock(
    n_beats: int = 30,
    record_len: int = 130_000,
) -> MagicMock:
    """Мок wfdb-аннотации для MIT-BIH."""
    ann = MagicMock()
    # равномерно расставляем биты, отступив от краёв
    positions = np.linspace(500, record_len - 500, n_beats).astype(int).tolist()
    # чередуем классы N/S/V/F/Q
    symbol_pool = ["N", "N", "N", "S", "V", "N", "A", "V", "N", "F"]
    symbols = [symbol_pool[i % len(symbol_pool)] for i in range(n_beats)]
    ann.sample = positions
    ann.symbol = symbols
    return ann


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 1: Фильтры (preprocessing/filters.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestFilters:

    def test_design_lowpass_kaiser_returns_odd_length(self):
        from preprocessing.filters import design_lowpass_kaiser
        b = design_lowpass_kaiser(fs=500, cutoff=40, transition_width=5)
        assert len(b) % 2 == 1, "FIR должен иметь нечётное число тапов (type I)"
        assert b.dtype == np.float64

    def test_design_lowpass_kaiser_cutoff_exceeds_nyquist_raises(self):
        from preprocessing.filters import design_lowpass_kaiser
        with pytest.raises(ValueError):
            design_lowpass_kaiser(fs=500, cutoff=260)  # 260 > 500/2

    def test_apply_lowpass_preserves_shape(self):
        from preprocessing.filters import apply_lowpass
        sig = make_ecg_signal(12, 5000, 500.0)
        out = apply_lowpass(sig, fs=500)
        assert out.shape == sig.shape, f"shape изменилась: {out.shape} != {sig.shape}"

    def test_apply_lowpass_suppresses_high_freq(self):
        """80 Гц компонента должна быть ослаблена ≥ 40 дБ."""
        from preprocessing.filters import apply_lowpass
        from numpy.fft import rfft, rfftfreq

        fs = 500.0
        n  = 5000
        t  = np.arange(n) / fs
        sig_1ch = np.sin(2 * np.pi * 80 * t)  # только 80 Гц
        sig = np.tile(sig_1ch, (1, 1))         # [1, 5000]

        out = apply_lowpass(sig, fs=fs, cutoff=40)

        freqs  = rfftfreq(n, 1 / fs)
        sp_in  = np.abs(rfft(sig[0]))
        sp_out = np.abs(rfft(out[0]))

        idx_80 = int(np.argmin(np.abs(freqs - 80)))
        attn   = 20 * np.log10(sp_out[idx_80] / (sp_in[idx_80] + 1e-12))
        assert attn < -40, f"Ослабление 80 Гц = {attn:.1f} дБ, ожидали < -40 дБ"

    def test_apply_lowpass_passes_low_freq(self):
        """1 Гц компонента не должна ослабляться более чем на 3 дБ."""
        from preprocessing.filters import apply_lowpass
        from numpy.fft import rfft, rfftfreq

        fs = 500.0
        n  = 5000
        t  = np.arange(n) / fs
        sig_1ch = np.sin(2 * np.pi * 1.0 * t)
        sig = np.tile(sig_1ch, (1, 1))

        out = apply_lowpass(sig, fs=fs, cutoff=40)

        freqs  = rfftfreq(n, 1 / fs)
        sp_in  = np.abs(rfft(sig[0]))
        sp_out = np.abs(rfft(out[0]))

        idx_1  = int(np.argmin(np.abs(freqs - 1)))
        attn   = 20 * np.log10(sp_out[idx_1] / (sp_in[idx_1] + 1e-12))
        assert attn > -3, f"Полоса пропускания 1 Гц: {attn:.1f} дБ, ожидали > -3 дБ"

    def test_apply_notch_50hz_suppression(self):
        """Notch 50 Гц должен ослабить помеху ≥ 20 дБ."""
        from preprocessing.filters import apply_notch
        from numpy.fft import rfft, rfftfreq

        fs = 500.0
        n  = 5000
        t  = np.arange(n) / fs
        sig_1ch = np.sin(2 * np.pi * 50 * t)
        sig = np.tile(sig_1ch, (1, 1))

        out = apply_notch(sig, fs=fs, freq=50.0)

        freqs  = rfftfreq(n, 1 / fs)
        sp_in  = np.abs(rfft(sig[0]))
        sp_out = np.abs(rfft(out[0]))

        idx_50 = int(np.argmin(np.abs(freqs - 50)))
        attn   = 20 * np.log10(sp_out[idx_50] / (sp_in[idx_50] + 1e-12))
        assert attn < -20, f"Notch 50 Гц: {attn:.1f} дБ, ожидали < -20 дБ"

    def test_apply_notch_above_nyquist_is_noop(self):
        """Notch выше Найквиста — сигнал не меняется."""
        from preprocessing.filters import apply_notch
        sig = make_ecg_signal(2, 300, 360.0)
        out = apply_notch(sig, fs=360.0, freq=300.0)  # 300 Гц > 180 Гц
        np.testing.assert_array_equal(out, sig)

    def test_apply_ecg_filters_full_pipeline_shape(self):
        """Полный пайплайн: форма и dtype не меняются."""
        from preprocessing.filters import apply_ecg_filters
        sig = make_ecg_signal(12, 5000, 500.0)
        out = apply_ecg_filters(sig, fs=500.0, powerline_freq=50)
        assert out.shape == (12, 5000)
        assert out.dtype == np.float64

    def test_apply_ecg_filters_no_nan(self):
        """После фильтрации не должно быть NaN."""
        from preprocessing.filters import apply_ecg_filters
        sig = make_ecg_signal(12, 5000, 500.0)
        out = apply_ecg_filters(sig, fs=500.0, powerline_freq=50)
        assert not np.isnan(out).any(), "NaN после фильтрации"

    def test_apply_ecg_filters_beat_shape(self):
        """Пайплайн работает на битах [2, 300] (поток B)."""
        from preprocessing.filters import apply_ecg_filters
        beat = make_beat_signal(2, 300)
        out = apply_ecg_filters(beat, fs=360.0, powerline_freq=60)
        assert out.shape == (2, 300)

    @pytest.mark.parametrize("powerline_freq", [50, 60])
    def test_get_powerline_freq_returns_correct(self, powerline_freq):
        from preprocessing.filters import get_powerline_freq, DATASET_POWERLINE
        for ds_name, freq in DATASET_POWERLINE.items():
            result = get_powerline_freq(ds_name)
            assert result in (50, 60)

    def test_apply_notch_harmonics(self):
        """Гармоники 50 Гц подавляются."""
        from preprocessing.filters import apply_notch_harmonics
        from numpy.fft import rfft, rfftfreq

        fs = 500.0
        n  = 5000
        t  = np.arange(n) / fs
        sig_1ch = (
            np.sin(2 * np.pi * 50 * t)
            + np.sin(2 * np.pi * 100 * t)
        )
        sig = np.tile(sig_1ch, (1, 1))
        out = apply_notch_harmonics(sig, fs=fs, freq=50.0, harmonics=1)

        freqs  = rfftfreq(n, 1 / fs)
        sp_in  = np.abs(rfft(sig[0]))
        sp_out = np.abs(rfft(out[0]))

        for target_hz in (50, 100):
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            attn = 20 * np.log10(sp_out[idx] / (sp_in[idx] + 1e-12))
            assert attn < -15, f"{target_hz} Гц: {attn:.1f} дБ"


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 2: Нормализация (preprocessing/normalize.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalize:

    def test_resample_signal_same_fs_noop(self):
        from preprocessing.normalize import resample_signal
        sig = make_ecg_signal(12, 5000, 500.0)
        out = resample_signal(sig, fs_in=500, fs_out=500)
        assert out.shape == sig.shape

    @pytest.mark.parametrize("fs_in,fs_out,n_in", [
        (360, 500, 6480),    # MIT-BIH → 500 Гц
        (100, 500, 1000),    # PTB-XL lr → hr (теоретически)
        (500, 360, 5000),    # обратный ресэмплинг
        (250, 500, 2500),    # x2
    ])
    def test_resample_signal_output_length(self, fs_in, fs_out, n_in):
        from preprocessing.normalize import resample_signal
        sig = make_ecg_signal(2, n_in, float(fs_in))
        out = resample_signal(sig, fs_in=fs_in, fs_out=fs_out)
        expected = int(round(n_in * fs_out / fs_in))
        # допуск ±2 сэмпла из-за округления polyphase
        assert abs(out.shape[-1] - expected) <= 2, (
            f"fs={fs_in}→{fs_out}: ожидали {expected}, получили {out.shape[-1]}"
        )

    def test_crop_or_pad_crop(self):
        from preprocessing.normalize import crop_or_pad
        sig = make_ecg_signal(2, 6000, 500.0)
        out = crop_or_pad(sig, target_len=5000)
        assert out.shape == (2, 5000)

    def test_crop_or_pad_pad(self):
        from preprocessing.normalize import crop_or_pad
        sig = make_ecg_signal(2, 3000, 500.0)
        out = crop_or_pad(sig, target_len=5000)
        assert out.shape == (2, 5000)

    def test_crop_or_pad_exact_noop(self):
        from preprocessing.normalize import crop_or_pad
        sig = make_ecg_signal(2, 5000, 500.0)
        out = crop_or_pad(sig, target_len=5000)
        assert out.shape == (2, 5000)
        np.testing.assert_array_equal(out, sig)

    def test_crop_center_or_pad_symmetry(self):
        """Центральный кроп не должен сдвигать центр сигнала."""
        from preprocessing.normalize import crop_center_or_pad
        # сигнал с чётким пиком по центру
        sig_1d = np.zeros(8000)
        sig_1d[4000] = 1.0
        sig = sig_1d[np.newaxis, :]  # [1, 8000]
        out = crop_center_or_pad(sig, target_len=5000)
        assert out.shape == (1, 5000)
        peak_pos = int(np.argmax(out[0]))
        # пик должен быть близко к центру ±100 сэмплов
        assert abs(peak_pos - 2500) < 100, f"Пик смещён: позиция {peak_pos}"

    def test_znorm_zero_mean_unit_std(self):
        from preprocessing.normalize import znorm
        sig = make_beat_signal(2, 300).astype(np.float64)
        out = znorm(sig, axis=-1)
        assert out.shape == sig.shape
        for ch in range(2):
            assert abs(out[ch].mean()) < 1e-6, f"ch{ch}: mean={out[ch].mean():.2e}"
            assert abs(out[ch].std() - 1.0) < 1e-6, f"ch{ch}: std={out[ch].std():.4f}"

    def test_znorm_constant_signal_no_crash(self):
        """Константный сигнал не должен вызывать деление на 0."""
        from preprocessing.normalize import znorm
        sig = np.ones((2, 300), dtype=np.float64)
        out = znorm(sig)
        assert not np.isnan(out).any(), "NaN при константном входе"

    def test_normalize_record_stream_a_shape_and_dtype(self):
        """Stream A нормализация: выход [12, 5000] float64."""
        from preprocessing.normalize import normalize_record_stream_a
        sig = make_ecg_signal(12, 6480, 360.0)  # MIT-BIH-подобный
        out = normalize_record_stream_a(sig, fs_in=360, fs_out=500, target_len=5000)
        assert out.shape == (12, 5000), f"shape={out.shape}"
        assert out.dtype == np.float64

    def test_normalize_beat_stream_b_shape(self):
        """Stream B нормализация: выход [2, 300] float64, z-нормирован."""
        from preprocessing.normalize import normalize_beat_stream_b
        beat = make_beat_signal(2, 300)
        out = normalize_beat_stream_b(beat, target_len=300)
        assert out.shape == (2, 300)
        assert out.dtype == np.float64


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 3: SNOMED-маппинг (preprocessing/snomed_map.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestSnomedMap:

    def test_n_classes_is_positive(self):
        from preprocessing.snomed_map import N_CLASSES
        assert N_CLASSES > 0
        assert N_CLASSES <= 27

    def test_scored_snomed_classes_unique(self):
        from preprocessing.snomed_map import SCORED_SNOMED_CLASSES
        assert len(SCORED_SNOMED_CLASSES) == len(set(SCORED_SNOMED_CLASSES)), \
            "Дубликаты в SCORED_SNOMED_CLASSES"

    def test_snomed_to_index_is_inverse(self):
        from preprocessing.snomed_map import SCORED_SNOMED_CLASSES, SNOMED_TO_INDEX
        for idx, code in enumerate(SCORED_SNOMED_CLASSES):
            assert SNOMED_TO_INDEX[code] == idx, f"Несоответствие для code={code}"

    def test_encode_ptbxl_labels_afib(self):
        """AFIB → SNOMED 164889003 → должна быть 1 в label_vec."""
        from preprocessing.snomed_map import encode_ptbxl_labels, SNOMED_TO_INDEX
        scp = {"AFIB": 100.0}
        vec = encode_ptbxl_labels(scp)
        assert vec.shape[0] > 0
        afib_snomed = 164889003
        if afib_snomed in SNOMED_TO_INDEX:
            idx = SNOMED_TO_INDEX[afib_snomed]
            assert vec[idx] == 1.0, f"AFIB не закодирован: vec[{idx}]={vec[idx]}"

    def test_encode_ptbxl_labels_empty(self):
        """Пустой scp_codes → нулевой вектор."""
        from preprocessing.snomed_map import encode_ptbxl_labels, N_CLASSES
        vec = encode_ptbxl_labels({})
        assert vec.shape == (N_CLASSES,)
        assert vec.sum() == 0.0

    def test_encode_ptbxl_labels_unscored_ignored(self):
        """Коды вне 27 scored классов не попадают в вектор."""
        from preprocessing.snomed_map import encode_ptbxl_labels
        scp = {"LVH": 100.0, "AMI": 80.0}  # оба вне scored
        vec = encode_ptbxl_labels(scp)
        assert vec.sum() == 0.0

    def test_encode_ptbxl_labels_min_likelihood_filter(self):
        """Коды с вероятностью ниже порога фильтруются."""
        from preprocessing.snomed_map import encode_ptbxl_labels
        scp_low  = {"AFIB": 30.0}
        scp_high = {"AFIB": 80.0}
        vec_low  = encode_ptbxl_labels(scp_low,  min_likelihood=50.0)
        vec_high = encode_ptbxl_labels(scp_high, min_likelihood=50.0)
        # низкая вероятность не должна кодироваться
        assert vec_low.sum() == 0.0, "Код с низкой достоверностью прошёл фильтр"
        assert vec_high.sum() > 0.0, "Код с высокой достоверностью не закодирован"

    def test_encode_snomed_labels_known_code(self):
        """Прямой SNOMED код 164889003 (AFIB) должен быть закодирован."""
        from preprocessing.snomed_map import encode_snomed_labels, SNOMED_TO_INDEX
        codes = [164889003]  # AFIB
        vec = encode_snomed_labels(codes)
        if 164889003 in SNOMED_TO_INDEX:
            idx = SNOMED_TO_INDEX[164889003]
            assert vec[idx] == 1.0

    def test_encode_snomed_labels_unknown_code_ignored(self):
        """Несуществующий SNOMED код не должен вызывать ошибку."""
        from preprocessing.snomed_map import encode_snomed_labels
        vec = encode_snomed_labels([999999999])
        assert vec.sum() == 0.0

    def test_encode_snomed_labels_multiple_codes(self):
        """Несколько кодов → несколько единиц в векторе."""
        from preprocessing.snomed_map import (
            encode_snomed_labels, SCORED_SNOMED_CLASSES
        )
        # берём первые 3 scored класса
        codes = SCORED_SNOMED_CLASSES[:3]
        vec = encode_snomed_labels(codes)
        assert int(vec.sum()) == 3, f"Ожидали 3 единицы, получили {vec.sum()}"

    def test_decode_label_vector(self):
        """decode_label_vector обращает encode_snomed_labels."""
        from preprocessing.snomed_map import (
            encode_snomed_labels, decode_label_vector, SCORED_SNOMED_CLASSES
        )
        codes_in = SCORED_SNOMED_CLASSES[:2]
        vec = encode_snomed_labels(codes_in)
        codes_out = decode_label_vector(vec)
        assert set(codes_out) == set(codes_in), \
            f"decode вернул {codes_out}, ожидали {codes_in}"

    def test_scp_to_snomed_covers_common_rhythms(self):
        """Ключевые SCP-коды присутствуют в маппинге."""
        from preprocessing.snomed_map import SCP_TO_SNOMED
        for key in ("NORM", "AFIB", "LBBB", "CRBBB", "1AVB", "PVC", "SVPB"):
            assert key in SCP_TO_SNOMED, f"SCP-код {key!r} отсутствует в маппинге"


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 4: Базовые утилиты (data/_base.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestBase:

    def test_record_a_fields(self):
        """RecordA хранит все нужные поля с правильными типами."""
        from data._base import RecordA
        from preprocessing.snomed_map import N_CLASSES
        rec = RecordA(
            signal=np.zeros((12, 5000), dtype=np.float32),
            label_vec=np.zeros(N_CLASSES, dtype=np.float32),
            ecg_id="test_001",
            dataset="ptbxl",
            split="train",
            meta={"fs_orig": 500.0},
        )
        assert rec.signal.shape == (12, 5000)
        assert rec.label_vec.shape == (N_CLASSES,)
        assert rec.dataset == "ptbxl"

    def test_record_b_fields(self):
        """RecordB хранит все нужные поля."""
        from data._base import RecordB
        rec = RecordB(
            beat=np.zeros((2, 300), dtype=np.float32),
            beat_class=2,   # V
            record_id="100",
            beat_idx=5,
            sample_pos=1000,
        )
        assert rec.beat.shape == (2, 300)
        assert rec.beat_class == 2

    def test_processed_cache_put_get(self):
        """ProcessedCache: save → load → содержимое совпадает."""
        from data._base import ProcessedCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessedCache(tmpdir)
            sig = make_ecg_signal(12, 5000, 500.0)
            cache.put("test_ecg_001", sig)

            assert "test_ecg_001" in cache
            loaded = cache.get("test_ecg_001")
            assert loaded is not None
            assert loaded.shape == sig.shape
            np.testing.assert_allclose(loaded, sig.astype(np.float32), rtol=1e-5)

    def test_processed_cache_miss_returns_none(self):
        from data._base import ProcessedCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessedCache(tmpdir)
            assert "nonexistent_key" not in cache
            assert cache.get("nonexistent_key") is None

    def test_processed_cache_clear(self):
        from data._base import ProcessedCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessedCache(tmpdir)
            cache.put("k1", np.zeros(10))
            assert "k1" in cache
            cache.clear()
            assert "k1" not in cache

    def test_find_dataset_root_with_explicit_path(self, tmp_path):
        """find_dataset_root находит датасет если требуемый файл есть."""
        from data._base import find_dataset_root
        marker = "ptbxl_database.csv"
        (tmp_path / marker).touch()
        result = find_dataset_root("ptbxl", [marker], explicit_path=str(tmp_path))
        assert result == tmp_path

    def test_find_dataset_root_missing_returns_none(self, tmp_path):
        """find_dataset_root возвращает None если ничего не найдено."""
        from data._base import find_dataset_root
        result = find_dataset_root(
            "nonexistent_dataset_xyz",
            ["file_that_does_not_exist.csv"],
            explicit_path=str(tmp_path / "nowhere"),
        )
        assert result is None

    def test_parse_wfdb_header_labels(self, tmp_path):
        """parse_wfdb_header_labels читает SNOMED из .hea файла."""
        from data._base import parse_wfdb_header_labels
        hea = tmp_path / "test.hea"
        hea.write_text(
            "test 12 500 5000 05:00:00 01/01/2020\n"
            "#Dx: 164889003,270492004\n"
            "#Rx: Unknown\n"
        )
        codes = parse_wfdb_header_labels(hea)
        assert 164889003 in codes
        assert 270492004 in codes

    def test_parse_wfdb_header_labels_no_dx_returns_empty(self, tmp_path):
        from data._base import parse_wfdb_header_labels
        hea = tmp_path / "test.hea"
        hea.write_text("test 12 500 5000\n#Rx: Unknown\n")
        codes = parse_wfdb_header_labels(hea)
        assert codes == []

    def test_read_wfdb_record_mock(self):
        """read_wfdb_record возвращает [C, T] float32 и fs."""
        from data._base import read_wfdb_record
        mock_rec = make_wfdb_record_mock(12, 5000, 500.0)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            signal, fs, names = read_wfdb_record("fake/path")

        assert signal.shape == (12, 5000)
        assert signal.dtype == np.float32
        assert fs == 500.0
        assert not np.isnan(signal).any()

    def test_simple_progress_runs(self):
        """SimpleProgress не падает."""
        from data._base import SimpleProgress
        with SimpleProgress(10, "test") as p:
            for _ in range(10):
                p.update()


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 5: load_cpsc2018.py
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadCPSC2018:

    def _make_reference_csv(self, tmp_path: Path, records: Dict[str, List[int]]) -> None:
        ref = tmp_path / "REFERENCE.csv"
        lines = []
        for rec_id, labels in records.items():
            lines.append(",".join([rec_id] + [str(l) for l in labels]))
        ref.write_text("\n".join(lines))

    def test_label_to_snomed_mapping_coverage(self):
        """Все 9 CPSC меток присутствуют в маппинге."""
        from data.load_cpsc2018 import CPSC_LABEL_TO_SNOMED
        for i in range(1, 10):
            assert i in CPSC_LABEL_TO_SNOMED, f"Метка {i} отсутствует в маппинге"

    def test_load_cpsc_reference_parses_correctly(self, tmp_path):
        from data.load_cpsc2018 import load_cpsc_reference, CPSC_LABEL_TO_SNOMED
        self._make_reference_csv(tmp_path, {
            "A0001": [1],        # Normal
            "A0002": [2, 5],     # AF + RBBB
            "A0003": [7],        # PVC
        })
        df = load_cpsc_reference(tmp_path)
        assert len(df) == 3
        # A0001 → Normal → SNOMED 426783006
        row_0001 = df[df["record_id"] == "A0001"].iloc[0]
        assert CPSC_LABEL_TO_SNOMED[1] in row_0001["snomed_codes"]
        # A0002 → AF + RBBB
        row_0002 = df[df["record_id"] == "A0002"].iloc[0]
        assert CPSC_LABEL_TO_SNOMED[2] in row_0002["snomed_codes"]
        assert CPSC_LABEL_TO_SNOMED[5] in row_0002["snomed_codes"]

    def test_load_cpsc_reference_skips_header(self, tmp_path):
        """Строка заголовка игнорируется."""
        from data.load_cpsc2018 import load_cpsc_reference
        ref = tmp_path / "REFERENCE.csv"
        ref.write_text("Recording,First_label\nA0001,1\nA0002,2\n")
        df = load_cpsc_reference(tmp_path)
        assert len(df) == 2
        assert "A0001" in df["record_id"].values

    def test_load_cpsc_reference_missing_returns_empty(self, tmp_path):
        from data.load_cpsc2018 import load_cpsc_reference
        df = load_cpsc_reference(tmp_path)
        assert len(df) == 0

    def test_read_cpsc_signal_via_wfdb_mock(self, tmp_path):
        """read_cpsc_signal возвращает [12, 5000] при успешном чтении wfdb."""
        from data.load_cpsc2018 import read_cpsc_signal
        mock_rec = make_wfdb_record_mock(12, 5000, 500.0)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)
        assert sig.dtype == np.float32
        assert not np.isnan(sig).any()

    def test_read_cpsc_signal_pads_short_signal(self, tmp_path):
        """Сигнал короче 5000 → pad до [12, 5000]."""
        from data.load_cpsc2018 import read_cpsc_signal
        mock_rec = make_wfdb_record_mock(12, 3000, 500.0)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)

    def test_read_cpsc_signal_resamples_from_250hz(self, tmp_path):
        """Запись 250 Гц ресэмплируется в 500 Гц."""
        from data.load_cpsc2018 import read_cpsc_signal
        mock_rec = make_wfdb_record_mock(12, 2500, 250.0)  # 10 с при 250 Гц

        with patch("wfdb.rdrecord", return_value=mock_rec):
            sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)

    def test_read_cpsc_signal_pads_channels(self, tmp_path):
        """Запись с 8 отведениями → паддируем до 12."""
        from data.load_cpsc2018 import read_cpsc_signal
        mock_rec = make_wfdb_record_mock(8, 5000, 500.0)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)

    def test_read_cpsc_signal_fallback_mat(self, tmp_path):
        """Fallback на scipy.io.loadmat когда wfdb падает."""
        from data.load_cpsc2018 import read_cpsc_signal

        mat_data = {
            "val": make_ecg_signal(12, 5000, 500.0),  # [12, 5000]
        }

        # wfdb падает, scipy.io.loadmat возвращает mat_data
        with patch("wfdb.rdrecord", side_effect=Exception("wfdb error")):
            with patch("scipy.io.loadmat", return_value=mat_data):
                sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)

    def test_read_cpsc_signal_mat_transposed_shape(self, tmp_path):
        """loadmat возвращает [T, 12] → транспонируется в [12, T]."""
        from data.load_cpsc2018 import read_cpsc_signal
        # [5000, 12] — транспонированная форма
        mat_data = {"val": make_ecg_signal(12, 5000, 500.0).T}

        with patch("wfdb.rdrecord", side_effect=Exception("wfdb error")):
            with patch("scipy.io.loadmat", return_value=mat_data):
                sig = read_cpsc_signal(tmp_path, "A0001", cache=None)

        assert sig is not None
        assert sig.shape == (12, 5000)

    def test_iter_cpsc2018_end_to_end(self, tmp_path):
        """iter_cpsc2018 выдаёт RecordA с правильными полями."""
        from data.load_cpsc2018 import iter_cpsc2018

        self._make_reference_csv(tmp_path, {"A0001": [2], "A0002": [1]})
        mock_rec = make_wfdb_record_mock(12, 5000, 500.0)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            records = list(iter_cpsc2018(
                root=str(tmp_path), use_cache=False, show_progress=False
            ))

        assert len(records) == 2
        for rec in records:
            assert rec.signal.shape == (12, 5000)
            assert rec.dataset == "cpsc2018"
            assert rec.split == "train"
            assert not np.isnan(rec.signal).any()

    def test_iter_cpsc2018_no_reference_no_hea_raises(self, tmp_path):
        """Пустая папка → FileNotFoundError."""
        from data.load_cpsc2018 import iter_cpsc2018
        with pytest.raises(FileNotFoundError):
            list(iter_cpsc2018(root=str(tmp_path), show_progress=False))


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 6: load_mitbih.py (поток B)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadMITBIH:

    def _make_fake_mitbih_dir(self, tmp_path: Path) -> Path:
        """Создаёт минимальную структуру MIT-BIH директории."""
        (tmp_path / "100.hea").write_text(
            "100 2 360 650000 0:00 02/01/1990\n"
            "100.dat 212 200 11 1024 995 -22131 0 MLII\n"
            "100.dat 212 200 11 1024 1011 20052 0 V5\n"
        )
        (tmp_path / "100.dat").write_bytes(b"\x00" * 100)
        (tmp_path / "100.atr").write_bytes(b"\x00" * 10)
        return tmp_path

    def test_aami_map_covers_all_standard_symbols(self):
        from data.load_mitbih import AAMI_MAP
        standard = {
            "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
            "A": 1, "a": 1, "J": 1, "S": 1,
            "V": 2, "E": 2,
            "F": 3,
            "/": 4, "f": 4, "Q": 4,
        }
        for sym, cls in standard.items():
            assert AAMI_MAP.get(sym) == cls, \
                f"Символ {sym!r}: ожидали класс {cls}, получили {AAMI_MAP.get(sym)}"

    def test_aami_names_has_5_classes(self):
        from data.load_mitbih import AAMI_NAMES
        assert len(AAMI_NAMES) == 5
        assert AAMI_NAMES[0] == "N"
        assert AAMI_NAMES[2] == "V"

    def test_segment_beats_basic(self):
        """_segment_beats нарезает нужное число битов."""
        from data.load_mitbih import _segment_beats
        signal = make_ecg_signal(2, 10_000, 360.0)
        # 5 биений, все в безопасной зоне
        r_peaks = [500, 1500, 2500, 3500, 4500]
        symbols = ["N", "A", "V", "F", "/"]
        beats, classes = _segment_beats(signal, r_peaks, symbols)
        # все 5 биений должны быть в результате
        assert len(beats) == 5
        assert beats.shape == (5, 2, 300)
        assert classes.tolist() == [0, 1, 2, 3, 4]

    def test_segment_beats_drops_edge_beats(self):
        """Биты у края (lo<0 или hi>T) отбрасываются."""
        from data.load_mitbih import _segment_beats
        signal = make_ecg_signal(2, 1000, 360.0)
        r_peaks = [50, 500, 990]    # первый и последний выходят за край
        symbols = ["N", "N", "N"]
        beats, classes = _segment_beats(signal, r_peaks, symbols)
        assert len(beats) == 1
        assert int(classes[0]) == 0

    def test_segment_beats_skips_unknown_symbols(self):
        """Аннотации ~ и | не в AAMI — пропускаются без ошибки."""
        from data.load_mitbih import _segment_beats
        signal = make_ecg_signal(2, 5000, 360.0)
        r_peaks = [500, 1500, 2500]
        symbols = ["~", "|", "N"]   # первые два — не в AAMI
        beats, classes = _segment_beats(signal, r_peaks, symbols)
        assert len(beats) == 1
        assert int(classes[0]) == 0

    def test_znorm_beats_stats(self):
        """_znorm_beats: mean≈0, std≈1 по каждому биту/каналу."""
        from data.load_mitbih import _znorm_beats
        beats = np.stack([make_beat_signal(2, 300) for _ in range(10)])
        normed = _znorm_beats(beats)
        assert normed.shape == (10, 2, 300)
        for i in range(10):
            for ch in range(2):
                assert abs(normed[i, ch].mean()) < 1e-5
                assert abs(normed[i, ch].std()  - 1.0) < 1e-4

    def test_znorm_beats_constant_no_nan(self):
        """Константный бит не вызывает NaN."""
        from data.load_mitbih import _znorm_beats
        beats = np.ones((3, 2, 300), dtype=np.float32)
        normed = _znorm_beats(beats)
        assert not np.isnan(normed).any()

    def test_iter_mitbih_beats_mocked(self, tmp_path):
        """iter_mitbih_beats выдаёт RecordB с правильными полями."""
        from data.load_mitbih import iter_mitbih_beats

        self._make_fake_mitbih_dir(tmp_path)
        raw_signal = make_ecg_signal(2, 130_000, 360.0)

        mock_rec = MagicMock()
        mock_rec.p_signal = raw_signal.T.astype(np.float32)
        mock_rec.fs = 360.0
        mock_rec.sig_name = ["MLII", "V5"]

        mock_ann = make_wfdb_annotation_mock(n_beats=20, record_len=130_000)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            with patch("wfdb.rdann", return_value=mock_ann):
                beats = list(iter_mitbih_beats(
                    root=str(tmp_path),
                    split=None,
                    use_cache=False,
                    show_progress=False,
                    record_ids=["100"],
                ))

        assert len(beats) > 0, "Ни одного бита не выдано"
        b = beats[0]
        assert b.beat.shape == (2, 300), f"shape={b.beat.shape}"
        assert b.beat.dtype == np.float32
        assert 0 <= b.beat_class <= 4
        assert b.record_id == "100"
        assert not np.isnan(b.beat).any()

    def test_iter_mitbih_class_distribution(self, tmp_path):
        """Классы битов соответствуют аннотациям."""
        from data.load_mitbih import iter_mitbih_beats
        import collections

        self._make_fake_mitbih_dir(tmp_path)
        raw_signal = make_ecg_signal(2, 130_000, 360.0)

        mock_rec = MagicMock()
        mock_rec.p_signal = raw_signal.T.astype(np.float32)
        mock_rec.fs = 360.0
        mock_rec.sig_name = ["MLII", "V5"]

        # 10 биений: 5 N, 3 V, 2 A
        n_beats = 10
        positions = [500 + i * 1000 for i in range(n_beats)]
        symbols   = ["N", "N", "N", "N", "N", "V", "V", "V", "A", "A"]
        mock_ann  = MagicMock()
        mock_ann.sample = positions
        mock_ann.symbol = symbols

        with patch("wfdb.rdrecord", return_value=mock_rec):
            with patch("wfdb.rdann", return_value=mock_ann):
                beats = list(iter_mitbih_beats(
                    root=str(tmp_path),
                    use_cache=False,
                    show_progress=False,
                    record_ids=["100"],
                    znorm=False,
                ))

        counter = collections.Counter(b.beat_class for b in beats)
        assert counter[0] == 5, f"N: {counter[0]}, ожидали 5"
        assert counter[2] == 3, f"V: {counter[2]}, ожидали 3"
        assert counter[1] == 2, f"S: {counter[1]}, ожидали 2"

    def test_load_mitbih_arrays_shape(self, tmp_path):
        """load_mitbih_arrays возвращает X [N, 2, 300], y [N]."""
        from data.load_mitbih import load_mitbih_arrays

        self._make_fake_mitbih_dir(tmp_path)
        raw_signal = make_ecg_signal(2, 130_000, 360.0)
        mock_rec = MagicMock()
        mock_rec.p_signal = raw_signal.T.astype(np.float32)
        mock_rec.fs = 360.0
        mock_rec.sig_name = ["MLII", "V5"]
        mock_ann = make_wfdb_annotation_mock(15, 130_000)

        with patch("wfdb.rdrecord", return_value=mock_rec):
            with patch("wfdb.rdann", return_value=mock_ann):
                X, y = load_mitbih_arrays(
                    root=str(tmp_path),
                    show_progress=False,
                    use_cache=False,
                    record_ids=["100"],
                )

        assert X.ndim == 3
        assert X.shape[1] == 2
        assert X.shape[2] == 300
        assert y.shape[0] == X.shape[0]
        assert y.dtype == np.int64

    def test_ds1_ds2_splits_are_disjoint(self):
        from data.load_mitbih import DS1_RECORDS, DS2_RECORDS
        overlap = DS1_RECORDS & DS2_RECORDS
        assert len(overlap) == 0, f"DS1 и DS2 пересекаются: {overlap}"

    def test_get_kfold_splits_coverage(self):
        from data.load_mitbih import get_kfold_splits, DS1_RECORDS
        folds = get_kfold_splits(k=5, seed=42)
        assert len(folds) == 5
        # проверяем, что все DS1 записи встречаются как val хотя бы один раз
        all_val = set()
        for train_ids, val_ids in folds:
            all_val.update(val_ids)
            # train и val не пересекаются
            assert len(set(train_ids) & set(val_ids)) == 0
        assert all_val == DS1_RECORDS, (
            f"Не все DS1 записи покрыты k-fold: не покрыты {DS1_RECORDS - all_val}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 7: unified_dataset.py
# ══════════════════════════════════════════════════════════════════════════════

class TestUnifiedDataset:

    def _make_fake_records(self, n: int, dataset: str, split: str = "train") -> List:
        from data._base import RecordA
        from preprocessing.snomed_map import N_CLASSES
        return [
            RecordA(
                signal=make_ecg_signal(12, 5000, 500.0).astype(np.float32),
                label_vec=np.eye(N_CLASSES, dtype=np.float32)[i % N_CLASSES],
                ecg_id=f"{dataset}_{i:05d}",
                dataset=dataset,
                split=split,
                meta={"fs_orig": 500.0},
            )
            for i in range(n)
        ]

    # ── Взвешенный семплинг ────────────────────────────────────────────────

    def test_compute_weights_inv_sqrt_smaller_gets_higher(self):
        from data.unified_dataset import _compute_weights
        # 100 записей ptbxl + 10 записей cpsc2018
        labels = ["ptbxl"] * 100 + ["cpsc2018"] * 10
        w = _compute_weights(labels, strategy="inv_sqrt")
        w_ptbxl  = w[0]
        w_cpsc   = w[100]
        # cpsc (меньше) должен иметь больший вес
        assert w_cpsc > w_ptbxl, (
            f"cpsc вес {w_cpsc:.4f} должен > ptbxl вес {w_ptbxl:.4f}"
        )

    def test_compute_weights_uniform_all_equal(self):
        from data.unified_dataset import _compute_weights
        labels = ["ptbxl"] * 100 + ["cpsc2018"] * 10
        w = _compute_weights(labels, strategy="uniform")
        np.testing.assert_array_equal(w, np.ones(110))

    def test_compute_weights_dataset_equal_total(self):
        """Стратегия 'dataset': суммарный вес одинаков для каждого датасета."""
        from data.unified_dataset import _compute_weights
        labels = ["ptbxl"] * 100 + ["cpsc2018"] * 10
        w = _compute_weights(labels, strategy="dataset")
        total_ptbxl = w[:100].sum()
        total_cpsc  = w[100:].sum()
        np.testing.assert_allclose(total_ptbxl, total_cpsc, rtol=1e-5)

    def test_compute_weights_unknown_strategy_raises(self):
        from data.unified_dataset import _compute_weights
        with pytest.raises(ValueError, match="стратегия"):
            _compute_weights(["ptbxl"] * 5, strategy="magic")

    # ── UnifiedStreamADataset ─────────────────────────────────────────────

    def test_dataset_len_and_getitem(self):
        from data.unified_dataset import UnifiedStreamADataset
        recs = self._make_fake_records(20, "ptbxl")
        weights = np.ones(20)
        ds = UnifiedStreamADataset(recs, weights)
        assert len(ds) == 20
        sig, lbl = ds[0]
        # совместимо и с numpy и с torch
        assert hasattr(sig, "shape")
        assert sig.shape == (12, 5000)

    def test_dataset_filter_by_split(self):
        from data.unified_dataset import UnifiedStreamADataset
        recs_train = self._make_fake_records(15, "ptbxl", "train")
        recs_val   = self._make_fake_records(5,  "ptbxl", "val")
        recs_test  = self._make_fake_records(3,  "ptbxl", "test")
        all_recs = recs_train + recs_val + recs_test
        weights  = np.ones(len(all_recs))
        ds = UnifiedStreamADataset(all_recs, weights)

        ds_train = ds.filter_by_split("train")
        ds_val   = ds.filter_by_split("val")
        ds_test  = ds.filter_by_split("test")

        assert len(ds_train) == 15
        assert len(ds_val)   == 5
        assert len(ds_test)  == 3

    def test_dataset_get_labels_shape(self):
        from data.unified_dataset import UnifiedStreamADataset
        from preprocessing.snomed_map import N_CLASSES
        recs = self._make_fake_records(10, "ptbxl")
        ds = UnifiedStreamADataset(recs, np.ones(10))
        labels = ds.get_labels()
        assert labels.shape == (10, N_CLASSES)

    def test_dataset_get_ecg_ids(self):
        from data.unified_dataset import UnifiedStreamADataset
        recs = self._make_fake_records(5, "cpsc2018")
        ds = UnifiedStreamADataset(recs, np.ones(5))
        ids = ds.get_ecg_ids()
        assert len(ids) == 5
        assert all(isinstance(i, str) for i in ids)

    def test_dataset_weights_shape_mismatch_raises(self):
        from data.unified_dataset import UnifiedStreamADataset
        recs = self._make_fake_records(5, "ptbxl")
        with pytest.raises(AssertionError):
            UnifiedStreamADataset(recs, weights=np.ones(10))

    # ── compute_pos_weight ─────────────────────────────────────────────────

    def test_compute_pos_weight_shape_and_range(self):
        """pos_weight: shape [N_CLASSES], значения в [1, 100]."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch не установлен")

        from data.unified_dataset import UnifiedStreamADataset, compute_pos_weight
        from preprocessing.snomed_map import N_CLASSES

        recs = self._make_fake_records(50, "ptbxl")
        ds = UnifiedStreamADataset(recs, np.ones(50))
        pw = compute_pos_weight(ds)

        assert pw.shape == (N_CLASSES,)
        assert float(pw.min()) >= 1.0
        assert float(pw.max()) <= 100.0

    # ── build_unified_dataset (мок загрузчиков) ───────────────────────────

    def test_build_unified_dataset_combines_datasets(self):
        """build_unified_dataset объединяет PTB-XL + PhysioNet."""
        from data.unified_dataset import build_unified_dataset

        ptbxl_recs   = self._make_fake_records(10, "ptbxl",   "train")
        physionet_recs = self._make_fake_records(20, "physionet2020", "train")

        def fake_iter_ptbxl(*args, **kwargs):
            return iter(ptbxl_recs)

        def fake_iter_physionet(*args, **kwargs):
            return iter(physionet_recs)

        with patch("data.load_ptbxl.iter_ptbxl", side_effect=fake_iter_ptbxl):
            with patch("data.load_physionet2020.iter_physionet2020",
                       side_effect=fake_iter_physionet):
                ds = build_unified_dataset(
                    split="train",
                    use_ptbxl=True,
                    use_physionet2020=True,
                    use_cpsc2018=False,
                )

        assert len(ds) == 30
        assert ds.weights is not None
        assert len(ds.weights) == 30

    def test_build_unified_dataset_missing_both_raises(self):
        """Если оба датасета недоступны → RuntimeError."""
        from data.unified_dataset import build_unified_dataset

        with patch("data.load_ptbxl.iter_ptbxl",
                   side_effect=FileNotFoundError("ptbxl not found")):
            with patch("data.load_physionet2020.iter_physionet2020",
                       side_effect=FileNotFoundError("pn2020 not found")):
                with pytest.raises(RuntimeError):
                    build_unified_dataset(
                        split="train",
                        use_ptbxl=True,
                        use_physionet2020=True,
                        use_cpsc2018=False,
                    )

    def test_build_unified_dataset_deduplication(self):
        """PTB-XL ecg_id передаётся в PhysioNet как exclude_ptbxl_ids."""
        from data.unified_dataset import build_unified_dataset
        from data._base import RecordA
        from preprocessing.snomed_map import N_CLASSES

        ptbxl_recs = self._make_fake_records(5, "ptbxl", "train")
        # Делаем так, чтобы ptbxl ecg_ids были видны
        for i, r in enumerate(ptbxl_recs):
            object.__setattr__(r, "ecg_id", str(i + 1))

        captured: Dict = {}

        def fake_iter_ptbxl(*args, **kwargs):
            return iter(ptbxl_recs)

        def fake_iter_physionet(*args, exclude_ptbxl_ids=None, **kwargs):
            captured["exclude_ptbxl_ids"] = exclude_ptbxl_ids
            return iter([])

        with patch("data.load_ptbxl.iter_ptbxl", side_effect=fake_iter_ptbxl):
            with patch("data.load_physionet2020.iter_physionet2020",
                       side_effect=fake_iter_physionet):
                build_unified_dataset(
                    split="train",
                    use_ptbxl=True,
                    use_physionet2020=True,
                    use_cpsc2018=False,
                    skip_physionet_ptbxl_subdir=True,
                )

        # PhysioNet получил набор PTB-XL ecg_ids для дедупликации
        assert "exclude_ptbxl_ids" in captured
        assert captured["exclude_ptbxl_ids"] is not None
        assert len(captured["exclude_ptbxl_ids"]) == 5

    def test_build_unified_dataset_weights_inv_sqrt(self):
        """Веса вычисляются корректно: меньший датасет → больший вес."""
        from data.unified_dataset import build_unified_dataset

        ptbxl_recs     = self._make_fake_records(100, "ptbxl",         "train")
        physionet_recs = self._make_fake_records(400, "physionet2020",  "train")

        with patch("data.load_ptbxl.iter_ptbxl",
                   side_effect=lambda *a, **k: iter(ptbxl_recs)):
            with patch("data.load_physionet2020.iter_physionet2020",
                       side_effect=lambda *a, **k: iter(physionet_recs)):
                ds = build_unified_dataset(
                    split="train",
                    sampling_strategy="inv_sqrt",
                )

        # вес ptbxl (n=100): 1/sqrt(100) = 0.1
        # вес physionet (n=400): 1/sqrt(400) = 0.05
        w_ptbxl    = float(ds.weights[0])
        w_physionet = float(ds.weights[100])
        assert w_ptbxl > w_physionet, (
            f"ptbxl вес {w_ptbxl:.4f} должен быть больше physionet {w_physionet:.4f}"
        )

    # ── get_sampler ────────────────────────────────────────────────────────

    def test_get_sampler_returns_correct_num_samples(self):
        try:
            import torch
            from torch.utils.data import WeightedRandomSampler
        except ImportError:
            pytest.skip("torch не установлен")

        from data.unified_dataset import UnifiedStreamADataset, get_sampler

        recs = self._make_fake_records(50, "ptbxl")
        ds = UnifiedStreamADataset(recs, np.ones(50))
        sampler = get_sampler(ds, num_samples=100)

        assert isinstance(sampler, WeightedRandomSampler)
        assert sampler.num_samples == 100

    def test_get_sampler_no_weights_raises(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch не установлен")

        from data.unified_dataset import UnifiedStreamADataset, get_sampler
        recs = self._make_fake_records(5, "ptbxl")
        ds = UnifiedStreamADataset(recs, weights=None)

        with pytest.raises(ValueError, match="weights"):
            get_sampler(ds)


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 8: Интеграционные тесты (полный пайплайн без датасетов)
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Полный пайплайн фильтр → нормализация → кодирование метки.
    Датасеты не нужны — используем синтетические сигналы.
    """

    def test_stream_a_full_pipeline(self):
        """
        Поток A: raw [12, 5000] 500 Гц
          → apply_ecg_filters
          → normalize_record_stream_a
          → encode_ptbxl_labels
          → RecordA
        Итог: signal [12, 5000] float32, label_vec [N_CLASSES] float32, без NaN.
        """
        from preprocessing.filters import apply_ecg_filters
        from preprocessing.normalize import normalize_record_stream_a
        from preprocessing.snomed_map import encode_ptbxl_labels, N_CLASSES
        from data._base import RecordA

        raw = make_ecg_signal(12, 5000, 500.0)

        # 1. Фильтрация
        filtered = apply_ecg_filters(raw, fs=500.0, powerline_freq=50)
        assert filtered.shape == (12, 5000)
        assert not np.isnan(filtered).any()

        # 2. Нормализация (500 → 500 = noop)
        normalized = normalize_record_stream_a(filtered, fs_in=500, fs_out=500)
        assert normalized.shape == (12, 5000)

        # 3. Метка
        scp = {"AFIB": 100.0, "CRBBB": 80.0}
        label_vec = encode_ptbxl_labels(scp).astype(np.float32)
        assert label_vec.shape == (N_CLASSES,)
        assert label_vec.sum() > 0

        # 4. RecordA
        rec = RecordA(
            signal=normalized.astype(np.float32),
            label_vec=label_vec,
            ecg_id="synth_001",
            dataset="ptbxl",
            split="train",
            meta={"fs_orig": 500.0},
        )
        assert rec.signal.shape == (12, 5000)
        assert rec.signal.dtype == np.float32

    def test_stream_a_pipeline_from_360hz(self):
        """
        Поток A, ресэмплинг 360 → 500 Гц.
        Типичный сценарий для MIT-BIH в контексте backbone (если бы использовался).
        """
        from preprocessing.filters import apply_ecg_filters
        from preprocessing.normalize import normalize_record_stream_a

        raw = make_ecg_signal(12, 7200, 360.0)   # 20 с при 360 Гц

        filtered = apply_ecg_filters(raw, fs=360.0, powerline_freq=60)
        normalized = normalize_record_stream_a(
            filtered, fs_in=360.0, fs_out=500.0, target_len=5000
        )

        assert normalized.shape == (12, 5000)
        assert normalized.dtype == np.float64

    def test_stream_b_full_pipeline_single_beat(self):
        """
        Поток B: raw_record [2, T] 360 Гц
          → apply_ecg_filters
          → сегментация [2, 300]
          → _znorm_beats
          → RecordB
        """
        from preprocessing.filters import apply_ecg_filters
        from data.load_mitbih import _segment_beats, _znorm_beats
        from data._base import RecordB

        raw = make_ecg_signal(2, 10_000, 360.0)

        # 1. Фильтрация всей записи
        filtered = apply_ecg_filters(raw, fs=360.0, powerline_freq=60)

        # 2. Нарезка биений
        r_peaks = [500, 1500, 2500]
        symbols = ["N", "V", "A"]
        beats, classes = _segment_beats(filtered, r_peaks, symbols)
        assert beats.shape == (3, 2, 300)

        # 3. Z-нормализация
        normed = _znorm_beats(beats)
        assert not np.isnan(normed).any()

        # 4. RecordB из первого бита
        rec = RecordB(
            beat=normed[0].astype(np.float32),
            beat_class=int(classes[0]),
            record_id="100",
            beat_idx=0,
            sample_pos=500,
        )
        assert rec.beat.shape == (2, 300)
        assert rec.beat_class == 0   # N

    def test_cache_integration(self):
        """Кэш ускоряет повторную загрузку: hash-путь корректен."""
        from data._base import ProcessedCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ProcessedCache(tmpdir)
            sig = make_ecg_signal(12, 5000, 500.0)

            # первое сохранение
            cache.put("ecg_123", sig)
            assert "ecg_123" in cache

            # повторная загрузка
            loaded = cache.get("ecg_123")
            np.testing.assert_allclose(loaded, sig.astype(np.float32), rtol=1e-5)

            # другой ключ не влияет
            cache.put("ecg_456", sig * 2)
            loaded_456 = cache.get("ecg_456")
            np.testing.assert_allclose(loaded_456, (sig * 2).astype(np.float32), rtol=1e-5)

    @pytest.mark.parametrize("n_leads,n_samples,fs,target_fs", [
        (12, 5000, 500, 500),    # PTB-XL native
        (12, 1000, 100, 500),    # PTB-XL lr
        (12, 6250, 500, 500),    # длинная запись
        (12, 2500, 500, 500),    # короткая запись
        (8,  5000, 500, 500),    # меньше 12 каналов
    ])
    def test_stream_a_shape_invariant(self, n_leads, n_samples, fs, target_fs):
        """Выход stream A всегда [12, 5000] независимо от входа."""
        from preprocessing.filters import apply_ecg_filters
        from preprocessing.normalize import normalize_record_stream_a

        raw = make_ecg_signal(n_leads, n_samples, float(fs))
        # паддируем каналы если их меньше 12
        if n_leads < 12:
            pad = np.zeros((12 - n_leads, n_samples), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)

        filtered = apply_ecg_filters(raw, fs=float(fs), powerline_freq=50)
        out = normalize_record_stream_a(
            filtered, fs_in=float(fs), fs_out=float(target_fs), target_len=5000
        )
        assert out.shape == (12, 5000), \
            f"n_leads={n_leads}, n_samples={n_samples}, fs={fs}: shape={out.shape}"


# ══════════════════════════════════════════════════════════════════════════════
# БЛОК 9: Опциональные тесты с реальными датасетами (--real-data)
# ══════════════════════════════════════════════════════════════════════════════

def _has_dataset(env_key: str, required_file: str) -> bool:
    import os
    root = os.environ.get(env_key)
    if root:
        return Path(root, required_file).exists()
    return False

_HAVE_PTBXL       = _has_dataset("ECG_PTBXL_ROOT", "ptbxl_database.csv")
_HAVE_PHYSIONET   = _has_dataset("ECG_PHYSIONET2020_ROOT", "RECORDS")
_HAVE_CPSC        = _has_dataset("ECG_CPSC2018_ROOT", "REFERENCE.csv")
_HAVE_MITBIH      = _has_dataset("ECG_MITBIH_ROOT", "100.hea")


@pytest.mark.skipif(not _HAVE_PTBXL, reason="PTB-XL не найден (ECG_PTBXL_ROOT)")
class TestRealPTBXL:

    def test_ptbxl_loads_one_record(self):
        from data.load_ptbxl import iter_ptbxl
        records = list(iter_ptbxl(splits="test", limit=1, show_progress=False))
        assert len(records) == 1
        rec = records[0]
        assert rec.signal.shape == (12, 5000)
        assert rec.signal.dtype == np.float32
        assert not np.isnan(rec.signal).any()
        assert rec.split == "test"

    def test_ptbxl_split_sizes(self):
        from data.load_ptbxl import load_ptbxl_manifest, find_dataset_root
        import os
        root = find_dataset_root("ptbxl", ["ptbxl_database.csv"],
                                 os.environ.get("ECG_PTBXL_ROOT"))
        df = load_ptbxl_manifest(root)
        n_train = (df["split"] == "train").sum()
        n_val   = (df["split"] == "val").sum()
        n_test  = (df["split"] == "test").sum()
        # PTB-XL: ~17 500 записей, folds 1–8 = train
        assert n_train > 10_000, f"train={n_train} подозрительно мало"
        assert n_val   >  1_000, f"val={n_val} подозрительно мало"
        assert n_test  >  1_000, f"test={n_test} подозрительно мало"

    def test_ptbxl_signal_not_all_zeros(self):
        from data.load_ptbxl import iter_ptbxl
        records = list(iter_ptbxl(splits="test", limit=5, show_progress=False))
        for rec in records:
            assert rec.signal.std() > 0.0, \
                f"ecg_id={rec.ecg_id}: нулевой сигнал!"


@pytest.mark.skipif(not _HAVE_CPSC, reason="CPSC 2018 не найден (ECG_CPSC2018_ROOT)")
class TestRealCPSC2018:

    def test_cpsc_loads_one_record(self):
        from data.load_cpsc2018 import iter_cpsc2018
        records = list(iter_cpsc2018(limit=1, show_progress=False))
        assert len(records) == 1
        rec = records[0]
        assert rec.signal.shape == (12, 5000)
        assert rec.dataset == "cpsc2018"
        assert not np.isnan(rec.signal).any()


@pytest.mark.skipif(not _HAVE_MITBIH, reason="MIT-BIH не найден (ECG_MITBIH_ROOT)")
class TestRealMITBIH:

    def test_mitbih_loads_beats_from_record_100(self):
        from data.load_mitbih import iter_mitbih_beats
        beats = list(iter_mitbih_beats(
            record_ids=["100"], use_cache=False, show_progress=False
        ))
        # запись 100 содержит ~2273 аннотированных бита
        assert len(beats) > 100, f"Мало битов из записи 100: {len(beats)}"
        b = beats[0]
        assert b.beat.shape == (2, 300)
        assert not np.isnan(b.beat).any()

    def test_mitbih_class_balance(self):
        """В записи 100 преобладают нормальные биты (класс N)."""
        from data.load_mitbih import iter_mitbih_beats
        import collections
        beats = list(iter_mitbih_beats(
            record_ids=["100"], use_cache=False, show_progress=False
        ))
        counter = collections.Counter(b.beat_class for b in beats)
        # N должны составлять большинство
        total = sum(counter.values())
        n_frac = counter.get(0, 0) / total
        assert n_frac > 0.5, f"Слишком мало N-битов: {n_frac:.1%}"


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа для запуска напрямую
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-x"],
        cwd=str(_PROJECT),
    )
    sys.exit(result.returncode)