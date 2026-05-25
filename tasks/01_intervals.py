"""
tasks/01_intervals.py
Задача 01: Измерение ЭКГ-интервалов через NeuroKit2.

Измеряются: RR, QRS, QT, PQ для каждой записи потока [A].
Отведение: II (индекс 1 в 12-канальной матрице [12, 5000]).
Метод: neurokit2.ecg_delineate() → медиана и SD по всем битам записи.

Нормы (мс):
  RR:  600–1000
  QRS: <120
  QT:  <440
  PQ:  120–200

Покрытие: % записей с ≥5 детектируемых R-пиков.

Использование:
  python -m tasks.01_intervals --config configs/default.yaml
  python -m tasks.01_intervals --config configs/default.yaml --limit 200
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Константы нормативных значений интервалов (мс)
# ─────────────────────────────────────────────────────────────────────────────

INTERVAL_NORMS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "rr_ms":  (600.0,  1000.0),   # (min, max)
    "qrs_ms": (None,   120.0),
    "qt_ms":  (None,   440.0),
    "pq_ms":  (120.0,  200.0),
}

# Отведение II (индекс 1 из 12)
LEAD_II_IDX = 1

# Минимальное число R-пиков для включения записи в статистику
MIN_RPEAKS = 5

# Частота дискретизации для потока [A]
FS = 500.0


# ─────────────────────────────────────────────────────────────────────────────
# Результат по одной записи
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntervalResult:
    """Результат измерения интервалов для одной записи."""
    ecg_id:        str
    dataset:       str
    n_rpeaks:      int       # число найденных R-пиков

    # Медианы интервалов (мс), None если не определено
    rr_median:     Optional[float]
    rr_sd:         Optional[float]
    qrs_median:    Optional[float]
    qrs_sd:        Optional[float]
    qt_median:     Optional[float]
    qt_sd:         Optional[float]
    pq_median:     Optional[float]
    pq_sd:         Optional[float]

    # Флаги отклонения от нормы
    rr_abnormal:   bool = False
    qrs_abnormal:  bool = False
    qt_abnormal:   bool = False
    pq_abnormal:   bool = False

    error:         Optional[str] = None   # сообщение об ошибке если есть


def _check_norm(
    value: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
) -> bool:
    """True если значение выходит за пределы нормы."""
    if value is None:
        return False
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция измерения интервалов
# ─────────────────────────────────────────────────────────────────────────────

def measure_intervals(
    signal: np.ndarray,
    fs: float = FS,
    lead_idx: int = LEAD_II_IDX,
) -> Dict:
    """
    Измеряет ЭКГ-интервалы из одной записи с помощью NeuroKit2.

    Parameters
    ----------
    signal : np.ndarray  [12, 5000] float32
        Нормализованный сигнал после preprocessing-пайплайна.
    fs : float
        Частота дискретизации (500 Гц для потока [A]).
    lead_idx : int
        Индекс отведения (1 = отведение II).

    Returns
    -------
    dict с ключами:
        n_rpeaks, rr_ms, qrs_ms, qt_ms, pq_ms (каждый: median, sd),
        error (str | None)
    """
    try:
        import neurokit2 as nk
    except ImportError:
        return {"error": "neurokit2 не установлен: pip install neurokit2"}

    lead = signal[lead_idx].astype(np.float64)

    result: Dict = {
        "n_rpeaks": 0,
        "rr_ms":   {"median": None, "sd": None, "values": []},
        "qrs_ms":  {"median": None, "sd": None, "values": []},
        "qt_ms":   {"median": None, "sd": None, "values": []},
        "pq_ms":   {"median": None, "sd": None, "values": []},
        "error":   None,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Детекция R-пиков
            _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs), method="pantompkins1985")
            r_peaks = r_info.get("ECG_R_Peaks", np.array([]))

            if len(r_peaks) < MIN_RPEAKS:
                result["n_rpeaks"] = int(len(r_peaks))
                result["error"] = f"Мало R-пиков: {len(r_peaks)} < {MIN_RPEAKS}"
                return result

            result["n_rpeaks"] = int(len(r_peaks))

            # RR-интервалы (мс)
            rr_samples = np.diff(r_peaks).astype(float)
            rr_ms = rr_samples * 1000.0 / fs
            # Фильтрация физиологически возможных значений: 200–2000 мс
            rr_ms = rr_ms[(rr_ms >= 200) & (rr_ms <= 2000)]
            if len(rr_ms) >= 2:
                result["rr_ms"]["median"] = float(np.median(rr_ms))
                result["rr_ms"]["sd"]     = float(np.std(rr_ms))
                result["rr_ms"]["values"] = rr_ms.tolist()

            # Деляниация волн P/Q/R/S/T
            try:
                _, waves_peak = nk.ecg_delineate(
                    lead,
                    r_peaks,
                    sampling_rate=int(fs),
                    method="peaks",
                )
            except Exception as exc_del:
                logger.debug("ecg_delineate ошибка: %s", exc_del)
                return result

            # ── QRS-комплекс ─────────────────────────────────────────────────
            q_onsets  = np.array(waves_peak.get("ECG_Q_Peaks", []), dtype=float)
            s_offsets = np.array(waves_peak.get("ECG_S_Peaks", []), dtype=float)

            if len(q_onsets) > 0 and len(s_offsets) > 0:
                # Берём пары Q–S для каждого бита
                n_beats = min(len(q_onsets), len(s_offsets), len(r_peaks))
                qrs_vals = []
                for i in range(n_beats):
                    q = q_onsets[i]
                    s = s_offsets[i]
                    if np.isnan(q) or np.isnan(s):
                        continue
                    dur = (s - q) * 1000.0 / fs
                    if 20 < dur < 300:
                        qrs_vals.append(dur)
                if len(qrs_vals) >= 2:
                    result["qrs_ms"]["median"] = float(np.median(qrs_vals))
                    result["qrs_ms"]["sd"]     = float(np.std(qrs_vals))
                    result["qrs_ms"]["values"] = qrs_vals

            # ── QT-интервал ───────────────────────────────────────────────────
            t_offsets = np.array(waves_peak.get("ECG_T_Offsets", []), dtype=float)
            # Если T_Offsets не доступны, попробуем через delineate с методом cwt
            if np.all(np.isnan(t_offsets)) or len(t_offsets) == 0:
                try:
                    _, waves_cwt = nk.ecg_delineate(
                        lead, r_peaks,
                        sampling_rate=int(fs),
                        method="cwt",
                    )
                    t_offsets = np.array(
                        waves_cwt.get("ECG_T_Offsets", []), dtype=float
                    )
                except Exception:
                    pass

            if len(q_onsets) > 0 and len(t_offsets) > 0:
                n_beats = min(len(q_onsets), len(t_offsets))
                qt_vals = []
                for i in range(n_beats):
                    q = q_onsets[i]
                    t = t_offsets[i]
                    if np.isnan(q) or np.isnan(t):
                        continue
                    dur = (t - q) * 1000.0 / fs
                    if 150 < dur < 700:
                        qt_vals.append(dur)
                if len(qt_vals) >= 2:
                    result["qt_ms"]["median"] = float(np.median(qt_vals))
                    result["qt_ms"]["sd"]     = float(np.std(qt_vals))
                    result["qt_ms"]["values"] = qt_vals

            # ── PQ-интервал ───────────────────────────────────────────────────
            p_onsets = np.array(waves_peak.get("ECG_P_Onsets", []), dtype=float)
            if np.all(np.isnan(p_onsets)) or len(p_onsets) == 0:
                p_onsets = np.array(waves_peak.get("ECG_P_Peaks", []), dtype=float)

            if len(p_onsets) > 0 and len(q_onsets) > 0:
                n_beats = min(len(p_onsets), len(q_onsets))
                pq_vals = []
                for i in range(n_beats):
                    p = p_onsets[i]
                    q = q_onsets[i]
                    if np.isnan(p) or np.isnan(q):
                        continue
                    dur = (q - p) * 1000.0 / fs
                    if 50 < dur < 500:
                        pq_vals.append(dur)
                if len(pq_vals) >= 2:
                    result["pq_ms"]["median"] = float(np.median(pq_vals))
                    result["pq_ms"]["sd"]     = float(np.std(pq_vals))
                    result["pq_ms"]["values"] = pq_vals

    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("Ошибка measure_intervals: %s", exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Обработка одной записи
# ─────────────────────────────────────────────────────────────────────────────

def process_record(
    signal: np.ndarray,
    ecg_id: str,
    dataset: str,
    fs: float = FS,
    lead_idx: int = LEAD_II_IDX,
) -> IntervalResult:
    """
    Обрабатывает одну ЭКГ-запись и возвращает IntervalResult.
    """
    raw = measure_intervals(signal, fs=fs, lead_idx=lead_idx)

    rr_med  = raw["rr_ms"].get("median")
    qrs_med = raw["qrs_ms"].get("median")
    qt_med  = raw["qt_ms"].get("median")
    pq_med  = raw["pq_ms"].get("median")

    return IntervalResult(
        ecg_id       = ecg_id,
        dataset      = dataset,
        n_rpeaks     = raw["n_rpeaks"],
        rr_median    = rr_med,
        rr_sd        = raw["rr_ms"].get("sd"),
        qrs_median   = qrs_med,
        qrs_sd       = raw["qrs_ms"].get("sd"),
        qt_median    = qt_med,
        qt_sd        = raw["qt_ms"].get("sd"),
        pq_median    = pq_med,
        pq_sd        = raw["pq_ms"].get("sd"),
        rr_abnormal  = _check_norm(rr_med,  *INTERVAL_NORMS["rr_ms"]),
        qrs_abnormal = _check_norm(qrs_med, *INTERVAL_NORMS["qrs_ms"]),
        qt_abnormal  = _check_norm(qt_med,  *INTERVAL_NORMS["qt_ms"]),
        pq_abnormal  = _check_norm(pq_med,  *INTERVAL_NORMS["pq_ms"]),
        error        = raw.get("error"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Агрегация по всему датасету
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results: List[IntervalResult]) -> Dict:
    """
    Вычисляет агрегированную статистику по всем записям.

    Returns
    -------
    dict с полями:
        n_total, n_with_rpeaks, coverage_pct,
        per_interval: {name: {median_of_medians, iqr, n_measured, n_abnormal, abnormal_pct}}
    """
    n_total = len(results)
    valid = [r for r in results if r.n_rpeaks >= MIN_RPEAKS and r.error is None]
    n_valid = len(valid)
    coverage_pct = 100.0 * n_valid / n_total if n_total > 0 else 0.0

    stats: Dict = {}
    for key in ("rr", "qrs", "qt", "pq"):
        med_field = f"{key}_median"
        abn_field = f"{key}_abnormal"
        medians = [
            getattr(r, med_field)
            for r in valid
            if getattr(r, med_field) is not None
        ]
        if medians:
            arr = np.array(medians)
            stats[f"{key}_ms"] = {
                "median_of_medians": float(np.median(arr)),
                "mean":              float(np.mean(arr)),
                "std":               float(np.std(arr)),
                "p5":                float(np.percentile(arr, 5)),
                "p95":               float(np.percentile(arr, 95)),
                "n_measured":        int(len(arr)),
                "n_abnormal":        int(sum(getattr(r, abn_field) for r in valid)),
                "abnormal_pct":      float(
                    100.0 * sum(getattr(r, abn_field) for r in valid) / len(arr)
                    if len(arr) > 0 else 0.0
                ),
            }
        else:
            stats[f"{key}_ms"] = {
                "n_measured": 0, "median_of_medians": None,
                "n_abnormal": 0, "abnormal_pct": 0.0,
            }

    return {
        "n_total":       n_total,
        "n_with_rpeaks": n_valid,
        "coverage_pct":  round(coverage_pct, 2),
        "per_interval":  stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────────────────────────────────────────

def run_intervals(
    cfg,
    split: str = "test",
    limit: Optional[int] = None,
) -> Dict:
    """
    Запускает измерение интервалов на PTB-XL fold 10 (split='test').

    Parameters
    ----------
    cfg : _DotDict
    split : str
        'test' | 'val' | 'train'
    limit : int | None

    Returns
    -------
    dict с агрегированными метриками и путём к сохранённым данным
    """
    from data.load_ptbxl import iter_ptbxl
    from training._common import load_config

    ptbxl_root = cfg.paths.get("ptbxl_root") or cfg.paths.data_root
    results_dir = Path(cfg.paths.results_dir) / "task01_intervals"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Задача 01: измерение интервалов, split=%s limit=%s", split, limit)

    all_results: List[IntervalResult] = []
    n_processed = 0

    for rec in iter_ptbxl(
        root=ptbxl_root,
        splits=[split],
        use_cache=True,
        show_progress=True,
        limit=limit,
    ):
        ir = process_record(
            signal  = rec.signal,
            ecg_id  = rec.ecg_id,
            dataset = rec.dataset,
        )
        all_results.append(ir)
        n_processed += 1

        if n_processed % 500 == 0:
            logger.info("Обработано %d записей…", n_processed)

    logger.info("Всего обработано: %d записей", n_processed)

    # Агрегация
    summary = aggregate_results(all_results)

    # Вывод покрытия
    logger.info(
        "Покрытие (≥%d R-пиков): %d/%d = %.1f%%",
        MIN_RPEAKS,
        summary["n_with_rpeaks"],
        summary["n_total"],
        summary["coverage_pct"],
    )

    # Вывод интервалов
    for interval, stat in summary["per_interval"].items():
        if stat["n_measured"] > 0:
            logger.info(
                "%s: медиана=%.1f мс  SD=%.1f  аномалий=%.1f%%  (n=%d)",
                interval,
                stat["median_of_medians"] or 0,
                stat.get("std", 0),
                stat["abnormal_pct"],
                stat["n_measured"],
            )

    # Сохранение детальных результатов
    records_list = [asdict(r) for r in all_results]
    detail_path = results_dir / f"intervals_{split}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(records_list, f, indent=2, ensure_ascii=False, default=str)

    # Сохранение сводки
    summary_path = results_dir / f"intervals_{split}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Результаты сохранены: %s", results_dir)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты: отдельный вызов для одной записи (для tasks/02*, 03*, 04*)
# ─────────────────────────────────────────────────────────────────────────────

def get_rr_features(
    signal: np.ndarray,
    fs: float = FS,
    lead_idx: int = LEAD_II_IDX,
) -> Dict:
    """
    Быстрое извлечение RR-статистик для задачи 03 (ФП).

    Возвращает: rmssd, sdnn, cv, sd1, sd2, n_rpeaks

    Parameters
    ----------
    signal : np.ndarray  [12, 5000]
    fs : float
    lead_idx : int

    Returns
    -------
    dict  — ключи: rmssd, sdnn, cv, sd1, sd2, n_rpeaks, error
    """
    try:
        import neurokit2 as nk
    except ImportError:
        return {"error": "neurokit2 не установлен"}

    lead = signal[lead_idx].astype(np.float64)
    out: Dict = {
        "n_rpeaks": 0,
        "rmssd":    None,
        "sdnn":     None,
        "cv":       None,
        "sd1":      None,
        "sd2":      None,
        "error":    None,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_info = nk.ecg_peaks(lead, sampling_rate=int(fs))
            r_peaks = r_info.get("ECG_R_Peaks", np.array([]))

        if len(r_peaks) < MIN_RPEAKS:
            out["n_rpeaks"] = int(len(r_peaks))
            out["error"] = f"Мало R-пиков: {len(r_peaks)}"
            return out

        out["n_rpeaks"] = int(len(r_peaks))

        rr = np.diff(r_peaks).astype(float) * 1000.0 / fs  # мс
        rr = rr[(rr >= 200) & (rr <= 2000)]

        if len(rr) < 3:
            out["error"] = "Слишком мало валидных RR"
            return out

        # Временные метрики HRV
        rr_diff = np.diff(rr)
        out["sdnn"]  = float(np.std(rr))
        out["rmssd"] = float(np.sqrt(np.mean(rr_diff ** 2)))
        out["cv"]    = float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else None

        # Пуанкаре-плот (SD1/SD2)
        sd1 = float(np.std(rr_diff) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.std(rr) ** 2 - sd1 ** 2))
        out["sd1"] = sd1
        out["sd2"] = sd2

    except Exception as exc:
        out["error"] = str(exc)

    return out


def get_qrs_duration(
    signal: np.ndarray,
    fs: float = FS,
    lead_idx: int = LEAD_II_IDX,
) -> Optional[float]:
    """
    Возвращает медианную длительность QRS (мс) для задачи 02/04.

    None если не удалось определить.
    """
    raw = measure_intervals(signal, fs=fs, lead_idx=lead_idx)
    return raw["qrs_ms"].get("median")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Задача 01: измерение ЭКГ-интервалов (NeuroKit2)"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "val", "test"],
        help="PTB-XL сплит (default: test = fold 10)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from training._common import load_config
    cfg = load_config(args.config)

    limit = args.limit or (20 if args.debug else None)
    summary = run_intervals(cfg, split=args.split, limit=limit)

    print("\n════ Интервалы ════")
    print(f"Покрытие: {summary['coverage_pct']:.1f}% "
          f"({summary['n_with_rpeaks']}/{summary['n_total']} записей)")
    for name, stat in summary["per_interval"].items():
        med = stat.get("median_of_medians")
        if med is not None:
            print(
                f"  {name:8s}: медиана={med:6.1f} мс  "
                f"p5–p95=[{stat['p5']:.0f}–{stat['p95']:.0f}]  "
                f"аномалий={stat['abnormal_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
