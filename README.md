# ECG — три пайплайна анализа 12-канальной ЭКГ (PhysioNet/CinC 2020)

Плоский, простой проект: три независимых решения задачи классификации ЭКГ на
27 диагностических классов (scored-классы PhysioNet/CinC Challenge 2020, SNOMED CT).
Без сложного мультипроцессинга — всё обычными последовательными проходами и
стандартным `DataLoader`.

| # | Решение | Файл | Метрика |
|---|---------|------|---------|
| 1 | Делинеация (neurokit) + rule-based диагноз по интервалам | `delineation.py` | precision/recall/F1 диагноза по интервалам vs реальные метки |
| — | **Отдельный** пайплайн извлечения ручных фич Prna → CSV | `extract_features.py` | — (готовит фичи для решения 3) |
| 2 | Гибридная модель **CTN** (CNN+Transformer) только на сигнале | `train_ctn.py` | метрика контеста + macro-F1 |
| 3 | Та же CTN + ручные фичи из CSV | `train_ctn.py --features` | метрика контеста для DL + ручные фичи |

## Структура

```
ecg_data.py          # чтение .hea+.mat, таблица записей (метки/возраст/пол из #Dx/#Age/#Sex),
                     #   детерминированные фолды, простой torch-Dataset (+ опц. кэш сигнала)
analytics.py         # распределение по классам + оценка качества (шумные записи, neurokit)
delineation.py       # РЕШЕНИЕ 1: делинеация + диагноз по интервалам + метрики + визуализация + CLI
extract_features.py  # извлечение ручных фич (feats.Features) -> CSV; CLI
model.py             # CTN (CNN-энкодер + Transformer, batch_first)
train_ctn.py         # РЕШЕНИЯ 2/3: обучение CTN; без --features = DL-only, с --features = +фичи; CLI
smoke_test.py        # самопроверка на синтетике (без данных и GPU)
feats/               # АУТЕНТИЧНЫЙ пакет ручных фич Prna/Goodfellow (features + 3 группы статистик + utils)
eval/                # официальный скоринг контеста (weights.csv, dx_mapping, evaluate_12ECG_score.py)
data/                # сюда кладёте датасет (.hea+.mat); в git не коммитится (см. .gitignore)
```

Оставлено только то, что нельзя извлечь из самого датасета: код `feats/` и
официальные артефакты `eval/`. Метки, возраст, пол — берутся из заголовков `.hea`;
фолды — детерминированный сплит по хэшу `record_id`; фичи — считаются `extract_features.py`.

## Данные

Положите записи в `data/` (любая вложенность), каждая — пара WFDB-файлов
`.hea` + `.mat` (формат PhysioNet/CinC 2020). Метки читаются из строки `#Dx:`
заголовка. Папка `data/` не коммитится.

## Запуск

```bash
pip install -r requirements.txt        # torch — отдельно, см. requirements.txt

# Решение 1: делинеация + диагноз + метрики + примеры разметки
python delineation.py --data-root data --delineate-method dwt --plot-dir outputs/plots

# Решение 2: CTN только на сигнале
python train_ctn.py --data-root data --epochs 20

# Решение 3: сначала извлечь фичи (один раз), потом обучить с ними
python extract_features.py --data-root data --out features.csv
python train_ctn.py --data-root data --features features.csv --epochs 20

# Быстрая проверка всего на синтетике (без данных и GPU)
python smoke_test.py
```

Обучение само использует CUDA, если доступна (`--device auto`). Слабый F1_macro
у решения 2 лечится перевзвешиванием классов (по умолчанию) и поклассовым
перебором порогов — отчёт печатает метрики при трёх режимах порогов
(0.5 / глобальный / поклассовый). Сигнал опционально кэшируется на диск
(`--cache-dir`, по умолчанию `cache/signals`) — первая эпоха заполняет кэш,
дальнейшие идут быстрее.

## Примечания

- Пакет `feats/` — авторское решение команды Prna (S. Goodfellow, PhysioNet 2017;
  модификации Philips для 2020). Атрибуция — в `AUTHORS.txt`. При импорте
  применяется тонкий слой совместимости со старыми версиями numpy/pandas/pyentrp.
- Расчёт фич детерминирован (фиксируется seed → воспроизводимый KMeans в HRV).
- Метрики считает официальный код `eval/evaluate_12ECG_score.py`.
