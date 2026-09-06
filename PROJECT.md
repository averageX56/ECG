# ECG: интервалы и классификация

**Единый ноутбук всех пайплайнов и просмотра качества:** [04_all_pipelines_a100.ipynb](notebooks/04_all_pipelines_a100.ipynb).
Одна A10040/80ГБ, Jupyter: [инструкция и данные для переноса](docs/CLUSTER.md). [Модульная структура](docs/STRUCTURE.md). [Новые BERT/ECGFounder результаты](reports/representation_results.md).

Локальный Python-проект с CLI, обученными весами и исследовательскими ноутбуками. Основной результат и реальные ограничения: [reports/RESULTS.md](reports/RESULTS.md). Научные источники: [reports/research_notes.md](reports/research_notes.md).

## Быстрый запуск

Из `C:\Projects\ECG` в установленном Python:

```powershell
python -m ecg_project analyze LUDB/1.hea --output reports/example_ludb
python -m ecg_project analyze data/mit-bih/200.csv --duration-seconds 60 --output reports/example_mit
python -m ecg_project analyze data/Training_WFDB/A0001.hea --output reports/example_challenge
python -m pytest -q
```

Без явного `--checkpoint` анализ использует адаптированный `artifacts/delineator_qt.pt`, если он существует, иначе исходный `artifacts/delineator.pt`. Классификатор проверяет хеш разметчика: несовместимые признаки не подаются в старые веса. Модели выбираются по `artifacts/{record,beat}_models/selection.json` на основе validation.

В новом окружении: `python -m pip install -e ".[dev]"`. CUDA-сборку PyTorch нужно установить под свою систему отдельно; в текущем окружении проверены Torch 2.7.1+cu118 и GTX 1050. Точные версии сохранены в `reports/environment.json`. Сетевые загрузки выполняются только явными скриптами, не при импорте/анализе.

## Выход анализа

- `analysis.json`: параметры записи, измерения по отведениям, оценки диагнозов, форма ФП как неизвестная без анамнеза, кандидаты ЖТ и происхождение моделей.
- `intervals_predicted.csv`: P/QRS/T, onset/peak/offset в локальных отсчётах, абсолютные координаты, секунды, длительность в мс и флаг QRS ≥120 мс.
- `intervals_reference.csv`: ручная разметка LUDB отдельно от предсказаний. Её отсутствие допустимо.
- `beats.npz`: полные биты фиксированной физической длительности, все исходные отведения, fs, единицы и координаты. Растяжение по времени не используется.
- `beat_classification.csv`: N/S/V/F/Q и модельные оценки. MLII проверен на MIT-BIH; перенос с MLII на другие отведения помечен как непроверенный.
- `delineation.png`: первые 10 секунд с ручной/предсказанной разметкой.
- Для длинной 12-канальной записи — `window_classification.json` по всем полным 10-секундным окнам; клинически откалиброванная агрегация всей записи не заявляется.

`--lead II`, `--start-seconds 30`, `--duration-seconds 10` задают просмотр/анализ фрагмента. Абсолютные координаты сохраняются. `--device cuda` ускоряет разметку; CPU поддерживается. Для произвольного CSV укажите `--csv-fs`; первый столбец должен содержать непрерывные нулевые sample indices. Для локального MIT-BIH калибровка автоматически сверяется с официальным заголовком. Другие CSV без gain/baseline остаются в ADC units.

Пакетный режим: CSV с колонкой `path`, затем `python -m ecg_project batch manifest.csv --output reports/batch --checkpoint artifacts/delineator_qt.pt`. Ошибки каждой записи фиксируются в `batch_status.json`.

## Воспроизводимое обучение

```powershell
python scripts/download_metadata.py
python -m ecg_project audit
python -m ecg_project benchmark-dwt
python -m ecg_project train-delineator --epochs 30 --minutes 25
python -m ecg_project adapt-delineator --epochs 12 --minutes 20
python scripts/download_qtdb.py
python -m ecg_project train-qt --epochs 10 --minutes 15
python -m ecg_project evaluate-qt --checkpoint artifacts/delineator_qt.pt --output reports/qtdb_adapted.json
python -m ecg_project evaluate-delineator --checkpoint artifacts/delineator_qt.pt --output reports/segmentation_qt_test.json
python -m ecg_project prepare-records --limit 500 --checkpoint artifacts/delineator_qt.pt --output artifacts/record_features_qt --device cuda
python -m ecg_project train-records --manifest artifacts/record_features_qt/manifest.csv --minutes 45
python -m ecg_project prepare-beats --checkpoint artifacts/delineator_qt.pt --output artifacts/beat_features_qt --device cuda
python -m ecg_project train-beats --root artifacts/beat_features_qt --minutes 40
python -m ecg_project train-beat-cnn --root artifacts/beat_features_qt --epochs 25 --minutes 20
python -m ecg_project evaluate-vt
python scripts/make_report.py
```

Указанный лимит — исследовательский пилот, а не весь датасет. Для полного извлечения признаков используйте `--limit 0` и новый output-каталог. Лимиты обучения суммарно 165 минут; обработка исходных данных может занимать существенно больше времени и не входит в обучение. Кеш защищён хешами сигнала, заголовка и разметчика. После изменения модели выбирайте новый output-каталог, не смешивайте старые признаки с новыми.

Три варианта классификации сохранены отдельно: `interval.joblib`, `waveform.joblib`, `fusion.joblib`. Waveform у классификатора битов — реальные отсчёты фиксированного окна; у классификатора записи — свёртки медианного и наиболее отличающегося бита по отведениям. Отчёты содержат AUROC/AUPRC и пороги по классам, confusion matrices, учёт ошибок детектора, совместную встречаемость меток train. `joblib`/PyTorch checkpoints следует загружать из созданных здесь доверенных артефактов.

Дополнительная ветвь `cnn_fusion.joblib` обучает компактный Conv1D encoder формы бита совместно с интервальными признаками. Imputation/scaling обучаются только на train, эпоха и выбор среди четырёх моделей — только на validation. Максимальный бюджет этой ветви 20 минут; её фактический результат сравнивается с бустингом в едином ноутбуке.

## Ноутбуки и структура

[04_all_pipelines_a100.ipynb](notebooks/04_all_pipelines_a100.ipynb) — единый ноутбук: аудит, обучение, графики разметки и аналитика классификации.

Код организован по областям ответственности: `data`, `processing`, `models`, `training`, `evaluation`, `experiments`, `workflows`. Подробная навигация: [docs/STRUCTURE.md](docs/STRUCTURE.md). Датасеты, веса и сгенерированные примеры хранятся локально и не входят в Git.

Проект различает наличие меток и их отсутствие. ЖТ, ЖЭС/НадЖЭС, ФП, АВ-блокады и внутрижелудочковые нарушения представлены в целевом mapping; редкие/неподтверждённые возможности явно отражены в отчёте, без выдуманных метрик или диагноза по одному QRS.
