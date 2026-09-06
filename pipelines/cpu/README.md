# CPU: выполнить локально

Команды запускаются из корня репозитория. Этот launcher отключает CUDA до импорта PyTorch.

`--workers N` задаёт число процессов подготовки (по умолчанию до 4, `--workers 1` — последовательный режим). Обрабатываются независимые записи, порядок результатов сохраняется, на процесс выделяется один численный поток. Очередь ограничена 2×workers. Параллельны LUDB/Qwen, HuBERT, ECGFounder, MIT beats и record features. Аудит метаданных и единичный инференс не разбиваются этим флагом на процессы. Для HGB --workers задаёт число численных потоков одного процесса. Каждый worker разметочных features держит свою копию небольшого U-Net; подбирайте N по RAM и скорости диска.

```powershell
python -m pipelines.cpu.run audit
python -m pipelines.cpu.run prepare-beats --checkpoint artifacts/delineator_qt.pt --output artifacts/beat_features_qt
python -m pipelines.cpu.run prepare-records --checkpoint artifacts/delineator_qt.pt --output artifacts/record_features_full --limit 0
python -m pipelines.cpu.run prepare-hubert --workers 4
python -m pipelines.cpu.run prepare-qwen --workers 4
```

Выбирайте только нужные этапы. Для HuBERT достаточно аудита и `prepare-hubert`: LUDB/MIT features ему не нужны. Для Beat-BERT достаточно `prepare-beats`. Извлечение признаков использует инференс разметчика на CPU и может идти долго. После изменения весов разметчика нужен новый каталог признаков. Подготовка HuBERT пишет новый каталог, повторный запуск поверх готового запрещён.

На CPU также выполняются обучение HGB и финальный инференс:

```powershell
python -m pipelines.cpu.run train-records --manifest artifacts/record_features_full/manifest.csv --output artifacts/record_models_full --minutes 45
python -m pipelines.cpu.run train-beats --root artifacts/beat_features_qt --output artifacts/beat_models_new --minutes 40
python -m pipelines.cpu.run run LUDB/1.hea --output reports/final_example
python -m pipelines.cpu.run report
```

Обучение HGB тоже расходует локальный бюджет обучения; не запускайте все команды автоматически. `report` собирает имеющиеся результаты, не обучает модели. Отдельная опциональная подготовка ECGFounder: `python -m pipelines.cpu.run prepare-founder`.

Переносимые файлы и порядок работы: [../README.md](../README.md).
