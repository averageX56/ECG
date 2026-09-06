# Разделение CPU / GPU

| Папка | Где запускать | Работа |
|---|---|---|
| [cpu](cpu/README.md) | Локальный CPU | Аудит, фильтрация, ресэмплинг, нарезка, features, HGB, финальный инференс, отчёты |
| [gpu](gpu/README.md) | Одна A100 | Обучение нейросетей из готовых кэшей |

Общие реализации остаются в `ecg_project`: перенос внутренних модулей сломал бы импорты сохранённых моделей. Физически разделены запускаемые этапы; единственный ноутбук управляет ими.

## Что перенести на кластер

Для **Qwen QLoRA-разметчика**: локально `python -m pipelines.cpu.run prepare-qwen --workers 4` и `# Qwen downloads on the cluster from the unified notebook`; перенести `artifacts/qwen_delineation_inputs/` и `artifacts/qwen3_4b/`. LUDB нужна дополнительно для графиков и сравнения границ. Для локального smoke используется размер 0.6B. См. [описание](../docs/LORA.md).

- Всегда: репозиторий и необходимые зависимости. Запускайте из корня, сохраняя относительные пути.
- **HuBERT LoRA:** целиком `artifacts/hubert_inputs_full/` (signals.npy, manifest.csv, provenance.json) и `artifacts/hubert_large/` (веса, config, provenance). Исходный PTB-XL после подготовки этой ветке не нужен. Скачивание весов выполните локально: `python -u scripts/download_hubert.py --weights`. Для локальной подготовки нужен `data/WFDB_PTB-XL/` и каталог после `audit`; используются только folds1–9.
- **Beat-BERT:** `artifacts/beat_features_qt/` целиком. Для SSL дополнительно подготовленный `artifacts/unlabeled_beats/`. Исходный MIT-BIH для обучения из кэша не нужен.
- **Разметчик:** `LUDB/`, `data/qtdb_external/`, `data/Training_2/`, `artifacts/catalog.csv` и необходимые исходные checkpoints. Здесь исходные записи действительно нужны.
- **ECGFounder, опционально:** `artifacts/founder_full_inputs/` и `artifacts/ecgfounder/`.
- **Графики и готовый финальный анализ:** четыре артефакта из `configs/final_pipeline.json` и нужные исходные ЭКГ (например, файлы записи LUDB/1). Для сравнительных таблиц также `reports/`, `artifacts/representation_experiments/` и `artifacts/beat_features_qt/`.

После обучения верните run-каталоги целиком на CPU для аналитики. Новые веса не заменяют выбранный финальный пайплайн автоматически: требуется validation, совместимое повторное извлечение признаков и новый manifest. Полное переобучение разметчика образует цикл GPU → CPU features → GPU Beat-BERT; старый кэш после него использовать нельзя.

Уточнение запуска: Qwen локально не скачивается. Загрузка перенесена в GPU-ячейку единого notebook, см. [FRESH_START](../docs/FRESH_START.md). Реальный Qwen smoke на предобученных весах локально не выполнялся.
