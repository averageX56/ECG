# Запуск после очистки результатов

По запросу удалены производные кэши, обученные checkpoints/адаптеры и отчёты прогонов. Исходные `data/`, `LUDB/`, официальные метаданные и скачанные базы HuBERT/ECGFounder сохраняются. Незавершённая локальная загрузка Qwen остановлена; Qwen скачивается **на кластере из notebook**. Исторический manifest финального пайплайна не заменяет отсутствующие обученные веса.

## Локальный CPU

Из корня проекта, после установки `python -m pip install -e ".[dev]"`:

```powershell
python -m pipelines.cpu.run audit
python -m pipelines.cpu.run prepare-qwen --workers 4
python -m pipelines.cpu.run prepare-hubert --workers 4
```

Эти ветви независимы: Qwen использует LUDB train/valid, HuBERT — PTB folds1–9. Перенесите выбранные кэши на кластер. `--workers 1` отключает процессы; начальный выбор — до четырёх. Для Founder: `prepare-founder --workers 4`. Для повторения MIT/record features сначала необходим обученный разметчик.

## Кластер / единый Jupyter notebook

Откройте `notebooks/04_all_pipelines_a100.ipynb`, установите `.[dev,qlora]` с CUDA-сборкой PyTorch. По умолчанию обучение выключено и отсутствующие результаты пропускаются.

- **Qwen-разметка:** перенесите `artifacts/qwen_delineation_inputs/`, для графиков также LUDB. Включите `RUN_QWEN=True`: ячейка скачает Qwen3-4B на кластер, выполнит короткий smoke, затем полное обучение. Скачать Qwen локально не требуется.
- **HuBERT frozen/LoRA/QLoRA:** перенесите `artifacts/hubert_inputs_full/` и сохранённый `artifacts/hubert_large/`; включите `RUN_HUBERT_LORA=True`.
- **U-Net и адаптация:** перенесите LUDB, QT, Training_2 и каталог, включите `RUN_DELINEATION=True`. Новые веса сохраняются в `artifacts/cluster/`, чтобы не подменять исторические модели.
- **Beat-BERT и HGB:** после обучения разметчика верните его на CPU, выполните `prepare-beats`/`prepare-records` с явным checkpoint и свежим output, затем перенесите features для GPU-BERT. HGB можно обучить локально.

После изменения разметчика нельзя использовать старые features. Финальный manifest пересоздаётся только после validation новых моделей; их SHA256 не будут совпадать с удалёнными историческими весами.

## Сохранённые варианты сравнения

| Задача | Реализации |
|---|---|
| Разметка | DWT benchmark, U-Net, consistency adaptation, QT adaptation, Qwen QLoRA |
| Beat-классификация | HGB interval/waveform/fusion, CNN, RR-context, scratch/masked Beat-BERT, latent features, большой cluster BERT |
| Record-классификация | HGB interval/waveform/fusion, frozen/fine-tuned ECGFounder, HuBERT frozen/LoRA/QLoRA |
| Общая teacher/student модель | Пока [предложение](NEXT_EXPERIMENTS.md), не реализация |

Общие модули `ecg_project/experiments`, `training`, `models` сохранены. Очистка затрагивает результаты, а не варианты сравнения. Новые датасеты из предложения не скачивались и ожидают утверждения. Счётчик ранее потраченного локального бюджета сохранён отдельно в `configs/local_budget.json`; очистка не обнуляет лимит 180 минут.

В едином notebook `QWEN_SIZES=["1.7B","4B"]`: обе модели скачиваются только на кластере и обучаются в отдельных run-каталогах на одинаковом LUDB train/valid. Размеры 0.6B и 8B можно явно добавить в список.
