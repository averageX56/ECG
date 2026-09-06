# Структура проекта

```text
pipelines/cpu/   локальная подготовка, --workers, HGB и инференс
pipelines/gpu/   A100-точки запуска BERT, HuBERT LoRA/QLoRA, Qwen
ecg_project/
  data/         чтение и аудит, splits
  processing/   фильтрация, features, ограниченный пул CPU workers
  models/       U-Net, CNN, BERT, HuBERT, Qwen-разметчик, LoRA
  training/     подготовка и обучение, checkpoint/resume
  evaluation/   границы, классы, эпизоды
  experiments/  сохранённые варианты сравнения
  workflows/    CLI-анализ и закреплённый маршрут
notebooks/      один общий notebook
scripts/        явные загрузки, preflight и отчёты
configs/        manifest выбора и накопленный локальный бюджет
docs/           инструкции и предложения
data/, LUDB/    исходные данные, не Git
artifacts/      сохранённые базы и будущие кэши/обученные модели, не Git
reports/        будущие результаты, не Git
```

Общие модули не перемещены из `ecg_project`, чтобы сохранять стабильные импорты моделей. Все запускаемые CPU/GPU стадии физически разделены в `pipelines`. [Запуск после очистки](FRESH_START.md).
