# GPU: выполнить на кластере

Откройте единственный [ноутбук](../../notebooks/04_all_pipelines_a100.ipynb) из корня проекта. Подготовьте кэши локально по инструкции [CPU](../cpu/README.md), перенесите их до запуска GPU-сессии. Ветка HuBERT больше не запускает подготовку или скачивание внутри обучения.

```python
from pipelines.gpu.run import train_hubert
runs = train_hubert()  # frozen head, LoRA rank16, QLoRA NF4 rank16
from pipelines.gpu.run import train_qwen
qwen_run = train_qwen()  # отдельная задача разметки P/QRS/T
```

CLI-эквивалент: `python -m pipelines.gpu.run --epochs 30`.

Начальные microbatch: 32 на 40 ГБ, 64 на 80 ГБ; две 5-секундные проекции на запись, accumulation=2. BF16, TF32, gradient checkpointing; сохранение адаптеров и возобновление с завершённой эпохи. Это стартовые настройки, а не измеренный максимум A100: контролируйте память и throughput, меняйте batch в `LoRAConfig` для нового run. Нельзя обещать максимальную утилизацию без запуска на вашей карте.

Beat-BERT принимает подготовленные MIT features через `train_bert(config)`; размеры и автоматический подбор batch задаёт ноутбук. Обучение разметчика требует исходных LUDB/QT/Training_2, поскольку разметка и аугментации читаются при обучении. Его нельзя заменить кэшем HuBERT.
