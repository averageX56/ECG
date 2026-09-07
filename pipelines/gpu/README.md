# A100: только обучение из готовых caches

Открыть [единый notebook](../../notebooks/04_all_pipelines_a100.ipynb) в Jupyter из корня репозитория.
Настроить `RUN_DELINEATION`, `RUN_BERT`, `RUN_FOUNDER`, `RUN_HUBERT_LORA`, `RUN_QWEN`.
Все false по умолчанию; `RUN_SMOKE=False` полностью пропускает smoke перед full run.

GPU 1 обучает U-Net на LUDB + partial manual QTDB и optional Training_2 consistency.
Выбранный checkpoint: `artifacts/cluster/delineator_qt.pt`. Вернуть его локально, выполнить CPU 2,
перенести новые caches и запускать GPU 2. Notebook не генерирует teacher predictions on-the-fly.

Qwen experiments: A LUDB, B LUDB+QT, C +Training_2 pseudo hard, D +extended pseudo hard,
E +soft KD. Выбор `QWEN_EXPERIMENTS`, размеры `QWEN_SIZES=['1.7B']` или `['1.7B','4B']`.
Загрузка base Qwen происходит только в явно включённой кластерной ветке.
JSON configs находятся в `configs/qwen_v2`; factory — `pipelines.gpu.experiments.qwen_experiment`.
`RUN_QWEN_CURRICULUM=True` включает последовательные distilled циклы с `PSEUDO_RATIOS=[1,2,4]`:
warm start последней принятой модели, validation gate, остановка на plateau. Детали и пороги
в [teacher–student protocol](../../docs/QWEN_TEACHER_STUDENT.md).

Начальные Qwen microbatch: 16 на 40 GB, 32 на 80 GB, accumulation 4/2.
Effective manual batch 64; pseudo batches добавляются с контролируемой долей.
Это стартовые профили, а не обещание максимальной утилизации или отсутствия OOM для любой архитектуры.
Для 4B разрешено уменьшить batch и увеличить accumulation в новом run. Beat-BERT использует auto-batch probe.

Доступны BERT supervised (`pretrain_epochs=0`), MIT masked pretrain и extended SSL с явным списком roots.
Founder full fine-tuning и HuBERT frozen/LoRA/QLoRA используют один record source policy.
HuBERT по умолчанию microbatch32/64, Founder64/128; измеряйте throughput и peak memory.

Early stopping: U-Net/Qwen mean validation P/QRS/T Dice, Beat-BERT supported NSVF selection F1,
Founder/HuBERT macro AUROC. Patience9 в notebook. Resume восстанавливает optimizer/RNG/stale counter;
изменение конфигурации или cache требует нового run. `latest.pt`, `best.pt`, `history.json`,
`best_metrics.json`, `run.json` и validation predictions сохраняются отдельно.

`run.json`: actual/effective batch, PyTorch peak allocated/reserved memory, throughput, epochs,
best epoch, complete/interrupted. Данные nvidia-smi не используются как единственная оценка памяти.
Ни один test/external cohort не используется для выбора checkpoint.
