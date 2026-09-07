# A100: только обучение из готовых caches

Открыть [единый notebook](../../notebooks/04_all_pipelines_a100.ipynb) в Jupyter из корня репозитория.
Настроить `RUN_DELINEATION`, `RUN_BERT`, `RUN_FOUNDER`, `RUN_HUBERT_LORA`, `RUN_QWEN`.
Текущий чистый Qwen profile включает `RUN_QWEN=True`, `RUN_QWEN_CURRICULUM=True`, E/1.7B и ratios1/2/4;
остальные branches выключены. `RUN_SMOKE=False` полностью пропускает smoke перед full run.

В Colab первая ячейка монтирует Drive, клонирует отсутствующий `/content/ECG` и подключает
`ECG_DATA/artifacts` и `reports` симлинками. Optional `ECG_DATA/data` и `LUDB` также подключаются.
Существующий непустой checkout/каталог не удаляется. Аргументы `subprocess.run` передаются списком.
Зависимости устанавливаются только в Colab; обычный Jupyter bootstrap пропускает.

GPU 1 обучает U-Net на LUDB + partial manual QTDB и optional Training_2 consistency.
Выбранный checkpoint: `artifacts/cluster/delineator_qt.pt`. Для Beat-BERT выполнить CPU 2.
Для Qwen C/D/E оставить checkpoint в Colab: `PREPARE_QWEN_PSEUDO_GPU=True` запускает отдельную
генерацию pseudo-cache на CUDA (default batch64). `PSEUDO_IO_WORKERS=8` читает/фильтрует сигналы;
CUDA-модель находится только в главном процессе. Готовые shards пропускаются при resume.
Student затем читает cache, без повторного teacher inference на каждой training iteration.

Qwen preparation: **Drive raw → local SSD → CPU preprocess → CUDA U-Net → local cache → Drive sync**.
Extended E: строго CPSC_EXTRA, PTBXL, CPSC, CHAPMAN; **без Ningbo**. Другие branches не меняются.
Отдельные progress bars: Protection registry, Pseudo input preprocessing, GPU inference/write, Final sync to Drive.
Raw symlinks временно указывают на SSD и восстанавливаются после подготовки. Кэш публикуется на Drive
только после завершения и проверки. Ранее простаивающий GPU ожидал Drive I/O/preprocessing и полный
batch; это не свидетельствовало о проблеме CUDA. Численное ускорение на A100 ещё нужно измерить.
Snapshot reuse, strict verification и interruption: [подробности](../../docs/QWEN_PREPARATION_PERFORMANCE.md).

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
