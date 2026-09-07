# A100: только обучение из готовых caches

Открыть [единый notebook](../../notebooks/04_all_pipelines_a100.ipynb) в Jupyter из корня репозитория.
Настроить `RUN_DELINEATION`, `RUN_BERT`, `RUN_FOUNDER`, `RUN_HUBERT_LORA`, `RUN_QWEN`.
Текущий чистый Qwen profile включает `RUN_QWEN=True`, `RUN_QWEN_CURRICULUM=True`, E/1.7B и ratios1/2/4;
остальные branches выключены. `RUN_SMOKE=False` полностью пропускает smoke перед full run.

Запуск: [окружение JupyterLab / A100](../../docs/CLUSTER.md).
Notebook использует существующий checkout, data, LUDB и artifacts; не монтирует Drive, не удаляет каталоги,
не обновляет git и не устанавливает зависимости в работающий kernel.

GPU 1: optional U-Net → `artifacts/cluster/delineator_qt.pt`.
Qwen C/D/E: existing raw → CPU workers → batched CUDA teacher → persistent artifacts cache.
Batch64, workers8; CUDA только в parent. Optional `ECG_SCRATCH` переносит только запись нового cache на SSD,
после завершения выполняется проверенная синхронизация. Исходные каталоги остаются на своих местах.
Extended E: CPSC_EXTRA, PTBXL, CPSC, CHAPMAN; без Ningbo. Registry сохраняет все проверки holdout overlap.
Resume проверяет готовые shards до WFDB decoding. Старые результаты не очищаются.
Подробности: [performance](../../docs/QWEN_PREPARATION_PERFORMANCE.md).

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
