# ECG pipelines v2

Один управляющий notebook: [04_all_pipelines_a100.ipynb](../notebooks/04_all_pipelines_a100.ipynb).
Текущий notebook настроен на Qwen E/1.7B с curriculum; остальные GPU-ветки выключены.
[Полный запуск](../docs/FRESH_START.md).

```text
CPU 1: metadata/audit → manual delineation cache + Founder/HuBERT inputs
GPU 1: U-Net teacher → artifacts/cluster/delineator_qt.pt
CPU 2: beat features + unlabeled Beat-BERT features + record features
GPU 2: Qwen pseudo logits (batched U-Net, Colab) → Qwen supervised/distilled
       Beat-BERT / Founder / HuBERT LoRA, QLoRA
CPU:   HGB, inference, plots, analysis bundle
```

Базовая подготовка выполняется в [cpu](cpu/README.md); обучение — в [gpu](gpu/README.md).
Qwen extended pseudo-cache по последнему требованию строится прямо на GPU в Colab после teacher,
с `PREPARE_QWEN_PSEUDO_GPU=True`. Возвращать teacher на локальный CPU для этой ветки не нужно.
Общие реализации находятся в `ecg_project/data`, `processing`, `models`, `training`, `evaluation`.
Baseline trainers сохранены; новые runs не заменяют исторический final selection автоматически.

Canonical teacher передаётся явно через `--checkpoint artifacts/cluster/delineator_qt.pt`.
Checkpoint между папками не копируется молча. Новый teacher требует нового CPU cache.
Кэши содержат SHA256, preprocessing version, manifest/provenance; смена исходных данных,
checkpoint или конфигурации отклоняет reuse. Дорогие stages продолжаются по готовым shards.

Qwen A/B обучаются только на ручной разметке. C/D/E дополнительно получают supervision от U-Net.
Это разные бюджеты supervision: сравнивать их как одинаково обученные модели нельзя.
Протокол, формула loss и ограничения: [Qwen teacher–student](../docs/QWEN_TEACHER_STUDENT.md).
