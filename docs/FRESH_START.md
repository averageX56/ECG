# Запуск ECG v2 с нуля

В репозитории сохранены базы и исходные datasets. Старые trained results ранее удалены по запросу;
новая подготовка не меняет существующие checkpoints/results. Исторический final pipeline — baseline,
а не результат уже выполненного v2 обучения. A100 и pretrained Qwen v2 пока не измерены.

## Порядок

```text
CPU 1 (локально)
  audit / metadata / dataset manifest
  manual LUDB/QTDB inputs, optional train-only consistency inputs
  Founder inputs, HuBERT inputs
GPU 1 (Jupyter A100)
  train U-Net → artifacts/cluster/delineator_qt.pt
CPU 2 (локально, с новым checkpoint)
  prepare-beats, prepare-unlabeled
  prepare-records → HGB (optional)
GPU 2 (Jupyter A100)
  Qwen pseudo-cache: batched CUDA U-Net in Colab, persistent Drive output
  Beat-BERT, Founder, HuBERT LoRA/QLoRA, Qwen supervised/distilled
CPU / notebook analysis
  validation plots and reports; fixed final test protocol; analysis bundle
```

Команды: [CPU README](../pipelines/cpu/README.md). Notebook:
[`04_all_pipelines_a100.ipynb`](../notebooks/04_all_pipelines_a100.ipynb).
Установить зависимости проекта согласно pyproject.toml в отдельное окружение и выбрать его Jupyter kernel.
Запускать из корня репозитория. Никаких больших datasets notebook автоматически не скачивает.

**Colab:** первая ячейка подключает `/content/drive/MyDrive/ECG_DATA`, репозиторий — `/content/ECG`.
Artifacts/reports связываются с Drive и переживают смену сессии. Qwen extended больше не требует
локального CPU 2: `PREPARE_QWEN_PSEUDO_GPU=True` генерирует teacher logits непосредственно на GPU
перед student training. Для этого на Drive должны быть исходные `data/`, `LUDB/` и audited catalog.
Если там только artifacts/reports, нужно добавить raw signals или готовый pseudo cache.
Это требование к входам teacher, а не необходимость запускать подготовку на локальном компьютере.

## Датасеты

| Ветка | Источники / локальные папки |
|---|---|
| U-Net, Qwen manual | `LUDB/`; `data/qtdb_external/` с `split.json`, `.q1c` |
| U-Net consistency, Qwen C | `data/Training_2/` = CPSC_EXTRA |
| Beat-BERT supervised | `data/mit-bih/` + `artifacts/mit_headers/`; optional `data/svdb/`, `data/incart/` |
| Record models, HGB | `data/WFDB_PTB-XL/`, `Training_WFDB/`, `Training_2/`, `WFDB_PTB/`, `WFDB_ChapmanShaoxing/`, `WFDB_Ningbo/` |
| Extended SSL, Qwen D/E | train-only PTBXL, CPSC, CPSC_EXTRA, CHAPMAN, NINGBO |
| External protection/evaluation | `data/WFDB_GEORGIA/`, `data/Training_StPetersburg/` |

SVDB/INCART поддержаны source adapters. Загрузка — только отдельная явная команда:
`python scripts/download_beat_datasets.py --source svdb` или `--source incart` (проверьте `--help`).
Остальные новые sources нужно положить в указанные папки в PhysioNet Challenge-style WFDB формате
и заново выполнить audit. Для отсутствующего явно указанного source подготовка завершается ошибкой.
Расширенные datasets ещё не скачивались автоматически.

PTB-XL: folds1–8 train,9 valid,10 test. CPSC/extra/PTB/Chapman/Ningbo — весь source train.
Georgia external, St Petersburg external_long. При неизвестных patient identities нет случайного
record/fragment split. Это не доказывает отсутствие неизвестных cross-source пациентов;
доступные patient IDs и exact signal hashes дополнительно проверяются для pseudo/SSL.
MIT historical split сохранён, 201/202 не разделяются. SVDB/INCART — entire train, optional patients.csv.
INCART/St Petersburg могут иметь общее происхождение: независимый beat external после INCART training не заявляется.

QT: только manual `.q1c`; `.pu/.pu0/.pu1` запрещены как supervised ground truth.
Из прежних 8 adaptation records фиксированно6 train/2 valid, прежние10 external остаются untouched.
Исторический QT baseline уже использовал все8 adaptation records; новый QT-valid не является
независимым от этого старого baseline. Неизвестные samples имеют -100; отсутствие T onset не выдумывается.
Для A/B/C/D/E одинаковый validation cache protocol через `--validation-datasets LUDB QTDB`.

Record labels пересчитываются из общих SNOMED TARGETS; несовпадение catalog mapping — ошибка.
Q сохраняется в AAMI dataset; BERT явно сообщает поддержку и исключение Q из selection,
опция `include_q=False` сохраняет original support report. Редкий Q не объявляется надёжно обученным.

## Что переносить на кластер

```text
artifacts/
  catalog.csv
  delineator_inputs_v2/          # GPU1; manual + optional consistency
  qwen_ludb_inputs_v2/           # A; same validation as B
  qwen_delineation_inputs_v2/    # B–E
  founder_inputs_v2/            # signals.npy + manifest.csv + provenance.json
  hubert_inputs_v2/             # signals.npy + manifest.csv + provenance.json
  ecgfounder/                   # preserved pretrained base
  hubert_large/                 # preserved pretrained base
  beat_features_v2/             # after GPU1 → CPU2
  unlabeled_beats_v2/           # after GPU1 → CPU2
  qwen_pseudo_training2/        # C; generated on Colab GPU, saved to Drive
  qwen_pseudo_extended/         # D/E; generated on Colab GPU, saved to Drive
  record_features_v2/           # optional CPU HGB
  cluster/
    delineator/                 # latest/best/history/metrics
    delineator_qt.pt            # canonical selected teacher
    bert_*/ founder_*/ hubert_*/ qwen_*/
```

GPU training потребляет готовые caches. Для разрешённой GPU-генерации Qwen pseudo cache
notebook также читает исходные ECG из Drive; повторного копирования teacher на локальную машину нет.
Qwen base 1.7B/4B скачивается самим кластером в явно включённой ветке.
Для identity-only exclusion registry нужны holdout raw signals в месте подготовки pseudo cache
(в Colab — Drive/data и Drive/LUDB); labels не используются для tuning.
После изменения teacher/data/preprocessing cache stale: задайте новый output, не подменяйте provenance.

HuBERT v2: canonical12leads, deterministic first10sec, <10sec исключаются, zero padding нет.
Founder v2: leadI,500Hz, notch50Hz, bandpass0.67–40Hz, median baseline, normalization.
Manifest содержит original duration, окна, исключения. Для neural/HGB сравнивайте также пересечение
доступных records: требования к leads/duration различаются. HuBERT pretrained источники могут
пересекаться с PTB/CPSC/Georgia: независимость pretraining не заявляется.

## Анализ и возобновление

Результаты сравнивать по validation и supervision regime. После фиксации selection оценивать test один раз.
В notebook есть таблица runs и графики raw cached ECG, partial manual labels, predictions.
`python scripts/make_analysis_bundle.py --output analysis_v2.zip` собирает маленькие reports,
configs/history/run/best metrics, provenance/manifests и validation predictions. Weights, raw datasets,
signals.npy и полный feature cache исключены. Большие metadata-файлы вызывают явную ошибку лимита.

Resume — повтор конфигурации в тот же output: stale counter сохраняется. Изменение конфигурации
отклоняется до перезаписи run. При OOM profile менять в новом run. Patience9 ограничивает plateau;
кластерного лимита времени нет. Локальный training budget по-прежнему хранится в configs/local_budget.json.

Teacher–student детали: [QWEN_TEACHER_STUDENT](QWEN_TEACHER_STUDENT.md).
Клинические ограничения baseline (AF form/history, редкие AVB/VT, uncertainty gain) сохраняются.
