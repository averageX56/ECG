# Локальная CPU-часть

Запускать из корня репозитория. Все команды подготовки поддерживают общий `--workers N`;
используется bounded spawn multiprocessing, один Torch/BLAS thread на worker.
До запуска workers проверяется существование зависимого checkpoint.

```bash
python -m pipelines.cpu.run audit --workers 4
python -m pipelines.cpu.run prepare-record-datasets --workers 4
python -m pipelines.cpu.run prepare-qwen --datasets LUDB QTDB --validation-datasets LUDB QTDB --workers 4
python -m pipelines.cpu.run prepare-hubert --workers 4
python -m pipelines.cpu.run prepare-founder --workers 4
```

`--sources` задаёт явный список record sources. Без списка record preparation использует
доступные источники с известной политикой; явно запрошенный отсутствующий source — ошибка.
Полный pool: `PTBXL CPSC CPSC_EXTRA PTB CHAPMAN NINGBO`. Georgia/STP — только external.
HuBERT требует 12 canonical leads; Founder — lead I. Короткие записи исключаются;
длинные дают фиксированные первые 10 секунд. `--train-windows N` разрешает несколько
детерминированных train windows; valid всегда одна. Reasons записываются в exclusions.json.

После GPU 1 вернуть новый U-Net checkpoint локально:

Для Beat-BERT/HGB. **Qwen C/D/E готовит pseudo-cache прямо на GPU кластера в JupyterLab**;
приведённые ниже `prepare-qwen-pseudo` команды сохранены только как optional CPU fallback.

```bash
python -m pipelines.cpu.run prepare-beats --sources MIT SVDB INCART --checkpoint artifacts/cluster/delineator_qt.pt --workers 4
python -m pipelines.cpu.run prepare-unlabeled --sources CPSC_EXTRA PTBXL CPSC CHAPMAN NINGBO --checkpoint artifacts/cluster/delineator_qt.pt --workers 4
python -m pipelines.cpu.run prepare-qwen-pseudo --sources CPSC_EXTRA --checkpoint artifacts/cluster/delineator_qt.pt --output artifacts/qwen_pseudo_training2 --tau 0.95 --workers 4
python -m pipelines.cpu.run prepare-qwen-pseudo --sources CPSC_EXTRA PTBXL CPSC CHAPMAN --checkpoint artifacts/cluster/delineator_qt.pt --output artifacts/qwen_pseudo_extended --tau 0.95 --workers 4
python -m pipelines.cpu.run prepare-records --checkpoint artifacts/cluster/delineator_qt.pt --workers 4
python -m pipelines.cpu.run train-records --manifest artifacts/record_features_v2/manifest.csv --workers 4
```

Для MIT-only задайте `--sources MIT --output artifacts/beat_mit_v2`.
Для LUDB-only Qwen: `prepare-qwen --datasets LUDB --validation-datasets LUDB QTDB --output artifacts/qwen_ludb_inputs_v2`.
Для U-Net consistency: `prepare-qwen --datasets LUDB QTDB --validation-datasets LUDB QTDB --unlabeled-sources CPSC_EXTRA --output artifacts/delineator_inputs_v2`.

Сохраните весь законченный cache, включая provenance. Повтор той же команды продолжает
готовые shards, но не разрешает смену данных/модели. Для новой версии используйте другой `--output`.
Pseudo stage строит отдельный resumable registry защищённых identities: `artifacts/protected_identities_v2`.
Exact signal duplicates защищённых cohorts исключаются; совпадение patient/record — hard error.
Это проверка identities, не использование holdout labels для обучения.

Датасеты и pretrained базы не удаляются. Qwen локально не скачивается.
Подробные пути и перенос на кластер: [FRESH_START](../../docs/FRESH_START.md).
