# Qwen pseudo preparation: Colab SSD

Точечное изменение I/O пути, не архитектур или objective student.

```text
Drive raw → local SSD → CPU preprocess → CUDA U-Net → local cache → Drive sync
```

Extended E: CPSC_EXTRA, PTBXL, CPSC, CHAPMAN. Ningbo исключён только из Qwen profile.
Staging: data/Training_2, Training_WFDB, WFDB_PTB-XL, WFDB_ChapmanShaoxing, qtdb_external, LUDB.
Catalog добавляет обязательные holdouts (обычно Georgia/St Petersburg). Их identities и exact hashes
по-прежнему исключают leakage; evaluation labels не используются.

Default scratch `/content/ecg_qwen`: raw/, caches/qwen_pseudo_extended/, protected_identities/.
`stage_qwen_raw` делает первоначальную копию с SHA256 и сохраняет ownership/completion manifests.
Полная локальная копия повторно не читается с Drive: проверяются список файлов, size/mtime.
Это повторное использование **того же immutable snapshot**, а не автоматическое обновление Drive данных.
`verify_sources=True` / `VERIFY_QWEN_SOURCES=True` проверяет полные Drive/local hashes и source hashes
при resume. Изменения содержимого с сохранением size/mtime обнаруживаются в этом строгом режиме.
При изменении датасетов используйте новый scratch; старые/чужие непустые каталоги не удаляются.

На время generation symlinks data/LUDB переключаются на SSD, затем восстанавливаются в finally.
Attach Drive для остальных branches не меняется. Teacher читается один раз; CUDA только в parent,
torch.inference_mode, batch64. Spawn workers8 выполняют чтение и preprocessing, без модели.
Protection registry и псевдо-shards пишутся на SSD. Tqdm показывает registry/preprocess/inference/sync
отдельно; preprocessing виден до первого batch. Batch остаётся configurable.

Resume сначала проверяет metadata, teacher hash, tau, version, source/patient identity, source stat
и SHA256 готового shard. Затем использует сохранённый signal hash для leakage check, без load_record,
preprocessing или повторного decoded-signal hash. Совместимые старые metadata обновляются после
однократной проверки исходных SHA256 и исходного cache_identity.json, без WFDB decoding.
Изменённый teacher/cohort/tau по-прежнему отклоняется. Для нового record raw bytes хешируются один раз, WFDB decoding выполняется один раз;
этот rec используется и для signal hash, и для preprocessing. Полное устранение raw reads для первого
прохода не заявляется: file SHA256 и WFDB decoder читают bytes отдельно, но с local SSD.

После publish проверяется local provenance. Затем одна финальная фаза copy в `<output>.syncing`
на Drive, verification и переименование в публичный output. Незавершённый local cache не публикуется.
Прерванная final sync может быть повторена. Существующий другой Drive cache не перезаписывается.
Это по-прежнему много файлов при final sync; они больше не тормозят CUDA hot loop.
Архивный storage format не вводится, CPU/GPU NPZ schema совместима.

Periodic sync по умолчанию отсутствует ради скорости. Прерывание ячейки сохраняет local resume;
сброс Colab runtime теряет незавершённый scratch. Завершённые artifacts/results остаются на Drive.
Сохранённые source manifests и SHA256 обеспечивают audit, fast stat checks не заменяют strict rehash
для недоверенных/изменяемых snapshots.

Notebook настроен на чистый Qwen E/1.7B, curriculum1/2/4, остальные branches выключены.
Если teacher готов — RUN_DELINEATION=False; иначе включите его. Итоговый порядок:
mount → clone/update → dependencies → stage raw → optional teacher → pseudo generation → sync → Qwen.
Низкая GPU utilization в прежнем пути объяснялась ожиданием Drive I/O/preprocessing и batch;
это не доказательство сбоя CUDA. A100 throughput после изменения ещё нужно измерить.
