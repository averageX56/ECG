# Проверки v2, 2026-09-07

- Полный `python -m pytest -q`: 57 passed, 1 skipped. Skip относится к optional trained
  checkpoint, отсутствующему после ранее согласованной очистки.
- Проверены split policy, AAMI mapping, cache/provenance SHA256 и stale rejection,
  immediate missing checkpoint, partial QT masks, duration policy, hard/soft losses,
  pseudo worker, warm start/resume, curriculum rollback и analysis bundle exclusions.
- Общий notebook исполнен с выключенными training flags: загрузок Qwen и GPU обучения не было.
  Для dry-run flags отключены только в памяти: сохранённый notebook настроен на Qwen E/1.7B.
- Performance-refactor: проверены staging/reuse без удаления чужих dirs, обязательные holdout sources,
  исключение Ningbo из Qwen, fast resume без raw reads, strict SHA256 verification, прерванный final sync
  и validation завершённого Drive cache. Фильтрация одного выбранного lead совпадает с прежним input
  с atol1e-7. A100/Drive throughput не измерялся локально.
- Дополнительно проверен batched pseudo-inference executor на CPU и CUDA (маленькая тестовая модель),
  отсутствие teacher forward при cache resume и сохранение существующей базы при Colab attach.
  Сам Google Colab/Drive runtime и полный GPU U-Net pseudo pool ещё не запускались.
- Реальный audit: LUDB200, QT18, Training_2 3453 (3452 readable), CPSC6877,
  St Petersburg74, Georgia10344, PTB-XL21837.
- Реальная CPU подготовка manual cache с workers2: LUDB train1680/valid360 lead-windows,
  QTDB train70/valid24; всего1750 train и384 valid, shape1×2500 при250Hz.
  Это проверка подготовки, не метрика качества нейросети.

Не выполнены: полное обучение v2 на A100, pretrained Qwen training, полный teacher pseudo cache
на расширенном pool и внешняя клиническая оценка. Новые большие datasets автоматически не скачивались.
Синтетические tests проверяют исполнение и защиту от ошибок, не превосходство моделей.

После проверок CPU/GPU outputs очищаются по запросу пользователя. Датасеты, pretrained базы
и исходные metadata сохраняются; результаты перечисленных smoke checks не используются для final model selection.
# JupyterLab cluster migration validation (2026-09-07)

`python -m pytest -q`: **65 passed, 1 skipped** (optional checkpoint unavailable).
Notebook schema and all code-cell syntax validated; setup/training cells dry-run with GPU branches disabled in memory.
`bash -n scripts/setup_cluster.sh` passed. Existing data/artifacts were not cleaned, moved or relinked.
Cluster tests cover directory preservation, missing teacher, teacher mismatch, interrupted scratch publication,
persistent partial resume and the exact Qwen source set. Existing leakage/hash/schema tests remain passing.
Linux dependency installation and a full A100/Qwen training run were not executed on this Windows host.
The selected torch/transformers/bitsandbytes versions match the existing local software versions apart from the CUDA wheel;
cluster NF4 forward/backward is checked by the notebook before model download/training.
