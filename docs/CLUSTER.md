# A100 40/80 GB: Linux-кластер и JupyterLab

Используйте существующий checkout ECG, `data/`, `LUDB/`, `artifacts/` и `reports/`.
Ничего из них не удаляется, не перемещается и не заменяется ссылками. Существующие symlinks допустимы.
Все относительные пути считаются от корня репозитория; catalog должен содержать доступные на кластере пути.

## Окружение

Профиль: Linux x86_64, Python 3.11, одна выделенная A100. В терминале JupyterLab из корня ECG:

```bash
bash scripts/setup_cluster.sh
```

Скрипт создаёт новое изолированное `$HOME/.venvs/ecg-a100`, ставит CUDA PyTorch и зависимости,
проверяет `pip check`, регистрирует kernel **ECG A100 (Python 3.11)**.
Существующее окружение не изменяет: при повторном запуске останавливается. Для другой установки:

```bash
ECG_ENV="$HOME/.venvs/ecg-a100-v2" ECG_KERNEL=ecg-a100-v2 bash scripts/setup_cluster.sh
```

Выберите этот kernel в `notebooks/04_all_pipelines_a100.ipynb`. После установки перезапустите kernel.
JupyterLab-сервер может оставаться окружением администратора, но kernel должен запускаться на GPU allocation,
а не на login node. Способ получения allocation зависит от scheduler кластера.
Для самостоятельного сервера: `$HOME/.venvs/ecg-a100/bin/jupyter lab --no-browser`.

`requirements/cluster-a100.txt`: torch 2.7.1 (CUDA 12.6 wheel), transformers 5.8.1,
bitsandbytes 0.49.2, numpy 2.1.3, WFDB 4.3.1, NeuroKit2 0.2.13.
Остальные версии ограничены диапазонами; resolved environment сохраняется внутри venv в `requirements-resolved.txt`.
Это профиль установки, а не уже проверенный A100 training run. Notebook проверяет CUDA, VRAM,
импорт Qwen3 и маленький NF4 forward/backward без загрузки весов.
Flash-attn, DeepSpeed, PEFT и torchaudio не нужны: используются SDPA и собственные LoRA/NF4 modules.

CUDA wheel требует совместимого драйвера хоста. Если CUDA check не проходит, согласуйте runtime с администратором.
Не переустанавливайте драйвер из notebook. Источники для выбора wheels:
[PyTorch versions](https://pytorch.org/get-started/previous-versions/),
[bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation).

## Пути и запуск

Notebook определяет корень из текущей директории или `ECG_ROOT`; допустим запуск из `notebooks/`.
Нет `/content`, Google Drive, автоматического clone/pull или pip в kernel.
Caches проходят provenance checks; несовместимые inputs требуют нового output.
Включены только Qwen E/1.7B, CUDA pseudo preparation и curriculum `[1,2,4]`.
Teacher существует: `RUN_DELINEATION=False`. `RUN_SMOKE=False` пропускает тренировочный smoke;
короткая проверка окружения остаётся обязательной.

Порядок: CPU 1 (manual/record caches) → optional GPU U-Net → CPU 2 для Beat-BERT/HGB
и CUDA Qwen pseudo preparation → выбранные GPU branches → notebook analysis.
CPU-команды выполняются в том же окружении на CPU allocation. Возвращать данные на ноутбук не нужно.

Qwen pseudo: `data/Training_2`, `data/Training_WFDB`, `data/WFDB_PTB-XL`, `data/WFDB_ChapmanShaoxing`;
также `LUDB`, `data/qtdb_external` и protected holdouts из catalog. Ningbo не входит в Qwen E.
Полная таблица веток: [FRESH_START](FRESH_START.md).

По умолчанию raw читается на месте, новые shards пишутся в persistent artifacts.
Для медленного сетевого FS задайте `QWEN_SCRATCH='/allocated/local/ssd/ecg-pseudo'` в notebook.
Тогда новый cache записывается на SSD, валидируется и синхронизируется в artifacts после завершения.
Raw не копируется автоматически. Незавершённый persistent cache продолжается на месте.
Scratch может исчезнуть после allocation; незавершённая подготовка не публикуется как complete.
Автоматической очистки нет. Изменённый config требует нового output; старые эксперименты сохраняйте.

Проверка inputs: `python scripts/preflight.py --profile qwen`.
Существующие базы используются на месте; только отсутствующая выбранная Qwen base скачивается включённой веткой.
