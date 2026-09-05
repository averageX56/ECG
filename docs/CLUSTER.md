# Одна A100 40/80 ГБ, запуск через Jupyter

Откройте **`notebooks/04_all_pipelines_a100.ipynb`**. Это единая точка входа: аудит, smoke, разметчик, SSL/BERT, ECGFounder, классификация, графики и аналитика. По умолчанию полный training выключен; RUN_SMOKE включён. Лимита времени на кластере нет. Запуск задач на кластер из локальной среды не выполнялся.

## Что выгрузить

Сохраняйте относительную структуру каталогов. Ядро ноутбука работает из корня проекта, независимо от Windows/Linux.

| Назначение | Обязательные файлы/датасеты |
|---|---|
| Код и единый ноутбук | `ecg_project/`, `pyproject.toml`, `scripts/`, `configs/`, `docs/`, `notebooks/04_all_pipelines_a100.ipynb` |
| Быстрый BERT/SSL на уже извлечённых битах | **`artifacts/beat_features_qt/` целиком**, включая manifest.json и все44 NPZ/JSON; `artifacts/delineator_qt.pt`; сами исходные MIT CSV для такого обучения не нужны |
| Пересоздание битов, анализ ЖТ и графики MIT | **`data/mit-bih/`**:48CSV и annotation TXT; **`artifacts/mit_headers/`** официальные заголовки |
| Обучение/графики разметчика | **`LUDB/`** целиком:200hea/dat и все отведенческие аннотации |
| QT adaptation/external control | **`data/qtdb_external/`** целиком:18записей и manifest; не объединяйте NSR train и European ST-T control |
| Дополнительное SSL без меток | **`data/Training_2/`** и `artifacts/catalog.csv`; используются только readable записи без Dx. Можно передать готовый `artifacts/unlabeled_beats/` вместо raw |
| ECGFounder — полный PTB train/valid | **`data/WFDB_PTB-XL/`** с hea и соответствующими mat, `artifacts/catalog.csv`, `artifacts/ptbxl_database.csv`. Используются folds1–9; fold10 остаётся закрытым |
| ECGFounder — воспроизвести только локальный пилот | Готовые **`artifacts/founder_experiments/inputs.npz`** и `manifest.csv`; raw PTB не нужны |
| Предобученные веса | **`artifacts/ecgfounder/`**, включая `1_lead_ECGFounder.pth`, `net1d.py`, `sha256.json`, README. Или загрузка официальным `scripts/download_founder.py` (~370МБ) |
| Полная классификация нескольких источников | Все **`data/Training_WFDB/`, `data/Training_2/`, `data/WFDB_PTB-XL/`, `data/WFDB_GEORGIA/`**, catalog и PTB metadata; StPetersburg опционален для exploratory длинных записей |
| Просмотр уже полученных результатов | `reports/`, `artifacts/record_models/`, `artifacts/beat_models/`, `artifacts/representation_experiments/`, `artifacts/context_experiments/`, `artifacts/founder_experiments/`; для графика LUDB нужен raw LUDB |

`data/ludb_beats/` не нужен: дублирует исходную LUDB. Не переносите старые baseline caches, если не воспроизводите их отдельно. HEEDB/MIMIC/Icentia11k **не требуются** текущему пайплайну; использование готового ECGFounder не требует скачивания его pretraining corpus.

Проверьте точное имя локального каталога PTB/Georgia в `artifacts/catalog.csv`: используйте пути из catalog, не переименовывайте данные при переносе. В рамках подготовки каталога не подменяйте отсутствующие DAT соседними MAT.

## A100

- BERT40ГБ: dim512,8слоёв;80ГБ:dim768,12слоёв. Длина65битов. BF16, TF32, подбор batch с30%резервом памяти; компактные tokens и сборка окон наGPU.
- Отдельный A100 замер не выполнялся. Автоподбор максимизирует размер batch в установленном диапазоне, **не гарантирует максимальную аппаратную загрузку**. Используйте `nvidia-smi` и историю времени эпох; узким местом может оказаться небольшой датасет.
- Полное дообучение ECGFounder30,8M: BF16/TF32; стартовые batch32/64 для40/80ГБ. Увеличивайте после первой эпохи в **новом run**; resume требует того же batch. Все параметры backbone разморожены.
- Jupyter: одна GPU-ветвь за раз. Несколько одновременно работающих ядер конкурируют заVRAM. Для BERT DataLoader не нужен при gpu_resident=True; для fullFounder Linux workers4/pinned memory.
- Установка: `python -m pip install -e ".[dev]"`, подходящий CUDA PyTorch; ipywidgets опционален. Никаких Slurm-скриптов не требуется.
- Resume загружает последнюю завершённую эпоху; прерванная незавершённая эпоха повторяется. Сохраняются optimizer/RNG. Не меняйте данные/архитектуру в том же output.

## Границы проверки

Локально обучены маленький BERT с нуля/SSL, latent-HGB, frozen ECGFounder и fine-tuning последнего блока. Полные A100 профили подготовлены, но не обучались на1050. Крупные модели на небольшом MIT корпусе могут переобучаться. Внешний test уже просмотрен в прошлом цикле; новый выбор делается на validation. Автоматическая клиническая интерпретация формыФП/критичности блокад не заявляется.
