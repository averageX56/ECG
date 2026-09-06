# Навигация по проекту

Главная точка входа для кластера и просмотра: **notebooks/04_all_pipelines_a100.ipynb**.

```text
ecg_project/
  data/          чтение WFDB/MAT/CSV, каталог, patient splits
  processing/    фильтрация, нарезка битов, интервальные признаки
  models/        разметчик, CNN, адаптер BERT для инференса
  training/      обучение, адаптация, A100/Jupyter, подготовка SSL
  evaluation/    matching, метрики, robustness, эпизоды ЖТ
  experiments/   сравнения RR-context, BERT/latent, ECGFounder
  workflows/     анализ записи и batch CLI
  __main__.py    командная строка
  utils.py       seed и сериализация JSON
notebooks/       единый pipeline обучения и просмотра качества
scripts/         загрузки, генерация отчётов/ноутбуков
docs/            структура и инструкция переноса на A100
configs/         описание экспериментов
artifacts/       веса, caches, manifests, run histories
reports/         метрики, примеры графиков, исследовательские выводы
tests/           проверки корректности
LUDB/, data/     исходные датасеты
```

Два файла `ecg_project/beat_cnn.py` и `sequence_predictor.py` оставлены только как compatibility imports: их старые пути записаны внутри существующих joblib-моделей. Основная реализация находится в models. Старые ноутбуки и одноразовые генераторы исключены из репозитория. JSON-метрики и исследовательские отчёты оставлены для проверки опубликованных результатов; примеры сигналов и графиков генерируются локально.

CLI остаётся `python -m ecg_project`. Например, экспериментальный BERT:

```powershell
python -m ecg_project analyze data/mit-bih/200.csv --duration-seconds 20 --beat-model-path artifacts/representation_experiments/scratch_bert.joblib --output reports/example_bert
```

Экспериментальные веса не подменяют автоматически исходную выбранную модель. Данные пользователей и исходные удаления старого Git-кода сохранены.
