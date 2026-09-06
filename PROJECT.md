# ECG: разметка интервалов и классификация

Единый [Jupyter notebook](notebooks/04_all_pipelines_a100.ipynb) управляет обучением и сравнением моделей на одной A100 40/80 ГБ и показывает границы волн и метрики.

**После очистки обученные модели и результаты отсутствуют.** Исходные датасеты, официальные метаданные и базы HuBERT/ECGFounder сохранены. Qwen скачивается только на кластере из notebook. Все экспериментальные реализации сохранены.

- [Повторный запуск после очистки](docs/FRESH_START.md).
- [CPU: подготовка с workers](pipelines/cpu/README.md); [GPU: обучение](pipelines/gpu/README.md); [что перенести](pipelines/README.md).
- [Исторически выбранный финальный маршрут](docs/FINAL_PIPELINE.md).
- [HuBERT LoRA/QLoRA и Qwen-разметчик](docs/LORA.md). В notebook сравниваются Qwen3-1.7B и 4B; 0.6B/8B доступны опционально.
- [Предложение общей модели и новых датасетов — на утверждение](docs/NEXT_EXPERIMENTS.md).
- [Структура кода](docs/STRUCTURE.md); [научные источники](docs/RESEARCH.md).

```powershell
python -m pip install -e ".[dev]"
python -m pipelines.cpu.run audit
python -m pipelines.cpu.run prepare-qwen --workers 4
python -m pipelines.cpu.run prepare-hubert --workers 4
python -m pytest -q
```

На кластере установите `.[dev,qlora]` и подходящую CUDA-сборку PyTorch, откройте notebook, включите нужные RUN-флаги. Полное обучение по умолчанию выключено. Необходимые кэши проверяются до запуска. Предобработка и инженерные features поддерживаются; train/valid/test разделены по доступным идентичностям пациентов и источникам.

При наличии обученных совместимых моделей CLI `python -m ecg_project run RECORD` выводит CSV интервалов, нарезанные beats, классификацию и график. `analyze` позволяет явно выбрать экспериментальные модели. Qwen пока имеет отдельный Predictor и evaluation в notebook; он не подменяет разметчик основного маршрута автоматически.

QRS ≥120 мс — измеряемый признак. Полнота кандидатов ЖТ низкая, редкие блокады недостаточно подтверждены, форму ФП без анамнеза определить нельзя. Пилотные метрики и smoke не доказывают клиническую готовность.
