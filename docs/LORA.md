# HuBERT LoRA / QLoRA и Qwen-разметчик

Общий запуск и перенос кэшей: [CPU/GPU](../pipelines/README.md). Установка исследовательских зависимостей: `python -m pip install -e ".[dev,qlora]"`.

## HuBERT-ECG Large: классификация

Используются [официальные веса автора](https://huggingface.co/Edoardo-Coppola/hubert-ecg-large), revision `f297337d4e33198800441e3d64e0dd393774637f`, и нативный `HubertModel`. Custom Python из репозитория весов не исполняется. Модель имеет около 188 млн параметров. Подробности исходной ECG-предобработки: [код авторов](https://github.com/Edoar-do/HuBERT-ECG).

Вход: канонические 12 отведений, 10 секунд, FIR 0.05–47 Гц, ресэмплинг, нормализация каждого отведения в [-1,1], две 5-секундные половины. Каждая половина преобразуется в 6000 значений в порядке отведений по схеме авторов. Неполные отведения и нечисловые значения отвергаются. Используются PTB folds1–8/9; пациенты не пересекаются. Fold10 не загружается.

Сравниваются frozen encoder + head, LoRA Q/V rank16 alpha32 и QLoRA с теми же адаптерами. NF4 с double quantization применяется к линейным слоям; свёртки, нормализация и обучаемые параметры остаются floating point. Оптимизатор — обычный AdamW для адаптеров и головы; paged optimizer в этом варианте не используется. BF16 на A100, FP32 compute на GTX1050. Основание квантизации: [документация bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Локальный smoke использует 8 train, 8 valid, одну эпоху и только две minibatch. Он проверяет загрузку реальных весов, backward, сохранение и инференс. Перед очисткой: frozen/LoRA/QLoRA(fixed seed) дали smoke AUROC 0.6190/0.6190/0.6302 на 8 valid; это не сравнение итогового качества. Peak allocated CUDA LoRA ~789 MiB, QLoRA ~202 MiB. Run-файлы и адаптеры удалены по запросу. В A100-ветке лимиты smoke сняты.

PTB/CPSC/Georgia присутствовали в предобучении HuBERT. Их оценки нельзя называть независимым внешним подтверждением. Лицензия весов CC BY-NC4.0. Новая модель не заменяет выбранный HGB автоматически.

## Qwen3 + QLoRA: разметка P/QRS/T

Это эксперимент переноса языкового декодера на сигнал, а не готовая медицинская модель. Основа — [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3), с входом `inputs_embeds` и без текстового tokenizer/LM head. Локально проверена только малая случайно инициализированная Qwen-архитектура. Предобученные Qwen3-1.7B/4B проверяются и обучаются на кластере; опционально 0.6B/8B. Веса скачиваются отдельно и фиксируются revision/SHA256; готовность 4B/8B не заявляется по запуску 0.6B.

1. На CPU LUDB преобразуется той же функцией, что используется U-Net: 250 Гц, 10 секунд по одному отведению, метки background/P/QRS/T. Неразмеченные края имеют ignore_index=-100. Пациенты 140/30 train/valid; test не читается.
2. 20 отсчётов (80 мс) → обучаемый linear adapter и LayerNorm → один ECG token. Языковой embedding table удалён.
3. Общий Qwen обрабатывает последовательность токенов слева направо и справа налево. Causal attention сохранён; обратные состояния разворачиваются обратно и усредняются. Это офлайн-анализ с будущим контекстом.
4. Linear head выдаёт 20×4 логита на token, поэтому выход сохраняет шаг **4 мс**, а не округляет границы до 80 мс. Padding обрезается до исходной длины.
5. NF4/double quantization замораживает линейные базовые веса; обучаются Q/V LoRA, input adapter и sample head. Loss — взвешенный cross entropy с весами только из train. Эпоха выбирается по validation wave Dice; окончательное сравнение использует также event F1, ошибки границ и ширины QRS.

```powershell
python -m pipelines.cpu.run prepare-qwen
# Qwen downloads on the cluster from the unified notebook
```

На кластер перенести `artifacts/qwen_delineation_inputs/` и `artifacts/qwen3_4b/`. Исходная LUDB нужна дополнительно для оценки границ и графиков, но не для обучения из кэша. Загрузка базы сначала использует CPU RAM: для 4B разумно выделить не менее 32 ГБ RAM, для 8B — 64 ГБ; это не требование к VRAM. Удаление embedding table выполняется после загрузки.

```python
from pipelines.gpu.run import train_qwen
run_path = train_qwen()  # Qwen3-4B; microbatch 4/8 для A100 40/80 ГБ
from ecg_project.models.qwen_delineator import Predictor
from ecg_project.models.segmentation import evaluate
predictor = Predictor(run_path)
metrics = evaluate(split='valid', predictor=predictor,
                   output='reports/qwen_delineation_valid.json')
```

Локальную загрузку Qwen остановили по запросу; реальный smoke выполняет GPU-ячейка notebook.

Два шага не обучают новый интерфейс сигнала до приемлемого качества. До подтверждённого преимущества Qwen не входит в финальный маршрут и не используется для подготовки признаков финальных классификаторов. Источники и ограничения должны учитываться при интерпретации любого улучшения.

Уточнение запуска: Qwen локально не скачивается. Загрузка перенесена в GPU-ячейку единого notebook, см. [FRESH_START](../docs/FRESH_START.md). Реальный Qwen smoke на предобученных весах локально не выполнялся.

В едином notebook `QWEN_SIZES=["1.7B","4B"]`: обе модели скачиваются только на кластере и обучаются в отдельных run-каталогах на одинаковом LUDB train/valid. Размеры 0.6B и 8B можно явно добавить в список.
