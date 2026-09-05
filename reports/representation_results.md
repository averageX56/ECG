# BERT, latent features и ECGFounder — фактические пробы

Все новые результаты — validation; test не открывался этими экспериментами.

| MIT вариант | End-to-end macro F1 N/S/V/F | S F1 | V F1 |
|---|---:|---:|---:|
| PCA latent + интервалы + HGB | 0.4464 | 0.3077 | 0.5322 |
| BERT с нуля | **0.5621** | **0.4366** | 0.7005 |
| Скрытые признаки supervised BERT + HGB | 0.5418 | 0.3622 | 0.7088 |
| Masked SSL → frozen latent HGB | 0.5088 | 0.3769 | 0.6957 |
| Masked SSL → fine-tuned BERT | 0.4958 | 0.1417 | 0.6919 |
| Скрытые признаки fine-tuned SSL BERT + HGB | 0.4640 | 0.0574 | **0.7380** |

Небольшой BERT:292859параметров,3слоя,dim96,контекст17сокращений; tokens PCA32+11интервалов+11индикаторов отсутствия. PCA/нормировки fit только train. SSL8эпох masked reconstruction, supervised15эпох. Это адаптация принципа bidirectional masked modeling к непрерывным признакам, **не точное воспроизведение HeartBERT**. Отрицательный результат SSL сохранён; большее число стадий не дало общего улучшения. Четыре validation-пациента недостаточны для уверенного обобщения; сильная вариативность по эпохам требует расширенной проверки.

| PTB pilot, только отведение I | Validation macro AUROC |
|---|---:|
| Интервалы I + HGB | 0.8789 |
| Frozen ECGFounder embeddings + HGB | 0.8936 |
| Frozen embeddings + интервалы I | **0.8950** |
| Fine-tuning последнего блока ECGFounder | 0.8948 |

Одинаковый cohort:524train/499valid; классы с≥20train-позитивами. ECGFounder30,8Mпараметров, оригинальные одноотведённые веса,500Гц/10с, notch50Гц,bandpass0.67–40Гц,median baseline и z-score по официальному коду. Строго загружен state_dict, SHA256 сохранён. Полное дообучение всей модели подготовлено для A100, локально эта крупная ветвь не обучалась.

Время новых representation-проб409.49с; founder-проб122.13с (включая extraction/evaluation в измеренном времени). Пиковая выделенная CUDA-память≈70МиБ и419МиБ соответственно. Это не измерение полной зарезервированной/системной VRAM.

## Основания и происхождение весов

- [HeartBERT, авторский репозиторий](https://github.com/ecgResearch/HeartBert): pretraining включает MIT-BIH, PTB-XL и European ST-T. Готовые веса исключены из независимой MIT-проверки.
- [ECGFounder, статья](https://arxiv.org/html/2410.04133v1): обучение на HEEDB, PTB как внешний benchmark. Индивидуальная таблица пересечений с HEEDB нам недоступна; происхождение фиксируется по сведениям авторов.
- [Официальные веса](https://huggingface.co/PKUDigitalHealth/ECGFounder), [код](https://github.com/NickLJLee/ECGFounder).

Запуск всех ветвей, A100-конфигурации и графики: `notebooks/04_all_pipelines_a100.ipynb`. Список переносимых данных: `docs/CLUSTER.md`.
