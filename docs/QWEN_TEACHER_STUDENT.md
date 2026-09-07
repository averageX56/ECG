# Qwen delineation: supervised и teacher–student

Teacher — выбранный U-Net `artifacts/cluster/delineator_qt.pt`; student — Qwen3 1.7B/4B,
NF4 QLoRA, ECG patch adapter и sample segmentation head. Qwen получает сигнал через embeddings;
генерация текстовых ответов и словесных координат интервалов не используется.

## Эксперименты

| Config | Manual train | Pseudo train | Regime |
|---|---|---|---|
| A | LUDB | — | supervised_only |
| B | LUDB + manual QTDB | — | supervised_only |
| C | B | Training_2 | teacher_student_hard |
| D | B | Training_2 + PTBXL1–8/CPSC/Chapman | teacher_student_hard |
| E | B | D + soft KD | teacher_student_hard_soft |

JSON configs: `configs/qwen_v2/{A..E}_{1.7B,4B}.json`. В notebook выбираются явно;
factory `pipelines.gpu.experiments.qwen_experiment` задаёт 40/80GB profile.
Для загрузки JSON: `DelineationConfig(**json.loads(Path(config_path).read_text()))`, затем
`pipelines.gpu.experiments.run_qwen(cfg, run_smoke=False)` после скачивания base на кластере.

Для сравнения A–E оба manual caches готовятся с `--validation-datasets LUDB QTDB`.
LUDB test, QT external, PTB9/10 и external records не участвуют в selection/training.
Новые размеры/режимы имеют отдельные run directories. Supervised baseline не удалён.

## Colab GPU cache

В notebook включить `PREPARE_QWEN_PSEUDO_GPU=True`. C/D/E автоматически вызывают отдельный
GPU preparation stage после появления teacher, до загрузки Qwen base:

```python
from pipelines.gpu.colab import prepare_qwen_pseudo_gpu
prepare_qwen_pseudo_gpu(
    output='artifacts/qwen_pseudo_extended',
    sources=['CPSC_EXTRA','PTBXL','CPSC','CHAPMAN'],
    checkpoint='artifacts/cluster/delineator_qt.pt',
    batch_size=64, workers=8,
)
```

Исходные `data/` и `LUDB/`, catalog и manual teacher-training manifest должны быть доступны
через Drive. U-Net работает батчами на CUDA, workers только читают/фильтруют входы.
Полностью готовый cache не вызывает teacher forward; частичный продолжается по готовым shards.
Выход сохраняется в Drive/artifacts; CPU fallback command остаётся доступной как optional baseline.

Горячий цикл выполняется на локальном SSD Colab: Drive raw → staging → CPU workers → CUDA U-Net
→ local cache → финальная Drive sync. Raw symlinks временные; registry тоже локальный.
Fast resume и strict rehash описаны в [performance protocol](QWEN_PREPARATION_PERFORMANCE.md).

Extended pool задаётся явно в sources и отдельным output.
Teacher inference делается один раз; student не загружает U-Net. Один record даёт детерминированное
первое10sec окно, leadII или первый доступный lead. Общая фильтрация/resampling250Hz и normalization
совпадают с input teacher; edge125 samples игнорируются. Неполные короткие записи не дополняются нулями.

Каждый shard содержит signal, unscaled logits4×time, confidence, hard target (-100 при ignore),
valid mask, lead/window. Manifest содержит source/record/patient/split, source-file SHA256,
decoded-signal SHA256, teacher SHA256 и preprocessing version. Resume проверяет каждый готовый shard.
Смена checkpoint/tau/data требует нового output; недописанный shard пересчитывается.

Protected registry читает только identity и сигналы holdout cohorts, не target labels.
Проверяются known patient IDs, source-qualified record IDs и decoded signal hash
(float32 samples, fs и ordered lead names). Exact duplicates protected↔pseudo исключаются,
train↔train deduplicated. Identity conflicts завершают подготовку ошибкой.
Registry resumable в `artifacts/protected_identities_v2`. Его изменение также запрещает reuse.
Проверяется и train-pool teacher: нужен v2 U-Net checkpoint с identity и исходный CPU manual
manifest по пути config.input_root. SHA256 manifest должен совпасть с checkpoint; teacher train
identities не могут пересекать protected cohorts. Старый teacher без проверяемой provenance
отклоняется: сначала обучить новый U-Net. Это особенно важно для прежнего QT baseline,
который использовал все8 adaptation records, включая нынешние2 validation records.
Signal hash не обнаруживает near-duplicates, другое масштабирование, обрезки и перестановку leads;
неизвестные cross-source patient identities не объявляются доказанно различными.

## Objective и sampling

```text
L = lambda_supervised * L_manual
  + ramp(epoch) * (lambda_pseudo * L_hard + lambda_kd * L_KD)
```

Manual CE учитывает только размеченные samples; LUDB и QT contributions взвешиваются отдельно
через `qt_weight`. В Qwen profiles qt_weight=1.0. lambda_supervised=1.0 всегда положительна.
Hard CE: argmax teacher только при confidence≥tau, по умолчанию tau=.95; остальное ignore=-100.
KD: T² KL(softmax(teacher_logits/T) || softmax(student_logits/T)), default T=2,
только valid samples; `kd_confidence_threshold` позволяет отдельный, более мягкий threshold.
Сохранённые logits не temperature-scaled, поэтому T применяется ровно один раз.

lambda_pseudo=.5; lambda_kd=0 для C/D, .5 для E. `pseudo_ramp_epochs=5` даёт линейный ramp,
0 отключает ramp. `pseudo_per_manual=1` означает один pseudo microbatch на каждый manual microbatch;
2 даёт приблизительно manual:pseudo1:2. Manual loader проходит полностью каждый full epoch.
Pseudo loader циклически перемешивается, большой pseudo pool не удлиняет epoch и не вытесняет manual.
Доли могут отличаться на последнем неполном batch; smoke с max_batches намеренно ограничен.
Extra losses усредняются по pseudo_per_manual; увеличение sampling ratio не умножает lambda.

Student augmentations: amplitude scaling, polarity flip, Gaussian noise, baseline drift.
Они не меняют sample coordinates. Time warping не применяется. Teacher cache canonical.

## Reports и интерпретация

### Повторные циклы с большей долей примеси

В notebook включить `RUN_QWEN=True`, выбрать C/D/E, затем `RUN_QWEN_CURRICULUM=True`.
`PSEUDO_RATIOS=[1,2,4]` задаёт последовательные manual:pseudo1:1 →1:2 →1:4.
Каждый цикл начинает student с лучшего принятого checkpoint. Optimizer нового цикла
создаётся заново; resume внутри цикла восстанавливает его и stale counter.
Чтобы продолжить уже обученный E, перед вызовом задать `cfg=replace(cfg,warm_start='.../best.pt')`.

В `*_curriculum/selection.json` сохраняются все предложения, принятые/отклонённые циклы,
checkpoint SHA256 и последний лучший результат. По умолчанию требуется прирост mean Dice≥.001,
без падения отдельной P/QRS/T Dice любого validation source более .005. После двух
неулучшающих циклов — остановка; отклонённый student не становится исходным для следующего.
Пороги настраиваются до запуска, test для них не используется. Число циклов явно конечно;
для следующего плана нужен новый output и warm_start последней принятой модели.

Teacher остаётся фиксированным, его pseudo cache не переписывается. При новом teacher
сначала повторить GPU pseudo generation в новый cache. Сам рост доли примеси не устраняет
ошибки teacher и не доказывает способность размечать все ЭКГ. Нужны ручные данные новых
доменов и заранее фиксированный независимый test; validation тоже может переобучиться
при неограниченном числе циклов, поэтому вечного auto-loop нет.

`reports/qwen_pseudo_dataset_report.json` и cache provenance содержат coverage, ignored fraction,
mean confidence, class counts и per-source stats. Этот общий report отражает последний подготовленный
cache; исходные per-cache provenance сохраняют статистику каждого эксперимента.
В run.json дополнительно сохраняются manual class counts и pseudo statistics, training regime,
best epoch, PyTorch peak memory и throughput. effective_batch означает manual batch×accumulation;
число pseudo examples дополнительно регулируется pseudo_per_manual.

Best checkpoint выбирается по среднему validation P/QRS/T Dice, отдельно публикуются LUDB/QT reports.
Partial QT Dice рассчитан только на известных samples и не равен fully annotated record Dice.
Исторические boundary metrics в baseline evaluation доступны отдельно; v2 selection — sample Dice.
Клиническое превосходство Qwen и лучшая A100 утилизация пока не установлены запуском.

Если distilled Qwen превзойдёт U-Net, прежде чем делать вывод: проверить teacher/student train
provenance против evaluation identities, один и тот же test set, отсутствие test coverage в pseudo cache,
и явно указать дополнительный teacher supervision budget. Unknown pretraining overlap не скрывать.

QT annotation semantics: [официальный QTDB](https://physionet.org/content/qtdb/1.0.0/).
Новые beat sources: [SVDB](https://physionet.org/content/svdb/1.0.0/),
[INCART](https://physionet.org/content/incartdb/1.0.0/). Automatic QT annotations не подменяют manual ground truth.
