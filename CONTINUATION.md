# Промпт для продолжения работы над ECG

## Самый новый статус: wrapping up, единый Jupyter на A100

Пользователь попросил закончить при10%usage. Последние требования: BERT/latent/большие ECGмодели; затем один notebook со всеми pipelines и аналитикой; кластер одна A10040или80ГБ, запускJupyter, без лимита времени; модульная структура. Всё собрано в `notebooks/04_all_pipelines_a100.ipynb`, notebook реально выполнен в smoke-режиме с checkpoint/resume. Полные A100ветви выключены флагами, на1050 не запускались. Данные для переноса — `docs/CLUSTER.md`, структура — `docs/STRUCTURE.md`.

Код разнесён в data/processing/models/training/evaluation/experiments/workflows. Два compatibility shim сохранены для старых joblib. CLI прежний, добавлен --beat-model-path. Новые результаты: `reports/representation_results.md`; scratchBERT macroF1valid0.5621; maskedBERT0.4958; замороженныйECGFounder+интервалы AUROCvalid0.8950 противsame-cohortinterval0.8789. Test не читался в этих экспериментах. Representation409.49с,Founder122.13с дополнительно. Новых тренировок до дальнейшей команды не запускать.

A100 профили: BERT512×8/768×12, BF16/TF32, auto batch, tokens/windows наGPU; masked→supervised, epoch checkpoints и resume. ПолныйFounder размораживает все30.8M, полныйPTB folds1–9; rawdata preprocessing отдельной ячейкой. Максимальная загрузкаA100 НЕ измерялась локально, это конфигурации, не обещание скорости. Главные остаточные риски: маленький MITtrain/valid и переобучение; токеныPCA32 ограничивают морфологию; provenanceHEEDB по авторам безpatientmap; fullA100/FP16ветви требуют проверки нацелевомGPU. Исторические статусы ниже не отменяют этот.

## Актуальный статус на 2026-09-05 — читать прежде исторического плана

Новый запрос пользователя: «ищи sota решения текущих проблем, пробуй разное». Выполнен следующий цикл: `reports/sota_experiments.md`. Четыре context/weight HGB ablation, только train/valid, без повторного test. S validation F1 0.303→0.461, но V0.637→0.573; VT не улучшилась. Новые веса `artifacts/context_experiments`, в основной CLI не внедрены. Время66.97с дополнительно, 20 tests passed. Следующие гипотезы и источники в новом отчёте; foundation weights пока НЕ испытывались. Основные результаты ниже относятся к предыдущей поставке.

Исследовательская поставка реализована: CLI, веса, метрики и три выполненных ноутбука. Исторические TODO и session IDs ниже относятся к промежуточному состоянию; обучение CNN, оценка VT, генерация отчёта и финальные проверки уже завершены. Продолжать будущие улучшения из раздела Prospects, не перезапускать выполненные эксперименты автоматически.

- Проверки: `python -m pytest -q` — **19 passed**, compileall успешен; все три ноутбука выполнены через nbclient. CLI LUDB, Challenge, MIT успешно выдаёт интервалы и классификацию; QT поддерживает частичную ручную разметку.
- Итог: `reports/RESULTS.md`, запуск: `PROJECT.md`, первичные статьи: `reports/research_notes.md`, версии: `reports/environment.json`.
- Сумма измеренного времени обучения, включая пробы и CNN: **17.82 минуты**; `reports/training_budget.json`. Предобработка/инференс отдельно.
- Выбранный разметчик `artifacts/delineator_qt.pt`: LUDB test P/QRS/T F1 **0.9706 / 0.9969 / 0.9846**, QRS width MAE **10.914 мс**, согласие порога ≥120 мс **92.49%**. Внешний QT улучшился по QRS, ухудшился по T offset; не утверждать универсального улучшения.
- Record model выбран **interval** по validation AUROC: PTB test macro AUROC **0.8763**, Georgia **0.9157**. Fusion test 0.8981, но это не основание менять выбор по test. Это пилот, не полный исходный корпус.
- Beat model выбран **fusion HGB** по validation N/S/V/F macro F1. Test end-to-end F1 S **0.136**, V **0.862**. CNN обучена и сохранена, но не победила на validation; все четыре ablation есть в отчёте.
- VT test: найдены кандидаты, пересекающие **7 из 21** размеченных эпизодов, 9 кандидатов, 2 без пересечения. Низкая полнота **33.3%** даже при мягком критерии любого пересечения. Не считать задачу клинической диагностики ЖТ решённой.
- Калибровка PTB/StP остаётся неподтверждённой; физические амплитуды исключены. Высокие степени AV-блокады имеют недостаточный test support; форма ФП требует анамнеза.
- Исходные удаления Git сохранены; commit/push не выполнялись.

При дальнейшем планировании главный приоритет — улучшить наджелудочковые экстрасистолы и эпизодную ЖТ на отдельной validation, затем расширить cohort и независимые проверки. Не подбирать параметры по уже просмотренному test; для новых циклов учитывать риск адаптации к тестовым результатам.

## Инструкция следующему агенту

Продолжай работу в `C:\Projects\ECG`, PowerShell, Python 3.13. Не начинай заново и не останавливайся на плане. Основной пользовательский запрос ещё выполняется. Сначала прочитай этот файл, `PROJECT.md`, `reports/research_notes.md`, фактические JSON отчёты и проверь живые процессы/артефакты. Новые сообщения пользователя — уточнение текущей задачи, если явно не отменяют её. Делегирование запрещено текущими developer instructions, если пользователь отдельно его не разрешит. Не восстанавливай удаления старого кода в Git.

Пользователь попросил реализовать целый исследовательский проект: (1) вывод ручных и предсказанных интервалов неоднородных ЭКГ/отведений; (2) классификация по интервалам И нарезанным сигналам. Нужны CLI и ноутбуки, без веб-интерфейса. Можно обучать любые архитектуры, бюджет GTX1050 4GB и до 3 часов обучения, инференс не ограничен. Нужно читать статьи, измерять метрики, делать sanity checks и исключать утечки. Спорные решения, особенно про исходные данные, уточнять.

## Уточнения пользователя

- Цели: желудочковая тахикардия, желудочковая/наджелудочковая экстрасистолия, ФП, АВ-блокады, оценка внутрижелудочковых блокад.
- Порог широкого QRS **>=120 мс**, только в совокупности с другими признаками, не самостоятельный диагноз.
- Форма ФП «впервые выявленная/постоянная» по одной ЭКГ не определяется без анамнеза; это уже объяснено пользователю.
- Таблиц пациентов CPSC/Georgia нет. Пользователь подтвердил: **междатасетного совпадения пациентов нет**. Внутри PTB-XL используем official patient IDs, внутри MIT 201/202 вместе.
- Разрешены и запрошены transfer/unsupervised методы разметки на других датасетах.
- Разрешено скачать дополнительный небольшой независимый датасет, обучаться на части и валидироваться на другой.
- Последняя просьба: при исчерпании контекста сохранить отдельный промпт с планом, статусом и prospects/todo. Этот файл выполняет просьбу; обновлять по ходу работы.

## Состояние репозитория до нашей работы

В рабочем дереве было только LUDB/, data/, ludb_beats_dataset.ipynb, .git. Git уже показывал удаление 40 старых файлов (`README.md`, `.gitignore`, model.py, train_ctn.py и др.). Это **исходные пользовательские удаления**, мы их не делали, не восстанавливать. Новый код — `ecg_project/`, pyproject.toml, PROJECT.md, tests/, scripts/, notebooks/, reports/, configs/. Исходный notebook и исходные данные не менялись. Сетевые загрузки — только metadata и новый `data/qtdb_external/`.

## Данные и обнаруженные проблемы

- LUDB: 200 x 12 x 10 сек, 500Гц, WFDB DAT + 12 аннотаторов (`i`, `ii`, ..., `v6`). Границы включительные, duration=(offset-onset)/fs. Train/valid/test 140/30/30 пациентов, seed42. Все отведения/биты пациента вместе; неразмеченные края игнорируются в loss и benchmark.
- Уже нарезанный `data/ludb_beats`: 1832 бита, не независимый датасет; новый код работает с исходной LUDB.
- MIT-BIH: 48 CSV (650000 отсчётов, 360Гц, 2 отведения) + текстовые annotations. Официальные заголовки скачаны в artifacts/mit_headers; проверка длины, lead names, first sample, checksum перед переводом ADC в mV. Четыре paced записи исключены. Split de Chazal-style с 201 перенесённым к 202 в test; valid={108,114,207,223}. 44 записи используются полностью.
- Challenge: Training_WFDB 6877, Training_2 3453, StPetersburg74, Georgia10344, PTB-XL21837.
- **Training_2:** 1020 заголовков были преобразованы на DAT, потеряли Dx; один DAT отсутствует. MAT/NPY рядом другой шкалы! Не подменять DAT на MAT. 2433 записи имеют Dx и используются для supervision, unlabeled DAT — для consistency adaptation.
- **Незакрытый вопрос пользователю:** менялись ли калибровки? В PTB-XL local gain=200, в пяти official original headers gain=1000, initial samples/checksums совпадают. StPetersburg local gain=306000 также требует выяснения преобразования (original INCART I01 != local I0001 по raw first samples). Пользователь пока не ответил. Код консервативно помечает PTB/StP units `header-scaled (calibration unverified)`, исключает их амплитудные признаки; normalized morphology и интервалы допускаются. Не делай неподтверждённого исправления коэффициентов.
- PTB metadata v1.0.1 скачаны в artifacts/ptbxl_database.csv. Mapping HRxxxxx→ecg_id проверен по age/sex всех записей. Folds1-8 train,9valid,10test.
- Для классификации Georgia полностью external, StP external_long (первые30сек — только exploratory, НЕ полноценный тест общей метки записи). CPSC/Training2 — train. Exact MAT SHA256 duplicates удаляются с приоритетом protected splits.
- **Новый QTDB:** 18 записей (~12MB), SHA256 verified. 8 NSR (`sel16265`, `sel16272`, `sel16273`, `sel16420`, `sel16483`, `sel16539`, `sel16773`, `sel16786`) — adaptation_train. 10 European ST-T (`sele0104`, `sele0106`, `sele0107`, `sele0110`, `sele0111`, `sele0112`, `sele0114`, `sele0116`, `sele0121`, `sele0122`) — external_valid. Никаких QT записей из MIT-BIH Arrhythmia. Split по source, до оценки. q1c ручные границы совместные по двум отведениям, T-onset часто отсутствует; parser сохраняет None. Precision/F1 всей записи на sparse annotation неидентифицируемы — сообщаем recall размеченных волн и ошибки доступных границ. Не выбирать лучший lead по ground truth.

## Реализация

- `io.py`: WFDB DAT/MAT, strict header gains, CSV, LUDB wave parser, MIT text parser.
- `signal.py`: 0.5–40Hz zero-phase morphology; 5–25Hz polarity detector; NeuroKit DWT baseline; fixed-time beats; RR, PR, QT/QTc Fridericia, widths, missingness.
- `segmentation.py`: shared single-lead small 1D U-Net 214408 params, 250Hz, masked loss, GroupNorm, sign/noise augmentation. Long inference overlap-add. Были ускорены repeated medians: теперь baseline median один раз на канал; численно эквивалентно.
- `adaptation.py`: MeanTeacher EMA. Initial LUDB model + real LUDB train loss + .1 confidence>.95 consistency на 200 unlabeled Training2 records, только train source. Не mixed validation/test.
- `qtdb.py`: supervised partial-label adaptation на QT NSR channel0, с сохранением LUDB supervision; T onset unknown ignored; epoch selection LUDB valid Dice, внешний QT в обучение/early stopping не входит.
- `features.py`: interval features + ROCKET-inspired 32 fixed kernels (не официальный ROCKET), median и atypical beats per lead; 12 canonical leads. Beat features: previous/next RR, relative RR, compensation, QRS, PR, P presence, template correlation/distance, relative width. Офлайн-контекст, не causal realtime.
- `record_model.py`: interval / waveform / fusion HistGradientBoosting, train-support>=20; validation thresholds, selected by valid macroAUROC, explicit unsupported VT. No sklearn random internal early stopping. Caches tied to SHA256 signal/header/delineator.
- `beats.py`: 5 classes N,S,V,F,Q; V только symbolV, E→Q (не называть ventricular escape PVC). Actual automatic detected peaks; reference labels только для supervision/evaluation. End-to-end recall includes missed/edge beats; precision penalizes unmatched detector outputs. HGB ablations completed.
- `beat_cnn.py` **новый, сейчас обучается**: маленькая Conv1D morphology + scaled/imputed interval branch, train-only stats, class weights sqrt, validation epoch selection macro F1 N/S/V/F. Wrapper `BeatPredictor` supports predict_proba/predict for existing pipeline; artifact `cnn_fusion.joblib`. It appends metrics and reselects best using valid only.
- `analysis.py`: analyze / batch CLI, JSON, CSV, NPZ, plotting; default delineator_qt if exists. Hash mismatch→no incompatible classifier use. Shared QT manual annotation supported, partial T only known region shown. Full long12lead inference windows10sec, no calibrated record aggregate.
- `episodes.py`: >=3 consecutive predicted V with prob>=.8, mean rate>=100; candidates only. Evaluate against MIT `(VT` rhythm segments with lenient any-overlap retrieval, no boundary-accuracy claim.
- `robustness.py`: fixed 5 LUDB valid patients, II/aVR/V1, synthetic20dB noise, drift,50Hz.
- `scripts/uncertainty.py`: 500 patient-cluster bootstraps on selected LUDB test model, retaining all leads per patient.

## Завершённые эксперименты / метрики

1. Full DWT valid30patients12leads: QRS F1 raw .87485, filtered .86610; T .74594/.75744. Reports/dwt_valid_full.json.
2. Baseline U-Net `artifacts/delineator.pt`: 30epochs, best24, 174.37sec GTX1050, max torch GPU allocated59.86MB. Valid P/QRS/T F1 .9576/.9843/.9710; QRS width MAE16.10ms. Test .95997/.99343/.97728; width14.72ms.
3. MeanTeacher `artifacts/delineator_transfer.pt`:12epochs143.54sec; valid width worsens18.13ms though T slightly improves. Не объявлять универсальным улучшением.
4. QT-supervised `artifacts/delineator_qt.pt`:10epochs114.30sec; selected by LUDB valid Dice. Valid F1 .9571/.9864/.9760; width12.90ms, threshold agreement88.71%. **Test F1 .97058/.99686/.98462; width10.914ms, bias-0.116ms, >=120 agreement92.490%, matchedQRS3329**. This is selected delineator.
5. ExternalQT baseline→QTadapted QRS widthMAE: channel0 20.679→17.593ms, channel1 17.519→12.136ms. T boundaries still poor, e.g T offset42.7→48.6ms channel0,25.9→30.8ms channel1; report honestly. All JSON in reports/qtdb_*.json.
6. Record pilot: **3239 records** (limit500 per source/split + all rare training positives; duplicates maybe reduce count—inspect manifest). Baseline feature classifier completed and saved in artifacts/record_models_baseline. Final features with selected QT delineator completed `artifacts/record_features_qt/manifest.csv` (~637sec inference). **Final record HGB training completed** artifacts/record_models/{metrics,selection,budget}.json; inspect actual values, do not reuse baseline metrics. Baseline interval valid macroAUROC .8693, test.8684, Georgia.8719; fusion test.8846 but selection valid interval. Final may differ. VT train only1→unsupported. Rare AVB2/3 test pilot may have0positives; AUROC=None.
7. MIT baseline features completed artifacts/beat_features (2202sec extraction); QT features completed artifacts/beat_features_qt (936.6sec extraction). No need redo.
8. Beat HGB final QT ablations completed artifacts/beat_models; validation macro including Q: interval .4489, waveform .2603, fusion .3924. Selection uses N/S/V/F not Q. New CNN branch is being trained because waveformHGB weak. Need inspect final CNN+test performance.
9. `reports/delineation_uncertainty.json`: QRS test F1 CI95 [.99249,.99968], widthMAE CI95 [9.03,13.72]ms. Patient bootstrap, not independent leads.
10. Tests **17 passed** before latest BeatCNN/episodes additions. Need rerun final suite. Compileall passed earlier. Plot example LUDB visually inspected, proper overlays; unknown boundary cases handled after initial plotting.

## Живые инструменты на момент создания файла

Последний `python -u -m ecg_project train-beat-cnn` имеет **exec session_id=3005**. Другие computations завершены кроме возможного analyzeQT session22932 (проверить) и оставшегося output eval_qt_test93278 (артефакт точно создан). Старые tool session IDs могут уже быть закрыты, проверять процессы/файлы при сомнении.

Не запускать train-beats до final manifest: один преждевременный запуск упал на отсутствующем manifest, затем был успешно повторён после окончания extraction. Ошибка исправлена информативным сообщением.

## Обязательное TODO до финального ответа

1. Дождаться CNN, проверить метрики/selection. Если ошибка — исправить, перезапустить только необходимое. Убедиться wrapper joblib load predict работает.
2. Запустить `python -m ecg_project evaluate-vt` после окончательного beat selection. Проверить TP/false candidates, не выдавать candidate retrieval за clinical VT accuracy.
3. Полный end-to-end CLI smoke на LUDB1, ChallengeA0001, MIT200 (60сек), QTsele0104 (600–610сек). Убедиться финальные record/beat models согласованы с `delineator_qt.pt`, record predictions непусты на12lead, beat predictions есть наMLII, JSON finite/None, csv absolute coords корректны. Старые examples созданы до финальных моделей — обновить.
4. Проверить 3 notebooks; они сгенерированы `scripts/make_notebooks.py`, ещё НЕ выполнены. `scripts/execute_notebooks.py` uses nbclient; при sandbox socket failure требовать escalation (не обходить). Approved prefixes downloads only; kernel execution may need permission. Не говорить notebooks выполнены, пока это не так.
5. Обновить make_report.py/PROJECT.md/configs с CNN: command train-beat-cnn, четвёртая ablation, бюджет add artifacts/beat_models/cnn_budget.json. Сейчас make_report budgets учитывают baseline+finalrecords, delineators, beatHGB, но ещё НЕCNN.
6. `python scripts/make_report.py` создаёт reports/RESULTS.md, training_budget.json, environment.json. Пока не запускался / RESULT ещё нет. Проверить фактическую сумму обучения <180мин, GPU<4GB. Actual training пока ~15мин, plenty of budget; preprocessing separately.
7. `python -m pytest -q` финально, добавить meaningful smoke tests BeatCNN wrapper shape/proba и QT partial plot if useful. Проверить все JSON/NPZ, git status preserves original deletions. Не commit/push.
8. Проверить документацию соответствует реальности. PROJECT.md уже есть; research_notes.md нужно дописать MeanTeacher/QT/CNN actual results/citations; user asked internet research. Sources in file primary links.
9. Опционально выдать research packaging / requirements snapshot and audit counts, but no new huge experiments unless evidence requires.
10. Обновить **этот файл** итоговым статусом, exact metrics, future prospects. Затем final по-русски: ссылки на PROJECT.md/RESULTS/notebooks; главные измерения; явные границы (редкие VT/highgradeAVB, формаAF, amplitudequestion). Не утверждать production/clinical readiness. Упомянуть project implemented and tested, training time actual. Не завершать раньше необходимой проверки.

## Prospects после текущей поставки

- После ответа о калибровке восстановить физические амплитуды с официальными/подтверждёнными gains; не угадывать.
- Полный training cohort вместо pilot500/source/split (`prepare-records --limit 0`, новый cache dir); primary PTB test целиком для большего support редких блокад. Inference unlimited, но новый full training держать в3h budget.
- Больше независимо размеченных abnormal QT/P/T cases, т.к QTDB mostlynormal sparse labels не доказывает quality приVT/AF.
- Валидация переносящегося beat classifier на иных отведениях, lead dropout/multilead shared encoder, external beat labels.
- Надёжная VT/AV dissociation/AVB2/3 episode model требует дополнительных эпизодных labels и пациентов; current candidates недостаточно для clinical validation.
- Calibrated probabilities, per-patient bootstrap on disease models, full-record weak-label MIL для длинных ECG, richer T-boundary supervision, uncertainty-aware interval extraction.
- Label correlation refinement можно сравнить только train-fitted/out-of-fold и с контролем sourceconfounding, не использовать true test co-occurrence.

## Полезные команды

```powershell
python -m ecg_project --help
python -m ecg_project analyze LUDB/1.hea --output reports/example_ludb
python -m ecg_project analyze data/mit-bih/200.csv --duration-seconds 60 --output reports/example_mit
python -m ecg_project analyze data/Training_WFDB/A0001.hea --output reports/example_challenge
python -m ecg_project evaluate-vt
python scripts/execute_notebooks.py
python scripts/make_report.py
python -m pytest -q
```
