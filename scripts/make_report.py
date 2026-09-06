"""Generate a concise Russian report from saved measurements, not estimated scores."""
from pathlib import Path
import json
import platform
import importlib.metadata as im
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'reports').mkdir(exist_ok=True)
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def fmt(v):return '—' if v is None else f'{v:.3f}'

lines=['# Реализованный ECG-проект: измеренные результаты','',
'Код, CLI, обученные локальные веса и единый проверенный ноутбук. Метрики относятся к указанным выборкам и протоколам; это исследовательская система.', '',
'## Разметка интервалов','',
'| Модель / выборка | P F1 | QRS F1 | T F1 | QRS MAE, мс | Согласие ≥120 мс |',
'|---|---:|---:|---:|---:|---:|']
for label,p in [('LUDB baseline / valid','reports/segmentation_valid.json'),('Consistency / valid','reports/segmentation_transfer_valid.json'),
                ('QT adaptation / valid','reports/segmentation_qt_valid.json'),('LUDB baseline / test','reports/segmentation_test.json'),('QT adaptation / test','reports/segmentation_qt_test.json')]:
    if not (ROOT/p).exists():continue
    d=read(p);s=d['summary'];q=d['qrs_width']
    lines.append('| '+label+' | '+' | '.join(fmt(v) for v in [s['P']['f1'],s['QRS']['f1'],s['T']['f1'],q['mae_ms'],q['threshold_120_agreement']])+' |')
lines+=['','LUDB: 140/30/30 пациентов, все отведения пациента в одной части. One-to-one сопоставление пиков в пределах 150 мс, оценка в размеченном диапазоне. Ошибка границ и ширины условна на найденных волнах; пропуски входят в recall. QRS — по каждому отведению, не глобальная длительность по всем отведениям.',
       '', '## Независимый QTDB', '',
       '18 записей (~12 МБ), SHA256 проверены. Восемь NSR записей — адаптация, десять European ST-T — внешний контроль. Все записи QT из MIT-BIH Arrhythmia исключены. T-onset часто отсутствует; точность всей записи по частичной разметке не вычисляется. Оба канала сравниваются с общей ручной разметкой, без выбора лучшего по истинным границам.', '',
       '| Модель | Канал | QRS recall | QRS width MAE, мс | T offset MAE, мс |','|---|---|---:|---:|---:|']
for label,p in [('baseline','reports/qtdb_baseline.json'),('consistency','reports/qtdb_transfer.json'),('QT adaptation','reports/qtdb_adapted.json')]:
    if not (ROOT/p).exists():continue
    d=read(p)['summary']
    for c in [0,1]:
        q=d[f'channel{c}/QRS'];t=d[f'channel{c}/T']
        lines.append(f'| {label} | {c} | {fmt(q["recall"])} | {fmt(q["width_mae_ms"])} | {fmt(t["offset"]["mae_ms"])} |')
lines+=['','Точность T-границ в новом домене остаётся ограничением. Улучшение QRS не означает универсального улучшения всех волн.', '', '## Классификация записей', '']
manifest=ROOT/'artifacts/record_features_qt/manifest.csv'
if manifest.exists():
    frame=pd.read_csv(manifest)
    lines+=['Обработанная выборка:', '', '| Источник | Часть | Записей |','|---|---|---:|']
    for (s,part),n in frame.groupby(['source','split']).size().items():lines.append(f'| {s} | {part} | {n} |')
    lines+=['','Пилот: до 500 записей на source/split плюс все редкие положительные train-записи. Это не полный прогон по всем 42 тысячам исходных записей. Patient IDs PTB-XL взяты из официальной таблицы; folds 1–8/9/10. Georgia целиком внешний источник. Совпадение пациентов между источниками исключено по подтверждению владельца. Точные дубликаты сигналов удалены между частями; неизвестные ID не выдаются за проверенные.', '']
if (ROOT/'artifacts/record_models/metrics.json').exists():
    data=read('artifacts/record_models/metrics.json');selection=read('artifacts/record_models/selection.json');best=selection['selected']
    lines+=['| Вариант | PTB-XL test macro AUROC | Georgia macro AUROC |','|---|---:|---:|']
    for m,d in data.items():lines.append(f'| {m} | {fmt(d["test"]["macro_auroc"])} | {fmt(d["external"]["macro_auroc"])} |')
    lines+=['',f'Выбран по validation macro AUROC: **{best}**. Все три варианта сохранены. AUPRC, precision/recall/F1, пороги и число позитивов доступны в JSON и ноутбуке.', '',
       '| Класс | Позитивов PTB test | AUROC | F1 |','|---|---:|---:|---:|']
    for c,v in data[best]['test']['per_class'].items():lines.append(f'| {c} | {v["positive"]} | {fmt(v["auroc"])} | {fmt(v["f1"])} |')
    lines+=['','Если положительных случаев нет, AUROC не определён. Редкие AVB2/AVB3 нельзя считать проверенными по такому тесту. Отдельная модель записи VT не обучена: в training только один положительный пример. Общие метки StPetersburg не подтверждают событие в первых 30 секундах; exploratory_long не включён в главную оценку.', '']
if (ROOT/'artifacts/beat_models/metrics.json').exists():
    data=read('artifacts/beat_models/metrics.json');selection=read('artifacts/beat_models/selection.json');best=selection['selected']
    lines+=['## Beat model ablations', '', '| Model | Validation macro F1 (N,S,V,F) | Test S end-to-end F1 | Test V end-to-end F1 |', '|---|---:|---:|---:|']
    for name, scores in data.items():
        score=np.mean([scores['valid'][c]['f1-score'] for c in ['N','S','V','F']])
        lines.append(f'| {name} | {fmt(score)} | {fmt(scores["test"]["S"]["end_to_end_f1"])} | {fmt(scores["test"]["V"]["end_to_end_f1"])} |')
    lines+=['## Классификация сокращений MIT-BIH','',f'Выбрана на validation: **{best}**. Автоматические R-пики; тестовые аннотации используются только для оценки. 201 и 202 в одной части. Четыре записи со стимулятором исключены.', '',
       '| Класс | Все reference beats test | F1 на matched detections | End-to-end recall | End-to-end precision |','|---|---:|---:|---:|---:|']
    for c in ['N','S','V','F','Q']:
        v=data[best]['test'][c];lines.append(f'| {c} | {v["all_reference_support"]} | {fmt(v["f1-score"])} | {fmt(v["end_to_end_recall"])} | {fmt(v["end_to_end_precision"])} |')
    lines+=['','End-to-end recall учитывает пропущенные и ненарезанные краевые биты; precision дополнительно штрафует классификацию ложных детекций. Q — остаточная неоднородная группа, не отдельная болезнь. Перенос модели сокращений с MLII на другие отведения не валидирован.', '']
if (ROOT/'reports/vt_episodes.json').exists():
    lines+=['## Кандидаты ЖТ', '', '```json',json.dumps(read('reports/vt_episodes.json')['summary'],indent=2),'```', '',
    'Критерий: ≥3 подряд предсказанных V с оценкой ≥0.8 и частотой ≥100/мин. Оценка эпизодов использует любое временное пересечение и является оценкой поиска кандидатов, не диагностической точности механизма тахикардии.']
baseline=ROOT/'configs/local_budget.json'
budgets=json.loads(baseline.read_text())['runs'].copy() if baseline.exists() else {}
for p in [ROOT/'artifacts/delineator.budget.json',ROOT/'artifacts/delineator_transfer.budget.json',ROOT/'artifacts/delineator_qt.budget.json',ROOT/'artifacts/record_models_baseline/budget.json',ROOT/'artifacts/record_models/budget.json',ROOT/'artifacts/beat_models/budget.json',ROOT/'artifacts/beat_models/cnn_budget.json']:
    if p.exists():budgets[str(p.relative_to(ROOT))]=json.loads(p.read_text())
for name in ['context_experiments','representation_experiments','founder_experiments']:
    extra=ROOT/'artifacts'/name/'budget.json'
    if extra.exists():budgets[str(extra.relative_to(ROOT))]=json.loads(extra.read_text())
for pattern in ['hubert_local_*','qwen_local_*']:
    for root in (ROOT/'artifacts').glob(pattern):
        if (root/'budget.json').exists():budgets[str((root/'budget.json').relative_to(ROOT))]=json.loads((root/'budget.json').read_text())
        elif (root/'run.json').exists():budgets[str((root/'run.json').relative_to(ROOT))]={'seconds':json.loads((root/'run.json').read_text())['seconds_this_invocation']}
total=sum(d['seconds'] for d in budgets.values())
lines+=['','## Ресурсы и практические ограничения','',f'Суммарное измеренное время обучения, включая пробный классификатор: **{total/60:.1f} мин**. Предобработка и инференс считаются отдельно. Бюджет обучения 180 минут; 4 ГБ GPU. Разметчик содержит 214 408 параметров.', '',
       '- QRS ≥120 мс — измеряемый признак, не достаточное правило ЖЭС или критичности блокады.',
       '- Форма ФП «впервые выявленная/постоянная» без анамнеза не определяется.',
       '- У части Training_2 потеряны #Dx и изменены заголовки; отсутствующий DAT не подменяется MAT.',
       '- Калибровка PTB-XL/StPetersburg помечена непроверенной; спорные физические амплитуды исключены из признаков.',
       '- Корреляции диагнозов рассчитаны только на train и не интерпретируются причинно.',
       '- Предобработка zero-phase и RR/template-контекст используют будущее внутри записи: это офлайн-анализ.',
       '', 'Источники и основания решений: [research_notes.md](research_notes.md). Запуск и форматы: [PROJECT.md](../PROJECT.md).']
(ROOT/'reports/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT/'reports/training_budget.json').write_text(json.dumps(dict(total_seconds=total,limit_seconds=10800,within_budget=total<=10800,runs=budgets),indent=2),encoding='utf-8')
versions={'python':platform.python_version()}
for p in ['numpy','scipy','pandas','scikit-learn','torch','wfdb','neurokit2','matplotlib','joblib','pytest','nbformat','nbclient']:versions[p]=im.version(p)
(ROOT/'reports/environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
print('Wrote reports/RESULTS.md; total training minutes:',round(total/60,2))
