from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
def make(name,cells):
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if kind=='md' else nbf.v4.new_code_cell(s) for kind,s in cells])
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nb.metadata.language_info={'name':'python','version':'3.13'}
    nbf.write(nb,ROOT/'notebooks'/name)

SETUP="""from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
if not (ROOT / 'ecg_project').exists(): ROOT = ROOT.parent
os.chdir(ROOT)
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
"""

make('01_data_audit.ipynb',[
 ('md','# Датасеты и контроль утечек\nВоспроизводимый обзор фактических файлов. Исходные данные и прежний ноутбук не изменяются.'),
 ('code',SETUP),
 ('code',"catalog = pd.read_csv('artifacts/catalog.csv').fillna('')\ncatalog.groupby('source').agg(records=('record_id','size'), readable=('readable','sum'), labeled=('has_labels','sum'), min_seconds=('duration_s','min'), max_seconds=('duration_s','max'))"),
 ('md','Отсутствие #Dx означает неизвестную метку. Типы сокращений MIT-BIH не являются диагнозами всей записи. LUDB и data/ludb_beats — один источник, не независимые выборки.'),
 ('code',"from ecg_project.data.catalog import TARGETS\ncatalog[catalog.has_labels].groupby('source')[list(TARGETS)].sum()"),
 ('code',"from ecg_project.data.catalog import ludb_split, assert_disjoint\nfrom ecg_project.training.beats import split_for\nassert split_for('201') == split_for('202')\nsplits = ludb_split(range(1,201))\npd.Series(splits).value_counts()"),
 ('code',"f = pd.read_csv('artifacts/record_features_qt/manifest.csv').fillna('')\nassert_disjoint(f)\nf.groupby(['source','split']).size().unstack(fill_value=0)"),
 ('md','Основной PTB-XL split использует официальные patient_id. Georgia целиком внешний источник; совпадения пациентов между источниками исключены по подтверждению владельца данных. Пилот ограничен 500 записями на source/split, с сохранением всех редких train-позитивов. Это не полное обучение на всём архиве.'),
 ('code',"from ecg_project.data.io import load_record, annotations\nr=load_record('LUDB/1.hea')\nprint(r.fs, r.signal.shape, r.leads, r.unit)\npd.DataFrame(annotations('LUDB/1.hea','II')).head()"),
 ('md','Сведения о спорной калибровке PTB-XL/StPetersburg и изменённых заголовках Training_2: ../reports/research_notes.md. Амплитуды из спорной калибровки не используются как физические признаки.')
])

make('02_delineation_transfer.ipynb',[
 ('md','# Интервалы, предобработка и перенос\nP/QRS/T, ошибка ширины QRS, сравнение с DWT и независимым QTDB. Порог расширения QRS: ≥120 мс.'),
 ('code',SETUP),
 ('code',"rows=[]\nfor name,path in [('LUDB baseline','reports/segmentation_valid.json'),('consistency','reports/segmentation_transfer_valid.json'),('QT adaptation','reports/segmentation_qt_valid.json')]:\n    d=read_json(path)\n    for wave,m in d['summary'].items(): rows.append(dict(model=name,wave=wave,f1=m['f1'],onset_mae_ms=m['onset']['mae_ms'],offset_mae_ms=m['offset']['mae_ms'],qrs_width_mae_ms=d['qrs_width']['mae_ms']))\npd.DataFrame(rows)"),
 ('md','Сравнение выше — validation. Исходный и адаптированный test показываются для оценки, а не подбора параметров. Ошибка ширины условна на сопоставленных QRS; пропуски входят в event recall.'),
 ('code',"d=read_json('reports/segmentation_qt_test.json')\nprint('QRS width:',d['qrs_width'])\npd.DataFrame({k:{q:v[q] for q in ['precision','recall','f1']} for k,v in d['summary'].items()}).T"),
 ('code',"from ecg_project.workflows.analysis import analyze\nresult=analyze('LUDB/1.hea',output='reports/notebook_ludb',checkpoint='artifacts/delineator_qt.pt')\nfrom IPython.display import Image, display\ndisplay(Image(filename='reports/notebook_ludb/delineation.png'))\npd.read_csv('reports/notebook_ludb/intervals_predicted.csv').query(\"lead == 'II'\").head(12)"),
 ('md','QTDB: восемь NSR записей для адаптации, десять European ST-T только для внешней проверки. Разметка q1c частичная и совместная для двух каналов. Неизвестные T-onset не заполняются. Precision/F1 всей записи по этим данным неидентифицируемы; не выбираем лучший канал по истинной разметке.'),
 ('code',"rows=[]\nfor model,path in [('baseline','reports/qtdb_baseline.json'),('consistency','reports/qtdb_transfer.json'),('QT adaptation','reports/qtdb_adapted.json')]:\n    for key,m in read_json(path)['summary'].items(): rows.append(dict(model=model,channel_wave=key,recall=m['recall'],onset_mae_ms=m['onset']['mae_ms'],offset_mae_ms=m['offset']['mae_ms'],onset_support=m['onset']['n']))\npd.DataFrame(rows)"),
 ('code',"robust=read_json('reports/robustness.json')\nrows=[dict(model=Path(model).stem,condition=key,f1=m['f1']) for model,conditions in robust['results'].items() for key,m in conditions.items()]\npd.DataFrame(rows).pivot(index='condition',columns='model',values='f1').plot.bar(figsize=(13,5)); plt.ylabel('Event F1'); plt.tight_layout(); plt.show()"),
 ('md','Обучение запускается явно через CLI: `python -m ecg_project train-delineator`, `adapt-delineator`, `train-qt`. Повторное обучение требует нового каталога признаков: кеш защищён хешем разметчика.')
])

make('03_classification.ipynb',[
 ('md','# Классификация: интервалы, нарезанные ЭКГ и объединение\nВероятностные выходы class-balanced boosting — модельные оценки, не клинически откалиброванные вероятности. Порог выбирается на validation; редкие классы без достаточного validation support используют 0.5 с явным ограничением.'),
 ('code',SETUP),
 ('code',"m=read_json('artifacts/record_models/metrics.json')\nrows=[]\nfor mode,splits in m.items():\n    for split,metrics in splits.items():\n        if split=='external_long':continue\n        rows.append(dict(mode=mode,split=split,macro_auroc=metrics['macro_auroc'],macro_f1=metrics['macro_f1']))\npd.DataFrame(rows)"),
 ('code',"selection=read_json('artifacts/record_models/selection.json'); print(selection)\npd.DataFrame(m[selection['selected']]['test']['per_class']).T"),
 ('md','AUROC отсутствует, если в выборке нет положительного или отрицательного класса. Общую метку 30-минутной StPetersburg записи нельзя считать подтверждением события в первых 30 секундах; exploratory_long не входит в основной вывод о качестве.'),
 ('code',"co=read_json('artifacts/record_models/label_dependencies.json')\nfig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(co['conditional'],vmin=0,vmax=1,cmap='Blues'); ax.set_xticks(range(len(co['classes'])),co['classes'],rotation=45); ax.set_yticks(range(len(co['classes'])),co['classes']); ax.set_title('P(recorded label j | recorded label i), train only'); fig.colorbar(im); plt.tight_layout(); plt.show()"),
 ('md','Это совместная встречаемость записанных меток. Она включает особенности кодирования источников и не доказывает причинную связь заболеваний.'),
 ('code',"b=read_json('artifacts/beat_models/metrics.json')\nrows=[]\nfor mode,splits in b.items():\n    for split,v in splits.items():\n        for c in ['N','S','V','F','Q']:\n            rows.append(dict(mode=mode,split=split,beat_class=c,f1_matched=v[c]['f1-score'],recall_end_to_end=v[c]['end_to_end_recall'],precision_end_to_end=v[c]['end_to_end_precision'],support=v[c]['all_reference_support']))\npd.DataFrame(rows)"),
 ('code',"best=read_json('artifacts/beat_models/selection.json')['selected']\ncm=np.array(b[best]['test']['confusion_matrix']); fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(cm/np.maximum(cm.sum(1,keepdims=True),1),vmin=0,vmax=1,cmap='Blues'); names=['N','S','V','F','Q']; ax.set_xticks(range(5),names); ax.set_yticks(range(5),names); ax.set_xlabel('Predicted'); ax.set_ylabel('Reference'); ax.set_title('MIT-BIH test, matched detections'); fig.colorbar(im); plt.show()"),
 ('md','N включает нормально проведённые/блокадные и некоторые escape сокращения; S — суправентрикулярная эктопия; V — метка V; F — fusion; Q — прочие/неизвестные, включая ventricular escape E. Широкий QRS не является достаточным правилом ЖЭС. Классификатор битов проверен на MLII; перенос на другие отведения отдельно помечается как непроверенный.'),
 ('code',"p=Path('reports/vt_episodes.json')\nread_json(p) if p.exists() else {'status':'run python -m ecg_project evaluate-vt'}"),
 ('md','Форма ФП впервые выявленная/постоянная требует анамнеза. Гемодинамическая критичность блокады требует клинического контекста. Проект выдаёт измерения и модельные признаки, не подменяя отсутствующие сведения.')
])

print('Created 3 notebooks')
