"""One-time migration to domain packages; preserves legacy pickle imports."""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
groups={
 'data':['io','catalog'],
 'processing':['signal','features'],
 'models':['segmentation','beat_cnn','sequence_predictor'],
 'training':['adaptation','qtdb','beats','record_model','cluster_training','cluster_founder','unlabeled_beats'],
 'evaluation':['metrics','benchmark','robustness','episodes'],
 'experiments':['context_experiments','representation_experiments','founder_experiments'],
 'workflows':['analysis'],
}
mapping={name:group+'.'+name for group,names in groups.items() for name in names}
for group,names in groups.items():
    target=root/'ecg_project'/group;target.mkdir(exist_ok=True)
    (target/'__init__.py').touch()
    for name in names:
        src=root/'ecg_project'/(name+'.py');dest=target/(name+'.py')
        if src.exists() and not dest.exists():src.rename(dest)
files=[]
for folder in ['ecg_project','scripts','tests','notebooks','reports','configs']:
    files.extend(p for p in (root/folder).rglob('*') if p.suffix in ['.py','.md','.ipynb','.json'] and '__pycache__' not in p.parts)
files += [root/'PROJECT.md',root/'CONTINUATION.md']
for path in files:
    if path.name=='reorganize_project.py':continue
    text=path.read_text(encoding='utf-8')
    if path.suffix=='.py' and 'ecg_project' in path.parts:
        text=re.sub(r'from \.(\w+) import',lambda m:'from ecg_project.'+mapping.get(m[1],m[1])+' import',text)
    text=re.sub(r'ecg_project\.(\w+)',lambda m:'ecg_project.'+mapping.get(m[1],m[1]),text)
    path.write_text(text,encoding='utf-8')
for name in ['beat_cnn','sequence_predictor']:
    (root/'ecg_project'/(name+'.py')).write_text('"""Compatibility for already saved joblib artifacts."""\nfrom ecg_project.'+mapping[name]+' import *\n')
print('Organized domain packages; legacy artifact imports retained')
