"""Accumulate local invocations without overwriting time on checkpoint resume."""
import json
from pathlib import Path
from ecg_project.utils import save_json


def read_ledger():
    path=Path('reports/training_budget.json')
    if not path.exists():path=Path('configs/local_budget.json')
    return json.loads(path.read_text())


def check_reservation(minutes):
    if minutes<=0:raise ValueError('Positive local time reservation required')
    ledger=read_ledger()
    # Exclude dynamically added run budgets, then include their current values once.
    spent=sum(v['seconds'] for k,v in ledger['runs'].items() if 'hubert_local_' not in k and 'qwen_local_' not in k)
    for pattern in ('hubert_local_*','qwen_local_*'):
        for root in Path('artifacts').glob(pattern):
            if (root/'budget.json').exists():spent+=json.loads((root/'budget.json').read_text())['seconds']
            elif (root/'run.json').exists():spent+=json.loads((root/'run.json').read_text())['seconds_this_invocation']
    if spent+minutes*60>10800:raise RuntimeError('Local reservation exceeds 180 minutes; ask user first')


def record_invocation(root,seconds,local):
    if not local:return
    path=Path(root)/'budget.json'
    old=json.loads(path.read_text())['seconds'] if path.exists() else 0.
    if not path.exists() and (Path(root)/'run.json').exists():old=json.loads((Path(root)/'run.json').read_text())['seconds_this_invocation']
    save_json(path,dict(seconds=old+seconds,local=True))
