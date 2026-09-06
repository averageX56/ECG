"""Pinned end-to-end research pipeline selected by validation, not smoke scores."""
from pathlib import Path
import json
from ecg_project.data.catalog import file_hash
from ecg_project.workflows.analysis import analyze


def run(path,output='reports/final_example',pipeline='configs/final_pipeline.json',
        csv_fs=None,lead=None,start_seconds=0,duration_seconds=None,device='cpu'):
    config=json.loads(Path(pipeline).read_text(encoding='utf-8'))
    for name,item in config['models'].items():
        p=Path(item['path'])
        if not p.exists():raise FileNotFoundError(f'{name}: transfer required artifact {p}; see docs/FINAL_PIPELINE.md')
        if file_hash(p)!=item['sha256']:raise ValueError(f'{name}: artifact differs from the selected model; create a new pipeline manifest after validation')
    models=config['models']
    return analyze(path,output=output,checkpoint=models['delineator']['path'],
        beat_model_path=models['beat_classifier']['path'],record_model_path=models['record_classifier']['path'],
        csv_fs=csv_fs,lead=lead,start_seconds=start_seconds,duration_seconds=duration_seconds,device=device)
