"""Read-only readiness check before opening the unified cluster notebook."""
from pathlib import Path
import argparse
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--profile',choices=['smoke','bert','founder','delineation','lora'],default='smoke')
args=parser.parse_args()
required={'smoke':['artifacts/beat_features_qt/manifest.json','LUDB/1.hea','artifacts/delineator_qt.pt'],
          'bert':['artifacts/beat_features_qt/manifest.json'],
          'founder':['artifacts/catalog.csv','data/WFDB_PTB-XL','artifacts/ecgfounder/1_lead_ECGFounder.pth','artifacts/ecgfounder/net1d.py'],
          'delineation':['LUDB','data/qtdb_external','artifacts/catalog.csv'],
          'lora':['artifacts/catalog.csv','data/WFDB_PTB-XL','artifacts/hubert_large/model.safetensors','artifacts/hubert_large/config.json']}[args.profile]
missing=[p for p in required if not (ROOT/p).exists()]
if args.profile in ['smoke','bert'] and not missing:
    manifest=json.loads((ROOT/required[0]).read_text())
    missing += [str(Path(required[0]).parent/(r['record']+'.npz')) for r in manifest
                if r['split'] in ['train','valid'] and not (ROOT/Path(required[0]).parent/(r['record']+'.npz')).exists()]
import torch
gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None
vram=round(torch.cuda.get_device_properties(0).total_memory/2**30,2) if gpu else 0
if args.profile in ['bert','founder','lora'] and vram<35:missing.append('GPU with >=35 GiB VRAM for the configured A100 profile')
if args.profile=='lora':
    import importlib.util
    for package in ['transformers','safetensors']:
        if importlib.util.find_spec(package) is None:missing.append('Optional lora dependency: '+package)
print(json.dumps(dict(profile=args.profile,root=str(ROOT),python=sys.version.split()[0],torch=torch.__version__,gpu=gpu,vram_gib=vram,missing=missing,ready=not missing),indent=2))
sys.exit(bool(missing))
