"""Read-only readiness check before opening the unified cluster notebook."""
from pathlib import Path
import argparse
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--profile',choices=['smoke','bert','founder','delineation','lora','qlora','qwen','final'],default='final')
args=parser.parse_args()
required={'smoke':['artifacts/beat_features_v2/manifest.json','LUDB/1.hea','artifacts/cluster/delineator_qt.pt'],
          'bert':['artifacts/beat_features_v2/manifest.json'],
          'founder':['artifacts/founder_inputs_v2/signals.npy','artifacts/founder_inputs_v2/provenance.json','artifacts/ecgfounder/1_lead_ECGFounder.pth','artifacts/ecgfounder/net1d.py'],
          'delineation':['artifacts/delineator_inputs_v2/provenance.json','artifacts/delineator_inputs_v2/train_x.npy'],
          'lora':['artifacts/hubert_inputs_v2/signals.npy','artifacts/hubert_inputs_v2/manifest.csv','artifacts/hubert_inputs_v2/provenance.json','artifacts/hubert_large/model.safetensors','artifacts/hubert_large/config.json'],
          'qlora':['artifacts/hubert_inputs_v2/signals.npy','artifacts/hubert_inputs_v2/manifest.csv','artifacts/hubert_inputs_v2/provenance.json','artifacts/hubert_large/model.safetensors','artifacts/hubert_large/config.json'],
          'qwen':['artifacts/qwen_delineation_inputs_v2/provenance.json','artifacts/qwen_delineation_inputs_v2/train_x.npy','artifacts/qwen_delineation_inputs_v2/train_y.npy','artifacts/qwen_delineation_inputs_v2/valid_x.npy','artifacts/qwen_delineation_inputs_v2/valid_y.npy'],
          'final':['configs/final_pipeline.json']}[args.profile]
missing=[p for p in required if not (ROOT/p).exists()]
if args.profile=='final' and not missing:
    import hashlib
    for name,item in json.loads((ROOT/required[0]).read_text())['models'].items():
        p=ROOT/item['path']
        if not p.exists():missing.append(str(p))
        else:
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                while chunk:=stream.read(2**20):digest.update(chunk)
            if digest.hexdigest()!=item['sha256']:missing.append(name+': SHA256 mismatch')
if args.profile in ['smoke','bert'] and not missing:
    manifest=json.loads((ROOT/required[0]).read_text())
    missing += [str(Path(required[0]).parent/(r['record']+'.npz')) for r in manifest
                if r['split'] in ['train','valid'] and not (ROOT/Path(required[0]).parent/(r['record']+'.npz')).exists()]
import torch
gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None
vram=round(torch.cuda.get_device_properties(0).total_memory/2**30,2) if gpu else 0
if args.profile in ['bert','founder','lora','qlora','qwen'] and vram<35:missing.append('GPU with >=35 GiB VRAM for the configured A100 profile')
if args.profile in ['lora','qlora','qwen']:
    import importlib.util
    for package in ['transformers','safetensors']:
        if importlib.util.find_spec(package) is None:missing.append('Optional lora dependency: '+package)
    if args.profile in ['qlora','qwen'] and importlib.util.find_spec('bitsandbytes') is None:missing.append('Optional qlora dependency: bitsandbytes')
print(json.dumps(dict(profile=args.profile,root=str(ROOT),python=sys.version.split()[0],torch=torch.__version__,gpu=gpu,vram_gib=vram,missing=missing,ready=not missing),indent=2))
sys.exit(bool(missing))
