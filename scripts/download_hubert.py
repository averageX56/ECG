"""Pin and download the author's SSL HuBERT-ECG Large checkpoint and source."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

parser=argparse.ArgumentParser()
parser.add_argument('--weights',action='store_true')
args=parser.parse_args()
root=Path('artifacts/hubert_large');root.mkdir(parents=True,exist_ok=True)
repo='Edoardo-Coppola/hubert-ecg-large'
def get(url,path):
    if path.exists():return
    print('Download',path.name,flush=True)
    with urllib.request.urlopen(url,timeout=120) as response,path.with_suffix(path.suffix+'.partial').open('wb') as stream:
        while chunk:=response.read(2**20):stream.write(chunk)
    path.with_suffix(path.suffix+'.partial').replace(path)
get('https://huggingface.co/api/models/'+repo,root/'model_info.json')
info=json.loads((root/'model_info.json').read_text());revision=info['sha']
for entry in info['siblings']:
    name=entry['rfilename']
    if '/' in name:continue
    if name.endswith(('.json','.py','.md')) or args.weights and name.endswith('.safetensors'):
        get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',root/name)
source=root/'upstream';source.mkdir(exist_ok=True)
for name in ['hubert_ecg/dataset.py','hubert_ecg/utils.py','README.md','LICENSE']:
    dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True)
    get(f'https://raw.githubusercontent.com/Edoar-do/HuBERT-ECG/master/{name}',dest)
manifest={'model_id':repo,'revision':revision,'source_revision':'master snapshot; exact files identified by SHA256',
    'sha256':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and p.name not in ['provenance.json'] and not p.name.endswith('.partial')}}
(root/'provenance.json').write_text(json.dumps(manifest,indent=2))
print('Pinned',revision,flush=True)
