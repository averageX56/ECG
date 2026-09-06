"""Explicit pinned tensor-only download of Qwen3; no remote code is executed."""
import argparse
import json
from pathlib import Path
import urllib.request
import hashlib


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(2**20):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',choices=['0.6B','1.7B','4B','8B'],default='1.7B')
    args=p.parse_args();repo='Qwen/Qwen3-'+args.size
    root=Path('artifacts/qwen3_'+args.size.lower());root.mkdir(parents=True,exist_ok=True)
    def get(url,path):
        if path.exists():return
        print('Downloading',path.name,flush=True)
        temporary=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as stream:
            while chunk:=response.read(2**20):stream.write(chunk)
        temporary.replace(path)
    get('https://huggingface.co/api/models/'+repo,root/'model_info.json')
    info=json.loads((root/'model_info.json').read_text());revision=info['sha']
    for entry in info['siblings']:
        name=entry['rfilename']
        if '/' not in name and (name in ('config.json','README.md','LICENSE','model.safetensors.index.json') or name.endswith('.safetensors')):
            get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',root/name)
    manifest=dict(model_id=repo,revision=revision,sha256={p.name:file_hash(p) for p in root.iterdir()
        if p.is_file() and p.name!='provenance.json' and not p.name.endswith('.partial')})
    (root/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Pinned',repo,revision,flush=True)


if __name__=='__main__':main()
