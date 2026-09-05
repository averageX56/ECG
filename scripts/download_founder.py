"""Download official ECGFounder source and single-lead checkpoint for review."""
from pathlib import Path
import urllib.request
import json
import hashlib

out=Path('artifacts/ecgfounder');out.mkdir(parents=True,exist_ok=True)
def get(url,path):
    if path.exists(): return
    print('Downloading',url,flush=True)
    with urllib.request.urlopen(url,timeout=90) as response, path.with_suffix(path.suffix+'.partial').open('wb') as stream:
        while block:=response.read(1024*1024):stream.write(block)
    path.with_suffix(path.suffix+'.partial').replace(path)

get('https://api.github.com/repos/NickLJLee/ECGFounder/contents',out/'repository.json')
for item in json.loads((out/'repository.json').read_text()):
    if item['type']=='file' and item['name'].endswith(('.py','.md')):
        get(item['download_url'],out/item['name'])
get('https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/1_lead_ECGFounder.pth',out/'1_lead_ECGFounder.pth')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='sha256.json'}
(out/'sha256.json').write_text(json.dumps(hashes,indent=2))
print('Download complete',flush=True)
