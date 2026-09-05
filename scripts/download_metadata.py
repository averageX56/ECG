"""Download metadata only, never ECG signal archives."""
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
URLS={
 'artifacts/dx_mapping_scored.csv':'https://raw.githubusercontent.com/physionetchallenges/evaluation-2020/master/dx_mapping_scored.csv',
 'artifacts/dx_mapping_unscored.csv':'https://raw.githubusercontent.com/physionetchallenges/evaluation-2020/master/dx_mapping_unscored.csv',
 'artifacts/ptbxl_database.csv':'https://physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv',
}
for p in (ROOT/'data/mit-bih').glob('*.csv'):
    URLS[f'artifacts/mit_headers/{p.stem}.hea']=f'https://physionet.org/files/mitdb/1.0.0/{p.stem}.hea'
for i in [1,2,100,1000,20000]:
    folder=(i//1000)*1000
    URLS[f'artifacts/ptb_headers/{i:05d}_hr.hea']=f'https://physionet.org/files/ptb-xl/1.0.1/records500/{folder:05d}/{i:05d}_hr.hea'
for i in [1,2,3]:
    URLS[f'artifacts/incart_headers/I{i:02d}.hea']=f'https://physionet.org/files/incartdb/1.0.0/I{i:02d}.hea'

def download(item):
    name,url=item;p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(url,timeout=60) as r: payload=r.read()
        p.write_bytes(payload)
    return name

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(download,URLS.items()):print(name,flush=True)
