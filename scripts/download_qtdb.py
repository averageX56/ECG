"""Download a fixed ~12MB QTDB subset, without MIT-BIH-origin records."""
from pathlib import Path
import urllib.request
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
TRAIN=['sel16265','sel16272','sel16273','sel16420','sel16483','sel16539','sel16773','sel16786']
VALID=['sele0104','sele0106','sele0107','sele0110','sele0111','sele0112','sele0114','sele0116','sele0121','sele0122']
BASE='https://physionet.org/files/qtdb/1.0.0/'

def main():
    out=ROOT/'data/qtdb_external';out.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(BASE+'SHA256SUMS.txt',timeout=60) as r:checks=r.read()
    (out/'SHA256SUMS.txt').write_bytes(checks)
    expected={line.split()[-1].lstrip('*'):line.split()[0] for line in checks.decode().splitlines() if line.strip()}
    def get(name):
        p=out/name
        if not p.exists():
            with urllib.request.urlopen(BASE+name,timeout=60) as r:data=r.read()
            p.write_bytes(data)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if name not in expected or digest!=expected[name]:raise ValueError('SHA256 mismatch: '+name)
        return name
    names=[r+ext for r in TRAIN+VALID for ext in ['.hea','.dat','.q1c','.man']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name in pool.map(get,names):print(name,flush=True)
    rows=[dict(record_id=r,split='adapt_train' if r in TRAIN else 'external_valid',
           origin='MIT-BIH Normal Sinus Rhythm' if r in TRAIN else 'European ST-T',
           path=f'data/qtdb_external/{r}.hea') for r in TRAIN+VALID]
    (out/'split.json').write_text(json.dumps(dict(records=rows,source=BASE,
       manual_annotation='q1c',excluded='All MIT-BIH Arrhythmia-origin QT records',
       split_protocol='Source-disjoint adaptation vs external validation; fixed before model evaluation',
       license='Open Data Commons Attribution License v1.0'),indent=2),encoding='utf-8')

if __name__=='__main__':main()
