"""Explicit pinned tensor-only download of Qwen3; no remote code is executed."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error
import hashlib
import http.client
import re
import time


def download(url, path, attempts=8, timeout=300):
    """Retry interrupted reads and resume a pinned file using HTTP byte ranges."""
    path = Path(path)
    if path.exists():
        return path
    if attempts < 1 or timeout <= 0:
        raise ValueError('Positive attempts and timeout required')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    restart = False
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() and not restart else 0
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        print(f'Downloading {path.name}: attempt {attempt+1}/{attempts}, offset {offset:,} bytes', flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                length = response.headers.get('Content-Length')
                expected = int(length) if length is not None else None
                total = None
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match:
                        raise ValueError(f'Invalid Content-Range for {path.name}')
                    start, end, total = map(int, match.groups())
                    if start != offset or end < start or end >= total:
                        raise ValueError(f'Unexpected resume range for {path.name}')
                    if expected is not None and expected != end-start+1:
                        raise ValueError(f'Conflicting response lengths for {path.name}')
                    expected = end-start+1
                elif status == 200:
                    # The server may ignore Range. Replace the partial stream
                    # with its full response instead of appending duplicate bytes.
                    offset = 0
                    total = expected
                else:
                    raise ValueError(f'Unexpected HTTP status {status} for {path.name}')
                restart = False
                received = 0
                with temporary.open('ab' if offset else 'wb') as stream:
                    while chunk := response.read(2**20):
                        stream.write(chunk)
                        received += len(chunk)
                if expected is not None and received != expected:
                    raise http.client.IncompleteRead(b'', max(0, expected-received))
                if total is not None and temporary.stat().st_size != total:
                    raise http.client.IncompleteRead(b'', max(0, total-temporary.stat().st_size))
            temporary.replace(path)
            return path
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                # Validate via a fresh full response rather than trusting a
                # possibly stale/oversized partial file solely by its length.
                restart = True
            elif exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            error = exc
            exc.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            error = exc
        if attempt+1 == attempts:
            raise RuntimeError(f'Download failed after {attempts} attempts: {path}. '
                               f'Partial file preserved at {temporary}; rerun to resume.') from error
        delay = min(2**(attempt+1), 30)
        print(f'Download interrupted: {error}; retrying in {delay}s', flush=True)
        time.sleep(delay)


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(2**20):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size',choices=['0.6B','1.7B','4B','8B'],default='1.7B')
    p.add_argument('--attempts',type=int,default=8)
    p.add_argument('--timeout',type=float,default=300,help='Socket timeout in seconds')
    args=p.parse_args();repo='Qwen/Qwen3-'+args.size
    root=Path('artifacts/qwen3_'+args.size.lower());root.mkdir(parents=True,exist_ok=True)
    def get(url,path):
        return download(url,path,attempts=args.attempts,timeout=args.timeout)
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
