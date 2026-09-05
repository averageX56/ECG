from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
for p in sorted((ROOT/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(p,as_version=4)
    NotebookClient(nb,timeout=300,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,p)
    print('Executed',p.name,flush=True)
