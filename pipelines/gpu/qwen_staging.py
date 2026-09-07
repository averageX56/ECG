"""Qwen-only local SSD snapshots and publish-last Drive synchronization."""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import shutil
from tqdm.auto import tqdm
from ecg_project.data.cache import atomic_json,validate_cache
from ecg_project.data.catalog import file_hash

EXTENDED_QWEN_SOURCES=('CPSC_EXTRA','PTBXL','CPSC','CHAPMAN')
RAW_DIRS=dict(CPSC_EXTRA='data/Training_2',PTBXL='data/WFDB_PTB-XL',CPSC='data/Training_WFDB',CHAPMAN='data/WFDB_ChapmanShaoxing')


def _inventory(root):
    return {p.relative_to(root).as_posix():[p.stat().st_size,p.stat().st_mtime_ns]
            for p in root.rglob('*') if p.is_file()}


def copy_snapshot(src,dst,control,verify_sources=False):
    """Owned scratch only. Complete immutable snapshots reuse local stat checks."""
    src=Path(src);dst=Path(dst);control=Path(control)
    owner=control.with_suffix('.owner.json');identity=dict(source=str(src.resolve()),destination=str(dst.absolute()))
    if owner.exists():
        if json.loads(owner.read_text())!=identity:raise ValueError(f'Staging identity mismatch: {dst}')
    else:
        if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):raise FileExistsError(f'Nonempty unrelated staging directory: {dst}')
        atomic_json(owner,identity)
    if control.exists():
        saved=json.loads(control.read_text());local=_inventory(dst)
        if local!=saved['stats']:raise ValueError(f'Staged snapshot changed: {dst}; choose new scratch')
        if verify_sources:
            if _inventory(src)!=saved['stats'] or any(file_hash(src/n)!=h for n,h in saved['sha256'].items()):
                raise ValueError(f'Drive source changed: {src}; choose new scratch')
            if any(file_hash(dst/n)!=h for n,h in saved['sha256'].items()):raise ValueError(f'Local snapshot hash mismatch: {dst}')
        print(f'Reusing complete local snapshot: {dst} ({len(local)} files)',flush=True)
        return
    if not src.is_dir():raise FileNotFoundError(f'Missing Drive dataset: {src}')
    print(f'Raw staging inventory: {src}',flush=True)
    files=[p for p in tqdm(src.rglob('*'),desc=f'Inventory {src.name}',unit='file') if p.is_file()]
    stats={};hashes={}
    for p in tqdm(files,desc=f'Stage {src.name}',unit='file'):
        name=p.relative_to(src).as_posix();target=dst/name
        before=[p.stat().st_size,p.stat().st_mtime_ns]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)  # One Drive read; hashing happens on local SSD.
        if before!=[p.stat().st_size,p.stat().st_mtime_ns]:raise ValueError(f'Source changed during staging: {p}')
        stats[name]=[target.stat().st_size,target.stat().st_mtime_ns];hashes[name]=file_hash(target)
    atomic_json(control,dict(**identity,stats=stats,sha256=hashes))


def stage_qwen_raw(root='/content/ECG',drive='/content/drive/MyDrive/ECG_DATA',
                   scratch='/content/ecg_qwen/raw',sources=EXTENDED_QWEN_SOURCES,verify_sources=False):
    from ecg_project.data.policy import build_record_manifest
    root=Path(root);drive=Path(drive);scratch=Path(scratch)
    if scratch.resolve().is_relative_to(drive.resolve()):raise ValueError('Raw staging must be outside Drive')
    if not set(sources)<=set(EXTENDED_QWEN_SOURCES):raise ValueError('Qwen pseudo sources must exclude Ningbo and unknown sources')
    required={RAW_DIRS[s] for s in sources}|{'data/qtdb_external','LUDB'}
    frame=build_record_manifest(root/'artifacts/catalog.csv')
    for row in frame.itertuples():
        if row.split!='train' and row.readable:
            path=Path(row.path)
            if path.is_absolute() or '..' in path.parts or path.parts[0]!='data':raise ValueError(f'Unsupported raw catalog path: {path}')
            required.add('/'.join(path.parts[:2]))
    # Check all inputs and destinations before beginning a potentially large copy.
    for name in sorted(required):
        if not (drive/name).is_dir():raise FileNotFoundError(f'Required raw/holdout dataset missing on Drive: {drive/name}')
        dest=scratch/name;owner=scratch/'.staging'/name.replace('/','_')
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())) and not owner.with_suffix('.owner.json').exists():
            raise FileExistsError(f'Nonempty unrelated staging directory: {dest}')
    for name in sorted(required):
        copy_snapshot(drive/name,scratch/name,scratch/'.staging'/(name.replace('/','_')+'.json'),verify_sources)
    return scratch


@contextmanager
def local_raw_links(root,scratch):
    """Restore Drive links on success or interruption; never replace real data dirs."""
    root=Path(root);scratch=Path(scratch);previous={}
    for name in ('data','LUDB'):
        p=root/name
        if p.exists() and not p.is_symlink():raise FileExistsError(f'Refusing to replace unrelated real directory: {p}')
        if not (scratch/name).is_dir():raise FileNotFoundError(scratch/name)
    try:
        for name in ('data','LUDB'):
            p=root/name;previous[name]=os.readlink(p) if p.is_symlink() else None
            if p.is_symlink():p.unlink()
            os.symlink((scratch/name).resolve(),p,target_is_directory=True)
        yield
    finally:
        for name,target in previous.items():
            p=root/name
            if p.is_symlink() and p.resolve()==(scratch/name).resolve():p.unlink()
            elif p.exists():raise RuntimeError(f'Raw link changed during Qwen preparation: {p}')
            if target is not None:os.symlink(target,p,target_is_directory=True)


def sync_completed_cache(local,destination):
    """One final sync phase. The public destination appears only after validation."""
    local=Path(local);destination=Path(destination)
    meta=validate_cache(local)  # Interrupted local outputs must not create a Drive cache.
    digest=file_hash(local/'provenance.json')
    if destination.exists():
        if (destination/'provenance.json').is_file() and file_hash(destination/'provenance.json')==digest:
            validate_cache(destination);return destination
        if not destination.is_dir() or any(destination.iterdir()):raise FileExistsError(f'Existing different Drive cache: {destination}; use a new output')
    pending=destination.with_name(destination.name+'.syncing');owner=pending/'.sync_identity.json'
    if pending.exists() and any(pending.iterdir()) and not owner.is_file():raise FileExistsError(f'Unrelated sync directory: {pending}')
    if owner.exists() and json.loads(owner.read_text())!={'provenance_sha256':digest}:raise ValueError('Interrupted sync belongs to another cache')
    atomic_json(owner,dict(provenance_sha256=digest))
    files=[p for p in local.rglob('*') if p.is_file() and p.name!='provenance.json' and not p.name.endswith('.partial')]
    for p in tqdm(files,desc='Final sync to persistent storage',unit='file'):
        target=pending/p.relative_to(local);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(local/'provenance.json',pending/'provenance.json')
    validate_cache(pending)
    if destination.exists():destination.rmdir()  # Only an empty directory, checked above.
    pending.replace(destination)
    return destination
