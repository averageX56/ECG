"""Persistent Colab paths without deleting a checkout, datasets or base models."""
from pathlib import Path
import os


def attach_drive(root='/content/ECG',drive='/content/drive/MyDrive/ECG_DATA'):
    root=Path(root);drive=Path(drive)
    for name in ('artifacts','reports','data','LUDB'):
        src=drive/name;dst=root/name
        if name in ('artifacts','reports'):src.mkdir(parents=True,exist_ok=True)
        elif not src.is_dir():continue  # Optional raw datasets; GPU preparation checks them explicitly.
        if dst.is_symlink():
            if dst.resolve()==src.resolve():continue
            raise FileExistsError(f'{dst} points to a different location; preserve its data before changing the link')
        if dst.exists():
            if not dst.is_dir() or any(dst.iterdir()):raise FileExistsError(f'Nonempty local path {dst}; copy its contents to {src} before attaching Drive')
            dst.rmdir()
        os.symlink(src,dst,target_is_directory=True)
    return root


def prepare_qwen_pseudo_gpu(output='artifacts/qwen_pseudo_extended',
        sources=('CPSC_EXTRA','PTBXL','CPSC','CHAPMAN','NINGBO'),
        checkpoint='artifacts/cluster/delineator_qt.pt',batch_size=256,workers=2,tau=.95):
    from ecg_project.data.cache import require_checkpoint
    require_checkpoint(checkpoint)
    if not Path('artifacts/catalog.csv').is_file():
        raise FileNotFoundError('Missing artifacts/catalog.csv. Upload the audited catalog to ECG_DATA/artifacts on Drive.')
    from ecg_project.data.policy import build_record_manifest
    frame=build_record_manifest('artifacts/catalog.csv',sources)
    missing=[p for p in frame.path if not Path(p).is_file()]
    if missing or not Path('LUDB').is_dir() or not Path('data/qtdb_external/split.json').is_file():
        raise FileNotFoundError('GPU pseudo generation needs raw ECG files and the holdout identity registry. '
            'Attach ECG_DATA/data -> /content/ECG/data and ECG_DATA/LUDB -> /content/ECG/LUDB. '
            f'First missing headers: {missing[:3]}')
    from ecg_project.data.qwen_pseudo import prepare
    return prepare(output=output,sources=sources,checkpoint=checkpoint,device='cuda',batch_size=batch_size,workers=workers,tau=tau)
