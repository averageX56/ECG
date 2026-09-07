"""Persistent Colab paths without deleting a checkout, datasets or base models."""
from pathlib import Path
import os
EXTENDED_QWEN_SOURCES=('CPSC_EXTRA','PTBXL','CPSC','CHAPMAN')


def stage_qwen_raw(*args,**kwargs):
    # Bootstrap imports attach_drive before installing WFDB/numerical dependencies.
    from pipelines.gpu.qwen_staging import stage_qwen_raw as stage
    return stage(*args,**kwargs)


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
        sources=EXTENDED_QWEN_SOURCES,
        checkpoint='artifacts/cluster/delineator_qt.pt',batch_size=64,workers=8,tau=.95,
        root='/content/ECG',drive='/content/drive/MyDrive/ECG_DATA',scratch='/content/ecg_qwen',
        verify_sources=False):
    from ecg_project.data.cache import require_checkpoint
    from pipelines.gpu.qwen_staging import local_raw_links,sync_completed_cache
    import shutil
    root=Path(root);drive=Path(drive);scratch=Path(scratch)
    if scratch.resolve().is_relative_to(drive.resolve()):raise ValueError('Qwen scratch must be on local SSD, outside Drive')
    require_checkpoint(checkpoint)
    if not Path('artifacts/catalog.csv').is_file():
        raise FileNotFoundError('Missing artifacts/catalog.csv. Upload the audited catalog to ECG_DATA/artifacts on Drive.')
    staged=stage_qwen_raw(root,drive,scratch/'raw',sources,verify_sources)
    scratch.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'artifacts/catalog.csv',scratch/'catalog.csv')
    # No shard/registry/report writes go through the artifacts -> Drive symlink.
    local=scratch/'caches'/Path(output).name
    from ecg_project.data.qwen_pseudo import prepare
    with local_raw_links(root,staged):
        prepare(output=str(local),sources=sources,checkpoint=checkpoint,device='cuda',batch_size=batch_size,workers=workers,tau=tau,
            catalog=str(scratch/'catalog.csv'),registry_output=str(scratch/'protected_identities'),strict_sources=verify_sources,
            report_path=str(local/'qwen_pseudo_dataset_report.json'))
    return sync_completed_cache(local,output)
