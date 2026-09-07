"""JupyterLab cluster entry points. Existing datasets and directories stay in place."""
from pathlib import Path
import hashlib

EXTENDED_QWEN_SOURCES = ('CPSC_EXTRA', 'PTBXL', 'CPSC', 'CHAPMAN')


def prepare_qwen_pseudo_gpu(output='artifacts/qwen_pseudo_extended',
        sources=EXTENDED_QWEN_SOURCES, checkpoint='artifacts/cluster/delineator_qt.pt',
        batch_size=64, workers=8, tau=.95, scratch=None, verify_sources=False):
    """Read existing raw paths; optionally publish new cache from node-local scratch.

    Run from the existing repository root. No mount, link changes or raw copies.
    A partial persistent cache is resumed in place, never replaced by scratch.
    """
    from ecg_project.data.cache import require_checkpoint, validate_cache
    from ecg_project.data.catalog import file_hash
    from ecg_project.data.qwen_pseudo import prepare, VERSION
    from pipelines.gpu.qwen_staging import sync_completed_cache
    require_checkpoint(checkpoint)
    destination = Path(output).absolute()
    if not set(sources) <= set(EXTENDED_QWEN_SOURCES):
        raise ValueError('Qwen extended sources must exclude Ningbo and unknown sources')
    if not Path('artifacts/catalog.csv').is_file():
        raise FileNotFoundError('Missing artifacts/catalog.csv; run python -m pipelines.cpu.run audit --workers 8')
    if (destination/'provenance.json').is_file():
        meta = validate_cache(destination)
        if meta['teacher_sha256'] != file_hash(checkpoint):
            raise ValueError('Teacher hash mismatch: choose a new pseudo output; existing artifacts are preserved')
        if set(meta['source_datasets']) != set(sources) or meta['confidence_threshold'] != tau:
            raise ValueError('Pseudo source/threshold mismatch: choose a new output')
        if meta['preprocessing'] != VERSION:
            raise ValueError('Pseudo preprocessing version mismatch: choose a new output')
        if not verify_sources:
            return destination
    local = destination
    if scratch is not None and not destination.exists():
        token = hashlib.sha256(str(destination).encode()).hexdigest()[:12]
        local = Path(scratch).absolute() / (destination.name+'-'+token)
        if local == destination or local.is_relative_to(destination):
            raise ValueError('Scratch must be separate from the published cache')
    registry = (local.parent/'protected_identities_v2') if scratch is not None else Path('artifacts/protected_identities_v2')
    prepare(output=str(local), sources=sources, checkpoint=checkpoint, device='cuda',
            batch_size=batch_size, workers=workers, tau=tau, registry_output=str(registry),
            strict_sources=verify_sources, report_path=str(local/'qwen_pseudo_dataset_report.json'))
    return sync_completed_cache(local, destination) if local != destination else destination


def check_environment(qlora=True):
    """Small runtime compatibility probe; no downloads or dataset/output mutations."""
    import sys
    import importlib.metadata as metadata
    import torch
    from pipelines.gpu.run import require_a100_memory
    vram = require_a100_memory()
    packages = ['torch', 'numpy', 'scipy', 'pandas', 'wfdb', 'neurokit2', 'jupyterlab', 'ipykernel']
    if qlora:
        from transformers import Qwen3Model  # Validate the actual model API.
        import bitsandbytes as bnb
        packages += ['transformers', 'bitsandbytes', 'safetensors']
        layer = bnb.nn.Linear4bit(64, 32, compute_dtype=torch.bfloat16,
                                 quant_type='nf4', compress_statistics=True).cuda()
        x = torch.randn(2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        loss = layer(x).float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
            raise RuntimeError('NF4 forward/backward returned non-finite values')
        del layer, x, loss
        torch.cuda.empty_cache()
    return dict(python=sys.executable, versions={p: metadata.version(p) for p in packages},
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), vram_gib=vram)
