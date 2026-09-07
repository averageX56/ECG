"""Cluster entry points consume caches prepared in pipelines/cpu."""
from pathlib import Path


def require_a100_memory():
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError('This training profile requires a GPU with >=35 GiB VRAM')
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def train_hubert(input_root='artifacts/hubert_inputs_v2', output_root='artifacts/cluster', epochs=30, include_qlora=True):
    vram = require_a100_memory()
    root = Path(input_root)
    for name in ('manifest.csv', 'signals.npy', 'provenance.json'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing {root / name}; run pipelines.cpu.run prepare-hubert locally first')
    from ecg_project.training.hubert_lora import LoRAConfig, run
    outputs = []
    variants=[('hubert_r0',0,'none'),('hubert_r16',16,'none')]
    if include_qlora:variants.append(('hubert_qlora_r16',16,'nf4'))
    for name,rank,quantization in variants:
        outputs.append(run(LoRAConfig(input_root=str(root), output=str(Path(output_root) / name),
            rank=rank, quantization=quantization, epochs=epochs, batch_size=64 if vram > 60 else 32, accumulation=2)))
    return outputs


def train_bert(config):
    require_a100_memory()
    from ecg_project.training.cluster_training import train_cluster
    return train_cluster(config)


def train_qwen(config=None):
    vram=require_a100_memory()
    from ecg_project.training.qwen_delineation import QwenConfig,train as baseline
    if isinstance(config,QwenConfig):return baseline(config)
    from pipelines.gpu.experiments import qwen_experiment,run_qwen
    return run_qwen(config or qwen_experiment(vram_gb=vram))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='A100 HuBERT frozen/LoRA comparison')
    parser.add_argument('--input-root', default='artifacts/hubert_inputs_v2')
    parser.add_argument('--output-root', default='artifacts/cluster')
    parser.add_argument('--epochs', type=int, default=30)
    train_hubert(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
