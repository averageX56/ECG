"""Train fresh Qwen 1.7B QLoRA (manual warmup, full cached E), then classifiers."""
import argparse
from dataclasses import dataclass, replace, asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ecg_project.data.catalog import file_hash, assert_disjoint
from ecg_project.data.classifier_labels import LABEL_VERSION, diagnosis_labels, select_common_classes
from ecg_project.training.interval_classification import ClassificationConfig, train_branch
from ecg_project.training.trust_intervals import METHOD, FEATURE_NAMES
from ecg_project.utils import save_json


@dataclass
class Job:
    stage: str
    variant: str
    root: Path
    output: Path
    feature_spec: dict | None


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def available(root):
    return (root/'cache_identity.json').is_file() and (root/'manifest.csv').is_file()


def make_jobs(base, trust, masks, stages, variants=('unet', 'qwen'), include_waveform=True):
    jobs, missing = [], []
    for stage in stages:
        if stage in ('waveform', 'intervals'):
            roots = [(base/'inputs', base)]
        elif stage == 'point':
            roots = [(trust/'point_inputs', trust/'point'), (masks/'point_inputs', masks/'point')]
        elif stage == 'trust':
            roots = [(trust/'inputs', trust/'trust'), (masks/'inputs', masks/'trust')]
        else:
            roots = [(masks/'inputs', masks/'masks')]
        selected_variants = (('waveform',) if include_waveform else ()) if stage == 'waveform' else variants
        for variant in selected_variants:
            branch = 'unet' if variant == 'waveform' else variant
            candidates = [(root, out) for root, out in roots if available(root) and (root/branch).is_dir()]
            # Prefer a completed or interrupted run to avoid retraining a duplicate.
            candidates.sort(key=lambda pair: not (pair[1]/LABEL_VERSION/variant/'cache_identity.json').is_file())
            if not candidates:
                missing.append(dict(stage=stage, variant=variant, reason='Missing prepared cache',
                                    paths=[str(root) for root, _ in roots]))
                continue
            root, out = candidates[0]
            spec = None
            if stage in ('point', 'trust', 'masks'):
                identity = read_json(root/'cache_identity.json')
                if identity.get('method') != METHOD:
                    raise ValueError(f'Unsupported feature cache method: {root}')
                if stage == 'masks' and identity.get('confidence_masks') != 'softmax_250hz_background_P_QRS_T_v1':
                    missing.append(dict(stage=stage, variant=variant, reason='Cache has no soft confidence masks', paths=[str(root)]))
                    continue
                spec = dict(method=METHOD, config=identity['config'], mode=stage,
                    feature_names=FEATURE_NAMES if stage == 'point' else
                    [f'{stat}/{name}' for stat in ('point', 'lower', 'upper', 'width', 'confidence') for name in FEATURE_NAMES])
                if stage == 'masks':
                    spec.update(confidence_masks=True, mask_channels=['background', 'P', 'QRS', 'T'], mask_fs=250)
            jobs.append(Job(stage, variant, root, out, spec))
    return jobs, missing


def load_frame(root):
    frame = pd.read_csv(root/'manifest.csv', dtype={'key': str}).fillna('')
    required = {'key', 'source', 'split', 'labels', 'beats', 'signal_hash'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Incomplete classifier manifest: {root}')
    if frame.key.duplicated().any():
        raise ValueError(f'Duplicate cache keys: {root}')
    frame = diagnosis_labels(frame)
    assert_disjoint(frame)
    # Capture sources before filtering unusable beats.
    from ecg_project.data.policy import source_name
    sources = set(frame.loc[frame.split != 'inference', 'source'].map(source_name))
    frame = frame[frame.beats > 0].reset_index(drop=True)
    classes, coverage = select_common_classes(frame, sources)
    if not classes:
        raise ValueError(f'No supported common diagnosis classes in {root}')
    return frame, classes, coverage


def job_config(job, defaults):
    path = job.output/LABEL_VERSION/job.variant/'cache_identity.json'
    if path.exists():
        cfg = ClassificationConfig(**read_json(path)['config'])
        if Path(cfg.output).resolve() != job.output.resolve():
            raise ValueError(f'Existing run config points to {cfg.output}, expected {job.output}; keep run paths consistent')
        return cfg
    # Match notebook configs, including the evaluate_manual flag stored in identity.
    return replace(defaults, output=str(job.output),
                   evaluate_manual=defaults.evaluate_manual if job.feature_spec is None else False)


def validate_shards(job, frame):
    branch = job.root/('unet' if job.variant == 'waveform' else job.variant)
    root_identity = read_json(job.root/'cache_identity.json')
    if read_json(branch/'cache_identity.json') != root_identity:
        raise ValueError(f'Branch identity mismatch: {branch}')
    for key in tqdm(frame.key, desc=f'Verify {branch}', unit='shard'):
        # A manifest must not be able to reference files outside its branch.
        if not isinstance(key, str) or not key or Path(key).name != key or '/' in key or '\\' in key:
            raise ValueError(f'Invalid shard key: {key}')
        data, meta = branch/(key+'.npz'), branch/(key+'.json')
        if not data.is_file() or not meta.is_file():
            raise FileNotFoundError(f'Missing completed shard: {data}; no preparation is performed by this script')
        record = read_json(meta)
        if record['sha256'] != file_hash(data):
            raise ValueError(f'Modified cache shard: {data}')
        identity = record['identity']
        if 'models' in identity and identity['models'] != root_identity:
            raise ValueError(f'Shard model identity mismatch: {data}')
        if 'experiment' in identity and identity['experiment'] != root_identity:
            raise ValueError(f'Shard experiment identity mismatch: {data}')
        with np.load(data, allow_pickle=False) as arrays:
            required = {'wave', 'interval'} | ({'confidence_mask'} if job.stage == 'masks' else set())
            if not required.issubset(arrays.files):
                raise ValueError(f'Cache lacks required arrays {required}: {data}')


def qwen_configs(args):
    from pipelines.gpu.experiments import qwen_experiment
    common = dict(input_root=args.manual_root, model_root=args.qwen_model_root,
                  batch_size=args.qwen_batch_size, accumulation=args.qwen_accumulation,
                  quantization='nf4', consistency_weight=0., final_checkpoint='', max_batches=0, valid_limit=0)
    warm = replace(qwen_experiment('B', '1.7B'), **common, output=args.qwen_output+'_manual_warmup',
                   epochs=args.warmup_epochs, patience=args.warmup_epochs, warm_start=None)
    full = replace(qwen_experiment('E', '1.7B'), **common, output=args.qwen_output,
                   pseudo_root=args.pseudo_root, pseudo_full_pass=True, pseudo_per_manual=4,
                   epochs=args.qwen_epochs, patience=args.qwen_patience, learning_rate=5e-5,
                   warm_start=str(Path(warm.output)/'best.pt'))
    return warm, full


def run_qwen_stage(cfg):
    from pipelines.gpu.experiments import run_qwen
    provenance = read_json(Path(cfg.model_root)/'provenance.json')
    if cfg.quantization != 'nf4' or provenance.get('model_id') != 'Qwen/Qwen3-1.7B':
        raise ValueError('This runner requires a Qwen3-1.7B NF4/QLoRA checkpoint and base model')
    output = Path(cfg.output)
    print(f'Qwen 1.7B QLoRA: warm_start={cfg.warm_start}; output={output}; pseudo_cache={cfg.pseudo_root}', flush=True)
    complete = output/'run.json'
    if complete.is_file() and (output/'best.pt').is_file() and (output/'latest.pt').is_file():
        previous = read_json(output/'config.json')
        matching = all(previous.get(k) == v for k,v in asdict(cfg).items())
        if matching and read_json(complete)['status'] == 'complete':
            import torch
            saved = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
            identity = saved['identity']
            from ecg_project.data.cache import validate_cache
            from ecg_project.data.delineation_v2 import VERSION as manual_version
            from ecg_project.data.qwen_pseudo import VERSION as pseudo_version
            # Reuse completed training only with the same inputs and incumbent.
            if (identity.get('warm_start_sha256') != (file_hash(cfg.warm_start) if cfg.warm_start else None) or identity['base'] != provenance
                or identity['cache'] != validate_cache(cfg.input_root, manual_version)
                or (cfg.pseudo_root and identity['pseudo'] != validate_cache(cfg.pseudo_root, pseudo_version))):
                raise ValueError('Completed Qwen run inputs changed; use a new run directory')
            print('Qwen stage already complete; reusing best.pt', flush=True)
            return cfg
    run_qwen(cfg, run_smoke=False)
    return cfg


def train_fresh_qwen(args):
    from pipelines.gpu.experiments import assert_qwen_inputs
    warm, full = qwen_configs(args)
    # Fail on missing manual/pseudo caches before spending time on warmup.
    assert_qwen_inputs(warm)
    assert_qwen_inputs(full)
    if not (Path(warm.model_root)/'provenance.json').is_file():
        if Path(warm.model_root).resolve() != Path('artifacts/qwen3_1.7b').resolve():
            raise FileNotFoundError(f'Missing Qwen3-1.7B base at {warm.model_root}')
        import subprocess
        print('Downloading the official Qwen3-1.7B base; existing ECG caches are reused.', flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('download_qwen.py')), '--size', '1.7B'], check=True)
    run_qwen_stage(warm)
    return run_qwen_stage(full)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qwen_E_classifier.json')
    parser.add_argument('--base-output', help='Original classifier output, containing inputs/')
    parser.add_argument('--trust-output', help='Existing probability-band experiment directory')
    parser.add_argument('--masks-output', help='Existing soft-mask experiment directory')
    parser.add_argument('--cache-only', action='store_true', help='Only run classifiers from existing caches; no Qwen fine-tuning or new features')
    parser.add_argument('--qwen-output', default='artifacts/cluster/qwen_1.7b_E_fresh_full_cache')
    parser.add_argument('--qwen-model-root', default='artifacts/qwen3_1.7b')
    parser.add_argument('--pseudo-root', default='artifacts/qwen_pseudo_extended', help='Existing pseudo cache')
    parser.add_argument('--manual-root', default='artifacts/qwen_delineation_inputs_v2', help='Existing manual cache')
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--qwen-batch-size', type=int, default=16)
    parser.add_argument('--qwen-accumulation', type=int, default=4)
    parser.add_argument('--qwen-epochs', type=int, default=20)
    parser.add_argument('--qwen-patience', type=int, default=5)
    parser.add_argument('--stages', nargs='+', choices=['waveform', 'intervals', 'point', 'trust', 'masks'],
                        default=['waveform', 'intervals', 'point', 'trust', 'masks'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=.001, help='Minimum cumulative validation AUROC gain to reset patience')
    parser.add_argument('--dry-run', action='store_true', help='List jobs without hashing shards or training')
    parser.add_argument('--report', default='reports/classifier_console.json')
    args = parser.parse_args(argv)
    if min(args.patience, args.qwen_epochs, args.qwen_patience, args.warmup_epochs,
           args.qwen_batch_size, args.qwen_accumulation) < 1 or not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error('patience must be positive; min-delta must be finite and nonnegative')
    defaults = ClassificationConfig(**read_json(args.config))
    base = Path(args.base_output or defaults.output)
    trust, masks = Path(args.trust_output or (str(base)+'_trust')), Path(args.masks_output or (str(base)+'_trust_masks'))
    stages = list(dict.fromkeys(args.stages))
    jobs, missing = make_jobs(base, trust, masks, stages,
                              variants=('unet', 'qwen') if args.cache_only else ('unet',))
    report = dict(stopping_policy=dict(patience=args.patience, min_delta=args.min_delta), jobs=[], missing=missing)
    qwen_defaults = None
    qwen_jobs = []
    needs_qwen = not args.cache_only and any(stage != 'waveform' for stage in stages)
    if needs_qwen:
        if args.dry_run:
            print(f'PLAN: fresh Qwen3-1.7B NF4; {args.warmup_epochs} manual warmup epochs; full existing pseudo cache; '
                  f'output={args.qwen_output}. Then refresh Qwen features and run {stages}. No CPU/pseudo preparation.', flush=True)
        else:
            if not available(base/'inputs'):
                raise FileNotFoundError(f'Missing original classifier cache: {base}/inputs')
            full = train_fresh_qwen(args)
            qwen_defaults = replace(defaults, qwen_run=full.output, model_root=full.model_root,
                                    output=str(base)+'_qwen17_fresh_E')
            from ecg_project.training.classifier_refresh import refresh_qwen
            from ecg_project.training.trust_intervals import TrustConfig
            # Keep the exact probability-band settings of the U-Net comparison.
            config = next((read_json(p/'cache_identity.json')['config'] for p in (masks/'inputs', trust/'inputs') if available(p)), None)
            trust_cfg = TrustConfig(**config) if config else TrustConfig()
            missing_unet = sorted({m['stage'] for m in missing if m['variant'] == 'unet' and m['stage'] in ('point', 'trust', 'masks')})
            if missing_unet:
                print('Preparing only missing U-Net feature stages:', missing_unet, flush=True)
                unet_output = Path(str(base)+'_unet_missing_features')
                _, unet_bands = refresh_qwen(base/'inputs', unet_output, defaults, trust_cfg,
                    need_bands=True, need_masks='masks' in missing_unet, variant='unet')
                added, still_missing = make_jobs(unet_output, unet_bands, unet_bands, missing_unet,
                    variants=('unet',), include_waveform=False)
                jobs += added
                missing[:] = [m for m in missing if not (m['variant'] == 'unet' and m['stage'] in missing_unet)]
                missing += still_missing
            _, new_bands = refresh_qwen(base/'inputs', qwen_defaults.output, qwen_defaults, trust_cfg,
                need_bands=any(s in stages for s in ('point', 'trust', 'masks')), need_masks='masks' in stages)
            qwen_jobs, qwen_missing = make_jobs(Path(qwen_defaults.output), new_bands, new_bands,
                stages, variants=('qwen',), include_waveform=False)
            jobs += qwen_jobs
            missing += qwen_missing
            report['qwen'] = dict(config=asdict(full), checkpoint=str(Path(full.output)/'best.pt'))
    for entry in missing:
        print('MISSING:', entry, flush=True)
    verified = set()
    for job in jobs:
        out = job.output/LABEL_VERSION/job.variant
        status = 'completed' if (out/'metrics.json').is_file() else 'resume' if (out/'latest.pt').is_file() else 'new'
        print(f'{job.stage}/{job.variant}: {status}; cache={job.root}; output={out}', flush=True)
        if args.dry_run:
            continue
        try:
            frame, classes, coverage = load_frame(job.root)
            cfg = job_config(job, qwen_defaults if job in qwen_jobs else defaults)
            key = (str(job.root.resolve()), 'unet' if job.variant == 'waveform' else job.variant, job.stage == 'masks')
            if status != 'completed' and key not in verified:
                validate_shards(job, frame)
                verified.add(key)
            result = train_branch(cfg, job.root, frame, job.variant, classes, job.feature_spec,
                                  stopping_patience=args.patience, min_delta=args.min_delta)
            report['jobs'].append(dict(stage=job.stage, variant=job.variant, status='completed',
                action=status, output=str(out), classes=classes, diagnosis_coverage=coverage, metrics=result))
        except (FileNotFoundError, ValueError) as exc:
            report['jobs'].append(dict(stage=job.stage, variant=job.variant, status='blocked', error=str(exc)))
            print(f'BLOCKED {job.stage}/{job.variant}: {exc}', flush=True)
        save_json(args.report, report)
    if not args.dry_run:
        save_json(args.report, report)
    return 2 if missing or any(j['status'] == 'blocked' for j in report['jobs']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
