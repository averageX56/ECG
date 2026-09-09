"""Foreground probability bands for frozen delineators and downstream diagnoses."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ecg_project.data.cache import begin_cache, shard
from ecg_project.data.catalog import file_hash
from ecg_project.data.classifier_labels import select_common_classes
from ecg_project.data.policy import build_record_manifest, source_name
from ecg_project.processing.features import BEAT_FEATURE_NAMES
from ecg_project.processing.trust import feature_bands
from ecg_project.training import interval_classification as base
from ecg_project.utils import save_json

METHOD = 'foreground_probability_bands_v1'
FEATURE_NAMES = [*BEAT_FEATURE_NAMES, 'qt_ms', 't_ms']


@dataclass
class TrustConfig:
    confidence_threshold: float = .5
    uncertainty_threshold: float = .3
    probability_smoothing_ms: float = 20.
    max_extension_ms: float = 40.

    def validate(self):
        if not 0 <= self.uncertainty_threshold < self.confidence_threshold <= 1:
            raise ValueError('Require 0 <= uncertainty_threshold < confidence_threshold <= 1')
        if not all(np.isfinite(v) and v >= 0 for v in (self.probability_smoothing_ms, self.max_extension_ms)):
            raise ValueError('Nonnegative finite smoothing and extension required')


def record_features(signal, fs, peaks, predictor, cfg):
    cfg.validate()
    waves = predictor.predict(signal, fs, confidence_threshold=cfg.confidence_threshold,
                              uncertainty_threshold=cfg.uncertainty_threshold,
                              probability_smoothing_ms=cfg.probability_smoothing_ms,
                              max_extension_ms=cfg.max_extension_ms)[0]
    wave, features, ids = base.interval_features(signal, peaks, waves, fs)
    point, trust = feature_bands(features, np.asarray(peaks)[ids], waves, fs)
    return wave, point, trust, dict(ids=ids, waves=waves)


def _load_signal(row):
    if row.source == 'MIT' and Path(row.path).suffix == '.csv':
        from ecg_project.training.beats import load_mit
        record = load_mit(row.path)
    else:
        record = base.load_record(row.path)
    lead = next((l for l in ('II', 'MLII') if l in record.leads), record.leads[0])
    return record.signal[:, record.leads.index(lead)], record.fs


def prepare(cfg, trust_cfg, original_root, frame, qwen, output):
    identity = dict(**json.loads((original_root/'cache_identity.json').read_text()),
                    method=METHOD, config=asdict(trust_cfg),
                    manifest=file_hash(original_root/'manifest.csv'))
    root = begin_cache(output/'inputs', identity)
    for variant in ('unet', 'qwen'):
        branch = begin_cache(root/variant, identity)
        predictor = None
        try:
            for row in tqdm(frame.itertuples(), total=len(frame), desc=f'{variant} trust intervals'):
                original = original_root/variant/f'{row.key}.npz'
                def compute():
                    nonlocal predictor
                    if predictor is None:
                        if variant == 'unet':
                            from ecg_project.models.segmentation import Predictor
                            predictor = Predictor(cfg.unet_checkpoint, cfg.device)
                        else:
                            from ecg_project.models.qwen_delineator import Predictor
                            predictor = Predictor(qwen, cfg.model_root, cfg.device)
                    signal, fs = _load_signal(row)
                    # Use the complete R-peak list, including edge beats, to keep
                    # RR context identical to the deterministic preparation.
                    peaks, _ = base.rpeaks(base.preprocess(signal, fs), fs)
                    wave, point, trust, meta = record_features(signal, fs, peaks, predictor, trust_cfg)
                    with np.load(original) as old:
                        if not np.array_equal(old['peaks'], peaks[meta['ids']]) or not np.array_equal(old['wave'], wave):
                            raise ValueError(f'Trust/base crop mismatch: {row.key}')
                    return dict(wave=wave, interval=trust, point=point, waves=json.dumps(meta['waves']), fs=fs)
                shard(branch, row.key, dict(original_sha256=file_hash(original), experiment=identity), compute)
        finally:
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    frame.to_csv(root/'manifest.csv', index=False)
    # Materialize the lightweight point control with identical cohorts/crops.
    point_root = begin_cache(output/'point_inputs', identity)
    for variant in ('unet', 'qwen'):
        branch = begin_cache(point_root/variant, identity)
        for row in tqdm(frame.itertuples(), total=len(frame), desc=f'{variant} point control'):
            original = root/variant/f'{row.key}.npz'
            def compute_point():
                with np.load(original) as z:
                    return dict(wave=z['wave'], interval=z['point'])
            shard(branch, row.key, dict(trust_sha256=file_hash(original)), compute_point)
    frame.to_csv(point_root/'manifest.csv', index=False)
    return root, point_root


def run(cfg=None, trust_cfg=None, output=None):
    cfg = cfg or base.ClassificationConfig()
    if min(cfg.epochs, cfg.patience, cfg.batch_size, cfg.max_beats) < 1 or cfg.learning_rate <= 0:
        raise ValueError('Invalid classifier training parameters')
    trust_cfg = trust_cfg or TrustConfig()
    trust_cfg.validate()
    output = Path(output or (cfg.output + '_trust'))
    if output.resolve() == Path(cfg.output).resolve():
        raise ValueError('Trust experiment requires a separate output')
    qwen = base.selected_qwen(cfg.qwen_run)
    original_root, frame = base.prepare(cfg, qwen)
    frame = frame[frame.beats > 0].reset_index(drop=True)
    required = set(build_record_manifest(cfg.catalog).source.map(source_name))
    classes, label_report = select_common_classes(frame, required)
    if not classes:
        raise ValueError('No common supported diagnosis classes')
    begin_cache(output, dict(method=METHOD, config=asdict(trust_cfg), classifier=asdict(cfg),
                             original_inputs=json.loads((original_root/'cache_identity.json').read_text())))
    save_json(output/'diagnosis_coverage.json', label_report)
    root, point_root = prepare(cfg, trust_cfg, original_root, frame, qwen, output)
    results, coverage = {}, []
    for variant in ('unet', 'qwen'):
        for mode, inputs in (('point', point_root), ('trust', root)):
            spec = dict(method=METHOD, config=asdict(trust_cfg), mode=mode,
                        feature_names=FEATURE_NAMES if mode == 'point' else
                        [f'{stat}/{name}' for stat in ('point', 'lower', 'upper', 'width', 'confidence') for name in FEATURE_NAMES])
            results[f'{variant}/{mode}'] = base.train_branch(
                replace(cfg, output=str(output/mode), evaluate_manual=False), inputs, frame, variant, classes, spec)
        for source, cohort in frame.groupby('source'):
            width_sum = np.zeros(len(FEATURE_NAMES))
            accepted = np.zeros(len(FEATURE_NAMES))
            beats = 0
            for row in cohort.itertuples():
                with np.load(root/variant/f'{row.key}.npz') as z:
                    width_sum += np.nansum(z['interval'][:, 3*len(FEATURE_NAMES):4*len(FEATURE_NAMES)], axis=0)
                    accepted += np.isfinite(z['point']).sum(0)
                    beats += len(z['point'])
            coverage.extend(dict(variant=variant, source=source, feature=name, beats=beats,
                                 mean_band_width=float(width_sum[i]/accepted[i]) if accepted[i] else None,
                                 accepted_fraction=float(accepted[i]/beats))
                            for i, name in enumerate(FEATURE_NAMES))
    selected = max(results, key=lambda name: results[name]['valid']['macro_auroc'])
    report = dict(method=METHOD, config=asdict(trust_cfg), classes=classes, selected=selected,
                  metrics=results, interpretation='Confident wave core >= high; attached uncertain foreground >= low. '
                  'Class 0 and competing wave classes stop envelopes. Feature bounds come from boundary envelopes. '
                  'Probability bands, not statistical coverage guarantees; fixed R peaks and morphology.',
                  selection='Validation macro AUROC only; disease thresholds from validation only.',
                  delineator_overlap='Delineator training overlap is not independently audited; stability does not establish accuracy.')
    save_json(output/'comparison.json', report)
    pd.DataFrame([dict(variant=v, cohort=c, records=m['records'], macro_auroc=m['macro_auroc'], macro_f1=m['macro_f1'])
                  for v, cohorts in results.items() for c, m in cohorts.items()]).to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(coverage).to_csv(output/'feature_coverage.csv', index=False)
    return report
