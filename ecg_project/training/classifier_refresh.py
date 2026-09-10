"""Refresh Qwen classifier features after fine-tuning, preserving existing U-Net caches."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ecg_project.data.cache import begin_cache, shard
from ecg_project.data.catalog import file_hash
from ecg_project.processing.signal import resample, beat_windows
from ecg_project.processing.trust import decode_probability_bands, feature_bands, smooth_probabilities
from ecg_project.training import interval_classification as base
from ecg_project.training.trust_intervals import METHOD, _load_signal
from dataclasses import asdict


def refresh_qwen(original_root, output, cfg, trust_cfg, need_bands=True, need_masks=True, variant='qwen'):
    """One new Qwen pass per record; no teacher/pseudo/CPU cache preparation.

    Existing crop/record selection is authoritative. Source and crop changes
    fail instead of silently mixing representations from different signals.
    """
    original_root, output = Path(original_root), Path(output)
    trust_cfg.validate()
    if variant not in ('unet', 'qwen'):
        raise ValueError('Expected unet or qwen feature refresh')
    qwen = base.selected_qwen(cfg.qwen_run) if variant == 'qwen' else None
    original_identity = json.loads((original_root/'cache_identity.json').read_text())
    identity = dict(original_identity)
    if qwen is not None:
        identity['qwen'] = file_hash(qwen/'best.pt')
    if file_hash(cfg.unet_checkpoint) != identity['unet']:
        raise ValueError('Original U-Net checkpoint differs from the cached comparison')
    root = begin_cache(output/'inputs', identity)
    branch = begin_cache(root/variant, identity)
    frame = pd.read_csv(original_root/'manifest.csv', dtype={'key': str}).fillna('')
    frame = frame[frame.beats > 0].reset_index(drop=True)
    band_output = Path(str(output)+'_trust_masks')
    band_root = point_root = None
    if need_bands:
        band_identity = dict(identity, method=METHOD, config=asdict(trust_cfg),
                             manifest=file_hash(original_root/'manifest.csv'))
        if need_masks:
            band_identity['confidence_masks'] = 'softmax_250hz_background_P_QRS_T_v1'
        band_root = begin_cache(band_output/'inputs', band_identity)
        begin_cache(band_root/variant, band_identity)
        point_root = begin_cache(band_output/'point_inputs', band_identity)
        begin_cache(point_root/variant, band_identity)
    predictor = None
    try:
        for row in tqdm(frame.itertuples(), total=len(frame), desc=f'Refresh {variant} features'):
            if not row.key or '/' in row.key or '\\' in row.key or Path(row.key).name != row.key:
                raise ValueError('Invalid original shard key')
            original = original_root/variant/f'{row.key}.npz'
            metadata = json.loads(original.with_suffix('.json').read_text())
            if metadata['identity']['models'] != original_identity or file_hash(original) != metadata['sha256']:
                raise ValueError(f'Original cache changed: {original}')
            arrays = None
            def compute_all():
                nonlocal predictor, arrays
                if arrays is not None:
                    return arrays
                # No rehash of raw data on resume: only needed for new features.
                for path, digest in metadata['identity']['source_sha256'].items():
                    if file_hash(path) != digest:
                        raise ValueError(f'Source changed since original classification: {path}')
                if predictor is None:
                    if variant == 'qwen':
                        from ecg_project.models.qwen_delineator import Predictor
                        predictor = Predictor(qwen, cfg.model_root, cfg.device)
                    else:
                        from ecg_project.models.segmentation import Predictor
                        predictor = Predictor(cfg.unet_checkpoint, cfg.device)
                signal, fs = _load_signal(row)
                peaks, _ = base.rpeaks(base.preprocess(signal, fs), fs)
                waves, probabilities = predictor.predict(signal, fs, return_probabilities=True)
                wave, interval, ids = base.interval_features(signal, peaks, waves[0], fs)
                with np.load(original) as cached:
                    if not np.array_equal(wave, cached['wave']) or not np.array_equal(peaks[ids], cached['peaks']):
                        raise ValueError(f'Original crop alignment changed: {row.key}')
                    arrays = dict(wave=wave, interval=interval, peaks=peaks[ids], waves=json.dumps(waves[0]),
                                  signal_hash=cached['signal_hash'], lead=cached['lead'])
                if need_bands:
                    processed = resample(base.preprocess(signal, fs), fs)
                    bands = [] if np.ptp(signal) < 1e-8 else decode_probability_bands(probabilities[:, 0], processed, fs, len(signal),
                        trust_cfg.uncertainty_threshold, trust_cfg.confidence_threshold,
                        trust_cfg.probability_smoothing_ms, trust_cfg.max_extension_ms)
                    _, features, band_ids = base.interval_features(signal, peaks, bands, fs)
                    if not np.array_equal(ids, band_ids):
                        raise ValueError('Band crop alignment changed')
                    point, trust = feature_bands(features, peaks[ids], bands, fs)
                    band = dict(wave=wave, interval=trust, point=point, waves=json.dumps(bands), fs=fs)
                    if need_masks:
                        probabilities = smooth_probabilities(probabilities[:, 0], trust_cfg.probability_smoothing_ms)
                        masks, mask_ids = beat_windows(probabilities, np.round(peaks*250/fs).astype(int), 250)
                        if not np.array_equal(ids, mask_ids):
                            raise ValueError('Mask crop alignment changed')
                        band['confidence_mask'] = masks.transpose(0, 2, 1)
                    arrays['_band'] = band
                return arrays
            shard(branch, row.key, dict(source_sha256=metadata['identity']['source_sha256'], models=identity),
                  lambda: {k:v for k,v in compute_all().items() if k != '_band'})
            if need_bands:
                band_path = shard(band_root/variant, row.key,
                    dict(original_sha256=file_hash(branch/f'{row.key}.npz'), experiment=band_identity),
                    lambda: compute_all()['_band'])
                def compute_point():
                    with np.load(band_path) as cached:
                        return dict(wave=cached['wave'], interval=cached['point'])
                shard(point_root/variant, row.key, dict(trust_sha256=file_hash(band_path)), compute_point)
    finally:
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Preserve the exact manifest bytes; no labels/splits/record selection change.
    for target in (root, band_root, point_root):
        if target is not None:
            (target/'manifest.csv').write_bytes((original_root/'manifest.csv').read_bytes())
    return root, band_output
