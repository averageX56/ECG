"""Frozen Qwen E/U-Net comparison on identical crops and record diagnosis labels."""
from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from ecg_project.data.cache import begin_cache, shard, source_hashes, atomic_json
from ecg_project.data.catalog import file_hash, assert_disjoint
from ecg_project.data.classifier_labels import TARGETS, LABEL_VERSION, diagnosis_labels, select_common_classes
from ecg_project.data.policy import build_record_manifest
from ecg_project.data.io import load_record
from ecg_project.processing.signal import preprocess, rpeaks
from ecg_project.processing.features import beat_features
from ecg_project.evaluation.metrics import multilabel_metrics, thresholds_on_validation
from ecg_project.models.interval_classifier import IntervalClassifier
from ecg_project.training.cluster_training import _atomic_save
from ecg_project.utils import save_json, seed_all


@dataclass
class ClassificationConfig:
    qwen_run: str = 'artifacts/cluster/qwen_4b_E_curriculum'
    unet_checkpoint: str = 'artifacts/cluster/delineator_qt.pt'
    catalog: str = 'artifacts/catalog.csv'
    data_root: str = 'data'
    output: str = 'artifacts/cluster/qwen_E_classifier'
    model_root: str | None = None
    device: str = 'cuda'
    epochs: int = 60
    patience: int = 10
    batch_size: int = 8
    max_beats: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    manual_root: str = 'artifacts/qwen_delineation_inputs_v2'
    ludb_root: str = 'LUDB'
    qt_root: str = 'data/qtdb_external'
    evaluate_manual: bool = True
    qwen_seen_all_datasets: bool = True


def selected_qwen(path):
    """Respect accepted curriculum selection, including an earlier accepted cycle."""
    root = Path(path)
    if root.is_file():
        if root.name != 'best.pt':
            raise ValueError('Qwen loader requires best.pt or its run directory')
        root = root.parent
    journal = root / 'selection.json'
    if journal.exists():
        state = json.loads(journal.read_text())
        checkpoint = state.get('best_checkpoint')
        if not checkpoint:
            raise ValueError('Curriculum has no accepted checkpoint')
        checkpoint = Path(checkpoint.replace('\\', '/'))
        if not checkpoint.is_file():
            # Permit moving the complete curriculum directory to another machine.
            local = root / checkpoint.parent.name / checkpoint.name
            if not local.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint = local
        accepted = [c for c in state.get('cycles', []) if c.get('accepted')]
        if accepted and file_hash(checkpoint) != accepted[-1]['best_checkpoint_sha256']:
            raise ValueError('Selected Qwen checkpoint hash differs from curriculum journal')
        root = checkpoint.parent
    for name in ('best.pt', 'config.json'):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root


def interval_features(signal, peaks, waves, fs):
    wave, features, ids = beat_features(signal, peaks, waves, fs)
    # Add T/QT information to the existing RR, PR and QRS features.
    extra = np.full((len(ids), 2), np.nan, np.float32)
    for j, i in enumerate(ids):
        qrs = [w for w in waves if w['wave'] == 'QRS' and abs(w['peak']-peaks[i]) <= .15*fs]
        if not qrs:
            continue
        q = min(qrs, key=lambda w: abs(w['peak']-peaks[i]))
        ts = [w for w in waves if w['wave'] == 'T' and 0 < w['onset']-q['offset'] < .6*fs]
        if ts:
            t = min(ts, key=lambda w: w['onset'])
            extra[j] = [(t['offset']-q['onset'])*1000/fs, (t['offset']-t['onset'])*1000/fs]
    return wave, np.concatenate([features, extra], 1), ids


def prepare(cfg, qwen):
    """All readable catalog records; unsupported diagnosis sources get inference only."""
    known = build_record_manifest(cfg.catalog)
    raw = pd.read_csv(cfg.catalog).fillna('')
    extra = raw[~raw.path.isin(known.path)].copy()
    extra['split'] = 'inference'
    for c in TARGETS:
        extra[c] = -1
    frame = pd.concat([known, extra], ignore_index=True).fillna('')
    # The metadata catalog scans WFDB headers; the local MIT distribution is CSV.
    # Its beat annotations are not record-level diagnosis ground truth.
    mit_rows = []
    existing_paths = {str(Path(p).resolve()) for p in frame.path}
    for p in sorted((Path(cfg.data_root)/'mit-bih').glob('*.csv')):
        if str(p.resolve()) in existing_paths:
            continue
        patient = '201_202' if p.stem in ('201', '202') else p.stem
        mit_rows.append(dict(path=str(p), source='MIT', record_id=p.stem, patient_id='MIT:'+patient,
                             readable=True, split='inference', **{c: -1 for c in TARGETS}))
    if mit_rows:
        frame = pd.concat([frame, pd.DataFrame(mit_rows)], ignore_index=True).fillna('')
    frame = frame[frame.readable == True].copy()
    frame['key'] = [__import__('hashlib').sha256(str(p).encode()).hexdigest()[:24] for p in frame.path]
    if frame.key.duplicated().any():
        raise ValueError('Duplicate catalog paths')
    frame = diagnosis_labels(frame)
    identity = dict(version=1, catalog=file_hash(cfg.catalog), qwen=file_hash(qwen/'best.pt'),
                    unet=file_hash(cfg.unet_checkpoint), lead='II/MLII/first', duration='full record',
                    additional_mit_paths=[r['path'] for r in mit_rows])
    root = begin_cache(Path(cfg.output)/'inputs', identity)
    # Each delineator is loaded once, and only if at least one shard is missing.
    for variant in ('unet', 'qwen'):
        branch = begin_cache(root/variant, identity)
        predictor = None
        try:
            for row in tqdm(frame.to_dict('records'), desc=f'{variant} full-record intervals'):
                is_mit_csv = row['source'] == 'MIT' and Path(row['path']).suffix == '.csv'
                hashes = source_hashes(row['path'], [Path('artifacts/mit_headers')/(Path(row['path']).stem+'.hea')]) if is_mit_csv else source_hashes(row['path'])
                def compute():
                    nonlocal predictor
                    if predictor is None:
                        if variant == 'qwen':
                            from ecg_project.models.qwen_delineator import Predictor
                            predictor = Predictor(qwen, cfg.model_root, cfg.device)
                        else:
                            from ecg_project.models.segmentation import Predictor
                            predictor = Predictor(cfg.unet_checkpoint, cfg.device)
                    if is_mit_csv:
                        from ecg_project.training.beats import load_mit
                        rec = load_mit(row['path'])
                    else:
                        rec = load_record(row['path'])
                    lead = next((l for l in ('II', 'MLII') if l in rec.leads), rec.leads[0])
                    signal = rec.signal[:, rec.leads.index(lead)]
                    peaks, _ = rpeaks(preprocess(signal, rec.fs), rec.fs)
                    waves = predictor.predict(signal, rec.fs)[0]
                    wave, intervals, ids = interval_features(signal, peaks, waves, rec.fs)
                    from ecg_project.data.protection import signal_hash
                    return dict(wave=wave.astype(np.float32), interval=intervals, peaks=peaks[ids],
                                signal_hash=signal_hash(rec), lead=lead, waves=json.dumps(waves))
                shard(branch, row['key'], dict(source_sha256=hashes, models=identity), compute)
        finally:
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    lengths, hashes, agreement = [], [], {}
    for row in frame.itertuples():
        with np.load(root/'unet'/f'{row.key}.npz') as a, np.load(root/'qwen'/f'{row.key}.npz') as b:
            if not np.array_equal(a['wave'], b['wave']) or not np.array_equal(a['peaks'], b['peaks']):
                raise ValueError(f'Unpaired crops: {row.key}')
            lengths.append(len(a['wave']))
            hashes.append(str(a['signal_hash']))
            left, right = a['interval'], b['interval']
            pair = np.isfinite(left) & np.isfinite(right)
            stats = agreement.setdefault(row.source, dict(beats=0, records=0,
                paired=np.zeros(left.shape[1], dtype=np.int64),
                unet_known=np.zeros(left.shape[1], dtype=np.int64),
                qwen_known=np.zeros(left.shape[1], dtype=np.int64),
                absolute_error=np.zeros(left.shape[1], dtype=np.float64)))
            stats['records'] += 1
            stats['beats'] += len(left)
            stats['paired'] += pair.sum(0)
            stats['unet_known'] += np.isfinite(left).sum(0)
            stats['qwen_known'] += np.isfinite(right).sum(0)
            stats['absolute_error'] += np.where(pair, abs(left-right), 0).sum(0)
    from ecg_project.processing.features import BEAT_FEATURE_NAMES
    for stats in agreement.values():
        stats['mean_absolute_difference'] = np.divide(stats.pop('absolute_error'), stats['paired'],
            out=np.full(len(stats['paired']), np.nan), where=stats['paired'] > 0)
    save_json(root/'interval_agreement.json', dict(feature_names=[*BEAT_FEATURE_NAMES, 'qt_ms', 't_ms'],
        by_source=agreement, interpretation='Qwen/U-Net agreement on matched detected beats, not accuracy against manual truth; before deduplication'))
    frame['beats'] = lengths
    frame['signal_hash'] = hashes
    # Retain the most protected copy; never train on a duplicate held-out signal.
    rank = dict(external_long=0, external=1, test=2, valid=3, inference=4, train=5)
    frame['_rank'] = frame.split.map(rank)
    duplicates = frame[frame.duplicated('signal_hash', keep=False)]
    duplicates.to_csv(root/'duplicates.csv', index=False)
    frame = frame.sort_values('_rank').drop_duplicates('signal_hash').drop(columns='_rank').reset_index(drop=True)
    assert_disjoint(frame)
    frame.to_csv(root/'manifest.csv', index=False)
    save_json(root/'coverage.json', dict(catalog_records=len(raw), additional_mit_csv_records=len(mit_rows), unreadable=int((raw.readable != True).sum()),
        evaluated_records=len(frame), zero_beat_records=int((frame.beats == 0).sum()),
        per_source=frame.groupby('source').agg(records=('key', 'size'), beats=('beats', 'sum')).to_dict('index'),
        label_policy='Only existing SNOMED adapters; unknown labels and unsupported sources are inference only'))
    return root, frame


def scale_features(features, median, scale):
    missing = ~np.isfinite(features)
    return np.concatenate([np.clip((np.where(missing, median, features)-median)/scale, -10, 10), missing], 1).astype(np.float32)


class Bags(Dataset):
    def __init__(self, frame, root, variant, classes, median, scale, max_beats):
        self.frame = frame.reset_index(drop=True)
        self.root, self.variant, self.classes = Path(root), variant, classes
        self.median, self.scale, self.max_beats = median, scale, max_beats

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        branch = 'unet' if self.variant == 'waveform' else self.variant
        with np.load(self.root/branch/f'{row.key}.npz') as z:
            n = len(z['wave'])
            # Even coverage of the complete record, identical in every branch.
            ids = np.linspace(0, n-1, min(n, self.max_beats)).astype(int)
            wave = z['wave'][ids]
            feat = scale_features(z['interval'][ids], self.median, self.scale)
        return wave, feat, row[self.classes].to_numpy(dtype=np.float32)


def collate(items):
    size = max(len(w) for w, _, _ in items)
    wave = torch.zeros(len(items), size, items[0][0].shape[-1])
    feat = torch.zeros(len(items), size, items[0][1].shape[-1])
    mask = torch.zeros(len(items), size, dtype=torch.bool)
    for i, (w, f, _) in enumerate(items):
        wave[i, :len(w)] = torch.from_numpy(w)
        feat[i, :len(w)] = torch.from_numpy(f)
        mask[i, :len(w)] = True
    return wave, feat, mask, torch.tensor(np.stack([y for _, _, y in items]))


def metrics_by_cohort(frame, y, probability, classes, thresholds):
    labeled = (y >= 0).all(1)
    cohorts = {'all_labeled_descriptive': labeled,
               'heldout_all_sources': labeled & frame.split.isin(['test', 'external', 'external_long']).to_numpy()}
    for split in frame.split.unique():
        cohorts[str(split)] = labeled & (frame.split == split).to_numpy()
        for source in frame.source.unique():
            cohorts[f'{split}/{source}'] = labeled & ((frame.split == split) & (frame.source == source)).to_numpy()
    return {name: dict(records=int(mask.sum()), **multilabel_metrics(y[mask], probability[mask], classes, thresholds))
            for name, mask in cohorts.items() if mask.any()}


def train_branch(cfg, root, frame, variant, classes, feature_spec=None):
    seed_all(cfg.seed)
    out = begin_cache(Path(cfg.output)/LABEL_VERSION/variant, dict(config=asdict(cfg), manifest=file_hash(root/'manifest.csv'),
        inputs=json.loads((root/'cache_identity.json').read_text()), variant=variant, classes=classes,
        label_version=LABEL_VERSION, diagnosis_codes={c: TARGETS[c] for c in classes},
        **({'feature_spec': feature_spec} if feature_spec is not None else {})))
    if (out/'metrics.json').exists():
        return json.loads((out/'metrics.json').read_text())
    labeled = (frame[classes].to_numpy(dtype=int) >= 0).all(1)
    tr = (frame.split == 'train').to_numpy() & labeled
    va = (frame.split == 'valid').to_numpy() & labeled
    if not tr.any() or not va.any():
        raise ValueError('Labeled train and validation records required')
    branch = 'unet' if variant == 'waveform' else variant
    # Bounded per-record sample, train only; no holdout imputation statistics.
    samples = []
    for row in frame.loc[tr].itertuples():
        with np.load(root/branch/f'{row.key}.npz') as z:
            samples.append(z['interval'][np.linspace(0, len(z['interval'])-1, min(16, len(z['interval']))).astype(int)])
    f = np.concatenate(samples)
    f[~np.isfinite(f)] = np.nan
    median = np.nan_to_num(np.nanmedian(f, axis=0)).astype(np.float32)
    scale = np.maximum(np.std(np.where(np.isfinite(f), f, median), axis=0), 1e-5).astype(np.float32)
    def loader(subset, shuffle=False):
        return DataLoader(Bags(subset, root, variant, classes, median, scale, cfg.max_beats),
                          batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate)
    train_loader, valid_loader = loader(frame.loc[tr], True), loader(frame.loc[va])
    model = IntervalClassifier(len(classes), len(median)*2, variant != 'waveform').to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    yt = frame.loc[tr, classes].to_numpy(dtype=np.float32)
    pos = yt.sum(0)
    weight = torch.tensor(np.clip((len(yt)-pos)/np.maximum(pos, 1), .2, 20), device=cfg.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    def infer(dl):
        model.eval()
        probabilities = []
        with torch.no_grad():
            for w, f, m, _ in dl:
                probabilities.append(model(w.to(cfg.device), f.to(cfg.device), m.to(cfg.device)).sigmoid().cpu().numpy())
        return np.concatenate(probabilities)
    best, stale, start, history = -1., 0, 0, []
    latest = out/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location=cfg.device, weights_only=True)
        model.load_state_dict(saved['state_dict']); opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        best, stale, start, history = saved['best'], saved['stale'], saved['epoch'], saved['history']
        torch.set_rng_state(saved['rng'].cpu())
        if cfg.device.startswith('cuda') and 'cuda_rng' in saved:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    for epoch in range(start, cfg.epochs):
        if stale >= cfg.patience:
            break
        model.train(); losses = []
        for w, f, m, y in train_loader:
            w, f, m, y = [a.to(cfg.device) for a in (w, f, m, y)]
            w = w*(.9+.2*torch.rand(len(w), 1, 1, device=cfg.device)) + torch.randn_like(w)*.01
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(w, f, m), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
        p = infer(valid_loader)
        metric = multilabel_metrics(frame.loc[va, classes].to_numpy(dtype=int), p, classes)
        score = metric['macro_auroc']
        if score is None:
            raise ValueError('Validation has no class with both positive and negative labels')
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), valid_macro_auroc=float(score)))
        if score > best:
            best, stale = float(score), 0
            _atomic_save(dict(state_dict=model.state_dict(), classes=classes, median=median.tolist(), scale=scale.tolist(),
                use_intervals=variant != 'waveform', max_beats=cfg.max_beats, variant=variant,
                input_identity=json.loads((root/'cache_identity.json').read_text()),
                label_version=LABEL_VERSION, diagnosis_codes={c: TARGETS[c] for c in classes},
                **({'feature_spec': feature_spec} if feature_spec is not None else {})), out/'best.pt')
        else:
            stale += 1
        state = dict(state_dict=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                     best=best, stale=stale, epoch=epoch+1, history=history, rng=torch.get_rng_state())
        if cfg.device.startswith('cuda'):
            state['cuda_rng'] = torch.cuda.get_rng_state_all()
        _atomic_save(state, latest)
        print(variant, history[-1], flush=True)
    saved = torch.load(out/'best.pt', map_location=cfg.device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    probabilities = infer(loader(frame))
    y = frame[classes].to_numpy(dtype=int)
    thresholds = thresholds_on_validation(y[va], probabilities[va])
    saved['thresholds'] = thresholds.tolist()
    _atomic_save(saved, out/'best.pt')
    np.savez_compressed(out/'predictions.npz', probability=probabilities, labels=y, classes=np.array(classes),
        key=frame.key.to_numpy(dtype=str), source=frame.source.to_numpy(dtype=str),
        record_id=frame.record_id.to_numpy(dtype=str), split=frame.split.to_numpy(dtype=str))
    report = metrics_by_cohort(frame, y, probabilities, classes, thresholds)
    save_json(out/'history.json', history)
    save_json(out/'metrics.json', report)
    return report


def evaluate_manual(cfg, qwen):
    """Identical manual validation arrays plus untouched LUDB/QT evaluation."""
    from ecg_project.models.segmentation import Predictor as Unet, evaluate as ludb_evaluate
    from ecg_project.models.qwen_delineator import Predictor as Qwen
    from ecg_project.training.qtdb import evaluate as qt_evaluate
    root = Path(cfg.manual_root)
    x = np.load(root/'valid_x.npy', mmap_mode='r')
    y = np.load(root/'valid_y.npy', mmap_mode='r')
    rows = [r for r in json.loads((root/'manifest.json').read_text()) if r['split'] == 'valid']
    if len(rows) != len(x) or len(y) != len(x):
        raise ValueError('Manual manifest/array alignment mismatch')
    results = {}
    for variant in ('unet', 'qwen'):
        predictor = Unet(cfg.unet_checkpoint, cfg.device) if variant == 'unet' else Qwen(qwen, cfg.model_root, cfg.device)
        counts = {}
        with torch.no_grad():
            for i in tqdm(range(len(x)), desc=f'{variant} manual validation'):
                p = predictor.model(torch.from_numpy(np.array(x[i:i+1])).to(cfg.device)).argmax(1).cpu().numpy()[0]
                truth = y[i]; mask = truth >= 0
                cm = counts.setdefault(rows[i]['source'], np.zeros((4, 4), dtype=np.int64))
                cm += np.bincount(truth[mask]*4+p[mask], minlength=16).reshape(4, 4)
        result = {'valid': {s: dict(confusion_matrix=cm.tolist(),
            dice=[float(2*cm[c,c]/(cm[c].sum()+cm[:,c].sum())) if cm[c].sum()+cm[:,c].sum() else None for c in range(4)])
            for s, cm in counts.items()}}
        result['LUDB_test'] = ludb_evaluate(root=cfg.ludb_root, predictor=predictor,
            output=str(Path(cfg.output)/f'{variant}_ludb_test.json'))
        result['QT_external'] = qt_evaluate(root=cfg.qt_root, predictor=predictor,
            output=str(Path(cfg.output)/f'{variant}_qt_external.json'))
        results[variant] = result
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(Path(cfg.output)/'manual_comparison.json', results)
    return results


def run(cfg=ClassificationConfig()):
    if min(cfg.epochs, cfg.patience, cfg.batch_size, cfg.max_beats) < 1 or cfg.learning_rate <= 0:
        raise ValueError('Invalid classifier training parameters')
    qwen = selected_qwen(cfg.qwen_run)
    if cfg.evaluate_manual:
        for path in (Path(cfg.manual_root)/'valid_x.npy', Path(cfg.manual_root)/'valid_y.npy',
                     Path(cfg.manual_root)/'manifest.json', Path(cfg.qt_root)/'split.json'):
            if not path.is_file():
                raise FileNotFoundError(f'Missing manual evaluation input: {path}')
        if not list(Path(cfg.ludb_root).glob('*.hea')):
            raise FileNotFoundError(f'Missing LUDB: {cfg.ludb_root}')
    root, frame = prepare(cfg, qwen)
    # Keep the original source universe even if deduplication or zero-beat
    # filtering removes all usable records from a source.
    from ecg_project.data.policy import source_name
    catalog_frame = build_record_manifest(cfg.catalog)
    required_sources = set(catalog_frame.source.map(source_name))
    frame = frame[frame.beats > 0].reset_index(drop=True)
    classes, label_report = select_common_classes(frame, required_sources)
    save_json(Path(cfg.output)/'diagnosis_coverage.json', label_report)
    print('Common diagnosis classes:', ', '.join(classes), flush=True)
    if not classes:
        raise ValueError('No common diagnosis with >=20 positive and negative training records; see diagnosis_coverage.json')
    results = {variant: train_branch(cfg, root, frame, variant, classes) for variant in ('waveform', 'unet', 'qwen')}
    selected = max(results, key=lambda v: results[v]['valid']['macro_auroc'])
    report = dict(config=asdict(cfg), qwen_checkpoint=str(qwen/'best.pt'), classes=classes,
        label_version=LABEL_VERSION, diagnosis_coverage=label_report,
        classifier_root=str(Path(cfg.output)/LABEL_VERSION),
        unsupported_classes=[c for c in TARGETS if c not in classes], selected=selected,
        selection_criterion='validation macro AUROC only', metrics=results,
        evaluation_status='Qwen saw all datasets: downstream holdouts are not proven unseen by the delineator'
            if cfg.qwen_seen_all_datasets else 'Delineator pretraining overlap has not been independently audited',
        all_labeled_status='Descriptive, includes classifier training records; use heldout_all_sources for downstream evaluation',
        protocol='Single II/MLII/first lead; full-record intervals, uniformly sampled beat bags; no diagnosis metrics for unknown labels')
    save_json(Path(cfg.output)/'comparison.json', report)
    table = [dict(variant=v, cohort=c, records=m['records'], macro_auroc=m['macro_auroc'], macro_f1=m['macro_f1'])
             for v, cohorts in results.items() for c, m in cohorts.items()]
    pd.DataFrame(table).to_csv(Path(cfg.output)/'comparison.csv', index=False)
    if cfg.evaluate_manual:
        evaluate_manual(cfg, qwen)
    return report


class ClassificationPredictor:
    """Load a final classifier and its frozen delineator for a new ECG record."""
    def __init__(self, checkpoint, delineator_checkpoint=None, model_root=None, device='cuda'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.device, self.classes = device, saved['classes']
        self.median = np.asarray(saved['median'], np.float32)
        self.scale = np.asarray(saved['scale'], np.float32)
        self.thresholds = np.asarray(saved['thresholds'])
        self.max_beats = saved['max_beats']
        self.feature_spec = saved.get('feature_spec')
        self.model = IntervalClassifier(len(self.classes), len(self.median)*2, saved['use_intervals']).to(device)
        self.model.load_state_dict(saved['state_dict']); self.model.eval()
        self.delineator = None
        if saved['use_intervals']:
            if delineator_checkpoint is None:
                raise ValueError('Supply the interval checkpoint used to train this classifier')
            if saved['variant'] == 'qwen':
                root = selected_qwen(delineator_checkpoint)
                if file_hash(root/'best.pt') != saved['input_identity']['qwen']:
                    raise ValueError('Qwen checkpoint mismatch')
                from ecg_project.models.qwen_delineator import Predictor
                self.delineator = Predictor(root, model_root, device)
            else:
                if file_hash(delineator_checkpoint) != saved['input_identity']['unet']:
                    raise ValueError('U-Net checkpoint mismatch')
                from ecg_project.models.segmentation import Predictor
                self.delineator = Predictor(delineator_checkpoint, device)

    def predict(self, record):
        lead = next((l for l in ('II', 'MLII') if l in record.leads), record.leads[0])
        signal = record.signal[:, record.leads.index(lead)]
        peaks, _ = rpeaks(preprocess(signal, record.fs), record.fs)
        if self.feature_spec is not None:
            from ecg_project.training.trust_intervals import TrustConfig, record_features
            spec = self.feature_spec
            if spec['method'] != 'foreground_probability_bands_v1':
                raise ValueError('Unknown classifier uncertainty method')
            wave, point, trust, _ = record_features(signal, record.fs, peaks, self.delineator,
                                                   TrustConfig(**spec['config']))
            features = trust if spec['mode'] == 'trust' else point
        else:
            waves = self.delineator.predict(signal, record.fs)[0] if self.delineator is not None else []
            wave, features, _ = interval_features(signal, peaks, waves, record.fs)
        if not len(wave):
            return dict(status='no_usable_beats', predictions={}, lead=lead)
        ids = np.linspace(0, len(wave)-1, min(len(wave), self.max_beats)).astype(int)
        w = torch.from_numpy(wave[ids].astype(np.float32))[None].to(self.device)
        f = torch.from_numpy(scale_features(features[ids], self.median, self.scale))[None].to(self.device)
        with torch.no_grad():
            p = self.model(w, f, torch.ones(w.shape[:2], dtype=torch.bool, device=self.device)).sigmoid()[0].cpu().numpy()
        return dict(status='ok', lead=lead, available_beats=len(wave), selected_beats=len(ids),
                    predictions={c: dict(probability=float(v), threshold=float(t), flag=bool(v >= t))
                                 for c, v, t in zip(self.classes, p, self.thresholds)})
