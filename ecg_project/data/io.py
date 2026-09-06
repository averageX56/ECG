from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat

LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')

def canonical_lead(name):
    return {x.lower(): x for x in LEADS}.get(name.lower(), name)

@dataclass
class Record:
    record_id: str
    signal: np.ndarray  # samples, channels; physical mV unless explicitly unknown
    fs: float
    leads: list[str]
    metadata: dict = field(default_factory=dict)
    unit: str = 'mV'

def header(path):
    p = Path(path).with_suffix('.hea')
    lines = p.read_text(encoding='utf-8-sig').splitlines()
    first = lines[0].split()
    n = int(first[1])
    meta = {}
    for line in lines[1+n:]:
        if line.startswith('#') and ':' in line:
            key, value = line[1:].split(':', 1)
            meta[key.strip('<> ').lower()] = value.strip()
    return dict(path=str(p), record_id=p.stem, fs=float(first[2].split('/')[0]),
                n_samples=int(first[3]), n_leads=n, channels=[s.split() for s in lines[1:n+1]],
                metadata=meta, labels=[x.strip() for x in meta.get('dx', '').split(',') if x.strip()])

def load_record(path, start=0, stop=None, csv_fs=None, csv_gain=None, csv_baseline=None):
    p = Path(path)
    if p.suffix == '.csv':
        if csv_fs is None:
            raise ValueError('CSV requires explicit sampling frequency; amplitude units require gain and baseline.')
        df = pd.read_csv(p)
        index = df.iloc[:, 0].to_numpy()
        if not np.array_equal(index, np.arange(len(index))):
            raise ValueError('CSV first column must be contiguous zero-based sample indices.')
        x = df.iloc[start:stop, 1:].to_numpy(dtype=np.float64)
        unit = 'ADC (uncalibrated)'
        if csv_gain is not None and csv_baseline is not None:
            if csv_gain <= 0: raise ValueError('gain must be positive')
            x = (x-csv_baseline)/csv_gain
            unit = 'mV'
        return Record(p.stem, x, float(csv_fs), [canonical_lead(c.strip("'\" ")) for c in df.columns[1:]],
                      {'sample_start':start, 'source':'mit-bih'}, unit)
    h = header(p)
    stop = h['n_samples'] if stop is None else min(stop, h['n_samples'])
    if not 0 <= start < stop: raise ValueError('Invalid sample range')
    files = {c[0] for c in h['channels']}
    if len(files) != 1: raise ValueError('Multi-file WFDB records are not supported')
    signal_file = p.parent / next(iter(files))
    if signal_file.suffix == '.dat':
        if not signal_file.exists():
            raise ValueError('Header references missing DAT; refusing to apply its gain to an unrelated MAT/NPY.')
        rec = wfdb.rdrecord(str(p.with_suffix('')), sampfrom=start, sampto=stop)
        x = rec.p_signal
    elif signal_file.suffix == '.mat':
        raw = loadmat(signal_file)['val']
        if raw.shape != (h['n_leads'], h['n_samples']): raise ValueError('MAT shape/header mismatch')
        x = raw[:,start:stop].T.astype(np.float64)
        for i, c in enumerate(h['channels']):
            m = re.fullmatch(r'([\d.eE+\-]+)(?:\((-?\d+)\))?/([^\s]+)', c[2])
            if m is None or m[3] != 'mV': raise ValueError('Unsupported gain/unit')
            gain = float(m[1]); base = float(m[2]) if m[2] else float(c[4])
            if gain <= 0: raise ValueError('Nonpositive gain')
            x[:,i] = (x[:,i]-base)/gain
    else: raise ValueError('Expected DAT or MAT referenced by header')
    if not np.isfinite(x).all(): raise ValueError('Signal has nonfinite samples')
    meta=dict(h['metadata'], sample_start=start)
    unit='mV'
    if p.parent.name in ('WFDB_PTB-XL','Training_StPetersburg'):
        meta['calibration_status']='unverified_local_header_gain'
        unit='header-scaled (calibration unverified)'
    return Record(h['record_id'], x, h['fs'], [canonical_lead(c[-1]) for c in h['channels']],meta,unit)

def annotations(path, lead):
    """WFDB offsets retained as inclusive annotation sample coordinates."""
    a = wfdb.rdann(str(Path(path).with_suffix('')), lead.lower())
    out=[]; onset=None; peak=None; wave=None
    for sample, symbol in zip(a.sample, a.symbol):
        if symbol == '(':
            onset=int(sample); peak=None; wave=None
        elif symbol in ('p','N','t'):
            peak=int(sample); wave={'p':'P','N':'QRS','t':'T'}[symbol]
        elif symbol == ')':
            if onset is not None and peak is not None and onset <= peak <= sample:
                out.append(dict(wave=wave, onset=onset, peak=peak, offset=int(sample), lead=canonical_lead(lead), source='annotation'))
            onset=None; peak=None; wave=None
    return out

def mit_annotations(path):
    rows=[]
    for line in Path(path).read_text().splitlines()[1:]:
        v=line.split()
        if len(v)>=6:
            rows.append(dict(sample=int(v[1]), symbol=v[2], aux=' '.join(v[6:])))
    return rows
