"""Separate preprocessing for HuBERT; never reuse ECGFounder-normalized inputs."""
import numpy as np
from scipy.signal import firwin,filtfilt,resample,decimate

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
VERSION='hubert_fir005_47_minmax500_flatdecimate5_twoview_v1'
VERSION_V2='hubert_fir005_47_twoview_fixed10s_no_padding_v2'


def select_window(rec,start_seconds=0):
    from copy import copy
    if start_seconds<0:raise ValueError('Negative window start')
    n=round(10*rec.fs);start=round(start_seconds*rec.fs)
    if start+n>len(rec.signal):raise ValueError('shorter_than_10_seconds; excluded without padding')
    selected=copy(rec);selected.signal=rec.signal[start:start+n].copy()
    return selected,dict(original_duration=len(rec.signal)/rec.fs,window_start=start/rec.fs,window_end=(start+n)/rec.fs)


def prepare_signal_v2(rec,start_seconds=0):
    selected,metadata=select_window(rec,start_seconds)
    return prepare_signal(selected),metadata


def prepare_signal(rec):
    if len(rec.leads)!=12 or set(rec.leads)!=set(LEADS):raise ValueError('HuBERT branch requires all twelve canonical leads')
    if abs(len(rec.signal)/rec.fs-10)>.02:raise ValueError('Expected ten-second record; window long records explicitly')
    x=rec.signal[:,[rec.leads.index(l) for l in LEADS]].T.astype(np.float64)
    if not np.isfinite(x).all():raise ValueError('Non-finite input; do not silently fabricate an ECG')
    # BioSPPy FIR order=.3*fs is made odd and passed as numtaps.
    taps=int(.3*rec.fs);taps+=1 if taps%2==0 else 0
    b=firwin(taps,[.05,47],pass_zero=False,fs=rec.fs)
    x=filtfilt(b,[1.],x,axis=-1)
    x=resample(x,5000,axis=-1)
    x=2*(x-x.min(1,keepdims=True))/(np.ptp(x,axis=1,keepdims=True)+1e-8)-1
    # Upstream flattens leads then decimates, including filtering at lead joins.
    return np.stack([decimate(x[:,start:start+2500].reshape(-1),5) for start in [0,2500]]).astype(np.float32)
