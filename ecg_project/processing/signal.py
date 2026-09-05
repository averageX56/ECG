import warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction

def preprocess(x, fs, mode='morphology'):
    """Zero-phase processing. Morphology uses 0.5–40 Hz; raw remains available."""
    x=np.asarray(x,dtype=np.float64)
    if fs < 100 or len(x)<fs: raise ValueError('At least 1 s and fs >=100 Hz are required')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite signal')
    if mode=='raw':return x.copy()
    if mode not in ('morphology','detector'): raise ValueError(mode)
    lo,hi=(.5,40) if mode=='morphology' else (5,25)
    sos=butter(3,[lo,min(hi,fs*.45)],fs=fs,btype='bandpass',output='sos')
    return sosfiltfilt(sos,x,axis=0)

def resample(x, fs, target=250):
    f=Fraction(float(target)/float(fs)).limit_denominator(10000)
    return resample_poly(x,f.numerator,f.denominator,axis=0)

def rpeaks(x, fs):
    import neurokit2 as nk
    # Sign chosen from QRS-band extrema; needed for inverted leads such as aVR.
    band=preprocess(x,fs,'detector')
    sign=1 if np.quantile(band,.995)>=abs(np.quantile(band,.005)) else -1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, info=nk.ecg_peaks(sign*x,sampling_rate=fs,method='neurokit',correct_artifacts=False)
    return np.asarray(info['ECG_R_Peaks'],dtype=int),sign

def delineate(x, fs, mode='morphology', method='dwt'):
    import neurokit2 as nk
    y=preprocess(x,fs,mode)
    if np.ptp(y)<1e-8: return [],np.array([],dtype=int),{'status':'flat_signal'}
    peaks,sign=rpeaks(y,fs)
    if len(peaks)<3: return [],peaks,{'status':'too_few_peaks'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _,w=nk.ecg_delineate(sign*y,peaks,sampling_rate=fs,method=method,check=False)
    except (ValueError,IndexError,ZeroDivisionError) as e:
        return [],peaks,{'status':'delineation_failed','error':str(e)}
    out=[]
    for wave,keys in {'P':('ECG_P_Onsets','ECG_P_Peaks','ECG_P_Offsets'),
                      'QRS':('ECG_R_Onsets',None,'ECG_R_Offsets'),
                      'T':('ECG_T_Onsets','ECG_T_Peaks','ECG_T_Offsets')}.items():
        values=[peaks if k is None else w.get(k,[np.nan]*len(peaks)) for k in keys]
        for i,abc in enumerate(zip(*values)):
            if not np.isfinite(abc).all():continue
            a,b,c=map(int,abc)
            if 0<=a<=b<=c<len(x) and c>a:
                out.append(dict(wave=wave,onset=a,peak=b,offset=c,beat=i,source='predicted'))
    return sorted(out,key=lambda z:z['onset']),peaks,{'status':'ok','polarity':sign,'n_peaks':len(peaks)}

def beat_windows(signal, peaks, fs, before=.35, after=.55):
    """Fixed physical time; never time-warp QRS duration. Return only complete beats."""
    pre,post=round(before*fs),round(after*fs)
    ids=np.array([i for i,p in enumerate(peaks) if p-pre>=0 and p+post<=len(signal)],dtype=int)
    if len(ids)==0:return np.empty((0,pre+post)+signal.shape[1:],dtype=np.float32),ids
    return np.stack([signal[peaks[i]-pre:peaks[i]+post] for i in ids]).astype(np.float32),ids

def interval_features(waves, peaks, fs):
    """Only measured intervals; absent waves remain missing, not duration zero."""
    values={k:[] for k in ('p_ms','qrs_ms','t_ms','pr_ms','qt_ms','qtc_fridericia_ms','rr_ms')}
    values['rr_ms']=(np.diff(peaks)/fs*1000).tolist()
    for w in waves:values[w['wave'].lower()+'_ms'].append((w['offset']-w['onset'])/fs*1000)
    qs=sorted([w for w in waves if w['wave']=='QRS'],key=lambda w:w['peak'])
    for q in qs:
        ps=[w for w in waves if w['wave']=='P' and 0<q['onset']-w['offset']<.35*fs]
        ts=[w for w in waves if w['wave']=='T' and 0<w['onset']-q['offset']<.5*fs]
        nextq=next((v for v in qs if v['peak']>q['peak']),None)
        if ps:values['pr_ms'].append((q['onset']-max(ps,key=lambda w:w['offset'])['onset'])/fs*1000)
        if ts:
            t=min(ts,key=lambda w:w['onset'])
            if nextq is not None and t['offset']>=nextq['onset']:continue
            qt=(t['offset']-q['onset'])/fs
            values['qt_ms'].append(qt*1000)
            prior=peaks[peaks<q['peak']]
            if len(prior):
                rr=(q['peak']-prior[-1])/fs
                if rr>.2:values['qtc_fridericia_ms'].append(qt/rr**(1/3)*1000)
    features={}
    for k,v in values.items():
        features[k+'_median']=float(np.median(v)) if len(v) else np.nan
        features[k+'_iqr']=float(np.subtract(*np.percentile(v,[75,25]))) if len(v) else np.nan
        features[k+'_count']=len(v)
    rr=np.diff(peaks)/fs
    features['rr_cv']=float(np.std(rr)/np.mean(rr)) if len(rr)>1 else np.nan
    features['rmssd_ms']=float(np.sqrt(np.mean(np.diff(rr)**2))*1000) if len(rr)>1 else np.nan
    features['qrs_ge_120_fraction']=float(np.mean(np.asarray(values['qrs_ms'])>=120)) if values['qrs_ms'] else np.nan
    return features
