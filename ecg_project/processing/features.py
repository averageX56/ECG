"""Interval features and fixed convolutions of complete beat waveforms."""
import numpy as np
from scipy.ndimage import convolve1d
from ecg_project.data.io import LEADS
from ecg_project.processing.signal import preprocess,resample,beat_windows,interval_features,rpeaks

def waveform_features(beats):
    """ROCKET-inspired frozen kernels; not an implementation of MiniROCKET.

    Beats have fixed 0.9 second duration at 250 Hz. The kernels are generated
    independently of every dataset; median and atypical beats retain morphology.
    """
    if len(beats)==0:return np.full(128,np.nan,dtype=np.float32)
    median=np.median(beats,axis=0)
    distance=np.mean((beats-median)**2,axis=1)
    atypical=beats[int(np.argmax(distance))]
    rng=np.random.default_rng(1729);out=[]
    for k in range(32):
        size=int(rng.choice([7,9,11]));dilation=int(rng.choice([1,2,4,8]))
        w=rng.normal(size=size);w-=w.mean();w/=np.linalg.norm(w)
        kernel=np.zeros((size-1)*dilation+1);kernel[::dilation]=w
        bias=rng.uniform(-.5,.5)
        for template in [median,atypical]:
            v=convolve1d(template,kernel,mode='reflect')
            out.extend([float(v.max()),float(np.mean(v>bias))])
    return np.asarray(out,dtype=np.float32)

def extract_record(rec,predictor):
    x=resample(preprocess(rec.signal,rec.fs),rec.fs)
    waves=predictor.predict(rec.signal,rec.fs)
    intervals={};wf={};qc={};canonical={lead:i for i,lead in enumerate(rec.leads)}
    ref=canonical.get('II',0)
    peaks=np.array([w['peak'] for w in waves[ref] if w['wave']=='QRS'],dtype=int)
    if len(peaks)<3:
        peaks,_=rpeaks(preprocess(rec.signal[:,ref],rec.fs),rec.fs)
    for lead in LEADS:
        if lead not in canonical:
            f=interval_features([],np.array([],dtype=int),rec.fs)
            intervals.update({lead+'/'+k:np.nan for k in f});wf.update({lead+f'/conv_{i}':np.nan for i in range(128)})
            wf[lead+'/amplitude_std_mv']=np.nan
            continue
        i=canonical[lead];w=waves[i];p=np.array([v['peak'] for v in w if v['wave']=='QRS'],dtype=int)
        f=interval_features(w,p,rec.fs)
        intervals.update({lead+'/'+k:v for k,v in f.items()})
        # Keep absolute mV scale and variation in the feature branch.
        xx=x[:,i];scale=max(np.std(xx),1e-5)
        b,ids=beat_windows(xx/scale,np.round(peaks*250/rec.fs).astype(int),250)
        conv=waveform_features(b)
        wf.update({lead+f'/conv_{j}':v for j,v in enumerate(conv)})
        wf[lead+'/amplitude_std_mv']=np.std(xx) if rec.unit=='mV' else np.nan
        qc[lead+'/flat_fraction']=float(np.mean(np.diff(rec.signal[:,i])==0))
        qc[lead+'/qrs_count']=len(p)
    # Across-lead disagreement is a useful uncertainty feature, not a hard diagnosis rule.
    qrs=[intervals.get(l+'/qrs_ms_median',np.nan) for l in LEADS]
    valid=np.asarray(qrs)[np.isfinite(qrs)]
    intervals['global/qrs_lead_range_ms']=np.ptp(valid) if len(valid) else np.nan
    return intervals,wf,qc

BEAT_FEATURE_NAMES=['rr_previous_s','rr_next_s','rr_previous_relative','rr_next_relative','compensatory_ratio',
    'qrs_ms','pr_ms','p_present','template_correlation','template_distance','qrs_relative_to_record']

def beat_features(signal,peaks,waves,fs):
    """Offline context uses neighboring beats in this recording, never their labels."""
    peaks=np.asarray(peaks,dtype=int);x=resample(preprocess(signal,fs),fs)
    if x.ndim>1:x=x[:,0]
    x=(x-np.median(x))/max(np.std(x),1e-5)
    crops,ids=beat_windows(x,np.round(peaks*250/fs).astype(int),250)
    template=np.median(crops,axis=0) if len(crops) else np.zeros(226)
    qs=[w for w in waves if w['wave']=='QRS'];ps=[w for w in waves if w['wave']=='P']
    rr=np.diff(peaks)/fs;med=np.median(rr) if len(rr) else np.nan
    widths=[(q['offset']-q['onset'])/fs*1000 for q in qs];mw=np.median(widths) if widths else np.nan
    features=[]
    for crop,i in zip(crops,ids):
        p=peaks[i];prev=rr[i-1] if i else np.nan;following=rr[i] if i<len(rr) else np.nan
        local=rr[max(0,i-5):min(len(rr),i+5)];localmed=np.median(local) if len(local) else med
        near=[q for q in qs if abs(q['peak']-p)<=.15*fs]
        q=min(near,key=lambda w:abs(w['peak']-p)) if near else None
        width=(q['offset']-q['onset'])/fs*1000 if q else np.nan
        pp=[w for w in ps if q and 0<q['onset']-w['offset']<.35*fs]
        pr=(q['onset']-max(pp,key=lambda w:w['offset'])['onset'])/fs*1000 if pp else np.nan
        corr=np.corrcoef(crop,template)[0,1] if np.std(crop)>1e-6 and np.std(template)>1e-6 else np.nan
        features.append([prev,following,prev/localmed,following/localmed,(prev+following)/(2*localmed),
                         width,pr,float(bool(pp)),corr,np.mean((crop-template)**2),width/mw])
    return crops,np.asarray(features,dtype=np.float32).reshape(-1,len(BEAT_FEATURE_NAMES)),ids
