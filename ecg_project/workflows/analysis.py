from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from ecg_project.data.catalog import file_hash
from ecg_project.data.io import load_record,annotations,LEADS
from ecg_project.processing.signal import preprocess,rpeaks,interval_features,beat_windows
from ecg_project.models.segmentation import Predictor
from ecg_project.utils import save_json,seed_all

def ventricular_runs(beats,fs,min_rate=100,min_probability=.8):
    """Candidate VT episodes, not an etiological diagnosis from QRS width."""
    runs=[];active=[]
    def finish():
        if len(active)<3:return
        samples=np.array([b['peak'] for b in active]);rr=np.diff(samples)/fs
        if np.any(rr<=0):return
        rate=60/np.mean(rr)
        if rate<min_rate:return
        runs.append(dict(start_sample=int(samples[0]),end_sample=int(samples[-1]),
            n_beats=len(active),rate_bpm=float(rate),observed_duration_s=float((samples[-1]-samples[0])/fs),
            type='suspected_ventricular_tachycardia',confidence_min=min(b['probabilities'].get('V',0) for b in active),
            criteria=f'>=3 consecutive predicted V beats, mean rate >={min_rate}/min; ventricular origin not independently proven'))
    for b in beats:
        if b['class_name']=='V' and b['probabilities'].get('V',0)>=min_probability:
            if active and b['beat_index']!=active[-1]['beat_index']+1:finish();active=[]
            active.append(b)
        else:finish();active=[]
    finish();return runs

def selected_path(directory):
    p=Path(directory);s=p/'selection.json'
    if not s.exists():return None
    return p/(json.loads(s.read_text())['selected']+'.joblib')

def batch(manifest,output='reports/batch',checkpoint=None):
    """Input CSV has a path column. Failures are explicit and do not stop other records."""
    frame=pd.read_csv(manifest);rows=[]
    if 'path' not in frame:raise ValueError('Manifest must include path')
    for i,path in enumerate(frame.path):
        destination=Path(output)/(str(i)+'_'+Path(path).stem)
        try:
            analyze(path,output=destination,checkpoint=checkpoint,make_plot=False)
            rows.append(dict(path=path,status='ok',output=str(destination)))
        except Exception as e:rows.append(dict(path=path,status='failed',error=repr(e)))
        save_json(Path(output)/'batch_status.json',rows)
    return rows

def analyze(path,output='reports/example',checkpoint=None,csv_fs=None,
            lead=None,start_seconds=0,duration_seconds=None,device='cpu',make_plot=True,beat_model_path=None,
            hubert_run=None,hubert_model_root='artifacts/hubert_large',record_model_path=None):
    seed_all();p=Path(path);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=checkpoint or ('artifacts/delineator_qt.pt' if Path('artifacts/delineator_qt.pt').exists() else 'artifacts/delineator.pt')
    if p.suffix=='.csv' and (Path('artifacts/mit_headers')/(p.stem+'.hea')).exists():
        from ecg_project.training.beats import load_mit
        rec=load_mit(p)
    else:rec=load_record(p,csv_fs=csv_fs)
    start=round(start_seconds*rec.fs);stop=min(len(rec.signal),start+round(duration_seconds*rec.fs)) if duration_seconds else len(rec.signal)
    if not 0<=start<stop:raise ValueError('Invalid requested time range')
    rec.signal=rec.signal[start:stop];rec.metadata['sample_start']=start
    predictor=Predictor(checkpoint,device=device);predicted=predictor.predict(rec.signal,rec.fs)
    reference=[]
    for l in rec.leads:
        if p.with_suffix('.'+l.lower()).exists():
            for w in annotations(p,l):
                if start<=w['onset'] and w['offset']<stop:
                    reference.append({**w,**{k:w[k]-start for k in ['onset','peak','offset']}})
    if p.with_suffix('.q1c').exists():
        from ecg_project.training.qtdb import manual
        for w in manual(p):
            left=w['onset'] if w['onset'] is not None else w['peak']
            if start<=left and w['offset']<stop:
                reference.append({**w,**{k:w[k]-start if w[k] is not None else None for k in ['onset','peak','offset']},
                                  'lead':'shared_two_lead_reference','source':'annotation'})
    rows=[];features={};widths=[]
    for l,waves in zip(rec.leads,predicted):
        peaks=np.array([w['peak'] for w in waves if w['wave']=='QRS'],dtype=int)
        features[l]=interval_features(waves,peaks,rec.fs)
        for w in waves:
            ms=(w['offset']-w['onset'])*1000/rec.fs
            row={**w,'lead':l,'duration_ms':ms,'onset_s':(start+w['onset'])/rec.fs,'offset_s':(start+w['offset'])/rec.fs,
                 'onset_abs':start+w['onset'],'peak_abs':start+w['peak'],'offset_abs':start+w['offset'],
                 'qrs_ge_120':ms>=120 if w['wave']=='QRS' else None}
            rows.append(row)
            if w['wave']=='QRS':widths.append(ms)
    wave_columns=['wave','onset','peak','offset','confidence','source','lead','duration_ms','onset_s','offset_s','onset_abs','peak_abs','offset_abs','qrs_ge_120']
    pd.DataFrame(rows,columns=wave_columns).to_csv(out/'intervals_predicted.csv',index=False)
    pd.DataFrame(reference,columns=['wave','onset','peak','offset','lead','source']).to_csv(out/'intervals_reference.csv',index=False)
    chosen=lead or ('MLII' if 'MLII' in rec.leads else 'II' if 'II' in rec.leads else rec.leads[0])
    if chosen not in rec.leads:raise ValueError(f'Lead {chosen} absent; available: {rec.leads}')
    ch=rec.leads.index(chosen);xx=rec.signal[:,ch];peaks,_=rpeaks(preprocess(xx,rec.fs),rec.fs)
    crops,ids=beat_windows(rec.signal,peaks,rec.fs)
    np.savez_compressed(out/'beats.npz',signal=crops,peak=peaks[ids],peak_abs=peaks[ids]+start,
                        sample_start=peaks[ids]-round(.35*rec.fs)+start,fs=rec.fs,leads=np.array(rec.leads),unit=rec.unit)
    beat_model=Path(beat_model_path) if beat_model_path else selected_path('artifacts/beat_models');beat_results=[];beat_status='model_not_trained'
    if beat_model and joblib.load(beat_model)['feature_model_hash']!=file_hash(checkpoint):
        beat_status='classifier_requires_features_from_its_training_delineator';beat_model=None
    if beat_model:
        from ecg_project.training.beats import predict
        beat_results=predict(xx,peaks,predicted[ch],rec.fs,beat_model,checkpoint=checkpoint)
        beat_status='evaluated_lead_MLII' if chosen=='MLII' else 'unvalidated_transfer_from_MLII'
    pd.DataFrame(beat_results).to_csv(out/'beat_classification.csv',index=False)
    episodes=ventricular_runs(beat_results,rec.fs)
    for ep in episodes:
        ep['start_s']=(ep['start_sample']+start)/rec.fs;ep['end_s']=(ep['end_sample']+start)/rec.fs
    record_predictions={};unsupported={};record_status='model_not_trained'
    model=Path(record_model_path) if record_model_path else selected_path('artifacts/record_models')
    if model and joblib.load(model)['feature_model_hash']!=file_hash(checkpoint):
        record_status='classifier_requires_features_from_its_training_delineator';model=None
    if model:
        if set(LEADS).issubset(rec.leads):
            from ecg_project.training.record_model import predict
            if len(rec.signal)/rec.fs<=30:
                record_predictions,unsupported=predict(rec,predictor,model);record_status='screening_scores'
            else:
                # Process all non-overlapping 10s windows. No whole-record label is used as a window label.
                from ecg_project.data.io import Record
                window_rows=[]
                for a in range(0,len(rec.signal),round(10*rec.fs)):
                    b=min(len(rec.signal),a+round(10*rec.fs))
                    if b-a<5*rec.fs:continue
                    window=Record(rec.record_id,rec.signal[a:b],rec.fs,rec.leads,rec.metadata,rec.unit)
                    pp,unsupported=predict(window,predictor,model)
                    window_rows.append(dict(start_s=(a+start)/rec.fs,end_s=(b+start)/rec.fs,predictions=pp))
                save_json(out/'window_classification.json',window_rows)
                record_status='window_scores; full-record aggregation not calibrated'
        else:record_status='unsupported_record_lead_set; use beat results'
    result=dict(record_id=rec.record_id,fs=rec.fs,leads=rec.leads,unit=rec.unit,metadata=rec.metadata,
        start_seconds=start/rec.fs,analyzed_duration_seconds=len(rec.signal)/rec.fs,
        interval_features=features,record_predictions=record_predictions,record_status=record_status,
        unsupported_training_labels=unsupported,beat_status=beat_status,reference_lead=chosen,
        beat_counts=dict(pd.Series([b['class_name'] for b in beat_results],dtype=str).value_counts()),
        vt_candidates=episodes,af_form='not_determined_without_clinical_history',
        conduction_assessment=dict(qrs_ge_120_fraction=float(np.mean(np.array(widths)>=120)) if widths else None,
             clinical_criticality='not_determined_from_QRS_duration_alone',
             note='Per-lead widths, not global earliest-onset/latest-offset QRS; boundary error can change the 120ms flag.'),
        provenance=dict(delineator=str(checkpoint),record_model=str(model) if model else None,beat_model=str(beat_model) if beat_model else None))
    if hubert_run:
        from ecg_project.training.hubert_lora import predict_record
        result['hubert_predictions']=predict_record(rec,hubert_run,hubert_model_root,device)
    save_json(out/'analysis.json',result)
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(16,7),sharex=True)
        display_n=min(len(xx),round(10*rec.fs));time=np.arange(display_n)/rec.fs+start/rec.fs
        for ax,title,wave_rows in [(axes[0],'Manual annotations (unknown onset: peak to offset only)',[w for w in reference if w['lead'] in [chosen,'shared_two_lead_reference']]),
                                   (axes[1],'Predicted delineation',predicted[ch])]:
            ax.plot(time,xx[:display_n],lw=.7,color='#202530')
            for w in wave_rows:
                left=w['onset'] if w['onset'] is not None else w['peak']
                if left<display_n:
                    ax.axvspan((start+left)/rec.fs,(start+min(w['offset'],display_n))/rec.fs,
                               color={'P':'#35a86b','QRS':'#e95f54','T':'#5097db'}[w['wave']],alpha=.28)
            ax.set_title(title+' — '+chosen);ax.set_ylabel(rec.unit);ax.grid(alpha=.2)
        axes[-1].set_xlabel('Time, s');fig.tight_layout();fig.savefig(out/'delineation.png',dpi=160);plt.close(fig)
    print(f'Analysis written to {out.resolve()}',flush=True)
    return result
