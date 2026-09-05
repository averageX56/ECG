import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_fscore_support

def match_events(ref, pred, tolerance):
    """Maximum cardinality then minimum distance, one-to-one matching."""
    ref=np.asarray(ref);pred=np.asarray(pred)
    if not len(ref) or not len(pred):return []
    dist=np.abs(ref[:,None]-pred[None,:])
    penalty=(min(len(ref),len(pred))+1)*(tolerance+1)
    cost=np.where(dist<=tolerance,dist,penalty)
    a,b=linear_sum_assignment(cost)
    return [(int(i),int(j)) for i,j in zip(a,b) if dist[i,j]<=tolerance]

def event_metrics(ref,pred,fs,tolerance_ms=150):
    pairs=match_events([w['peak'] for w in ref],[w['peak'] for w in pred],tolerance_ms*fs/1000)
    tp=len(pairs); fp=len(pred)-tp;fn=len(ref)-tp
    result=dict(tp=tp,fp=fp,fn=fn)
    for key in ('onset','peak','offset'):
        result[key+'_errors_ms']=[(pred[j][key]-ref[i][key])*1000/fs for i,j in pairs]
    return result

def summarize_events(items):
    tp=sum(x['tp'] for x in items);fp=sum(x['fp'] for x in items);fn=sum(x['fn'] for x in items)
    out=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,
             recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    for key in ('onset','peak','offset'):
        e=np.array([v for x in items for v in x[key+'_errors_ms']])
        out[key]={'mae_ms':np.mean(abs(e)) if len(e) else None,'bias_ms':np.mean(e) if len(e) else None,
                  'sd_ms':np.std(e) if len(e) else None,'within_40ms':np.mean(abs(e)<=40) if len(e) else None}
    return out

def multilabel_metrics(y,p,classes,thresholds=None):
    out={};thresholds=np.full(len(classes),.5) if thresholds is None else thresholds
    for i,c in enumerate(classes):
        yt=y[:,i];pr=p[:,i];positive=int(yt.sum());negative=int(len(yt)-positive)
        prec,rec,f1,_=precision_recall_fscore_support(yt,pr>=thresholds[i],average='binary',zero_division=0)
        out[c]=dict(positive=positive,negative=negative,auroc=roc_auc_score(yt,pr) if positive and negative else None,
                    auprc=average_precision_score(yt,pr) if positive and negative else None,
                    precision=prec,recall=rec,f1=f1,threshold=thresholds[i],prevalence=positive/len(yt))
    defined=[v['auroc'] for v in out.values() if v['auroc'] is not None]
    f1s=[v['f1'] for v in out.values() if v['positive'] and v['negative']]
    return dict(per_class=out,macro_auroc=np.mean(defined) if defined else None,
                macro_f1=np.mean(f1s) if f1s else None)

def thresholds_on_validation(y,p):
    ts=[]
    for i in range(y.shape[1]):
        if y[:,i].sum()<5 or (1-y[:,i]).sum()<5:ts.append(.5);continue
        grid=np.arange(.05,.96,.05)
        scores=[precision_recall_fscore_support(y[:,i],p[:,i]>=t,average='binary',zero_division=0)[2] for t in grid]
        ts.append(float(grid[np.argmax(scores)]))
    return np.array(ts)
