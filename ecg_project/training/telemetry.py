"""Common invocation metrics, including interrupted GPU runs."""
from functools import wraps
from dataclasses import is_dataclass,asdict
from pathlib import Path
import inspect
import json
import time
import torch
from ecg_project.utils import save_json


def tracked(function):
    @wraps(function)
    def wrapper(*args,**kwargs):
        bound=inspect.signature(function).bind(*args,**kwargs);bound.apply_defaults();values=dict(bound.arguments)
        if args and is_dataclass(args[0]):values=asdict(args[0])
        elif 'cfg' in values and is_dataclass(values['cfg']):values=asdict(values['cfg'])
        elif 'config' in values and is_dataclass(values['config']):values=asdict(values['config'])
        out=Path(values['output']);start=time.monotonic();status='complete';error=None
        def read(p,default):return json.loads(p.read_text()) if p.exists() else default
        before=len(read(out/'history.json',[]))
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        try:return function(*args,**kwargs)
        except BaseException as e:status='interrupted';error=e;raise
        finally:
            # An incompatible resume is a read-only rejection, not a new run.
            incompatible=error and any(s in str(error).lower() for s in ('mismatch','differs','stale'))
            if (out/'latest.pt').exists() and not incompatible:
                elapsed=time.monotonic()-start;report=read(out/'run.json',{});history=read(out/'history.json',[]);best=read(out/'best_metrics.json',{})
                batch=report.get('actual_batch',values.get('batch_size',1))
                report.update(status=status,seconds_this_invocation=elapsed,actual_batch=batch,effective_batch=batch*values.get('accumulation',1),
                    max_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    max_memory_reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    epochs_completed=len(history),best_epoch=best.get('best_epoch'),
                    examples_per_second=sum(r.get('examples',0) for r in history[before:])/max(elapsed,1e-9))
                save_json(out/'run.json',report)
    return wrapper
