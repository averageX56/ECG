"""Bounded, ordered CPU processes; each worker uses one numerical thread."""
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from collections import deque


@lru_cache(maxsize=1)
def cached_predictor(checkpoint,device):
    from ecg_project.models.segmentation import Predictor
    return Predictor(checkpoint,device=device)


def initialize_worker():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import torch
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def ordered_map(function,items,workers=None):
    workers=int(os.environ.get('ECG_CPU_WORKERS','1')) if workers is None else workers
    if workers<1:raise ValueError('workers must be positive')
    if workers==1:
        yield from map(function,items)
    else:
        # spawn works consistently on Windows and avoids forking a CUDA runtime.
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            iterator=iter(items);pending=deque()
            for _ in range(workers*2):
                try:pending.append(pool.submit(function,next(iterator)))
                except StopIteration:break
            while pending:
                yield pending.popleft().result()
                try:pending.append(pool.submit(function,next(iterator)))
                except StopIteration:pass
