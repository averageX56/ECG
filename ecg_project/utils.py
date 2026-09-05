import json
import random
from pathlib import Path
import numpy as np

def save_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, Path): return str(x)
        return x
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def seed_all(seed=42):
    import os
    os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
