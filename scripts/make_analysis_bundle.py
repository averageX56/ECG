"""Portable, bounded analysis ZIP without weights, raw signals or feature shards."""
import argparse
from pathlib import Path
import zipfile

NAMES={'config.json','history.json','best_metrics.json','run.json','selection.json',
       'provenance.json','manifest.json','manifest.csv','class_support.json',
       'exclusions.json','excluded.json','protected_registry.json'}
PREDICTIONS={'valid_predictions.npz','validation_predictions.npz','valid_predictions.csv'}


def make_bundle(output,root='.',max_file_mb=128):
    root=Path(root).resolve();output=Path(output).resolve()
    output.parent.mkdir(parents=True,exist_ok=True);selected=[]
    for directory in ('artifacts/cluster','artifacts','reports'):
        base=root/directory
        if not base.is_dir():continue
        for path in base.rglob('*'):
            if not path.is_file() or path.is_symlink() or path==output:continue
            rel=path.relative_to(root);parts=rel.parts
            # Base-model metadata and raw datasets are not analysis inputs.
            if any(p.startswith(('qwen3_','hubert_large','ecgfounder')) for p in parts):continue
            allowed=path.name in NAMES or path.name in PREDICTIONS
            allowed |= path.suffix=='.json' and ('reports' in parts or path.name.endswith('_valid.json'))
            if allowed:
                if path.stat().st_size>max_file_mb*2**20:raise ValueError(f'Analysis file exceeds size limit: {rel}')
                selected.append(path)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(selected)):z.write(path,path.relative_to(root).as_posix())
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--root',default='.');p.add_argument('--max-file-mb',type=int,default=128)
    print(make_bundle(**vars(p.parse_args())))
