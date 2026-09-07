"""Shared configurable CPU preparation commands, dispatched by run.py."""
import argparse
from pathlib import Path
from ecg_project.data.cache import CANONICAL_DELINEATOR,atomic_json

STAGES={'prepare-qwen','prepare-hubert','prepare-founder','prepare-beats','prepare-unlabeled','prepare-records','prepare-record-datasets','prepare-qwen-pseudo'}


def main(argv):
    p=argparse.ArgumentParser(description='CPU preprocessing v2; use --workers with the parent launcher')
    p.add_argument('stage',choices=sorted(STAGES));p.add_argument('--output');p.add_argument('--sources',nargs='+')
    p.add_argument('--catalog',default='artifacts/catalog.csv');p.add_argument('--checkpoint',default=CANONICAL_DELINEATOR)
    p.add_argument('--datasets',nargs='+',choices=['LUDB','QTDB'],default=['LUDB','QTDB'])
    p.add_argument('--validation-datasets',nargs='+',choices=['LUDB','QTDB'])
    p.add_argument('--root',default='LUDB');p.add_argument('--qt-root',default='data/qtdb_external')
    p.add_argument('--unlabeled-sources',nargs='*',default=[]);p.add_argument('--train-windows',type=int,default=1)
    p.add_argument('--tau',type=float,default=.95);p.add_argument('--limit',type=int,default=0)
    args=p.parse_args(argv)
    if args.stage=='prepare-qwen':
        from ecg_project.data.delineation_v2 import prepare
        return prepare(output=args.output or 'artifacts/qwen_delineation_inputs_v2',datasets=args.datasets,ludb_root=args.root,
            qt_root=args.qt_root,catalog=args.catalog,unlabeled_sources=args.unlabeled_sources,validation_datasets=args.validation_datasets)
    if args.stage in ('prepare-hubert','prepare-founder'):
        from ecg_project.data.record_inputs import prepare
        return prepare(kind=args.stage.removeprefix('prepare-'),output=args.output,catalog=args.catalog,sources=args.sources,train_windows=args.train_windows)
    if args.stage in ('prepare-beats','prepare-unlabeled'):
        from ecg_project.data.beat_datasets import prepare
        ssl=args.stage=='prepare-unlabeled'
        return prepare(output=args.output or ('artifacts/unlabeled_beats_v2' if ssl else 'artifacts/beat_features_v2'),
            sources=args.sources or (['CPSC_EXTRA'] if ssl else ['MIT']),checkpoint=args.checkpoint,catalog=args.catalog,unlabeled=ssl)
    if args.stage=='prepare-qwen-pseudo':
        from ecg_project.data.qwen_pseudo import prepare
        return prepare(output=args.output or 'artifacts/qwen_pseudo_inputs',sources=args.sources or ['CPSC_EXTRA'],
            checkpoint=args.checkpoint,catalog=args.catalog,tau=args.tau)
    if args.stage=='prepare-records':
        from ecg_project.training.record_model import prepare
        return prepare(catalog=args.catalog,output=args.output or 'artifacts/record_features_v2',limit=args.limit,checkpoint=args.checkpoint,device='cpu',sources=args.sources)
    from ecg_project.data.policy import build_record_manifest,POLICY_VERSION
    from ecg_project.data.catalog import file_hash,TARGETS
    frame=build_record_manifest(args.catalog,args.sources);output=Path(args.output or 'artifacts/record_dataset_manifest.csv')
    output.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(output,index=False)
    atomic_json(output.with_suffix('.provenance.json'),dict(policy=POLICY_VERSION,catalog_sha256=file_hash(args.catalog),manifest_sha256=file_hash(output),targets=TARGETS))
    return output
