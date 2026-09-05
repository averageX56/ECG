import argparse

def main():
    parser=argparse.ArgumentParser(description='ECG: dataset audit, delineation, beat and multilabel classification')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit');p.add_argument('--root',default='.');p.add_argument('--output',default='artifacts/catalog.csv')
    p=sub.add_parser('benchmark-dwt');p.add_argument('--root',default='LUDB');p.add_argument('--split',choices=['train','valid','test'],default='valid');p.add_argument('--limit',type=int,default=0);p.add_argument('--output',default='reports/dwt_valid_full.json')
    p=sub.add_parser('train-delineator');p.add_argument('--root',default='LUDB');p.add_argument('--epochs',type=int,default=30);p.add_argument('--minutes',type=float,default=25);p.add_argument('--output',default='artifacts/delineator.pt');p.add_argument('--device',default='auto')
    p=sub.add_parser('evaluate-delineator');p.add_argument('--root',default='LUDB');p.add_argument('--split',choices=['valid','test'],default='test');p.add_argument('--output',default='reports/segmentation_test.json');p.add_argument('--checkpoint',default='artifacts/delineator.pt')
    p=sub.add_parser('prepare-records');p.add_argument('--catalog',default='artifacts/catalog.csv');p.add_argument('--output',default='artifacts/record_features');p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',default='artifacts/delineator.pt')
    p=sub.add_parser('train-records');p.add_argument('--manifest',default='artifacts/record_features/manifest.csv');p.add_argument('--output',default='artifacts/record_models');p.add_argument('--minutes',type=float,default=45)
    p=sub.add_parser('prepare-beats');p.add_argument('--root',default='data/mit-bih');p.add_argument('--output',default='artifacts/beat_features');p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',default='artifacts/delineator.pt')
    p=sub.add_parser('train-beats');p.add_argument('--root',default='artifacts/beat_features');p.add_argument('--output',default='artifacts/beat_models');p.add_argument('--minutes',type=float,default=40)
    p=sub.add_parser('adapt-delineator');p.add_argument('--epochs',type=int,default=12);p.add_argument('--minutes',type=float,default=20);p.add_argument('--unlabeled-records',type=int,default=200);p.add_argument('--output',default='artifacts/delineator_transfer.pt')
    p=sub.add_parser('train-qt');p.add_argument('--root',default='data/qtdb_external');p.add_argument('--epochs',type=int,default=10);p.add_argument('--minutes',type=float,default=15);p.add_argument('--output',default='artifacts/delineator_qt.pt')
    p=sub.add_parser('evaluate-qt');p.add_argument('--root',default='data/qtdb_external');p.add_argument('--checkpoint',default='artifacts/delineator.pt');p.add_argument('--output',default='reports/qtdb_baseline.json')
    p=sub.add_parser('batch');p.add_argument('manifest');p.add_argument('--output',default='reports/batch');p.add_argument('--checkpoint')
    p=sub.add_parser('evaluate-vt');p.add_argument('--root',default='artifacts/beat_features_qt');p.add_argument('--output',default='reports/vt_episodes.json')
    p=sub.add_parser('train-beat-cnn');p.add_argument('--root',default='artifacts/beat_features_qt');p.add_argument('--epochs',type=int,default=25);p.add_argument('--minutes',type=float,default=20);p.add_argument('--output',default='artifacts/beat_models')
    p=sub.add_parser('analyze');p.add_argument('path');p.add_argument('--output',default='reports/example');p.add_argument('--checkpoint');p.add_argument('--csv-fs',type=float);p.add_argument('--lead');p.add_argument('--start-seconds',type=float,default=0);p.add_argument('--duration-seconds',type=float);p.add_argument('--device',default='cpu')
    p.add_argument('--beat-model-path')
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='audit':from ecg_project.data.catalog import audit as run
    elif command=='benchmark-dwt':from ecg_project.evaluation.benchmark import benchmark_ludb as run
    elif command=='train-delineator':from ecg_project.models.segmentation import train as run
    elif command=='evaluate-delineator':from ecg_project.models.segmentation import evaluate as run
    elif command=='prepare-records':from ecg_project.training.record_model import prepare as run
    elif command=='train-records':from ecg_project.training.record_model import train as run
    elif command=='prepare-beats':from ecg_project.training.beats import prepare as run
    elif command=='train-beats':from ecg_project.training.beats import train as run
    elif command=='adapt-delineator':from ecg_project.training.adaptation import train as run
    elif command=='train-qt':from ecg_project.training.qtdb import train as run
    elif command=='evaluate-qt':from ecg_project.training.qtdb import evaluate as run
    elif command=='batch':from ecg_project.workflows.analysis import batch as run
    elif command=='evaluate-vt':from ecg_project.evaluation.episodes import evaluate as run
    elif command=='train-beat-cnn':from ecg_project.models.beat_cnn import train as run
    else:from ecg_project.workflows.analysis import analyze as run
    run(**args)

if __name__=='__main__':main()
