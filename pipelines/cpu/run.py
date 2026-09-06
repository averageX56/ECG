"""Run from the repository root: python -m pipelines.cpu.run --help."""
import argparse
import os
import sys


def main():
    # Set before importing torch or any project training module.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    worker_parser=argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument('--workers',type=int,default=min(4,max(1,(os.cpu_count() or 2)-1)),help='CPU preparation processes')
    worker_args,remaining=worker_parser.parse_known_args()
    if worker_args.workers<1:worker_parser.error('--workers must be positive')
    os.environ['ECG_CPU_WORKERS']=str(worker_args.workers)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    parser = argparse.ArgumentParser(description='Local CPU pipeline stages',epilog='Use --workers N before or after a stage (default: up to 4).')
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('audit', 'prepare-beats', 'prepare-records', 'train-beats', 'train-records', 'run'):
        sub.add_parser(name, add_help=False)
    p = sub.add_parser('prepare-hubert')
    p.add_argument('--output', default='artifacts/hubert_inputs_full')
    p.add_argument('--pilot', action='store_true')
    p = sub.add_parser('prepare-founder')
    p.add_argument('--output', default='artifacts/founder_full_inputs')
    p = sub.add_parser('prepare-unlabeled')
    p.add_argument('--output', default='artifacts/unlabeled_beats')
    p.add_argument('--checkpoint', default='artifacts/delineator_qt.pt')
    p = sub.add_parser('prepare-qwen')
    p.add_argument('--root', default='LUDB')
    p.add_argument('--output', default='artifacts/qwen_delineation_inputs')
    sub.add_parser('report')
    args, rest = parser.parse_known_args(remaining)
    if args.stage in ('train-beats','train-records'):
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name]=str(worker_args.workers)
    if args.stage in ('prepare-hubert', 'prepare-founder', 'prepare-unlabeled', 'prepare-qwen', 'report') and rest:
        parser.error('Unexpected arguments: ' + ' '.join(rest))
    if args.stage == 'prepare-hubert':
        from ecg_project.training.hubert_lora import prepare_inputs
        prepare_inputs(args.output, full=not args.pilot)
    elif args.stage == 'prepare-founder':
        from ecg_project.experiments.founder_experiments import prepare_full_ptb
        prepare_full_ptb(args.output)
    elif args.stage == 'prepare-unlabeled':
        from ecg_project.training.unlabeled_beats import prepare
        prepare(output=args.output,checkpoint=args.checkpoint,device='cpu')
    elif args.stage == 'prepare-qwen':
        from ecg_project.training.qwen_delineation import prepare
        prepare(root=args.root,output=args.output)
    elif args.stage == 'report':
        import runpy
        runpy.run_path('scripts/make_report.py', run_name='__main__')
    else:
        if args.stage in ('prepare-beats', 'prepare-records', 'run'):
            rest += ['--device', 'cpu']
        sys.argv = ['ecg_project', args.stage, *rest]
        from ecg_project.__main__ import main as cli
        cli()


if __name__ == '__main__':
    main()
