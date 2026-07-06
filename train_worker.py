import os
import sys
import json
import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--shared-state', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--leads', choices=['all', 'lead1'], required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.chdir(args.project_dir)
    sys.path.insert(0, args.project_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import ecg_pipeline as ep
    ep.apply_compat_patches()

    with open(args.config) as f:
        config = json.load(f)

    shared = np.load(args.shared_state, allow_pickle=True)
    data_df = shared['data_df'].item()
    top_feat_names = list(shared['top_feat_names'])
    feat_means = shared['feat_means']
    feat_stds = shared['feat_stds']
    weights_matrix = shared['weights_matrix']
    tst_fold = int(shared['tst_fold'])
    # Предвычисленный кэш фич (feats/<Dataset>/all_feats_ch{N}.zip), если он был передан
    # из ноутбука; без него воркер, как и раньше, будет считать фичи на лету.
    feats_lookup = shared['feats_lookup'].item() if 'feats_lookup' in shared else None

    trn_df, val_df, tst_df, _, _ = ep.build_splits(data_df, tst_fold)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    window = config['WINDOW_SECONDS'] * ep.FS

    result, _ = ep.run_pipeline(
        leads=args.leads, tag=args.tag,
        trn_df=trn_df, val_df=val_df, tst_df=tst_df,
        top_feat_names=top_feat_names, feat_means=feat_means, feat_stds=feat_stds,
        cache_dir=args.cache_dir, window=window, weights_matrix=weights_matrix,
        config=config, device=device, checkpoint_path=None,
        feats_lookup=feats_lookup,
    )

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
