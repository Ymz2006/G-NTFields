"""Train the original metric NTFields baseline (the self-contained bundle /home/jeffrey/export_metric, i.e.
Lie-Shape-NTFields' metric_baseline: models/metric with alpha = 1.0) with a fixed seed and a named run folder.

    python train/train_metric_baseline.py --dataPath /workspace/datasets/3dshape/data/Tshape3d_env1 --name T_mb_s0 \
        --seed 0 --epochs 8000 --bundle /export --out_root /workspace/Experiments/metric_baseline

The bundle's trainer is used unchanged (AdamW wd 0.5, lr 5e-4 in the loop, 5 batches of 2000 per epoch, accept /
reject rollback, checkpoint every 100 epochs). The current directory must be the bundle root (its trainer copies
./models/metric into the run folder). Afterwards only the epoch-4000 and final checkpoints are kept
(epoch4000.pt, latest.pt) and train_summary.json is written.
"""
import argparse
import glob
import json
import os
import pickle
import random
import re
import shutil
import sys
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out_root', default='/workspace/Experiments/metric_baseline')
    ap.add_argument('--bundle', default='/export')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8000)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--td-weight', type=float, default=0.0,
                    help='TD loss weight; the original metric NTFields (G-Shape-NTFields-old models/metric) uses 1e-3')
    ap.add_argument('--normal-weight', type=float, default=0.0, help='normal-alignment loss weight; original 1e-3')
    ap.add_argument('--keep', default='4000', help='comma list of intermediate epochs whose checkpoints are kept')
    a = ap.parse_args()

    os.chdir(a.bundle)
    sys.path.insert(0, a.bundle)
    sys.modules.setdefault('pickle5', pickle)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    from models.metric import model_train_metric as md

    folder = os.path.join(os.path.abspath(a.out_root), a.name)
    os.makedirs(folder, exist_ok=True)
    m = md.Model(os.path.abspath(a.out_root), a.dataPath.rstrip('/'), 6, [0.0] * 6, device=a.device)
    m.folder = folder
    m.Params['Training']['Number of Epochs'] = a.epochs
    m.Params['Training']['Alpha'] = a.alpha
    m.Params['Training']['Seed'] = a.seed
    m.Params['Training']['TD weight'] = a.td_weight
    m.Params['Training']['Normal weight'] = a.normal_weight
    print('data   : {}\noutput : {}\nseed   : {}  epochs {}  alpha {}  td {}  normal {}'.format(
        a.dataPath, folder, a.seed, a.epochs, a.alpha, a.td_weight, a.normal_weight), flush=True)
    t0 = time.time()
    m.train()
    wall = time.time() - t0
    print('Training time: {:.1f}s'.format(wall), flush=True)

    keep = {int(k) for k in a.keep.split(',') if k}
    ckpts = sorted(glob.glob(os.path.join(folder, 'Model_Epoch_*.pt')))
    final = None
    for c in ckpts:
        ep = int(re.search(r'Model_Epoch_(\d+)_', os.path.basename(c)).group(1))
        if ep == a.epochs:
            final = c
            shutil.copy2(c, os.path.join(folder, 'latest.pt'))
        elif ep in keep:
            shutil.copy2(c, os.path.join(folder, 'epoch{}.pt'.format(ep)))
        os.remove(c)
    for f in os.listdir(folder):                      # the trainer's per-100-epoch plots (100 MB per run)
        if f.startswith(('plots', 'tauplots')) and f.endswith(('.jpg', '.obj')):
            os.remove(os.path.join(folder, f))
    loss = [float(x) for x in m.total_train_loss] if hasattr(m, 'total_train_loss') else []
    with open(os.path.join(folder, 'train_summary.json'), 'w') as fh:
        json.dump({'wall_time_s': wall, 'epochs': a.epochs, 'seed': a.seed, 'alpha': a.alpha, 'td_weight': a.td_weight,
                   'normal_weight': a.normal_weight, 'dataPath': a.dataPath,
                   'final_checkpoint': os.path.basename(final) if final else None,
                   'final_loss': loss[-1] if loss else None, 'kept_epochs': sorted(keep)}, fh, indent=1)


if __name__ == '__main__':
    main()
