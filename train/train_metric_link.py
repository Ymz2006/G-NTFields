"""Train the original metric NTFields (the bundle /media/4tbssd/jeffrey/export_metric, mounted at /export) on a
stretchable-link cell (models/drone_link): train_metric_baseline.py with the dimension of the cell, 7 =
(drone a's position, rotvec / 2 pi, link length). Trainer, network, loss, speed warp and every hyperparameter are the
bundle's; the network reads the 7-d chart through its Fourier features (B: 128 x 7) and the Eikonal residual is the
chart gradient over all 7 coordinates.

    python /workspace/train/train_metric_link.py --dataPath /workspace/datasets/3dshape/data/link_forest \
        --name forest_ntf_s0 --seed 0 --epochs 8000 --td-weight 1e-3 --normal-weight 1e-3 --bundle /export \
        --out_root /workspace/Experiments/drone_link

The current directory becomes the bundle root (its trainer copies ./models/metric into the run folder). Only the
epoch-4000 and final checkpoints are kept (epoch4000.pt, latest.pt); train_summary.json is written. Run with
NTF_NO_PLOT=1 (the trainer's 2-d plots assume the SE(2) / SE(3) layout).
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
    ap.add_argument('--out_root', default='/workspace/Experiments/drone_link')
    ap.add_argument('--bundle', default='/export')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8000)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--td-weight', type=float, default=0.0)
    ap.add_argument('--normal-weight', type=float, default=0.0)
    ap.add_argument('--keep', default='4000')
    a = ap.parse_args()

    data = os.path.abspath(a.dataPath).rstrip('/')
    meta = json.load(open(os.path.join(data, 'meta.json')))
    dim = int(meta.get('dim', 6))
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
    m = md.Model(os.path.abspath(a.out_root), data, dim, [0.0] * dim, device=a.device)
    m.folder = folder
    m.Params['Training']['Number of Epochs'] = a.epochs
    m.Params['Training']['Alpha'] = a.alpha
    m.Params['Training']['Seed'] = a.seed
    m.Params['Training']['TD weight'] = a.td_weight
    m.Params['Training']['Normal weight'] = a.normal_weight
    print('data   : {}\noutput : {}\ndim {}  seed {}  epochs {}  alpha {}  td {}  normal {}'.format(
        data, folder, dim, a.seed, a.epochs, a.alpha, a.td_weight, a.normal_weight), flush=True)
    t0 = time.time()
    m.train()
    wall = time.time() - t0
    print('Training time: {:.1f}s'.format(wall), flush=True)

    keep = {int(k) for k in a.keep.split(',') if k}
    final = None
    for c in sorted(glob.glob(os.path.join(folder, 'Model_Epoch_*.pt'))):
        ep = int(re.search(r'Model_Epoch_(\d+)_', os.path.basename(c)).group(1))
        if ep == a.epochs:
            final = c
            shutil.copy2(c, os.path.join(folder, 'latest.pt'))
        elif ep in keep:
            shutil.copy2(c, os.path.join(folder, 'epoch{}.pt'.format(ep)))
        os.remove(c)
    for f in os.listdir(folder):
        if f.startswith(('plots', 'tauplots')) and f.endswith(('.jpg', '.obj')):
            os.remove(os.path.join(folder, f))
    loss = [float(x) for x in m.total_train_loss] if hasattr(m, 'total_train_loss') else []
    with open(os.path.join(folder, 'train_summary.json'), 'w') as fh:
        json.dump({'wall_time_s': wall, 'epochs': a.epochs, 'seed': a.seed, 'alpha': a.alpha, 'dim': dim,
                   'td_weight': a.td_weight, 'normal_weight': a.normal_weight, 'dataPath': data,
                   'final_checkpoint': os.path.basename(final) if final else None,
                   'final_loss': loss[-1] if loss else None, 'loss_every_100': loss[::100],
                   'kept_epochs': sorted(keep)}, fh, indent=1)


if __name__ == '__main__':
    main()
