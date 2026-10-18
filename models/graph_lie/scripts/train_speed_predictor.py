"""Train the speed predictor s_hat(x): G(x) -> warped translational clearance speed S_d(x).

Supervision: every endpoint of the scene's training pairs (2 x 800k poses), target log S_d with S_d
warped exactly as the trainers do (model_train_lie.warp_speeds), MSE in log space; a 5 % held-out
split (by pair) reports log-space RMSE, the fraction within a factor 2, and the per-bin median
ratio s_hat / S_d.

    python models/graph_lie/scripts/train_speed_predictor.py --dataPath datasets/3dshape/data/Tshape3d_env1 \
        --out Experiments/graph_lie/speed_pred/Tshape3d_env1.pt --device cuda:0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie import model_train_lie as mtl  # noqa: E402
from models.graph_native import _sib  # noqa: E402
from models.graph_native.model_train_metric import load_dataset  # noqa: E402

sf = __import__(_sib.BASE + '.speed_fourier', fromlist=['speed_fourier'])
EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]


def endpoints(rows):
    """(N, 12) pair rows -> (2N, 12) G(x), (2N,) warped S_d."""
    sp = mtl.warp_speeds(rows)['Sd']
    x = torch.cat([rows[:, :6], rows[:, 6:12]], 0)
    g = _sib.se3.features(x[:, :3], _sib.se3.so3_exp(2.0 * math.pi * x[:, 3:6]))
    return g, torch.cat([sp[:, 0], sp[:, 1]], 0)


def evaluate(model, g, s, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, g.shape[0], 65536):
            out.append(model(g[i:i + 65536].to(device)).cpu())
    model.train()
    p = torch.cat(out).numpy()
    t = s.numpy()
    lr = np.log(p / t)
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (t >= lo) & (t < hi)
        if m.sum() > 20:
            bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(m.sum()), 'median_ratio': float(np.exp(np.median(lr[m]))),
                         'mean_abs_log10': float(np.mean(np.abs(lr[m])) / math.log(10))})
    return {'rmse_log': float(np.sqrt(np.mean(lr ** 2))), 'within_x2': float(np.mean(np.abs(lr) < math.log(2))),
            'mean_abs_log10': float(np.mean(np.abs(lr)) / math.log(10)), 'bins': bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_freq', type=int, default=128)
    ap.add_argument('--sigma_t', type=float, default=1.0)
    ap.add_argument('--sigma_r', type=float, default=0.3)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--dense', nargs='*', default=[], help='clearance_oracle.py npz files: training poses come from '
                    'these (dense geometric supervision); the dataset endpoints are then validation only')
    ap.add_argument('--band_weight', type=float, default=1.0, help='loss weight of poses with S_d < 1 (dense mode)')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rows = load_dataset(a.dataPath)
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(rows.shape[0], generator=gen)
    n_val = rows.shape[0] // 20
    g_val, s_val = endpoints(rows[perm[:n_val]])
    if a.dense:
        cfg, sd = [], []
        for f in a.dense:
            z = np.load(f)
            cfg.append(z['configs']), sd.append(z['speed_dists'])
        cfg, sd = np.concatenate(cfg), np.concatenate(sd)
        x = torch.from_numpy(cfg)
        g_tr = _sib.se3.features(x[:, :3], _sib.se3.so3_exp(2.0 * math.pi * x[:, 3:6]))
        sd = torch.from_numpy(sd)
        s_tr = torch.clamp(sd ** 2 * (2 - sd) ** 2, min=0.001)  # the trainers' warp
        g_val, s_val = endpoints(rows[perm])  # every dataset endpoint is held out
        print('dense supervision: {} poses, {:.1f} % in the band; validation on {} dataset endpoints'.format(
            g_tr.shape[0], 100.0 * (sd < 1).float().mean(), g_val.shape[0]), flush=True)
    else:
        g_tr, s_tr = endpoints(rows[perm[n_val:]])
    y_tr = torch.log(s_tr)
    w_tr = torch.where(s_tr < 0.999, torch.full_like(y_tr, a.band_weight), torch.ones_like(y_tr))
    kw = dict(n_freq=a.n_freq, sigma_t=a.sigma_t, sigma_r=a.sigma_r, width=a.width, depth=a.depth)
    model = sf.SpeedPredictor(**kw).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = a.epochs * (g_tr.shape[0] // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g_tr, y_tr, w_tr = g_tr.to(a.device), y_tr.to(a.device), w_tr.to(a.device)
    t0 = time.time()
    hist = []
    for ep in range(1, a.epochs + 1):
        idx = torch.randperm(g_tr.shape[0], device=a.device)
        tot, nb = 0.0, 0
        for i in range(0, g_tr.shape[0] - a.batch + 1, a.batch):
            b = idx[i:i + a.batch]
            loss = (w_tr[b] * (model.log_speed(g_tr[b]) - y_tr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        ev = evaluate(model, g_val, s_val, a.device)
        hist.append({'epoch': ep, 'train_mse_log': tot / nb, **{k: ev[k] for k in ('rmse_log', 'within_x2')}})
        print('epoch {} train mse(log) {:.4f} val rmse(log) {:.4f} within x2 {:.3f} {:.0f}s'.format(
            ep, tot / nb, ev['rmse_log'], ev['within_x2'], time.time() - t0), flush=True)
    ev = evaluate(model, g_val, s_val, a.device)
    for b in ev['bins']:
        print('  S_d [{:.3f},{:.3f}) n={:6d} median s_hat/S_d {:.2f} mean|log10| {:.3f}'.format(
            b['lo'], b['hi'], b['n'], b['median_ratio'], b['mean_abs_log10']))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'kw': kw,
                'dataPath': a.dataPath, 'val': ev, 'history': hist, 'args': vars(a)}, a.out)
    with open(a.out.replace('.pt', '.json'), 'w') as fh:
        json.dump({'kw': kw, 'val': ev, 'history': hist, 'args': vars(a), 'wall_s': time.time() - t0}, fh, indent=1)


if __name__ == '__main__':
    main()
