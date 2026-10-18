"""How the conformal branch head uses its terms (graph_branch_conf checkpoints).

* cone scale vs speed: at training endpoints p (stored clearance speed S_d(p)), the learned cone factor 1 + rho(p)
  against the Eikonal limit 1 / S_d(p): Spearman correlation of rho with 1 / S_d - 1, median ratio per speed decade;
* decomposition of tau on random pairs into T_free + C (conformal / escape term) + delta (selected residual), as
  shares of tau, binned by T_free (near the goal C ~ rho T_free, far away C ~ rho_bar lam_bar).

    python models/graph_branch/scripts/conf_diag.py --scene Tshape3d_env1 Experiments/graph_branch/T_bc4_s0/latest.pt ...
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie.model_train_lie import warp_speeds  # noqa: E402
from models.graph_method_triangle import branch_decoder as bd  # noqa: E402
from models.graph_native import _sib  # noqa: E402
from models.graph_native import model_train_metric as gn_train  # noqa: E402

TWO_PI = 2.0 * math.pi


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='Tshape3d_env1')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='results/graph_branch/summary/conf_diag.json')
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    dev = a.device
    base = os.path.join('datasets/3dshape/data', a.scene)
    cols = [np.array(np.load(os.path.join(base, f + '.npy'), mmap_mode='r')[:200000:5])
            for f in ('sampled_points', 'speed', 'normal', 'speed_dists', 'speed_angles')]
    d = torch.from_numpy(np.concatenate(cols, 1)).float()
    Sd = warp_speeds(d)['Sd']
    X = d[:, :12].to(dev)
    out = {}
    for ck in a.ckpts:
        m = gn_train.Model('/tmp/cd', base, 6, [0.0] * 6, device=dev)
        m.load(ck)
        net = m.network.eval()
        h = net.net
        g0 = net.features(X[:, :6])
        g1 = net.features(X[:, 6:])
        e0 = h.enc(g0)
        rho = torch.nn.functional.softplus(h.rho_head(e0)[:, 0]).cpu()
        lam = (h.lam_min + torch.nn.functional.softplus(h.lam_head(e0)[:, 0])).cpu()
        S = Sd[:, 0]
        res = {'spearman_rho_vs_inv_speed': spearman(rho, 1.0 / S - 1.0),
               'rho_median': float(rho.median()), 'lam_median': float(lam.median())}
        edges = [0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.01]
        res['cone_factor_times_S_by_speed'] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (S >= lo) & (S < hi)
            if sel.sum() > 20:
                res['cone_factor_times_S_by_speed']['{:.2f}-{:.2f}'.format(lo, hi)] = \
                    [float(((1 + rho[sel]) * S[sel]).median()), int(sel.sum())]
        # decomposition on random pairs (start of row i, goal of row j)
        perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0)).to(dev)
        gy = g1[perm]
        ey = h.enc(gy)
        qx = bd.quat_from_matrix(g0[:, 3:].reshape(-1, 3, 3))
        qy = bd.quat_from_matrix(gy[:, 3:].reshape(-1, 3, 3))
        tp, tm = bd.free_routes(g0[:, :3], qx, gy[:, :3], qy, h.s_t, h.s_r)
        BV = h.branch_values(g0, gy)                       # (n, 2, K) with C included
        n, _, K = BV.shape
        idx = BV.reshape(n, -1).argmin(1)
        sig = idx // K
        Tf = torch.where(sig == 0, tp, tm)
        rb = 0.5 * (rho.to(dev) + torch.nn.functional.softplus(h.rho_head(ey)[:, 0]))
        lb = 0.5 * (lam.to(dev) + h.lam_min + torch.nn.functional.softplus(h.lam_head(ey)[:, 0]))
        C = rb * lb * torch.tanh(Tf / lb)
        T = BV.reshape(n, -1).min(1).values
        delta = T - Tf - C
        bins = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 2.0]
        res['share_by_Tfree'] = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (Tf >= lo) & (Tf < hi)
            if sel.sum() > 20:
                res['share_by_Tfree']['{:.2f}-{:.2f}'.format(lo, hi)] = {
                    'n': int(sel.sum()), 'T_over_Tfree': float((T[sel] / Tf[sel]).median()),
                    'share_Tfree': float((Tf[sel] / T[sel]).median()), 'share_C': float((C[sel] / T[sel]).median()),
                    'share_delta': float((delta[sel] / T[sel]).median())}
        out[ck] = res
        print(ck, json.dumps(res, indent=1))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
