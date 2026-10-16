#!/usr/bin/env python
"""Branch diagnostics of trained min-of-branches checkpoints (spec section 7), for the multi-branch heads
(graph_mb, graph_mb_free) and the previous campaign's branch heads (graph_branch, graph_branch_conf):

  usage          fraction of pairs where each branch (route x unit) is the hard minimum; surviving units (> 1 %)
  gaps           quantiles of T_(2) - T_(1); fraction of pairs within eta (near ties) and within the band Delta
  fake cuts      among near ties, the fraction with gradient cosine > gamma (duplicates) vs real ridges (< gamma)
  cut residual   mean Eikonal residual (sqrt(S |c|) - 1)^2 of the winner near ties vs away from ties, and of the
                 losing branch in its band (a misplaced cut shows as a large losing-branch residual near the cut)

Pairs and speeds come from the training set of the checkpoint's scene (fixed random subset).

    python models/graph_mb/scripts/mb_diag.py --out results/graph_mb/diag/T_mb4_s0.json Experiments/graph_mb/T_mb4_s0/latest.pt
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')
sys.path.append('/workspace')

import numpy as np
import torch

from models.graph_lie import model_function_lie as mfl
from models.graph_lie import model_train_lie as mtl
from models.graph_native import _sib
from models.graph_native.model_train_metric import load_dataset

TWO_PI = 2.0 * math.pi
SCENE = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}


def branch_api(head):
    """all_values / value_branch for the previous campaign's BranchNN as well."""
    if hasattr(head, 'all_values'):
        return head.all_values, head.value_branch, head.n_branches, head.unit_of_branch
    K = head.K

    def all_values(gx, gy):
        with torch.no_grad():
            return head.branch_values(gx, gy).reshape(gx.shape[0], -1)

    def value_branch(gx, gy, b):
        return head.branch_values(gx, gy).reshape(gx.shape[0], -1).gather(1, b[:, None])[:, 0]
    return all_values, value_branch, 2 * K, (lambda b: b % K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--delta', type=float, default=0.03)
    ap.add_argument('--eta', type=float, default=0.006)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    name = os.path.basename(os.path.dirname(a.ckpt))
    scene = SCENE[name[0]]
    m = mtl.Model('/tmp', 'datasets/3dshape/' + scene, 6, [0] * 6, device=a.device)
    m.load(a.ckpt)
    net, head = m.network, m.network.net
    all_values, value_branch, nb, unit_of = branch_api(head)
    d = load_dataset('datasets/3dshape/' + scene)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(d.shape[0], generator=g)[:a.n]
    rows = d[idx].to(a.device)
    pts = rows[:, :12]
    speeds = mtl.warp_speeds(rows, 6, 1.0)
    out = {'ckpt': a.ckpt, 'scene': scene, 'n': a.n, 'n_branches': nb}
    with torch.no_grad():
        g0, g1 = net.features(pts[:, :6]), net.features(pts[:, 6:])
        V = all_values(g0, g1)
        srt, order = V.sort(1)
        Tm, win, second = srt[:, 0], order[:, 0], order[:, 1]
        gap = srt[:, 1] - srt[:, 0]
        gap = torch.where(torch.isfinite(gap), gap, torch.full_like(gap, 1e9))
        usage = torch.bincount(win, minlength=nb).float() / a.n
        out['usage_branch'] = [round(float(x), 4) for x in usage]
        units = unit_of(torch.arange(nb, device=a.device))
        nu = int(units.max()) + 1
        uu = torch.zeros(nu, device=a.device).index_add_(0, units, usage)
        out['usage_unit'] = [round(float(x), 4) for x in uu]
        out['surviving_units'] = int((uu > 0.01).sum())
        out['alive'] = [int(x) for x in head.alive] if hasattr(head, 'alive') else None
        out['T_median'] = float(Tm.median())
        out['gap_quantiles_1_10_50'] = [float(torch.quantile(gap.clamp(max=10), q)) for q in (0.01, 0.1, 0.5)]
        out['frac_near_eta'] = float((gap < a.eta).float().mean())
        out['frac_in_band_2plus'] = float(((V <= (Tm + a.delta)[:, None]).sum(1) >= 2).float().mean())
        out['rows_per_pair'] = float((V <= (Tm + a.delta)[:, None]).sum(1).float().mean())
    # gradients of the winner and the second branch
    def grads(b):
        gx = g0.detach().requires_grad_(True)
        gy = g1.detach().requires_grad_(True)
        T = value_branch(gx, gy, b)
        dgx, dgy = torch.autograd.grad(T.sum(), (gx, gy))
        cx = _sib.tangents.project_feature_gradient(dgx, gx.detach()[:, 3:].reshape(-1, 3, 3), 1.0, TWO_PI)
        cy = _sib.tangents.project_feature_gradient(dgy, gy.detach()[:, 3:].reshape(-1, 3, 3), 1.0, TWO_PI)
        return T.detach(), cx.detach(), cy.detach()
    T1, cx1, cy1 = grads(win)
    T2, cx2, cy2 = grads(second)

    def resid(cx, cy):
        N0 = mfl.eikonal_N(cx, speeds, 0, 'iso')
        N1 = mfl.eikonal_N(cy, speeds, 1, 'iso')
        return (torch.sqrt(N0) - 1) ** 2 + (torch.sqrt(N1) - 1) ** 2

    def cosine(u, v):
        return (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1) + 1e-12)
    near = gap < a.eta
    band2 = gap < a.delta
    cos = 0.5 * (cosine(cx1, cx2) + cosine(cy1, cy2))
    r1, r2 = resid(cx1, cy1), resid(cx2, cy2)
    out['near_ties'] = int(near.sum())
    if near.any():
        out['near_tie_dup_frac'] = float((cos[near] > a.gamma).float().mean())
        out['near_tie_cos_median'] = float(cos[near].median())
        out['winner_resid_near_ties'] = float(r1[near].mean())
        out['loser_resid_near_ties'] = float(r2[near].mean())
    out['winner_resid_all'] = float(r1.mean())
    if band2.any():
        out['loser_resid_in_band'] = float(r2[band2].mean())
        out['band_dup_frac'] = float((cos[band2] > a.gamma).float().mean())
    # real ridges: near ties with distinct gradients
    out['real_ridge_frac'] = float((near & (cos < a.gamma)).float().mean())
    if hasattr(head, 'K'):
        out['route_switch_frac'] = float(((win // head.K) != (second // head.K))[near].float().mean()) if near.any() else None
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
