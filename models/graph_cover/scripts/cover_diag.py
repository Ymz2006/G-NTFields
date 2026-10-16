"""Diagnostics of trained native-pipeline checkpoints (graph_native format: graph_cover, graph_branch, graph_metric,
graph_tri_metric ...): triangle inequality under attack, group-element (branch) usage, free-time share, speed
calibration by clearance bin, and branch-point statistics of the cover head.

    python models/graph_cover/scripts/cover_diag.py --out results/graph_cover/diag/T_lp2_s0.json \
        --ckpt Experiments/graph_cover/T_lp2_s0/latest.pt --dataPath datasets/3dshape/Tshape3d_env1

Triangle check: x, z are start / goal poses of random dataset pairs (collision-free), y the free-space geodesic
midpoint, then Adam ascent of T(x, z) - T(x, y) - T(y, z) over body twists of y only ('y', all in-domain) and of
all three poses ('all'), in float64 (models/graph_method_triangle/triangle.py). Speed calibration: predicted
speed 1 / |c| (Lie gradient) vs the warped dataset speed S_d at both endpoints of dataset pairs, per log bin.
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')

import numpy as np
import torch

from models.graph_lie import lie, model_train_lie as mtl
from models.graph_method_triangle import triangle
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_train_metric as gnm

TWO_PI = 2 * math.pi
BINS = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0001]


def load(path, device):
    m = gnm.Model('.', '.', 6, [0.0] * 6, device=device)
    m.load(path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataPath', default='datasets/3dshape/Tshape3d_env1')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--n_speed', type=int, default=8192)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=4242)
    a = ap.parse_args()
    dev = a.device
    m = load(a.ckpt, dev)
    net = m.network
    head = net.net
    out = {'ckpt': a.ckpt, 'graph': net.graph, 'speed_model': m.Params['Training'].get('Speed model')}
    data = gnm.load_dataset(a.dataPath)
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(data.shape[0], generator=g)
    # ---------------------------------------------------------------- speed calibration + branch usage (float32)
    d = data[idx[:a.n_speed]].to(dev)
    pts = d[:, :12]
    sp = mtl.warp_speeds(d, 6)
    T, cx, cy = lie.endpoint_gradients(net, pts, 'lie', create_graph=False)
    c = torch.cat([cx, cy]).detach()
    S = torch.cat([sp['Sd'][:, 0], sp['Sd'][:, 1]])
    pred = 1.0 / c.norm(dim=1).clamp_min(1e-12)
    ratio = (pred / S)
    cal = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mk = (S >= lo) & (S < hi)
        if mk.sum() > 10:
            cal['{:g}-{:g}'.format(lo, hi)] = {'n': int(mk.sum()), 'median_pred_over_true': float(ratio[mk].median()),
                                               'median_cn': float((c.norm(dim=1) * S)[mk].median())}
    out['speed_calibration'] = cal
    out['dex_bins'] = float(np.mean([abs(math.log10(v['median_pred_over_true'])) for v in cal.values()]))
    out['T_mean'] = float(T.mean())
    with torch.no_grad():
        gx, gy = net.features(pts[:, :6]), net.features(pts[:, 6:])
        if hasattr(head, 'all_candidates'):
            cand = head.all_candidates(gx, gy)
            k = cand.argmin(1)
            usage = torch.bincount(k, minlength=cand.shape[1]).double() / k.numel()
            out['element_usage'] = usage.tolist()
            out['minus_route_frac'] = float((head.sigmas[k] < 0).double().mean())
            out['cut_flip_frac'] = float((head.signs[k][:, head.me + 4 * head.mq:] < 0).any(1).double().mean()) if head.m > 0 else 0.0
            tf = head.free_time(gx, gy)
            Tv = head.value(gx, gy)
            out['free_share_of_T2_mean'] = float((tf ** 2 / Tv ** 2).mean())
            out['T_over_Tfree_median'] = float((Tv / tf).median())
            # runner-up gap: how close the second-best element is (ridge proximity), relative to T
            srt = cand.sort(1).values
            gap = (srt[:, 1].sqrt() - srt[:, 0].sqrt()) / srt[:, 0].sqrt().clamp_min(1e-9)
            out['runner_up_gap_rel'] = {'p10': float(gap.quantile(0.1)), 'p50': float(gap.quantile(0.5)),
                                        'frac_below_0.02': float((gap < 0.02).double().mean())}
            if head.m > 0:
                ex, _ = head.embed_pair(gx, gx[:1])
                _, _, z = head.features_cover(gx, ex)
                zn = z.norm(dim=-1)                                   # (N, m)
                out['z_norm'] = {'median': zn.median(1).values.tolist() if zn.shape[0] > 0 else None,
                                 'p05': zn.quantile(0.05, dim=0).tolist(), 'p50': zn.quantile(0.5, dim=0).tolist(),
                                 'p95': zn.quantile(0.95, dim=0).tolist()}
                out['z_norm']['median'] = zn.median(0).values.tolist()
        elif hasattr(head, 'branch_values'):
            bv = head.branch_values(gx, gy)
            k = bv.reshape(bv.shape[0], -1).argmin(1)
            out['element_usage'] = (torch.bincount(k, minlength=bv.shape[1] * bv.shape[2]).double() / k.numel()).tolist()
            out['minus_route_frac'] = float(((k // bv.shape[2]) == 1).double().mean())
    # ---------------------------------------------------------------- triangle inequality (float64)
    rows = data[idx[a.n_speed:a.n_speed + 3 * a.n]].double()
    net64 = net.double()
    xs = rows[:a.n, :6].to(dev)
    zs = rows[a.n:2 * a.n, 6:12].to(dev)
    tx, Rx = xs[:, :3], se3.so3_exp(TWO_PI * xs[:, 3:6])
    tz, Rz = zs[:, :3], se3.so3_exp(TWO_PI * zs[:, 3:6])
    allt = torch.cat([data[:, :3], data[:, 6:9]]).double().to(dev)
    lo, hi = allt.amin(0), allt.amax(0)
    gx, gz = se3.features(tx, Rx), se3.features(tz, Rz)
    tri = {}
    with torch.no_grad():
        ty, Ry = triangle.midpoint(tx, Rx, tz, Rz)
        gy = se3.features(ty, Ry)
        v = net64.net.value(gx, gz) - net64.net.value(gx, gy) - net64.net.value(gy, gz)
        tri['midpoint'] = triangle.summarize(v, net64.net.value(gx, gz))
    for name, move_ends in (('attack_all', True), ('attack_y', False)):
        va, Ta, P = triangle.attack(net64.net, tx, Rx, tz, Rz, steps=a.steps, lr=0.02, move_ends=move_ends)
        tri[name] = triangle.summarize(va, Ta)
        inside = torch.ones_like(va, dtype=torch.bool)
        for t, _ in P:
            inside &= ((t >= lo) & (t <= hi)).all(-1)
        bad = va / Ta.clamp_min(1e-12) > 1e-4
        tri[name]['in_domain'] = float((inside & bad).double().sum() / max(int(bad.sum()), 1))
    out['triangle'] = tri
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ('graph', 'speed_calibration')}, indent=1))
    print('speed calibration:', json.dumps({k: round(v['median_pred_over_true'], 3) for k, v in cal.items()}))


if __name__ == '__main__':
    main()
