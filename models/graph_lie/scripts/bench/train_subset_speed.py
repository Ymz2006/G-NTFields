"""Predicted vs true speed on targeted subsets of the TRAINING pairs, chart vs Lie gradient.

For every checkpoint, on subsets of its scene's training pairs (the data the field was fitted on, so no
generalisation is involved), both endpoints pooled:

    N_lie   = own-formulation Eikonal value with the Lie gradient c     (1/N_lie   = predicted / true speed, physical)
    N_chart = own-formulation Eikonal value with the chart gradient g   (1/N_chart = predicted / true speed, chart)
    perp    = share of the rotational chart gradient orthogonal to the rotation vector, |g_w - (g_w . u) u|^2 / |g_w|^2
              (only this component is rescaled by the chart, by (theta / 2) / sin(theta / 2))
    ratio   = |c_w| / |g_w|  (the realised chart distortion of the field's own gradient at that endpoint)

Subsets (endpoint-level unless stated): theta bins of the endpoint |rotvec|; pair-level: both endpoints < 45 deg,
both >= 150 deg, start >= 150 & goal < 45, rotation-dominated pairs (|dt| < 0.05, relative rotation >= 90 deg),
translation-dominated pairs (|dt| >= 0.3, relative rotation < 20 deg); clearance bins of the formulation's speed.

    python models/graph_lie/scripts/bench/train_subset_speed.py --out results/graph_lie/bench/train_subsets <ckpt> ...
(repo root inside pytorchserver)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')
from models.graph_lie import lie, model_function_lie as mfl, model_train_lie as mtl  # noqa: E402
from models.graph_native.model_train_metric import load_dataset  # noqa: E402
from models.graph_native import _sib  # noqa: E402

TWO_PI = 2.0 * math.pi
THETA_BINS = [(0, 45), (45, 90), (90, 135), (135, 160), (160, 180.01)]
SPEED_BINS = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01)]


def scene_of(ckpt):
    cfg = os.path.join(os.path.dirname(ckpt), 'config.json')
    return json.load(open(cfg))['DataPath']


def q(v, sel):
    v = v[sel]
    if v.size < 50:
        return None
    return {'n': int(v.size), 'p25': float(np.percentile(v, 25)), 'p50': float(np.median(v)), 'p75': float(np.percentile(v, 75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--out', default='results/graph_lie/bench/train_subsets')
    ap.add_argument('--n', type=int, default=200000)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    for ck in a.ckpts:
        name = os.path.basename(os.path.dirname(ck))
        dst = os.path.join(a.out, name + '.json')
        if os.path.exists(dst):
            continue
        m = mtl.Model('/tmp/tss', scene_of(ck), 6, [0.0] * 6, device=a.device)
        m.load(ck)
        own = m.function.speed_model
        sc = m.Params['DataPath']
        if sc not in cache:
            data = load_dataset(sc)
            g = torch.Generator().manual_seed(0)
            cache[sc] = data[torch.randperm(data.shape[0], generator=g)[:a.n]]
        d_all = cache[sc]
        acc = {}
        for i in range(0, d_all.shape[0], 4096):
            dd = d_all[i:i + 4096].to(a.device)
            pts = dd[:, :12]
            sp = mtl.warp_speeds(dd)
            T, gx, gy = lie.endpoint_gradients(m.network, pts, 'chart', create_graph=False)
            _, cx, cy = lie.endpoint_gradients(m.network, pts, 'lie', create_graph=False)
            r = torch.cat([pts[:, 3:6], pts[:, 9:12]])
            gw = torch.cat([gx[:, 3:], gy[:, 3:]])
            cw = torch.cat([cx[:, 3:], cy[:, 3:]])
            u = r / r.norm(dim=1, keepdim=True).clamp_min(1e-9)
            par = (gw * u).sum(1, keepdim=True) * u
            perp = ((gw - par) ** 2).sum(1) / (gw ** 2).sum(1).clamp_min(1e-12)
            R0 = _sib.se3.so3_exp(TWO_PI * pts[:, 3:6])
            R1 = _sib.se3.so3_exp(TWO_PI * pts[:, 9:12])
            rel = _sib.se3.rotation_angle(R0.transpose(-1, -2) @ R1) * 180 / math.pi
            dt = (pts[:, 6:9] - pts[:, :3]).norm(dim=1)
            rows = {
                'theta': r.norm(dim=1) * 360.0,
                'N_lie': torch.cat([mfl.eikonal_N(cx, sp, 0, own), mfl.eikonal_N(cy, sp, 1, own)]),
                'N_chart': torch.cat([mfl.eikonal_N(gx, sp, 0, own), mfl.eikonal_N(gy, sp, 1, own)]),
                'perp': perp, 'ratio': cw.norm(dim=1) / gw.norm(dim=1).clamp_min(1e-9),
                'S': torch.cat([sp['S' if own == 'combined' else 'Sd'][:, 0], sp['S' if own == 'combined' else 'Sd'][:, 1]]),
                'is_start': torch.cat([torch.ones(len(pts)), torch.zeros(len(pts))]).to(a.device),
                'theta_other': torch.cat([pts[:, 9:12].norm(dim=1), pts[:, 3:6].norm(dim=1)]) * 360.0,
                'rel': torch.cat([rel, rel]), 'dt': torch.cat([dt, dt]), 'T': torch.cat([T, T]),
            }
            for k, v in rows.items():
                acc.setdefault(k, []).append(v.detach().cpu())
        A = {k: torch.cat(v).double().numpy() for k, v in acc.items()}
        th, tho = A['theta'], A['theta_other']
        subsets = {'all': np.ones_like(th, bool)}
        for lo, hi in THETA_BINS:
            subsets['theta_{}_{}'.format(lo, int(min(hi, 180)))] = (th >= lo) & (th < hi)
        subsets['pair_both_lt45'] = (th < 45) & (tho < 45)
        subsets['pair_both_ge150'] = (th >= 150) & (tho >= 150)
        subsets['pair_start_ge150_goal_lt45'] = (A['is_start'] == 1) & (th >= 150) & (tho < 45)
        subsets['pair_goal_ge150_start_lt45_at_start'] = (A['is_start'] == 1) & (th < 45) & (tho >= 150)
        subsets['pair_rotation_dominated'] = (A['dt'] < 0.05) & (A['rel'] >= 90)
        subsets['pair_translation_dominated'] = (A['dt'] >= 0.3) & (A['rel'] < 20)
        subsets['pair_rotdom_theta_ge135'] = (A['dt'] < 0.05) & (A['rel'] >= 90) & (th >= 135)
        subsets['pair_rotdom_theta_lt45'] = (A['dt'] < 0.05) & (A['rel'] >= 90) & (th < 45)
        for lo, hi in SPEED_BINS:
            subsets['speed_{}_{}'.format(lo, min(hi, 1.0))] = (A['S'] >= lo) & (A['S'] < hi)
        subsets['speed_ge0.5_theta_ge150'] = (A['S'] >= 0.5) & (th >= 150)
        subsets['speed_ge0.5_theta_lt45'] = (A['S'] >= 0.5) & (th < 45)
        out = {'checkpoint': ck, 'scene': os.path.basename(sc.rstrip('/')), 'grad_mode': m.function.grad_mode,
               'speed_model': own, 'n_pairs': int(d_all.shape[0]), 'subsets': {}}
        for sname, sel in subsets.items():
            out['subsets'][sname] = {k: q(A[k], sel) for k in ('N_lie', 'N_chart', 'perp', 'ratio', 'T')}
            out['subsets'][sname]['n'] = int(sel.sum())
        with open(dst, 'w') as fh:
            json.dump(out, fh, indent=1)
        print('{:28s} {:8s} {:5s} N_lie/N_chart: all {:.3f}/{:.3f}  theta>=160 {:.3f}/{:.3f}  rotdom>=135 {}/{}  both>=150 {:.3f}/{:.3f}'.format(
            name, own, m.function.grad_mode,
            out['subsets']['all']['N_lie']['p50'], out['subsets']['all']['N_chart']['p50'],
            out['subsets']['theta_160_180']['N_lie']['p50'], out['subsets']['theta_160_180']['N_chart']['p50'],
            *(['{:.3f}'.format(out['subsets']['pair_rotdom_theta_ge135'][k]['p50']) if out['subsets']['pair_rotdom_theta_ge135'][k] else '-' for k in ('N_lie', 'N_chart')]),
            out['subsets']['pair_both_ge150']['N_lie']['p50'], out['subsets']['pair_both_ge150']['N_chart']['p50']), flush=True)


if __name__ == '__main__':
    main()
