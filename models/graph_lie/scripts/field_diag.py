"""Field diagnostics for graph_lie / graph_native checkpoints: which Eikonal equation does a trained field satisfy?

For a fixed random subset of its training scene's pairs, every checkpoint is evaluated with BOTH gradients
(chart g = d tau / d(t, rotvec / 2 pi), Lie c = body-frame derivative) and all three speed formulations:

    N = S |grad| (combined),  S_d |grad| (iso),  sqrt(S_d^2 |grad_v|^2 + S_a^2 |grad_w|^2) (aniso)

N = 1 is the Eikonal equation. Reported per checkpoint (start and goal endpoints pooled):
  * res_<f>_<g>: median |N - 1| and mean 1e-2 (sqrt(N) - 1)^2 for formulation f and gradient g;
  * by_angle: median N of the checkpoint's own formulation, Lie and chart, binned by the endpoint's
    rotation angle |rotvec| (the chart's distortion grows with it; a physically consistent field has a flat
    Lie profile);
  * rot_share: median |c_w|^2 / |c|^2 (how much of the Lie gradient is rotational);
  * block: median S_d |c_v| and S_a |c_w| (the per-block speeds the aniso formulation pins);
  * by_speed: median N (own formulation, Lie gradient) binned by the endpoint's warped speed of that
    formulation (S, S_d, or S_d for aniso): N < 1 in the slow bins = a weaker barrier than the speed asks
    for near obstacles, which predicts collisions.

    python models/graph_lie/scripts/field_diag.py --out results/graph_lie/diag <ckpt> [<ckpt> ...]
(inside pytorchserver from the repo root; a checkpoint's scene is read from its run's config.json)
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

BINS = [0, 45, 90, 135, 160, 180.01]
SPEED_BINS = [0, 0.01, 0.1, 0.5, 1.01]


def scene_of(ckpt):
    cfg = os.path.join(os.path.dirname(ckpt), 'config.json')
    return json.load(open(cfg))['DataPath']


def diagnose(ckpt, n, device, seed=0):
    m = mtl.Model('/tmp/diag', scene_of(ckpt), 6, [0.0] * 6, device=device)
    m.load(ckpt)
    data = load_dataset(m.Params['DataPath'])
    g = torch.Generator().manual_seed(seed)
    d = data[torch.randperm(data.shape[0], generator=g)[:n]].to(device)
    out = {'checkpoint': ckpt, 'scene': os.path.basename(m.Params['DataPath'].rstrip('/')),
           'grad_mode': m.function.grad_mode, 'speed_model': m.function.speed_model,
           'arch': m.network.graph['arch'], 'n': n}
    acc = {}
    for i in range(0, n, 4096):
        dd = d[i:i + 4096]
        pts = dd[:, :12]
        sp = mtl.warp_speeds(dd)
        T, gx, gy = lie.endpoint_gradients(m.network, pts, 'chart', create_graph=False)
        _, cx, cy = lie.endpoint_gradients(m.network, pts, 'lie', create_graph=False)
        rows = {'T': T, 'ang': torch.cat([pts[:, 3:6].norm(dim=1), pts[:, 9:12].norm(dim=1)]) * 360.0,
                'rot_share': torch.cat([(cx[:, 3:] ** 2).sum(1) / (cx ** 2).sum(1).clamp_min(1e-12),
                                        (cy[:, 3:] ** 2).sum(1) / (cy ** 2).sum(1).clamp_min(1e-12)]),
                'blk_v': torch.cat([sp['Sd'][:, 0] * cx[:, :3].norm(dim=1), sp['Sd'][:, 1] * cy[:, :3].norm(dim=1)]),
                'blk_w': torch.cat([sp['Sa'][:, 0] * cx[:, 3:].norm(dim=1), sp['Sa'][:, 1] * cy[:, 3:].norm(dim=1)]),
                'S': torch.cat([sp['S'][:, 0], sp['S'][:, 1]]), 'Sd': torch.cat([sp['Sd'][:, 0], sp['Sd'][:, 1]])}
        for f in mfl.SPEED_MODELS:
            for gname, (ga, gb) in (('lie', (cx, cy)), ('chart', (gx, gy))):
                rows['N_{}_{}'.format(f, gname)] = torch.cat([mfl.eikonal_N(ga, sp, 0, f), mfl.eikonal_N(gb, sp, 1, f)])
        for k, v in rows.items():
            acc.setdefault(k, []).append(v.detach().cpu())
    A = {k: torch.cat(v).double().numpy() for k, v in acc.items()}
    out['T_mean'] = float(A['T'].mean())
    for f in mfl.SPEED_MODELS:
        for gname in ('lie', 'chart'):
            N = A['N_{}_{}'.format(f, gname)]
            out['res_{}_{}'.format(f, gname)] = {'med_abs': float(np.median(np.abs(N - 1))),
                                                 'loss': float(np.mean(1e-2 * (np.sqrt(N) - 1) ** 2)),
                                                 'within10': float(np.mean(np.abs(N - 1) < 0.1))}
    own = m.function.speed_model
    out['by_angle'] = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = (A['ang'] >= lo) & (A['ang'] < hi)
        out['by_angle'].append({'lo': lo, 'hi': min(hi, 180), 'n': int(sel.sum()),
                                'N_lie': float(np.median(A['N_{}_lie'.format(own)][sel])),
                                'N_chart': float(np.median(A['N_{}_chart'.format(own)][sel])),
                                'rot_share': float(np.median(A['rot_share'][sel]))})
    spd = A['S'] if own == 'combined' else A['Sd']
    out['by_speed'] = []
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        sel = (spd >= lo) & (spd < hi)
        out['by_speed'].append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(sel.sum()),
                                'N_lie': float(np.median(A['N_{}_lie'.format(own)][sel])) if sel.any() else None})
    out['rot_share'] = float(np.median(A['rot_share']))
    out['block'] = {'Sd_cv': float(np.median(A['blk_v'])), 'Sa_cw': float(np.median(A['blk_w']))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--out', default='results/graph_lie/diag')
    ap.add_argument('--n', type=int, default=32768)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--name', nargs='*', default=None, help='output names (default: run folder names)')
    ap.add_argument('--force', action='store_true', help='recompute existing outputs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    names = a.name or [os.path.basename(os.path.dirname(c)) for c in a.ckpts]
    for name, ck in zip(names, a.ckpts):
        dst = os.path.join(a.out, name + '.json')
        if os.path.exists(dst) and not a.force:
            continue
        res = diagnose(ck, a.n, a.device)
        with open(dst, 'w') as fh:
            json.dump(res, fh, indent=1)
        own = res['speed_model']
        print('{:28s} {:5s} {:8s} res(own, lie) {:.3f} res(own, chart) {:.3f} | N_lie by angle {} | N_chart {}'.format(
            name, res['grad_mode'], own, res['res_{}_lie'.format(own)]['med_abs'],
            res['res_{}_chart'.format(own)]['med_abs'],
            ' '.join('{:.2f}'.format(b['N_lie']) for b in res['by_angle']),
            ' '.join('{:.2f}'.format(b['N_chart']) for b in res['by_angle'])), flush=True)
        print('{:28s} N_lie by speed bin {}'.format(name, ' '.join(
            '{:.2f}'.format(b['N_lie']) if b['N_lie'] is not None else '-' for b in res['by_speed'])), flush=True)


if __name__ == '__main__':
    main()
