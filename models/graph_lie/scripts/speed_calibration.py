"""Predicted vs ground-truth speed for isotropic graph_lie runs (chart vs Lie gradient).

For every finished run <scene>_<head>_iso_<grad>_s<seed>, endpoints of a fixed random subset of the
scene's training pairs are evaluated:

    ground truth   S_d          warped translational clearance speed, the isotropic Eikonal's target
    predicted      1 / |grad|   with grad = the run's own gradient (chart: d tau / d(t, rotvec/2pi); Lie: c)
    predicted_lie  1 / |c|      Lie (body-frame) gradient for both, i.e. both fields in one physical metric

Speeds are binned by ground truth (log and linear bins); per bin and method the quantiles of the predicted
speed are written, seeds pooled. A perfect field lies on predicted = ground truth.

    python models/graph_lie/scripts/speed_calibration.py --out results/graph_lie/iso_report/speed_calibration.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie import lie, model_train_lie as mtl  # noqa: E402
from models.graph_native.model_train_metric import load_dataset  # noqa: E402

NAME = re.compile(r'^([TLA])_(gmetric|trieq)_iso_(chart|lie)_s(\d+)$')
SCENE = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
LOG_EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]
LIN_EDGES = [round(0.1 * i, 1) for i in range(10)] + [1.0001]
Q = [10, 25, 50, 75, 90]


def endpoint_speeds(run_dir, rows, device):
    m = mtl.Model('/tmp/cal', 'datasets/3dshape/x', 6, [0.0] * 6, device=device)
    m.load(os.path.join(run_dir, 'latest.pt'))
    gt, own, lie_s = [], [], []
    for i in range(0, rows.shape[0], 4096):
        d = rows[i:i + 4096].to(device)
        sp = mtl.warp_speeds(d)
        _, gx, gy = lie.endpoint_gradients(m.network, d[:, :12], m.function.grad_mode, create_graph=False)
        _, cx, cy = lie.endpoint_gradients(m.network, d[:, :12], 'lie', create_graph=False)
        gt.append(torch.cat([sp['Sd'][:, 0], sp['Sd'][:, 1]]).cpu())
        own.append(1.0 / torch.cat([gx.norm(dim=1), gy.norm(dim=1)]).clamp_min(1e-9).cpu())
        lie_s.append(1.0 / torch.cat([cx.norm(dim=1), cy.norm(dim=1)]).clamp_min(1e-9).cpu())
    return torch.cat(gt).numpy(), torch.cat(own).numpy(), torch.cat(lie_s).numpy()


def binned(gt, pred, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (gt >= lo) & (gt < hi)
        if s.sum() < 20:
            out.append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(s.sum())})
            continue
        q = np.percentile(pred[s], Q)
        out.append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(s.sum()), 'gt_median': float(np.median(gt[s])),
                    **{'p{}'.format(k): float(v) for k, v in zip(Q, q)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_lie/iso_report/speed_calibration.json')
    ap.add_argument('--rows', type=int, default=8192, help='pairs per run (2 endpoints each)')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    runs = {}
    for d in sorted(glob.glob('Experiments/graph_lie/*_iso_*')):
        m = NAME.match(os.path.basename(d))
        if m and os.path.exists(os.path.join(d, 'train_summary.json')):
            runs.setdefault((m[1], m[2], m[3]), []).append(d)
    data_cache, res = {}, {'log_edges': LOG_EDGES, 'lin_edges': LIN_EDGES, 'quantiles': Q, 'groups': {}}
    for (scene, head, grad), dirs in sorted(runs.items()):
        if scene not in data_cache:
            data = load_dataset(os.path.join('datasets/3dshape', SCENE[scene]))
            g = torch.Generator().manual_seed(0)
            data_cache[scene] = data[torch.randperm(data.shape[0], generator=g)[:a.rows]]
        rows = data_cache[scene]
        G, O, L = [], [], []
        for d in dirs:
            gt, own, lie_s = endpoint_speeds(d, rows, a.device)
            G.append(gt), O.append(own), L.append(lie_s)
        G, O, L = np.concatenate(G), np.concatenate(O), np.concatenate(L)
        res['groups']['{}|{}|{}'.format(scene, head, grad)] = {
            'seeds': [os.path.basename(d) for d in dirs], 'n_endpoints': int(G.size),
            'own_log': binned(G, O, LOG_EDGES), 'own_lin': binned(G, O, LIN_EDGES),
            'lie_log': binned(G, L, LOG_EDGES), 'lie_lin': binned(G, L, LIN_EDGES),
            'ratio_median_own': float(np.median(O / G)), 'ratio_median_lie': float(np.median(L / G))}
        print('{} {} {}: {} seeds, median predicted/true speed {:.2f} (own grad) {:.2f} (Lie grad)'.format(
            scene, head, grad, len(dirs), np.median(O / G), np.median(L / G)), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
