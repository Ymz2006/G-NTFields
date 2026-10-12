"""Measure triangle-inequality violations of trained checkpoints (any head).

    python models/graph_method_triangle/scripts/triangle_check.py --out results/x.json \
        Experiments/graph_method_triangle_free/tri_eq_s0/best.pt Experiments/graph_method_final/graph_gram_s0/best.pt

Obstacle-free checkpoints: x, z uniform in the training box. Stage-E checkpoints
(mode 'shape'): x, z are start / goal poses of random dataset pairs (collision-
free). Reports violations v = T(x, z) - T(x, y) - T(y, z), relative to T(x, z),
for random y and after the adversarial ascent of triangle.attack. Evaluated in
float64 so that rounding cannot fake or hide a violation. Two attacks: 'all'
moves x, y and z, 'y' keeps the sampled x, z and moves only y. `in_domain`
is the fraction of violating triples whose three translations stay inside the
training domain's translation bounding box (the attack is unconstrained).
"""
import argparse
import json
import os
import sys

sys.path.append('.')

import numpy as np
import torch

from models.graph_method_triangle import model_train_graph as mt, triangle
from models.graph_method_triangle.data.poses import PoseSampler
from models.graph_method_triangle.geometry import se3
from models.graph_method_triangle.model_function_shape import coords_to_features


def endpoints(cfg, n, seed, device):
    """Three pose batches and the domain's translation bounding box (lo, hi)."""
    if cfg['mode'] != 'shape':
        s = PoseSampler(seed=seed, device=device, dtype=torch.float64, box=cfg['box'])
        lo = torch.full((3,), -cfg['box'], dtype=torch.float64, device=device)
        return (s.uniform_poses(n), s.uniform_poses(n), s.uniform_poses(n)), (lo, -lo)
    pts = torch.from_numpy(np.load(os.path.join(cfg['data_path'], 'sampled_points.npy'))).double()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(pts.shape[0], generator=g)[:3 * n]
    G = coords_to_features(pts[idx, :6]).to(device)
    t, R = se3.split_features(G)
    allt = torch.cat([pts[:, :3], pts[:, 6:9]]).to(device)
    return ((t[:n], R[:n]), (t[n:2 * n], R[n:2 * n]), (t[2 * n:], R[2 * n:])), (allt.amin(0), allt.amax(0))


def in_domain(poses, viol, Txz, box, tol_rel=1e-4):
    lo, hi = box
    inside = torch.ones_like(viol, dtype=torch.bool)
    for t, _ in poses:
        inside &= ((t >= lo) & (t <= hi)).all(-1)
    bad = viol / Txz.clamp_min(1e-12) > tol_rel
    return (inside & bad).double().sum().item() / max(int(bad.sum()), 1)


def check(path, n, steps, device):
    f, ck = mt.load_function(path, device=device)
    cfg = f.cfg
    if cfg.get('coords'):
        return {'skipped': 'coords-mode checkpoint'}
    net = f.network.double()
    ((tx, Rx), (ty, Ry), (tz, Rz)), box = endpoints(cfg, n, 4242, device)
    gx, gy, gz = se3.features(tx, Rx), se3.features(ty, Ry), se3.features(tz, Rz)
    with torch.no_grad():
        Txz = net.value(gx, gz)
        v = Txz - net.value(gx, gy) - net.value(gy, gz)
    out = {'arch': cfg['arch'], 'mode': cfg['mode'], 'rot_mode': cfg.get('rot_mode', 'n/a'),
           'data': cfg.get('data_path') if cfg['mode'] == 'shape' else 'free',
           'random': triangle.summarize(v, Txz)}
    for name, move_ends in (('attack', True), ('attack_y', False)):
        va, Ta, P = triangle.attack(net, tx, Rx, tz, Rz, steps=steps, lr=0.02, move_ends=move_ends)
        out[name] = triangle.summarize(va, Ta)
        out[name]['in_domain'] = in_domain(P, va, Ta, box)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('ckpts', nargs='+')
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    res = {}
    for c in a.ckpts:
        res[c] = check(c, a.n, a.steps, a.device)
        print(c, json.dumps(res[c]), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
