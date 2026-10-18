"""Write a synthetic SE(2)-in-SE(3) problem (models/graph_branch/synth.py) as an NTFields training set, so the
heads can be trained with the physics loss alone (no labels) and scored against the exact travel time.

    python models/graph_branch/scripts/synth_dataset.py --kind sector --A 4 --out datasets/3dshape/data/synth_sector_A4

Files as in datasets/3dshape/data/<scene>: sampled_points (N, 12) chart rows (t, rotvec / 2 pi) of start and goal,
speed / normal / speed_dists / speed_angles. The trainer warps the stored speeds (model_train_lie.warp_speeds), so
the stored values are the pre-images of the intended speeds: translational S_d = 1 (stored 1), rotational
S_a = 1 / h(psi) (stored raw_from_warped(1 / h) * ANGLE_MAX). With --speed aniso --planar the loss is
sqrt(S_d^2 (c_v1^2 + c_v2^2) + S_a^2 c_wz^2) = 1 at both endpoints, the exact Eikonal equation of the problem.

Pairs mimic the 3-D shape datasets: the goal is the start plus a random displacement (|dt| ~ |N(0, 0.2)| in the
plane, dpsi ~ N(0, 1) rad; median |dt| 0.13, median |dpsi| 39 deg), plus 20 % uniform pairs.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_branch import synth  # noqa: E402
from models.graph_lie.model_train_lie import ANGLE_MAX  # noqa: E402
from models.graph_method_triangle.speed_fourier import raw_from_warped  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='sector')
    ap.add_argument('--A', type=float, default=4.0)
    ap.add_argument('--n', type=int, default=400000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    prob = synth.Problem(a.kind, A=a.A)
    g = torch.Generator().manual_seed(a.seed)
    n = a.n
    p0 = torch.cat([prob.sample_t(n, g), torch.rand(n, 1, generator=g, dtype=torch.float64) * 2 * math.pi], 1)
    n_loc = int(0.8 * n)
    p1 = p0.clone()
    for _ in range(20):  # resample displacements that leave the square (or enter the disc)
        bad = torch.ones(n, dtype=torch.bool) if _ == 0 else bad
        k = int(bad[:n_loc].sum())
        if k == 0:
            break
        d = torch.randn(k, 2, generator=g, dtype=torch.float64)
        d = d / d.norm(dim=1, keepdim=True) * (torch.randn(k, 1, generator=g, dtype=torch.float64).abs() * 0.2)
        idx = torch.nonzero(bad[:n_loc])[:, 0]
        p1[idx, :2] = p0[idx, :2] + d
        p1[idx, 2] = p0[idx, 2] + torch.randn(k, generator=g, dtype=torch.float64)
        bad = torch.zeros(n, dtype=torch.bool)
        bad[:n_loc] = (p1[:n_loc, :2].abs() > prob.half).any(1)
        if prob.r0 > 0:
            bad[:n_loc] |= p1[:n_loc, :2].norm(dim=1) <= prob.r0 + 0.01
    p1[n_loc:] = torch.cat([prob.sample_t(n - n_loc, g),
                            torch.rand(n - n_loc, 1, generator=g, dtype=torch.float64) * 2 * math.pi], 1)
    p1[:, 2] = torch.remainder(p1[:, 2], 2 * math.pi)
    X = torch.cat([synth.to_chart(p0), synth.to_chart(p1)], 1).float().numpy()
    Sa = torch.stack([1.0 / prob.h(p0[:, 2]), 1.0 / prob.h(p1[:, 2])], 1)
    stored_sa = (raw_from_warped(Sa) * ANGLE_MAX).float().numpy()
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, 'sampled_points.npy'), X)
    np.save(os.path.join(a.out, 'speed.npy'), np.ones((n, 2), np.float32))
    np.save(os.path.join(a.out, 'normal.npy'), np.zeros((n, 12), np.float32))
    np.save(os.path.join(a.out, 'speed_dists.npy'), np.ones((n, 2), np.float32))
    np.save(os.path.join(a.out, 'speed_angles.npy'), stored_sa)
    np.save(os.path.join(a.out, 'planar_rows.npy'), torch.cat([p0, p1], 1).numpy())
    meta = {'synthetic': True, 'kind': a.kind, 'A': prob.A, 'w': prob.w, 'psi0': prob.psi0, 'r0': prob.r0,
            'half': prob.half, 'n': n, 'seed': a.seed, 'note': 'models/graph_branch/synth.py; exact T* available'}
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    dt = (p1[:, :2] - p0[:, :2]).norm(dim=1)
    dpsi = (torch.remainder(p1[:, 2] - p0[:, 2] + math.pi, 2 * math.pi) - math.pi).abs()
    print('wrote', a.out, 'median |dt| {:.3f}, median |dpsi| {:.1f} deg, S_a range {:.3f}..1'.format(
        float(dt.median()), float(dpsi.median()) * 180 / math.pi, float(Sa.min())))


if __name__ == '__main__':
    main()
