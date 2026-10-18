"""Kink census (property 7, "peaks only"; evaluated in float64, since float32 rounding of T over ds = 2e-4 fakes kinks): count concave (ridge) and convex (valley) kinks of T(., y) along
random geodesic lines through the pose space, away from the goal.

Along a line x(s) = x_0 exp(s u) (u a unit scaled twist) the field is sampled at spacing ds; a kink shows as a
second difference D2 = (T(s + ds) - 2 T(s) + T(s - ds)) / ds whose magnitude is a slope jump, far above the
smooth background ~ ds * |T''|. Points with |D2| / local slope > thr are kinks: D2 < 0 is a ridge (peak, allowed), D2 > 0 a
valley (V-shaped kink, forbidden away from the goal). The goal is excluded by requiring T(s) > T_min. For the
branch / cover heads the active group element is also recorded, so a kink can be matched to a branch switch.

Works with graph_native-format checkpoints (obstacle scenes; --kind graph_native, poses from the dataset) and
graph_method-format ones (obstacle-free; --kind graph). Reports kinks per unit path length.

    python models/graph_cover/scripts/kink_census.py --out results/graph_cover/kinks/T_lp2_s0.json \
        --kind graph_native --ckpt Experiments/graph_cover/T_lp2_s0/latest.pt --dataPath datasets/3dshape/data/Tshape3d_env1
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')

import numpy as np
import torch

from models.graph_method_triangle.geometry import se3

TWO_PI = 2 * math.pi


def load_native(path, device):
    from models.graph_native import model_train_metric as gnm
    m = gnm.Model('.', '.', 6, [0.0] * 6, device=device)
    m.load(path)
    net = m.network.double()
    return (lambda gx, gy: net.net.value(gx, gy)), 1.0, TWO_PI, net


def load_graph(path, device):
    from models.graph_method_triangle import model_train_graph as mt
    fn, _ = mt.load_function(path, device=device)
    net = fn.network.double()
    return net.value, fn.s_t, fn.s_r, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--kind', default='graph_native', choices=['graph_native', 'graph'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--n_lines', type=int, default=256)
    ap.add_argument('--length', type=float, default=0.3, help='half-length of each line (scaled twist units)')
    ap.add_argument('--ds', type=float, default=2e-4)
    ap.add_argument('--thr', type=float, default=0.3, help='kink threshold: slope jump / local slope (smooth background ~ ds / field scale)')
    ap.add_argument('--slope_floor', type=float, default=0.2)
    ap.add_argument('--T_min', type=float, default=0.05)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(a.seed)
    if a.kind == 'graph_native':
        value, s_t, s_r, net = load_native(a.ckpt, dev)
        pts = torch.from_numpy(np.load(os.path.join(a.dataPath, 'sampled_points.npy'))).double()
        idx = torch.randperm(pts.shape[0])[:a.n_lines]
        rows = pts[idx].to(dev)
        tx, Rx = rows[:, :3], se3.so3_exp(TWO_PI * rows[:, 3:6])
        ty, Ry = rows[:, 6:9], se3.so3_exp(TWO_PI * rows[:, 9:12])
    else:
        value, s_t, s_r, net = load_graph(a.ckpt, dev)
        from models.graph_method_triangle.data.poses import PoseSampler
        smp = PoseSampler(seed=a.seed, device=dev, dtype=torch.float64, box=1.0)
        tx, Rx = smp.uniform_poses(a.n_lines)
        ty, Ry = smp.uniform_poses(a.n_lines)
    n = tx.shape[0]
    u = torch.randn(n, 6, device=dev, dtype=torch.float64)
    u = u / u.norm(dim=1, keepdim=True)
    scale = torch.tensor([s_t] * 3 + [s_r] * 3, device=dev, dtype=torch.float64)
    ss = torch.arange(-a.length, a.length + 1e-9, a.ds, device=dev, dtype=torch.float64)
    head = getattr(net, 'net', net)
    has_idx = hasattr(head, 'all_candidates')
    T_all, idx_all = [], []
    with torch.no_grad():
        for i in range(n):
            t2, R2 = se3.right_perturb(tx[i:i + 1].expand(len(ss), -1), Rx[i:i + 1].expand(len(ss), -1, -1),
                                       ss[:, None] * u[i:i + 1] * scale)
            gx, gy = se3.features(t2, R2), se3.features(ty[i:i + 1].expand(len(ss), -1), Ry[i:i + 1].expand(len(ss), -1, -1))
            T_all.append(value(gx, gy))
            if has_idx:
                idx_all.append(head.all_candidates(gx, gy).argmin(1))
    T = torch.stack(T_all)                                     # (n, S)
    D2 = (T[:, 2:] - 2 * T[:, 1:-1] + T[:, :-2]) / a.ds          # slope jump across the centre sample
    D1 = (T[:, 1:] - T[:, :-1]) / a.ds
    slope = torch.maximum(D1[:, 1:].abs(), D1[:, :-1].abs()).clamp_min(a.slope_floor)
    D2 = D2 / slope                                              # relative to the local slope (obstacle fields
    valid = T[:, 1:-1] > a.T_min                                 # have slopes up to 1 / S_d ~ 1000)
    ridge_m = valid & (D2 < -a.thr)
    valley_m = valid & (D2 > a.thr)
    # merge consecutive flagged samples into one kink event (rising edges of the mask)
    def count_events(m):
        return int((m[:, 1:] & ~m[:, :-1]).sum().item() + m[:, 0].sum().item())

    n_ridge, n_valley = count_events(ridge_m), count_events(valley_m)
    length_total = float(valid.sum().item() * a.ds)
    out = {'ckpt': a.ckpt, 'kind': a.kind, 'n_lines': n, 'ds': a.ds, 'thr': a.thr, 'T_min': a.T_min,
           'path_length_total': length_total, 'ridge_kinks': n_ridge, 'valley_kinks': n_valley,
           'ridge_per_unit_length': n_ridge / max(length_total, 1e-9),
           'valley_per_unit_length': n_valley / max(length_total, 1e-9),
           'D2_background_p50': float(D2[valid].abs().median().item()),
           'D2_background_p99': float(D2[valid].abs().quantile(0.99).item()),
           'D2_min': float(D2[valid].min().item()), 'D2_max': float(D2[valid].max().item())}
    if has_idx:
        I = torch.stack(idx_all)
        sw = (I[:, 1:] != I[:, :-1])                       # (n, S-1): switch between samples j and j+1
        rm = ridge_m                                      # (n, S-2): D2 centred at sample j+1
        out['branch_switches'] = int(sw.sum().item())
        out['ridge_kinks_at_branch_switch'] = int((rm & (sw[:, :-1] | sw[:, 1:])).sum().item())
        out['branch_switches_with_ridge_kink'] = int((sw[:, 1:-1] & (rm[:, :-1] | rm[:, 1:])).sum().item())
        out['valley_kinks_at_branch_switch'] = int((valley_m & (sw[:, :-1] | sw[:, 1:])).sum().item())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
