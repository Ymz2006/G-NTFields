"""Offline study of planner-side rules for choosing between the two sheet plans of the dual cover planner.

    python models/graph_homotopy/scripts/select_study.py --ckpt Experiments/graph_homotopy/T_hmin_s0/latest.pt \
        --paths results/graph_homotopy/plan_paths/T_hmin_s0/Tshape3d_env1__dual_paths.npz

Every rule uses only what a planner knows (the field, its own trajectories): SE(3) convergence, iterations, the
field's cost at the start (sigma*), path length, the minimum predicted speed 1 / |grad T~| along the plan (a field-
side clearance estimate), the fraction of increasing field values along the plan. Success is scored with the
evaluator's collision check of each sheet plan (stored in the npz), so the rules are compared on the same cases.
"""
import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_method_triangle import homotopy_decoder as hd  # noqa: E402
from models.graph_native import model_train_metric as md  # noqa: E402

TWO_PI = 2.0 * math.pi


def path_stats(head, P, goal, sigma, dev):
    """P (n, 6) chart path (nan-padded), goal (6,): min / median predicted speed along the path, field values."""
    ok = ~np.isnan(P[:, 0])
    P = torch.tensor(P[ok], dtype=torch.float32, device=dev)
    tg = torch.tensor(goal[:3], dtype=torch.float32, device=dev)[None].expand(P.shape[0], 3)
    qg = sigma * hd.quat_exp_turns(torch.tensor(goal[3:], dtype=torch.float32, device=dev)[None]).expand(P.shape[0], 4)
    t = P[:, :3].clone().requires_grad_(True)
    r = P[:, 3:].clone().requires_grad_(True)
    T = head.value_cover(t, hd.quat_exp_turns(r), tg, qg)
    gt, gr = torch.autograd.grad(T.sum(), (t, r))
    # Lie-gradient norm: translation block as is, rotation block via the chart Jacobian approximated by the chart
    # gradient (turn units); the ranking between the two plans is what matters here
    n = torch.sqrt((gt ** 2).sum(1) + (gr ** 2).sum(1) + 1e-12)
    sp = (1.0 / n).detach().cpu().numpy()
    Tv = T.detach().cpu().numpy()
    inc = float((np.diff(Tv) > 1e-4).mean()) if len(Tv) > 2 else 0.0
    length = float(np.linalg.norm(np.diff(P[:, :3].cpu().numpy(), axis=0), axis=1).sum())
    return {'min_speed': float(sp.min()), 'p10_speed': float(np.percentile(sp, 10)), 'inc': inc, 'len': length, 'T0': float(Tv[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--paths', required=True)
    ap.add_argument('--dataPath', default='/lieshape/testing_data/3dshape/Tshape3d_env1')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    import os
    m = md.Model('/tmp/sel_model', 'datasets/3dshape/data/Tshape3d_env1', 6, [0.0] * 6, device=a.device)
    m.load(a.ckpt)
    head = m.network.net
    z = np.load(a.paths)
    arr = np.load(os.path.join(a.dataPath, 'sampled_points.npy')).astype(np.float32)
    n = len(z['case'])
    feats = {'plus': [], 'minus': []}
    for i in range(n):
        goal = arr[z['case'][i], 6:]
        for k, sig in (('plus', 1.0), ('minus', -1.0)):
            P = z['full_' + k][i]
            k3 = int(z['iters3_' + k][i])
            seg = P[:k3 + 2] if k3 >= 0 else P
            st = path_stats(head, seg, goal, sig, a.device)
            st['conv'] = k3 >= 0
            st['iters'] = k3 if k3 >= 0 else 10 ** 6
            feats[k].append(st)
    okp, okm = z['sheet_ok_plus'], z['sheet_ok_minus']
    sig = z['sigma_star']
    rules = {}

    def score(choose):
        return float(np.mean([okp[i] if choose(i) else okm[i] for i in range(n)]))

    fp, fm = feats['plus'], feats['minus']
    rules['always +'] = score(lambda i: True)
    rules['sigma*'] = score(lambda i: sig[i] > 0)
    rules['either (oracle)'] = float(np.mean(okp | okm))
    rules['conv, then sigma*'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else sig[i] > 0)
    rules['conv, then fewer iters (dual)'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else (fp[i]['iters'] < fm[i]['iters'] if fp[i]['iters'] != fm[i]['iters'] else sig[i] > 0))
    rules['conv, then larger min speed'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else fp[i]['min_speed'] >= fm[i]['min_speed'])
    rules['conv, then larger p10 speed'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else fp[i]['p10_speed'] >= fm[i]['p10_speed'])
    rules['conv, then fewer increases'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else (fp[i]['inc'] <= fm[i]['inc'] if fp[i]['inc'] != fm[i]['inc'] else sig[i] > 0))
    rules['conv, then shorter path'] = score(lambda i: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else fp[i]['len'] <= fm[i]['len'])
    for thr in (0.5, 0.8, 1.0):
        rules['conv, then min speed unless within x{} '.format(thr)] = score(
            lambda i, thr=thr: fp[i]['conv'] if fp[i]['conv'] != fm[i]['conv'] else (fp[i]['min_speed'] >= thr * fm[i]['min_speed']))
    for k, v in sorted(rules.items(), key=lambda kv: -kv[1]):
        print('{:48s} {:.1f}'.format(k, 100 * v))
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(rules, fh, indent=1)


if __name__ == '__main__':
    main()
