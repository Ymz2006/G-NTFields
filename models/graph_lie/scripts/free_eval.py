"""Obstacle-free ground-truth evaluation of graph_lie checkpoints trained on datasets/3dshape/free_se3.

With all speeds 1 the exact travel time is the SE(3) geodesic T* = sqrt(|dt|^2 + (theta / 2 pi)^2). On
fresh held-out pose pairs (uniform translations in the scene box, Haar rotations) it reports

  * field error: relative error |tau - T*| / T* (pairs with T* > 0.05) and signed bias (tau - T*) / T*,
    overall, by start-goal relative angle, and by the larger endpoint rotation angle |rotvec| (the chart's
    distortion depends on the latter, the true geodesic does not);
  * Eikonal norms at both endpoints: median |c| (Lie) and |g| (chart) by |rotvec| (exact solution: |c| = 1);
  * planning: the official forward MPPI dynamics (plan_eval_graph.py, chart-space samples, no obstacles) on
    1000 pairs; SE(3) success (pose within 0.01), chart success, and path cost = sum of SE(3) step lengths
    over T*. The exact field ('oracle') is planned with the same MPPI as the planner's own floor.

    python models/graph_lie/scripts/free_eval.py --out results/graph_lie/free [--oracle] <ckpt> [<ckpt> ...]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie import lie, model_train_lie as mtl  # noqa: E402
from models.graph_native import _sib  # noqa: E402

TWO_PI = 2.0 * math.pi
LO = torch.tensor([-0.465, -0.323, -0.489], dtype=torch.float64)
HI = torch.tensor([0.465, 0.323, 0.490], dtype=torch.float64)
STEP, TEMP, SAMPLES, HORIZON, ITERS = 0.015, 50.0, 50, 5, 200


def haar_rotvec(n, g):
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=1, keepdim=True)
    q = torch.where(q[:, :1] < 0, -q, q)
    s = q[:, 1:].norm(dim=1)
    ang = 2.0 * torch.atan2(s, q[:, 0])
    return q[:, 1:] / s.clamp_min(1e-12)[:, None] * (ang / TWO_PI)[:, None]


def pairs(n, seed):
    g = torch.Generator().manual_seed(seed)
    t0 = LO + (HI - LO) * torch.rand(n, 3, generator=g, dtype=torch.float64)
    t1 = LO + (HI - LO) * torch.rand(n, 3, generator=g, dtype=torch.float64)
    return torch.cat([t0, haar_rotvec(n, g), t1, haar_rotvec(n, g)], 1)


def geodesic(X):
    """Exact T* for chart rows (N, 12), float64; also the relative angle (deg)."""
    R0 = _sib.se3.so3_exp(TWO_PI * X[:, 3:6])
    R1 = _sib.se3.so3_exp(TWO_PI * X[:, 9:12])
    th = _sib.se3.rotation_angle(R0.transpose(-1, -2) @ R1)
    return torch.sqrt(((X[:, 6:9] - X[:, :3]) ** 2).sum(1) + (th / TWO_PI) ** 2), th * 180.0 / math.pi


def binned(values, key, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (key >= lo) & (key < hi)
        out.append({'lo': lo, 'hi': min(hi, 180), 'n': int(sel.sum()),
                    'med': float(np.median(values[sel])) if sel.any() else None,
                    'p95': float(np.percentile(values[sel], 95)) if sel.any() else None})
    return out


def mppi(value_fn, X, device, seed=0):
    """Verbatim forward MPPI of the evaluator (chart-space samples), no collision test."""
    torch.manual_seed(seed)
    XP = X.clone().float().to(device)
    B = XP.shape[0]
    done_c = torch.zeros(B, dtype=torch.bool, device=device)
    it_se3 = torch.full((B,), -1, dtype=torch.long, device=device)
    path = [XP[:, :6].clone()]
    for it in range(ITERS):
        XT = XP.clone()[:, None, None, :].repeat(1, SAMPLES, HORIZON, 1)
        dP = STEP * torch.randn(B, SAMPLES, 1, 6, device=device) + STEP * torch.randn(B, SAMPLES, HORIZON, 6, device=device)
        nrm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(nrm, min=STEP) / STEP)
        XT[..., :6] = XT[..., :6] + torch.cumsum(dP, dim=2)
        ends = XT[:, :, [0, -1], :].reshape(-1, 12)
        with torch.no_grad():
            cost = value_fn(ends).reshape(B, SAMPLES, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        w = torch.softmax(-TEMP * cost, dim=1)
        step = torch.bmm(w.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        active = ~done_c & (it_se3 < 0)
        XP[:, :6] = XP[:, :6] + step * active.unsqueeze(1)
        path.append(XP[:, :6].clone())
        done_c = done_c | (torch.norm(XP[:, 6:] - XP[:, :6], dim=1) < 0.01)
        d3, _ = geodesic(XP.double())
        newly = (d3 < 0.01) & (it_se3 < 0)
        it_se3[newly] = it
        if bool(((it_se3 >= 0) | done_c).all()):
            break
    P = torch.stack(path, 1).double().cpu()                      # (B, n, 6)
    seg = torch.cat([P[:, :-1], P[:, 1:]], -1).reshape(-1, 12)
    steps, _ = geodesic(seg)
    length = steps.reshape(B, -1).sum(1)
    return it_se3.cpu(), done_c.cpu(), length


def evaluate(value_fn, net, device, n_field=20000, n_plan=1000):
    X = pairs(n_field, 12345)
    Ts, ang = geodesic(X)
    with torch.no_grad():
        tau = torch.cat([value_fn(X[i:i + 8192].float().to(device)).double().cpu() for i in range(0, n_field, 8192)])
    reg = Ts > 0.05
    rel = ((tau - Ts).abs() / Ts)[reg].numpy()
    bias = ((tau - Ts) / Ts)[reg].numpy()
    rmax = (torch.maximum(X[:, 3:6].norm(dim=1), X[:, 9:12].norm(dim=1)) * 360.0)[reg].numpy()
    out = {'rel_p50': float(np.median(rel)), 'rel_p95': float(np.percentile(rel, 95)), 'rel_mean': float(rel.mean()),
           'bias_p50': float(np.median(bias)),
           'by_rel_angle': binned(rel, ang[reg].numpy(), [0, 90, 150, 170, 180.01]),
           'bias_by_rel_angle': binned(bias, ang[reg].numpy(), [0, 90, 150, 170, 180.01]),
           'by_rotvec': binned(rel, rmax, [0, 90, 135, 160, 180.01]),
           'bias_by_rotvec': binned(bias, rmax, [0, 90, 135, 160, 180.01])}
    if net is not None:
        Xe = X[:4096].float().to(device)
        _, gx, gy = lie.endpoint_gradients(net, Xe, 'chart', create_graph=False)
        _, cx, cy = lie.endpoint_gradients(net, Xe, 'lie', create_graph=False)
        rv = torch.cat([Xe[:, 3:6].norm(dim=1), Xe[:, 9:12].norm(dim=1)]).cpu().numpy() * 360.0
        c = torch.cat([cx.norm(dim=1), cy.norm(dim=1)]).cpu().numpy()
        g = torch.cat([gx.norm(dim=1), gy.norm(dim=1)]).cpu().numpy()
        out['eik_lie_by_rotvec'] = binned(c, rv, [0, 45, 90, 135, 160, 180.01])
        out['eik_chart_by_rotvec'] = binned(g, rv, [0, 45, 90, 135, 160, 180.01])
        out['eik_lie_med'] = float(np.median(c))
        out['eik_chart_med'] = float(np.median(g))
    Xp = pairs(n_plan, 777)
    Tp, angp = geodesic(Xp)
    it3, donec, length = mppi(value_fn, Xp, device)
    succ = (it3 >= 0).numpy()
    ratio = (length / Tp).numpy()
    out['plan'] = {'n': n_plan, 'succ_se3': float(succ.mean() * 100), 'succ_chart': float(donec.numpy().mean() * 100),
                   'cost_p50': float(np.median(ratio[succ])) if succ.any() else None,
                   'cost_p95': float(np.percentile(ratio[succ], 95)) if succ.any() else None,
                   'iters_p50': float(np.median(it3.numpy()[succ])) if succ.any() else None,
                   'by_rel_angle': [{'lo': lo, 'hi': min(hi, 180),
                                     'succ_se3': float(succ[(angp.numpy() >= lo) & (angp.numpy() < hi)].mean() * 100),
                                     'cost_p50': float(np.median(ratio[succ & (angp.numpy() >= lo) & (angp.numpy() < hi)]))
                                     if (succ & (angp.numpy() >= lo) & (angp.numpy() < hi)).any() else None}
                                    for lo, hi in ((0, 90), (90, 150), (150, 180.01))]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='*')
    ap.add_argument('--out', default='results/graph_lie/free')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--oracle', action='store_true', help='also plan with the exact field')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [('oracle', None)] if a.oracle else []
    jobs += [(os.path.basename(os.path.dirname(c)), c) for c in a.ckpts]
    for name, ck in jobs:
        dst = os.path.join(a.out, name + '.json')
        if os.path.exists(dst):
            continue
        if ck is None:
            res = evaluate(lambda Z: geodesic(Z.double())[0].float(), None, a.device)
            res.update(grad_mode='exact', arch='analytic')
        else:
            m = mtl.Model('/tmp/free', 'datasets/3dshape/free_se3', 6, [0.0] * 6, device=a.device)
            m.load(ck)
            res = evaluate(m.function.TravelTimes, m.network, a.device)
            res.update(grad_mode=m.function.grad_mode, arch=m.network.graph['arch'], checkpoint=ck)
        with open(dst, 'w') as fh:
            json.dump(res, fh, indent=1)
        print('{:24s} rel p50 {:.4f} p95 {:.4f} bias {:+.4f} | by |rotvec| {} | plan SE(3) {:.1f}% cost {}'.format(
            name, res['rel_p50'], res['rel_p95'], res['bias_p50'],
            ' '.join('{:.3f}'.format(b['med']) for b in res['by_rotvec']), res['plan']['succ_se3'],
            res['plan']['cost_p50']), flush=True)


if __name__ == '__main__':
    main()
