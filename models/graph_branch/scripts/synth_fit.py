"""Representation test with exact labels: fit each head to T* of a synthetic problem (synth.py) by regression and
measure where its ridges end up and how large a time it can express (properties 8, 9).

    python models/graph_branch/scripts/synth_fit.py --kind sector --A 4 --heads ggram,br1,br4 --seed 0 \
        --out results/graph_branch/synth/sector_A4

Training: Adam (lr 1e-3, cosine to 1e-5), batch 2048 fresh pairs per step (50 % uniform, 25 % near the true ridge,
25 % local), relative squared error ((T - T*) / (T* + 0.05))^2, every head with the graph_native network defaults
(G-Shape encoder, Fourier 0.5 / 0.1, input skip). Evaluation on fixed held-out sets (seed 777):

* field     relative error |T - T*| / T* on 20 000 uniform pairs (T* > 0.05) and on 10 000 near-ridge pairs;
* ridge     for 64 starts, the goal yaw scanned over a full turn at the start's position (pure rotations, 1441
            samples): ridge location error = circular distance between argmax T and the true ridge (deg), and
            the height error at the ridge; with a disc, also goals on a circle behind the disc (argmax angle);
* cap       max over the scans of T vs max T* (pure rotations): graph_gram's rotation term is capped at 1/2;
* planning  steepest descent on the decoded time in the problem's own metric (planar: v_1, v_2, w_z), step
            0.01 in metric length, 600 steps, 300 random start / goal pairs; success = metric distance to the goal
            < 0.02 (then the exact final segment is charged); cost ratio = path length / T*.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')

from models.graph_branch import synth  # noqa: E402
from models.graph_branch import chart_metric  # noqa: E402,F401  (arch 'metric_chart')
from models.graph_native import model_network_metric as gn_network  # noqa: E402

TWO_PI = 2.0 * math.pi
HEADS = {
    'ggram': ('graph_gram', {}),
    'gmetric': ('graph_metric', {}),
    'br1': ('graph_branch', dict(n_branch=1)),
    'br4': ('graph_branch', dict(n_branch=4)),
    'br4L': ('graph_branch', dict(n_branch=4, res_kind='lse', res_even=128, res_odd=128)),
    'br4P': ('graph_branch', dict(n_branch=4, res_kind='lp', res_even=128, res_odd=128)),
    'br4H': ('graph_branch', dict(n_branch=4, res_huber=0.05)),
    'br4PH': ('graph_branch', dict(n_branch=4, res_kind='lp', res_even=128, res_odd=128, res_huber=0.05)),
    'bc4': ('graph_branch_conf', dict(n_branch=4)),
    'morig': ('metric_chart', {}),   # the original metric NTFields network (chart input), models/graph_branch/chart_metric.py
}


def build(tag, seed, device):
    arch, extra = HEADS[tag]
    torch.manual_seed(seed)
    net = gn_network.NN(device, 6, None, graph=dict(arch=arch, fourier_seed=seed, **extra))
    return net.to(device).float()


def value(net, p0, p1, device, chunk=65536):
    out = []
    for i in range(0, p0.shape[0], chunk):
        g0 = synth.to_features(p0[i:i + chunk]).to(device, torch.float32)
        g1 = synth.to_features(p1[i:i + chunk]).to(device, torch.float32)
        out.append(net.net.value(g0, g1).double().cpu())
    return torch.cat(out)


def train(net, prob, steps, batch, seed, device, lr=1e-3, log=None):
    g = torch.Generator().manual_seed(seed + 1000)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
    t0 = time.time()
    for it in range(1, steps + 1):
        p0, p1 = prob.sample_pairs(batch, g)
        Ts = prob.T_star(p0, p1).to(device, torch.float32)
        T = net.net.value(synth.to_features(p0).to(device, torch.float32), synth.to_features(p1).to(device, torch.float32))
        loss = (((T - Ts) / (Ts + 0.05)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if log and (it % 500 == 0 or it == 1):
            log('  step {:5d} loss {:.3e} ({:.0f}s)'.format(it, loss.item(), time.time() - t0))
    return net


def circ_deg(a, b):
    return (torch.remainder(a - b + math.pi, TWO_PI) - math.pi).abs() * 180 / math.pi


@torch.no_grad()
def evaluate(net, prob, device):
    g = torch.Generator().manual_seed(777)
    res = {}
    # field error, uniform and near-ridge pairs
    p0, p1 = prob.sample_pairs(20000, g, mix=(1.0, 0.0, 0.0))
    Ts, T = prob.T_star(p0, p1), value(net, p0, p1, device)
    m = Ts > 0.05
    rel = ((T - Ts).abs() / Ts)[m]
    res['uniform_rel_median'], res['uniform_rel_p95'] = float(rel.median()), float(rel.quantile(0.95))
    p0, p1 = prob.sample_pairs(10000, g, mix=(0.0, 1.0, 0.0))
    Ts, T = prob.T_star(p0, p1), value(net, p0, p1, device)
    rel = ((T - Ts).abs() / Ts)[Ts > 0.05]
    res['ridge_rel_median'], res['ridge_rel_p95'] = float(rel.median()), float(rel.quantile(0.95))
    res['lower_bound_viol'] = float(((T - prob.T_free(p0, p1)) < -1e-4).double().mean())
    # rotational ridge scans (pure rotations at the start position)
    n_s, n_psi = 64, 1441
    st = torch.cat([prob.sample_t(n_s, g), torch.rand(n_s, 1, generator=g, dtype=torch.float64) * TWO_PI], 1)
    psi = torch.linspace(0, TWO_PI, n_psi, dtype=torch.float64)[:-1]
    P0 = st.repeat_interleave(len(psi), 0)
    P1 = torch.cat([st[:, :2].repeat_interleave(len(psi), 0), psi.repeat(n_s)[:, None]], 1)
    T = value(net, P0, P1, device).reshape(n_s, -1)
    Ts = prob.T_star(P0, P1).reshape(n_s, -1)
    ridge_true = psi[Ts.argmax(1)]
    ridge_net = psi[T.argmax(1)]
    err = circ_deg(ridge_net, ridge_true)
    res['rot_ridge_err_deg_median'], res['rot_ridge_err_deg_max'] = float(err.median()), float(err.max())
    res['rot_ridge_true_offset_from_halfturn_deg_median'] = float(circ_deg(ridge_true, st[:, 2] + math.pi).median())
    res['rot_ridge_net_offset_from_halfturn_deg_median'] = float(circ_deg(ridge_net, st[:, 2] + math.pi).median())
    res['rot_scan_rel_err_median'] = float(((T - Ts).abs() / Ts.clamp_min(0.05)).median())
    res['rot_max_T'], res['rot_max_T_star'] = float(T.max()), float(Ts.max())
    res['rot_peak_height_rel_err_median'] = float(((T.max(1).values - Ts.max(1).values).abs() / Ts.max(1).values).median())
    # translational ridge behind the disc: goals on a circle of radius 0.4 around the origin, same yaw
    if prob.r0 > 0:
        ang = torch.linspace(-math.pi, math.pi, 1441, dtype=torch.float64)[:-1]
        stt = []
        while len(stt) < 32:
            c = prob.sample_t(1, g)[0]
            if c.norm() > 0.3:
                stt.append(c)
        stt = torch.stack(stt)
        yaw = torch.rand(len(stt), generator=g, dtype=torch.float64) * TWO_PI
        P0 = torch.cat([stt, yaw[:, None]], 1).repeat_interleave(len(ang), 0)
        circ = torch.stack([0.4 * torch.cos(ang), 0.4 * torch.sin(ang)], 1)
        P1 = torch.cat([circ.repeat(len(stt), 1), yaw.repeat_interleave(len(ang))[:, None]], 1)
        T = value(net, P0, P1, device).reshape(len(stt), -1)
        Ts = prob.T_star(P0, P1).reshape(len(stt), -1)
        e = circ_deg(ang[T.argmax(1)], ang[Ts.argmax(1)])
        res['trans_ridge_err_deg_median'], res['trans_ridge_err_deg_max'] = float(e.median()), float(e.max())
        res['trans_scan_rel_err_median'] = float(((T - Ts).abs() / Ts.clamp_min(0.05)).median())
    res.update(plan(net, prob, device, g))
    return res


def plan(net, prob, device, g, n=300, steps=600, h=0.01):
    """Steepest descent of T(., goal) in the problem metric diag(1, 1, h(psi)^2 / (2 pi)^2) on (t1, t2, psi)."""
    p0, p1 = prob.sample_pairs(n, g, mix=(0.5, 0.5, 0.0))
    x = p0.clone()
    cost = torch.zeros(n, dtype=torch.float64)
    done = torch.zeros(n, dtype=torch.bool)
    G1 = synth.to_features(p1).to(device, torch.float32)
    for _ in range(steps):
        xx = x.clone().to(device, torch.float32).requires_grad_(True)
        with torch.enable_grad():
            T = net.net.value(synth.to_features(xx), G1)
            gr = torch.autograd.grad(T.sum(), xx)[0].double().cpu()
        hh = prob.h(x[:, 2])
        # natural gradient: u = -G^{-1} grad, unit metric speed
        ginv = torch.stack([torch.ones(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64),
                            (TWO_PI / hh) ** 2], 1)
        u = -ginv * gr
        nrm = torch.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2 + (hh * u[:, 2] / TWO_PI) ** 2).clamp_min(1e-12)
        dist = prob.T_star(x, p1)
        step = torch.minimum(torch.full_like(dist, h), dist)
        x_new = x + (step / nrm)[:, None] * u
        if prob.r0 > 0:  # hard disc: project back onto its boundary (the planner may not cross the obstacle)
            r = x_new[:, :2].norm(dim=1, keepdim=True)
            x_new[:, :2] = torch.where(r < prob.r0, x_new[:, :2] / r * prob.r0, x_new[:, :2])
        x_new[:, 2] = torch.remainder(x_new[:, 2], TWO_PI)
        seg = prob.T_star(x, x_new)
        cost = cost + torch.where(done, torch.zeros_like(seg), seg)
        x = torch.where(done[:, None], x, x_new)
        close = prob.T_star(x, p1) < 0.02
        cost = cost + torch.where(close & ~done, prob.T_star(x, p1), torch.zeros_like(seg))
        done = done | close
        if bool(done.all()):
            break
    ratio = cost / prob.T_star(p0, p1).clamp_min(1e-6)
    out = {'plan_success': float(done.double().mean())}
    out['plan_cost_ratio_median'] = float(ratio[done].median()) if done.any() else None
    out['plan_cost_ratio_p90'] = float(ratio[done].quantile(0.9)) if done.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', default='sector')
    ap.add_argument('--A', type=float, default=4.0)
    ap.add_argument('--heads', default='ggram,br1,br4')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prob = synth.Problem(a.kind, A=a.A)
    logf = open(os.path.join(a.out, 'log_s{}.txt'.format(a.seed)), 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()
    log('problem {} A={} H_tot/2pi={:.3f} (max rotational time {:.3f}; graph_gram cap 0.5)'.format(
        a.kind, prob.A, prob.H_tot / TWO_PI, prob.H_tot / (2 * TWO_PI)))
    for tag in a.heads.split(','):
        dst = os.path.join(a.out, '{}_s{}.json'.format(tag, a.seed))
        if os.path.exists(dst):
            log('skip {} (done)'.format(dst))
            continue
        log('head {} seed {}'.format(tag, a.seed))
        t0 = time.time()
        net = build(tag, a.seed, a.device)
        train(net, prob, a.steps, a.batch, a.seed, a.device, log=log)
        net.eval()
        res = evaluate(net, prob, a.device)
        res.update({'head': tag, 'arch': HEADS[tag][0], 'seed': a.seed, 'kind': a.kind, 'A': prob.A,
                    'steps': a.steps, 'seconds': time.time() - t0})
        log(json.dumps(res))
        with open(dst, 'w') as fh:
            json.dump(res, fh, indent=1)
        torch.save({'graph': net.graph, 'state': net.state_dict()}, os.path.join(a.out, '{}_s{}.pt'.format(tag, a.seed)))


if __name__ == '__main__':
    main()
