#!/usr/bin/env python
"""Stage 0 of the spec (mechanics check of the learned cut): travel time on the circle, goal at 0, unit speed.

  0a  no obstacle:  T* = min(delta, 2 pi - delta), two route families, ridge at 180 deg
  0b  arc [90, 120] deg blocked, start in the free arc: T* = distance along the free arc, one family, no ridge

Model: the spec's free heads on the circle (direct head T_0 = sqrt((|A (e(x) - e(0))|^2 + rho |G(x) - G(0)|^2))
exp(tanh g_0), detours T_k = softplus f_k, e = small Fourier MLP of G = (cos, sin)), min over K branches, trained by
the spec's losses (decoded Eikonal with the 3 tau gap mask, sheet Eikonal on the band, distinctness at ties),
temperature schedule, near-cut sampling (20 %), usage checks every 500 steps with revival / pruning / merging.
Unit speed, no causality weight (the obstacle-free convention of graph_lie). K = 1 is the single-head baseline.

Metrics (spec section 9, 3 seeds): surviving branches (usage > 1 %), ridge-type cuts (ties with gradient cosine
< gamma) and their distance to 180 deg, hand-overs (ties with parallel gradients), median relative error, ridge-band
(|delta - pi| < 0.05) median / p95 error, second-best accuracy on the ridge band (0a), Eikonal residual near the ridge.

    python models/graph_mb/scripts/stage0_circle.py --out results/graph_mb/stage0 --seeds 0 1 2
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from torch.nn import Linear

TWO_PI = 2.0 * math.pi
DT = torch.float32


def inv_softplus(y):
    return math.log(math.expm1(y))


class Enc(torch.nn.Module):
    def __init__(self, n_freq=16, sigma=1.0, h=64, out=32):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer('B', torch.randn(2, n_freq, generator=g) * sigma)
        self.l1, self.l2, self.l3 = Linear(2 * n_freq, h), Linear(h, h), Linear(h, out)
        self.act = torch.nn.Softplus(beta=10)

    def forward(self, d):
        G = torch.stack([torch.cos(d), torch.sin(d)], -1)
        x = G @ (TWO_PI * self.B)
        x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return self.l3(self.act(self.l2(self.act(self.l1(x)))))


class Head(torch.nn.Module):
    def __init__(self, inp=32, h=32, bias=0.0):
        super().__init__()
        self.l1, self.l2 = Linear(inp, h), Linear(h, 1)
        self.act = torch.nn.Softplus(beta=10)
        with torch.no_grad():
            self.l2.weight.mul_(0.1)
            self.l2.bias.fill_(bias)

    def forward(self, e):
        return self.l2(self.act(self.l1(e)))[:, 0]


class Circle(torch.nn.Module):
    def __init__(self, K, rho0=0.01, offset_init=0.3, level=1.0):
        super().__init__()
        self.K = K
        self.enc = Enc()
        self.A = Linear(32, 8, bias=False)
        self.g0 = Head()
        self.f = torch.nn.ModuleList([Head(bias=inv_softplus(level + offset_init * k)) for k in range(1, K)])
        self.rho0 = rho0
        self.register_buffer('alive', torch.ones(K))

    def all_values(self, d):
        e = self.enc(d)
        e0 = self.enc(torch.zeros(1, dtype=d.dtype, device=d.device))
        G = torch.stack([torch.cos(d), torch.sin(d)], -1)
        base = (self.A(e - e0) ** 2).sum(-1) + self.rho0 * ((G - torch.tensor([1.0, 0.0], device=d.device)) ** 2).sum(-1)
        T0 = torch.sqrt(base.clamp_min(0.0)) * torch.exp(torch.tanh(self.g0(e)))
        V = [T0] + [torch.nn.functional.softplus(f(e)) for f in self.f]
        V = torch.stack(V, 1)
        return torch.where(self.alive[None, :] > 0.5, V, torch.full_like(V, float('inf')))

    def value_branch(self, d, b):
        return self.all_values(d).gather(1, b[:, None])[:, 0]

    def alive_units(self):
        return [k for k in range(self.K) if self.alive[k] > 0.5]

    def copy_unit(self, dst, src, noise, raise_by, pool=None):
        with torch.no_grad():
            if src >= 1:
                for pd, ps in zip(self.f[dst - 1].parameters(), self.f[src - 1].parameters()):
                    pd.copy_(ps)
                    pd.add_(noise * (pd.std() if pd.numel() > 1 else pd.abs()) * torch.randn_like(pd))
                self.f[dst - 1].l2.bias.add_(raise_by)
                self.alive[dst] = 1.0
                return
            # the direct head has another parametrisation: copy its function by regression on the pool
            self.alive[dst] = 1.0
            e = self.enc(pool).detach()
            target = self.all_values(pool)[:, src].detach() + raise_by
        head = self.f[dst - 1]
        for p in head.parameters():
            p.data.normal_(0, 0.05)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(400):
            opt.zero_grad()
            loss = ((torch.nn.functional.softplus(head(e)) - target) ** 2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for p in head.parameters():
                p.add_(noise * (p.std() if p.numel() > 1 else p.abs()) * torch.randn_like(p))

    def unit_params(self, k):
        return list(self.f[k - 1].parameters()) if k >= 1 else list(self.g0.parameters()) + [self.A.weight]


def truth(d, case):
    """0a: min(delta, 2 pi - delta). 0b (free arc only): delta below 90 deg, 2 pi - delta above 120 deg
    (the route through the slow arc costs 0.52 / 0.05 = 10.5 more, never optimal in the free arc)."""
    d = d % TWO_PI
    if case == '0a':
        return torch.minimum(d, TWO_PI - d), torch.maximum(d, TWO_PI - d)
    return torch.where(d < math.pi / 2, d, TWO_PI - d), None


def sample(n, case, g, device):
    """Uniform on the whole circle (0b: the blocked arc is a slow region, as obstacles are in the NTFields data)."""
    return torch.rand(n, generator=g) * TWO_PI


def speed(d, case, s_block=0.05):
    if case == '0a':
        return torch.ones_like(d)
    dm = d % TWO_PI
    return torch.where((dm >= math.pi / 2) & (dm <= 2 * math.pi / 3), torch.full_like(d, s_block), torch.ones_like(d))


def train(case, K, seed, steps, spec, device, log):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Circle(K).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec['lr'], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=spec['lr'] * 0.03)
    B = 512
    n_near = int(spec['near_cut_frac'] * B)
    pool = sample(4096, case, torch.Generator().manual_seed(999), device).to(device)
    Delta, eta, gam = spec['delta'], spec['eta'], spec['gamma']
    streak, n_rev, events, checks = [0] * K, [0] * K, [], []
    splits = {}   # unit k -> (src, v, until_step): forced ownership by the sign of v * dT_src/d delta (split revival)
    st = dict(win=torch.zeros(K), res=torch.zeros(K), resn=torch.zeros(K), shared=torch.zeros(K, K), dup=torch.zeros(K, K),
              n=0, near=0, neardup=0)
    for step in range(1, steps + 1):
        f = min(1.0, (step - 1) / (spec['anneal_frac'] * steps))
        tau = spec['tau0'] * (spec['tau_end'] / spec['tau0']) ** f
        d = sample(B, case, g, device).to(device)
        if n_near > 0:
            pd = sample(2 * B, case, g, device).to(device)
            with torch.no_grad():
                srt = net.all_values(pd).sort(1).values
                gp = srt[:, 1] - srt[:, 0] if K > 1 else torch.full_like(srt[:, 0], 1e9)
                gp = torch.where(torch.isfinite(gp), gp, torch.full_like(gp, 1e9))
                d = torch.cat([d[:B - n_near], pd[gp.topk(n_near, largest=False).indices]])
        with torch.no_grad():
            V = net.all_values(d)
            srt, order = V.sort(1)
            Tmin, win = srt[:, 0], order[:, 0]
            second = order[:, 1] if K > 1 else order[:, 0]
            gap = srt[:, 1] - srt[:, 0] if K > 1 else torch.full_like(Tmin, 1e9)
            gap = torch.where(torch.isfinite(gap), gap, torch.full_like(gap, 1e9))
            band = V <= (Tmin + Delta)[:, None]
            ri, rb = band.nonzero(as_tuple=True)
            row_of = torch.full((B, K), -1, dtype=torch.long, device=device)
            row_of[ri, rb] = torch.arange(ri.shape[0], device=device)
            is_win = rb == win[ri]
        x = d[ri].clone().requires_grad_(True)
        T = net.value_branch(x, rb)
        gr = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        S = speed(x.detach(), case)
        # NTFields residual: (sqrt(S |g|) - 1)^2 has an infinite gradient at g = 0 (a flat region is not a
        # stationary point); the spec's (g^2 - 1)^2 has zero gradient there and the single head gets stuck flat.
        r = (torch.sqrt(S * gr.abs() + 1e-8) - 1.0) ** 2 * torch.exp(-0.5 * T)   # NTFields residual + causality weight
        m_dec = (is_win & (gap[ri] > 3 * tau)).float()
        w_sheet = torch.ones_like(r)
        for k, (src, v, until) in list(splits.items()):
            if step > until:
                del splits[k]
                continue
            with torch.no_grad():
                owner_ok = (win == src) | (win == k)
                rs, rk = row_of[:, src], row_of[:, k]
                both = owner_ok & (rs >= 0) & (rk >= 0)
                ib = both.nonzero(as_tuple=True)[0]
                if ib.numel() == 0:
                    continue
                sgn = gr.detach()[rs[ib]] * v
                to_k = sgn < 0
                m_dec[rs[ib]] = (~to_k).float()
                m_dec[rk[ib]] = to_k.float()
                w_sheet[rs[ib]] = (~to_k).float()
                w_sheet[rk[ib]] = to_k.float()
        L_dec = (r * m_dec).sum() / B
        L_sheet = (r * w_sheet).sum() / max(float(w_sheet.sum()), 1.0)
        near = gap < eta
        i_near = near.nonzero(as_tuple=True)[0]
        L_dist = torch.zeros((), device=device)
        if i_near.numel():
            r1, r2 = row_of[i_near, win[i_near]], row_of[i_near, second[i_near]]
            ok = (r1 >= 0) & (r2 >= 0)
            r1, r2, i_near = r1[ok], r2[ok], i_near[ok]
            if i_near.numel():
                cos = gr[r1] * gr[r2] / (gr[r1].abs() * gr[r2].abs() + 1e-8)
                L_dist = (torch.relu(cos - gam) ** 2).sum() / B
                with torch.no_grad():
                    dup = cos > gam
                    st['near'] += int(i_near.numel())
                    st['neardup'] += int(dup.sum())
                    a_, b_ = torch.minimum(win[i_near], second[i_near]), torch.maximum(win[i_near], second[i_near])
                    for aa, bb in zip(a_[dup].tolist(), b_[dup].tolist()):
                        st['dup'][aa, bb] += 1
        loss = spec['w_dec'] * L_dec + spec['w_sheet'] * L_sheet + spec['w_dist'] * L_dist
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            st['win'] += torch.bincount(win, minlength=K).float().cpu()
            st['res'] += torch.bincount(rb, weights=r.detach(), minlength=K).float().cpu()
            st['resn'] += torch.bincount(rb, minlength=K).float().cpu()
            st['n'] += B
            both = row_of[torch.arange(B, device=device), second] >= 0
            a_, b_ = torch.minimum(win[both], second[both]), torch.maximum(win[both], second[both])
            flat = torch.bincount(a_ * K + b_, minlength=K * K).reshape(K, K).float().cpu()
            st['shared'] += flat
        if step % spec['check_every'] == 0:
            with torch.no_grad():
                u = torch.bincount(net.all_values(pool).argmin(1), minlength=K).float() / pool.shape[0]
            res = (st['res'] / st['resn'].clamp_min(1)).numpy()
            ev = []
            for k in net.alive_units():
                if k == 0:
                    continue
                streak[k] = streak[k] + 1 if u[k] < spec['dead_frac'] else 0
                if streak[k] >= spec['patience']:
                    if n_rev[k] >= spec['max_revive']:
                        net.alive[k] = 0.0
                        ev.append(('prune', k))
                    else:
                        cands = [j for j in net.alive_units() if j != k]
                        src = max(cands, key=lambda j: res[j])
                        if spec['split']:
                            net.copy_unit(k, src, spec['split_noise'], 0.0, pool)
                            v = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                            splits[k] = (src, v, step + int(spec['warm']))
                        else:
                            net.copy_unit(k, src, spec['revive_noise'], spec['revive_raise'], pool)
                        for p in net.unit_params(k):
                            if p in opt.state:
                                for key in ('exp_avg', 'exp_avg_sq'):
                                    opt.state[p][key].zero_()
                        n_rev[k] += 1
                        ev.append(('revive', k, src) + ((splits[k][1],) if k in splits else ()))
                    streak[k] = 0
            sh, du = st['shared'].numpy(), st['dup'].numpy()
            for a in net.alive_units():
                for b in net.alive_units():
                    if b > a and sh[a, b] >= spec['merge_min_shared'] and du[a, b] / sh[a, b] > spec['merge_frac']:
                        victim = b if (a == 0 or u[a] >= u[b]) else a
                        if victim != 0 and net.alive[victim] > 0.5:
                            net.alive[victim] = 0.0
                            ev.append(('merge', a, b, victim, round(float(du[a, b] / sh[a, b]), 2)))
            rec = dict(step=step, tau=tau, usage=[round(float(x), 4) for x in u], alive=[int(x) for x in net.alive],
                       near=st['near'] / st['n'], neardup=st['neardup'] / max(st['near'], 1), events=ev,
                       loss=float(loss), dec=float(L_dec), sheet=float(L_sheet))
            checks.append(rec)
            events += [(step,) + e for e in ev]
            if log:
                print('  step {:5d} tau {:.4f} usage {} alive {} near {:.3f} dup {:.2f} loss {:.2e} {}'.format(
                    step, tau, rec['usage'], rec['alive'], rec['near'], rec['neardup'], rec['loss'], ev or ''), flush=True)
            st = dict(win=torch.zeros(K), res=torch.zeros(K), resn=torch.zeros(K), shared=torch.zeros(K, K),
                      dup=torch.zeros(K, K), n=0, near=0, neardup=0)
    return net, checks, events


@torch.no_grad()
def evaluate(net, case, spec, device):
    if case == '0a':
        d = torch.linspace(0, TWO_PI, 7201, device=device)[1:-1]
    else:
        d = torch.cat([torch.linspace(0, math.pi / 2, 1801, device=device)[1:-1],
                       torch.linspace(2 * math.pi / 3, TWO_PI, 4801, device=device)[1:-1]])
    Tt, T2t = truth(d, case)
    V = net.all_values(d)
    T, win = V.min(1)
    rel = ((T - Tt).abs() / Tt).cpu().numpy()
    out = {'median_rel_err_pct': 100 * float(np.median(rel)), 'p95_rel_err_pct': 100 * float(np.percentile(rel, 95))}
    with torch.no_grad():
        u = torch.bincount(win, minlength=net.K).float() / d.shape[0]
    out['usage'] = [round(float(x), 4) for x in u]
    out['surviving'] = int((u > 0.01).sum())
    out['alive'] = [int(x) for x in net.alive]
    # ties: argmin changes between neighbouring grid points (same arc)
    dd = d.cpu().numpy()
    w = win.cpu().numpy()
    x = d.clone().requires_grad_(True)
    with torch.enable_grad():
        Vg = net.all_values(x)
        grads = []
        for k in range(net.K):
            if net.alive[k] > 0.5:
                grads.append(torch.autograd.grad(Vg[:, k].sum(), x, retain_graph=True)[0])
            else:
                grads.append(torch.zeros_like(x))
        G = torch.stack(grads, 1).detach().cpu().numpy()
    ridges, handovers = [], []
    for i in range(len(dd) - 1):
        if w[i] != w[i + 1] and abs(dd[i + 1] - dd[i]) < 0.01:
            c = np.sign(G[i, w[i]] * G[i, w[i + 1]])
            (handovers if c > 0 else ridges).append(0.5 * (dd[i] + dd[i + 1]))
    out['ridge_cuts_deg'] = [round(math.degrees(v), 2) for v in ridges]
    out['handovers_deg'] = [round(math.degrees(v), 2) for v in handovers]
    if case == '0a':
        out['cut_err_deg'] = min([abs(math.degrees(v) - 180.0) for v in ridges]) if ridges else None
        band = ((d - math.pi).abs() < 0.05).cpu().numpy()
        out['ridge_band_median_pct'] = 100 * float(np.median(rel[band]))
        out['ridge_band_p95_pct'] = 100 * float(np.percentile(rel[band], 95))
        srt = V.sort(1).values
        if net.K > 1 and torch.isfinite(srt[:, 1]).all():
            rel2 = ((srt[:, 1] - T2t).abs() / T2t).cpu().numpy()
            out['second_best_band_median_pct'] = 100 * float(np.median(rel2[band]))
        gmin = np.take_along_axis(G, w[:, None], 1)[:, 0]
        out['eik_resid_band_median'] = float(np.median(np.abs(np.abs(gmin[band]) - 1.0)))
    out['T_mean_ratio'] = float((T / Tt).mean())
    return out, (dd, V.cpu().numpy(), w, Tt.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_mb/stage0')
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--K', nargs='*', type=int, default=[4, 1])
    ap.add_argument('--cases', nargs='*', default=['0a', '0b'])
    ap.add_argument('--spec', nargs='*', default=[])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    spec = dict(delta=0.1, tau0=0.1, tau_end=1e-3, anneal_frac=0.6, w_dec=1.0, w_sheet=1.0, w_dist=0.1, eta=0.02,
                gamma=0.95, near_cut_frac=0.2, check_every=500, dead_frac=0.01, patience=3, revive_noise=0.1,
                revive_raise=0.05, max_revive=2, merge_frac=0.5, merge_min_shared=20,
                split=1, split_noise=0.02, warm=600, lr=3e-3)   # tau_ref = 1 (T ~ pi); split: gradient-sign split revival
    for s in a.spec:
        k, v = s.split('=')
        spec[k] = float(v)
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for case in a.cases:
        for K in a.K:
            for seed in a.seeds:
                tag = '{}_K{}_s{}'.format(case, K, seed)
                print('== ' + tag, flush=True)
                net, checks, events = train(case, K, seed, a.steps, spec, a.device, not a.quiet)
                res, curves = evaluate(net, case, spec, a.device)
                res['events'] = events
                results[tag] = res
                print('   ' + json.dumps({k: v for k, v in res.items() if k != 'events'}), flush=True)
                np.savez(os.path.join(a.out, tag + '_curves.npz'), d=curves[0], V=curves[1], win=curves[2], T=curves[3])
                with open(os.path.join(a.out, tag + '_checks.json'), 'w') as fh:
                    json.dump(checks, fh, indent=1)
    with open(os.path.join(a.out, 'stage0_results.json'), 'w') as fh:
        json.dump({'spec': spec, 'steps': a.steps, 'results': results}, fh, indent=1)


if __name__ == '__main__':
    sys.exit(main())
