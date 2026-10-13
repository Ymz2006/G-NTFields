"""Path audit for the chart-vs-Lie benchmark: what the field predicts along the executed MPPI path.

For every executed path x_0 .. x_K (up to the SE(3) convergence point, or the chart freeze, or the full
budget) with fixed goal y, and for the field's OWN speed formulation f (combined / iso / aniso):

  ground truth  Scene.clearance (Lie-Shape-NTFields placement evaluator, the quantity the training data was
                labelled with) -> dist, angle -> warped S_d, S_a, S_comb (analysis.common.target_speeds)
  gradients     chart g = d tau / d(t, r) and Lie c = body-frame derivative, both w.r.t. the moving endpoint
  Eikonal       N_lie = f(S, c), N_chart = f(S, g); 1 / N is predicted / true speed in that metric
  steps         chart length ds_chart = |x_{k+1} - x_k|, Lie length ds_lie = sqrt(|R^T dt|^2 + |log(R^T R')|^2 / (2 pi)^2),
                physical cost increments dc_lie = ds_lie / S (aniso: sqrt(|dv|^2 / S_d^2 + |dw|^2 / S_a^2)) and the
                chart analogue dc_chart, with 1 / S averaged over the two ends of the step; dtau = tau_{k+1} - tau_k

Per-step rows are returned as flat arrays (one row per waypoint; step quantities NaN on the last waypoint) so
the aggregator can bin them by |rotvec|, clearance, rotation share etc. Per-case summaries: tau_start against the
physical path cost (Lie and chart), medians of N along the path, local slope ratios -dtau / dc.

Everything here runs after planning (Scene() reseeds numpy / torch) and is pure bookkeeping: the planner, the
collision test and the success labels are untouched.
"""
import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi
EPS = 1e-8


def eikonal_N(grad, Sd, Sa, Sc, speed_model):
    v2 = (grad[:, :3] ** 2).sum(1)
    w2 = (grad[:, 3:] ** 2).sum(1)
    if speed_model == 'combined':
        return Sc * torch.sqrt(v2 + w2 + EPS)
    if speed_model == 'iso':
        return Sd * torch.sqrt(v2 + w2 + EPS)
    if speed_model == 'aniso':
        return torch.sqrt(Sd ** 2 * v2 + Sa ** 2 * w2 + EPS)
    raise ValueError(speed_model)


def cost_increment(dv2, dw2, Sd, Sa, Sc, speed_model):
    """Cost of a step with squared translation / rotation lengths dv2, dw2 at speeds (Sd, Sa, Sc)."""
    if speed_model == 'combined':
        return torch.sqrt(dv2 + dw2) / Sc
    if speed_model == 'iso':
        return torch.sqrt(dv2 + dw2) / Sd
    if speed_model == 'aniso':
        return torch.sqrt(dv2 / Sd ** 2 + dw2 / Sa ** 2)
    raise ValueError(speed_model)


def gradients(network, sib, coords, chunk):
    """coords (N, 12) float32 on device -> tau (N,), chart g (N, 6), Lie c (N, 6) w.r.t. the first endpoint."""
    taus, gs, cs = [], [], []
    for i in range(0, coords.shape[0], chunk):
        x = coords[i:i + chunk]
        tau, _, Xp = network.out(x)
        d = torch.autograd.grad(tau, Xp, torch.ones_like(tau), only_inputs=True)[0]
        gs.append(d[:, :6].detach())
        g0 = network.features(x[:, :6]).detach()
        g1 = network.features(x[:, 6:]).detach()
        T, cx, _ = sib.tangents.tangent_gradients(network.net.value, g0, g1, 1.0, TWO_PI, create_graph=False)
        taus.append(T.detach())
        cs.append(cx.detach())
    return torch.cat(taus), torch.cat(gs), torch.cat(cs)


def lie_steps(P, sib):
    """P (K+1, 6) float64 chart path -> per-step squared body translation / rotation lengths (turns^2)."""
    R = sib.se3.so3_exp(TWO_PI * P[:, 3:6])
    dt = P[1:, :3] - P[:-1, :3]
    dv = (R[:-1].transpose(-1, -2) @ dt[..., None])[..., 0]
    dw = sib.se3.so3_log(R[:-1].transpose(-1, -2) @ R[1:]) / TWO_PI
    return (dv ** 2).sum(1), (dw ** 2).sum(1)


def audit_paths(records, network, sib, speed_model, scene, target_speeds, device, chunk=8192, clear_chunk=200):
    """records: list of dicts with '_full' (n_rec, 6) path, '_kend' (number of waypoints to use) and 'goal' (6,).
    Returns (rows: dict of flat np arrays, one per waypoint; summaries: list of dicts aligned with records)."""
    paths, owner = [], []
    for k, rec in enumerate(records):
        P = np.asarray(rec['_full'][:rec['_kend']], dtype=np.float64)
        # drop frozen duplicates (episodes that converged keep repeating their last pose)
        if len(P) > 1:
            keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9])
            P = P[keep]
        # the Lie sampler appends the goal pose to every path (lie_mppi chain_b); for an episode that did not
        # converge that is a jump, not a step: drop it
        goal = np.asarray(rec['goal'], dtype=np.float64)
        if len(P) > 1 and np.abs(P[-1] - goal).max() < 1e-6 and np.linalg.norm(P[-1] - P[-2]) > 0.05:
            P = P[:-1]
        paths.append(P)
        owner.append(np.full(len(P), k))
    owner = np.concatenate(owner)
    W = np.concatenate(paths)                                              # (M, 6)
    goals = np.concatenate([np.repeat(np.asarray(rec['goal'], dtype=np.float64)[None], len(P), 0)
                            for rec, P in zip(records, paths)])
    # ground truth clearance and speeds
    c = scene.clearance(torch.tensor(W, dtype=torch.float32), chunk=clear_chunk)
    Sd, Sa, Sc = target_speeds(c['dist'], c['angle'])
    Sd, Sa, Sc = Sd.float(), Sa.float(), Sc.float()
    free = c['free'].numpy()
    # field: tau and both gradients w.r.t. the moving endpoint
    coords = torch.tensor(np.concatenate([W, goals], 1), dtype=torch.float32, device=device)
    tau, g, cg = gradients(network, sib, coords, chunk)
    tau, g, cg = tau.cpu(), g.cpu(), cg.cpu()
    N_lie = eikonal_N(cg, Sd, Sa, Sc, speed_model)
    N_chart = eikonal_N(g, Sd, Sa, Sc, speed_model)
    M = W.shape[0]
    rows = {
        'case': np.concatenate([np.full(len(P), rec['case']) for rec, P in zip(records, paths)]).astype(np.int32),
        'k': np.concatenate([np.arange(len(P)) for P in paths]).astype(np.int32),
        'theta_deg': (np.linalg.norm(W[:, 3:6], axis=1) * 360.0).astype(np.float32),
        'free': free, 'dist': c['dist'].numpy().astype(np.float32), 'angle': c['angle'].numpy().astype(np.float32),
        'Sd': Sd.numpy(), 'Sa': Sa.numpy(), 'Sc': Sc.numpy(),
        'tau': tau.numpy().astype(np.float32),
        'N_lie': N_lie.numpy().astype(np.float32), 'N_chart': N_chart.numpy().astype(np.float32),
        'cv': cg[:, :3].norm(dim=1).numpy().astype(np.float32), 'cw': cg[:, 3:].norm(dim=1).numpy().astype(np.float32),
        'gv': g[:, :3].norm(dim=1).numpy().astype(np.float32), 'gw': g[:, 3:].norm(dim=1).numpy().astype(np.float32),
    }
    for key in ('ds_lie', 'ds_chart', 'dc_lie', 'dc_chart', 'dtau', 'rotshare'):
        rows[key] = np.full(M, np.nan, dtype=np.float32)
    summaries = []
    off = 0
    for rec, P in zip(records, paths):
        n = len(P)
        sl = slice(off, off + n)
        summ = {'audit_n_waypoints': int(n), 'audit_frac_free': float(free[sl].mean()),
                'audit_frac_theta_gt135': float((rows['theta_deg'][sl] > 135).mean())}
        if n > 1:
            Pt = torch.tensor(P, dtype=torch.float64)
            dv2, dw2 = lie_steps(Pt, sib)
            dch = Pt[1:] - Pt[:-1]
            # chart rotation step: the Lie sampler writes canonical rotation vectors, which flip across the seam
            # (+pi n -> -pi n); take the shorter of the two representations of the next pose
            r_next = Pt[1:, 3:]
            nn = r_next.norm(dim=1, keepdim=True)
            r_alt = r_next - r_next / nn.clamp_min(1e-12)
            dr2 = torch.minimum((dch[:, 3:] ** 2).sum(1), ((r_alt - Pt[:-1, 3:]) ** 2).sum(1))
            dt2 = (dch[:, :3] ** 2).sum(1)
            inv = lambda S: 0.5 * (1.0 / S[sl][:-1].double() + 1.0 / S[sl][1:].double())  # noqa: E731
            # speeds averaged (harmonically) over the two ends of the step
            Sd_m, Sa_m, Sc_m = 1.0 / inv(Sd), 1.0 / inv(Sa), 1.0 / inv(Sc)
            dc_l = cost_increment(dv2, dw2, Sd_m, Sa_m, Sc_m, speed_model)
            dc_c = cost_increment(dt2, dr2, Sd_m, Sa_m, Sc_m, speed_model)
            dtau = tau[sl][1:].double() - tau[sl][:-1].double()
            ds_l = torch.sqrt(dv2 + dw2)
            rows['ds_lie'][off:off + n - 1] = ds_l.numpy()
            rows['ds_chart'][off:off + n - 1] = torch.sqrt(dt2 + dr2).numpy()
            rows['dc_lie'][off:off + n - 1] = dc_l.numpy()
            rows['dc_chart'][off:off + n - 1] = dc_c.numpy()
            rows['dtau'][off:off + n - 1] = dtau.numpy()
            rows['rotshare'][off:off + n - 1] = (dw2 / (dv2 + dw2).clamp_min(1e-16)).numpy()
            fr = torch.tensor(free[sl][:-1] & free[sl][1:])
            ok = fr & (dc_l > 0)
            summ.update({
                'tau_start': float(tau[sl][0]), 'tau_end': float(tau[sl][-1]),
                'cost_lie': float(dc_l[fr].sum()), 'cost_chart': float(dc_c[fr].sum()),
                'len_lie': float(ds_l.sum()), 'len_chart': float(torch.sqrt(dt2 + dr2).sum()),
                'slope_lie_med': float((-dtau[ok] / dc_l[ok]).median()) if ok.any() else None,
                'slope_chart_med': float((-dtau[ok] / dc_c[ok]).median()) if ok.any() else None,
                'N_lie_med': float(N_lie[sl].median()), 'N_chart_med': float(N_chart[sl].median()),
                'rotshare_med': float(rows['rotshare'][off:off + n - 1].mean()),
            })
        summaries.append(summ)
        off += n
    return rows, summaries
