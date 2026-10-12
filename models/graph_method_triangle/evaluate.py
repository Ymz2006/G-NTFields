"""Evaluation of learned SE(3) travel-time fields (spec page 7).

    python -m models.graph_method_triangle.evaluate --ckpt Experiments/graph_method_triangle/<run>/best.pt \
        --out results/graph_method_triangle/<run> [--n_plan 1000] [--device cuda:0]
    python -m models.graph_method_triangle.evaluate --analytic --out results/graph_method_triangle/analytic

Writes metrics.json, metrics.csv (flat key/value) and plots (cut-locus slice,
error vs. relative angle). All errors are measured against the independent
atan2 oracle in float64.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.append('.')

import torch

from models.graph_method_triangle.data.poses import build_heldout_suites, suites_to, PoseSampler
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3, tangents
from models.graph_method_triangle.losses import eikonal
from models.graph_method_triangle.planning import rollout
from models.graph_method_triangle import model_function_graph as mf
from models.graph_method_triangle import model_network_graph as nets
from models.graph_method_triangle import model_train_graph as mt

ANGLE_BINS = [(0, 90), (90, 150), (150, 170), (170, 179), (179, 180.0001)]
TIME_BINS = [(0, 0.05), (0.05, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]


def q(x, p):
    return x.quantile(p).item() if x.numel() else float('nan')


def stats(err, prefix):
    if err.numel() == 0:
        return {prefix + '_n': 0}
    return {prefix + '_n': int(err.numel()), prefix + '_mean': err.mean().item(),
            prefix + '_p50': q(err, 0.5), prefix + '_p95': q(err, 0.95), prefix + '_max': err.max().item()}


def values(model, tx, Rx, ty, Ry, chunk=32768):
    gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
    out = []
    with torch.no_grad():
        for i in range(0, gx.shape[0], chunk):
            out.append(model.value(gx[i:i + chunk], gy[i:i + chunk]))
    return torch.cat(out).double()


def grads(model, tx, Rx, ty, Ry, s_t, s_r, chunk=8192):
    gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
    cxs, cys = [], []
    for i in range(0, gx.shape[0], chunk):
        _, cx, cy = tangents.tangent_gradients(model.value, gx[i:i + chunk], gy[i:i + chunk], s_t, s_r,
                                               create_graph=False)
        cxs.append(cx.detach().double())
        cys.append(cy.detach().double())
    return torch.cat(cxs), torch.cat(cys)


def field_metrics(model, suites, s_t, s_r, tau_ref, diag_band, pi_band):
    M = {}
    pool = {k: [] for k in ('T', 'Ts', 'ang', 'b')}
    for name, s in suites.items():
        tx, Rx, ty, Ry = s['tx'], s['Rx'], s['ty'], s['Ry']
        T = values(model, tx, Rx, ty, Ry)
        T64 = [v.double() for v in (tx, Rx, ty, Ry)]
        Ts = oracle.time_star(*T64, s_t, s_r)
        ang = oracle.relative_angle(T64[1], T64[3]) * 180 / math.pi
        err = (T - Ts).abs()
        M[name + '_nonfinite'] = int((~torch.isfinite(T)).sum())
        M.update(stats(err, name + '_abs'))
        M[name + '_rmse'] = torch.sqrt((err ** 2).mean()).item()
        rel = mt.relative_errors(T, Ts, tau_ref)
        M.update(stats(rel, name + '_rel'))
        if name in ('uniform', 'mixture', 'near_pi'):
            pool['T'].append(T); pool['Ts'].append(Ts); pool['ang'].append(ang)
        if name in ('uniform', 'mixture'):
            # symmetry
            Tsw = values(model, ty, Ry, tx, Rx)
            M.update(stats((T - Tsw).abs(), name + '_sym'))
            # latent errors (only interpretable for supervised graph models)
            if getattr(model, 'has_latent', False):
                with torch.no_grad():
                    a, b = model.predict_latent(se3.features(tx, Rx), se3.features(ty, Ry))
                a_s, b_s = oracle.latent_star(*T64)
                M.update(stats((a.double() - a_s).abs(), name + '_latent_a'))
                M.update(stats((b.double() - b_s.clamp(0, 1)).abs(), name + '_latent_b'))
            # endpoint PDE residuals, regular vs excluded bands
            cx, cy = grads(model, tx, Rx, ty, Ry, s_t, s_r)
            rx = ((cx ** 2).sum(-1) - 1).abs()
            ry = ((cy ** 2).sum(-1) - 1).abs()
            reg = eikonal.regular_mask(*T64, s_t, s_r, tau_ref, diag_band, pi_band)
            M.update(stats(rx[reg], name + '_eik_x_regular'))
            M.update(stats(ry[reg], name + '_eik_y_regular'))
            M[name + '_eik_nonfinite'] = int((~torch.isfinite(rx) | ~torch.isfinite(ry)).sum())
            for k, band in enumerate((1e-2, 1e-3)):
                inband = (1 - oracle.latent_star(*T64)[1]) < band
                M.update(stats(rx[inband], '{}_eik_x_piband{:.0e}'.format(name, band)))
    if 'diagonal' in suites:
        s = suites['diagonal']
        T = values(model, s['tx'], s['Rx'], s['ty'], s['Ry'])
        M.update(stats(T.abs(), 'goal_T'))
    # stratification by relative angle / by goal time
    T, Ts, ang = torch.cat(pool['T']), torch.cat(pool['Ts']), torch.cat(pool['ang'])
    for lo, hi in ANGLE_BINS:
        m = (ang >= lo) & (ang < hi)
        rel = ((T - Ts).abs() / Ts.clamp_min(1e-12))[m & (Ts >= 0.05 * tau_ref)]
        M.update(stats(rel, 'angle_{}-{}_rel'.format(lo, int(hi))))
        M.update(stats((T - Ts).abs()[m], 'angle_{}-{}_abs'.format(lo, int(hi))))
    for lo, hi in TIME_BINS:
        m = (Ts >= lo * tau_ref) & (Ts < hi * tau_ref)
        M.update(stats((T - Ts).abs()[m], 'time_{}-{}_abs'.format(lo, hi)))
    return M


def triangle_metrics(model, s_t, s_r, n=10000, seed=31337, device='cpu', dtype=torch.float32):
    s = PoseSampler(seed=seed, device=device, dtype=dtype)
    x, y, z = s.uniform_poses(n), s.uniform_poses(n), s.uniform_poses(n)
    Txy = values(model, *x, *y)
    Txz = values(model, *x, *z)
    Tzy = values(model, *z, *y)
    viol = (Txy - Txz - Tzy).clamp_min(0)
    return {'triangle_viol_frac': (viol > 1e-4).double().mean().item(), 'triangle_viol_mean': viol.mean().item(),
            'triangle_viol_max': viol.max().item()}


def planning_metrics(model, suite, s_t, s_r, n_plan, h, max_steps, n_cand, seed=0):
    M = {}
    tx, Rx, ty, Ry = (suite[k][:n_plan] for k in ('tx', 'Rx', 'ty', 'Ry'))
    Ts = oracle.time_star(tx.double(), Rx.double(), ty.double(), Ry.double(), s_t, s_r)
    for mode in ('gradient', 'sampling'):
        res = rollout.rollout(model, tx, Rx, ty, Ry, s_t, s_r, mode=mode, h=h, max_steps=max_steps,
                              n_cand=n_cand, seed=seed)
        succ = res['success']
        ratio = (res['cost'].double() / Ts.clamp_min(1e-12))[succ]
        p = 'plan_' + mode
        M[p + '_success'] = succ.double().mean().item()
        M.update(stats(ratio, p + '_cost_ratio'))
        M.update(stats(res['err_t'].double(), p + '_err_t'))
        M.update(stats(res['err_r'].double(), p + '_err_r'))
        M[p + '_steps_mean'] = res['steps'].double().mean().item()
        M[p + '_fallback_frac'] = (res['fallback'] > 0).double().mean().item()
        M[p + '_fallback_steps_mean'] = res['fallback'].double().mean().item()
        M[p + '_runtime_s'] = res['runtime']
    return M


def cut_locus_slice(model, s_t, s_r, out_png, device, dtype):
    """T along R_y = R_x exp(theta n) for theta in [0, 2 pi): crosses the cut locus at pi."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = 721
    th = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    tx = torch.zeros(n, 3, dtype=torch.float64)
    ty = torch.full((n, 3), 0.3, dtype=torch.float64)
    Rx = se3.so3_exp(torch.tensor([[0.3, -0.5, 0.2]], dtype=torch.float64)).expand(n, 3, 3)
    axis = torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64)
    Ry = Rx @ se3.so3_exp(th[:, None] * axis)
    Ts = oracle.time_star(tx, Rx, ty, Ry, s_t, s_r)
    _, bs = oracle.latent_star(tx, Rx, ty, Ry)
    cast = lambda v: v.to(device=device, dtype=dtype)
    T = values(model, cast(tx), cast(Rx), cast(ty), cast(Ry)).cpu()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(th.numpy(), Ts.numpy(), 'k-', label='oracle T*')
    ax[0].plot(th.numpy(), T.numpy(), 'r--', label='learned T')
    ax[0].axvline(math.pi, color='gray', lw=0.5)
    ax[0].set_xlabel('rotation about fixed axis (rad)'); ax[0].set_ylabel('time'); ax[0].legend()
    ax[1].plot(th.numpy(), bs.numpy(), 'k-', label='analytic b*')
    if getattr(model, 'has_latent', False):
        with torch.no_grad():
            _, b = model.predict_latent(se3.features(cast(tx), cast(Rx)), se3.features(cast(ty), cast(Ry)))
        ax[1].plot(th.numpy(), b.double().cpu().numpy(), 'r--', label='learned b')
    ax[1].axvline(math.pi, color='gray', lw=0.5)
    ax[1].set_xlabel('rotation about fixed axis (rad)'); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    m = (th - math.pi).abs() < 0.05
    return {'slice_abs_max': (T - Ts).abs().max().item(), 'slice_abs_near_pi_max': (T - Ts).abs()[m].max().item()}


def evaluate(model, s_t, s_r, tau_ref, device, dtype, out, test_seed=2002, n_main=10000, n_plan=1000,
             plan_h=0.02, plan_max_steps=600, n_cand=128, diag_band=1e-3, pi_band=1e-4, sampler_kw=None):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    suites = suites_to(build_heldout_suites(test_seed, n_main=n_main, n_suite=1000, **(sampler_kw or {})),
                       device, dtype)
    M = field_metrics(model, suites, s_t, s_r, tau_ref, diag_band, pi_band)
    M.update(triangle_metrics(model, s_t, s_r, device=device, dtype=dtype))
    M.update(cut_locus_slice(model, s_t, s_r, os.path.join(out, 'cut_locus_slice.png'), device, dtype))
    if n_plan > 0:
        M.update(planning_metrics(model, suites['uniform'], s_t, s_r, n_plan, plan_h * tau_ref,
                                  plan_max_steps, n_cand))
    M['eval_time_s'] = time.time() - t0
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(M, f, indent=1)
    with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k in sorted(M):
            w.writerow([k, M[k]])
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--analytic', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_plan', type=int, default=1000)
    p.add_argument('--n_main', type=int, default=10000)
    p.add_argument('--use_target', action='store_true', help='evaluate the EMA target network')
    a = p.parse_args()
    if a.analytic:
        cfg = mf.default_config(device=a.device)
        model = nets.build_network('analytic').to(a.device)
        dtype = torch.float64
        s_t, s_r, tau_ref = cfg['s_t'], cfg['s_r'], cfg['L_ref'] / cfg['s_t']
        info = {'arch': 'analytic'}
    else:
        f, ckpt = mt.load_function(a.ckpt, device=a.device, use_target=a.use_target)
        model, cfg = f.network, f.cfg
        dtype = f.dtype
        s_t, s_r, tau_ref = f.s_t, f.s_r, f.tau_ref
        info = {'arch': cfg['arch'], 'mode': cfg['mode'], 'step': ckpt['step'], 'ckpt': a.ckpt,
                'params': nets.count_parameters(model), 'n_anchor_train': ckpt.get('n_anchor')}
    M = evaluate(model, s_t, s_r, tau_ref, a.device, dtype, a.out, test_seed=cfg['test_seed'],
                 n_main=a.n_main, n_plan=a.n_plan, diag_band=cfg['diag_band'], pi_band=cfg['pi_band'],
                 sampler_kw=dict(box=cfg['box'], s_t=s_t, s_r=s_r))
    M.update(info)
    with open(os.path.join(a.out, 'metrics.json'), 'w') as fh:
        json.dump(M, fh, indent=1)
    keys = ['uniform_rel_p50', 'uniform_rel_p95', 'mixture_rel_p50', 'mixture_rel_p95', 'exact_pi_abs_mean',
            'uniform_eik_x_regular_p50', 'plan_gradient_success', 'plan_gradient_cost_ratio_p50',
            'plan_sampling_success', 'plan_sampling_cost_ratio_p50']
    print(json.dumps({k: M.get(k) for k in keys}, indent=1))


if __name__ == '__main__':
    main()
