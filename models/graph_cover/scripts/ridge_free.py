"""Stage 1 validation of the ridge detector on the obstacle-free half-turn ridge.

For anchors (goal p, R_g; random unit axis n) the start is x(phi) = (p + d m, R_g exp(phi n)): the true field
has a concave corner at phi = pi (the two rotation routes tie). The detector (models/graph_cover/ridge.py)
is run along phi for (a) the exact hard-min field and the soft-min model at several widths tau, where the
rounding is known, and (b) trained obstacle-free checkpoints (graph_method / graph_method_triangle format).
Per model: ridge location bias (phi* - pi, degrees) from the zero of the cross derivative r = c . v inside
the band, band width (phi-range with S|c| < band), lambda_min at the ridge, and the fraction of anchors on
which the detector fires. Writes JSON and a figure.

    python models/graph_cover/scripts/ridge_free.py --out results/graph_cover/ridge_free \
        --ckpt metric=Experiments/graph_method_final/metric_best_recipe_s0/best.pt ... [--anchors 48] [--offset 0.2]
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')

import numpy as np
import torch

from models.graph_cover import ridge
from models.graph_method_triangle import model_train_graph as mt
from models.graph_method_triangle.geometry import se3

TWO_PI = 2 * math.pi


def anchors(n, seed, box, offset, device, dtype):
    g = torch.Generator().manual_seed(seed)
    p = (torch.rand(n, 3, generator=g, dtype=dtype) * 2 - 1) * box * 0.6
    q = torch.randn(n, 4, generator=g, dtype=dtype)
    q = q / q.norm(dim=1, keepdim=True)
    Rg = se3.quat_to_matrix(q)
    axis = torch.randn(n, 3, generator=g, dtype=dtype)
    axis = axis / axis.norm(dim=1, keepdim=True)
    m = torch.randn(n, 3, generator=g, dtype=dtype)
    m = m / m.norm(dim=1, keepdim=True)
    return p.to(device), Rg.to(device), axis.to(device), (p + offset * m).to(device)


def scan_model(value_fn, p, Rg, axis, p0, s_t, s_r, phis, band, kappa):
    """Run the detector along phi for every anchor; returns per-anchor stats and the curves of anchor 0."""
    n = p.shape[0]
    res = {'bias_deg': [], 'band_deg': [], 'lam_at_ridge': [], 'cn_at_ridge': [], 'fired': 0, 'T_err_pi': []}
    curves = None
    for i in range(n):
        Rg_i, ax_i = Rg[i:i + 1], axis[i:i + 1]
        orient = torch.cat([torch.zeros(3, dtype=p.dtype, device=p.device), ax_i[0]])[None]

        def pose_fn(ph):
            R = Rg_i.expand(len(ph), -1, -1) @ se3.so3_exp(ph[:, None] * ax_i.expand(len(ph), -1))
            return p0[i:i + 1].expand(len(ph), -1), R

        sc = ridge.scan_line(value_fn, pose_fn, phis, p[i:i + 1], Rg_i, s_t, s_r, orient=orient)
        mask = (sc['cn'] < band) & (sc['lam_min'] < -kappa)
        zs = ridge.zero_crossings(phis, sc['r'], mask)
        if zs:
            z = min(zs, key=lambda x: abs(x - math.pi))
            res['fired'] += 1
            res['bias_deg'].append((z - math.pi) * 180 / math.pi)
            j = int(torch.argmin((phis - z).abs()))
            res['lam_at_ridge'].append(float(sc['lam_min'][j]))
            res['cn_at_ridge'].append(float(sc['cn'][j]))
        inb = (sc['cn'] < band).nonzero().squeeze(-1)
        res['band_deg'].append(float((phis[inb.max()] - phis[inb.min()]) * 180 / math.pi) if inb.numel() else 0.0)
        # value error at exact pi vs the hard-min truth
        j = int(torch.argmin((phis - math.pi).abs()))
        tx, Rx = pose_fn(phis[j:j + 1])
        Tt = ridge.hard_min_free(se3.features(tx, Rx), se3.features(p[i:i + 1], Rg_i), s_t, s_r)
        res['T_err_pi'].append(float((sc['T'][j] - Tt[0]) / Tt[0]))
        if i == 0:
            curves = {k: sc[k].cpu().tolist() for k in ('T', 'cn', 'lam_min', 'r')}
    out = {'n': n, 'fired_frac': res['fired'] / n}
    for k in ('bias_deg', 'band_deg', 'lam_at_ridge', 'cn_at_ridge', 'T_err_pi'):
        a = np.array(res[k]) if res[k] else np.array([np.nan])
        out[k] = {'mean': float(np.nanmean(a)), 'median': float(np.nanmedian(a)), 'std': float(np.nanstd(a)),
                  'absmax': float(np.nanmax(np.abs(a)))}
    return out, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', action='append', default=[], metavar='TAG=PATH')
    ap.add_argument('--taus', default='0,0.01,0.03,0.1')
    ap.add_argument('--anchors', type=int, default=48)
    ap.add_argument('--offset', type=float, default=0.2, help='translation offset of the start from the goal')
    ap.add_argument('--span', type=float, default=40.0, help='phi scan half-width around pi (degrees)')
    ap.add_argument('--n_phi', type=int, default=321)
    ap.add_argument('--band', type=float, default=0.95)
    ap.add_argument('--kappa', type=float, default=5.0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    span = a.span * math.pi / 180
    results, curves = {}, {}
    # (a) synthetic fields in float64, s_t = 1, s_r = 1 (rotation in radians)
    dt = torch.float64
    phis = torch.linspace(math.pi - span, math.pi + span, a.n_phi, dtype=dt, device=dev)
    p, Rg, axis, p0 = anchors(a.anchors, a.seed, 1.0, a.offset, dev, dt)
    for tau in [float(t) for t in a.taus.split(',')]:
        f = ridge.SoftMinFreeField(tau, 1.0, 1.0)
        tag = 'softmin_tau{:g}'.format(tau) if tau > 0 else 'hardmin'
        results[tag], curves[tag] = scan_model(f.value, p, Rg, axis, p0, 1.0, 1.0, phis, a.band, a.kappa)
        print(tag, json.dumps({k: (round(v['median'], 4) if isinstance(v, dict) else v) for k, v in results[tag].items()}), flush=True)
    # (b) trained checkpoints (their own scales and dtype)
    for spec in a.ckpt:
        tag, path = spec.split('=', 1)
        fn, ck = mt.load_function(path, device=dev)
        model, cfg = fn.network, fn.cfg
        s_t, s_r = fn.s_t, fn.s_r
        dtm = fn.dtype
        phis_m = phis.to(dtm)
        pm, Rgm, axm, p0m = [x.to(dtm) for x in anchors(a.anchors, a.seed, cfg['box'], a.offset, dev, torch.float64)]
        results[tag], curves[tag] = scan_model(model.value, pm, Rgm, axm, p0m, s_t, s_r, phis_m, a.band, a.kappa)
        results[tag]['arch'] = cfg['arch']
        print(tag, json.dumps({k: (round(v['median'], 4) if isinstance(v, dict) else v) for k, v in results[tag].items()}), flush=True)
    with open(os.path.join(a.out, 'ridge_free.json'), 'w') as fh:
        json.dump({'args': vars(a), 'phis_deg': (phis * 180 / math.pi).cpu().tolist(), 'results': results,
                   'curves': curves}, fh, indent=1)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tags = list(curves.keys())
        fig, axs = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
        x = (phis * 180 / math.pi).cpu().numpy()
        for tag in tags:
            cv = curves[tag]
            axs[0].plot(x, cv['T'], label=tag)
            axs[1].plot(x, cv['cn'])
            axs[2].plot(x, np.clip(cv['lam_min'], -200, 50))
            axs[3].plot(x, cv['r'])
        axs[0].set_ylabel('T'); axs[1].set_ylabel('|c|'); axs[2].set_ylabel('lambda_min (clipped)'); axs[3].set_ylabel('r = c . v')
        axs[3].axhline(0, color='k', lw=0.5); axs[3].set_xlabel('phi (deg), ridge at 180')
        axs[1].axhline(a.band, color='k', lw=0.5, ls='--'); axs[2].axhline(-a.kappa, color='k', lw=0.5, ls='--')
        axs[0].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'ridge_free.png'), dpi=130)
    except Exception as e:  # noqa: BLE001
        print('figure skipped:', e)


if __name__ == '__main__':
    main()
