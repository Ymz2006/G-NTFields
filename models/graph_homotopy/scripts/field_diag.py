"""Diagnostics of a trained homotopy (or any graph_native) field on its training scene.

    python models/graph_homotopy/scripts/field_diag.py --ckpt Experiments/graph_homotopy/T_hrand_s0/latest.pt \
        --dataPath datasets/3dshape/data/Tshape3d_env1 --out results/graph_homotopy/diag/T_hrand_s0.json

* speed calibration: predicted speed 1 / |c| (Lie gradient at the start endpoint of dataset pairs) against the
  true warped clearance speed S_d, per decade of S_d: median log10 ratio per decade and dex = mean |log10 ratio|
  (the quantity that tracked planning success in the cover campaign);
* Eikonal residual statistics on dataset pairs on the selected sheet and on the far sheet (homotopy head);
* sheet statistics on random test-like pairs: fraction with sigma* = -1, relative gap between the sheets, the
  fraction of pairs within 5 % of the ridge, mean |T~_+ - T~_-|;
* property checks on trained weights: max T(p, p), max asymmetry, pairs below T_free, cone slope x speed.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie import lie  # noqa: E402
from models.graph_lie.model_train_lie import warp_speeds  # noqa: E402
from models.graph_native import model_train_metric as md  # noqa: E402
from models.graph_native.model_train_metric import load_dataset  # noqa: E402

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=40000)
    a = ap.parse_args()
    torch.manual_seed(0)
    m = md.Model('/tmp/diag_model', a.dataPath, 6, [0.0] * 6, device=a.device)
    m.load(a.ckpt)
    net = m.network
    head = net.net
    data = load_dataset(a.dataPath)
    idx = torch.randperm(data.shape[0])[:a.n]
    d = data[idx].to(a.device)
    pts = d[:, :12]
    sp = warp_speeds(d, 6, 1.0)
    out = {'ckpt': a.ckpt, 'arch': net.graph.get('arch')}
    # --- speed calibration (base field, selected sheet) in chunks
    pred, true, resid = [], [], []
    for s in range(0, pts.shape[0], 4000):
        p = pts[s:s + 4000]
        T, gx, gy = lie.endpoint_gradients(net, p, 'lie', create_graph=False)
        n = torch.sqrt((gx ** 2).sum(1) + 1e-12)
        pred.append((1.0 / n).detach().cpu())
        true.append(sp['Sd'][s:s + 4000, 0].cpu())
        resid.append(((sp['Sd'][s:s + 4000, 0] * n).sqrt() - 1).pow(2).detach().cpu())
    pred, true, resid = torch.cat(pred).numpy(), torch.cat(true).numpy(), torch.cat(resid).numpy()
    lr = np.log10(pred / true)
    edges = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]
    dec = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (true >= lo) & (true < hi)
        if sel.sum() > 20:
            dec.append({'lo': lo, 'hi': hi, 'n': int(sel.sum()), 'median_log10_ratio': float(np.median(lr[sel])),
                        'median_resid': float(np.median(resid[sel]))})
    out['speed_calibration'] = {'decades': dec, 'dex': float(np.mean([abs(x['median_log10_ratio']) for x in dec])),
                                'median_log10_ratio': float(np.median(lr)), 'median_resid': float(np.median(resid))}
    # --- far-sheet residual (homotopy head)
    if hasattr(head, 'sheet_value'):
        far_res, near_res = [], []
        for s in range(0, min(pts.shape[0], 16000), 4000):
            p = pts[s:s + 4000]
            g0 = net.features(p[:, :6]).detach()
            g1 = net.features(p[:, 6:]).detach()
            sv = head.sheet_values(g0, g1)
            near = torch.where(sv[:, 0] <= sv[:, 1], 1.0, -1.0)
            for sig, store in ((near, near_res), (-near, far_res)):
                T, gx, gy = lie._sib.tangents.tangent_gradients(lambda x, y: head.sheet_value(x, y, sig), g0, g1, 1.0, TWO_PI,
                                                                create_graph=False)
                n = torch.sqrt((gx ** 2).sum(1) + 1e-12)
                store.append(((sp['Sd'][s:s + 4000, 0] * n).sqrt() - 1).pow(2).detach().cpu())
        out['sheet_resid'] = {'near_median': float(torch.cat(near_res).median()), 'far_median': float(torch.cat(far_res).median()),
                              'near_mean': float(torch.cat(near_res).mean()), 'far_mean': float(torch.cat(far_res).mean())}
        # sheet statistics on random pairs of the workspace (uniform in the training box, rotations uniform)
        g = torch.Generator(device='cpu').manual_seed(1)
        n = 20000
        lo, hi = data[:, :3].min(0).values, data[:, :3].max(0).values
        t = lo + (hi - lo) * torch.rand(2 * n, 3, generator=g)
        q = torch.randn(2 * n, 4, generator=g)
        q = q / q.norm(dim=1, keepdim=True)
        from models.graph_method_triangle.geometry import se3
        G = se3.features(t, se3.quat_to_matrix(q)).to(a.device)
        with torch.no_grad():
            sv = head.sheet_values(G[:n], G[n:])
            tmin, tmax = sv.min(1).values, sv.max(1).values
            gap = (tmax - tmin) / tmin.clamp_min(1e-6)
            tf = head.free_time(G[:n], G[n:])
            c = (torch.einsum('ni,ni->n', *[__import__('models.graph_method_triangle.branch_decoder', fromlist=['quat_from_matrix']).quat_from_matrix(x[:, 3:].reshape(-1, 3, 3)) for x in (G[:n], G[n:])]))
            sigma = torch.where(sv[:, 0] <= sv[:, 1], 1.0, -1.0)
            long_way = ((sigma * c) < 0).float().mean().item()
        out['sheets'] = {'frac_near_ridge_5pct': float((gap < 0.05).float().mean()), 'median_gap': float(gap.median()),
                         'long_way_frac': long_way, 'median_T_over_Tfree': float((tmin / tf.clamp_min(1e-6)).median()),
                         'min_T_over_Tfree': float((tmin / tf.clamp_min(1e-6)).min())}
        # properties on trained weights
        with torch.no_grad():
            Tpp = head.value(G[:2000], G[:2000]).abs().max().item()
            asym = (head.value(G[:2000], G[n:n + 2000]) - head.value(G[n:n + 2000], G[:2000])).abs().max().item()
        out['properties'] = {'max_T_pp': Tpp, 'max_asymmetry': asym}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'speed_calibration'}), 'dex', out['speed_calibration']['dex'],
          'decades', [round(x['median_log10_ratio'], 2) for x in dec])


if __name__ == '__main__':
    main()
