"""Cone calibration at the goal: does the slope of the trained field at its goal match the Eikonal limit of the
equation the field was trained on?

For a goal p and unit directions xi, the slope s(p, eps) = (T(p exp(eps xi), p) - T(p, p)) / eps (mean over 12
directions). In the limit x -> y the Eikonal equation S |grad T| = 1 asks for S(p) s(p) = 1, the condition the
end constraint imposes on the conformal branch head (S (1 + rho) = 1). Reported per speed bin: the median and
quartiles of S(p) s(p, eps) at eps = 1e-2, 1e-3, 1e-4, with each field's own speed model and metric:

* graph and branch heads (graph_lie pipeline, iso speed): S = warped S_d, directions = unit body twists of the
  Lie metric |v|^2 + |w / 2 pi|^2;
* the original metric head (--orig; combined speed, chart gradient): S = warped combined speed, directions = unit
  vectors of the chart (t, rotvec / 2 pi), i.e. the metric of its own loss.

    python models/graph_branch/scripts/cone_calibration.py --scene Tshape3d_env1 --out <json> \
        ggram=Experiments/graph_branch/T_ggram_s0/latest.pt orig=orig:/lieshape/Experiments/.../Model_...pt
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_branch import chart_metric  # noqa: E402
from models.graph_branch.scripts.decoder_properties import load, unit_twists, V  # noqa: E402
from models.graph_lie.model_train_lie import warp_speeds  # noqa: E402
from models.graph_native import _sib  # noqa: E402

TWO_PI = 2.0 * math.pi
se3 = _sib.se3
EDGES = [0.0, 0.01, 0.03, 0.1, 0.3, 0.6, 1.01]
EPS = (1e-2, 1e-3, 1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--m', type=int, default=4000)
    ap.add_argument('models', nargs='+', help='label=checkpoint, or label=orig:checkpoint for the original trainer')
    a = ap.parse_args()
    dev = a.device
    torch.set_grad_enabled(False)
    g = torch.Generator().manual_seed(0)
    root = os.path.join('datasets/3dshape', a.scene)
    cols = [np.asarray(np.load(os.path.join(root, f + '.npy'), mmap_mode='r')[:400000:7][: 3 * a.m])
            for f in ('sampled_points', 'speed', 'normal', 'speed_dists', 'speed_angles')]
    d = torch.from_numpy(np.concatenate(cols, 1)).float()
    rows = d[:, :12]
    P = torch.cat([rows[:, :6], rows[:, 6:]], 0)
    W = warp_speeds(d)
    Sd = torch.cat([W['Sd'][:, 0], W['Sd'][:, 1]], 0)
    Sc = torch.cat([W['S'][:, 0], W['S'][:, 1]], 0)
    perm = torch.randperm(P.shape[0], generator=g)[: a.m]
    P, Sd, Sc = P[perm].to(dev), Sd[perm].to(dev), Sc[perm].to(dev)
    tP, RP = P[:, :3], se3.so3_exp(TWO_PI * P[:, 3:6])
    GP = se3.features(tP, RP)
    xi = unit_twists(12, g, dev)
    u = torch.randn(12, 6, generator=g).to(dev)
    u = u / u.norm(dim=1, keepdim=True)
    out = {'scene': a.scene, 'edges': EDGES, 'eps': list(EPS), 'models': {}}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for spec in a.models:
        lab, ck = spec.split('=', 1)
        orig = ck.startswith('orig:')
        ck = ck[5:] if orig else ck
        if lab in out['models'] and out['models'][lab].get('ckpt') == ck:
            print('skip', lab)
            continue
        net = chart_metric.load_original(ck, dev) if orig else load(ck, dev)
        S = Sc if orig else Sd
        res = {'ckpt': ck, 'arch': net.graph['arch'], 'speed': 'combined' if orig else 'iso (S_d)',
               'directions': 'chart unit vectors' if orig else 'Lie unit twists'}
        if orig:
            off = net.net.value_chart(P, P)
        else:
            off = V(net, GP, GP)
        for eps in EPS:
            sl = []
            for j in range(12):
                if orig:
                    T = net.net.value_chart(P + eps * u[j][None], P)
                else:
                    tq, Rq = se3.right_perturb(tP, RP, (eps * xi[j]).expand(len(P), 6))
                    T = V(net, se3.features(tq, Rq), GP)
                sl.append((T - off) / eps)
            s = torch.stack(sl, 1).mean(1)
            prod = (S * s).cpu()
            bins = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                mk = ((S.cpu() >= lo) & (S.cpu() < hi))
                if mk.sum() < 20:
                    bins.append(None)
                    continue
                v = prod[mk]
                bins.append({'n': int(mk.sum()), 'median': float(v.median()), 'q25': float(v.quantile(0.25)),
                             'q75': float(v.quantile(0.75)), 'S_median': float(S.cpu()[mk].median())})
            res['eps_{:g}'.format(eps)] = {'bins': bins, 'median_all': float(prod.median()),
                                           'dex_median': float(torch.log10(prod.clamp_min(1e-12)).abs().median())}
        if getattr(net.net, 'conf', False):
            res['exact_slope_check'] = float((net.net.endpoint_slope(GP) - torch.stack(
                [(V(net, se3.features(*se3.right_perturb(tP, RP, (1e-4 * xi[j]).expand(len(P), 6))), GP) - off) / 1e-4
                 for j in range(12)], 1).mean(1)).abs().median())
        out['models'][lab] = res
        print(lab, json.dumps({k: v['median_all'] for k, v in res.items() if k.startswith('eps_')}))
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as fh:
            json.dump(out, fh, indent=1)


if __name__ == '__main__':
    main()
