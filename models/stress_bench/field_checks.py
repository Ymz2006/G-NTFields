"""Field-level consistency checks of a travel-time field (no planner): identity, symmetry, triangle
inequality on free midpoints, and detour awareness on the blocked / ctrl stress sets.

Loads a checkpoint exactly as plan_eval_graph.py does (--kind graph_native for this repo's heads, --kind metric
for the original NTFields) and queries T(x, y) in the chart (t, rotvec / 2 pi). Reported per checkpoint:

  identity     T(x, x) over the standard set's endpoints (mean, p95)                     [0 for a metric]
  symmetry     |T(x,y) - T(y,x)| / mean(T(x,y), T(y,x)) over the standard pairs (median, p95, max, % > 5 %)
  tri_mid      V = T(x,z) - T(x,y) - T(y,z), y = SE(3) midpoint of the ctrl pairs (straight line free, clearance
               > 0.02): % V > 0 (violations), p95 and max of V / T(x,z)                  [<= 0 for a metric]
  tri_rand     the same with y a random other endpoint of the standard set
  detour       median T(x,z) / d_SE3(x,z) on the blocked set divided by the same median on ctrl
               (d_SE3 = sqrt(|dp|^2 + (theta_geo / 2 pi)^2)); 1 = the field does not see the obstacle in the way

    python /workspace/models/stress_bench/field_checks.py --kind graph_native --checkpoint ... --scene Tshape3d_env1 \
        --out /workspace/results/stress_bench/field/<run>.json      (cwd /lieshape, as plan_eval_graph.py)
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--scene', default='Tshape3d_env1')
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--lie-root', default='/lieshape')
ap.add_argument('--graph-root', default='/workspace')
ap.add_argument('--bench', default='/workspace/datasets/3dshape/data/bench')
args = ap.parse_args()

sys.path.insert(0, args.lie_root)
os.chdir(args.lie_root)
torch.manual_seed(0)
DIM = 6
TWO_PI = 2.0 * math.pi
from analysis.common import so3_exp, so3_log, geodesic_angle  # noqa: E402


def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_pkg('graph_method_triangle', os.path.join(args.graph_root, 'models', 'graph_method_triangle'))
if args.kind == 'graph_native':
    _load_pkg('graph_native', os.path.join(args.graph_root, 'models', 'graph_native'))
    md = importlib.import_module('graph_native.model_train_metric')
    m = md.Model('/tmp/analysis_model', 'datasets/3dshape/data/Tshape3d_env1/', DIM, [0.0] * DIM, device=args.device)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
else:
    from analysis.common import load_model  # noqa: E402
    _, fn = load_model(args.kind, args.checkpoint, args.device)


def T(x, y):
    X = torch.cat([torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)], 1).to(args.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(fn.TravelTimes(X[i:i + 8192]).detach().double().cpu().reshape(-1))
    return torch.cat(out)


def midpoint(q0, q1):
    R0 = so3_exp(TWO_PI * q0[:, 3:].double())
    R1 = so3_exp(TWO_PI * q1[:, 3:].double())
    w = so3_log(R0.transpose(-1, -2) @ R1)
    r = so3_log(R0 @ so3_exp(0.5 * w)) / TWO_PI
    return torch.cat([0.5 * (q0[:, :3] + q1[:, :3]), r.float()], 1)


def d_se3(q0, q1):
    th = geodesic_angle(q0[:, 3:], q1[:, 3:]).double()
    return torch.sqrt(((q1[:, :3] - q0[:, :3]).double() ** 2).sum(1) + (th / TWO_PI) ** 2)


def load_set(name):
    for base in (os.path.join(args.bench, name), os.path.join('testing_data/3dshape', name)):
        p = os.path.join(base, 'sampled_points.npy')
        if os.path.exists(p):
            a = torch.tensor(np.load(p).astype(np.float32))
            return a[:, :6], a[:, 6:]
    return None, None


def pct(v, q):
    return float(np.percentile(v, q))


res = {'kind': args.kind, 'checkpoint': args.checkpoint, 'scene': args.scene}
x, z = load_set(args.scene)
# identity
e = torch.cat([x, z])
ti = T(e, e).numpy()
txz, tzx = T(x, z).numpy(), T(z, x).numpy()
res['identity'] = {'mean': float(ti.mean()), 'p95': pct(ti, 95), 'max': float(ti.max()),
                   'rel_to_median_T': float(ti.mean() / np.median(txz))}
# symmetry
asym = np.abs(txz - tzx) / (0.5 * (txz + tzx))
res['symmetry'] = {'median': float(np.median(asym)), 'p95': pct(asym, 95), 'max': float(asym.max()),
                   'frac_gt_5pct': float((asym > 0.05).mean()), 'n': int(asym.size)}
# triangle, random third point
g = torch.Generator().manual_seed(0)
y = e[torch.randperm(len(e), generator=g)[:len(x)]]
V = txz - T(x, y).numpy() - T(y, z).numpy()
res['tri_rand'] = {'frac_viol': float((V > 1e-6).mean()), 'p95_rel': pct(V / txz, 95), 'max_rel': float((V / txz).max()),
                   'median_ratio': float(np.median((txz - V) / txz)), 'n': int(V.size)}
# triangle, free midpoints of the ctrl set
xc, zc = load_set(args.scene + '_ctrl')
if xc is not None:
    yc = midpoint(xc, zc)
    tc = T(xc, zc).numpy()
    Vc = tc - T(xc, yc).numpy() - T(yc, zc).numpy()
    res['tri_mid'] = {'frac_viol': float((Vc > 1e-6).mean()), 'p95_rel': pct(Vc / tc, 95), 'max_rel': float((Vc / tc).max()),
                      'median_ratio': float(np.median((tc - Vc) / tc)), 'n': int(Vc.size)}
    # detour awareness
    xb, zb = load_set(args.scene + '_blocked')
    rb = (T(xb, zb) / d_se3(xb, zb)).numpy()
    rc = (torch.tensor(tc) / d_se3(xc, zc)).numpy()
    res['detour'] = {'blocked_T_over_d_median': float(np.median(rb)), 'ctrl_T_over_d_median': float(np.median(rc)),
                     'factor': float(np.median(rb) / np.median(rc)), 'blocked_p25': pct(rb, 25), 'blocked_p75': pct(rb, 75)}
    # tight sets: T / d on tightboth vs ctrl (slow band should raise the time)
    for s in ('tightboth', 'tightrot', 'tightgoal', 'squeeze'):
        xs, zs = load_set('{}_{}'.format(args.scene, s))
        if xs is None:
            continue
        rs = (T(xs, zs) / d_se3(xs, zs)).numpy()
        res['T_over_d_' + s] = {'median': float(np.median(rs)), 'p25': pct(rs, 25), 'p75': pct(rs, 75)}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('checkpoint',)}))
