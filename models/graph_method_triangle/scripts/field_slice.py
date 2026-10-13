"""tau along a full rotation of the start about a body axis, at a free position, for several fields.

For each anchor (position p of a test-case goal, goal rotation R_g, random unit axis n) the
start is x(phi) = (p, R_g exp(phi n)), phi in [0, 2 pi): the true travel time is the rotation
time min(phi, 2 pi - phi) / (2 pi S) with a corner (cut locus) at phi = pi and no jump
anywhere. Fields that take the rotation vector as input see the chart seam |rotvec| = pi
wherever the canonical rotation vector of R_g exp(phi n) flips sign; fields on SO(3) do not.
Writes tau(phi) per anchor and model plus the chart angle |rotvec(phi)|, and a figure.

    python /workspace/models/graph_method_triangle/scripts/field_slice.py --out /workspace/results/graph_method_cusp/slices \
        --model 'original=metric:/lieshape/Experiments/.../Model_Epoch_04000_*.pt' \
        --model 'graph_tri_eq=graph:/workspace/Experiments/.../latest.pt' [--anchors 8]
(cwd /lieshape inside the container, as plan_eval_graph.py)
"""
import os
import sys
import json
import math
import glob
import argparse
import importlib
import importlib.util

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', action='append', required=True, metavar='LABEL=KIND:CKPT')
ap.add_argument('--dataPath', default='testing_data/3dshape/Tshape3d_env1')
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--anchors', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--lie-root', default='/lieshape')
ap.add_argument('--graph-root', default='/workspace')
args = ap.parse_args()
sys.path.insert(0, args.lie_root)
os.chdir(args.lie_root)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm = _load_pkg('graph_method_triangle', os.path.join(args.graph_root, 'models', 'graph_method_triangle'))
se3 = importlib.import_module('graph_method_triangle.geometry.se3')
TWO_PI = 2 * math.pi


def load(kind, ckpt):
    ckpt = sorted(glob.glob(ckpt))[0]
    if kind == 'graph':
        mt = importlib.import_module('graph_method_triangle.model_train_graph')
        f, ck = mt.load_function(ckpt, device=args.device)
        c2f = importlib.import_module('graph_method_triangle.model_function_shape').coords_to_features
        net = f.network

        def tt(Xp):
            with torch.no_grad():
                Xp = Xp.to(args.device, torch.float32)
                return net.value(c2f(Xp[:, :6]), c2f(Xp[:, 6:]))
        return tt
    if kind == 'graph_native':
        _load_pkg('graph_native', os.path.join(args.graph_root, 'models', 'graph_native'))
        md = importlib.import_module('graph_native.model_train_metric')
        m = md.Model('/tmp/analysis_model', 'datasets/3dshape/Tshape3d_env1/', 6, [0.0] * 6, device=args.device)
        m.load(ckpt)
        return m.function.TravelTimes
    from analysis.common import load_model
    model, fn = load_model(kind, ckpt, args.device)
    return fn.TravelTimes


models = []
for spec in args.model:
    label, rest = spec.split('=', 1)
    kind, ckpt = rest.split(':', 1)
    models.append((label, load(kind, ckpt)))

arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
rng = np.random.RandomState(args.seed)
pick = rng.choice(len(arr), args.anchors, replace=False)
phi = np.linspace(0, TWO_PI, 721)
out = {'phi': phi.tolist(), 'anchors': []}
for i in pick:
    goal = arr[i, 6:]
    Rg = se3.so3_exp(torch.tensor(goal[3:] * TWO_PI))
    n = rng.randn(3)
    n /= np.linalg.norm(n)
    Rs = Rg[None] @ se3.so3_exp(torch.tensor(phi[:, None] * n[None]))
    rv = se3.so3_log(Rs) / TWO_PI                       # canonical rotation vectors (turns)
    chart_angle = rv.norm(dim=1).numpy() * 360
    X = np.concatenate([np.repeat(goal[None, :3], len(phi), 0), rv.numpy(),
                        np.repeat(goal[None], len(phi), 0)], 1).astype(np.float32)
    seam_idx = np.where(np.linalg.norm(np.diff(rv.numpy(), axis=0), axis=1) > 0.25)[0]
    rec = {'case': int(i), 'axis': n.tolist(), 'goal_theta_deg': float(np.linalg.norm(goal[3:]) * 360),
           'chart_angle_deg': chart_angle.tolist(), 'seam_phi_deg': np.degrees(phi[seam_idx]).tolist(), 'tau': {}}
    for label, tt in models:
        rec['tau'][label] = tt(torch.tensor(X).to(args.device)).detach().cpu().numpy().tolist()
    out['anchors'].append(rec)
os.makedirs(args.out, exist_ok=True)
json.dump(out, open(os.path.join(args.out, 'slices.json'), 'w'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
k = min(4, len(out['anchors']))
fig, axes = plt.subplots(k, 1, figsize=(8, 2.6 * k), sharex=True)
axes = np.atleast_1d(axes)
for ax, rec in zip(axes, out['anchors'][:k]):
    for label, _ in models:
        ax.plot(np.degrees(phi), rec['tau'][label], label=label)
    for s in rec['seam_phi_deg']:
        ax.axvline(s, color='gray', ls=':', lw=1)
    ax.axvline(180, color='k', lw=0.5)
    ax.set_ylabel('tau(start(phi), goal)')
    ax.set_title('case {}: goal angle {:.0f} deg; dotted = chart seam, solid = cut locus'.format(rec['case'], rec['goal_theta_deg']), fontsize=9)
axes[0].legend(fontsize=8)
axes[-1].set_xlabel('rotation of the start about a body axis (deg)')
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'slices.png'), dpi=120)
print('wrote', args.out)
