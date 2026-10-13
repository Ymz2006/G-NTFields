"""Travel time against exact ground truth on saturated-clearance pairs, by endpoint rotation angle.

Where the translational clearance is >= the margin along the whole way, S_d = 1 and the isotropic Eikonal
has an exact solution: the Lie formulation gives the SE(3) geodesic T_geo = sqrt(|dt|^2 + (phi / 2 pi)^2),
the chart formulation the straight chart segment T_chart = sqrt(|dt|^2 + |r1 - r0|^2) (chart lengths are
never shorter than geodesic ones; they coincide at the identity and differ most at large |rotvec|). The
combined speed never saturates (it needs a rotational clearance of pi), so the exact value is known only
for the iso fields; the others are evaluated for the ratio trends, not for the absolute value.

Pairs: start from the training sampler with |rotvec| in a bin, goal = start moved by |dp| <= --dmax and rotated
by phi in [--phi_lo, --phi_hi] deg about a random body axis; kept when the endpoints, three interior points of
the SE(3) geodesic and three interior points of the chart segment are all free with clearance >= MARGIN and
inside the box. --n pairs per bin.

    docker ... -w /lieshape ... python /workspace/models/graph_lie/scripts/bench/free_tau_eval.py \
        --scene Tshape3d_env1 --pairs /workspace/datasets/3dshape/bench/Tshape3d_env1_freepairs.npz \
        --out /workspace/results/graph_lie/bench/freepairs <ckpt> [<ckpt> ...]
Per checkpoint: tau in both directions, |c| and |g| at both endpoints (JSON next to the pairs' arrays).
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')
from analysis.common import Scene, TWO_PI, MARGIN, random_rotvec, so3_exp, so3_log, geodesic_angle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('ckpts', nargs='*')
ap.add_argument('--scene', default='Tshape3d_env1')
ap.add_argument('--pairs', required=True)
ap.add_argument('--out', default='/workspace/results/graph_lie/bench/freepairs')
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--dmax', type=float, default=0.15)
ap.add_argument('--phi_lo', type=float, default=30.0)
ap.add_argument('--phi_hi', type=float, default=90.0)
ap.add_argument('--bins', default='0,45,90,135,160,180')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--graph-root', default='/workspace')
ap.add_argument("--tries", type=int, default=200)
ap.add_argument("--clear-chunk", type=int, default=400)
args = ap.parse_args()
D2R = math.pi / 180.0
bins = [float(b) for b in args.bins.split(',')]


def compose(r_turns, w_rad):
    R = so3_exp(TWO_PI * r_turns.double()) @ so3_exp(w_rad.double())
    return (so3_log(R) / TWO_PI).float()


def chart_alt(r):
    """The other rotation vector of the same rotation (|r| in turns -> |r| - 1 along the same axis)."""
    n = r.norm(dim=1, keepdim=True)
    return r - r / n.clamp_min(1e-12)


if not os.path.exists(args.pairs):
    shape, env = args.scene.split('_')[0], args.scene.split('_')[1]
    scene = Scene(env_obj='datasets/3dshape/{}.obj'.format(env), shape_obj='datasets/3dshape/{}.obj'.format(shape),
                  device=args.device)
    gen = torch.Generator().manual_seed(args.seed)
    he = torch.tensor(scene.half_extent, dtype=torch.float32)
    rows, labels = [], []
    t0 = time.time()
    for lo, hi in zip(bins[:-1], bins[1:]):
        got, tries = [], 0
        while sum(len(g) for g in got) < args.n and tries < args.tries:
            tries += 1
            m = 3000
            p0 = torch.stack([(torch.rand(m, generator=gen) * 2 - 1) * float(he[k]) for k in range(3)], 1)
            r0 = random_rotvec(m, gen, lo * D2R, min(hi, 179.9) * D2R) / TWO_PI
            d = torch.randn(m, 3, generator=gen)
            d = d / d.norm(dim=1, keepdim=True) * (args.dmax * torch.rand(m, 1, generator=gen))
            p1 = p0 + d
            ax = torch.randn(m, 3, generator=gen)
            ax = ax / ax.norm(dim=1, keepdim=True)
            phi = (args.phi_lo + (args.phi_hi - args.phi_lo) * torch.rand(m, 1, generator=gen)) * D2R
            w = phi * ax
            r1 = compose(r0, w)
            ok = (p1.abs() <= he).all(1)
            # interior samples: SE(3) geodesic and chart segment
            qs = [torch.cat([p0, r0], 1), torch.cat([p1, r1], 1)]
            for t in (0.25, 0.5, 0.75):
                qs.append(torch.cat([p0 + t * d, compose(r0, t * w)], 1))
                qs.append(torch.cat([p0 + t * d, r0 + t * (r1 - r0)], 1))
            Q = torch.cat(qs)
            c = scene.clearance(Q, chunk=args.clear_chunk)
            sat = (c['free'] & (c['dist'] >= MARGIN)).view(len(qs), m).all(0)
            keep = ok & sat
            got.append(torch.cat([p0[keep], r0[keep], p1[keep], r1[keep]], 1))
            print('  bin {:.0f}-{:.0f}: try {} +{} ({})'.format(lo, hi, tries, int(keep.sum()), sum(len(g) for g in got)), flush=True)
        g = torch.cat(got)[:args.n]
        rows.append(g)
        labels.append(torch.full((len(g),), lo))
    X = torch.cat(rows).double()
    lab = torch.cat(labels).numpy()
    phi = geodesic_angle(X[:, 3:6].float(), X[:, 9:12].float()).double()
    dt2 = ((X[:, 6:9] - X[:, :3]) ** 2).sum(1)
    T_geo = torch.sqrt(dt2 + (phi / TWO_PI) ** 2)
    r0, r1 = X[:, 3:6], X[:, 9:12]
    cands = [torch.sqrt(dt2 + ((b - a) ** 2).sum(1)) for a in (r0, chart_alt(r0)) for b in (r1, chart_alt(r1))]
    T_chart = cands[0]
    T_chart_min = torch.stack(cands, 1).min(1).values
    os.makedirs(os.path.dirname(args.pairs) or '.', exist_ok=True)
    np.savez(args.pairs, pairs=X.float().numpy(), bin_lo=lab, phi_deg=(phi / D2R).numpy(), T_geo=T_geo.numpy(),
             T_chart=T_chart.numpy(), T_chart_min=T_chart_min.numpy(),
             theta0_deg=(r0.norm(dim=1) * 360).numpy(), theta1_deg=(r1.norm(dim=1) * 360).numpy(),
             dp=torch.sqrt(dt2).numpy(), rotshare=((phi / TWO_PI) ** 2 / (dt2 + (phi / TWO_PI) ** 2)).numpy(),
             args=json.dumps(vars(args)))
    print('wrote {} pairs to {} in {:.0f}s'.format(len(X), args.pairs, time.time() - t0), flush=True)
    for lo in bins[:-1]:
        s = lab == lo
        print('  bin {:.0f}: n={} median T_chart/T_geo {:.3f} (p90 {:.3f}), T_chart_min/T_geo {:.3f}'.format(
            lo, int(s.sum()), float(np.median((T_chart / T_geo).numpy()[s])), float(np.percentile((T_chart / T_geo).numpy()[s], 90)),
            float(np.median((T_chart_min / T_geo).numpy()[s]))), flush=True)

if not args.ckpts:
    sys.exit(0)

# ----------------------------------------------------------------------------- field evaluation
def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm = _load_pkg('graph_method_triangle', os.path.join(args.graph_root, 'models', 'graph_method_triangle'))
gn = _load_pkg('graph_native', os.path.join(args.graph_root, 'models', 'graph_native'))
md = importlib.import_module('graph_native.model_train_metric')
spec = importlib.util.spec_from_file_location('bench_audit', os.path.join(args.graph_root, 'models/graph_lie/scripts/bench/audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

P = np.load(args.pairs)
X = torch.tensor(P['pairs'], dtype=torch.float32, device=args.device)
Xb = torch.cat([X[:, 6:], X[:, :6]], 1)
os.makedirs(args.out, exist_ok=True)
for ck in args.ckpts:
    name = os.path.basename(os.path.dirname(ck))
    m = md.Model('/tmp/analysis_model', 'datasets/3dshape/Tshape3d_env1/', 6, [0.0] * 6, device=args.device)
    m.load(ck)
    m.network.eval()
    ckd = torch.load(ck, map_location='cpu')
    tau_f, g_f, c_f = audit.gradients(m.network, gn._sib, X, 8192)
    tau_b, g_b, c_b = audit.gradients(m.network, gn._sib, Xb, 8192)
    out = {'checkpoint': ck, 'speed_model': ckd.get('speed_model', 'combined'), 'grad_mode': ckd.get('grad_mode', 'chart'),
           'tau_fwd': tau_f.cpu().numpy().tolist(), 'tau_bwd': tau_b.cpu().numpy().tolist(),
           'c_start': c_f.norm(dim=1).cpu().numpy().tolist(), 'g_start': g_f.norm(dim=1).cpu().numpy().tolist(),
           'c_goal': c_b.norm(dim=1).cpu().numpy().tolist(), 'g_goal': g_b.norm(dim=1).cpu().numpy().tolist(),
           'cw_start': c_f[:, 3:].norm(dim=1).cpu().numpy().tolist(), 'gw_start': g_f[:, 3:].norm(dim=1).cpu().numpy().tolist()}
    with open(os.path.join(args.out, name + '.json'), 'w') as fh:
        json.dump(out, fh)
    tf = tau_f.cpu().numpy()
    print('{:28s} {:8s} {:5s} median tau/T_geo by bin: {}   tau/T_chart: {}'.format(
        name, out['speed_model'], out['grad_mode'],
        ' '.join('{:.3f}'.format(np.median((tf / P['T_geo'])[P['bin_lo'] == lo])) for lo in bins[:-1]),
        ' '.join('{:.3f}'.format(np.median((tf / P['T_chart'])[P['bin_lo'] == lo])) for lo in bins[:-1])), flush=True)
