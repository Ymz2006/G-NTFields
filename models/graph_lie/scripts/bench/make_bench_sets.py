"""Targeted planning subsets for the chart-vs-Lie gradient benchmark (models/graph_lie/scripts/bench).

Every subset is a planning test set in the layout of Lie-Shape-NTFields/testing_data (sampled_points.npy
(N, 12) = start, goal in the chart (t, rotvec / 2 pi); meta.json copied from the scene's standard test set),
drawn from the training sampler (analysis.common.Scene.sample_configs: uniform position in the box, uniform
axis, angle uniform in [0, pi]) and restricted so that the chart and Lie Eikonal equations disagree by a
known amount along the way, or agree (controls). Endpoints are free with clearance > --offset (0.005, as the
Lie repo's make_sets.py). The chart metric charges a body rotation at a pose with |rotvec| = theta up to
(theta / 2) / sin(theta / 2) times its physical cost (1 at the identity, 1.27 at 134 deg, pi / 2 at 180 deg;
only the component orthogonal to the rotation vector), so the subsets are parametrised by the endpoint
rotation angles and by the rotation axis relative to the rotation vector:

  nearid        both |rotvec| < 45 deg, relative rotation >= 45 deg, |dp| in [0.1, 0.4]   (control: chart = Lie)
  bothfar       both |rotvec| in [135, 180) deg, |dp| in [0.1, 0.4]                        (chart factor >= 1.27 at both ends)
  startfar      start |rotvec| in [150, 180), goal |rotvec| < 45 deg, |dp| in [0.1, 0.4]  (the planner leaves the distorted region)
  goalfar       start |rotvec| < 45 deg, goal |rotvec| in [150, 180), |dp| in [0.1, 0.4]  (the planner enters it)
  perp / par    paired: same position (|dp| <= 0.05), same start (|rotvec| in [120, 165] deg), same geodesic
                angle phi in [60, 120] deg; the goal rotates the start about a body axis orthogonal to the
                rotation vector (perp: chart factor maximal) or about the rotation vector itself towards the
                identity (par: factor 1). Pairs are kept only when both goals are free, so perp and par share
                their cases one to one.
  longtrans_near  |dp| >= 0.5, both |rotvec| < 45 deg                                      (long path, no distortion)
  longtrans_far   |dp| >= 0.5, both |rotvec| in [135, 180), relative rotation < 90 deg    (long path inside the distorted region)
  tightstart    start clearance in [offset, 0.015] (S_d <= 0.26 after the warp), goal clearance > 0.02,
                |dp| in [0.1, 0.4]                                                           (speed-prediction stress)

    docker ... -v Lie-Shape-NTFields:/lieshape:ro -w /lieshape ... python /workspace/models/graph_lie/scripts/bench/make_bench_sets.py \
        --scene Tshape3d_env1 --out /workspace/datasets/3dshape/data/bench --n 300 --device cuda:0
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.append('.')
from analysis.common import Scene, TWO_PI, random_rotvec, so3_exp, so3_log, geodesic_angle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--scene', default='Tshape3d_env1', help='<shape>_<env> of testing_data/3dshape')
ap.add_argument('--out', default='/workspace/datasets/3dshape/data/bench')
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--offset', type=float, default=0.005)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--seed', type=int, default=20261012)
ap.add_argument('--only', default='', help='comma list of subsets')
ap.add_argument('--chunk', type=int, default=500)
ap.add_argument('--tries', type=int, default=60)
args = ap.parse_args()

shape, env = args.scene.split('_')[0], args.scene.split('_')[1]
scene = Scene(env_obj='datasets/3dshape/data/{}.obj'.format(env), shape_obj='datasets/3dshape/data/{}.obj'.format(shape),
              device=args.device)
gen = torch.Generator().manual_seed(args.seed)
meta_src = os.path.join('testing_data/3dshape', args.scene, 'meta.json')
meta = json.load(open(meta_src))
he = torch.tensor(scene.half_extent, dtype=torch.float32)
D2R = math.pi / 180.0


def free(q, offset=None):
    c = scene.clearance(q, chunk=args.chunk)
    return c['free'] & (c['dist'] > (args.offset if offset is None else offset)), c


def inbox(p):
    return (p.abs() <= he).all(1)


def rotvec_in(n, lo_deg, hi_deg):
    return random_rotvec(n, gen, lo_deg * D2R, hi_deg * D2R) / TWO_PI


def compose(r_turns, w_rad, left=False):
    """exp(2 pi r) exp(w) (body / right) or exp(w) exp(2 pi r) (left) -> canonical turns."""
    R = so3_exp(TWO_PI * r_turns.double())
    W = so3_exp(w_rad.double())
    M = W @ R if left else R @ W
    return (so3_log(M) / TWO_PI).float()


def positions(n):
    p = torch.empty(n, 3)
    for k in range(3):
        p[:, k] = (torch.rand(n, generator=gen) * 2 - 1) * float(he[k])
    return p


def pair_with_offset(n, dlo, dhi):
    """start positions and goal positions |dp| in [dlo, dhi] inside the box."""
    p0 = positions(n)
    d = torch.randn(n, 3, generator=gen)
    d = d / d.norm(dim=1, keepdim=True)
    d = d * (dlo + (dhi - dlo) * torch.rand(n, 1, generator=gen))
    return p0, p0 + d


def generic(name, ang0, ang1, dlo, dhi, rel_lo=None, rel_hi=None, start_clear=None, goal_clear=None):
    rows, tries = [], 0
    while sum(len(r) for r in rows) < args.n and tries < args.tries:
        tries += 1
        m = 4000
        p0, p1 = pair_with_offset(m, dlo, dhi)
        r0, r1 = rotvec_in(m, *ang0), rotvec_in(m, *ang1)
        ok = inbox(p1)
        if rel_lo is not None or rel_hi is not None:
            rel = geodesic_angle(r0, r1).float() / D2R
            if rel_lo is not None:
                ok &= rel >= rel_lo
            if rel_hi is not None:
                ok &= rel < rel_hi
        q0, q1 = torch.cat([p0, r0], 1), torch.cat([p1, r1], 1)
        idx = torch.nonzero(ok)[:, 0]
        if len(idx) == 0:
            continue
        f0, c0 = free(q0[idx], start_clear[0] if start_clear else None)
        if start_clear:
            f0 &= c0['dist'] <= start_clear[1]
        f1, c1 = free(q1[idx], goal_clear if goal_clear else None)
        keep = idx[f0 & f1]
        rows.append(torch.cat([q0[keep], q1[keep]], 1))
        print('  {}: try {} +{} ({})'.format(name, tries, len(keep), sum(len(r) for r in rows)), flush=True)
    return torch.cat(rows)[:args.n]


def perp_par():
    """Shared cases for perp and par (see module docstring)."""
    P, Q, R = [], [], []
    tries = 0
    while sum(len(x) for x in P) < args.n and tries < args.tries:
        tries += 1
        m = 4000
        p0, p1 = pair_with_offset(m, 0.0, 0.05)
        r0 = rotvec_in(m, 120, 165)
        u = r0 / r0.norm(dim=1, keepdim=True)
        phi = (60 + 60 * torch.rand(m, 1, generator=gen)) * D2R
        # body axis orthogonal to u: random vector projected out of u
        a = torch.randn(m, 3, generator=gen)
        a = a - (a * u).sum(1, keepdim=True) * u
        a = a / a.norm(dim=1, keepdim=True)
        r_perp = compose(r0, phi * a)                 # R0 exp(phi a), a _|_ u in the body frame
        r_par = compose(r0, -phi * u)                 # R0 exp(-phi u): angle theta0 - phi, same axis
        ok = inbox(p1)
        q0 = torch.cat([p0, r0], 1)
        qp = torch.cat([p1, r_perp], 1)
        qq = torch.cat([p1, r_par], 1)
        idx = torch.nonzero(ok)[:, 0]
        f0, _ = free(q0[idx])
        fp, _ = free(qp[idx])
        fq, _ = free(qq[idx])
        keep = idx[f0 & fp & fq]
        P.append(q0[keep]), Q.append(qp[keep]), R.append(qq[keep])
        print('  perp/par: try {} +{} ({})'.format(tries, len(keep), sum(len(x) for x in P)), flush=True)
    q0, qp, qq = (torch.cat(x)[:args.n] for x in (P, Q, R))
    gp, gq = geodesic_angle(q0[:, 3:], qp[:, 3:]), geodesic_angle(q0[:, 3:], qq[:, 3:])
    print('  perp/par geodesic angle check: max |perp - par| = {:.2e} deg'.format(float((gp - gq).abs().max() / D2R)))
    return torch.cat([q0, qp], 1), torch.cat([q0, qq], 1)


def write(name, rows):
    d = os.path.join(args.out, '{}_{}'.format(args.scene, name))
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'sampled_points.npy'), rows.numpy().astype(np.float32))
    m = dict(meta)
    m.update({'set': '{}_{}'.format(args.scene, name), 'generator': 'models/graph_lie/scripts/bench/make_bench_sets.py',
              'bench_subset': name, 'n': int(rows.shape[0]), 'offset_endpoints': args.offset, 'seed': args.seed,
              'rrt_verified': False})
    with open(os.path.join(d, 'meta.json'), 'w') as fh:
        json.dump(m, fh, indent=1)
    th0 = rows[:, 3:6].norm(dim=1) * 360
    th1 = rows[:, 9:12].norm(dim=1) * 360
    rel = geodesic_angle(rows[:, 3:6], rows[:, 9:12]) / D2R
    dp = (rows[:, 6:9] - rows[:, :3]).norm(dim=1)
    print('{:16s} n={:4d}  |rotvec| start {:5.1f}-{:5.1f} (med {:5.1f})  goal {:5.1f}-{:5.1f} (med {:5.1f})  rel {:5.1f}-{:5.1f} (med {:5.1f})  |dp| med {:.3f}'.format(
        name, rows.shape[0], th0.min(), th0.max(), th0.median(), th1.min(), th1.max(), th1.median(),
        rel.min(), rel.max(), rel.median(), dp.median()), flush=True)


SPECS = {
    'nearid': dict(ang0=(0, 45), ang1=(0, 45), dlo=0.1, dhi=0.4, rel_lo=45),
    'bothfar': dict(ang0=(135, 180), ang1=(135, 180), dlo=0.1, dhi=0.4),
    'startfar': dict(ang0=(150, 180), ang1=(0, 45), dlo=0.1, dhi=0.4),
    'goalfar': dict(ang0=(0, 45), ang1=(150, 180), dlo=0.1, dhi=0.4),
    'longtrans_near': dict(ang0=(0, 45), ang1=(0, 45), dlo=0.5, dhi=1.2),
    'longtrans_far': dict(ang0=(135, 180), ang1=(135, 180), dlo=0.5, dhi=1.2, rel_hi=90),
    'tightstart': dict(ang0=(0, 180), ang1=(0, 180), dlo=0.1, dhi=0.4, start_clear=(args.offset, 0.015), goal_clear=0.02),
}
only = set(args.only.split(',')) if args.only else set(SPECS) | {'perp', 'par'}
t0 = time.time()
for name, spec in SPECS.items():
    if name in only:
        write(name, generic(name, **spec))
if 'perp' in only or 'par' in only:
    rp, rq = perp_par()
    write('perp', rp)
    write('par', rq)
print('done in {:.0f}s'.format(time.time() - t0))
