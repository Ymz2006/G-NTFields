"""Stress-test planning sets (models/stress_bench): tight spaces, blocked straight lines, reversed queries.

Same layout and sampler as models/graph_lie/scripts/bench/make_bench_sets.py (sampled_points.npy (N, 12) =
start, goal in the chart (t, rotvec / 2 pi); meta.json copied from the scene's standard test set; endpoints
drawn from the training sampler). Clearances are the preprocess clearance of the placed shape (normalized
units; the training margin is 0.05, the standard test sets use endpoint offset 0.02, the "tight_3d" sets
0.005). Subsets:

  tightboth   start AND goal clearance in [offset, 0.015], |dp| in [0.1, 0.4]
              (both ends in the slow band: speed calibration at both ends of the field)
  tightgoal   start clearance > 0.02, goal clearance in [offset, 0.015], |dp| in [0.1, 0.4]
              (the goal is the field's source: the mirror of the bench 'tightstart' subset)
  tightrot    |dp| <= 0.02, both clearances in [offset, 0.015], relative rotation >= 60 deg
              (rotate in place inside clutter: the rotational part of the field in the slow band)
  blocked     both clearances > 0.02, |dp| in [0.2, 0.5], and >= 20 % of the poses on the straight SE(3)
              interpolation collide (an obstacle sits in the way: the field must encode a detour)
  squeeze     both clearances > 0.02, |dp| in [0.2, 0.5], straight interpolation free but with minimum
              clearance in (0, 0.008] (a narrow gap on the direct route: go through or around)
  ctrl        both clearances > 0.02, |dp| in [0.2, 0.5], straight interpolation clearance > 0.02 throughout
              (control for blocked / squeeze: same length distribution, nothing in the way)
  *_rev       start / goal swapped copies of tightgoal and blocked
              (direction asymmetry of the field: T(x, y) vs T(y, x), and the goal-side vs start-side slow band)

    docker ... -v Lie-Shape-NTFields:/lieshape:ro -w /lieshape ... python /workspace/models/stress_bench/make_stress_sets.py \
        --scene Tshape3d_env1 --out /workspace/datasets/3dshape/data/bench --n 300 --device cuda:0
"""
import argparse
import json
import math
import os
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
ap.add_argument('--seed', type=int, default=20261013)
ap.add_argument('--only', default='', help='comma list of subsets')
ap.add_argument('--chunk', type=int, default=500)
ap.add_argument('--tries', type=int, default=200)
ap.add_argument('--interp', type=int, default=25, help='interior poses on the straight SE(3) interpolation')
args = ap.parse_args()

shape, env = args.scene.split('_')[0], args.scene.split('_')[1]
scene = Scene(env_obj='datasets/3dshape/data/{}.obj'.format(env), shape_obj='datasets/3dshape/data/{}.obj'.format(shape),
              device=args.device)
gen = torch.Generator().manual_seed(args.seed)
meta_src = os.path.join('testing_data/3dshape', args.scene, 'meta.json')
meta = json.load(open(meta_src))
he = torch.tensor(scene.half_extent, dtype=torch.float32)
D2R = math.pi / 180.0
TIGHT = (args.offset, 0.015)
FREE = 0.02


def clearance(q):
    c = scene.clearance(q, chunk=args.chunk)
    d = c['dist'].clone()
    d[~c['free']] = -1.0          # colliding placements: clearance undefined -> negative
    return d


def inbox(p):
    return (p.abs() <= he).all(1)


def positions(n):
    p = torch.empty(n, 3)
    for k in range(3):
        p[:, k] = (torch.rand(n, generator=gen) * 2 - 1) * float(he[k])
    return p


def pair_with_offset(n, dlo, dhi):
    p0 = positions(n)
    d = torch.randn(n, 3, generator=gen)
    d = d / d.norm(dim=1, keepdim=True)
    d = d * (dlo + (dhi - dlo) * torch.rand(n, 1, generator=gen))
    return p0, p0 + d


def rotvecs(n):
    return random_rotvec(n, gen, 0.0, math.pi) / TWO_PI


def in_band(d, band):
    return (d >= band[0]) & (d <= band[1])


def interpolate(q0, q1, k):
    """k interior poses on the straight SE(3) interpolation (position linear, rotation geodesic). (N, k, 6)"""
    R0 = so3_exp(TWO_PI * q0[:, 3:].double())
    R1 = so3_exp(TWO_PI * q1[:, 3:].double())
    w = so3_log(R0.transpose(-1, -2) @ R1)
    ts = torch.linspace(0, 1, k + 2)[1:-1]
    out = []
    for t in ts:
        p = q0[:, :3] + float(t) * (q1[:, :3] - q0[:, :3])
        r = so3_log(R0 @ so3_exp(float(t) * w)) / TWO_PI
        out.append(torch.cat([p, r.float()], 1))
    return torch.stack(out, 1)


def endpoint_subset(name, dlo, dhi, start_band, goal_band, rel_lo=None, m=8000):
    rows, tries = [], 0
    while sum(len(r) for r in rows) < args.n and tries < args.tries:
        tries += 1
        p0, p1 = pair_with_offset(m, dlo, dhi)
        r0, r1 = rotvecs(m), rotvecs(m)
        ok = inbox(p1)
        if rel_lo is not None:
            ok &= geodesic_angle(r0, r1).float() / D2R >= rel_lo
        idx = torch.nonzero(ok)[:, 0]
        q0, q1 = torch.cat([p0, r0], 1)[idx], torch.cat([p1, r1], 1)[idx]
        c0 = clearance(q0)
        k0 = in_band(c0, start_band)
        if k0.sum() == 0:
            continue
        c1 = clearance(q1[k0])
        k1 = in_band(c1, goal_band)
        rows.append(torch.cat([q0[k0][k1], q1[k0][k1]], 1))
        print('  {}: try {} +{} ({})'.format(name, tries, int(k1.sum()), sum(len(r) for r in rows)), flush=True)
    return torch.cat(rows)[:args.n]


def line_subsets(m=1500):
    """blocked / squeeze / ctrl share the endpoint distribution (free > 0.02, |dp| in [0.2, 0.5])."""
    got = {'blocked': [], 'squeeze': [], 'ctrl': []}
    want = lambda k: sum(len(r) for r in got[k])  # noqa: E731
    tries = 0
    while min(want(k) for k in got) < args.n and tries < args.tries:
        tries += 1
        p0, p1 = pair_with_offset(m, 0.2, 0.5)
        r0, r1 = rotvecs(m), rotvecs(m)
        idx = torch.nonzero(inbox(p1))[:, 0]
        q0, q1 = torch.cat([p0, r0], 1)[idx], torch.cat([p1, r1], 1)[idx]
        keep = clearance(q0) > FREE
        q0, q1 = q0[keep], q1[keep]
        keep = clearance(q1) > FREE
        q0, q1 = q0[keep], q1[keep]
        if len(q0) == 0:
            continue
        P = interpolate(q0, q1, args.interp)
        d = clearance(P.reshape(-1, 6)).reshape(len(q0), args.interp)
        frac_coll = (d < 0).float().mean(1)
        dmin = d.min(1).values
        sel = {'blocked': frac_coll >= 0.2, 'squeeze': (dmin > 0) & (dmin <= 0.008), 'ctrl': dmin > FREE}
        for k, s in sel.items():
            if want(k) < args.n:
                got[k].append(torch.cat([q0[s], q1[s], frac_coll[s, None], dmin[s, None]], 1))
        print('  line: try {} '.format(tries) + ' '.join('{} {}'.format(k, want(k)) for k in got), flush=True)
    return {k: torch.cat(v)[:args.n] for k, v in got.items()}


def write(name, rows, extra=None):
    d = os.path.join(args.out, '{}_{}'.format(args.scene, name))
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'sampled_points.npy'), rows[:, :12].numpy().astype(np.float32))
    if extra is not None:
        np.save(os.path.join(d, 'line_stats.npy'), extra.numpy().astype(np.float32))
    m = dict(meta)
    m.update({'set': '{}_{}'.format(args.scene, name), 'generator': 'models/stress_bench/make_stress_sets.py',
              'stress_subset': name, 'n': int(rows.shape[0]), 'offset_endpoints': args.offset, 'seed': args.seed,
              'rrt_verified': False})
    with open(os.path.join(d, 'meta.json'), 'w') as fh:
        json.dump(m, fh, indent=1)
    th0 = rows[:, 3:6].norm(dim=1) * 360
    th1 = rows[:, 9:12].norm(dim=1) * 360
    rel = geodesic_angle(rows[:, 3:6], rows[:, 9:12]) / D2R
    dp = (rows[:, 6:9] - rows[:, :3]).norm(dim=1)
    print('{:16s} n={:4d}  |rotvec| start med {:5.1f}  goal med {:5.1f}  rel {:5.1f}-{:5.1f} (med {:5.1f})  |dp| med {:.3f}'.format(
        name, rows.shape[0], th0.median(), th1.median(), rel.min(), rel.max(), rel.median(), dp.median()), flush=True)


def reverse(src_name, dst_name):
    src = os.path.join(args.out, '{}_{}'.format(args.scene, src_name))
    if not os.path.exists(os.path.join(src, 'sampled_points.npy')):
        print('skip {}: {} missing'.format(dst_name, src))
        return
    a = np.load(os.path.join(src, 'sampled_points.npy'))
    rows = torch.tensor(np.concatenate([a[:, 6:], a[:, :6]], 1))
    write(dst_name, rows)


only = set(args.only.split(',')) if args.only else {'tightboth', 'tightgoal', 'tightrot', 'line', 'rev'}
t0 = time.time()
if 'tightboth' in only:
    write('tightboth', endpoint_subset('tightboth', 0.1, 0.4, TIGHT, TIGHT))
if 'tightgoal' in only:
    write('tightgoal', endpoint_subset('tightgoal', 0.1, 0.4, (FREE, 1.0), TIGHT))
if 'tightrot' in only:
    write('tightrot', endpoint_subset('tightrot', 0.0, 0.02, TIGHT, TIGHT, rel_lo=60))
if 'line' in only:
    for k, v in line_subsets().items():
        write(k, v[:, :12], extra=v[:, 12:])
if 'rev' in only:
    reverse('tightgoal', 'tightgoal_rev')
    reverse('blocked', 'blocked_rev')
print('done in {:.0f}s'.format(time.time() - t0))
