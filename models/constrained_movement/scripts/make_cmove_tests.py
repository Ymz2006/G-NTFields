"""Test sets of a constrained-movement cell, every case verified by a forward-monotone RRT-Connect on M.

A case is (t0, R0) -> (t1, R1) with t0 < t1 on the cell's path: the followed shape moves forward from t0 to t1 and
the camera pyramid (anchored at the shape's centre) must turn from R0 to R1 without ever touching the scene.

    full   t0 = 0, t1 = L: the whole path, the camera's start and end orientations given (the camera-follow query)
    seg    t0 < t1 uniform on [0, L] with t1 - t0 >= --min-seg (sub-paths)
    tight  full, both endpoint clearances in [0.005, 0.02]
    bigrot full, relative rotation between start and goal >= 150 deg
    hard   full, clearances in [0.005, 0.02] and relative rotation >= 135 deg

R0, R1 are uniform on SO(3) and kept if the exact pyramid clearance there is >= --offset (0.02). A case is kept if
an RRT-Connect on M = [0, L] x SO(3) connects it with t never decreasing (start tree forward, goal tree backward),
each motion (t linear, R slerp) checked with the exact clearance >= --margin at resolution --res in the metric
sqrt(dt^2 + (theta / 2 pi)^2). Every case also records whether the direct motion (t linear, R slerp the short way /
the long way) is collision-free: the trivial solution. (models/path_constrained/scripts/make_path_tests.py with
the pyramid in place of the prism.)

    python models/constrained_movement/scripts/make_cmove_tests.py --cell datasets/3dshape/data/cm_forest_a \
        --variant full --n 1000 --out testing_data/3dshape --workers 12

Output testing_data/3dshape/<cell>_<variant>/: sampled_points.npy (N, 12) [start | goal] in the arclength chart
(t = (l - L / 2, 0, 0), rotvec / 2 pi), cases.npz (l0, rv0, l1, rv1 in radians; the RRT path of each case),
meta.json (the cell's meta + the variant), path.npz, verification.json.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.constrained_movement import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cell', required=True, help='training cell directory (meta.json with the constraint, path.npz)')
ap.add_argument('--variant', required=True, choices=['full', 'seg', 'tight', 'bigrot', 'hard'])
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--out', default='testing_data/3dshape')
ap.add_argument('--name', default='')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--workers', type=int, default=12)
ap.add_argument('--offset', type=float, default=0.02)
ap.add_argument('--min-seg', type=float, default=0.25)
ap.add_argument('--step', type=float, default=0.05)
ap.add_argument('--res', type=float, default=0.003)
ap.add_argument('--margin', type=float, default=0.002)
ap.add_argument('--rrt-iters', type=int, default=4000)
ap.add_argument('--rrt-time', type=float, default=60.0)
args = ap.parse_args()

meta = json.load(open(os.path.join(args.cell, 'meta.json')))
con = meta['constraint']
path = G.Path.load(os.path.join(args.cell, con['path_file']))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
scene = G.scene_from_meta(meta, REPO)
pyr = G.Pyramid(con['pyramid']['W'], con['pyramid']['H'], con['pyramid']['D'])
L = path.L
name = args.name or '{}_{}'.format(os.path.basename(args.cell.rstrip('/')), args.variant)
TWO_PI = 2 * math.pi


def clearance(ell, q):
    """Exact clearance (k,) at arc lengths ell (k,) and quaternions q (k, 4) (CPU, float64)."""
    t = path.pos(torch.as_tensor(ell, dtype=torch.float64))
    R = torch.as_tensor(G.quat_matrix(q), dtype=torch.float64)
    return G.pyramid_clearance(scene, pyr, t, R).numpy()


def dist(a, b):
    """Metric distance between (l, q) nodes: a (4 + 1,) vs b (m, 5)."""
    dl = b[:, 0] - a[0]
    th = G.quat_angle(a[None, 1:], b[:, 1:])
    return np.sqrt(dl * dl + (th / TWO_PI) ** 2)


def motion_ok(a, b):
    """Collision-free motion a -> b (l linear, R slerp short way); a itself is assumed valid."""
    d = float(dist(a, b[None])[0])
    k = max(1, int(math.ceil(d / args.res)))
    u = np.arange(1, k + 1) / k
    q = G.slerp(a[1:], b[1:], u)
    ell = a[0] + u * (b[0] - a[0])
    return bool((clearance(ell, q) >= args.margin).all())


def interp(a, b, frac):
    q = G.slerp(a[1:], b[1:], np.array([frac]))[0]
    return np.concatenate([[a[0] + frac * (b[0] - a[0])], q])


class Tree:
    def __init__(self, root, forward):
        self.nodes = [root]
        self.parent = [-1]
        self.forward = forward

    def nearest(self, x):
        N = np.asarray(self.nodes)
        ok = N[:, 0] <= x[0] + 1e-12 if self.forward else N[:, 0] >= x[0] - 1e-12
        if not ok.any():
            return -1
        d = dist(x, N)
        d[~ok] = np.inf
        return int(d.argmin())

    def add(self, x, p):
        self.nodes.append(x)
        self.parent.append(p)
        return len(self.nodes) - 1

    def branch(self, i):
        out = []
        while i >= 0:
            out.append(self.nodes[i])
            i = self.parent[i]
        return out


TRAPPED, ADVANCED, REACHED = 0, 1, 2


def extend(T, x):
    i = T.nearest(x)
    if i < 0:
        return TRAPPED, -1
    near = T.nodes[i]
    d = float(dist(near, x[None])[0])
    new = x if d <= args.step else interp(near, x, args.step / d)
    if not motion_ok(near, new):
        return TRAPPED, -1
    j = T.add(new, i)
    return (REACHED if d <= args.step else ADVANCED), j


def rrt_connect(start, goal, rng):
    TA, TB = Tree(start, True), Tree(goal, False)
    t0 = time.time()
    for it in range(args.rrt_iters):
        if time.time() - t0 > args.rrt_time:
            break
        x = np.concatenate([[rng.uniform(start[0], goal[0])], G.random_quats(1, rng)[0]])
        st, j = extend(TA, x)
        if st != TRAPPED:
            xn = TA.nodes[j]
            while True:
                s2, k = extend(TB, xn)
                if s2 != ADVANCED:
                    break
            if s2 == REACHED:
                a, b = (TA, TB) if TA.forward else (TB, TA)
                ia, ib = (j, k) if TA.forward else (k, j)
                fw = a.branch(ia)[::-1]
                bw = b.branch(ib)
                return np.asarray(fw + bw[1:]), it + 1
        TA, TB = TB, TA
    return None, args.rrt_iters


def direct_ok(a, b, short):
    d = math.hypot(b[0] - a[0], G.quat_angle(a[1:], b[1:]) / TWO_PI if short else
                   (2 * math.pi - G.quat_angle(a[1:], b[1:])) / TWO_PI)
    k = max(2, int(math.ceil(d / args.res)))
    u = np.linspace(0, 1, k + 1)
    q = G.slerp(a[1:], b[1:], u, short=short)
    return bool((clearance(a[0] + u * (b[0] - a[0]), q) >= 0).all())


VARIANTS = {'full': dict(clear=(None, math.inf), angle=0.0), 'seg': dict(clear=(None, math.inf), angle=0.0),
            'tight': dict(clear=(0.005, 0.02), angle=0.0), 'bigrot': dict(clear=(None, math.inf), angle=150.0),
            'hard': dict(clear=(0.005, 0.02), angle=135.0)}
CLEAR = VARIANTS[args.variant]['clear']
CLEAR = (args.offset if CLEAR[0] is None else CLEAR[0], CLEAR[1])
MIN_ANGLE = math.radians(VARIANTS[args.variant]['angle'])


def draw_end(ell, rng, ref=None):
    """A uniform rotation with exact clearance in CLEAR at arc length ell (and, given ref, at least MIN_ANGLE
    away from it)."""
    while True:
        q = G.random_quats(256, rng)
        if ref is not None and MIN_ANGLE > 0:
            q = q[G.quat_angle(q, ref[None]) >= MIN_ANGLE]
            if not len(q):
                continue
        c = clearance(np.full(len(q), ell), q)
        ok = np.nonzero((c >= CLEAR[0]) & (c <= CLEAR[1]))[0]
        if len(ok):
            return q[ok[0]], float(c[ok[0]])


def job(k):
    torch.set_num_threads(1)
    rng = np.random.default_rng(1000003 * args.seed + k)
    if args.variant != 'seg':
        l0, l1 = 0.0, L
    else:
        while True:
            l0, l1 = np.sort(rng.uniform(0, L, 2))
            if l1 - l0 >= args.min_seg:
                break
    q0, c0 = draw_end(l0, rng)
    q1, c1 = draw_end(l1, rng, ref=q0)
    a = np.concatenate([[l0], q0])
    b = np.concatenate([[l1], q1])
    t0 = time.time()
    P, iters = rrt_connect(a, b, rng)
    rec = {'k': k, 'l0': float(l0), 'l1': float(l1), 'clear0': c0, 'clear1': c1, 'rrt_iters': int(iters),
           'rrt_s': time.time() - t0, 'solved': P is not None,
           'angle_deg': float(np.degrees(G.quat_angle(q0, q1))),
           'direct_short': direct_ok(a, b, True), 'direct_long': direct_ok(a, b, False)}
    if P is not None:
        dl = np.diff(P[:, 0])
        th = G.quat_angle(P[:-1, 1:], P[1:, 1:])
        rec['rrt_len'] = float(np.sqrt(dl ** 2 + (th / TWO_PI) ** 2).sum())
        rec['rrt_nodes'] = int(len(P))
        assert (dl >= -1e-9).all()
    return rec, a, b, P


if __name__ == '__main__':
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    recs, starts, goals, paths = [], [], [], []
    tried = n_fail = 0
    with mp.get_context('fork').Pool(args.workers) as pool:
        k = 0
        while len(recs) < args.n:
            need = args.n - len(recs)
            batch = list(range(k, k + int(need * 1.15) + 4))
            k += len(batch)
            for rec, a, b, P in pool.imap(job, batch):
                tried += 1
                if rec['solved'] and len(recs) < args.n:
                    recs.append(rec)
                    starts.append(a)
                    goals.append(b)
                    paths.append(P)
                elif not rec['solved']:
                    n_fail += 1
            print('{} / {} cases ({} tried, {:.0f}s)'.format(len(recs), args.n, tried, time.time() - t0), flush=True)
    S, Gl = np.asarray(starts), np.asarray(goals)
    rv0, rv1 = G.rotvec_from_quat(S[:, 1:]), G.rotvec_from_quat(Gl[:, 1:])
    pts = np.zeros((len(S), 12), np.float32)
    pts[:, 0] = S[:, 0] - 0.5 * L
    pts[:, 3:6] = rv0 / TWO_PI
    pts[:, 6] = Gl[:, 0] - 0.5 * L
    pts[:, 9:12] = rv1 / TWO_PI
    np.save(os.path.join(out, 'sampled_points.npy'), pts)
    lens = np.array([len(p) for p in paths])
    flat = np.concatenate(paths)
    np.savez(os.path.join(out, 'cases.npz'), l0=S[:, 0], rv0=rv0, l1=Gl[:, 0], rv1=rv1, rrt_flat=flat, rrt_len=lens)
    shutil.copy2(os.path.join(args.cell, con['path_file']), os.path.join(out, con['path_file']))
    m = dict(meta)
    m['testing_data'] = True
    m['constraint'] = dict(con, chart='arclength')
    m['variant'] = args.variant
    m['test_settings'] = {k_: v for k_, v in vars(args).items()}
    json.dump(m, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    ds = np.array([r['direct_short'] for r in recs])
    dl_ = np.array([r['direct_long'] for r in recs])
    summ = {'n': len(recs), 'tried': tried, 'unsolved': n_fail,
            'seconds': time.time() - t0, 'direct_short_ok': float(ds.mean()), 'direct_long_ok': float(dl_.mean()),
            'direct_either_ok': float((ds | dl_).mean()),
            'rrt_iters_median': float(np.median([r['rrt_iters'] for r in recs])),
            'angle_deg_median': float(np.median([r['angle_deg'] for r in recs])),
            'span_median': float(np.median([r['l1'] - r['l0'] for r in recs]))}
    json.dump({'summary': summ, 'cases': recs}, open(os.path.join(out, 'verification.json'), 'w'), indent=1)
    print(json.dumps(summ))
