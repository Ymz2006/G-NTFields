"""Test sets for a stretchable-link cell, every case verified by an RRT-Connect in the 7-d pose space.

A case is a start and a goal pose (t, R, L) of the link (drone a's position, the orientation of a -> b, the length),
both with exact clearance >= --offset (0.02, the standard sets' endpoint offset; the clearance includes the length
limits, so L is in [L_min + 0.02, L_max - 0.02]). Variants:

    std    endpoints uniform: t in the room, R uniform on SO(3), L uniform
    cross  the scene's crossing task: both drones in region A at the start and in region B at the goal (every
           other case the reverse); regions in raw OBJ units (5 units off the walls that separate them):
               forest    z > -15 (in front of the trees)  <->  z < -165 (behind them)
               windows   z > -80 (front room)             <->  z < -100 (back room)
               corridor  z < -136 (first leg)             <->  z > -43 (last leg)
               scaffold  x < 30                           <->  x > 170

A case is kept if an RRT-Connect joins start and goal, each motion (t linear, R slerp the short way, L linear)
checked with the exact clearance >= --margin at resolution --res in the metric sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2):
the drones never touch the scene, always see each other and the length stays within its limits. The RRT path is
then shortened by random shortcutting (the stored path and rrt_len; rrt_len_raw before). Every case also records
whether the direct motion (the same interpolation from start to goal) is collision-free.

    python models/drone_link/scripts/make_link_tests.py --cell datasets/3dshape/data/link_forest --variant std \
        --n 1000 --out testing_data/3dshape --workers 24

Output testing_data/3dshape/<cell>_<variant>/: sampled_points.npy (N, 14) [start | goal] in the chart
(t, rotvec / 2 pi, L), cases.npz (the RRT path of each case: rrt_flat (sum K, 8) rows (t, q, L), rrt_len (N,)),
meta.json (the cell's meta + the variant), verification.json.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.drone_link import geometry as G  # noqa: E402
from models.drone_link.cases import REGIONS, clear_np, draw_case  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cell', required=True, help='training cell directory (its meta.json names the scene and link)')
ap.add_argument('--variant', required=True, choices=['std', 'cross'])
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--out', default='testing_data/3dshape')
ap.add_argument('--name', default='')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--workers', type=int, default=24)
ap.add_argument('--offset', type=float, default=0.02)
ap.add_argument('--step', type=float, default=0.05)
ap.add_argument('--res', type=float, default=0.003)
ap.add_argument('--margin', type=float, default=0.002)
ap.add_argument('--rrt-iters', type=int, default=20000)
ap.add_argument('--rrt-time', type=float, default=120.0)
ap.add_argument('--shortcut', type=int, default=300, help='random shortcut attempts on each RRT path')
args = ap.parse_args()

meta = json.load(open(os.path.join(args.cell, 'meta.json')))
scene = G.load_scene(meta, REPO)
link = G.load_link(meta)
SC = meta['link']['scene']
name = args.name or '{}_{}'.format(os.path.basename(args.cell.rstrip('/')), args.variant)
TWO_PI = 2 * math.pi
LO, HI = scene.room_lo, scene.room_hi


def clear(t, q, L):
    return clear_np(scene, link, t, q, L)


def dist(a, b):
    """Metric distance between nodes (t, q, L): a (8,) vs b (m, 8)."""
    th = G.quat_angle(a[None, 3:7], b[:, 3:7])
    return np.sqrt(((b[:, :3] - a[:3]) ** 2).sum(1) + (th / TWO_PI) ** 2 + (b[:, 7] - a[7]) ** 2)


def interp(a, b, u):
    """Poses at fractions u (k,) of the motion a -> b: (t (k, 3), q (k, 4), L (k,))."""
    u = np.asarray(u, np.float64)
    return (a[:3] + u[:, None] * (b[:3] - a[:3]), G.slerp(a[3:7], b[3:7], u), a[7] + u * (b[7] - a[7]))


def motion_ok(a, b, margin):
    d = float(dist(a, b[None])[0])
    k = max(1, int(math.ceil(d / args.res)))
    t, q, L = interp(a, b, np.arange(1, k + 1) / k)
    return bool((clear(t, q, L) >= margin).all())


class Tree:
    def __init__(self, root):
        self.nodes = [root]
        self.parent = [-1]

    def nearest(self, x):
        return int(dist(x, np.asarray(self.nodes)).argmin())

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
    near = T.nodes[i]
    d = float(dist(near, x[None])[0])
    if d <= args.step:
        new = x
    else:
        t, q, L = interp(near, x, np.array([args.step / d]))
        new = np.concatenate([t[0], q[0], L])
    if not motion_ok(near, new, args.margin):
        return TRAPPED, -1
    return (REACHED if d <= args.step else ADVANCED), T.add(new, i)


def random_node(rng):
    t = rng.uniform(LO, HI)
    return np.concatenate([t, G.random_quats(1, rng)[0], [rng.uniform(link.L_min, link.L_max)]])


def rrt_connect(start, goal, rng):
    TA, TB = Tree(start), Tree(goal)
    a_is_start = True
    t0 = time.time()
    for it in range(args.rrt_iters):
        if time.time() - t0 > args.rrt_time:
            break
        st, j = extend(TA, random_node(rng))
        if st != TRAPPED:
            xn = TA.nodes[j]
            while True:
                s2, k = extend(TB, xn)
                if s2 != ADVANCED:
                    break
            if s2 == REACHED:
                fw, bw = (TA.branch(j), TB.branch(k)) if a_is_start else (TB.branch(k), TA.branch(j))
                return np.asarray(fw[::-1] + bw[1:]), it + 1, len(TA.nodes) + len(TB.nodes)
        TA, TB = TB, TA
        a_is_start = not a_is_start
    return None, args.rrt_iters, len(TA.nodes) + len(TB.nodes)


def shortcut(P, rng):
    """Random shortcutting: replace the nodes between two random nodes by the direct motion when it is valid."""
    P = list(P)
    for _ in range(args.shortcut):
        if len(P) < 3:
            break
        i, j = sorted(rng.choice(len(P), 2, replace=False))
        if j - i < 2:
            continue
        if motion_ok(P[i], P[j], args.margin):
            P = P[:i + 1] + P[j:]
    return np.asarray(P)


def path_len(P):
    th = G.quat_angle(P[:-1, 3:7], P[1:, 3:7])
    return float(np.sqrt((np.diff(P[:, :3], axis=0) ** 2).sum(1) + (th / TWO_PI) ** 2 + np.diff(P[:, 7]) ** 2).sum())


def job(k):
    torch.set_num_threads(1)
    a, c0, b, c1, rng = draw_case(scene, link, SC, args.variant, k, args.seed, args.offset)
    t0 = time.time()
    P, iters, nodes = rrt_connect(a, b, rng)
    rec = {'k': k, 'clear0': c0, 'clear1': c1, 'rrt_iters': int(iters), 'rrt_nodes': int(nodes),
           'rrt_s': time.time() - t0, 'solved': P is not None,
           'dist_t': float(np.linalg.norm(b[:3] - a[:3])), 'angle_deg': float(np.degrees(G.quat_angle(a[3:7], b[3:7]))),
           'L0': float(a[7]), 'L1': float(b[7]), 'direct_ok': motion_ok(a, b, 0.0)}
    if P is not None:
        rec['rrt_len_raw'] = path_len(P)
        P = shortcut(P, rng)
        rec['rrt_len'] = path_len(P)
        rec['rrt_waypoints'] = int(len(P))
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
            batch = list(range(k, k + int(need * 1.1) + 4))
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
                    print('  unsolved case {} ({} iterations, {:.0f}s)'.format(rec['k'], rec['rrt_iters'], rec['rrt_s']),
                          flush=True)
            print('{} / {} cases ({} tried, {:.0f}s)'.format(len(recs), args.n, tried, time.time() - t0), flush=True)
    S, Gl = np.asarray(starts), np.asarray(goals)
    x0 = G.to_chart(S[:, :3], G.rotvec_from_quat(S[:, 3:7]), S[:, 7])
    x1 = G.to_chart(Gl[:, :3], G.rotvec_from_quat(Gl[:, 3:7]), Gl[:, 7])
    np.save(os.path.join(out, 'sampled_points.npy'), np.concatenate([x0, x1], 1).astype(np.float32))
    np.savez(os.path.join(out, 'cases.npz'), start=S, goal=Gl, rrt_flat=np.concatenate(paths),
             rrt_len=np.array([len(p) for p in paths]))
    m = dict(meta)
    m['testing_data'] = True
    m['offset'] = args.offset
    m['variant'] = args.variant
    m['test_settings'] = dict(vars(args))
    if args.variant == 'cross':
        m['regions_raw'] = REGIONS[SC]
    json.dump(m, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    summ = {'n': len(recs), 'tried': tried, 'unsolved': n_fail, 'seconds': time.time() - t0,
            'direct_ok': float(np.mean([r['direct_ok'] for r in recs])),
            'rrt_iters_median': float(np.median([r['rrt_iters'] for r in recs])),
            'rrt_s_median': float(np.median([r['rrt_s'] for r in recs])),
            'rrt_len_median': float(np.median([r['rrt_len'] for r in recs])),
            'dist_t_median': float(np.median([r['dist_t'] for r in recs])),
            'angle_deg_median': float(np.median([r['angle_deg'] for r in recs]))}
    json.dump({'summary': summ, 'cases': recs}, open(os.path.join(out, 'verification.json'), 'w'), indent=1)
    print(json.dumps(summ))
