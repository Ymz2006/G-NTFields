"""Paths of the followed shape (RRT), scored for the camera pyramid anchored at the shape's centre.

A candidate: a start and a goal for the shape's centre drawn uniformly in the room (or in the raw height band
--y-band, e.g. below the crowns and roofs so the shape has to fly between the obstacles; shape clearance >=
--end-clear, at least --min-span apart, and a fraction >= --end-free of random pyramid rotations free there),
connected by an RRT-Connect in R^3 (the shape keeps its orientation; each motion checked with the exact shape
clearance >= --clear at resolution --res; samples in the band with probability --band-bias), shortcut (--shortcuts
random chords, then a greedy pass to the farthest visible node) and smoothed (averaging iterations that never move
a point below --clear), then resampled at 1e-3. Kept if its length is in [--min-len, --max-len].

For every kept path the pyramid is scored: the free fraction of uniformly random rotations at --stations points
along the path (exact pyramid clearance > 0; where it is small the camera has little room to turn), and the
success of the straight motion (t linear, R slerp the short way) between random endpoint rotations with clearance
>= 0.02 at t = 0 and t = L (the trivial plan). Paths whose camera room never drops below --min-free and whose
straight motion fails often are the interesting ones.

    python models/constrained_movement/scripts/make_shape_paths.py --env forest --seed 0
    python models/constrained_movement/scripts/make_shape_paths.py --env forest --export 12   # -> cm_forest_a.npz

Writes <out>/cm_<env>_cand.json (statistics) and cm_<env>_cand.npz (dense paths, RRT and shortcut waypoints).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.constrained_movement import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--env', required=True)
ap.add_argument('--shape', default='bird', choices=sorted(G.SHAPES))
ap.add_argument('--obj-dir', default='datasets/3dshape/obj')
ap.add_argument('--out', default='datasets/3dshape/paths')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n-candidates', type=int, default=40)
ap.add_argument('--max-tries', type=int, default=400)
ap.add_argument('--clear', type=float, default=0.01, help='shape clearance along the path (normalized units)')
ap.add_argument('--end-clear', type=float, default=0.02)
ap.add_argument('--end-free', type=float, default=0.1)
ap.add_argument('--y-band', default='', help='raw y range lo,hi of the endpoints (and of the biased RRT samples)')
ap.add_argument('--band-bias', type=float, default=0.0, help='probability of an RRT sample in the y band')
ap.add_argument('--min-span', type=float, default=0.6)
ap.add_argument('--min-len', type=float, default=0.8)
ap.add_argument('--max-len', type=float, default=1.8)
ap.add_argument('--step', type=float, default=0.04, help='RRT step')
ap.add_argument('--res', type=float, default=0.003, help='motion check resolution')
ap.add_argument('--rrt-iters', type=int, default=20000)
ap.add_argument('--shortcuts', type=int, default=1000)
ap.add_argument('--smooth-iters', type=int, default=300)
ap.add_argument('--stations', type=int, default=64)
ap.add_argument('--rots', type=int, default=2048)
ap.add_argument('--pairs', type=int, default=128, help='endpoint pairs of the straight-motion score')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--export', nargs='*', type=int, default=None, help='candidate indices to export as paths a, b, ...')
args = ap.parse_args()

dev = args.device if torch.cuda.is_available() else 'cpu'
scene = G.Scene(os.path.join(args.obj_dir, args.env + '.obj'))
shape = G.Shape(G.SHAPES[args.shape], scene.env_scale)
pyr = G.default_pyramid(scene.env_scale)
os.makedirs(args.out, exist_ok=True)
stem = os.path.join(args.out, 'cm_{}'.format(args.env))

if args.export is not None:
    z = np.load(stem + '_cand.npz', allow_pickle=True)
    stats = json.load(open(stem + '_cand.json'))
    for j, k in enumerate(args.export):
        name = 'cm_{}_{}'.format(args.env, 'abcdefgh'[j])
        p = G.Path(z['path_{}'.format(k)])
        st = stats['candidates'][k]
        p.save(os.path.join(args.out, name + '.npz'), env=args.env, shape=args.shape, candidate=k,
               rrt=z['rrt_{}'.format(k)], shortcut=z['short_{}'.format(k)], clear=args.clear,
               shape_parts_raw=np.asarray(G.SHAPES[args.shape], np.float64), camera=json.dumps(G.CAMERA))
        print('exported candidate {} -> {}: L = {:.3f}, free mean {:.3f} min {:.3f}, straight {:.3f}'.format(
            k, name, p.L, st['free_mean'], st['free_min'], st['straight_ok']))
    sys.exit(0)

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
lo_c = scene.room_lo - shape.lo.min(0)          # the centre's box: the shape inside the room
hi_c = scene.room_hi - shape.hi.max(0)
lo_e, hi_e = lo_c.copy(), hi_c.copy()            # the endpoints' box
if args.y_band:
    yb = [(float(v) - scene.env_center[1]) / scene.env_scale for v in args.y_band.split(',')]
    lo_e[1], hi_e[1] = max(lo_c[1], yb[0]), min(hi_c[1], yb[1])


def sclear(P):
    """Exact shape clearance (n,) at centres P (n, 3) (float64, CPU)."""
    return shape.clearance(scene, torch.as_tensor(np.atleast_2d(P), dtype=torch.float64)).numpy()


def seg_free(a, b, clear=None):
    clear = args.clear if clear is None else clear
    k = max(1, int(math.ceil(np.linalg.norm(b - a) / args.res)))
    u = np.arange(1, k + 1)[:, None] / k
    return bool((sclear(a + u * (b - a)) >= clear).all())


def rand_rot(n):
    return G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng), dtype=torch.float32, device=dev))


def pclear(P, R):
    return G.pyramid_clearance(scene, pyr, torch.as_tensor(P, dtype=torch.float32, device=dev), R)


def free_frac(P, n):
    """Fraction of random rotations with pyramid clearance > 0 (and > 0.02) at each centre P (m, 3)."""
    R = rand_rot(n)
    m = len(P)
    T = torch.as_tensor(P, dtype=torch.float32, device=dev)[:, None].expand(-1, n, -1).reshape(-1, 3)
    c = G.pyramid_clearance(scene, pyr, T, R[None].expand(m, -1, -1, -1).reshape(-1, 3, 3)).reshape(m, n)
    return (c > 0).float().mean(1).cpu().numpy(), (c > 0.02).float().mean(1).cpu().numpy()


def draw_end():
    while True:
        P = rng.uniform(lo_e, hi_e, (512, 3))
        P = P[sclear(P) >= args.end_clear]
        for p in P[:16]:
            if free_frac(p[None], 256)[0][0] >= args.end_free:
                return p


def rrt_connect(a, b):
    trees = [([a], [-1]), ([b], [-1])]
    for it in range(args.rrt_iters):
        x = rng.uniform(lo_e, hi_e) if rng.random() < args.band_bias else rng.uniform(lo_c, hi_c)
        A, B_ = trees[it % 2], trees[(it + 1) % 2]
        N = np.asarray(A[0])
        i = int(np.linalg.norm(N - x, axis=1).argmin())
        d = np.linalg.norm(x - N[i])
        new = x if d <= args.step else N[i] + (x - N[i]) * (args.step / d)
        if not seg_free(N[i], new):
            continue
        A[0].append(new)
        A[1].append(i)
        j = len(A[0]) - 1
        # connect B towards new
        while True:
            NB = np.asarray(B_[0])
            k = int(np.linalg.norm(NB - new, axis=1).argmin())
            d = np.linalg.norm(new - NB[k])
            nb = new if d <= args.step else NB[k] + (new - NB[k]) * (args.step / d)
            if not seg_free(NB[k], nb):
                break
            B_[0].append(nb)
            B_[1].append(k)
            if d <= args.step:
                def branch(T, i):
                    out = []
                    while i >= 0:
                        out.append(T[0][i])
                        i = T[1][i]
                    return out
                pa, pb = branch(A, j), branch(B_, len(B_[0]) - 1)
                path = pa[::-1] + pb[1:] if it % 2 == 0 else pb[::-1] + pa[1:]
                return np.asarray(path), it + 1
    return None, args.rrt_iters


def shortcut(P):
    P = list(P)
    for _ in range(args.shortcuts):
        if len(P) < 3:
            break
        i, j = sorted(rng.choice(len(P), 2, replace=False))
        if j - i < 2:
            continue
        if seg_free(P[i], P[j]):
            P = P[:i + 1] + P[j:]
    out, i = [P[0]], 0
    while i < len(P) - 1:
        j = next(j for j in range(len(P) - 1, i, -1) if j == i + 1 or seg_free(P[i], P[j]))
        out.append(P[j])
        i = j
    return np.asarray(out)


def smooth(P):
    Q = G.PP.resample(P, 0.01)
    for _ in range(args.smooth_iters):
        S = Q.copy()
        S[1:-1] = 0.5 * Q[1:-1] + 0.25 * (Q[:-2] + Q[2:])
        ok = sclear(S) >= args.clear
        Q = np.where(ok[:, None], S, Q)
    return G.PP.resample(Q, 1e-3)


def straight_score(path, n):
    """Fraction of n endpoint pairs (clearance >= 0.02 at both ends) whose straight motion is collision-free."""
    ends = []
    for p in (path.pts[:1], path.pts[-1:]):
        got = []
        for _ in range(64):
            q = G.random_quats(4096, rng)
            c = pclear(np.repeat(p, len(q), 0), torch.as_tensor(G.quat_matrix(q), dtype=torch.float32, device=dev))
            got += list(q[(c >= 0.02).cpu().numpy()])
            if len(got) >= n:
                break
        if len(got) < n:
            return float('nan')
        ends.append(got[:n])
    k = max(64, int(math.ceil(math.hypot(path.L, 0.5) / 0.004)))
    u = np.linspace(0, 1, k + 1)
    ell = torch.as_tensor(u * path.L, dtype=torch.float32, device=dev)
    T = path.pos(ell)
    ok = []
    for q0, q1 in zip(*ends):
        q = G.slerp(q0, q1, u)
        R = torch.as_tensor(G.quat_matrix(q), dtype=torch.float32, device=dev)
        ok.append(bool((G.pyramid_clearance(scene, pyr, T, R) >= 0).all()))
    return float(np.mean(ok))


cands, paths, rrts, shorts = [], [], [], []
reject = {'span': 0, 'rrt': 0, 'length': 0, 'clearance': 0}
t0 = time.time()
tries = 0
print('{}: scale {:.3f}, {} boxes, shape {} {} raw, pyramid {:.3f} x {:.3f} x {:.3f} (fov {:.1f} x {:.1f} deg)'.format(
    args.env, scene.env_scale, scene.n_boxes, args.shape, shape.extent.tolist(), pyr.W, pyr.H, pyr.D, *pyr.fov_deg),
    flush=True)
while len(cands) < args.n_candidates and tries < args.max_tries:
    tries += 1
    a = draw_end()
    b = draw_end()
    if np.linalg.norm(b - a) < args.min_span:
        reject['span'] += 1
        continue
    raw, iters = rrt_connect(a, b)
    if raw is None:
        reject['rrt'] += 1
        continue
    short = shortcut(raw)
    dense = smooth(short)
    path = G.Path(dense)
    mc = float(sclear(path.pts).min())
    if not (args.min_len <= path.L <= args.max_len):
        reject['length'] += 1
        continue
    if mc < 0.9 * args.clear:
        reject['clearance'] += 1
        continue
    ell = np.linspace(0, path.L, args.stations)
    f0, f2 = free_frac(path.pos(torch.as_tensor(ell, dtype=torch.float32, device=dev)).cpu().numpy(), args.rots)
    st = {'index': len(cands), 'L': path.L, 'span': float(np.linalg.norm(b - a)), 'rrt_iters': iters,
          'rrt_nodes': len(raw), 'shortcut_nodes': len(short), 'min_shape_clear': mc,
          'free_mean': float(f0.mean()), 'free_min': float(f0.min()), 'free_q10': float(np.quantile(f0, 0.1)),
          'free02_mean': float(f2.mean()), 'free02_min': float(f2.min()),
          'straight_ok': straight_score(path, args.pairs), 'profile': f0.round(3).tolist(),
          'start_raw': scene.to_raw(path.pts[0]).round(1).tolist(), 'goal_raw': scene.to_raw(path.pts[-1]).round(1).tolist()}
    cands.append(st)
    paths.append(path.pts)
    rrts.append(raw)
    shorts.append(short)
    print('cand {:3d}: L {:.3f} rrt {:4d} it, free mean {:.3f} min {:.3f} (>0.02: min {:.3f}), straight {:.3f}  '
          '({} tries, {:.0f}s)'.format(st['index'], path.L, iters, st['free_mean'], st['free_min'], st['free02_min'],
                                     st['straight_ok'], tries, time.time() - t0), flush=True)

np.savez(stem + '_cand.npz', **{'path_{}'.format(i): x for i, x in enumerate(paths)},
         **{'rrt_{}'.format(i): x for i, x in enumerate(rrts)}, **{'short_{}'.format(i): x for i, x in enumerate(shorts)})
json.dump({'env': args.env, 'args': vars(args), 'tries': tries, 'rejected': reject, 'env_scale': scene.env_scale,
           'pyramid': {'W': pyr.W, 'H': pyr.H, 'D': pyr.D, 'fov_deg': pyr.fov_deg}, 'candidates': cands},
          open(stem + '_cand.json', 'w'), indent=1)
print('{} candidates from {} tries in {:.0f}s -> {}_cand.json; rejected: {}'.format(
    len(cands), tries, time.time() - t0, stem, reject))
