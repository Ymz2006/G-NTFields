"""Targeted, RRT-verified test sets in the layout of testing_data/3dshape/<cell> (2026-10-17, second request).

Every pair is (1) sampled like the standard sets (independent uniform placements: position uniform in the room
box, rotation = uniform axis x angle uniform in [0, pi]), (2) filtered to the variant's constraints, (3) exact
collision-checked at both ends against three independently drawn 50k-point clouds of the environment surface
(the evaluator's own test: an environment surface point inside the placed shape = collision; clearance = the
smallest signed distance, as plan_eval_homotopy.path_clearance), and (4) connected by an RRT-Connect in SE(3)
whose motion check keeps a 0.003 margin to the shape surface (trilinear signed-distance grid of the shape), after
which the returned path is re-walked with the exact test. OMPL is not installed here, so the planner is this
file's own (bidirectional, extend / connect, step 0.05, resolution 0.005 in the metric
sqrt(|dp|^2 + (theta / 2 pi)^2), uniform sampling); a pair it cannot connect within the budget is dropped.

Variants (constraints in normalized units; clearance band = exact endpoint clearance on the first cloud, the
other two clouds must give >= offset as well):

  long     offset 0.02, |dp| >= 0.6                                  (top ~15 % of the standard sets' distances)
  bigrot   offset 0.02, relative rotation >= 150 deg                  (the ridge region: ~25 % of the standard sets)
  tight    offset 0.005, both clearances in [0.005, 0.02]             (both ends inside the slow band, below the
                                                                        standard sets' 0.02 endpoint offset)
  hard     offset 0.005, clearances in [0.005, 0.02], |dp| >= 0.5, rotation >= 135 deg
  through  (wall-with-hole scenes) offset 0.02, start and goal on opposite sides of the slab, at least 0.05 away
           from it; verification = RRT to an aligned approach pose in front of the hole, a straight translation
           through the hole with the aligned orientation (exact check at 0.0025), RRT to the goal.

Run in the container with the Lie repo at /lieshape (dataprocessing.preprocess_obj helpers) and this repo at
/workspace:

    python /workspace/models/graph_homotopy/scripts/make_test_sets.py --scene Tshape3d_env1 --variant long \
        --n 1000 --out /workspace/testing_data/3dshape --workers 14

Output: sampled_points.npy (N, 12) float32 [start | goal] in the chart (t, rotvec / 2 pi), meta.json (the scene's
standard meta plus the variant's settings), env.npy (copied), verification.json (per-pair constraints, clearances,
RRT path lengths, rejection counts).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scene', required=True, help='<shape>_<env> of testing_data/3dshape (standard) or datasets/3dshape (hole)')
ap.add_argument('--variant', required=True, choices=['long', 'bigrot', 'tight', 'hard', 'through'])
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--out', default='/workspace/testing_data/3dshape')
ap.add_argument('--name', default='', help='output cell name (default <scene>_<variant>)')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--workers', type=int, default=14)
ap.add_argument('--lie-root', default='/lieshape')
ap.add_argument('--graph-root', default='/workspace')
ap.add_argument('--rrt-iters', type=int, default=6000)
ap.add_argument('--rrt-time', type=float, default=120.0)
ap.add_argument('--step', type=float, default=0.05)
ap.add_argument('--resolution', type=float, default=0.005)
ap.add_argument('--sdf-margin', type=float, default=0.003)
ap.add_argument('--grid-h', type=float, default=0.0025)
ap.add_argument('--round', type=int, default=6000, help='placements drawn per round')
ap.add_argument('--max-rounds', type=int, default=60)
args = ap.parse_args()

sys.path.insert(0, args.lie_root)
os.chdir(args.lie_root)
import igl  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from dataprocessing.preprocess_obj import load_obj, sample_surface_points, _rotvec_to_matrix_np  # noqa: E402

TWO_PI = 2.0 * math.pi
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
CLEAR_CAP = 0.03
INF = float('inf')
VARIANTS = {
    'long': dict(offset=0.02, clear=(0.02, INF), dp=(0.6, INF), ang=(0.0, 180.0)),
    'bigrot': dict(offset=0.02, clear=(0.02, INF), dp=(0.0, INF), ang=(150.0, 180.0)),
    'tight': dict(offset=0.005, clear=(0.005, 0.02), dp=(0.0, INF), ang=(0.0, 180.0)),
    'hard': dict(offset=0.005, clear=(0.005, 0.02), dp=(0.5, INF), ang=(135.0, 180.0)),
    'through': dict(offset=0.02, clear=(0.02, INF), dp=(0.0, INF), ang=(0.0, 180.0), side_gap=0.05),
}
V = VARIANTS[args.variant]
name = args.name or '{}_{}'.format(args.scene, args.variant)

# ----------------------------------------------------------------------------- scene
std_dir = os.path.join(args.lie_root, 'testing_data/3dshape', args.scene)
if not os.path.exists(os.path.join(std_dir, 'meta.json')):
    std_dir = os.path.join(args.graph_root, 'datasets/3dshape/data', args.scene)      # hole cells: the training set's meta
meta = json.load(open(os.path.join(std_dir, 'meta.json')))


def _resolve(path):
    for c in (path, os.path.join(std_dir, os.path.basename(path)),
              os.path.join(args.lie_root, 'datasets/3dshape', os.path.basename(path)),
              os.path.join(args.graph_root, 'datasets/3dshape/obj', os.path.basename(path))):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(path)


env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
V_sh, F_sh, _ = load_obj(_resolve(meta['shape_obj']))
shape_V = np.ascontiguousarray((V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / env_scale * float(meta['shape_scale']), dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
V_env, F_env, names_env = load_obj(_resolve(meta['env_obj']))
V_env_n = (V_env - env_center) / env_scale
half_extent = (V_env_n.max(0) - V_env_n.min(0)) * 0.5 - 0.01
clouds, trees = [], []
for k in range(3):
    np.random.seed(1000 * args.seed + k)
    pts = np.ascontiguousarray(sample_surface_points(V_env_n, F_env, 50000), dtype=np.float64)
    clouds.append(pts)
    trees.append(cKDTree(pts))
np.random.seed(args.seed)

# signed-distance grid of the shape (for the planner's conservative motion check)
pad = 0.03
lo = shape_V.min(0) - pad
hi = shape_V.max(0) + pad
h = args.grid_h
dims = np.ceil((hi - lo) / h).astype(int) + 1
gx, gy, gz = [lo[k] + h * np.arange(dims[k]) for k in range(3)]
G = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), -1).reshape(-1, 3)
t0 = time.time()
SDF = igl.signed_distance(np.ascontiguousarray(G), shape_V, shape_F, _SDF_SIGN)[0].reshape(tuple(dims))
print('scene {}: shape radius {:.4f}, half extent {}, sdf grid {} ({:.1f}s)'.format(
    args.scene, shape_radius, np.round(half_extent, 3).tolist(), tuple(dims), time.time() - t0), flush=True)

# wall / hole geometry (through variant)
slab = None
if args.variant == 'through':
    frame = np.array(['frame' in str(n).lower() for n in names_env])
    assert frame.any(), 'through needs a frame (wall-with-hole) environment'
    W = V_env_n[np.unique(F_env[frame])]
    slab = dict(z0=float(W[:, 2].min()), z1=float(W[:, 2].max()))
    print('slab z {:.4f} .. {:.4f}'.format(slab['z0'], slab['z1']), flush=True)


# ----------------------------------------------------------------------------- geometry helpers
def rotvec_to_quat(rv):
    """(N, 3) rotation vectors (radians) -> (N, 4) unit quaternions (w, x, y, z), w >= 0."""
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    ax = rv / np.maximum(th, 1e-12)
    q = np.concatenate([np.cos(th / 2), np.sin(th / 2) * ax], -1)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def quat_to_rotvec(q):
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)
    v = q[:, 1:]
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    th = 2.0 * np.arctan2(nv[:, 0], q[:, 0])
    return th[:, None] * v / np.maximum(nv, 1e-12)


def quat_to_matrix(q):
    return Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], -1)).as_matrix()


def chart_to_state(X):
    """(N, 6) chart (t, rotvec / 2 pi) -> (N, 7) (t, q)."""
    return np.concatenate([X[:, :3], rotvec_to_quat(X[:, 3:6] * TWO_PI)], 1)


def state_to_chart(S):
    return np.concatenate([S[:, :3], quat_to_rotvec(S[:, 3:7]) / TWO_PI], 1)


def quat_angle(qa, qb):
    d = np.abs((qa * qb).sum(-1)).clip(max=1.0)
    return 2.0 * np.arccos(d)


def sdist(A, B):
    """metric between (N, 7) states (broadcast): sqrt(|dp|^2 + (theta / 2 pi)^2)."""
    dp = np.linalg.norm(A[..., :3] - B[..., :3], axis=-1)
    th = quat_angle(A[..., 3:7], B[..., 3:7])
    return np.sqrt(dp ** 2 + (th / TWO_PI) ** 2)


def interp(a, b, ts):
    """states a, b (7,), ts (K,) -> (K, 7): linear translation, slerp rotation (shortest arc)."""
    qa, qb = a[3:7], b[3:7].copy()
    if (qa * qb).sum() < 0:
        qb = -qb
    d = float(np.clip((qa * qb).sum(), -1.0, 1.0))
    om = math.acos(d)
    ts = np.asarray(ts, dtype=np.float64)[:, None]
    if om < 1e-6:
        q = qa[None] * (1 - ts) + qb[None] * ts
    else:
        q = (np.sin((1 - ts) * om) * qa[None] + np.sin(ts * om) * qb[None]) / math.sin(om)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.concatenate([a[None, :3] * (1 - ts) + b[None, :3] * ts, q], 1)


def grid_free(S, margin=args.sdf_margin):
    """conservative collision test of (K, 7) states on the SDF grid: True = free with margin."""
    R = quat_to_matrix(S[:, 3:7])
    sel = trees[0].query_ball_point(S[:, :3], r=shape_radius + margin)
    out = np.ones(len(S), bool)
    for k in range(len(S)):
        if not sel[k]:
            continue
        P = (clouds[0][sel[k]] - S[k, :3]) @ R[k]
        c = ((P - lo) / h).T
        d = map_coordinates(SDF, c, order=1, mode='nearest')
        out[k] = d.min() >= margin
    return out


def exact_clear(X, tree_pts=0):
    """exact clearance (capped) of (K, 6) chart poses against cloud tree_pts."""
    X = np.asarray(X, dtype=np.float64)
    out = np.full(len(X), CLEAR_CAP)
    tree, pts = trees[tree_pts], clouds[tree_pts]
    for k, sel in enumerate(tree.query_ball_point(X[:, :3], r=shape_radius + CLEAR_CAP)):
        if not sel:
            continue
        R = _rotvec_to_matrix_np(X[k, 3:6] * TWO_PI)
        S = igl.signed_distance(np.ascontiguousarray((pts[sel] - X[k, :3]) @ R), shape_V, shape_F, _SDF_SIGN)[0]
        out[k] = min(float(S.min()), CLEAR_CAP)
    return out


def exact_free_path(S, res=args.resolution):
    """re-walk a state path (M, 7) at resolution res with the exact test; returns (ok, min clearance, n poses)."""
    poses = [S[:1]]
    for i in range(len(S) - 1):
        d = float(sdist(S[i], S[i + 1]))
        k = max(1, int(math.ceil(d / res)))
        poses.append(interp(S[i], S[i + 1], np.arange(1, k + 1) / k))
    P = state_to_chart(np.concatenate(poses))
    c = exact_clear(P)
    return bool((c >= 0).all()), float(c.min()), len(P)


# ----------------------------------------------------------------------------- sampling
def sample_placements(n, rng):
    X = np.empty((n, 6))
    for k in range(3):
        X[:, k] = rng.uniform(-half_extent[k], half_extent[k], n)
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True) + 1e-12
    X[:, 3:6] = ax * rng.uniform(0, math.pi, (n, 1)) / TWO_PI
    return X


def placement_ok(X):
    """(clearance on cloud 0, accepted) for (K, 6) chart placements."""
    c0 = exact_clear(X, 0)
    ok = (c0 >= V['clear'][0]) & (c0 <= V['clear'][1])
    for t in (1, 2):
        idx = np.where(ok)[0]
        if len(idx):
            ok[idx] &= exact_clear(X[idx], t) >= V['offset']
    return c0, ok


def pair_ok(xs, xg):
    dp = float(np.linalg.norm(xg[:3] - xs[:3]))
    ang = float(quat_angle(rotvec_to_quat(xs[None, 3:6] * TWO_PI), rotvec_to_quat(xg[None, 3:6] * TWO_PI))[0]) * 180 / math.pi
    if not (V['dp'][0] <= dp <= V['dp'][1] and V['ang'][0] <= ang <= V['ang'][1]):
        return None
    if slab is not None:
        g = V['side_gap']
        a, b = xs[2], xg[2]
        if not ((a <= slab['z0'] - g and b >= slab['z1'] + g) or (b <= slab['z0'] - g and a >= slab['z1'] + g)):
            return None
    return dp, ang


# ----------------------------------------------------------------------------- RRT-Connect
class Tree:
    def __init__(self, root):
        self.S = [root]
        self.parent = [-1]
        self.arr = np.asarray([root])

    def add(self, s, parent):
        self.S.append(s)
        self.parent.append(parent)
        self.arr = np.asarray(self.S)
        return len(self.S) - 1

    def nearest(self, s):
        return int(np.argmin(sdist(self.arr, s[None])))

    def path_to_root(self, i):
        out = []
        while i >= 0:
            out.append(self.S[i])
            i = self.parent[i]
        return out


def edge_free(a, b):
    d = float(sdist(a, b))
    k = max(1, int(math.ceil(d / args.resolution)))
    return bool(grid_free(interp(a, b, np.arange(1, k + 1) / k)).all())


def extend(tree, target):
    i = tree.nearest(target)
    near = tree.S[i]
    d = float(sdist(near, target))
    if d <= args.step:
        new, reached = target, True
    else:
        new, reached = interp(near, target, [args.step / d])[0], False
    if not edge_free(near, new):
        return 'trapped', -1
    j = tree.add(new, i)
    return ('reached' if reached else 'advanced'), j


def connect(tree, target):
    while True:
        st, j = extend(tree, target)
        if st != 'advanced':
            return st, j


def rrt_connect(s0, s1, rng, iters=args.rrt_iters, tlim=args.rrt_time):
    """path (M, 7) from s0 to s1 or None; also the iteration count."""
    if edge_free(s0, s1):
        return np.stack([s0, s1]), 0
    ta, tb = Tree(s0), Tree(s1)
    swapped = False
    t0 = time.time()
    for it in range(iters):
        r = np.empty(7)
        for k in range(3):
            r[k] = rng.uniform(-half_extent[k], half_extent[k])
        q = rng.normal(size=4)
        r[3:7] = q / np.linalg.norm(q)
        st, j = extend(ta, r)
        if st != 'trapped':
            st2, j2 = connect(tb, ta.S[j])
            if st2 == 'reached':
                pa = ta.path_to_root(j)[::-1]
                pb = tb.path_to_root(j2)
                path = pa + pb[1:]
                if swapped:
                    path = path[::-1]
                return np.asarray(path), it + 1
        ta, tb = tb, ta
        swapped = not swapped
        if time.time() - t0 > tlim:
            break
    return None, iters


def path_stats(P):
    d = sdist(P[:-1], P[1:])
    tr = np.linalg.norm(P[1:, :3] - P[:-1, :3], axis=1).sum()
    rot = quat_angle(P[:-1, 3:7], P[1:, 3:7]).sum() / TWO_PI
    return float(d.sum()), float(tr), float(rot)


# aligned passage through the hole (through variant): orientation candidates with the plate parallel to the slab
PASSAGE = None
if slab is not None:
    best = None
    zc = 0.5 * (slab['z0'] + slab['z1'])
    reach = shape_radius + 0.5 * (slab['z1'] - slab['z0']) + 0.03
    for sgn in (1.0, -1.0):
        for phi in (0.0, 90.0, 180.0, 270.0):
            R = Rotation.from_euler('z', phi, degrees=True) * Rotation.from_euler('x', sgn * 90.0, degrees=True)
            rv = R.as_rotvec()
            zs = np.linspace(zc - reach, zc + reach, int(math.ceil(2 * reach / 0.0025)) + 1)
            X = np.zeros((len(zs), 6))
            X[:, 2] = zs
            X[:, 3:6] = rv / TWO_PI
            c = exact_clear(X)
            if best is None or c.min() > best[0]:
                best = (float(c.min()), rv, zc - reach, zc + reach)
    assert best[0] > 0, 'no aligned orientation passes the hole (min clearance {:.4f})'.format(best[0])
    PASSAGE = best
    print('aligned passage: min clearance {:.4f}, rotvec {}'.format(best[0], np.round(best[1], 3).tolist()), flush=True)


def verify_pair(job):
    """job = (idx, xs, xg) chart poses -> dict(ok, stats...)."""
    idx, xs, xg = job
    rng = np.random.default_rng(args.seed * 100003 + idx)
    s0, s1 = chart_to_state(xs[None])[0], chart_to_state(xg[None])[0]
    t0 = time.time()
    if PASSAGE is None:
        P, it = rrt_connect(s0, s1, rng)
    else:
        q = rotvec_to_quat(PASSAGE[1][None])[0]
        a = np.concatenate([[0.0, 0.0, PASSAGE[2]], q])
        b = np.concatenate([[0.0, 0.0, PASSAGE[3]], q])
        if s0[2] > s1[2]:
            a, b = b, a
        P1, it1 = rrt_connect(s0, a, rng)
        P2, it2 = rrt_connect(b, s1, rng) if P1 is not None else (None, 0)
        P = None if (P1 is None or P2 is None) else np.concatenate([P1, P2])
        it = it1 + it2
    if P is None:
        return dict(idx=idx, ok=False, reason='rrt', time=time.time() - t0, iters=it)
    ok, cmin, npose = exact_free_path(P)
    L, tr, rot = path_stats(P)
    # straight interpolation: how much of it collides (metadata)
    ks = np.arange(1, 26) / 26.0
    cs = exact_clear(state_to_chart(interp(s0, s1, ks)))
    return dict(idx=idx, ok=ok, reason='' if ok else 'recheck', time=time.time() - t0, iters=it, path_len=L,
                path_trans=tr, path_rot_turns=rot, path_min_clear=cmin, n_states=len(P),
                straight_blocked=float((cs < 0).mean()), straight_min_clear=float(cs.min()))


# ----------------------------------------------------------------------------- main loop
import multiprocessing as mp  # noqa: E402

rng = np.random.default_rng(args.seed)
pool = mp.get_context('fork').Pool(args.workers)
pairs, info = [], []
stats = dict(placements_drawn=0, placements_in_band=0, pairs_formed=0, rrt_failed=0, recheck_failed=0, rounds=0)
t_start = time.time()
try:
    for rnd in range(args.max_rounds):
        if len(pairs) >= args.n:
            break
        stats['rounds'] += 1
        X = sample_placements(args.round, rng)
        stats['placements_drawn'] += len(X)
        chunks = [X[i:i + 100] for i in range(0, len(X), 100)]
        res = pool.map(placement_ok, chunks)
        c0 = np.concatenate([r[0] for r in res])
        ok = np.concatenate([r[1] for r in res])
        Xb, cb = X[ok], c0[ok]
        stats['placements_in_band'] += len(Xb)
        # pair the accepted placements (each used once): random partner search
        perm = rng.permutation(len(Xb))
        used = np.zeros(len(Xb), bool)
        jobs, meta_pairs = [], []
        for ii in perm:
            if used[ii]:
                continue
            cand = rng.permutation(len(Xb))[:200]
            for jj in cand:
                if jj == ii or used[jj]:
                    continue
                pk = pair_ok(Xb[ii], Xb[jj])
                if pk is not None:
                    used[ii] = used[jj] = True
                    jobs.append((stats['pairs_formed'], Xb[ii], Xb[jj]))
                    meta_pairs.append(dict(dp=pk[0], rel_angle_deg=pk[1], clear_start=float(cb[ii]), clear_goal=float(cb[jj])))
                    stats['pairs_formed'] += 1
                    break
        need = args.n - len(pairs)
        jobs, meta_pairs = jobs[:max(need + 20, int(need * 1.3))], meta_pairs[:max(need + 20, int(need * 1.3))]
        t0 = time.time()
        results = {}
        for r in pool.imap_unordered(verify_pair, jobs, chunksize=1):
            results[r['idx']] = r
        for (idx, xs, xg), mp_ in zip(jobs, meta_pairs):
            r = results[idx]
            if not r['ok']:
                stats['rrt_failed' if r['reason'] == 'rrt' else 'recheck_failed'] += 1
                continue
            if len(pairs) >= args.n:
                break
            pairs.append(np.concatenate([xs, xg]))
            d = dict(mp_)
            d.update({k: v for k, v in r.items() if k not in ('idx', 'ok', 'reason')})
            info.append(d)
        print('[round {}] drawn {} in-band {} formed {} verified {}/{} ({:.0f}s)  banked {}/{}  rrt-fail {} recheck-fail {}'.format(
            rnd + 1, len(X), len(Xb), len(jobs), sum(r['ok'] for r in results.values()), len(jobs), time.time() - t0,
            len(pairs), args.n, stats['rrt_failed'], stats['recheck_failed']), flush=True)
finally:
    pool.terminate()
    pool.join()

n_final = len(pairs)
if n_final < args.n:
    print('WARNING: only {} of {} pairs after {} rounds'.format(n_final, args.n, stats['rounds']), flush=True)
out_dir = os.path.join(args.out, name)
tmp_dir = out_dir + '.tmp'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)
np.save(os.path.join(tmp_dir, 'sampled_points.npy'), np.asarray(pairs, dtype=np.float32))
if os.path.exists(os.path.join(std_dir, 'env.npy')):
    shutil.copy2(os.path.join(std_dir, 'env.npy'), os.path.join(tmp_dir, 'env.npy'))
m = dict(meta)
m.update({'testing_data': True, 'offset': float(V['offset']), 'variant': args.variant, 'scene': args.scene,
          'constraints': {k: (list(v) if isinstance(v, tuple) else v) for k, v in V.items()},
          'rrt_verified': True, 'generator': 'models/graph_homotopy/scripts/make_test_sets.py', 'seed': args.seed,
          'shape_obj': _resolve(meta['shape_obj']), 'env_obj': _resolve(meta['env_obj'])})
with open(os.path.join(tmp_dir, 'meta.json'), 'w') as fh:
    json.dump(m, fh, indent=2, default=float)
ver = dict(pairs_written=n_final, pairs_requested=args.n, stats=stats, settings=vars(args),
           sdf_margin=args.sdf_margin, rrt=dict(step=args.step, resolution=args.resolution, iters=args.rrt_iters, time=args.rrt_time),
           passage=None if PASSAGE is None else dict(min_clear=PASSAGE[0], rotvec=PASSAGE[1].tolist(), z=[PASSAGE[2], PASSAGE[3]]),
           wall_clock_s=time.time() - t_start, per_pair=info)
with open(os.path.join(tmp_dir, 'verification.json'), 'w') as fh:
    json.dump(ver, fh, indent=1, default=float)
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.rename(tmp_dir, out_dir)
if info:
    a = np.array([d['rel_angle_deg'] for d in info]); dp = np.array([d['dp'] for d in info])
    cl = np.array([min(d['clear_start'], d['clear_goal']) for d in info]); pl = np.array([d['path_len'] for d in info])
    print('wrote {} pairs to {} ({:.0f}s): rel angle p50 {:.0f} p90 {:.0f}; |dp| p50 {:.2f}; min clearance p50 {:.4f}; '
          'rrt path length p50 {:.2f}; straight blocked frac mean {:.2f}'.format(
              n_final, out_dir, time.time() - t_start, np.median(a), np.percentile(a, 90), np.median(dp), np.median(cl),
              np.median(pl), np.mean([d['straight_blocked'] for d in info])), flush=True)
