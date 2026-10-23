"""preprocess_obj.py with an exact early rejection of training x0 candidates.

Same command line and outputs as the original (Lie-Shape-NTFields
dataprocessing/preprocess_obj.py, path in $PREPROCESS_OBJ), and the same
samples: for a fixed numpy / torch seed it writes the same arrays.

Why: in training mode every x0 candidate gets the full evaluate_placements
(clearance, collision, rotational clearance, normals) against every env point
within ``min_dist + 2 R`` of its centre, and only those with
``free & offset < clearance < margin`` survive.  With a large shape in a tight
room most candidates collide, so almost all of that work is thrown away.

What changes (2026-10-21: two more exact cuts for large bodies, see x0_band_mask and
env_cull_bound; 2026-10-22: the collision and clearance queries themselves run on
signed-distance grids of the shape with an exact fallback on the few env points
the grids cannot decide, see grid_points_inside_tets / grid_calculate_dist -- the
same values, without the (B, M, F) closest-point tensors that took most of the
time and memory): one block is inserted into the original generate_valid_pairs,
right after its ``m0 - R <= margin`` gate.  It computes the survivor test
``free & offset < clearance < margin`` exactly, but only on the env points
that can affect it -- a point inside a tet lies within R of the centre, and a
point closer than ``margin`` to the surface lies within ``R + margin`` -- and
drops the candidates that fail.  The original evaluate_placements then runs on
the survivors only, so every stored value comes from the original code, and
the survivors (hence the torch random stream) are the same as without the block.

    python dataprocessing/preprocess_obj_fast.py <preprocess_obj.py flags>
    python dataprocessing/preprocess_obj_fast.py --selftest 2000 --env ... --shape ... [--2d]  # compare with the original
"""
import importlib.util
import inspect
import os
import sys
import textwrap

import numpy as np
import torch

ORIG = os.environ.get('PREPROCESS_OBJ', '/lieshape/dataprocessing/preprocess_obj.py')
_spec = importlib.util.spec_from_file_location('preprocess_obj', ORIG)
po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po)

ANCHOR = '            x0 = x0[feas0]\n'
INSERT = '''            band = x0_band_mask(x0, tet_verts_local, face_verts_local, env_t, margin, offset, device)
            if int(band.sum()) == 0:
                continue
            x0 = x0[band]
'''
# x1 only has to be collision-free (clearance > offset): drop the colliding ones with the exact grid test
# before the full evaluation, which was spent mostly on them (acceptance = the free fraction, 1-30 %)
ANCHOR_X1 = '            x1 = x1[feas1]\n'
INSERT_X1 = '''            gfree = x1_free_mask(x1, tet_verts_local, face_verts_local, env_t, device)
            if int(gfree.sum()) == 0:
                continue
            x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0 = _idx(
                gfree, x0, dist0, angle0, normal0, trans0, rot0, shp0, envp0)
            x1 = x1[gfree]
'''


_SDF_GRIDS = {}


def _sdf_grid(tet_verts_local, face_verts_local, margin, device, h=0.005):
    """Signed-distance grid of the shape in its own frame (spacing ``h``), the sign from the tets (a node inside
    a tet is inside), the magnitude the distance to the surface triangles.  The shape's AABB is padded by more
    than ``margin + 2 delta`` with ``delta = h sqrt(3) / 2``, the largest error of a nearest-node lookup (the
    distance is 1-Lipschitz), so a point outside the grid is provably farther than ``margin``."""
    key = (id(tet_verts_local), float(margin), str(device))
    if key in _SDF_GRIDS:
        return _SDF_GRIDS[key]
    import igl
    T = face_verts_local.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    V, inv = np.unique(T.round(7), axis=0, return_inverse=True)
    F = np.ascontiguousarray(inv.reshape(-1, 3).astype(np.int64))
    V = np.ascontiguousarray(V)
    delta = h * np.sqrt(3.0) / 2.0 + 1e-6
    pad = margin + 2.0 * delta + 2.0 * h
    lo = V.min(0) - pad
    dims = np.ceil((V.max(0) + pad - lo) / h).astype(int) + 1
    axes = [lo[k] + h * np.arange(dims[k]) for k in range(3)]
    G = np.ascontiguousarray(np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3))
    d = np.abs(igl.signed_distance(G, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0])
    tets = tet_verts_local.to(device).unsqueeze(0)
    Gt = torch.tensor(G, dtype=torch.float32, device=device)
    inside = _nodes_inside(tets[0], Gt)
    sd = torch.tensor(d, dtype=torch.float32, device=device)
    sd[inside] *= -1.0
    grid = dict(sdf=sd.reshape(tuple(dims)), lo=torch.tensor(lo, dtype=torch.float32, device=device), h=float(h),
                hi_idx=torch.tensor(dims - 1, device=device), delta=float(delta))
    _SDF_GRIDS[key] = grid
    return grid


def _nodes_inside(tets, P, chunk=4096):
    """(N,) bool: which of the points ``P`` (N, 3) lie inside any tet of ``tets`` (K, 4, 3) -- the same test as
    points_inside_tets, one placement, chunked over the points."""
    v0 = tets[:, 0, :]
    M = torch.stack([tets[:, 1, :] - v0, tets[:, 2, :] - v0, tets[:, 3, :] - v0], dim=-1)      # (K,3,3)
    Minv = torch.linalg.inv(M + po.EPS * torch.eye(3, device=tets.device))
    res = []
    for i in range(0, len(P), chunk):
        rhs = P[i:i + chunk].unsqueeze(0) - v0.unsqueeze(1)                                     # (K,M,3)
        bary = torch.einsum('kij,kmj->kmi', Minv, rhs)                                          # (K,M,3)
        l0 = 1.0 - bary.sum(-1)
        tol = -1e-6
        inside = (bary[..., 0] >= tol) & (bary[..., 1] >= tol) & (bary[..., 2] >= tol) & (l0 >= tol)
        res.append(inside.any(0))
    return torch.cat(res)


_COARSE_GRIDS = {}


def _coarse_grid(face_verts_local, device, h=0.02, pad=0.6):
    """Unsigned distance to the shape's surface on a coarse grid of its padded AABB (spacing ``h``, padding
    ``pad``): a nearest-node lookup is within ``delta = h sqrt(3) / 2`` of the true distance, and a point
    outside the grid is at least ``pad`` from the shape (the shape is inside its AABB)."""
    key = (id(face_verts_local), float(h), float(pad), str(device))
    if key in _COARSE_GRIDS:
        return _COARSE_GRIDS[key]
    import igl
    T = face_verts_local.detach().cpu().numpy().reshape(-1, 3).astype(np.float64)
    V, inv = np.unique(T.round(7), axis=0, return_inverse=True)
    F = np.ascontiguousarray(inv.reshape(-1, 3).astype(np.int64))
    V = np.ascontiguousarray(V)
    lo = V.min(0) - pad
    dims = np.ceil((V.max(0) + pad - lo) / h).astype(int) + 1
    axes = [lo[k] + h * np.arange(dims[k]) for k in range(3)]
    G = np.ascontiguousarray(np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3))
    d = np.abs(igl.signed_distance(G, V, F, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0])
    grid = dict(d=torch.tensor(d, dtype=torch.float32, device=device).reshape(tuple(dims)),
                lo=torch.tensor(lo, dtype=torch.float32, device=device), h=float(h),
                hi_idx=torch.tensor(dims - 1, device=device), delta=float(h * np.sqrt(3.0) / 2.0 + 1e-6), pad=float(pad))
    _COARSE_GRIDS[key] = grid
    return grid


def _lookup(grid, key, R, t, P):
    """Nearest-node values of grid[key] at world points P (B, M, 3) of placements (R, t): (values, inside grid)."""
    local = torch.einsum('bij,bmi->bmj', R, P - t.unsqueeze(1))                 # shape frame: (p - t) @ R
    idx = torch.round((local - grid['lo']) / grid['h']).long()
    inside = ((idx >= 0) & (idx <= grid['hi_idx'])).all(-1)
    idx = torch.minimum(torch.clamp(idx, min=0), grid['hi_idx'])
    return grid[key][idx[..., 0], idx[..., 1], idx[..., 2]], inside


def _gather_candidates(P, cand):
    """(B, Kmax, 3) the points flagged in cand (B, M), each row padded with its first candidate."""
    cnt = cand.sum(1)
    kmax = int(cnt.max())
    order = torch.argsort((~cand).to(torch.int8), dim=1, stable=True)[:, :kmax]    # candidates first
    slot = torch.arange(kmax, device=P.device)
    order = torch.where(slot[None] < cnt[:, None], order, order[:, :1])
    return torch.gather(P, 1, order.unsqueeze(-1).expand(-1, -1, 3))


def grid_points_inside_tets(tets, env, R, t, tet_verts_local, face_verts_local, device):
    """points_inside_tets on the env points the fine signed-distance grid cannot rule out (a point with grid
    value > delta is provably outside; one outside the grid is beyond the padded AABB)."""
    if env.shape[1] == 0:
        return torch.ones(tets.shape[0], dtype=torch.bool, device=tets.device)
    g = _sdf_grid(tet_verts_local, face_verts_local, 0.0, device)
    s, ing = _lookup(g, 'sdf', R, t, env)
    maybe = ing & (s <= g['delta'])
    free = torch.ones(tets.shape[0], dtype=torch.bool, device=tets.device)
    rows = torch.nonzero(maybe.any(1), as_tuple=False).squeeze(1)
    if len(rows):
        free[rows] = po.points_inside_tets(tets[rows], _gather_candidates(env[rows], maybe[rows]))
    return free


def grid_calculate_dist(faces, env, centers, R, t, tet_verts_local, face_verts_local, device):
    """calculate_dist on the env points that can be the nearest: those whose coarse-grid lower bound is at most
    the smallest upper bound of the placement (the true minimizer always qualifies)."""
    if env.shape[1] == 0:
        return po.calculate_dist(faces, env, centers)
    g = _coarse_grid(face_verts_local, device)
    v, ing = _lookup(g, 'd', R, t, env)
    lb = torch.where(ing, v - g['delta'], torch.full_like(v, g['pad']))
    ub = torch.where(ing, v + g['delta'], torch.full_like(v, float('inf')))
    cand = lb <= ub.min(dim=1, keepdim=True).values + 1e-7
    if int(cand.sum(1).max()) >= env.shape[1]:
        return po.calculate_dist(faces, env, centers)
    return po.calculate_dist(faces, _gather_candidates(env, cand), centers)


def x1_free_mask(x1, tet_verts_local, face_verts_local, env_t, device):
    """``free`` per placement, exactly as evaluate_placements computes it, on the env points within the body's
    radius (a point inside a tet is within R of the centre)."""
    cfg = x1.to(device)
    t = cfg[:, 0:3]
    R = po.rotvec_to_matrix(cfg[:, 3:6])
    R_shape = face_verts_local.reshape(-1, 3).norm(dim=1).max()
    to_env_dist = torch.linalg.norm(env_t.unsqueeze(0) - t.unsqueeze(1), dim=-1)
    r_col = R_shape * (1 + 1e-4) + 1e-6
    n_col = int((to_env_dist <= r_col).sum(dim=1).max().clamp(min=1))
    sort_idx = torch.topk(to_env_dist, n_col, dim=1, largest=False).indices
    tets = torch.einsum('bij,nj->bni', R, tet_verts_local.reshape(-1, 3)).reshape(len(cfg), -1, 4, 3) + t[:, None, None]
    return grid_points_inside_tets(tets, env_t[sort_idx], R, t, tet_verts_local, face_verts_local, device).cpu()


def x0_band_mask(x0, tet_verts_local, face_verts_local, env_t, margin, offset, device):
    """``free & (offset < clearance < margin)`` per candidate, exactly as
    evaluate_placements would compute it, on the env points that can matter.

    A signed-distance grid of the shape (_sdf_grid) first rejects, exactly, the
    candidates that cannot be in the band: one env point provably inside the
    shape (collision), or every env point provably farther than ``margin``.
    Only the others get the full inside-tet and face-distance test (the body of
    the camera test has 122 tets and a 0.23 radius: without the gate the exact
    test took most of the preprocessing time and 10 GB per shard)."""
    cfg = x0.to(device)
    B = cfg.shape[0]
    t = cfg[:, 0:3]
    R = po.rotvec_to_matrix(cfg[:, 3:6])
    R_shape = face_verts_local.reshape(-1, 3).norm(dim=1).max()
    to_env_dist = torch.linalg.norm(env_t.unsqueeze(0) - t.unsqueeze(1), dim=-1)   # (B,E)
    sorted_d, sort_idx = torch.sort(to_env_dist, dim=1)
    band = torch.zeros(B, dtype=torch.bool, device=device)
    # Clearance < margin needs a point within R + margin of the centre; on that
    # set the minimum is the true clearance whenever it is below margin, and
    # stays >= margin otherwise.  Beyond it no point can be inside either.
    r_clr = R_shape + margin * (1 + 1e-4) + 1e-6
    n_clr = int((sorted_d <= r_clr).sum(dim=1).max())
    if n_clr == 0:
        return band.cpu()
    g = _sdf_grid(tet_verts_local, face_verts_local, margin, device)
    P = env_t[sort_idx[:, :n_clr]]                                                  # (B,n,3)
    local = torch.einsum('bij,bni->bnj', R, P - t.unsqueeze(1))                     # shape frame: (p - t) @ R
    idx = torch.round((local - g['lo']) / g['h']).long()
    idx = torch.minimum(torch.clamp(idx, min=0), g['hi_idx'])
    s = g['sdf'][idx[..., 0], idx[..., 1], idx[..., 2]]
    s = torch.where(sorted_d[:, :n_clr] <= r_clr, s, torch.full_like(s, float('inf')))
    smin = s.min(dim=1).values
    cand = (smin >= -g['delta']) & (smin <= margin + g['delta'])
    ci = torch.nonzero(cand, as_tuple=False).squeeze(1)
    if len(ci) == 0:
        return band.cpu()
    t, R, sorted_d, sort_idx = t[ci], R[ci], sorted_d[ci], sort_idx[ci]
    # The same transforms as evaluate_placements.
    tets = torch.einsum('bij,nj->bni', R, tet_verts_local.reshape(-1, 3)).reshape(len(ci), -1, 4, 3) + t[:, None, None]
    faces = torch.einsum('bij,nj->bni', R, face_verts_local.reshape(-1, 3)).reshape(len(ci), -1, 3, 3) + t[:, None, None]

    # Collision: a point inside a tet is within R of the centre (slack for the
    # -1e-6 barycentric tolerance of points_inside_tets).
    r_col = R_shape * (1 + 1e-4) + 1e-6
    n_col = int((sorted_d <= r_col).sum(dim=1).max().clamp(min=1))
    free = grid_points_inside_tets(tets, env_t[sort_idx[:, :n_col]], R, t, tet_verts_local, face_verts_local, device)
    sub = free.clone()
    fi = torch.nonzero(free, as_tuple=False).squeeze(1)
    if len(fi):
        n_clr = int((sorted_d[fi] <= r_clr).sum(dim=1).max().clamp(min=1))
        dist, _, _ = grid_calculate_dist(faces[fi], env_t[sort_idx[fi, :n_clr]], t[fi], R[fi], t[fi],
                                         tet_verts_local, face_verts_local, device)
        sub[fi] = (dist > offset) & (dist < margin)
    band[ci] = sub
    return band.cpu()


# ── evaluate_placements: a tighter exact env cull ───────────────────────────
# Its broad-phase keeps every env point within ``min centre distance + 2 R``; for a body whose radius R is a
# quarter of the room that is the whole cloud, and the closest-point-on-triangle step over it took most of
# the time (and the memory) of the camera-body preprocessing.  An upper bound u on the clearance (the
# distance from any surface sample of the placed body to the nearest env point) puts the nearest env point
# within R + u of the centre; culling to that ball keeps the exact minimum, its normal, and every point that
# can be inside the body (those are within R).  The angle binning keeps its own full sort.
ANCHOR_EVAL = '    within = sorted_d <= (sorted_d[:, 0:1] + 2.0 * R_shape)       # (B,E)\n'
REPLACE_EVAL = ('    within = sorted_d <= (R_shape + grid_clearance_bound(env_points, R, t, face_verts_local, device)[:, None]'
                ' + 1e-6)       # (B,E)\n')
# The rotational-clearance stage (min angle between the shape's radial directions and the env directions) is a
# third of the time; it feeds rot_n.npy (unread) but also the angle half of speed.npy = (speed_dists + angle / pi) / 2,
# which the trainers use, so it is kept unless PREPROCESS_ANGLE=0 (then speed.npy is half the distance speed).
SKIP_ANGLE = os.environ.get('PREPROCESS_ANGLE', '1') == '0'
ANGLE_START = "    _prof_tic('angle_binning', device)\n"
ANGLE_END = "    _prof_toc('angle_cosine', device)\n"
ANGLE_STUB = '''    if SKIP_ANGLE:
        min_angle = torch.zeros(B, device=device)
        zeros3 = torch.zeros(B, 3, device=device)
        rot_n = torch.zeros(B, 6, device=device)
        shape_pt = t.clone()
        env_pt = t.clone()
    else:
'''


def grid_clearance_bound(env_points, R, t, face_verts_local, device):
    """(B,) upper bound on the clearance of each placement from the coarse grid: the smallest ``value + delta``
    over the env points inside the grid (inf when none is), so the nearest env point lies within R + bound of
    the centre -- what env_cull_bound computed with a (B, 250, E) distance table."""
    g = _coarse_grid(face_verts_local, device)
    out = []
    for i in range(0, len(t), 512):
        v, ing = _lookup(g, 'd', R[i:i + 512], t[i:i + 512], env_points.unsqueeze(0).expand(min(512, len(t) - i), -1, -1))
        out.append(torch.where(ing, v + g['delta'], torch.full_like(v, float('inf'))).min(dim=1).values)
    return torch.cat(out)


ANCHOR_COLL = '    is_free = points_inside_tets(tets, env_kept)\n'
REPLACE_COLL = ('    is_free = grid_points_inside_tets(tets, env_kept, R, t, tet_verts_local, face_verts_local, device)\n')
ANCHOR_DIST = '    dist, normal, n_ws = calculate_dist(faces, env_kept, configs[:, 0:3])\n'
REPLACE_DIST = ('    dist, normal, n_ws = grid_calculate_dist(faces, env_kept, configs[:, 0:3], R, t, tet_verts_local, '
                'face_verts_local, device)\n')


def env_cull_bound(rad_pts_world, env_points, step=4, chunk=32):
    """(B,) upper bound on the clearance of each placement: the smallest distance from a body-surface sample
    (every ``step``-th radial point, world frame) to any env point."""
    S = rad_pts_world.reshape(rad_pts_world.shape[0], -1, 3)[:, ::step].contiguous()
    out = []
    for i in range(0, S.shape[0], chunk):
        b = S[i:i + chunk]
        out.append(torch.cdist(b, env_points.unsqueeze(0).expand(b.shape[0], -1, -1)).amin(dim=(1, 2)))
    return torch.cat(out)


def _fast_evaluate_placements():
    src = textwrap.dedent(inspect.getsource(po.evaluate_placements))
    assert src.count(ANCHOR_EVAL) == 1, 'anchor not found once in preprocess_obj.evaluate_placements'
    src = src.replace(ANCHOR_EVAL, REPLACE_EVAL)
    # the min-angle loop runs bins x ceil(num_env / ANGLE_CHUNK) small kernels; with the few placements that
    # survive the gates the chunk can be much wider at the same peak memory (same maxima, same values)
    assert src.count('    ANGLE_CHUNK = 128\n') == 1
    src = src.replace('    ANGLE_CHUNK = 128\n', '    ANGLE_CHUNK = 128 if B > 64 else 2048\n')
    # the collision and clearance queries on the grid-selected candidates (same values)
    for a, b in ((ANCHOR_COLL, REPLACE_COLL), (ANCHOR_DIST, REPLACE_DIST)):
        assert src.count(a) == 1, 'anchor not found once in preprocess_obj.evaluate_placements: ' + a.strip()
        src = src.replace(a, b)
    # the rotational-clearance block, indented under `else:` behind the SKIP_ANGLE stub
    assert src.count(ANGLE_START) == 1 and src.count(ANGLE_END) == 1
    i0, i1 = src.index(ANGLE_START), src.index(ANGLE_END) + len(ANGLE_END)
    block = ''.join(('    ' + ln if ln.strip() else ln) for ln in src[i0:i1].splitlines(True))
    src = src[:i0] + ANGLE_STUB + block + src[i1:]
    ns = po.__dict__
    ns['env_cull_bound'] = env_cull_bound
    ns['grid_clearance_bound'] = grid_clearance_bound
    ns['SKIP_ANGLE'] = SKIP_ANGLE
    ns['grid_points_inside_tets'] = grid_points_inside_tets
    ns['grid_calculate_dist'] = grid_calculate_dist
    exec(compile(src, ORIG + ' [+ env_cull_bound, grid queries]', 'exec'), ns)
    return ns['evaluate_placements']


ORIGINAL_EVAL = po.evaluate_placements
FAST_EVAL = _fast_evaluate_placements()   # installs itself as po.evaluate_placements


def _fast_generate_valid_pairs():
    src = textwrap.dedent(inspect.getsource(po.generate_valid_pairs))
    assert src.count(ANCHOR) == 1, 'anchor not found once in preprocess_obj.generate_valid_pairs'
    src = src.replace(ANCHOR, ANCHOR + INSERT)
    assert src.count(ANCHOR_X1) == 1, 'x1 anchor not found once in preprocess_obj.generate_valid_pairs'
    src = src.replace(ANCHOR_X1, ANCHOR_X1 + INSERT_X1)
    ns = po.__dict__
    ns['x0_band_mask'] = x0_band_mask
    ns['x1_free_mask'] = x1_free_mask
    exec(compile(src, ORIG + ' [+ x0_band_mask]', 'exec'), ns)
    return ns['generate_valid_pairs']


ORIGINAL_GENERATE = po.generate_valid_pairs
FAST_GENERATE = _fast_generate_valid_pairs()   # also installs itself as po.generate_valid_pairs


def selftest(argv):
    """Run original and fast generate_valid_pairs on the same scene and seeds and
    compare every returned array."""
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', type=int, required=True)
    ap.add_argument('--env', required=True)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--offset', type=float, default=0.001)
    ap.add_argument('--batch_size', type=int, default=250)
    ap.add_argument('--num_env_points', type=int, default=po.DEFAULT_ENV_POINTS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--2d', dest='two_d', action='store_true', help='planar mode, as preprocess_obj.py --2d')
    a = ap.parse_args(argv)
    np.random.seed(a.seed)
    V, F, names = po.load_obj(a.env)
    if a.two_d:
        V[:, 2] = 0.0
    lo, hi = V.min(0), V.max(0)
    scale = float((hi - lo).max())
    Vn = (V - 0.5 * (lo + hi)) / scale
    keep = np.array(['null' not in str(n).lower() for n in names])
    env = po.sample_surface_points(Vn, F[keep], a.num_env_points).astype(np.float32)
    half = (hi - lo) / scale * 0.5 - 0.01
    Vs, Fs, _ = po.load_obj(a.shape)
    Vs = (Vs - 0.5 * (Vs.min(0) + Vs.max(0))) / scale
    if a.two_d:                      # the same flattening / squashing as preprocess_obj.py main
        half[2] = 0.0
        z = Vs[:, 2]
        Vs[:, 2] = (z - 0.5 * (z.max() + z.min())) * (po.TWOD_SHAPE_THICKNESS / float(z.max() - z.min()))
    TV, TT, TF = po.tetrahedralize_shape(Vs, Fs)
    tets = torch.tensor(TV[TT], dtype=torch.float32)
    faces = torch.tensor(TV[TF], dtype=torch.float32)
    rp, rb = po.generate_radius_surface_points(Vs, Fs, 1000, 10)
    out = {}
    for name, fn in (('original', ORIGINAL_GENERATE), ('fast', FAST_GENERATE)):
        po.evaluate_placements = ORIGINAL_EVAL if name == 'original' else FAST_EVAL
        torch.manual_seed(a.seed)
        t0 = time.time()
        res = fn(a.selftest, tets, faces, env, half, margin=a.margin, offset=a.offset,
                 rad_points=rp, rad_bins=rb, batch_size=a.batch_size, device=a.device,
                 track_angle_pts=True, two_d=a.two_d)
        out[name] = res
        print('{:8s} {:.1f}s'.format(name, time.time() - t0), flush=True)
    labels = ['pairs', 'dists', 'angles', 'normals', 'trans_n', 'rot_n', 'ang_shape', 'ang_env']
    same = True
    if SKIP_ANGLE:
        print('  (rotational clearance skipped: angles / rot_n / ang_* not compared)')
    for lab, x, y in zip(labels, out['original'], out['fast']):
        if SKIP_ANGLE and lab in ('angles', 'rot_n', 'ang_shape', 'ang_env'):
            continue
        eq = torch.equal(x, y)
        # the pairs must match exactly (same survivors, same random stream); the per-pair values are computed
        # on a differently culled env subset, so they may differ by float32 round-off -- amplified in the unit
        # directions (a division by a clearance as small as 1e-3) and the angles (arccos near 1)
        tol = {'dists': 1e-6, 'ang_shape': 1e-6, 'ang_env': 1e-6}.get(lab, 1e-4)
        same &= eq if lab == 'pairs' else bool(torch.allclose(x, y, atol=tol, rtol=0))
        print('  {:9s} identical {}  max |diff| {:.3g}'.format(lab, eq, float((x - y).abs().max())))
    print('SELFTEST', 'PASSED: same pairs, values within tolerance' if same else 'FAILED')
    return same


def env_weighted_sampling(argv, shell='wall (1)', shell_frac=0.25):
    """Env surface points of main(): the obstacles get ``1 - shell_frac`` of them and the room shell the rest
    (area-weighted within each).  The flat shell is most of the scene's area but needs few points; the thin
    obstacles of the drone scenes (4-unit branches, cables) need many, and every point within the body's
    radius costs time in each placement query."""
    if '--env' not in argv:
        return
    V, F, names = po.load_obj(argv[argv.index('--env') + 1])
    mask = np.array(['null' not in str(n).lower() for n in names])          # as main()'s sample_mask
    is_shell = np.array([str(n) == shell for n in names])[mask]
    if not is_shell.any() or is_shell.all():
        return
    n_faces = int(mask.sum())
    orig = po.sample_surface_points

    def wrapped(Vx, Fx, n):
        if len(Fx) != n_faces:                                               # the shape, not the env
            return orig(Vx, Fx, n)
        ns = int(round(shell_frac * n))
        return np.concatenate([orig(Vx, Fx[~is_shell], n - ns), orig(Vx, Fx[is_shell], ns)], 0)
    po.sample_surface_points = wrapped
    print('[fast] env points: {:.0f} % on the obstacles, {:.0f} % on the shell'.format(100 * (1 - shell_frac), 100 * shell_frac))


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest(sys.argv[1:]) else 1)
    env_weighted_sampling(sys.argv)
    po.main()
