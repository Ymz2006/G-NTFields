"""Test-case endpoints of the stretchable-link sets (shared by the RRT and PRM certificates of
scripts/make_link_tests.py and scripts/make_link_tests_prm.py).

A case is a start and a goal pose of the link, nodes (t (3), q (4) unit quaternion (w, x, y, z), L), both with exact
clearance >= offset. Case k of a set is drawn from numpy's default_rng(1000003 * seed + k), so a case's endpoints do
not depend on the certificate that verifies it.

    std    t uniform in the room, R uniform on SO(3), L uniform
    cross  both drones in region A at the start and in region B at the goal (every other case the reverse)
"""
import numpy as np
import torch

from . import geometry as G

# crossing regions: (axis, raw bound, side) for A and B; side +1 = coordinate > bound (raw OBJ units, 5 units off
# the walls that separate the regions)
REGIONS = {'forest': ((2, -15.0, 1), (2, -165.0, -1)), 'windows': ((2, -80.0, 1), (2, -100.0, -1)),
           'corridor': ((2, -136.0, -1), (2, -43.0, 1)), 'scaffold': ((0, 30.0, -1), (0, 170.0, 1))}


def region_box(scene, reg):
    """Normalized (lo, hi) of the part of the room on the region's side of its bound."""
    ax, raw, side = reg
    v = (raw - scene.env_center[ax]) / scene.env_scale
    lo, hi = scene.room_lo.copy(), scene.room_hi.copy()
    if side > 0:
        lo[ax] = max(lo[ax], v)
    else:
        hi[ax] = min(hi[ax], v)
    return lo, hi


def clear_np(scene, link, t, q, L):
    """Exact clearance (k,) of poses t (k, 3), q (k, 4), L (k,) (CPU, float64)."""
    return G.clearance(scene, link, torch.as_tensor(t, dtype=torch.float64),
                       torch.as_tensor(G.quat_matrix(q), dtype=torch.float64),
                       torch.as_tensor(L, dtype=torch.float64)).numpy()


def draw_end(scene, link, rng, offset, box=None, reg=None):
    """A node with exact clearance >= offset (both drones inside ``box`` when a region is given)."""
    lo, hi = box if box is not None else (scene.room_lo, scene.room_hi)
    while True:
        n = 512
        t = rng.uniform(lo, hi, (n, 3))
        q = G.random_quats(n, rng)
        L = rng.uniform(link.L_min, link.L_max, n)
        ok = np.ones(n, bool)
        if reg is not None:
            b = t + G.quat_matrix(q)[:, :, 2] * L[:, None]
            ok &= (b >= lo).all(1) & (b <= hi).all(1)
        if not ok.any():
            continue
        t, q, L = t[ok], q[ok], L[ok]
        c = clear_np(scene, link, t, q, L)
        good = np.nonzero(c >= offset)[0]
        if len(good):
            i = good[0]
            return np.concatenate([t[i], q[i], [L[i]]]), float(c[i])


def draw_case(scene, link, scene_name, variant, k, seed, offset):
    """(start node, its clearance, goal node, its clearance, the case's rng after the draw)."""
    rng = np.random.default_rng(1000003 * seed + k)
    if variant == 'std':
        a, c0 = draw_end(scene, link, rng, offset)
        b, c1 = draw_end(scene, link, rng, offset)
    else:
        ra, rb = REGIONS[scene_name]
        if k % 2:
            ra, rb = rb, ra
        a, c0 = draw_end(scene, link, rng, offset, region_box(scene, ra), ra)
        b, c1 = draw_end(scene, link, rng, offset, region_box(scene, rb), rb)
    return a, c0, b, c1, rng
