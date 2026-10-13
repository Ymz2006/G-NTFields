"""Structured clearance (speed) predictor: env signed-distance grid composed with sampled shape points.

A direct 6-d regression G(x) -> S_d does not generalise from the available samples (graph_lie
EXPERIMENTS.md S0): the clearance changes on the scale of the margin (0.05) while the samples are
~0.05 apart. The clearance is, however, a composition of two simple objects:

    d(x) = min_{p in shape surface} sdf_env(R p + t)

sdf_env is a 3-d function (here a dense trilinear grid built from the env mesh with igl, so it is exact
up to the grid spacing), and the shape surface is sampled once (vertices, edges, faces). The predictor
is then

    d_hat(x)  = -tau * logsumexp_i(-sdf_env(R p_i + t) / tau)       (smooth min over the K shape points)
    s         = clip(d_hat / margin, offset / margin, 1),  S_d = s^2 (2 - s)^2  (dataset + trainer warp)

It is differentiable in the pose (trilinear interpolation, manual so that double backward works), has
no trainable parameters (the grid and the points are buffers, so they travel inside the network
checkpoint) and the same interface as speed_fourier.SpeedPredictor (forward -> S_d, log_speed).
Build + validate: models/graph_lie/scripts/build_clearance_predictor.py.
"""
import math

import torch

from .geometry import se3

LOG_S_MIN = math.log(1e-3)


class ClearancePredictor(torch.nn.Module):
    def __init__(self, res=192, n_points=1024, tau=0.0, margin=0.05, offset=0.001, s_floor=1e-3, mode='grid',
                 n_boxes=0, d_bias=0.0):
        super().__init__()
        assert mode in ('grid', 'boxes'), mode
        # the datasets measure the clearance to *sampled* env surface points, which overestimates the exact
        # clearance by a fraction of the point spacing; d_bias (>= 0) mimics that bias
        self.d_bias = float(d_bias)
        self.mode, self.res, self.n_points, self.n_boxes = mode, int(res), int(n_points), int(n_boxes)
        self.tau, self.margin, self.offset, self.s_floor = float(tau), float(margin), float(offset), float(s_floor)
        if mode == 'grid':
            self.register_buffer('grid', torch.zeros(res, res, res))      # sdf_env on the box bmin..bmax
            self.register_buffer('bmin', torch.zeros(3))
            self.register_buffer('bmax', torch.ones(3))
        else:  # exact union of (oriented) boxes: sdf = min_k sign_k * box_sdf_k, sign -1 for a container
            self.register_buffer('box_center', torch.zeros(n_boxes, 3))
            self.register_buffer('box_half', torch.ones(n_boxes, 3))
            self.register_buffer('box_rot', torch.eye(3).expand(n_boxes, 3, 3).clone())
            self.register_buffer('box_sign', torch.ones(n_boxes))
        self.register_buffer('points', torch.zeros(n_points, 3))          # shape surface samples, local frame

    # ------------------------------------------------------------------ env sdf
    def sdf(self, p):
        """Signed distance at world points p (..., 3): exact boxes, or trilinear grid (clamped to its box)."""
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        if self.mode == 'boxes':
            q = torch.einsum('kji,nkj->nki', self.box_rot, p[:, None, :] - self.box_center[None])  # local coords
            q = q.abs() - self.box_half[None]
            outside = torch.sqrt((q.clamp_min(0.0) ** 2).sum(-1) + 1e-12)
            inside = q.amax(-1).clamp_max(0.0)
            d = self.box_sign[None] * (outside + inside)
            return d.min(dim=1).values.reshape(shape)
        u = (p - self.bmin) / (self.bmax - self.bmin) * (self.res - 1)
        u = u.clamp(0.0, self.res - 1 - 1e-4)
        i0 = u.floor().long().clamp(0, self.res - 2)
        f = u - i0.float()
        g = self.grid
        r = self.res
        base = (i0[:, 0] * r + i0[:, 1]) * r + i0[:, 2]
        flat = g.reshape(-1)
        out = 0.0
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    out = out + wx * wy * wz * flat[base + (dx * r + dy) * r + dz]
        return out.reshape(shape)

    # ------------------------------------------------------------------ clearance
    def clearance(self, g):
        """G(x) = (t, vec R) (N, 12) -> clearance of the placed shape (N,): min over the K shape points.

        The minimising point i* is selected without autograd (O(N K) memory freed at once), then the sdf
        is re-evaluated at that single point with autograd: same value, and the gradient of a min over
        a finite set is the gradient of its active element, so the derivative is exact (a.e.) as well.
        With tau > 0 a smooth min over all points is used instead (biased by up to tau log K).
        """
        t, R = se3.split_features(g)
        if self.tau > 0:
            p = torch.einsum('nij,kj->nki', R, self.points) + t[:, None, :]  # (N, K, 3)
            return -self.tau * torch.logsumexp(-self.sdf(p) / self.tau, dim=1)
        with torch.no_grad():
            i_star = None
            best = None
            for k0 in range(0, self.n_points, 512):  # chunked to bound memory
                p = torch.einsum('nij,kj->nki', R, self.points[k0:k0 + 512]) + t[:, None, :]
                d = self.sdf(p)
                m, i = d.min(dim=1)
                if best is None:
                    best, i_star = m, i + k0
                else:
                    upd = m < best
                    best = torch.where(upd, m, best)
                    i_star = torch.where(upd, i + k0, i_star)
        p_star = torch.einsum('nij,nj->ni', R, self.points[i_star]) + t
        return self.sdf(p_star)

    def raw_speed(self, g):
        """clip(d / margin, offset / margin, 1): the dataset's speed_dists."""
        return ((self.clearance(g) + self.d_bias) / self.margin).clamp(self.offset / self.margin, 1.0)

    def forward(self, g):
        s = self.raw_speed(g)
        return (s * s * (2.0 - s) * (2.0 - s)).clamp_min(self.s_floor)

    def log_speed(self, g):
        return torch.log(self.forward(g))


def sample_shape_points(V, F, n_points, seed=0):
    """Vertices + edge samples + area-weighted face samples of a triangle mesh (numpy), n_points total."""
    import numpy as np
    rng = np.random.default_rng(seed)
    verts = V[np.unique(F.reshape(-1))]
    edges = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    n_edge = max(0, min(n_points - len(verts), 4 * len(edges)))
    e = edges[rng.integers(0, len(edges), n_edge)]
    r = rng.random((n_edge, 1))
    edge_pts = V[e[:, 0]] * (1 - r) + V[e[:, 1]] * r
    n_face = max(0, n_points - len(verts) - n_edge)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(F), size=n_face, p=areas / areas.sum())
    r1, r2 = np.sqrt(rng.random((n_face, 1))), rng.random((n_face, 1))
    face_pts = (1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]
    pts = np.concatenate([verts, edge_pts, face_pts], 0)[:n_points]
    return pts.astype(np.float32)


def fit_boxes(V, F, groups, probe):
    """Each mesh group with exactly 8 vertices forming an axis-aligned box -> (center, half, rot, sign) arrays,
    sign -1 for a group that contains `probe` (a container). Returns None if a group is not such a box."""
    import numpy as np
    centers, halves, rots, signs = [], [], [], []
    for gid in np.unique(groups):
        vs = V[np.unique(F[groups == gid].reshape(-1))]
        if len(vs) != 8:
            return None
        lo, hi = vs.min(0), vs.max(0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        if np.abs(np.sort(vs, axis=0) - np.sort(corners, axis=0)).max() > 1e-6 * (hi - lo).max():
            return None
        centers.append(0.5 * (lo + hi)), halves.append(0.5 * (hi - lo)), rots.append(np.eye(3))
        signs.append(-1.0 if np.all((probe > lo) & (probe < hi)) else 1.0)
    return (np.array(centers, np.float32), np.array(halves, np.float32), np.array(rots, np.float32),
            np.array(signs, np.float32))


def build_env_sdf_grid(V, F, bmin, bmax, res, groups=None, probe=None):
    """Signed distance of the env mesh on a res^3 grid over [bmin, bmax]: min over mesh groups of each
    group's igl signed distance (pseudonormal sign). A group that contains `probe` (default: the box
    centre), i.e. a room / container, has its sign flipped so that its interior is free space.
    Returns (grid tensor, list of flipped group ids)."""
    import igl
    import numpy as np
    axes = [np.linspace(bmin[i], bmax[i], res) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float64)
    probe = np.asarray(0.5 * (np.asarray(bmin) + np.asarray(bmax)) if probe is None else probe, np.float64)[None]
    groups = np.zeros(len(F), np.int64) if groups is None else np.asarray(groups)
    Vd, out, flipped = V.astype(np.float64), None, []
    for gid in np.unique(groups):
        Fg = F[groups == gid].astype(np.int64)
        sign = 1.0
        if igl.signed_distance(probe, Vd, Fg)[0][0] < 0:
            sign, flipped = -1.0, flipped + [int(gid)]
        d = np.empty(P.shape[0], np.float64)
        for i in range(0, P.shape[0], 2_000_000):
            d[i:i + 2_000_000] = igl.signed_distance(P[i:i + 2_000_000], Vd, Fg)[0]
        d = sign * d
        out = d if out is None else np.minimum(out, d)
    return torch.from_numpy(out.reshape(res, res, res).astype(np.float32)), flipped
