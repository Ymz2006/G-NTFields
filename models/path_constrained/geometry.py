"""Exact geometry of the path-constrained prism: box scenes, the square prism, the point path.

Scenes. env1, env3 and env4 are a room (the 'wall' group: a box, or a box shell whose inner faces bound the free
space) plus axis-aligned obstacle boxes (the 'Body*' groups). Coordinates are normalized as in
dataprocessing/preprocess_obj.py: (v - bbox centre) / largest bbox extent, so poses are in the units of every
other cell of datasets/3dshape/data. Free space = inside the room box and outside every obstacle box.

Prism. A square prism of side w and length h whose body frame has its origin at the centre of one end face (the
anchor, glued to the path) and its axis along body +z: the box [-w/2, w/2]^2 x [0, h]. The other end is the
camera, so a collision-free prism is a camera whose line of sight to the anchored point is clear.

Clearance. For a pose (t, R) (t = anchor position) the signed clearance is exact: distance to the room walls =
min over the prism's vertices of the inner-box slack (the room interior is convex, its slack concave); distance to
an obstacle box = the exact distance between two convex boxes when a separating axis exists (min over vertex-box,
box-vertex and the 12 x 12 edge-edge distances), minus the smallest overlap of the 15 separating-axis tests when
they intersect. No point sampling is involved, so a 0.02-wide prism cannot slip between samples.

Angular clearance. speed_angles of preprocess_obj.evaluate_placements, computed the same way: the smallest angle,
seen from the anchor, between a prism surface point and an environment surface point in the same radial shell
(10 shells of equal prism-point count), here with a dense sampling of the box faces.
"""
import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi
EPS = 1e-12


# ----------------------------------------------------------------------------- OBJ
def read_obj_groups(path):
    """OBJ -> (V (n, 3) float64, [(group name, triangles (m, 3) int64)])."""
    verts, groups, cur = [], [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith(('g ', 'o ')):
                cur = [line[2:].strip(), []]
                groups.append(cur)
            elif line.startswith('f '):
                if cur is None:
                    cur = ['', []]
                    groups.append(cur)
                idx = [int(p.split('/')[0]) for p in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    cur[1].append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, np.float64), [(n, np.asarray(f, np.int64)) for n, f in groups if f]


def _ray_hit(origin, direction, tris):
    """Smallest positive Moller-Trumbore hit distance of one ray with triangles (m, 3, 3); inf if none."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - tris[:, 0]
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q * direction).sum(1) * inv
    d = (e2 * q).sum(1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (d > 1e-9)
    return float(d[hit].min()) if hit.any() else math.inf


class BoxScene:
    """Room box minus axis-aligned obstacle boxes, normalized like the training cells (module doc)."""

    def __init__(self, env_obj):
        V, groups = read_obj_groups(env_obj)
        lo, hi = V.min(0), V.max(0)
        self.env_center = 0.5 * (lo + hi)
        self.env_scale = float((hi - lo).max())
        Vn = (V - self.env_center) / self.env_scale
        wall, blo, bhi = [], [], []
        for name, F in groups:
            if name.lower().startswith('wall'):
                wall.append(Vn[F])
                continue
            P = Vn[np.unique(F)]
            for k in range(3):
                vals = np.unique(np.round(P[:, k], 7))
                assert len(vals) == 2, 'group {!r} is not an axis-aligned box'.format(name)
            blo.append(P.min(0))
            bhi.append(P.max(0))
        assert wall, 'no wall group'
        tris = np.concatenate(wall)
        # inner faces of the room: first wall hit from the centre along +-x, +-y, +-z
        o = np.zeros(3)
        self.room_lo = np.array([-_ray_hit(o, -np.eye(3)[k], tris) for k in range(3)])
        self.room_hi = np.array([_ray_hit(o, np.eye(3)[k], tris) for k in range(3)])
        assert np.isfinite(self.room_lo).all() and np.isfinite(self.room_hi).all()
        self.box_lo = np.asarray(blo)
        self.box_hi = np.asarray(bhi)
        self._dev = {}

    @property
    def n_boxes(self):
        return len(self.box_lo)

    def tensors(self, device, dtype=torch.float32):
        key = (str(device), dtype)
        if key not in self._dev:
            f = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
            lo, hi = f(self.box_lo), f(self.box_hi)
            corners = torch.stack([torch.stack([hi[:, k] if b else lo[:, k] for k, b in enumerate(bits)], -1)
                                   for bits in _BITS], 1)                                               # (K, 8, 3)
            self._dev[key] = dict(room_lo=f(self.room_lo), room_hi=f(self.room_hi), lo=lo, hi=hi,
                                  c=0.5 * (lo + hi), h=0.5 * (hi - lo), corners=corners,
                                  edges=corners[:, _EDGES])                                             # (K, 12, 2, 3)
        return self._dev[key]

    # ------------------------------------------------------------------ points
    def point_clearance(self, p):
        """Signed distance of points (..., 3) to the obstacles (positive in free space; exact)."""
        T = self.tensors(p.device, p.dtype)
        room = torch.minimum(p - T['room_lo'], T['room_hi'] - p).min(-1).values
        q = (p[..., None, :] - T['c']).abs() - T['h']
        sd = q.clamp(min=0).norm(dim=-1) + q.max(-1).values.clamp(max=0)
        return torch.minimum(room, sd.min(-1).values)

    def surface_points(self, spacing, seed=0):
        """Points on every face that bounds the free space (room inner faces, obstacle faces), area-uniform with
        mean spacing ``spacing``; faces or parts of faces buried inside another box or outside the room dropped."""
        rng = np.random.default_rng(seed)
        boxes = [(self.room_lo, self.room_hi)] + list(zip(self.box_lo, self.box_hi))
        out = []
        for lo, hi in boxes:
            for k in range(3):
                a, b = [j for j in range(3) if j != k]
                area = (hi[a] - lo[a]) * (hi[b] - lo[b])
                for side in (lo[k], hi[k]):
                    n = max(int(math.ceil(area / spacing ** 2)), 4)
                    P = np.empty((n, 3))
                    P[:, k] = side
                    P[:, a] = rng.uniform(lo[a], hi[a], n)
                    P[:, b] = rng.uniform(lo[b], hi[b], n)
                    out.append(P)
        P = np.concatenate(out)
        d = self.point_clearance(torch.as_tensor(P, dtype=torch.float64)).numpy()
        return P[np.abs(d) < 1e-6].astype(np.float32)   # on the free-space boundary


_BITS = [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)]
_EDGES = torch.tensor([[i, j] for i in range(8) for j in range(i + 1, 8)
                       if sum(a != b for a, b in zip(_BITS[i], _BITS[j])) == 1])          # (12, 2) corner pairs


# ----------------------------------------------------------------------------- prism
class Prism:
    """Square prism [-w/2, w/2]^2 x [0, h], anchor at the body origin, axis body +z."""

    def __init__(self, width, length):
        self.w, self.h = float(width), float(length)
        self.half = np.array([self.w / 2, self.w / 2, self.h / 2])
        self.center_local = np.array([0.0, 0.0, self.h / 2])
        self.radius = math.sqrt(2 * (self.w / 2) ** 2 + self.h ** 2)
        self.corners_local = np.array([[(2 * bx - 1) * self.w / 2, (2 * by - 1) * self.w / 2, bz * self.h]
                                       for bx, by, bz in _BITS])
        self._dev = {}

    def mesh(self):
        """(V (8, 3), F (12, 3)) outward-oriented triangles."""
        V = self.corners_local
        quads = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
        F = []
        for a, b, c, d in quads:
            F += [[a, b, c], [a, c, d]]
        return V.copy(), np.asarray(F, np.int64)

    def tensors(self, device, dtype=torch.float32):
        key = (str(device), dtype)
        if key not in self._dev:
            f = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
            corners = f(self.corners_local)
            self._dev[key] = dict(half=f(self.half), center=f(self.center_local), corners=corners,
                                  edges=corners[_EDGES])                                                # (12, 2, 3)
        return self._dev[key]

    def radial_points(self, num=1000, bins=10, seed=0):
        """preprocess_obj.generate_radius_surface_points for this prism: (bins, num // bins, 3) surface points
        sorted by distance from the anchor and split into equal-count shells, and the (bins + 1,) shell edges."""
        rng = np.random.default_rng(seed)
        V, F = self.mesh()
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tri = rng.choice(len(F), size=num, p=areas / areas.sum())
        r1 = np.sqrt(rng.random((num, 1)))
        r2 = rng.random((num, 1))
        pts = (1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]
        rad = np.linalg.norm(pts, axis=1)
        order = np.argsort(rad)
        pts, rad = pts[order], rad[order]
        per = num // bins
        pts, rad = pts[:per * bins], rad[:per * bins]
        edges = np.empty(bins + 1)
        edges[:bins] = rad.reshape(bins, per)[:, 0]
        edges[bins] = rad[-1]
        return pts.reshape(bins, per, 3).astype(np.float32), edges.astype(np.float32)


# ----------------------------------------------------------------------------- exact prism clearance
def _segseg(p1, q1, p2, q2):
    """Closest points of segments [p1, q1], [p2, q2] (broadcast (..., 3), non-degenerate); returns (c1, c2)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    den = a * e - b * b
    s = torch.where(den > 1e-12 * a * e, ((b * f - c * e) / den.clamp_min(1e-30)).clamp(0, 1), torch.zeros_like(a))
    t = (b * s + f) / e
    s = torch.where(t < 0, (-c / a).clamp(0, 1), torch.where(t > 1, ((b - c) / a).clamp(0, 1), s))
    t = t.clamp(0, 1)
    return p1 + d1 * s[..., None], p2 + d2 * t[..., None]


def _argmin_pair(d, ps, pe):
    """min over the last axis of d with the matching point pairs (..., n, 3)."""
    v, i = d.min(-1)
    g = i[..., None, None].expand(*i.shape, 1, 3)
    return v, torch.gather(ps, -2, g).squeeze(-2), torch.gather(pe, -2, g).squeeze(-2)


def prism_clearance(scene, prism, t, R, witness=False, chunk=8192):
    """Exact signed clearance of the prism at anchor poses (t (B, 3), R (B, 3, 3)) in the scene.

    Returns clearance (B,) (< 0: collision, minus the smallest separating-axis overlap) and, with witness, the
    closest prism point s* and obstacle point e* (B, 3) each (meaningful where clearance > 0)."""
    if t.shape[0] > chunk:
        parts = [prism_clearance(scene, prism, t[i:i + chunk], R[i:i + chunk], witness, chunk)
                 for i in range(0, t.shape[0], chunk)]
        if witness:
            return tuple(torch.cat([p[k] for p in parts]) for k in range(3))
        return torch.cat(parts)
    S = scene.tensors(t.device, t.dtype)
    P = prism.tensors(t.device, t.dtype)
    B = t.shape[0]
    Vp = t[:, None, :] + torch.einsum('bij,vj->bvi', R, P['corners'])             # (B, 8, 3) prism corners
    # room: slack of every corner to every inner face (exact, concave over the convex prism)
    lo_s = Vp - S['room_lo']
    hi_s = S['room_hi'] - Vp
    slack = torch.cat([lo_s, hi_s], -1).reshape(B, -1)                              # (B, 8 * 6)
    room, ri = slack.min(-1)
    best = room
    if witness:
        corner = ri // 6
        face = ri % 6
        s_star = Vp[torch.arange(B, device=t.device), corner]
        e_star = s_star.clone()
        ax = face % 3
        val = torch.where(face < 3, S['room_lo'][ax], S['room_hi'][ax])
        e_star[torch.arange(B, device=t.device), ax] = val
    if scene.n_boxes == 0:
        return (best, s_star, e_star) if witness else best
    c_p = t + torch.einsum('bij,j->bi', R, P['center'])                             # (B, 3) prism centre
    a = P['half']                                                                    # (3,)
    hb = S['h']                                                                      # (K, 3)
    d = c_p[:, None, :] - S['c'][None]                                               # (B, K, 3)
    # --- separating axis test (15 axes) -> min overlap (> 0 on every axis = intersection)
    Ra = R.abs()                                                                     # (B, 3, 3) |R_ij|
    ov = []
    # world axes e_i: r_B = h_i, r_P = sum_j a_j |R_ij|
    ov.append((hb[None] + (Ra * a).sum(-1)[:, None, :]) - d.abs())                  # (B, K, 3)
    # prism axes R_:j: r_B = sum_i h_i |R_ij|, r_P = a_j, dist = |d . R_:j|
    dj = torch.einsum('bki,bij->bkj', d, R)
    ov.append((torch.einsum('ki,bij->bkj', hb, Ra) + a) - dj.abs())
    # cross axes e_i x R_:j (unnormalized), |n| = sqrt(1 - R_ij^2)
    E = torch.eye(3, dtype=t.dtype, device=t.device)
    n = torch.cross(E[None, :, None, :].expand(B, 3, 3, 3), R.transpose(1, 2)[:, None, :, :].expand(B, 3, 3, 3),
                    dim=-1).reshape(B, 9, 3)                                         # (B, 9, 3)
    nn_ = n.norm(dim=-1)                                                             # (B, 9)
    rP = (torch.einsum('bni,bij->bnj', n, R).abs() * a).sum(-1)                    # (B, 9)
    rB = torch.einsum('bni,ki->bkn', n.abs(), hb)                                    # (B, K, 9)
    dist = torch.einsum('bki,bni->bkn', d, n).abs()
    ovc = (rP[:, None, :] + rB - dist) / nn_.clamp_min(1e-9)[:, None, :]
    ovc = torch.where(nn_[:, None, :] > 1e-6, ovc, torch.full_like(ovc, math.inf))
    ov.append(ovc)
    ov = torch.cat(ov, -1)                                                           # (B, K, 15)
    min_ov = ov.min(-1).values                                                       # (B, K)
    inter = min_ov > 0
    # --- exact distance (valid where separated)
    lo, hi = S['lo'], S['hi']
    # prism corners -> boxes
    cp = torch.maximum(torch.minimum(Vp[:, None], hi[None, :, None]), lo[None, :, None])   # (B, K, 8, 3)
    d1, s1, e1 = _argmin_pair((Vp[:, None] - cp).norm(dim=-1), Vp[:, None].expand_as(cp), cp)
    # box corners -> prism (in the prism frame)
    bc = S['corners'][None].expand(B, -1, -1, -1)                                     # (B, K, 8, 3)
    loc = torch.einsum('bji,bkvj->bkvi', R, bc - t[:, None, None, :])                # R^T (x - t)
    pl = P['corners'].min(0).values
    ph = P['corners'].max(0).values
    cl = torch.maximum(torch.minimum(loc, ph), pl)
    cw = t[:, None, None, :] + torch.einsum('bij,bkvj->bkvi', R, cl)
    d2, s2, e2 = _argmin_pair((loc - cl).norm(dim=-1), cw, bc)
    # edges x edges
    pe = t[:, None, None, :] + torch.einsum('bij,ekj->beki', R, P['edges'])           # (B, 12, 2, 3)
    be = S['edges']                                                                  # (K, 12, 2, 3)
    p1 = pe[:, None, :, None, 0, :]
    q1 = pe[:, None, :, None, 1, :]                                                  # (B, 1, 12, 1, 3)
    p2 = be[None, :, None, :, 0, :]
    q2 = be[None, :, None, :, 1, :]                                                  # (1, K, 1, 12, 3)
    c1, c2 = _segseg(p1, q1, p2, q2)                                                 # (B, K, 12, 12, 3)
    c1 = c1.reshape(B, -1, 144, 3)
    c2 = c2.reshape(B, -1, 144, 3)
    d3, s3, e3 = _argmin_pair((c1 - c2).norm(dim=-1), c1, c2)
    dd = torch.stack([d1, d2, d3], -1)                                               # (B, K, 3)
    dk, which = dd.min(-1)
    clear_k = torch.where(inter, -min_ov, dk)                                        # (B, K)
    ck, kk = clear_k.min(-1)
    use = ck < best
    best = torch.where(use, ck, best)
    if not witness:
        return best
    bi = torch.arange(B, device=t.device)
    w = which[bi, kk]
    sk = torch.stack([s1[bi, kk], s2[bi, kk], s3[bi, kk]], 1)[bi, w]
    ek = torch.stack([e1[bi, kk], e2[bi, kk], e3[bi, kk]], 1)[bi, w]
    s_star = torch.where(use[:, None], sk, s_star)
    e_star = torch.where(use[:, None], ek, e_star)
    return best, s_star, e_star


# ----------------------------------------------------------------------------- angular clearance
class AngleClearance:
    """speed_angles of preprocess_obj.evaluate_placements (module doc) for anchors on a path.

    Environment points within reach of the prism are pre-gathered per path sample (the anchors only ever lie on
    the path), so each query compares the prism's 1000 shell points with a few thousand nearby surface points."""

    def __init__(self, env_points, prism, path_pts, device, num=1000, bins=10, seed=0, reach_pad=0.01):
        self.device = device
        rp, edges = prism.radial_points(num, bins, seed)
        self.rad = torch.as_tensor(rp, device=device)                                # (bins, n, 3)
        self.edges = torch.as_tensor(edges, device=device)                           # (bins + 1,)
        self.bins = bins
        reach = float(edges[-1]) + reach_pad
        from scipy.spatial import cKDTree
        tree = cKDTree(env_points)
        self.anchor_pts = torch.as_tensor(path_pts, dtype=torch.float32, device=device)
        lists = tree.query_ball_point(path_pts, r=reach)
        m = max(1, max(len(x) for x in lists))
        idx = np.zeros((len(lists), m), np.int64)
        valid = np.zeros((len(lists), m), bool)
        for i, x in enumerate(lists):
            idx[i, :len(x)] = x
            valid[i, :len(x)] = True
        self.env = torch.as_tensor(env_points, dtype=torch.float32, device=device)
        self.idx = torch.as_tensor(idx, device=device)
        self.valid = torch.as_tensor(valid, device=device)
        self.max_pts = m

    def __call__(self, t, R, sample_idx, chunk=128, env_chunk=512):
        """Min angle (B,) in [0, pi] for anchors t (B, 3) at path samples sample_idx (B,) with rotations R."""
        out = []
        for i in range(0, t.shape[0], chunk):
            out.append(self._one(t[i:i + chunk], R[i:i + chunk], sample_idx[i:i + chunk], env_chunk))
        return torch.cat(out)

    def _one(self, t, R, si, env_chunk):
        B = t.shape[0]
        idx, valid = self.idx[si], self.valid[si]                                    # (B, m)
        ev = self.env[idx] - t[:, None, :]                                           # (B, m, 3)
        dist = ev.norm(dim=-1)
        # shell of each env point: (edges[i], edges[i + 1]] (searchsorted right = bucketize - 1 of the original)
        shell = torch.searchsorted(self.edges, dist.contiguous(), right=False) - 1
        ok = valid & (shell >= 0) & (shell < self.bins) & (dist > self.edges[0])
        eu = ev / dist.clamp_min(EPS)[..., None]
        su = torch.einsum('bij,snj->bsni', R, self.rad)
        su = su / su.norm(dim=-1, keepdim=True).clamp_min(EPS)                       # (B, bins, n, 3)
        best = torch.full((B,), -1.0, device=t.device)
        for s in range(0, eu.shape[1], env_chunk):
            e_c = eu[:, s:s + env_chunk]
            sh = shell[:, s:s + env_chunk]
            okc = ok[:, s:s + env_chunk]
            cos = torch.einsum('bsni,bmi->bsnm', su, e_c).max(2).values               # (B, bins, mc) best per env pt
            sel = torch.gather(cos, 1, sh.clamp(0, self.bins - 1)[:, None, :]).squeeze(1)   # its own shell
            sel = torch.where(okc, sel, torch.full_like(sel, -1.0))
            best = torch.maximum(best, sel.max(-1).values)
        return torch.arccos(best.clamp(-1.0, 1.0))


def so3_exp(w):
    """(..., 3) rotation vectors (radians) -> (..., 3, 3), series-safe."""
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    th = th2.sqrt()
    small = th2 < 1e-8
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th.clamp_min(1e-12))
    Bc = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -w[..., 2], w[..., 1], -w[..., 0]
    K = K - K.transpose(-1, -2)
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + A * K + Bc * (K @ K)


def so3_log(R):
    """(..., 3, 3) -> rotation vectors (..., 3) in radians, angle in [0, pi] (robust near pi)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = ((tr - 1) / 2).clamp(-1, 1)
    th = torch.arccos(c)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = torch.sin(th)
    small = th < 1e-4
    w = torch.where(small[..., None], 0.5 * v * (1 + th[..., None] ** 2 / 6),
                    v * (th / (2 * s.clamp_min(1e-12)))[..., None])
    near_pi = th > math.pi - 1e-3
    if near_pi.any():
        # axis from the symmetric part: (R + R^T) / 2 = cos(th) I + (1 - cos(th)) u u^T
        I = torch.eye(3, dtype=R.dtype, device=R.device)
        M = (0.5 * (R + R.transpose(-1, -2)) - c[..., None, None] * I) / (1 - c).clamp_min(1e-12)[..., None, None]
        diag = torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]], -1)
        k = diag.argmax(-1)
        col = torch.gather(M, -1, k[..., None, None].expand(*k.shape, 3, 1)).squeeze(-1)
        u = col / col.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        sgn = torch.sign((u * v).sum(-1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        w = torch.where(near_pi[..., None], u * sgn * th[..., None], w)
    return w


def quat_from_rotvec(rv):
    """numpy (n, 3) radians -> (n, 4) unit quaternions (w, x, y, z), w >= 0."""
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    ax = rv / np.maximum(th, 1e-12)
    return np.concatenate([np.cos(th / 2), ax * np.sin(th / 2)], -1)


def rotvec_from_quat(q):
    """numpy (..., 4) (w, x, y, z) -> rotation vectors (..., 3) with angle in [0, pi]."""
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    vn = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(vn, q[..., :1])
    return q[..., 1:] / np.maximum(vn, 1e-12) * ang


def quat_matrix(q):
    """numpy (..., 4) (w, x, y, z) -> (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def quat_angle(q0, q1):
    """Rotation angle (radians, [0, pi]) between unit quaternions (..., 4)."""
    return 2 * np.arccos(np.clip(np.abs((q0 * q1).sum(-1)), 0, 1))


def slerp(q0, q1, u, short=True):
    """numpy slerp from q0 (4,) to q1 (4,) at fractions u (k,) -> (k, 4); short = the shorter arc (the other
    homotopy class with short=False)."""
    d = float((q0 * q1).sum())
    if short == (d < 0):
        q1, d = -q1, -d
    th = math.acos(max(-1.0, min(1.0, d)))
    u = np.asarray(u, np.float64)[:, None]
    if math.sin(th) < 1e-9:
        q = (1 - u) * q0 + u * q1
    else:
        q = (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def random_quats(n, rng):
    """Uniform unit quaternions (n, 4) (w, x, y, z), w >= 0."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([np.sqrt(u1) * np.cos(2 * np.pi * u3), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                  np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u3)], 1)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def random_rotvecs(n, rng):
    """Uniform rotations on SO(3) as rotation vectors with angle in [0, pi] (Shoemake quaternions)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    q = np.stack([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)], 1)   # (x, y, z, w)
    q = q * np.where(q[:, 3:] < 0, -1.0, 1.0)
    vn = np.linalg.norm(q[:, :3], axis=1)
    ang = 2 * np.arctan2(vn, q[:, 3])
    return q[:, :3] / np.maximum(vn, 1e-12)[:, None] * ang[:, None]
