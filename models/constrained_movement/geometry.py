"""Exact geometry of constrained movement: box scenes, the followed shape, the camera's view pyramid.

Scenes. A room box (the inner faces of the 'wall' group) minus axis-aligned obstacle boxes, normalized as every
cell of datasets/3dshape/data ((v - bbox centre) / largest bbox extent of the OBJ). The obstacles are the non-wall
groups of the OBJ, one box each (path_constrained.geometry.BoxScene), or the rows of <scene>_boxes.json next to
the OBJ when a group is a union of boxes (a tree of the forest: trunk + crown).

Shape. The followed object: a union of axis-aligned boxes at a fixed orientation, positioned by its centre (the
centre of its bounding box). Default: the bird of dataprocessing/make_camera3d.py (fuselage, head, wings, tail
plane, fin; 48 x 18 x 48 units, flying along +x). Its clearance at a centre position is exact: per part and
obstacle the box-box distance (minus the smallest axis overlap when they intersect), the room slack of every part.

Pyramid. The camera's view volume, a rectangle-based pyramid. Body frame: the base rectangle W x H in the plane
z = 0 centred at the origin (the anchor, glued to the shape's centre), the apex (the camera) at (0, 0, D):

    base corners (+-W/2, +-H/2, 0), apex (0, 0, D),    so R e_z points from the shape to the camera.

Only the rotation about the anchor is free. A collision-free pyramid is a camera that never touches the scene and
whose view of the rectangle around the shape is never blocked. The followed shape itself is the target, not an
obstacle. Default: the camera of the camera + object test (birdcam3d_camera.json): W x H = 48 x 22, D = 50
(51 x 25 degree field of view), with the base now centred on the bird.

Clearance. Exact signed distance between the pyramid and the scene. Room: the slack of the 5 vertices to the inner
faces. Obstacle box: when one of the 26 candidate separating axes (3 box normals, 5 pyramid face normals, the 3 x 6
cross products of their edge directions) separates them, the convex distance = min over pyramid vertex - box,
box corner - pyramid face (closest point on a triangle) and the 8 x 12 edge pairs; otherwise minus the smallest
overlap over those axes (the penetration depth of two convex polyhedra). No point sampling is involved.
"""
import importlib.util
import json
import math
import os

import numpy as np
import torch


def _load(name, fname):
    """A module of models/path_constrained by file (the NTFields bundle owns the name 'models' in its processes)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'path_constrained', fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PG = _load('_cm_pc_geometry', 'geometry.py')
PP = _load('_cm_pc_path', 'path.py')
Path = PP.Path

TWO_PI = PG.TWO_PI
EPS = PG.EPS
so3_exp, so3_log = PG.so3_exp, PG.so3_log
quat_from_rotvec, rotvec_from_quat, quat_matrix = PG.quat_from_rotvec, PG.rotvec_from_quat, PG.quat_matrix
quat_angle, slerp, random_quats, random_rotvecs = PG.quat_angle, PG.slerp, PG.random_quats, PG.random_rotvecs
AngleClearance = PG.AngleClearance

# raw OBJ units (rows (x0, x1, y0, y1, z0, z1)), as dataprocessing/make_camera3d.py BIRD / PLATE / CAM_L
BIRD = [(0, 40, 10, 20, 15, 25), (40, 48, 10, 18, 15, 25), (14, 26, 16, 20, -4, 44), (0, 8, 16, 20, -4, 44),
        (0, 8, 20, 28, 18, 22)]
CAMERA = dict(width=48.0, height=22.0, depth=50.0)
SHAPES = {'bird': BIRD}


# ----------------------------------------------------------------------------- scenes
class Scene(PG.BoxScene):
    """BoxScene whose obstacles may come from a box list (raw OBJ units, rows (x0, x1, y0, y1, z0, z1)); given
    none, from <scene>_boxes.json next to the OBJ, else every non-wall group must be one box (BoxScene)."""

    def __init__(self, env_obj, boxes=None):
        self.name = os.path.splitext(os.path.basename(env_obj))[0]
        if boxes is None:
            js = os.path.splitext(env_obj)[0] + '_boxes.json'
            if os.path.exists(js):
                with open(js) as fh:
                    boxes = json.load(fh)['boxes']
        if boxes is None:
            super().__init__(env_obj)
        else:
            V, groups = PG.read_obj_groups(env_obj)
            lo, hi = V.min(0), V.max(0)
            self.env_center = 0.5 * (lo + hi)
            self.env_scale = float((hi - lo).max())
            Vn = (V - self.env_center) / self.env_scale
            wall = [Vn[F] for name, F in groups if name.lower().startswith('wall')]
            assert wall, 'no wall group'
            tris = np.concatenate(wall)
            o = np.zeros(3)
            self.room_lo = np.array([-PG._ray_hit(o, -np.eye(3)[k], tris) for k in range(3)])
            self.room_hi = np.array([PG._ray_hit(o, np.eye(3)[k], tris) for k in range(3)])
            assert np.isfinite(self.room_lo).all() and np.isfinite(self.room_hi).all()
            b = np.asarray(boxes, np.float64)
            self.box_lo = (b[:, 0::2] - self.env_center) / self.env_scale
            self.box_hi = (b[:, 1::2] - self.env_center) / self.env_scale
            self._dev = {}
            # every obstacle vertex of the OBJ must lie in (or on) the listed boxes
            P = np.concatenate([Vn[np.unique(F)] for name, F in groups if not name.lower().startswith('wall')])
            inside = ((P[:, None] >= self.box_lo[None] - 1e-7) & (P[:, None] <= self.box_hi[None] + 1e-7)).all(-1)
            assert inside.any(1).all(), '{}: the box list does not cover the OBJ obstacles'.format(env_obj)
        self.boxes_raw = np.stack([self.box_lo, self.box_hi], -1).reshape(-1, 6) * self.env_scale \
            + np.repeat(self.env_center, 2)[None]

    def to_raw(self, p):
        return np.asarray(p) * self.env_scale + self.env_center


def scene_from_meta(meta, repo):
    """Scene of a cell / test set meta (the box rows stored in the meta, so a later edit of the JSON can't change
    an existing cell)."""
    env_obj = meta['env_obj'] if os.path.exists(meta['env_obj']) else os.path.join(repo, meta['env_obj_rel'])
    return Scene(env_obj, boxes=meta['constraint'].get('boxes_raw'))


# ----------------------------------------------------------------------------- followed shape
class Shape:
    """Union of axis-aligned boxes (raw rows) at a fixed orientation, positioned by its bounding-box centre and
    normalized by the scene's scale."""

    def __init__(self, parts_raw, scale):
        b = np.asarray(parts_raw, np.float64)
        lo, hi = b[:, 0::2], b[:, 1::2]
        self.parts_raw = b
        self.center_raw = 0.5 * (lo.min(0) + hi.max(0))
        self.lo = (lo - self.center_raw) / scale                  # (P, 3) parts relative to the centre
        self.hi = (hi - self.center_raw) / scale
        self.extent = hi.max(0) - lo.min(0)                       # raw
        self._dev = {}

    def tensors(self, device, dtype=torch.float32):
        key = (str(device), dtype)
        if key not in self._dev:
            self._dev[key] = (torch.as_tensor(self.lo, dtype=dtype, device=device),
                              torch.as_tensor(self.hi, dtype=dtype, device=device))
        return self._dev[key]

    def clearance(self, scene, c):
        """Exact signed clearance (N,) of the shape centred at c (N, 3)."""
        S = scene.tensors(c.device, c.dtype)
        plo, phi = self.tensors(c.device, c.dtype)
        lo = c[:, None, :] + plo                                  # (N, P, 3)
        hi = c[:, None, :] + phi
        room = torch.minimum(lo - S['room_lo'], S['room_hi'] - hi).min(-1).values.min(-1).values
        if scene.n_boxes == 0:
            return room
        g = torch.maximum(S['lo'][None, None] - hi[:, :, None], lo[:, :, None] - S['hi'][None, None])   # (N, P, K, 3)
        gm = g.max(-1).values
        d = torch.where(gm > 0, g.clamp(min=0).norm(dim=-1), gm)
        return torch.minimum(room, d.min(-1).values.min(-1).values)

    def boxes(self, c):
        """(P, 2, 3) normalized part boxes (lo, hi) of the shape centred at c (3,)."""
        c = np.asarray(c, np.float64)
        return np.stack([self.lo + c, self.hi + c], 1)


# ----------------------------------------------------------------------------- pyramid
_TRIS = [(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]       # base (2) + sides, outward
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]


class Pyramid:
    """Rectangle-based pyramid: base W x H at z = 0 centred on the anchor (body origin), apex at (0, 0, D)."""

    def __init__(self, width, height, depth):
        self.W, self.H, self.D = float(width), float(height), float(depth)
        a, b = self.W / 2, self.H / 2
        self.verts = np.array([[-a, -b, 0.0], [a, -b, 0.0], [a, b, 0.0], [-a, b, 0.0], [0.0, 0.0, self.D]])
        self.tris = np.asarray(_TRIS, np.int64)
        self.edges = np.asarray(_EDGES, np.int64)
        cen = self.verts.mean(0)
        nrm = []
        for i, j, k in _TRIS:
            n = np.cross(self.verts[j] - self.verts[i], self.verts[k] - self.verts[i])
            assert np.dot(n, self.verts[i] - cen) > 0, 'inward triangle'
            nrm.append(n / np.linalg.norm(n))
        self.normals = np.asarray(nrm[1:])                        # (5, 3): base once + 4 sides
        dirs = [self.verts[1] - self.verts[0], self.verts[2] - self.verts[1]] + \
               [self.verts[4] - self.verts[i] for i in range(4)]
        self.dirs = np.asarray([d / np.linalg.norm(d) for d in dirs])   # (6, 3) distinct edge directions
        self.radius = max(self.D, math.hypot(a, b))               # farthest point from the anchor
        self.fov_deg = (2 * math.degrees(math.atan(a / self.D)), 2 * math.degrees(math.atan(b / self.D)))
        self._dev = {}

    def mesh(self):
        """(V (5, 3), F (6, 3)) outward-oriented triangles."""
        return self.verts.copy(), self.tris.copy()

    def tensors(self, device, dtype=torch.float32):
        key = (str(device), dtype)
        if key not in self._dev:
            f = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
            V = f(self.verts)
            T = V[torch.as_tensor(self.tris)]
            self._dev[key] = dict(verts=V, edges=V[torch.as_tensor(self.edges)], tri_a=T[:, 0], tri_b=T[:, 1],
                                  tri_c=T[:, 2], normals=f(self.normals), dirs=f(self.dirs))
        return self._dev[key]

    def radial_points(self, num=1000, bins=10, seed=0):
        """preprocess_obj.generate_radius_surface_points for this pyramid: (bins, num // bins, 3) surface points
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

    def surface_points(self, spacing, seed=0):
        """Area-uniform points on the surface with mean spacing ``spacing`` (tests, rendering)."""
        rng = np.random.default_rng(seed)
        V, F = self.mesh()
        out = []
        for i, j, k in F:
            a, b, c = V[i], V[j], V[k]
            n = max(int(math.ceil(0.5 * np.linalg.norm(np.cross(b - a, c - a)) / spacing ** 2)), 3)
            r1 = np.sqrt(rng.random((n, 1)))
            r2 = rng.random((n, 1))
            out.append((1 - r1) * a + r1 * (1 - r2) * b + r1 * r2 * c)
        return np.concatenate(out + [V])


def default_pyramid(scale, camera=CAMERA):
    """The camera of the camera + object test in the units of a scene of scale ``scale``."""
    return Pyramid(camera['width'] / scale, camera['height'] / scale, camera['depth'] / scale)


# ----------------------------------------------------------------------------- exact pyramid clearance
def closest_point_triangle(p, a, b, c):
    """Closest points of triangles (a, b, c) to points p (broadcast (..., 3)); Ericson, Real-Time Collision
    Detection 5.1.5, the Voronoi regions tested in reverse order so the first matching region wins."""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    den = (va + vb + vc).clamp_min(1e-30)
    out = a + ab * (vb / den)[..., None] + ac * (vc / den)[..., None]                       # interior
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-30)
    out = torch.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * w[..., None], out)
    w = d2 / (d2 - d6).clamp_min(1e-30)
    out = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * w[..., None], out)   # edge AC
    out = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c.expand_as(out), out)                     # vertex C
    v = d1 / (d1 - d3).clamp_min(1e-30)
    out = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * v[..., None], out)   # edge AB
    out = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b.expand_as(out), out)                     # vertex B
    out = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a.expand_as(out), out)                      # vertex A
    return out


def pyramid_clearance(scene, pyr, t, R, witness=False, chunk=4096):
    """Exact signed clearance of the pyramid at anchor poses (t (B, 3), R (B, 3, 3)) in the scene.

    Returns clearance (B,) (< 0: collision, minus the penetration depth) and, with witness, the closest pyramid
    point s* and scene point e* (B, 3) each (meaningful where clearance > 0)."""
    if t.shape[0] > chunk:
        parts = [pyramid_clearance(scene, pyr, t[i:i + chunk], R[i:i + chunk], witness, chunk)
                 for i in range(0, t.shape[0], chunk)]
        if witness:
            return tuple(torch.cat([p[k] for p in parts]) for k in range(3))
        return torch.cat(parts)
    S = scene.tensors(t.device, t.dtype)
    P = pyr.tensors(t.device, t.dtype)
    B = t.shape[0]
    bi = torch.arange(B, device=t.device)
    Vw = t[:, None, :] + torch.einsum('bij,vj->bvi', R, P['verts'])                 # (B, 5, 3)
    slack = torch.cat([Vw - S['room_lo'], S['room_hi'] - Vw], -1).reshape(B, -1)   # (B, 5 * 6)
    best, ri = slack.min(-1)
    if witness:
        s_star = Vw[bi, ri // 6]
        e_star = s_star.clone()
        face = ri % 6
        ax = face % 3
        e_star[bi, ax] = torch.where(face < 3, S['room_lo'][ax], S['room_hi'][ax])
    if scene.n_boxes == 0:
        return (best, s_star, e_star) if witness else best
    K = scene.n_boxes
    # --- separating axes (B, 26, 3): box normals, pyramid face normals, box edge x pyramid edge directions
    E = torch.eye(3, dtype=t.dtype, device=t.device)
    nf = torch.einsum('bij,fj->bfi', R, P['normals'])                                 # (B, 5, 3)
    ed = torch.einsum('bij,ej->bei', R, P['dirs'])                                    # (B, 6, 3)
    cr = torch.cross(E[None, :, None, :].expand(B, 3, 6, 3), ed[:, None].expand(B, 3, 6, 3), dim=-1).reshape(B, 18, 3)
    crn = cr.norm(dim=-1)
    axes = torch.cat([E[None].expand(B, 3, 3), nf, cr / crn.clamp_min(1e-9)[..., None]], 1)
    valid = torch.cat([torch.ones(B, 8, dtype=torch.bool, device=t.device), crn > 1e-6], 1)
    pv = torch.einsum('bvi,bai->bav', Vw, axes)                                        # (B, 26, 5)
    pmin, pmax = pv.min(-1).values, pv.max(-1).values
    bc = torch.einsum('ki,bai->bka', S['c'], axes)                                     # (B, K, 26)
    br = torch.einsum('ki,bai->bka', S['h'], axes.abs())
    ov = torch.minimum(pmax[:, None] - (bc - br), (bc + br) - pmin[:, None])
    ov = torch.where(valid[:, None, :], ov, torch.full_like(ov, math.inf))
    min_ov = ov.min(-1).values                                                         # (B, K)
    inter = min_ov > 0
    # --- exact distance (valid where separated)
    lo, hi = S['lo'], S['hi']
    cp = torch.maximum(torch.minimum(Vw[:, None], hi[None, :, None]), lo[None, :, None])       # (B, K, 5, 3)
    d1, s1, e1 = PG._argmin_pair((Vw[:, None] - cp).norm(dim=-1), Vw[:, None].expand_as(cp), cp)
    bcn = S['corners'][None].expand(B, -1, -1, -1)                                     # (B, K, 8, 3)
    loc = torch.einsum('bji,bkvj->bkvi', R, bcn - t[:, None, None, :])                 # body frame R^T (x - t)
    q = closest_point_triangle(loc[..., None, :], P['tri_a'], P['tri_b'], P['tri_c'])  # (B, K, 8, 6, 3)
    dq = (loc[..., None, :] - q).norm(dim=-1).reshape(B, K, 48)
    qw = (t[:, None, None, None, :] + torch.einsum('bij,bkvfj->bkvfi', R, q)).reshape(B, K, 48, 3)
    d2, s2, e2 = PG._argmin_pair(dq, qw, bcn[:, :, :, None, :].expand(B, K, 8, 6, 3).reshape(B, K, 48, 3))
    pe = t[:, None, None, :] + torch.einsum('bij,ekj->beki', R, P['edges'])            # (B, 8, 2, 3)
    be = S['edges']                                                                    # (K, 12, 2, 3)
    c1, c2 = PG._segseg(pe[:, None, :, None, 0, :], pe[:, None, :, None, 1, :],
                        be[None, :, None, :, 0, :], be[None, :, None, :, 1, :])        # (B, K, 8, 12, 3)
    c1, c2 = c1.reshape(B, K, 96, 3), c2.reshape(B, K, 96, 3)
    d3, s3, e3 = PG._argmin_pair((c1 - c2).norm(dim=-1), c1, c2)
    dd = torch.stack([d1, d2, d3], -1)                                                 # (B, K, 3)
    dk, which = dd.min(-1)
    clear_k = torch.where(inter, -min_ov, dk)
    ck, kk = clear_k.min(-1)
    use = ck < best
    best = torch.where(use, ck, best)
    if not witness:
        return best
    w = which[bi, kk]
    sk = torch.stack([s1[bi, kk], s2[bi, kk], s3[bi, kk]], 1)[bi, w]
    ek = torch.stack([e1[bi, kk], e2[bi, kk], e3[bi, kk]], 1)[bi, w]
    return best, torch.where(use[:, None], sk, s_star), torch.where(use[:, None], ek, e_star)
