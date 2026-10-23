"""Exact geometry of the stretchable link box in box scenes, its speeds and its 7-d chart.

Link. The box [-w/2, w/2]^2 x [0, L] in its body frame: drone a at the body origin (centre of the end face z = 0),
drone b at (0, 0, L) (centre of the far face). A pose is (t, R, L) with t = a, R e_z = (b - a) / |b - a| and
L = |b - a| in [L_min, L_max]. The models read the chart x = (t, rotvec(R) / 2 pi, L) (7 coordinates), the SE(3)
chart of every cell in this repository plus the length in normalized room units.

Scenes. A room box (the inner faces of the 'wall' group, found by ray casting as path_constrained.geometry.BoxScene)
minus axis-aligned obstacle boxes, normalized like preprocess_obj.py ((v - bbox centre) / largest bbox extent of the
OBJ). The obstacle boxes come from a <scene>_boxes.json next to the OBJ (written by
dataprocessing/make_link_scenes.py; a tree of the forest is a trunk and a crown, two boxes in one OBJ group).

Clearance. Exact signed distance between the link box and the scene, as path_constrained.geometry.prism_clearance
but with a length per pose: room slack of the 8 corners, and per obstacle the box-box distance (vertex-box,
box-vertex, 12 x 12 edge-edge) when a separating axis exists, minus the smallest of the 15 separating-axis overlaps
otherwise. The length limits are the seventh coordinate's walls: the clearance of a pose is

    c(t, R, L) = min(c_scene(t, R, L), L - L_min, L_max - L),

so a pose with c >= 0 has both drones clear of the scene, an unblocked line of sight and an allowed length.

Speeds (preprocess_obj.py's formulas): speed_dists = clip(c / margin, offset / margin, 1), speed_angles = the
smallest angle about drone a between a link surface point and a scene surface point in the same radial shell
(10 shells of equal link-point count, the shells of the box of that length), divided by pi; speed = their mean.
"""
import importlib.util
import json
import math
import os

import numpy as np
import torch

_PC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'path_constrained', 'geometry.py')
_spec = importlib.util.spec_from_file_location('_drone_link_pc_geometry', _PC)
PG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PG)

TWO_PI = 2.0 * math.pi
CHUNK = 8192            # poses per exact-clearance batch (about 0.4 GB per 1000 poses with 23 obstacle boxes)
so3_exp, so3_log = PG.so3_exp, PG.so3_log
quat_from_rotvec, rotvec_from_quat, quat_matrix = PG.quat_from_rotvec, PG.rotvec_from_quat, PG.quat_matrix
quat_angle, slerp, random_quats, random_rotvecs = PG.quat_angle, PG.slerp, PG.random_quats, PG.random_rotvecs


# ----------------------------------------------------------------------------- scenes
class Scene(PG.BoxScene):
    """BoxScene whose obstacles are listed in ``boxes`` (raw OBJ units, rows (x0, x1, y0, y1, z0, z1)); without a
    list every non-wall group of the OBJ must be one axis-aligned box (BoxScene's rule)."""

    def __init__(self, env_obj, boxes=None):
        if boxes is None:
            js = os.path.splitext(env_obj)[0] + '_boxes.json'
            if os.path.exists(js):
                with open(js) as fh:
                    boxes = json.load(fh)['boxes']
        if boxes is None:
            super().__init__(env_obj)
            return
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
        b = np.asarray(boxes, np.float64).reshape(-1, 6)
        self.box_lo = (b[:, [0, 2, 4]] - self.env_center) / self.env_scale
        self.box_hi = (b[:, [1, 3, 5]] - self.env_center) / self.env_scale
        assert (self.box_hi > self.box_lo).all()
        self._dev = {}

    def to_raw(self, p):
        return np.asarray(p) * self.env_scale + self.env_center


# ----------------------------------------------------------------------------- the link
class Link:
    """The stretchable link: width w, length limits [L_min, L_max] (normalized room units)."""

    def __init__(self, width, L_min, L_max):
        self.w, self.L_min, self.L_max = float(width), float(L_min), float(L_max)
        assert 0 < self.L_min < self.L_max

    def corners(self, L):
        """(B, 8, 3) body-frame corners for lengths L (B,), in path_constrained.geometry's corner order."""
        hw = 0.5 * self.w
        bits = torch.as_tensor(PG._BITS, dtype=L.dtype, device=L.device)
        B = L.shape[0]
        x = ((2 * bits[:, 0] - 1) * hw).expand(B, 8)
        y = ((2 * bits[:, 1] - 1) * hw).expand(B, 8)
        z = bits[:, 2][None] * L[:, None]
        return torch.stack([x, y, z], -1)

    def mesh(self, L):
        """(V (8, 3), F (12, 3)) of the box of length L (outward triangles), body frame."""
        return PG.Prism(self.w, float(L)).mesh()

    def slack(self, L):
        """Distance of the lengths L to the nearer length limit (negative outside [L_min, L_max])."""
        return torch.minimum(L - self.L_min, self.L_max - L)

    def ends(self, t, R, L):
        """World positions of drone a and drone b (B, 3) each."""
        return t, t + R[..., :, 2] * L[..., None]


def scene_clearance(scene, link, t, R, L, witness=False, chunk=None):
    """Exact signed clearance (B,) of the link boxes at poses (t (B, 3), R (B, 3, 3), L (B,)) against the room and
    the obstacles (< 0: collision, minus the smallest separating-axis overlap); with ``witness`` also the closest
    link point s* and scene point e* (B, 3) each (meaningful where the clearance is > 0)."""
    chunk = chunk or CHUNK
    if t.shape[0] > chunk:
        parts = [scene_clearance(scene, link, t[i:i + chunk], R[i:i + chunk], L[i:i + chunk], witness, chunk)
                 for i in range(0, t.shape[0], chunk)]
        if witness:
            return tuple(torch.cat([p[k] for p in parts]) for k in range(3))
        return torch.cat(parts)
    S = scene.tensors(t.device, t.dtype)
    B = t.shape[0]
    bi = torch.arange(B, device=t.device)
    C = link.corners(L)                                                              # (B, 8, 3) body frame
    Vp = t[:, None, :] + torch.einsum('bij,bvj->bvi', R, C)                          # (B, 8, 3) world corners
    slack = torch.cat([Vp - S['room_lo'], S['room_hi'] - Vp], -1).reshape(B, -1)     # (B, 8 * 6)
    best, ri = slack.min(-1)
    if witness:
        s_star = Vp[bi, ri // 6]
        e_star = s_star.clone()
        face = ri % 6
        ax = face % 3
        e_star[bi, ax] = torch.where(face < 3, S['room_lo'][ax], S['room_hi'][ax])
    if scene.n_boxes == 0:
        return (best, s_star, e_star) if witness else best
    hw = 0.5 * link.w
    half = torch.stack([torch.full_like(L, hw), torch.full_like(L, hw), 0.5 * L], -1)  # (B, 3) half extents
    c_p = t + R[:, :, 2] * (0.5 * L)[:, None]                                        # (B, 3) box centre
    hb = S['h']                                                                      # (K, 3)
    d = c_p[:, None, :] - S['c'][None]                                               # (B, K, 3)
    # --- separating axis test (15 axes) -> smallest overlap (> 0 on every axis = intersection)
    Ra = R.abs()
    ov = [(hb[None] + (Ra * half[:, None, :]).sum(-1)[:, None, :]) - d.abs()]       # world axes
    dj = torch.einsum('bki,bij->bkj', d, R)
    ov.append((torch.einsum('ki,bij->bkj', hb, Ra) + half[:, None, :]) - dj.abs())  # link axes
    E = torch.eye(3, dtype=t.dtype, device=t.device)
    n = torch.cross(E[None, :, None, :].expand(B, 3, 3, 3), R.transpose(1, 2)[:, None, :, :].expand(B, 3, 3, 3),
                    dim=-1).reshape(B, 9, 3)                                         # e_i x R_:j
    nn_ = n.norm(dim=-1)
    rP = (torch.einsum('bni,bij->bnj', n, R).abs() * half[:, None, :]).sum(-1)      # (B, 9)
    rB = torch.einsum('bni,ki->bkn', n.abs(), hb)                                    # (B, K, 9)
    dist = torch.einsum('bki,bni->bkn', d, n).abs()
    ovc = (rP[:, None, :] + rB - dist) / nn_.clamp_min(1e-9)[:, None, :]
    ov.append(torch.where(nn_[:, None, :] > 1e-6, ovc, torch.full_like(ovc, math.inf)))
    min_ov = torch.cat(ov, -1).min(-1).values                                        # (B, K)
    inter = min_ov > 0
    # --- exact distance where separated: link corners -> boxes, box corners -> link, edges x edges
    lo, hi = S['lo'], S['hi']
    cp = torch.maximum(torch.minimum(Vp[:, None], hi[None, :, None]), lo[None, :, None])     # (B, K, 8, 3)
    d1, s1, e1 = PG._argmin_pair((Vp[:, None] - cp).norm(dim=-1), Vp[:, None].expand_as(cp), cp)
    bc = S['corners'][None].expand(B, -1, -1, -1)                                     # (B, K, 8, 3)
    loc = torch.einsum('bji,bkvj->bkvi', R, bc - t[:, None, None, :])                # R^T (x - t)
    pl = torch.stack([torch.full_like(L, -hw), torch.full_like(L, -hw), torch.zeros_like(L)], -1)
    ph = torch.stack([torch.full_like(L, hw), torch.full_like(L, hw), L], -1)
    cl = torch.maximum(torch.minimum(loc, ph[:, None, None, :]), pl[:, None, None, :])
    cw = t[:, None, None, :] + torch.einsum('bij,bkvj->bkvi', R, cl)
    d2, s2, e2 = PG._argmin_pair((loc - cl).norm(dim=-1), cw, bc)
    pe = t[:, None, None, :] + torch.einsum('bij,bekj->beki', R, C[:, PG._EDGES])    # (B, 12, 2, 3)
    be = S['edges']                                                                  # (K, 12, 2, 3)
    c1, c2 = PG._segseg(pe[:, None, :, None, 0, :], pe[:, None, :, None, 1, :],
                        be[None, :, None, :, 0, :], be[None, :, None, :, 1, :])      # (B, K, 12, 12, 3)
    c1 = c1.reshape(B, -1, 144, 3)
    c2 = c2.reshape(B, -1, 144, 3)
    d3, s3, e3 = PG._argmin_pair((c1 - c2).norm(dim=-1), c1, c2)
    dk, which = torch.stack([d1, d2, d3], -1).min(-1)                                # (B, K)
    ck, kk = torch.where(inter, -min_ov, dk).min(-1)
    use = ck < best
    best = torch.where(use, ck, best)
    if not witness:
        return best
    w_ = which[bi, kk]
    sk = torch.stack([s1[bi, kk], s2[bi, kk], s3[bi, kk]], 1)[bi, w_]
    ek = torch.stack([e1[bi, kk], e2[bi, kk], e3[bi, kk]], 1)[bi, w_]
    return best, torch.where(use[:, None], sk, s_star), torch.where(use[:, None], ek, e_star)


def clearance(scene, link, t, R, L, parts=False):
    """c = min(scene clearance, L - L_min, L_max - L) (B,); with ``parts`` also (scene clearance, length slack)."""
    cs = scene_clearance(scene, link, t, R, L)
    sl = link.slack(L)
    c = torch.minimum(cs, sl)
    return (c, cs, sl) if parts else c


def normal7(link, t, R, L, s_star, e_star, length_bound):
    """Unit gradient (B, 7) of the clearance in the chart (t, rotvec / 2 pi, L), the conventions of
    preprocess_obj.calculate_dist: translation block = the workspace normal n (from the scene point to the link),
    rotation block = 2 pi x its moment about drone a, length block = zeta n . R e_z with zeta = the witness point's
    body z / L (a point at fraction zeta of the box moves by zeta dL when the far face moves by dL). Where the length
    limit is the binding clearance (``length_bound`` (B,) in {-1: L_min, +1: L_max, 0: the scene}) the gradient is
    -+e_L."""
    n_ws = s_star - e_star
    n_ws = n_ws / n_ws.norm(dim=1, keepdim=True).clamp_min(1e-12)
    r = s_star - t
    mom = torch.cross(r, n_ws, dim=1)
    ez = R[:, :, 2]
    zeta = ((r * ez).sum(1) / L.clamp_min(1e-9)).clamp(0, 1)
    dL = zeta * (n_ws * ez).sum(1)
    v = torch.cat([n_ws, TWO_PI * mom, dL[:, None]], 1)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    eL = torch.zeros_like(v)
    eL[:, 6] = 1.0
    v = torch.where((length_bound < 0)[:, None], eL, v)
    v = torch.where((length_bound > 0)[:, None], -eL, v)
    return v


# ----------------------------------------------------------------------------- angular clearance
class LinkAngle:
    """speed_angles of preprocess_obj.evaluate_placements for the link box of any length (module doc).

    The box's ``num`` surface samples are area-uniform for each length from common random numbers (face chosen by
    area, position uniform on it), sorted by distance from drone a and split into ``bins`` shells of equal count;
    the scene samples (``env_points``, normalized) in shell i are those at distance (edge_i, edge_i+1] from a, as in
    the original. The answer is the smallest angle between a link direction and a scene direction of the same
    shell (pi if every shell is empty)."""

    def __init__(self, env_points, width, device, num=1000, bins=10, seed=0):
        rng = np.random.default_rng(seed)
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.w, self.num, self.bins, self.per = float(width), num, bins, num // bins
        self.s, self.u, self.v = f(rng.random(num)), f(rng.random(num)), f(rng.random(num))
        self.env = f(env_points)
        self.device = device

    def shape_points(self, L):
        """(B, num, 3) body-frame surface samples of the boxes of lengths L (B,)."""
        w, hw = self.w, 0.5 * self.w
        B = L.shape[0]
        side = w * L
        cap = torch.full_like(L, w * w)
        areas = torch.stack([side, side, side, side, cap, cap], 1)
        cum = torch.cumsum(areas, 1) / areas.sum(1, keepdim=True)
        face = torch.searchsorted(cum, self.s.expand(B, -1).contiguous(), right=True).clamp(max=5)
        u = ((self.u - 0.5) * w).expand(B, -1)
        cu = ((self.v - 0.5) * w).expand(B, -1)
        zl = self.v[None] * L[:, None]
        Lb = L[:, None].expand(B, self.num)
        x = torch.where(face == 0, -hw, torch.where(face == 1, hw, u))
        y = torch.where(face <= 1, u, torch.where(face == 2, -hw, torch.where(face == 3, hw, cu)))
        z = torch.where(face <= 3, zl, torch.where(face == 4, torch.zeros_like(zl), Lb))
        return torch.stack([x, y, z], -1)

    def __call__(self, t, R, L, chunk=64):
        out = []
        for i in range(0, t.shape[0], chunk):
            out.append(self._one(t[i:i + chunk], R[i:i + chunk], L[i:i + chunk]))
        return torch.cat(out) if out else torch.zeros(0, device=self.device)

    def _one(self, t, R, L):
        B, bins, per = t.shape[0], self.bins, self.per
        P = self.shape_points(L)
        rad = P.norm(dim=-1)
        rad, order = rad.sort(1)
        P = torch.gather(P, 1, order[..., None].expand(-1, -1, 3))
        keep = per * bins
        P, rad = P[:, :keep].reshape(B, bins, per, 3), rad[:, :keep].reshape(B, bins, per)
        edges = torch.cat([rad[:, :, 0], rad[:, -1, -1:]], 1).contiguous()           # (B, bins + 1)
        su = torch.einsum('bij,bspj->bspi', R, P)
        su = su / su.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ev = self.env[None] - t[:, None, :]                                          # (B, E, 3)
        dist = ev.norm(dim=-1)
        sd, si = dist.sort(1)
        pos = torch.searchsorted(sd, edges, right=True)                              # (B, bins + 1)
        counts = pos[:, 1:] - pos[:, :-1]
        m = int(counts.max().clamp(min=1))
        slot = torch.arange(m, device=t.device)
        starts = pos[:, :-1, None]
        valid = slot < counts[..., None]                                             # (B, bins, m)
        cols = torch.where(valid, starts + slot, starts).clamp(max=sd.shape[1] - 1)
        idx = torch.gather(si, 1, cols.reshape(B, -1))
        eu = torch.gather(ev, 1, idx[..., None].expand(-1, -1, 3)).reshape(B, bins, m, 3)
        eu = eu / eu.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        best = torch.full((B,), -1.0, device=t.device)
        for s in range(0, m, 512):
            cos = torch.einsum('bspi,bsmi->bspm', su, eu[:, :, s:s + 512]).max(2).values  # (B, bins, mc)
            cos = torch.where(valid[:, :, s:s + 512], cos, torch.full_like(cos, -1.0))
            best = torch.maximum(best, cos.amax((1, 2)))
        return torch.arccos(best.clamp(-1.0, 1.0))


# ----------------------------------------------------------------------------- chart
def to_chart(t, rotvec, L):
    """numpy or torch (N, 3), (N, 3) radians, (N,) -> (N, 7) chart (t, rotvec / 2 pi, L)."""
    if isinstance(t, torch.Tensor):
        return torch.cat([t, rotvec / TWO_PI, L[:, None]], 1)
    return np.concatenate([t, np.asarray(rotvec) / TWO_PI, np.asarray(L)[:, None]], 1)


def from_chart(x):
    """torch (N, 7) chart -> (t (N, 3), R (N, 3, 3), L (N,))."""
    return x[:, :3], so3_exp(TWO_PI * x[:, 3:6]), x[:, 6]


def chart_clearance(scene, link, x, parts=False):
    t, R, L = from_chart(x)
    return clearance(scene, link, t, R, L, parts=parts)


def pose_dist(x, y):
    """sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2) between chart poses (N, 7) (theta = geodesic angle)."""
    Rx, Ry = so3_exp(TWO_PI * x[:, 3:6]), so3_exp(TWO_PI * y[:, 3:6])
    tr = (Rx * Ry).sum((1, 2))
    th = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
    return torch.sqrt(((x[:, :3] - y[:, :3]) ** 2).sum(1) + (th / TWO_PI) ** 2 + (x[:, 6] - y[:, 6]) ** 2)


def wrap_rotvec(rv):
    """Rotation vectors (N, 3) (radians) with the angle wrapped into [0, pi] (the same rotation)."""
    ang = rv.norm(dim=1, keepdim=True)
    ax = rv / ang.clamp_min(1e-12)
    w = ((ang + math.pi) % TWO_PI) - math.pi
    return torch.where(ang < 1e-8, rv, ax * w)


def load_scene(meta, repo):
    """Scene of a cell / test set from its meta.json (absolute path, else the repository-relative one)."""
    p = meta['env_obj'] if os.path.exists(meta['env_obj']) else os.path.join(repo, meta['env_obj_rel'])
    return Scene(p)


def load_link(meta):
    c = meta['link']
    return Link(c['width'], c['L_min'], c['L_max'])
