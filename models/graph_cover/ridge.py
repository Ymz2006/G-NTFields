"""Ridge detection on a learned travel-time field (stage 1 of the "movable ridges" plan, 2026-10-13).

A smooth network cannot form the concave corner where two routes tie; it rounds it over some width. Model the
rounding as a soft minimum of two branches T_1, T_2 with unit gradients p_1, p_2:

    T_tau = -tau log(exp(-T_1 / tau) + exp(-T_2 / tau)).

On the tie the gradient is (p_1 + p_2) / 2, so (i) the derivative across the ridge (along n ~ p_1 - p_2) is
zero exactly on the tie, (ii) |grad T| = cos(alpha / 2) < 1 (alpha the angle between the routes), and (iii)
the curvature across the ridge is about -(1 - p_1 . p_2) / (2 tau): strongly negative. A point is on the
learned ridge when the speed drops (S |c_x| < 1 noticeably), the Hessian of T has a strongly negative
eigenvalue lambda_min with eigenvector v, and the cross derivative r = c_x . v vanishes. The goal cone has
|c| < 1 as well but positive curvature, which separates it.

All derivatives are body-frame derivatives of T(x exp(xi), y) at xi = 0 for the start x (scaled twists
xi = (s_t v, s_r w)), so nothing depends on a chart. In high dimensions the ridge is described as the zero
set {r = 0} inside the band, not traced: `bisect_along` locates it on any line; `scan_line` evaluates the
detector along a parametrised curve.
"""
import math

import torch

from models.graph_method_triangle.geometry import se3


def body_derivatives(value_fn, tx, Rx, ty, Ry, s_t=1.0, s_r=1.0, hessian=True):
    """T (B,), c (B, 6) and H (B, 6, 6): value, gradient and Hessian of xi -> T(x exp(xi * scale), y) at 0."""
    B = tx.shape[0]
    scale = torch.tensor([s_t] * 3 + [s_r] * 3, dtype=tx.dtype, device=tx.device)
    xi = torch.zeros(B, 6, dtype=tx.dtype, device=tx.device, requires_grad=True)
    with torch.enable_grad():
        t2, R2 = se3.right_perturb(tx, Rx, xi * scale)
        T = value_fn(se3.features(t2, R2), se3.features(ty, Ry))
        c, = torch.autograd.grad(T.sum(), xi, create_graph=hessian)
        H = None
        if hessian:
            rows = [torch.autograd.grad(c[:, i].sum(), xi, retain_graph=True)[0] for i in range(6)]
            H = torch.stack(rows, 1)
            H = 0.5 * (H + H.transpose(1, 2))
    return T.detach(), c.detach(), (H.detach() if H is not None else None)


def ridge_features(c, H, speed=None, orient=None):
    """Per-sample detector quantities.

    Returns dict: cn = S |c| (speed-corrected gradient norm, 1 away from kinks), lam_min, v (unit eigenvector of
    lam_min, oriented so that v . orient >= 0 when `orient` is given), r = c . v (cross derivative).
    """
    lam, V = torch.linalg.eigh(H)
    lam_min = lam[:, 0]
    v = V[:, :, 0]
    if orient is not None:
        sgn = torch.sign((v * orient).sum(-1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        v = v * sgn
    cn = torch.linalg.norm(c, dim=-1)
    if speed is not None:
        cn = cn * speed
    r = (c * v).sum(-1)
    return {'cn': cn, 'lam_min': lam_min, 'v': v, 'r': r}


def detect(c, H, speed=None, band=0.95, kappa=5.0, orient=None):
    """Boolean ridge-band mask: S|c| < band and lam_min < -kappa. (kappa in units of 1 / time; the goal cone
    has lam ~ +1 / T, so a negative threshold of a few units is far from it.)"""
    f = ridge_features(c, H, speed, orient)
    return (f['cn'] < band) & (f['lam_min'] < -kappa), f


def scan_line(value_fn, pose_fn, params, ty, Ry, s_t=1.0, s_r=1.0, orient=None, speed_fn=None):
    """Evaluate the detector along a 1-parameter family of starts x(params) = pose_fn(params).

    pose_fn: (P,) -> (t (P, 3), R (P, 3, 3)). Returns dict of (P,) tensors: T, cn, lam_min, r, plus c, v.
    """
    tx, Rx = pose_fn(params)
    n = tx.shape[0]
    ty = ty.expand(n, -1) if ty.dim() == 2 and ty.shape[0] == 1 else ty
    Ry = Ry.expand(n, -1, -1) if Ry.shape[0] == 1 else Ry
    T, c, H = body_derivatives(value_fn, tx, Rx, ty, Ry, s_t, s_r)
    speed = speed_fn(tx, Rx) if speed_fn is not None else None
    f = ridge_features(c, H, speed, orient)
    return {'T': T, 'c': c, 'cn': f['cn'], 'lam_min': f['lam_min'], 'r': f['r'], 'v': f['v']}


def zero_crossings(params, r, mask=None):
    """Parameter values where r changes sign (linear interpolation), optionally restricted to mask."""
    out = []
    for i in range(len(params) - 1):
        if mask is not None and not (mask[i] or mask[i + 1]):
            continue
        a, b = float(r[i]), float(r[i + 1])
        if a == 0.0:
            out.append(float(params[i]))
        elif a * b < 0:
            out.append(float(params[i]) + (float(params[i + 1]) - float(params[i])) * a / (a - b))
    return out


def bisect_along(value_fn, tx, Rx, ty, Ry, u, lo, hi, s_t=1.0, s_r=1.0, iters=30):
    """Locate r = 0 on the line x(s) = x exp(s u) between s = lo and s = hi (per sample, u (B, 6) scaled twists)
    by bisection on the cross derivative, with the eigenvector oriented along u."""
    lo = lo.clone()
    hi = hi.clone()

    def r_at(s):
        t2, R2 = se3.right_perturb(tx, Rx, s[:, None] * u)
        _, c, H = body_derivatives(value_fn, t2, R2, ty, Ry, s_t, s_r)
        return ridge_features(c, H, orient=u)['r']

    r_lo = r_at(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r_mid = r_at(mid)
        same = torch.sign(r_mid) == torch.sign(r_lo)
        lo = torch.where(same, mid, lo)
        r_lo = torch.where(same, r_mid, r_lo)
        hi = torch.where(same, hi, mid)
    return 0.5 * (lo + hi)


class SoftMinFreeField:
    """Synthetic smooth field: soft minimum over the two rotation routes of the exact free-space time, with a
    fixed rounding width tau (a model of what a smooth network does to the half-turn ridge)."""

    def __init__(self, tau, s_t=1.0, s_r=1.0):
        self.tau, self.s_t, self.s_r = float(tau), float(s_t), float(s_r)

    def value(self, gx, gy):
        from models.graph_method_triangle import cover_decoder as cd
        qx = cd.quat_from_matrix(gx[:, 3:12].reshape(-1, 3, 3))
        qy = cd.quat_from_matrix(gy[:, 3:12].reshape(-1, 3, 3))
        a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1) / (self.s_t * self.s_t)
        c = (qx * qy).sum(-1).clamp(-1, 1)
        s2 = cd.route_s2(qx, qy)
        T2 = cd.free_routes_sq(a, s2, c, self.s_r)
        T = torch.sqrt(T2.clamp_min(1e-30))
        if self.tau <= 0:
            return T.amin(1)
        return -self.tau * torch.logsumexp(-T / self.tau, dim=1)


def hard_min_free(gx, gy, s_t=1.0, s_r=1.0):
    return SoftMinFreeField(0.0, s_t, s_r).value(gx, gy)
