"""Analytic obstacle-free oracle for the weighted product metric (spec page 3).

The oracle time deliberately does NOT go through decoder.phi: the angle is
computed independently with atan2 so that decoder bugs cannot hide inside the
labels.  Latent targets (a*, b*) are provided for supervised diagnostics only.
"""

import torch

from ..geometry import se3


def latent_star(tx, Rx, ty, Ry):
    """a* = |t_x - t_y|^2, b* = (3 - tr(R_x^T R_y)) / 4 (raw, unclamped)."""
    a = ((tx - ty) ** 2).sum(-1)
    tr = (Rx * Ry).sum((-1, -2))  # tr(R_x^T R_y)
    b = (3.0 - tr) / 4.0
    return a, b


def clamp_b(b):
    """Clamp tiny float excursions of b* outside [0, 1]; report how many."""
    n_bad = int(((b < 0) | (b > 1)).sum())
    return b.clamp(0.0, 1.0), n_bad


def relative_angle(Rx, Ry):
    return se3.rotation_angle(Rx.transpose(-1, -2) @ Ry)


def time_star(tx, Rx, ty, Ry, s_t, s_r):
    a = ((tx - ty) ** 2).sum(-1)
    th = relative_angle(Rx, Ry)
    return torch.sqrt(a / (s_t * s_t) + th * th / (s_r * s_r))


def optimal_twist_direction(tx, Rx, ty, Ry, s_t, s_r):
    """Normalized control u* (B, 6) of the exact geodesic at its start.

    Body-frame translation R_x^T (t_y - t_x) and rotation log(R_x^T R_y), both
    divided by the speed scales and normalized. Used by evaluation / analytic
    Bellman checks only.
    """
    dv = (Rx.transpose(-1, -2) @ (ty - tx).unsqueeze(-1)).squeeze(-1) / s_t
    dw = se3.so3_log(Rx.transpose(-1, -2) @ Ry) / s_r
    u = torch.cat([dv, dw], -1)
    return u / torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-300)


def path_star(tx, Rx, ty, Ry, lam):
    """Exact shortest path at fractions lam (K,) -> t (B, K, 3), R (B, K, 3, 3)."""
    w = se3.so3_log(Rx.transpose(-1, -2) @ Ry)  # (B, 3)
    lam = lam.to(tx.dtype)
    t = (1 - lam)[None, :, None] * tx[:, None] + lam[None, :, None] * ty[:, None]
    R = Rx[:, None] @ se3.so3_exp(lam[None, :, None] * w[:, None])
    return t, R
