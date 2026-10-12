"""SE(3) / SO(3) utilities with autodiff-safe series expansions.

Conventions (spec page 2):
    pose x = (t, R), t in R^3, R in SO(3)
    x * y = (t_x + R_x t_y, R_x R_y)
    twist xi = (v, w) in R^6, body frame; exp(hat(xi)) = (V(w) v, exp([w]_x))
    G(x) = concat(t, vec(R)) with vec = row-major flatten (12 entries)

Every function that divides by an angle switches to a Taylor series below a
small threshold, and evaluates the exact branch on a *safe* substitute input so
that the unused branch can never inject NaN/Inf into the backward pass.
"""

import math

import torch

# Below this squared angle the closed forms are replaced by their series.
_SMALL_SQ = 1e-6


def hat(w):
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix [w]_x."""
    o = torch.zeros_like(w[..., 0])
    wx, wy, wz = w.unbind(-1)
    return torch.stack([
        torch.stack([o, -wz, wy], -1),
        torch.stack([wz, o, -wx], -1),
        torch.stack([-wy, wx, o], -1),
    ], -2)


def vee(W):
    """Axial vector of the skew part's layout: (..., 3, 3) -> (..., 3).

    Assumes W is skew-symmetric; for a general matrix pass (M - M^T) / 2.
    """
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], -1)


def se3_hat(xi):
    """(..., 6) -> (..., 4, 4) Lie-algebra matrix for twist (v, w)."""
    v, w = xi[..., :3], xi[..., 3:]
    top = torch.cat([hat(w), v.unsqueeze(-1)], -1)
    bottom = torch.zeros_like(top[..., :1, :])
    return torch.cat([top, bottom], -2)


def _coeffs(theta_sq):
    """A = sin(t)/t, B = (1-cos t)/t^2, C = (t - sin t)/t^3 with safe series."""
    small = theta_sq < _SMALL_SQ
    safe_sq = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    th = torch.sqrt(safe_sq)
    A_ex = torch.sin(th) / th
    B_ex = (1.0 - torch.cos(th)) / safe_sq
    C_ex = (th - torch.sin(th)) / (safe_sq * th)
    t2 = theta_sq
    A_se = 1.0 - t2 / 6.0 + t2 * t2 / 120.0
    B_se = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
    C_se = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
    return (torch.where(small, A_se, A_ex),
            torch.where(small, B_se, B_ex),
            torch.where(small, C_se, C_ex))


def so3_exp(w):
    """Rodrigues: (..., 3) -> (..., 3, 3)."""
    theta_sq = (w * w).sum(-1)
    A, B, _ = _coeffs(theta_sq)
    W = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    return eye + A[..., None, None] * W + B[..., None, None] * (W @ W)


def so3_left_jacobian(w):
    """V(w) = I + B [w] + C [w]^2, the SE(3) translation Jacobian."""
    theta_sq = (w * w).sum(-1)
    _, B, C = _coeffs(theta_sq)
    W = hat(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    return eye + B[..., None, None] * W + C[..., None, None] * (W @ W)


def se3_exp(xi):
    """Twist (..., 6) -> pose (t (..., 3), R (..., 3, 3))."""
    v, w = xi[..., :3], xi[..., 3:]
    R = so3_exp(w)
    t = (so3_left_jacobian(w) @ v.unsqueeze(-1)).squeeze(-1)
    return t, R


def rotation_angle(R):
    """Robust relative angle in [0, pi] via atan2 (spec page 3 oracle form)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = (tr - 1.0) / 2.0
    s = torch.linalg.norm(vee(R - R.transpose(-1, -2)), dim=-1) / 2.0
    return torch.atan2(s, c)


def so3_log(R):
    """Stable SO(3) logarithm (..., 3, 3) -> (..., 3), angle in [0, pi].

    Uses the antisymmetric part away from pi and the symmetric part near pi.
    At an exact half-turn the axis sign is chosen consistently from the
    largest diagonal entry (both directions are valid geodesics).
    Intended for oracle paths and analysis; not differentiated in training.
    """
    theta = rotation_angle(R)
    skew = vee(R - R.transpose(-1, -2)) / 2.0  # = sin(theta) * n
    small = theta < 1e-4
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    factor = torch.where(small, 1.0 + theta * theta / 6.0,
                         safe_theta / torch.sin(safe_theta).clamp_min(1e-300))
    w_regular = factor[..., None] * skew

    # Near pi: R + R^T = 2 I + 2 (1 - cos) (n n^T - I)
    cos_t = torch.cos(theta)
    S = (R + R.transpose(-1, -2)) / 2.0
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    denom = (1.0 - cos_t).clamp_min(1e-12)[..., None, None]
    nnT = (S - cos_t[..., None, None] * eye) / denom
    diag = torch.diagonal(nnT, dim1=-2, dim2=-1)
    k = diag.argmax(-1)
    col = torch.gather(nnT, -1, k[..., None, None].expand(*nnT.shape[:-1], 1)).squeeze(-1)
    n = col / torch.linalg.norm(col, dim=-1, keepdim=True).clamp_min(1e-300)
    sign = torch.sign((n * skew).sum(-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    w_pi = (sign * theta)[..., None] * n

    near_pi = theta > math.pi - 1e-3
    return torch.where(near_pi[..., None], w_pi, w_regular)


def se3_log(t, R):
    """Pose -> twist (..., 6). Inverse of se3_exp for angles below pi."""
    w = so3_log(R)
    theta_sq = (w * w).sum(-1)
    A, B, _ = _coeffs(theta_sq)
    small = theta_sq < _SMALL_SQ
    safe_sq = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    D_ex = (1.0 - A / (2.0 * B)) / safe_sq
    D_se = 1.0 / 12.0 + theta_sq / 720.0
    D = torch.where(small, D_se, D_ex)
    W = hat(w)
    eye = torch.eye(3, dtype=t.dtype, device=t.device).expand(W.shape)
    V_inv = eye - 0.5 * W + D[..., None, None] * (W @ W)
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([v, w], -1)


def compose(t1, R1, t2, R2):
    """x * y = (t1 + R1 t2, R1 R2)."""
    return t1 + (R1 @ t2.unsqueeze(-1)).squeeze(-1), R1 @ R2


def inverse(t, R):
    Rt = R.transpose(-1, -2)
    return -(Rt @ t.unsqueeze(-1)).squeeze(-1), Rt


def right_perturb(t, R, xi):
    """x * exp(hat(xi)) with body-frame twist xi = (v, w)."""
    dt, dR = se3_exp(xi)
    return compose(t, R, dt, dR)


def quat_to_matrix(q):
    """Unit quaternion (w, x, y, z) -> rotation matrix. Sign invariant."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def features(t, R):
    """Fixed smooth pose embedding G(x) = concat(t, vec(R)) -> (..., 12)."""
    return torch.cat([t, R.reshape(*R.shape[:-2], 9)], -1)


def split_features(g):
    """Inverse of features(): (..., 12) -> (t, R)."""
    return g[..., :3], g[..., 3:].reshape(*g.shape[:-1], 3, 3)


def project_to_so3(M):
    """Nearest rotation (SVD); used only to clean accumulated rollout drift."""
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    D = torch.ones_like(M[..., 0])
    D[..., -1] = d
    return U @ torch.diag_embed(D) @ Vh
