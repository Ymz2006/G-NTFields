"""Admissible finite-step Bellman targets (spec page 4).

    target(x, y) = min_u [ h + T_target(x exp(h * hat(s_t u_t, s_r u_r)), y) ]
    L_B = mean[(T_theta(x, y) - stopgrad(target))^2] / tau_ref^2

Candidates are admissible normalized controls (|u| <= 1): the online model's
greedy direction -c_x/|c_x|, Gaussian perturbations of it, uniform unit
directions and the zero action.  Finite action sampling only approximates the
minimum (upper-biased), and a finite constant-twist step only approximates the
dynamic-programming operator.

Terminal treatment ('reach'): if the constant body twist that lands exactly on
y, xi = log(x^-1 y), satisfies |scaled xi| <= h, the goal is reachable within
the step and only the elapsed time |scaled xi| is charged.  This uses pose
geometry only (no oracle time labels); it is the declared boundary treatment.
"""

import torch

from ..geometry import se3


def scale_vector(s_t, s_r, ref):
    return torch.tensor([s_t] * 3 + [s_r] * 3, dtype=ref.dtype, device=ref.device)


def candidate_controls(u_greedy, k_rand, k_pert, sigma_pert, gen=None, include_zero=True):
    """(B, 6) greedy directions -> (B, K, 6) admissible candidate controls."""
    B, dev, dt = u_greedy.shape[0], u_greedy.device, u_greedy.dtype
    cands = [u_greedy[:, None]]
    if k_pert > 0:
        p = u_greedy[:, None] + sigma_pert * torch.randn(B, k_pert, 6, device=dev, dtype=dt, generator=gen)
        cands.append(p / torch.linalg.norm(p, dim=-1, keepdim=True).clamp_min(1e-12))
    if k_rand > 0:
        r = torch.randn(B, k_rand, 6, device=dev, dtype=dt, generator=gen)
        cands.append(r / torch.linalg.norm(r, dim=-1, keepdim=True).clamp_min(1e-12))
    if include_zero:
        cands.append(torch.zeros(B, 1, 6, device=dev, dtype=dt))
    return torch.cat(cands, 1)


def step_poses(tx, Rx, controls, h, s_t, s_r):
    """x exp(h hat(s * u)) for every candidate: (B,K,6) -> t (B,K,3), R (B,K,3,3)."""
    K = controls.shape[1]
    xi = h * controls * scale_vector(s_t, s_r, controls)
    t2, R2 = se3.right_perturb(tx[:, None].expand(-1, K, -1), Rx[:, None].expand(-1, K, -1, -1), xi)
    return t2, R2


def reach_time(tx, Rx, ty, Ry, s_t, s_r):
    """Duration of the constant body twist from x exactly to y (screw motion)."""
    ti, Ri = se3.inverse(tx, Rx)
    tr, Rr = se3.compose(ti, Ri, ty, Ry)
    xi = se3.se3_log(tr, Rr)
    return torch.sqrt((xi[:, :3] ** 2).sum(-1) / s_t ** 2 + (xi[:, 3:] ** 2).sum(-1) / s_r ** 2)


@torch.no_grad()
def bellman_target(target_model, tx, Rx, ty, Ry, controls, h, s_t, s_r, terminal=True, chunk=65536):
    B, K = controls.shape[:2]
    t2, R2 = step_poses(tx, Rx, controls, h, s_t, s_r)
    gx2 = se3.features(t2, R2).reshape(B * K, 12)
    gy2 = se3.features(ty, Ry)[:, None].expand(-1, K, -1).reshape(B * K, 12)
    vals = []
    for i in range(0, B * K, chunk):
        vals.append(target_model.value(gx2[i:i + chunk], gy2[i:i + chunk]))
    T_next = torch.cat(vals).view(B, K)
    best, idx = T_next.min(-1)
    target = h + best
    reached = torch.zeros_like(target, dtype=torch.bool)
    if terminal:
        tau = reach_time(tx, Rx, ty, Ry, s_t, s_r)
        reached = tau <= h
        target = torch.where(reached, torch.minimum(target, tau), target)
    return target, idx, reached
