"""Adversarial triangle-inequality check for any pair-value model.

Random pose triples almost never come close to equality in T(x, z) <= T(x, y) + T(y, z)
(the slack is O(T) for unrelated poses), so "0 violations on random triples"
says nothing. Here the violation

    v(x, y, z) = T(x, z) - T(x, y) - T(y, z)

is maximised by gradient ascent over body-frame twists of all three poses,
starting from y at the free-space geodesic midpoint of x and z (the equality
case of the analytic field). Heads with an architectural guarantee
(graph_tri*) must stay at rounding level; heads without one need not.
"""

import torch

from .fields import oracle
from .geometry import se3


def _safe_value(model, gx, gy):
    # the attack may drive two poses together; avoid sqrt'(0) = inf in its gradient
    if getattr(model, 'has_latent', False):
        return torch.sqrt(model.value_sq(gx, gy) + 1e-20)
    return model.value(gx, gy)


def midpoint(tx, Rx, tz, Rz):
    t, R = oracle.path_star(tx, Rx, tz, Rz, torch.tensor([0.5], dtype=tx.dtype, device=tx.device))
    return t[:, 0], R[:, 0]


def attack(model, tx, Rx, tz, Rz, ty=None, Ry=None, steps=300, lr=0.02, move_ends=True):
    """Returns per-triple max violation, the T(x, z) at that point and the poses.

    Evaluates the exact model.value on every iterate and keeps the best one,
    so the reported violation is a real (not surrogate) value.
    """
    if ty is None:
        ty, Ry = midpoint(tx, Rx, tz, Rz)
    B = tx.shape[0]
    xi = torch.zeros(3, B, 6, dtype=tx.dtype, device=tx.device, requires_grad=True)
    opt = torch.optim.Adam([xi], lr=lr)
    base = ((tx, Rx), (ty, Ry), (tz, Rz))
    best = torch.full((B,), -float('inf'), dtype=tx.dtype, device=tx.device)
    best_Txz = torch.zeros_like(best)
    best_poses = [None, None, None]
    for k in range(steps + 1):
        poses = [se3.right_perturb(t, R, xi[i] if (move_ends or i == 1) else xi[i].detach() * 0)
                 for i, (t, R) in enumerate(base)]
        g = [se3.features(t, R) for t, R in poses]
        with torch.no_grad():
            Txz = model.value(g[0], g[2])
            v = Txz - model.value(g[0], g[1]) - model.value(g[1], g[2])
            upd = v > best
            best = torch.where(upd, v, best)
            best_Txz = torch.where(upd, Txz, best_Txz)
            for i in range(3):
                t, R = poses[i]
                if best_poses[i] is None:
                    best_poses[i] = (t.detach().clone(), R.detach().clone())
                else:
                    best_poses[i][0][upd] = t.detach()[upd]
                    best_poses[i][1][upd] = R.detach()[upd]
        if k == steps:
            break
        vs = (_safe_value(model, g[0], g[2]) - _safe_value(model, g[0], g[1]) - _safe_value(model, g[1], g[2]))
        opt.zero_grad()
        (-vs.sum()).backward()
        with torch.no_grad():
            xi.grad.nan_to_num_(0.0)
        opt.step()
    return best, best_Txz, best_poses


def summarize(viol, Txz, tol_rel=1e-4):
    rel = viol / Txz.clamp_min(1e-12)
    return {'viol_max': viol.max().item(), 'viol_rel_max': rel.max().item(),
            'viol_frac': (rel > tol_rel).double().mean().item(),
            'viol_rel_p99': rel.quantile(0.99).item(), 'n': int(viol.numel())}
