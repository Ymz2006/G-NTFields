"""Pose-space planners on SE(3) (spec page 7).

Both planners are batched over N start/goal pairs, step with the actual SE(3)
exponential x_next = x exp(h hat(s_t u_t, s_r u_r)) and charge elapsed time,
which for unit normalized controls equals the weighted twist-norm path cost.

Terminal treatment (identical for every field): when the constant body twist
reaching the goal exactly has duration <= tol (= the step h by default), the
planner takes that action, charges its duration and stops (success).

gradient_rollout:  u = -c_x / |c_x| from the decoded time (never the latent
                   alone), backtracking (halve h, up to `backtrack` times) when
                   the decoded value fails to decrease; non-finite / tiny
                   gradients or failed backtracking fall back to one sampling
                   step, and the event is counted.
sampling_rollout:  `n_cand` normalized twist candidates per step: zero action,
                   previous action, perturbations of the previous action and
                   uniform directions; picks argmin h + T(next, y).
"""

import time

import torch

from ..geometry import se3, tangents
from ..losses.bellman import reach_time, scale_vector


def _value(model, tx, Rx, ty, Ry, chunk=65536):
    gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
    out = []
    for i in range(0, gx.shape[0], chunk):
        out.append(model.value(gx[i:i + chunk], gy[i:i + chunk]))
    return torch.cat(out) if out else gx.new_zeros(0)


def _sample_candidates(prev, n_cand, sigma, gen):
    n, dev, dt = prev.shape[0], prev.device, prev.dtype
    n_rest = n_cand - 2
    n_pert = n_rest // 2
    n_rand = n_rest - n_pert
    pert = prev[:, None] + sigma * torch.randn(n, n_pert, 6, device=dev, dtype=dt, generator=gen)
    rand = torch.randn(n, n_rand, 6, device=dev, dtype=dt, generator=gen)
    cands = torch.cat([prev[:, None], pert, rand], 1)
    norms = torch.linalg.norm(cands, dim=-1, keepdim=True)
    cands = torch.where(norms > 1e-12, cands / norms.clamp_min(1e-12), cands)
    zero = torch.zeros(n, 1, 6, device=dev, dtype=dt)
    return torch.cat([zero, cands], 1)


@torch.no_grad()
def _sampling_step(model, tx, Rx, ty, Ry, prev, h, s_t, s_r, n_cand, sigma, gen):
    U = _sample_candidates(prev, n_cand, sigma, gen)
    n, K = U.shape[:2]
    xi = h * U * scale_vector(s_t, s_r, U)
    t2, R2 = se3.right_perturb(tx[:, None].expand(-1, K, -1), Rx[:, None].expand(-1, K, -1, -1), xi)
    T2 = _value(model, t2.reshape(-1, 3), R2.reshape(-1, 3, 3),
                ty[:, None].expand(-1, K, -1).reshape(-1, 3),
                Ry[:, None].expand(-1, K, -1, -1).reshape(-1, 3, 3)).view(n, K)
    idx = T2.argmin(-1)
    ar = torch.arange(n, device=tx.device)
    return t2[ar, idx], R2[ar, idx], U[ar, idx], (idx > 0).to(tx.dtype)


def _pose_errors(tx, Rx, ty, Ry):
    et = torch.linalg.norm(tx - ty, dim=-1)
    er = se3.rotation_angle(Rx.transpose(-1, -2) @ Ry)
    return et, er


def rollout(model, tx, Rx, ty, Ry, s_t=1.0, s_r=1.0, mode='gradient', h=0.02, max_steps=600,
            tol=None, n_cand=128, sigma=0.3, backtrack=4, seed=0, record_path=False):
    """Plan from x to y for every row. Returns a dict of per-pair tensors."""
    tol = h if tol is None else tol
    dev, dt = tx.device, tx.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = tx.shape[0]
    tx, Rx = tx.clone(), Rx.clone()
    cost = torch.zeros(n, device=dev, dtype=dt)
    steps = torch.zeros(n, device=dev, dtype=torch.long)
    fallback = torch.zeros(n, device=dev, dtype=torch.long)
    done = torch.zeros(n, device=dev, dtype=torch.bool)
    prev = torch.zeros(n, 6, device=dev, dtype=dt)
    path = [(tx.clone(), Rx.clone())] if record_path else None
    t0 = time.time()
    for k in range(max_steps):
        # terminal action: goal reachable within tolerance
        tau = reach_time(tx, Rx, ty, Ry, s_t, s_r)
        arrive = (~done) & (tau <= tol)
        if arrive.any():
            cost[arrive] += tau[arrive]
            tx[arrive], Rx[arrive] = ty[arrive], Ry[arrive]
            done |= arrive
        act = ~done
        if not act.any():
            break
        ia = act.nonzero().squeeze(-1)
        a_tx, a_Rx, a_ty, a_Ry = tx[ia], Rx[ia], ty[ia], Ry[ia]
        if mode == 'gradient':
            with torch.enable_grad():
                T, cx, _ = tangents.tangent_gradients(model.value, se3.features(a_tx, a_Rx),
                                                      se3.features(a_ty, a_Ry), s_t, s_r,
                                                      create_graph=False)
            T, cx = T.detach(), cx.detach()
            gnorm = torch.linalg.norm(cx, dim=-1)
            ok = torch.isfinite(gnorm) & (gnorm > 1e-6) & torch.isfinite(T)
            u = torch.where(ok[:, None], -cx / gnorm.clamp_min(1e-12)[:, None], torch.zeros_like(cx))
            new_t, new_R = a_tx.clone(), a_Rx.clone()
            accepted = torch.zeros_like(ok)
            step_h = torch.full_like(T, h)
            for _ in range(backtrack + 1):
                pend = ok & ~accepted
                if not pend.any():
                    break
                xi = step_h[pend, None] * u[pend] * scale_vector(s_t, s_r, u)
                t2, R2 = se3.right_perturb(a_tx[pend], a_Rx[pend], xi)
                with torch.no_grad():
                    T2 = _value(model, t2, R2, a_ty[pend], a_Ry[pend])
                good = T2 < T[pend]
                idx = pend.nonzero().squeeze(-1)
                gi = idx[good]
                new_t[gi], new_R[gi] = t2[good], R2[good]
                accepted[gi] = True
                step_h[idx[~good]] *= 0.5
            took = accepted
            cost_inc = torch.where(took, step_h, torch.zeros_like(step_h))
            fb = ~took
            if fb.any():
                fi = fb.nonzero().squeeze(-1)
                st, sR, su, _ = _sampling_step(model, a_tx[fi], a_Rx[fi], a_ty[fi], a_Ry[fi],
                                               prev[ia][fi], h, s_t, s_r, n_cand, sigma, gen)
                new_t[fi], new_R[fi] = st, sR
                u[fi] = su
                cost_inc[fi] = h
                fallback[ia[fi]] += 1
            tx[ia], Rx[ia] = new_t, new_R
            prev[ia] = u
            cost[ia] += cost_inc
        else:
            st, sR, su, _ = _sampling_step(model, a_tx, a_Rx, a_ty, a_Ry, prev[ia], h, s_t, s_r,
                                           n_cand, sigma, gen)
            tx[ia], Rx[ia] = st, sR
            prev[ia] = su
            cost[ia] += h
        steps[ia] += 1
        if k % 50 == 49:  # remove accumulated round-off from the rotation
            Rx = se3.project_to_so3(Rx)
        if record_path:
            path.append((tx.clone(), Rx.clone()))
    et, er = _pose_errors(tx, Rx, ty, Ry)
    return {'success': done, 'cost': cost, 'steps': steps, 'fallback': fallback,
            'err_t': et, 'err_r': er, 'runtime': time.time() - t0, 'path': path}
