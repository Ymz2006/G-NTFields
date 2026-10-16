"""Stages 2 and 3 of the ridge plan on obstacle-free SE(3): planning with the ridge detector.

Gradient rollouts fail in the rounded band of a smooth field because the gradient there averages two routes.
Three planners are compared on pairs stratified by relative rotation (as plan_by_angle_free.py):

  gradient        planning/rollout.py gradient descent with backtracking and its sampling fallback (baseline)
  ridge_sample    stage 2: when the detector fires at the current pose (S|c| < band and lambda_min < -kappa) the
                  step is a 128-candidate sampling step instead of the gradient step
  ridge_sharpen   stage 3: when the detector fires, the two routes are recovered just outside the band on either
                  side of the ridge (x exp(+-d v), v the eigenvector of lambda_min) and the field is replaced by
                  the minimum of their first-order extrapolations; the step follows the gradient of the winning
                  branch (= the gradient outside the band on that side)

Fields: trained obstacle-free checkpoints (graph_method / graph_method_triangle format) and the synthetic
soft-min model of the exact field (--softmin TAU). Per bin: success, cost ratio (path time / T*) median and p95,
steps, detector firings per path, fallbacks. Also reports the stage-3 value error at the exact half-turn
(sharpened vs raw) on the anchors of ridge_free.py.

    python models/graph_cover/scripts/ridge_plan_free.py --out results/graph_cover/ridge_plan \
        --ckpt metric_best=Experiments/graph_method_final/metric_best_recipe_s0/best.pt --softmin 0.1 [--n 300]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.append('.')

import torch

from models.graph_cover import ridge
from models.graph_method_triangle import model_train_graph as mt
from models.graph_method_triangle.data.poses import PoseSampler
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3
from models.graph_method_triangle.losses.bellman import reach_time, scale_vector
from models.graph_method_triangle.planning import rollout as ro

BINS = [('90-150', 90, 150), ('150-170', 150, 170), ('170-179', 170, 179), ('179-180', 179, 180)]


def make_pairs(sampler, n, lo, hi, pure_rot=False):
    tx, Rx = sampler.uniform_poses(n)
    ty = tx.clone() if pure_rot else sampler.translations(n)
    axis = sampler.unit_vectors(n, 3)
    th = (lo + (hi - lo) * sampler._rand(n)) * math.pi / 180.0
    Ry = Rx @ se3.so3_exp(th[:, None] * axis)
    return tx, Rx, ty, Ry


def sharpened_direction(value_fn, tx, Rx, ty, Ry, v, d, s_t, s_r):
    """Recover the two routes across the ridge: probes x exp(+-d v); returns (T_sharp, c of the winning side)."""
    out_T, out_c = [], []
    for sgn in (1.0, -1.0):
        t2, R2 = se3.right_perturb(tx, Rx, sgn * d * v)
        T2, c2, _ = ridge.body_derivatives(value_fn, t2, R2, ty, Ry, s_t, s_r, hessian=False)
        # first-order extrapolation back to x: x = x_i exp(-sgn d v)
        out_T.append(T2 - sgn * d * (c2 * v).sum(-1))
        out_c.append(c2)
    T1, T2 = out_T
    pick = (T1 <= T2)
    return torch.where(pick, T1, T2), torch.where(pick[:, None], out_c[0], out_c[1])


def rollout_ridge(model, tx, Rx, ty, Ry, s_t, s_r, mode, h, max_steps, band, kappa, probe, n_cand=128,
                  sigma=0.3, backtrack=4, seed=0):
    """Gradient planner with detector hooks (copy of rollout.rollout's gradient loop plus the ridge branch)."""
    tol = h
    dev, dt = tx.device, tx.dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = tx.shape[0]
    tx, Rx = tx.clone(), Rx.clone()
    cost = torch.zeros(n, device=dev, dtype=dt)
    steps = torch.zeros(n, device=dev, dtype=torch.long)
    fallback = torch.zeros(n, device=dev, dtype=torch.long)
    fired = torch.zeros(n, device=dev, dtype=torch.long)
    done = torch.zeros(n, device=dev, dtype=torch.bool)
    prev = torch.zeros(n, 6, device=dev, dtype=dt)
    t0 = time.time()
    for k in range(max_steps):
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
        T, cx, H = ridge.body_derivatives(model.value, a_tx, a_Rx, a_ty, a_Ry, s_t, s_r, hessian=True)
        hit, feat = ridge.detect(cx, H, band=band, kappa=kappa)
        fired[ia[hit]] += 1
        if mode == 'ridge_sharpen' and hit.any():
            hi = hit.nonzero().squeeze(-1)
            Ts, cs = sharpened_direction(model.value, a_tx[hi], a_Rx[hi], a_ty[hi], a_Ry[hi], feat['v'][hi],
                                         probe, s_t, s_r)
            cx = cx.clone()
            cx[hi] = cs
        gnorm = torch.linalg.norm(cx, dim=-1)
        ok = torch.isfinite(gnorm) & (gnorm > 1e-6) & torch.isfinite(T)
        if mode == 'ridge_sample':
            ok = ok & ~hit
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
                T2 = ro._value(model, t2, R2, a_ty[pend], a_Ry[pend])
            if mode == 'ridge_sharpen':
                # accept on the raw field unless inside the band, where any finite step along the branch is taken
                good = (T2 < T[pend]) | hit[pend]
            else:
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
            st, sR, su, _ = ro._sampling_step(model, a_tx[fi], a_Rx[fi], a_ty[fi], a_Ry[fi], prev[ia][fi], h,
                                              s_t, s_r, n_cand, sigma, gen)
            new_t[fi], new_R[fi] = st, sR
            u[fi] = su
            cost_inc[fi] = h
            fallback[ia[fi]] += 1
        tx[ia], Rx[ia] = new_t, new_R
        prev[ia] = u
        cost[ia] += cost_inc
        steps[ia] += 1
        if k % 50 == 49:
            Rx = se3.project_to_so3(Rx)
    et, er = ro._pose_errors(tx, Rx, ty, Ry)
    return {'success': done, 'cost': cost, 'steps': steps, 'fallback': fallback, 'fired': fired, 'err_t': et,
            'err_r': er, 'runtime': time.time() - t0}


def q(x, p):
    return x.quantile(p).item() if x.numel() else float('nan')


def summarize(r, Ts):
    s = r['success']
    cr = r['cost'][s] / Ts[s]
    return {'success': s.double().mean().item(), 'cost_ratio_p50': q(cr, 0.5), 'cost_ratio_p95': q(cr, 0.95),
            'steps_mean': r['steps'].double().mean().item(), 'fallback_mean': r['fallback'].double().mean().item(),
            'fired_mean': r['fired'].double().mean().item() if 'fired' in r else 0.0,
            'fired_frac_paths': (r['fired'] > 0).double().mean().item() if 'fired' in r else 0.0,
            'runtime': r['runtime']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', action='append', default=[], metavar='TAG=PATH')
    ap.add_argument('--softmin', action='append', default=[], type=float, help='synthetic soft-min width tau')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--h', type=float, default=0.02)
    ap.add_argument('--max_steps', type=int, default=600)
    ap.add_argument('--band', type=float, default=0.9)
    ap.add_argument('--kappa', type=float, default=0.5)
    ap.add_argument('--probe', type=float, default=0.3, help='stage-3 probe distance (scaled twist units)')
    ap.add_argument('--modes', default='gradient,ridge_sample,ridge_sharpen')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    fields = []
    for tau in a.softmin:
        fields.append(('softmin_tau{:g}'.format(tau), ridge.SoftMinFreeField(tau, 1.0, 1.0), 1.0, 1.0, 1.0,
                       torch.float64))
    for spec in a.ckpt:
        tag, path = spec.split('=', 1)
        fn, _ = mt.load_function(path, device=dev)
        fields.append((tag, fn.network, fn.s_t, fn.s_r, fn.cfg['box'], fn.dtype))
    allres = {'args': vars(a), 'fields': {}}
    for tag, model, s_t, s_r, box, dtype in fields:
        sampler = PoseSampler(seed=a.seed, device=dev, dtype=dtype, box=box)
        res = {'bins': {}}
        families = [(nm, lo, hi, False) for nm, lo, hi in BINS] + [('pure_rot_170-180', 170, 180, True)]
        for name, lo, hi, pure in families:
            tx, Rx, ty, Ry = make_pairs(sampler, a.n, lo, hi, pure)
            Ts = oracle.time_star(tx.double(), Rx.double(), ty.double(), Ry.double(), s_t, s_r).to(dtype)
            B = {}
            for mode in a.modes.split(','):
                if mode == 'gradient':
                    r = ro.rollout(model, tx, Rx, ty, Ry, s_t, s_r, mode='gradient', h=a.h, max_steps=a.max_steps,
                                   seed=0)
                else:
                    r = rollout_ridge(model, tx, Rx, ty, Ry, s_t, s_r, mode, a.h, a.max_steps, a.band, a.kappa,
                                      a.probe, seed=0)
                B[mode] = summarize(r, Ts)
                print('{:14s} {:16s} {:14s} succ {:.3f} cost p50 {:.4f} p95 {:.4f} steps {:.0f} fb {:.2f} fired {:.2f} ({:.0f}s)'.format(
                    tag, name, mode, B[mode]['success'], B[mode]['cost_ratio_p50'], B[mode]['cost_ratio_p95'],
                    B[mode]['steps_mean'], B[mode]['fallback_mean'], B[mode]['fired_mean'], B[mode]['runtime']), flush=True)
            res['bins'][name] = B
        allres['fields'][tag] = res
        with open(os.path.join(a.out, 'ridge_plan.json'), 'w') as fh:
            json.dump(allres, fh, indent=1)


if __name__ == '__main__':
    main()
