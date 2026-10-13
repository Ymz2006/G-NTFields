"""Obstacle-free planning stratified by the start -> goal relative rotation angle.

For every angle bin, N pairs are drawn (translations uniform in the box, R_x
Haar, R_y = R_x exp(theta n) with a random axis n and theta uniform in the bin;
the 'pure_rot' family keeps t_y = t_x) and planned with both SE(3) planners of
planning/rollout.py (gradient descent on the decoded time with backtracking and
sampling fallback; 128-candidate sampling). Per bin: success rate, cost ratio
(path time / T*) median / p95, steps, fallbacks, terminal errors.

    python models/graph_method_triangle/scripts/plan_by_angle_free.py --out results/graph_method_cusp/free \
        --ckpt graph_gram_s0=Experiments/graph_method_final/graph_gram_s0/best.pt ... [--n 500]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.append('.')

import torch

from models.graph_method_triangle.data.poses import PoseSampler
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3
from models.graph_method_triangle.planning import rollout
from models.graph_method_triangle import model_train_graph as mt

BINS = [('0-90', 0, 90), ('90-150', 90, 150), ('150-170', 150, 170), ('170-179', 170, 179), ('179-180', 179, 180)]


def make_pairs(sampler, n, lo, hi, pure_rot=False):
    tx, Rx = sampler.uniform_poses(n)
    ty = tx.clone() if pure_rot else sampler.translations(n)
    axis = sampler.unit_vectors(n, 3)
    th = (lo + (hi - lo) * sampler._rand(n)) * math.pi / 180.0
    Ry = Rx @ se3.so3_exp(th[:, None] * axis)
    return tx, Rx, ty, Ry


def q(x, p):
    return x.quantile(p).item() if x.numel() else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True, metavar='TAG=PATH')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--h', type=float, default=0.02)
    ap.add_argument('--max_steps', type=int, default=600)
    ap.add_argument('--n_cand', type=int, default=128)
    ap.add_argument('--modes', default='gradient,sampling')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    families = [(name, lo, hi, False) for name, lo, hi in BINS] + [('pure_rot_170-180', 170, 180, True)]
    for spec in a.ckpt:
        tag, path = spec.split('=', 1)
        out_json = os.path.join(a.out, tag + '.json')
        if os.path.exists(out_json):
            print('skip', tag)
            continue
        f, ck = mt.load_function(path, device=dev)
        model, cfg = f.network, f.cfg
        s_t, s_r = f.s_t, f.s_r
        sampler = PoseSampler(seed=a.seed, device=dev, dtype=f.dtype, box=cfg['box'])
        res = {'tag': tag, 'ckpt': path, 'arch': cfg['arch'], 'n': a.n, 'h': a.h, 'bins': {}}
        t0 = time.time()
        for name, lo, hi, pure in families:
            tx, Rx, ty, Ry = make_pairs(sampler, a.n, lo, hi, pure)
            Ts = oracle.time_star(tx.double(), Rx.double(), ty.double(), Ry.double(), s_t, s_r)
            ang = oracle.relative_angle(Rx.double(), Ry.double()) * 180 / math.pi
            B = {'angle_mean': ang.mean().item(), 'T_star_mean': Ts.mean().item()}
            for mode in a.modes.split(','):
                r = rollout.rollout(model, tx, Rx, ty, Ry, s_t, s_r, mode=mode, h=a.h, max_steps=a.max_steps,
                                    n_cand=a.n_cand, seed=0)
                succ = r['success']
                ratio = (r['cost'].double() / Ts.clamp_min(1e-12))
                rs = ratio[succ]
                B[mode] = {
                    'success': succ.double().mean().item(),
                    'cost_ratio_p50': q(rs, 0.5), 'cost_ratio_p95': q(rs, 0.95), 'cost_ratio_mean': rs.mean().item() if rs.numel() else float('nan'),
                    'cost_ratio_gt_1p05': (rs > 1.05).double().mean().item() if rs.numel() else float('nan'),
                    'cost_ratio_gt_1p2': (rs > 1.2).double().mean().item() if rs.numel() else float('nan'),
                    'steps_mean': r['steps'].double().mean().item(), 'steps_p95': q(r['steps'].double(), 0.95),
                    'fallback_frac': (r['fallback'] > 0).double().mean().item(),
                    'err_t_p50_fail': q(r['err_t'][~succ].double(), 0.5), 'err_r_deg_p50_fail': q(r['err_r'][~succ].double() * 180 / math.pi, 0.5),
                    'runtime_s': r['runtime'],
                    'cost_ratio_all': ratio.cpu().tolist(), 'success_all': succ.cpu().tolist(),
                    'steps_all': r['steps'].cpu().tolist(),
                }
                print('{:<22s} {:<16s} {:<9s} succ {:.3f} cost p50 {:.4f} p95 {:.4f} steps {:.0f} fb {:.2f}'.format(
                    tag, name, mode, B[mode]['success'], B[mode]['cost_ratio_p50'], B[mode]['cost_ratio_p95'],
                    B[mode]['steps_mean'], B[mode]['fallback_frac']), flush=True)
            res['bins'][name] = B
        res['seconds'] = time.time() - t0
        with open(out_json, 'w') as fh:
            json.dump(res, fh)


if __name__ == '__main__':
    main()
