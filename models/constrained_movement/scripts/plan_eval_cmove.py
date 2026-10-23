"""Per-case planning of constrained movement: the camera pyramid on M = [0, L] x SO(3).

models/path_constrained/scripts/plan_eval_path.py with the pyramid's exact clearance: the forward MPPI of the
SE(3) evaluations (50 samples, horizon 5, step 0.015, temperature 50, cost 10 T(first) + T(last), chart
convergence ball 0.01 freezes a case, SE(3) convergence recorded) on M. The planner state is y = (t, r): arc length
of the followed shape along its path and the unwrapped rotation vector of the pyramid in turns; perturbations are
sampled in the 4-d tangent space (dt, dr); the cost is the model's time from the chart pose ((t - L / 2), 0, 0, r)
to the goal's.

    planner base   the field T(x, goal) (metric NTFields; the homotopy head as well)
            dual   homotopy head only: both lifted goals planned on the cover, the converged plan kept
    mode    free     t anywhere in [0, L]
            forward  every sample has dt >= 0 and t <= t_goal: the followed shape only moves forward

A plan succeeds (SE(3) criterion, primary) if it converges to within 0.01 in sqrt(dt^2 + (theta / 2 pi)^2) and the
exact pyramid clearance (geometry.pyramid_clearance) is >= 0 at every waypoint and at --substeps points of the
chart-linear motion between consecutive waypoints, up to the converged waypoint: the camera never touches the scene
and its view of the shape is never blocked.

    python models/constrained_movement/scripts/plan_eval_cmove.py --kind metric --mode forward \
        --checkpoint /workspace/Experiments/constrained_movement/forest_ntf_s0/latest.pt \
        --dataPath testing_data/3dshape/cm_forest_a_full --out results/constrained_movement/plan/forest_ntf_s0/cm_forest_a_full__base_forward.json
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--kind', default='metric', choices=['graph', 'metric'])
ap.add_argument('--metric-root', default='/export')
ap.add_argument('--dataPath', required=True, help='test set directory (cases.npz, meta.json, path.npz)')
ap.add_argument('--out', required=True)
ap.add_argument('--planner', default='base', choices=['base', 'dual'])
ap.add_argument('--mode', default='free', choices=['free', 'forward'])
ap.add_argument('--chart', default='arclength', choices=['arclength', 'world'], help='chart the model was trained in')
ap.add_argument('--steps', type=int, default=400)
ap.add_argument('--batch', type=int, default=125)
ap.add_argument('--cases', type=int, default=0)
ap.add_argument('--substeps', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--temp', type=float, default=50.0)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--save-paths', action='store_true')
args = ap.parse_args()

torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DIM = 6
TWO_PI = 2 * math.pi
STEP, TEMP, SAMPLE_NUM, HORIZON = 0.015, args.temp, 50, 5
dev = args.device


def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# this package and the homotopy decoder by path, so that 'models' stays free for the metric bundle
_load_pkg('constrained_movement', os.path.join(REPO, 'models', 'constrained_movement'))
G = importlib.import_module('constrained_movement.geometry')
_load_pkg('graph_method_triangle', os.path.join(REPO, 'models', 'graph_method_triangle'))
hd = importlib.import_module('graph_method_triangle.homotopy_decoder')

# ----------------------------------------------------------------------------- scene, path, cases
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
con = meta['constraint']
path = G.Path.load(os.path.join(args.dataPath, con['path_file']))
scene = G.scene_from_meta(meta, REPO)
pyr = G.Pyramid(con['pyramid']['W'], con['pyramid']['H'], con['pyramid']['D'])
L = path.L
cz = np.load(os.path.join(args.dataPath, 'cases.npz'))
n = len(cz['l0']) if args.cases <= 0 else min(args.cases, len(cz['l0']))
Y0_all = np.concatenate([cz['l0'][:n, None], cz['rv0'][:n] / TWO_PI], 1).astype(np.float32)
Yg_all = np.concatenate([cz['l1'][:n, None], cz['rv1'][:n] / TWO_PI], 1).astype(np.float32)

# ----------------------------------------------------------------------------- model
if args.kind == 'graph':
    sys.path.insert(0, REPO)
    gn = _load_pkg('graph_native', os.path.join(REPO, 'models', 'graph_native'))
    md = importlib.import_module('graph_native.model_train_metric')
    m = md.Model('/tmp/cm_eval', args.dataPath, DIM, [0.0] * DIM, device=dev)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
    head = m.network.net
else:
    import pickle
    sys.modules.setdefault('pickle5', pickle)
    sys.modules.pop('models', None)
    sys.path.insert(0, args.metric_root)
    md = importlib.import_module('models.metric.model_train_metric')
    m = md.Model('/tmp/cm_eval', args.dataPath + '/', DIM, [0.0] * DIM, device=dev)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
    head = None
assert args.planner == 'base' or hasattr(head, 'value_cover'), 'the dual planner needs the homotopy head'


def chart(Y):
    """(N, 4) planner states (l, r) -> (N, 6) chart poses of the model."""
    return torch.cat([path.chart_t(Y[:, 0], args.chart), Y[:, 1:]], 1)


def base_cost(Y, Xg):
    with torch.no_grad():
        out = []
        X = torch.cat([chart(Y), Xg], 1)
        for i in range(0, X.shape[0], 65536):
            out.append(fn.TravelTimes(X[i:i + 65536]).reshape(-1))
        return torch.cat(out)


def cover_cost(Y, tg, qg):
    with torch.no_grad():
        out = []
        X = chart(Y)
        for i in range(0, X.shape[0], 65536):
            x = X[i:i + 65536]
            out.append(head.value_cover(x[:, :3], hd.quat_exp_turns(x[:, 3:6]), tg[i:i + 65536], qg[i:i + 65536]))
        return torch.cat(out)


def se3_dist(Y, Yg):
    q0, q1 = hd.quat_exp_turns(Y[:, 1:]), hd.quat_exp_turns(Yg[:, 1:])
    c = (q0 * q1).sum(1).abs().clamp(max=1.0)
    return torch.sqrt((Y[:, 0] - Yg[:, 0]) ** 2 + (2.0 * torch.arccos(c) / TWO_PI) ** 2)


def mppi(Y, Yg, steps, sigma=None):
    """Forward MPPI on M; sigma None = base field, else (B,) the lifted goal's sign (cover field)."""
    B = Y.shape[0]
    Y = Y.clone()
    Xg = chart(Yg)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    conv3 = torch.full((B,), -1, dtype=torch.long, device=dev)
    rec = [Y.clone()]
    if sigma is not None:
        tg = Xg[:, :3]
        qg = sigma[:, None] * hd.quat_exp_turns(Yg[:, 1:])
    lo = torch.zeros(B, device=dev)
    hi = Yg[:, 0].clone() if args.mode == 'forward' else torch.full((B,), L, device=dev)
    for it in range(steps):
        dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, 4), device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, 4), device=dev)
        if args.mode == 'forward':
            dP[..., 0] = dP[..., 0].clamp(min=0.0)
        nrm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(nrm, min=STEP) / STEP)
        Ys = Y[:, None, None, :] + torch.cumsum(dP, dim=2)
        Ys[..., 0] = torch.minimum(torch.maximum(Ys[..., 0], lo[:, None, None]), hi[:, None, None])
        ends = Ys[:, :, [0, -1], :].reshape(-1, 4)
        if sigma is None:
            cost = base_cost(ends, Xg.repeat_interleave(SAMPLE_NUM * 2, 0))
        else:
            rep = SAMPLE_NUM * 2
            cost = cover_cost(ends, tg.repeat_interleave(rep, 0), qg.repeat_interleave(rep, 0))
        cost = cost.reshape(B, SAMPLE_NUM, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        w = torch.softmax(-TEMP * cost, dim=1)
        step = torch.bmm(w.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        Y = Y + step * (~done).unsqueeze(1)
        Y[:, 0] = torch.minimum(torch.maximum(Y[:, 0], lo), hi)
        rec.append(Y.clone())
        dis = torch.norm(Y - Yg, dim=1)
        newly = (dis < 0.01) & (~done)
        conv[newly] = it
        done = done | (dis < 0.01)
        d3 = se3_dist(Y, Yg)
        conv3[(d3 < 0.01) & (conv3 < 0)] = it
        if bool(done.all()):
            break
    return dict(full=torch.stack(rec, 1), iters=conv.cpu().numpy(), succ=done.cpu().numpy(), iters3=conv3.cpu().numpy())


def plan_clearance(full):
    """Exact clearance (B, K, substeps) of the chart-linear motion between consecutive waypoints of full
    (B, K + 1, 4): element [b, k, j] = pose at fraction (j + 1) / substeps from waypoint k to k + 1."""
    B, K1, _ = full.shape
    u = torch.arange(1, args.substeps + 1, device=dev, dtype=full.dtype) / args.substeps
    a, b = full[:, :-1, None, :], full[:, 1:, None, :]
    Yi = (a + (b - a) * u[None, None, :, None]).reshape(-1, 4)
    t = path.pos(Yi[:, 0])
    R = G.so3_exp(TWO_PI * Yi[:, 1:])
    c = G.pyramid_clearance(scene, pyr, t, R)
    return c.reshape(B, K1 - 1, args.substeps)


def geo_turns(Y):
    """Summed geodesic rotation (turns) along waypoints (K, 4)."""
    q = hd.quat_exp_turns(torch.as_tensor(Y[:, 1:]))
    c = (q[1:] * q[:-1]).sum(1).abs().clamp(max=1.0)
    return float((2 * torch.arccos(c)).sum() / TWO_PI)


# ----------------------------------------------------------------------------- run
start_clear = G.pyramid_clearance(scene, pyr, path.pos(torch.as_tensor(Y0_all[:, 0], device=dev)),
                                G.so3_exp(TWO_PI * torch.as_tensor(Y0_all[:, 1:], device=dev))).cpu().numpy()
goal_clear = G.pyramid_clearance(scene, pyr, path.pos(torch.as_tensor(Yg_all[:, 0], device=dev)),
                               G.so3_exp(TWO_PI * torch.as_tensor(Yg_all[:, 1:], device=dev))).cpu().numpy()
records = []
saved = {'case': [], 'path': [], 'iters3': []}
t_plan = t_check = 0.0
t_all = time.time()
for s in range(0, n, args.batch):
    idx = np.arange(s, min(s + args.batch, n))
    Y0 = torch.as_tensor(Y0_all[idx], device=dev)
    Yg = torch.as_tensor(Yg_all[idx], device=dev)
    B = len(idx)
    ta = time.time()
    with torch.no_grad():
        if head is not None and hasattr(head, 'value_cover'):
            Xg = chart(Yg)
            tg, qg = Xg[:, :3], hd.quat_exp_turns(Yg[:, 1:])
            tp, tm = cover_cost(Y0, tg, qg), cover_cost(Y0, tg, -qg)
            sig = torch.where(tp <= tm, 1.0, -1.0)
            sig_np = sig.cpu().numpy()
            gap = ((torch.maximum(tp, tm) - torch.minimum(tp, tm)) / torch.minimum(tp, tm).clamp_min(1e-6)).cpu().numpy()
            tau0 = torch.minimum(tp, tm).cpu().numpy()
        else:
            sig_np = np.ones(B)
            gap = np.full(B, np.nan)
            tau0 = base_cost(Y0, chart(Yg)).cpu().numpy()
    if args.planner == 'base':
        runs = [mppi(Y0, Yg, args.steps)]
        chosen = np.zeros(B, int)
    else:
        ones = torch.ones(B, device=dev)
        runs = [mppi(Y0, Yg, args.steps, sigma=ones), mppi(Y0, Yg, args.steps, sigma=-ones)]
        chosen = np.zeros(B, int)
        for b in range(B):
            key = [(int(r['iters3'][b] >= 0), int(r['succ'][b]),
                    -int(r['iters3'][b] if r['iters3'][b] >= 0 else args.steps)) for r in runs]
            chosen[b] = (0 if sig_np[b] > 0 else 1) if key[0] == key[1] else int(np.argmax(
                [k[0] * 1e6 + k[1] * 1e3 + k[2] for k in key]))
    if 'cuda' in dev:
        torch.cuda.synchronize(dev)
    t_plan += time.time() - ta
    tb = time.time()
    clears = [plan_clearance(r['full']).cpu().numpy() for r in runs]          # (B, K, sub) each
    fulls = [r['full'].cpu().numpy() for r in runs]

    def seg_ok(ri, b, k3):
        """collision-free motion up to waypoint k3 + 1 (the first SE(3)-converged one; all if k3 < 0)."""
        c = clears[ri][b] if k3 < 0 else clears[ri][b, :k3 + 1]
        return bool((c >= 0).all()), float(c.min()), float((c < 0).mean())

    for j, b in enumerate(range(B)):
        ri = chosen[b]
        r = runs[ri]
        k3 = int(r['iters3'][b])
        kc = int(r['iters'][b]) if r['succ'][b] else -1
        F = fulls[ri][b]
        okc, _, _ = seg_ok(ri, b, kc)
        ok3, minc3, fc3 = seg_ok(ri, b, k3)
        end = k3 + 2 if k3 >= 0 else F.shape[0]
        Pp = F[:end]
        cb = clears[ri][b] if k3 < 0 else clears[ri][b, :k3 + 1]
        bad = np.argwhere(cb < 0)
        first_block = None if not len(bad) else float(
            F[bad[0][0], 0] + (F[bad[0][0] + 1, 0] - F[bad[0][0], 0]) * (bad[0][1] + 1) / args.substeps)
        dl = np.diff(Pp[:, 0])
        back = float((np.maximum.accumulate(Pp[:, 0]) - Pp[:, 0]).max())
        i = int(idx[b])
        rec = {
            'case': i, 'l0': float(Y0_all[i, 0]), 'l1': float(Yg_all[i, 0]),
            'converged': bool(r['succ'][b]), 'collision': not okc, 'success': bool(r['succ'][b]) and okc,
            'iters': int(r['iters'][b]) if r['succ'][b] else None,
            'converged_se3': k3 >= 0, 'iters_se3': k3 if k3 >= 0 else None, 'collision_se3': not ok3,
            'success_se3': (k3 >= 0) and ok3, 'min_clearance': minc3, 'frac_blocked': fc3,
            'backtrack': back, 'monotone': back <= 0.005, 'first_block_t': first_block, 'end_t': float(Pp[-1, 0]),
            'success_se3_monotone': (k3 >= 0) and ok3 and back <= 0.005,
            'trans_len': float(np.abs(dl).sum()), 'rot_len_geo': geo_turns(Pp),
            'straight_trans': float(abs(Yg_all[i, 0] - Y0_all[i, 0])),
            'straight_rot_geo': float(se3_dist(torch.as_tensor(np.array([[0.0, *Y0_all[i, 1:]]], np.float32)),
                                               torch.as_tensor(np.array([[0.0, *Yg_all[i, 1:]]], np.float32)))[0]),
            'final_dist_se3': float(se3_dist(torch.as_tensor(F[-1:]), torch.as_tensor(Yg_all[i:i + 1]))[0]),
            'start_clear': float(start_clear[i]), 'goal_clear': float(goal_clear[i]),
            'sigma_star': float(sig_np[b]), 'sheet_gap': float(gap[b]), 'tau_start': float(tau0[b]),
        }
        if args.planner == 'dual':
            rec['chosen_sigma'] = 1.0 if ri == 0 else -1.0
            for nm, rj in (('plus', 0), ('minus', 1)):
                k3j = int(runs[rj]['iters3'][b])
                rec['success_se3_' + nm] = (k3j >= 0) and seg_ok(rj, b, k3j)[0]
            rec['success_se3_either'] = rec['success_se3_plus'] or rec['success_se3_minus']
        records.append(rec)
        if args.save_paths:
            fp = np.full((args.steps + 1, 4), np.nan, np.float32)
            fp[:F.shape[0]] = F
            saved['case'].append(i)
            saved['path'].append(fp)
            saved['iters3'].append(k3)
    t_check += time.time() - tb
    print('{} / {} cases, success_se3 so far {:.3f} ({:.0f}s)'.format(
        len(records), n, np.mean([r['success_se3'] for r in records]), time.time() - t_all), flush=True)

R = records


def rate(k):
    return float(np.mean([r[k] for r in R])) if R else None


def med(k, cond='success_se3'):
    v = [r[k] for r in R if r[cond] and r[k] is not None]
    return float(np.median(v)) if v else None


summary = {
    'dataPath': args.dataPath, 'checkpoint': args.checkpoint, 'kind': args.kind, 'planner': args.planner,
    'mode': args.mode, 'chart': args.chart, 'steps': args.steps, 'substeps': args.substeps, 'seed': args.seed,
    'step': STEP, 'temperature': TEMP, 'samples': SAMPLE_NUM, 'horizon': HORIZON, 'n_cases': n,
    'success_rate_se3': rate('success_se3'), 'collision_rate_se3': rate('collision_se3'),
    'nonconv_rate_se3': float(np.mean([not r['converged_se3'] for r in R])),
    'success_rate': rate('success'), 'collision_rate': rate('collision'),
    'nonconv_rate': float(np.mean([not r['converged'] for r in R])),
    'success_rate_se3_monotone': rate('success_se3_monotone'), 'monotone_rate': rate('monotone'),
    'median_backtrack_success': med('backtrack'), 'median_iters_se3': med('iters_se3'),
    'median_trans_ratio_success': float(np.median([r['trans_len'] / max(r['straight_trans'], 1e-3)
                                                   for r in R if r['success_se3']])) if any(r['success_se3'] for r in R) else None,
    'median_rot_len_success': med('rot_len_geo'), 'median_min_clearance_success': med('min_clearance'),
    'mean_frac_blocked': rate('frac_blocked'),
    'frac_sigma_minus': float(np.mean([r['sigma_star'] < 0 for r in R])),
    'seconds': {'plan_gpu': t_plan, 'check': t_check, 'total': time.time() - t_all},
}
if args.planner == 'dual':
    for k in ('success_se3_plus', 'success_se3_minus', 'success_se3_either'):
        summary[k] = rate(k)
    summary['frac_chosen_minus'] = float(np.mean([r['chosen_sigma'] < 0 for r in R]))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
tmp = args.out + '.tmp'
with open(tmp, 'w') as fh:
    json.dump({'summary': summary, 'records': R}, fh, indent=1, default=float)
os.replace(tmp, args.out)
if args.save_paths:
    np.savez_compressed(args.out[:-5] + '_paths.npz', case=np.array(saved['case']), path=np.stack(saved['path']),
                        iters3=np.array(saved['iters3']))
print(json.dumps(summary))
