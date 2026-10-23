"""Per-case planning of the stretchable link (two drones and their line of sight) with a trained travel-time field.

The forward MPPI of models/graph_homotopy/scripts/plan_eval_homotopy.py (50 samples, horizon 5, step 0.015,
temperature 50, cost 10 T(first) + T(last), chart convergence ball 0.01 freezes a case, 600 iterations: the
crossing sets have paths of 2-3 normalized units) in the 7-d chart
x = (t, rotvec / 2 pi, L) of models/drone_link: perturbations in all 7 coordinates, the length clamped to
[L_min, L_max] in every rollout and every step (the limits on how far the link stretches and shrinks).

A plan succeeds (primary criterion) if it converges to within 0.01 in sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2) and the
exact clearance of the link box (geometry.clearance) is >= 0 at every waypoint and at --substeps poses of the
chart-linear motion between consecutive waypoints, up to the converged waypoint: both drones stay clear of the scene,
see each other at every moment and keep an allowed distance. Per case it also records the contact fraction (share of
the checked poses whose link is clear), whether a drone itself hit the scene, the drones' path lengths and the
lengths the link went through.

    python models/drone_link/scripts/plan_eval_link.py --checkpoint Experiments/drone_link/forest_ntf_s0/latest.pt \
        --dataPath testing_data/3dshape/link_forest_std --out results/drone_link/plan/forest_ntf_s0/link_forest_std.json
"""
import argparse
import importlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--metric-root', default='/export')
ap.add_argument('--dataPath', required=True, help='test set directory (sampled_points.npy, meta.json)')
ap.add_argument('--out', required=True)
ap.add_argument('--steps', type=int, default=600)
ap.add_argument('--batch', type=int, default=125)
ap.add_argument('--cases', type=int, default=0)
ap.add_argument('--substeps', type=int, default=4)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--temp', type=float, default=50.0)
ap.add_argument('--samples', type=int, default=50, help='MPPI rollouts per step (planner ablation; 50 everywhere else)')
ap.add_argument('--horizon', type=int, default=5)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--save-paths', action='store_true')
args = ap.parse_args()

torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DIM = 7
TWO_PI = 2 * math.pi
STEP, TEMP, SAMPLE_NUM, HORIZON = 0.015, args.temp, args.samples, args.horizon
dev = args.device

# this repository's package by path: the metric bundle owns the name 'models' in this process
_spec = importlib.util.spec_from_file_location('drone_link_geometry', os.path.join(REPO, 'models', 'drone_link', 'geometry.py'))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
scene = G.load_scene(meta, REPO)
link = G.load_link(meta)
arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float32)
n = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
arr = arr[:n]

sys.modules.setdefault('pickle5', pickle)
sys.modules.pop('models', None)
sys.path.insert(0, args.metric_root)
md = importlib.import_module('models.metric.model_train_metric')
m = md.Model('/tmp/link_eval', args.dataPath + '/', DIM, [0.0] * DIM, device=dev)
m.load(args.checkpoint)
m.network.eval()
fn = m.function


def cost(X, Xg):
    with torch.no_grad():
        out = []
        Z = torch.cat([X, Xg], 1)
        for i in range(0, Z.shape[0], 65536):
            out.append(fn.TravelTimes(Z[i:i + 65536]).reshape(-1))
        return torch.cat(out)


def mppi(X, Xg, steps):
    B = X.shape[0]
    X = X.clone()
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    conv3 = torch.full((B,), -1, dtype=torch.long, device=dev)
    rec = [X.clone()]
    for it in range(steps):
        dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), device=dev)
        nrm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(nrm, min=STEP) / STEP)
        Xs = X[:, None, None, :] + torch.cumsum(dP, dim=2)
        Xs[..., 6] = Xs[..., 6].clamp(link.L_min, link.L_max)
        ends = Xs[:, :, [0, -1], :].reshape(-1, DIM)
        c = cost(ends, Xg.repeat_interleave(SAMPLE_NUM * 2, 0)).reshape(B, SAMPLE_NUM, 2)
        c = 10 * c[:, :, 0] + c[:, :, 1]
        w = torch.softmax(-TEMP * c, dim=1)
        step = torch.bmm(w.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        X = X + step * (~done).unsqueeze(1)
        X[:, 6] = X[:, 6].clamp(link.L_min, link.L_max)
        rec.append(X.clone())
        dis = torch.norm(X - Xg, dim=1)
        newly = (dis < 0.01) & (~done)
        conv[newly] = it
        done = done | (dis < 0.01)
        conv3[(G.pose_dist(X, Xg) < 0.01) & (conv3 < 0)] = it
        if bool(done.all()):
            break
    return dict(full=torch.stack(rec, 1), iters=conv.cpu().numpy(), succ=done.cpu().numpy(), iters3=conv3.cpu().numpy())


def motion_clearance(full):
    """Exact clearance and scene clearance (B, K, substeps) of the chart-linear motion between consecutive waypoints
    of full (B, K + 1, 7), and the drones' point clearance (B, K, substeps, 2)."""
    B, K1, _ = full.shape
    u = torch.arange(1, args.substeps + 1, device=dev, dtype=full.dtype) / args.substeps
    a, b = full[:, :-1, None, :], full[:, 1:, None, :]
    Xi = (a + (b - a) * u[None, None, :, None]).reshape(-1, DIM)
    t, R, L = G.from_chart(Xi)
    c, cs, _ = G.clearance(scene, link, t, R, L, parts=True)
    pa, pb = link.ends(t, R, L)
    dr = torch.stack([scene.point_clearance(pa), scene.point_clearance(pb)], -1)
    sh = (B, K1 - 1, args.substeps)
    return c.reshape(sh), cs.reshape(sh), dr.reshape(*sh, 2)


def geo_turns(P):
    R = G.so3_exp(TWO_PI * torch.as_tensor(P[:, 3:6], dtype=torch.float64))
    tr = (R[1:] * R[:-1]).sum((1, 2))
    return float(torch.arccos(((tr - 1) / 2).clamp(-1, 1)).sum() / TWO_PI)


def drone_paths(P):
    t, R, L = G.from_chart(torch.as_tensor(P, dtype=torch.float64))
    pa, pb = link.ends(t, R, L)
    return pa.numpy(), pb.numpy()


Xall = torch.as_tensor(arr, device=dev)
c_start = G.chart_clearance(scene, link, Xall[:, :DIM]).cpu().numpy()
c_goal = G.chart_clearance(scene, link, Xall[:, DIM:]).cpu().numpy()
records = []
saved = {'case': [], 'path': [], 'clear': [], 'iters3': []}
t_plan = t_check = 0.0
t_all = time.time()
for s in range(0, n, args.batch):
    idx = np.arange(s, min(s + args.batch, n))
    X0 = Xall[idx, :DIM]
    Xg = Xall[idx, DIM:]
    ta = time.time()
    with torch.no_grad():
        tau0 = cost(X0, Xg).cpu().numpy()
        r = mppi(X0, Xg, args.steps)
    if 'cuda' in dev:
        torch.cuda.synchronize(dev)
    t_plan += time.time() - ta
    tb = time.time()
    with torch.no_grad():
        cl, cs, dr = (v.cpu().numpy() for v in motion_clearance(r['full']))
    full = r['full'].cpu().numpy()
    for b in range(len(idx)):
        i = int(idx[b])
        k3 = int(r['iters3'][b])
        kc = int(r['iters'][b]) if r['succ'][b] else -1
        seg = slice(0, k3 + 1) if k3 >= 0 else slice(None)
        segc = slice(0, kc + 1) if kc >= 0 else slice(None)
        c3, s3, d3 = cl[b, seg], cs[b, seg], dr[b, seg]
        end = k3 + 2 if k3 >= 0 else full.shape[1]
        P = full[b, :end]
        pa, pb = drone_paths(P)
        a0, b0 = drone_paths(arr[i:i + 1, :DIM])
        a1, b1 = drone_paths(arr[i:i + 1, DIM:])
        rec = {
            'case': i, 'converged': bool(r['succ'][b]), 'collision': bool((cl[b, segc] < 0).any()),
            'success': bool(r['succ'][b]) and not bool((cl[b, segc] < 0).any()),
            'iters': int(r['iters'][b]) if r['succ'][b] else None,
            'converged_se3': k3 >= 0, 'iters_se3': k3 if k3 >= 0 else None,
            'collision_se3': bool((c3 < 0).any()), 'success_se3': (k3 >= 0) and not bool((c3 < 0).any()),
            'min_clearance': float(c3.min()), 'contact_frac': float((s3 >= 0).mean()),
            'drone_hit': bool((d3 < 0).any()), 'los_blocked_frac': float(((s3 < 0) & (d3 >= 0).all(-1)).mean()),
            'trans_len_a': float(np.linalg.norm(np.diff(pa, axis=0), axis=1).sum()),
            'trans_len_b': float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()),
            'straight_a': float(np.linalg.norm(a1 - a0)), 'straight_b': float(np.linalg.norm(b1 - b0)),
            'rot_len_geo': geo_turns(P), 'L_min_used': float(P[:, 6].min()), 'L_max_used': float(P[:, 6].max()),
            'L_travel': float(np.abs(np.diff(P[:, 6])).sum()), 'L0': float(arr[i, 6]), 'L1': float(arr[i, 13]),
            'final_dist_se3': float(G.pose_dist(torch.as_tensor(full[b, -1:]), torch.as_tensor(arr[i:i + 1, DIM:]))[0]),
            'start_clear': float(c_start[i]), 'goal_clear': float(c_goal[i]), 'tau_start': float(tau0[b]),
        }
        records.append(rec)
        if args.save_paths:
            fp = np.full((args.steps + 1, DIM), np.nan, np.float32)
            fp[:full.shape[1]] = full[b]
            cp = np.full((args.steps, args.substeps), np.nan, np.float32)
            cp[:cl.shape[1]] = cl[b]
            saved['case'].append(i)
            saved['path'].append(fp)
            saved['clear'].append(cp)
            saved['iters3'].append(k3)
    t_check += time.time() - tb
    print('{} / {} cases, success_se3 so far {:.3f} ({:.0f}s)'.format(
        len(records), n, np.mean([q['success_se3'] for q in records]), time.time() - t_all), flush=True)

R_ = records


def rate(k):
    return float(np.mean([q[k] for q in R_])) if R_ else None


def med(k, cond='success_se3'):
    v = [q[k] for q in R_ if q[cond] and q[k] is not None]
    return float(np.median(v)) if v else None


def med_ratio(num, den, cond='success_se3'):
    v = [q[num] / max(q[den], 1e-3) for q in R_ if q[cond]]
    return float(np.median(v)) if v else None


summary = {
    'dataPath': args.dataPath, 'checkpoint': args.checkpoint, 'kind': 'metric_ntfields_7d', 'steps': args.steps,
    'substeps': args.substeps, 'seed': args.seed, 'step': STEP, 'temperature': TEMP, 'samples': SAMPLE_NUM,
    'horizon': HORIZON, 'n_cases': n, 'link': meta['link'],
    'success_rate_se3': rate('success_se3'), 'collision_rate_se3': rate('collision_se3'),
    'nonconv_rate_se3': float(np.mean([not q['converged_se3'] for q in R_])),
    'success_rate': rate('success'), 'collision_rate': rate('collision'),
    'nonconv_rate': float(np.mean([not q['converged'] for q in R_])),
    'mean_contact_frac': rate('contact_frac'), 'full_contact_rate': float(np.mean([q['contact_frac'] >= 1.0 for q in R_])),
    'drone_hit_rate': rate('drone_hit'), 'mean_los_blocked_frac': rate('los_blocked_frac'),
    'median_iters_se3': med('iters_se3'), 'median_min_clearance_success': med('min_clearance'),
    'median_trans_ratio_a_success': med_ratio('trans_len_a', 'straight_a'),
    'median_trans_ratio_b_success': med_ratio('trans_len_b', 'straight_b'),
    'median_rot_len_success': med('rot_len_geo'), 'median_L_travel_success': med('L_travel'),
    'frac_success_using_short_link': float(np.mean([q['L_min_used'] < min(q['L0'], q['L1']) - 0.02
                                                    for q in R_ if q['success_se3']])) if any(q['success_se3'] for q in R_) else None,
    'seconds': {'plan_gpu': t_plan, 'check': t_check, 'total': time.time() - t_all},
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
tmp = args.out + '.tmp'
with open(tmp, 'w') as fh:
    json.dump({'summary': summary, 'records': R_}, fh, indent=1, default=float)
os.replace(tmp, args.out)
if args.save_paths:
    np.savez_compressed(args.out[:-5] + '_paths.npz', case=np.array(saved['case']), path=np.stack(saved['path']),
                        clear=np.stack(saved['clear']), iters3=np.array(saved['iters3']), goal=arr[:, DIM:])
print(json.dumps(summary))
