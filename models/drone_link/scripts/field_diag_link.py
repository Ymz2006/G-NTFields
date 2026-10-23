"""Is the learned travel time descending along a valid route? T(x, goal) of a trained 7-d metric NTFields along each
case's certified path (the test set's roadmap / RRT path, slerp-interpolated, 40 poses per segment).

Per case: T at the start, the certified path length, the share of steps along the path where T goes up (> 1e-4), and
the correlation of T with the remaining path length. Split by the evaluation's outcome (solved / stalled), written to
results/drone_link/diagnostics/field_<run>_<set>.json. Also records the training cell's pair statistics (drone a's
move, chart distance) that bound how far the field was supervised.

    python models/drone_link/scripts/field_diag_link.py --run forest_ntf_s0 --set link_forest_cross   (cwd /workspace)
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--set', required=True)
ap.add_argument('--cases', type=int, default=150, help='per outcome')
ap.add_argument('--metric-root', default='/export')
ap.add_argument('--device', default='cuda:0')
args = ap.parse_args()

spec = importlib.util.spec_from_file_location('drone_link_geometry', os.path.join(REPO, 'models/drone_link/geometry.py'))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
sys.modules.setdefault('pickle5', pickle)
sys.modules.pop('models', None)
sys.path.insert(0, args.metric_root)
md = importlib.import_module('models.metric.model_train_metric')
TWO_PI = 2 * math.pi
dev = args.device
sets = os.path.join(REPO, 'testing_data', '3dshape', args.set)
m = md.Model('/tmp/field_diag', sets + '/', 7, [0.0] * 7, device=dev)
m.load(os.path.join(REPO, 'Experiments', 'drone_link', args.run, 'latest.pt'))
m.network.eval()
ev = json.load(open(os.path.join(REPO, 'results', 'drone_link', 'plan', args.run, args.set + '.json')))
X = np.load(os.path.join(sets, 'sampled_points.npy'))
cz = np.load(os.path.join(sets, 'cases.npz'))
off = np.concatenate([[0], np.cumsum(cz['rrt_len'])])

out = {'solved': [], 'stalled': []}
for r in ev['records']:
    kind = 'solved' if r['success_se3'] else ('stalled' if not r['converged_se3'] else None)
    if kind is None or len(out[kind]) >= args.cases:
        continue
    c = r['case']
    W = cz['rrt_flat'][off[c]:off[c + 1]]
    pts = []
    for i in range(len(W) - 1):
        u = np.linspace(0, 1, 40, endpoint=False)
        q = G.slerp(W[i, 3:7], W[i + 1, 3:7], u)
        pts.append(np.concatenate([W[i, :3] + u[:, None] * (W[i + 1, :3] - W[i, :3]), G.rotvec_from_quat(q) / TWO_PI,
                                   (W[i, 7] + u * (W[i + 1, 7] - W[i, 7]))[:, None]], 1))
    pts.append(np.concatenate([W[-1:, :3], G.rotvec_from_quat(W[-1:, 3:7]) / TWO_PI, W[-1:, 7:8]], 1))
    P = torch.as_tensor(np.concatenate(pts), dtype=torch.float32, device=dev)
    goal = torch.as_tensor(X[c, 7:], dtype=torch.float32, device=dev)[None].expand(len(P), -1)
    with torch.no_grad():
        tv = m.function.TravelTimes(torch.cat([P, goal], 1)).reshape(-1).cpu().numpy()
    seg = G.pose_dist(P[:-1].double().cpu(), P[1:].double().cpu()).numpy()
    rem = np.concatenate([[0], np.cumsum(seg)])
    rem = rem[-1] - rem
    out[kind].append(dict(case=int(c), T_start=float(tv[0]), cert_len=float(rem[0]),
                          frac_up=float((np.diff(tv) > 1e-4).mean()), corr=float(np.corrcoef(tv, rem)[0, 1])))

cell = os.path.join(REPO, 'datasets', '3dshape', 'data', 'link_' + args.set.split('_')[1])
D = np.load(os.path.join(cell, 'sampled_points.npy'), mmap_mode='r')[:200000]
move = np.linalg.norm(D[:, 7:10] - D[:, :3], axis=1)
chart = np.linalg.norm(D[:, 7:] - D[:, :7], axis=1)
summ = {k: {key: float(np.median([x[key] for x in v])) for key in ('T_start', 'cert_len', 'frac_up', 'corr')} | {'n': len(v)}
        for k, v in out.items() if v}
summ['training_pairs'] = {'drone_a_move_median': float(np.median(move)), 'drone_a_move_gt_0.7': float((move > 0.7).mean()),
                          'chart_distance_median': float(np.median(chart)), 'cell': os.path.basename(cell)}
dst = os.path.join(REPO, 'results', 'drone_link', 'diagnostics', 'field_{}_{}.json'.format(args.run, args.set))
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump({'summary': summ, 'cases': out}, open(dst, 'w'), indent=1)
print(json.dumps(summ))
