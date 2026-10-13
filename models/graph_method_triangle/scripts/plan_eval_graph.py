"""Per-case planning evaluation of travel-time fields with the Shape-NTFields forward MPPI.

A port of Lie-Shape-NTFields/analysis/plan_eval.py (same MPPI as
evaluate_training_3d_batched.py at its defaults: 50 samples, horizon 5, step
0.015, temperature 50, momentum 0, 200 iterations, chart convergence ball 0.01;
collision = an environment surface point inside the placed shape at any
waypoint) that can score

  * our checkpoints (--kind graph: models/graph_method_triangle, loaded by path),
  * our native-pipeline checkpoints (--kind graph_native: models/graph_native),
  * the Lie-Shape-NTFields packages (--kind metric / lie / ...: analysis.common.load_model).

Per case it records everything plan_eval.py records (convergence, collision,
iterations, path translation / rotation lengths in the chart and on SO(3), their
ratios to the straight start -> goal values, minimum clearance, tau at the start,
tau monotonicity) and in addition

  * crosses_seam     whether the shortest start -> goal rotation crosses the
                     rotation-vector seam |rotvec| = pi (chart discontinuity),
  * SE(3) convergence: first iteration at which the normalized product-geodesic
                     distance sqrt(|dp|^2 + (theta / 2 pi)^2) drops below 0.01,
                     and success / collision / path lengths under that criterion
                     (the MPPI dynamics are the official chart ones; only the
                     bookkeeping differs, so one run yields both criteria),
  * the final chart / SE(3) distance and geodesic angle to the goal.

Run inside the container with the Lie repo mounted read-only at /lieshape (its
dataprocessing / analysis / models packages and test sets are used) and this
repo at /workspace, cwd = /lieshape:

    python /workspace/models/graph_method_triangle/scripts/plan_eval_graph.py --kind graph \
        --checkpoint /workspace/Experiments/.../latest.pt --dataPath testing_data/3dshape/Tshape3d_env1 \
        --out /workspace/results/graph_method_cusp/plan/<tag>/Tshape3d_env1.json --device cuda:0
"""
import os
import sys
import json
import math
import time
import argparse
import importlib
import importlib.util

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--dataPath', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--cases', type=int, default=0, help='0 = all')
ap.add_argument('--batch', type=int, default=250)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--lie-root', default='/lieshape')
ap.add_argument('--graph-root', default='/workspace')
ap.add_argument('--temp', type=float, default=50.0)
ap.add_argument('--wrap', action='store_true', help='canonicalize the rotation vector (|rotvec| <= pi, same pose) after each executed MPPI step; the official planner never does')
args = ap.parse_args()

sys.path.insert(0, args.lie_root)
os.chdir(args.lie_root)
torch.set_num_threads(4)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

import igl  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from dataprocessing.preprocess_obj import load_obj, sample_surface_points, _rotvec_to_matrix_np  # noqa: E402

DIM = 6
TWO_PI = 2.0 * math.pi
STEP, TEMP, SAMPLE_NUM, HORIZON = 0.015, args.temp, 50, 5
_SDF_SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER
CLEAR_CAP = 0.03


# ----------------------------------------------------------------------------- our packages (by path)
def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GM_DIR = os.path.join(args.graph_root, 'models', 'graph_method_triangle')
gm = _load_pkg('graph_method_triangle', GM_DIR)
se3 = importlib.import_module('graph_method_triangle.geometry.se3')


def so3_exp_t(r):
    return se3.so3_exp(r)


def so3_log_t(R):
    return se3.so3_log(R)


def geodesic_angle(r0, r1):
    """Geodesic angle (radians) between rotations given as normalized rotvecs (turns)."""
    R0 = se3.so3_exp(TWO_PI * torch.as_tensor(np.asarray(r0), dtype=torch.float64))
    R1 = se3.so3_exp(TWO_PI * torch.as_tensor(np.asarray(r1), dtype=torch.float64))
    return se3.rotation_angle(R0.transpose(-1, -2) @ R1)


def crosses_seam(r0, r1, k=65):
    """Does the shortest rotation from r0 to r1 (turns) cross the chart seam |rotvec| = pi?
    The geodesic R0 exp(lambda log(R0^T R1)) is sampled at k points and written as
    canonical rotation vectors; a jump of more than a quarter turn between
    consecutive samples is the seam flip (+pi n -> -pi n)."""
    r0 = torch.as_tensor(np.asarray(r0), dtype=torch.float64)
    r1 = torch.as_tensor(np.asarray(r1), dtype=torch.float64)
    R0 = se3.so3_exp(TWO_PI * r0)
    R1 = se3.so3_exp(TWO_PI * r1)
    w = se3.so3_log(R0.transpose(-1, -2) @ R1)                      # (B, 3) radians, |w| <= pi
    lam = torch.linspace(0, 1, k, dtype=torch.float64)
    Rl = R0[:, None] @ se3.so3_exp(lam[None, :, None] * w[:, None, :])   # (B, k, 3, 3)
    rl = se3.so3_log(Rl.reshape(-1, 3, 3)).reshape(r0.shape[0], k, 3) / TWO_PI
    jump = (rl[:, 1:] - rl[:, :-1]).norm(dim=-1).max(dim=1).values
    return (jump > 0.25).numpy()


# ----------------------------------------------------------------------------- scene
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))


def _resolve(path):
    return path if os.path.exists(path) else os.path.join('./datasets/3dshape', os.path.basename(path))


env_scale = float(meta['env_scale'])
env_center = np.asarray(meta['env_center'], dtype=np.float64)
V_sh, F_sh, _ = load_obj(_resolve(meta['shape_obj']))
shape_V = np.ascontiguousarray((V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / env_scale * float(meta['shape_scale']),
                               dtype=np.float64)
shape_F = np.ascontiguousarray(F_sh, dtype=np.int64)
shape_radius = float(np.linalg.norm(shape_V, axis=1).max())
V_env, F_env, _ = load_obj(_resolve(meta['env_obj']))
env_pts = np.ascontiguousarray(sample_surface_points((V_env - env_center) / env_scale, F_env, 50000), dtype=np.float64)
env_tree = cKDTree(env_pts)


def path_clearance(P):
    """Per-waypoint clearance of the placed shape along a path P (T, 6), capped at CLEAR_CAP;
    negative = collision (the evaluator's check_trajectory_collision decision)."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), CLEAR_CAP)
    for k, sel in enumerate(env_tree.query_ball_point(P[:, :3], r=shape_radius + CLEAR_CAP)):
        if not sel:
            continue
        R = _rotvec_to_matrix_np(P[k, 3:6] * TWO_PI)
        S = igl.signed_distance(np.ascontiguousarray((env_pts[sel] - P[k, :3]) @ R),
                                shape_V, shape_F, _SDF_SIGN)[0]
        out[k] = min(float(S.min()), CLEAR_CAP)
    return out


import multiprocessing as mp  # noqa: E402
POOL = mp.get_context('fork').Pool(args.workers) if args.workers > 1 else None


# ----------------------------------------------------------------------------- model
class _GraphQueries:
    def __init__(self, network, device, coords):
        self.network, self.device, self.coords = network, device, coords
        self.c2f = importlib.import_module('graph_method_triangle.model_function_shape').coords_to_features

    def TravelTimes(self, Xp, chunk=65536):
        Xp = Xp.to(self.device, torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, Xp.shape[0], chunk):
                x = Xp[i:i + chunk]
                if self.coords:
                    out.append(self.network.value(x[:, :6], x[:, 6:]))
                else:
                    out.append(self.network.value(self.c2f(x[:, :6]), self.c2f(x[:, 6:])))
        return torch.cat(out)


if args.kind == 'graph':
    mt = importlib.import_module('graph_method_triangle.model_train_graph')
    f, ck = mt.load_function(args.checkpoint, device=args.device)
    fn = _GraphQueries(f.network, args.device, bool(ck['config'].get('coords', False)))
elif args.kind == 'graph_native':
    gn = _load_pkg('graph_native', os.path.join(args.graph_root, 'models', 'graph_native'))
    md = importlib.import_module('graph_native.model_train_metric')
    m = md.Model('/tmp/analysis_model', 'datasets/3dshape/Tshape3d_env1/', DIM, [0.0] * DIM, device=args.device)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
else:
    from analysis.common import load_model  # noqa: E402
    model, fn = load_model(args.kind, args.checkpoint, args.device)


# ----------------------------------------------------------------------------- planner (verbatim MPPI)
def pose_dist_chart(XP):
    return torch.norm(XP[:, DIM:2 * DIM] - XP[:, 0:DIM], dim=1)


def wrap_turns(r):
    """Canonical rotation vector (|theta| <= pi) of r (B, 3) stored in turns."""
    n = r.norm(dim=-1, keepdim=True)
    nw = torch.remainder(n + 0.5, 1.0) - 0.5
    return torch.where(n > 0.5, r / n.clamp(min=1e-12) * nw, r)


def pose_dist_se3(XP):
    dp = XP[:, DIM:DIM + 3] - XP[:, 0:3]
    q = []
    for rv in (XP[:, 3:6], XP[:, DIM + 3:DIM + 6]):
        phi = TWO_PI * rv
        th = phi.norm(dim=1, keepdim=True)
        q.append(torch.cat([torch.cos(th / 2), 0.5 * torch.sinc(th / TWO_PI) * phi], 1))
    c = (q[0] * q[1]).sum(1).abs().clamp(max=1.0)
    theta = 2.0 * torch.arccos(c)
    return torch.sqrt((dp ** 2).sum(1) + (theta / TWO_PI) ** 2)


def MPPI_batched(XP, steps):
    """Verbatim logic of evaluate_training_3d_batched.MPPI_batched at its defaults; episodes
    freeze on chart convergence (official rule). SE(3) convergence is only recorded."""
    B = XP.shape[0]
    dev = XP.device
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    conv_se3 = torch.full((B,), -1, dtype=torch.long, device=dev)
    recorded = [XP[:, 0:DIM].clone()]
    for it in range(steps):
        XP_tmp = XP.clone()[:, None, None, :].repeat(1, SAMPLE_NUM, HORIZON, 1)
        r = torch.full((B,), STEP, device=dev)[:, None, None, None]
        dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), dtype=torch.float32, device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), dtype=torch.float32, device=dev)
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
        endpoints = XP_tmp[:, :, [0, -1], :]
        with torch.no_grad():
            cost = fn.TravelTimes(endpoints.reshape(-1, DIM * 2)).reshape(B, SAMPLE_NUM, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-TEMP * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * (~done).unsqueeze(1)
        if args.wrap:
            XP[:, 3:6] = wrap_turns(XP[:, 3:6])
        dis = pose_dist_chart(XP)
        recorded.append(XP[:, 0:DIM].clone())
        newly = (dis < 0.01) & (~done)
        conv_step[newly] = it
        done = done | (dis < 0.01)
        dse3 = pose_dist_se3(XP)
        newly_se3 = (dse3 < 0.01) & (conv_se3 < 0)
        conv_se3[newly_se3] = it
        if bool(done.all()):
            break
    success = done.cpu().numpy()
    cs = conv_step.cpu().numpy()
    cs3 = conv_se3.cpu().numpy()
    goal = XP[:, DIM:2 * DIM]
    full = torch.stack(recorded, 1).cpu().numpy()          # (B, n_rec, 6)
    return full, cs.tolist(), success.tolist(), cs3.tolist(), goal.cpu().numpy()


# ----------------------------------------------------------------------------- run
arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float32)
n = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
arr = arr[:n]

t0 = time.time()
ends = [arr[i].reshape(2, 6) for i in range(n)]
end_clear = POOL.map(path_clearance, ends) if POOL is not None else [path_clearance(e) for e in ends]
ok_idx = [i for i in range(n) if (end_clear[i] >= 0.0).all()]
t_filter = time.time() - t0
cross_all = crosses_seam(arr[:, 3:6], arr[:, 9:12])

records = []
t_plan = t_check = 0.0
for s in range(0, len(ok_idx), args.batch):
    idx = ok_idx[s:s + args.batch]
    XP = torch.tensor(arr[idx]).to(args.device)
    ta = time.time()
    full, iters, succ, iters3, goal = MPPI_batched(XP.clone(), args.steps)
    if 'cuda' in args.device:
        torch.cuda.synchronize(args.device)
    t_plan += time.time() - ta
    tb = time.time()
    # official path: up to chart convergence (+ goal appended), full path otherwise
    paths = []
    for b in range(len(idx)):
        end = int(iters[b]) + 2 if succ[b] else full.shape[1]
        paths.append(np.concatenate([full[b, :end], goal[b:b + 1]], 0))
    # for the SE(3) criterion the clearance is needed along the full recorded path
    fulls = [full[b] for b in range(len(idx))]
    clears_full = POOL.map(path_clearance, fulls) if POOL is not None else [path_clearance(P) for P in fulls]
    goal_clear = POOL.map(path_clearance, [goal[b:b + 1] for b in range(len(idx))]) if POOL is not None \
        else [path_clearance(goal[b:b + 1]) for b in range(len(idx))]
    for b, i in enumerate(idx):
        P = paths[b]
        cf = clears_full[b]
        clear = np.concatenate([cf[:len(P) - 1], goal_clear[b]])
        coll = bool((clear < 0).any())
        start, goal_b = arr[i, :6], arr[i, 6:]
        dP = np.diff(P, axis=0)
        trans_len = float(np.linalg.norm(dP[:, :3], axis=1).sum())
        rot_chart = float(np.linalg.norm(dP[:, 3:], axis=1).sum())
        rot_geo = float(geodesic_angle(P[:-1, 3:], P[1:, 3:]).sum() / TWO_PI)
        # SE(3) criterion: path up to the first SE(3)-converged waypoint
        k3 = int(iters3[b])
        conv3 = k3 >= 0
        if conv3:
            P3 = full[b, :k3 + 2]
            c3 = cf[:k3 + 2]
            coll3 = bool((c3 < 0).any())
            d3 = np.diff(P3, axis=0)
            trans_len3 = float(np.linalg.norm(d3[:, :3], axis=1).sum())
            rot_geo3 = float(geodesic_angle(P3[:-1, 3:], P3[1:, 3:]).sum() / TWO_PI)
        else:
            coll3 = bool((cf < 0).any())
            trans_len3 = rot_geo3 = None
        last = full[b, -1]
        fin = np.concatenate([last, goal_b])[None]
        fin_t = torch.tensor(fin)
        records.append({
            'case': int(i), 'converged': bool(succ[b]), 'collision': coll,
            'success': bool(succ[b]) and not coll, 'iters': int(iters[b]) if succ[b] else None,
            'n_waypoints': int(len(P)),
            'trans_len': trans_len, 'rot_len_chart': rot_chart, 'rot_len_geo': rot_geo,
            'straight_trans': float(np.linalg.norm(goal_b[:3] - start[:3])),
            'straight_rot_chart': float(np.linalg.norm(goal_b[3:] - start[3:])),
            'straight_rot_geo': float(geodesic_angle(start[None, 3:], goal_b[None, 3:])[0] / TWO_PI),
            'min_clearance': float(clear.min()), 'frac_clear_lt_0p01': float((clear < 0.01).mean()),
            'theta_start_deg': float(np.linalg.norm(start[3:]) * 360), 'theta_goal_deg': float(np.linalg.norm(goal_b[3:]) * 360),
            'crosses_seam': bool(cross_all[i]),
            'converged_se3': conv3, 'iters_se3': k3 if conv3 else None, 'collision_se3': coll3,
            'success_se3': conv3 and not coll3,
            'trans_len_se3': trans_len3, 'rot_len_geo_se3': rot_geo3,
            'final_dist_chart': float(pose_dist_chart(fin_t)[0]), 'final_dist_se3': float(pose_dist_se3(fin_t)[0]),
            'final_geo_deg': float(geodesic_angle(last[None, 3:], goal_b[None, 3:])[0] * 180 / math.pi),
            'final_theta_deg': float(np.linalg.norm(last[3:]) * 360),
        })
        records[-1]['_path'] = P
    t_check += time.time() - tb

# tau along each (official) path for monotonicity, one batched pass
allw, owner = [], []
for k, rec in enumerate(records):
    P = rec.pop('_path')
    g = np.repeat(P[-1:], len(P), 0)
    allw.append(np.concatenate([P, g], 1))
    owner.append(np.full(len(P), k))
W = torch.tensor(np.concatenate(allw), dtype=torch.float32) if allw else torch.zeros(0, 12)
owner = np.concatenate(owner) if owner else np.zeros(0, int)
taus = []
with torch.no_grad():
    for s in range(0, len(W), 20000):
        taus.append(fn.TravelTimes(W[s:s + 20000].to(args.device)).detach().cpu().numpy())
taus = np.concatenate(taus) if taus else np.zeros(0)
for k, rec in enumerate(records):
    tk = taus[owner == k]
    d = np.diff(tk[:-1]) if len(tk) > 2 else np.zeros(0)
    rec['tau_start'] = float(tk[0])
    rec['tau_frac_increase'] = float((d > 1e-4).mean()) if len(d) else 0.0

R = records
nok = len(R)


def rate(key):
    return float(np.mean([r[key] for r in R])) if R else None


def med_succ(key, skey='success'):
    v = [r[key] for r in R if r[skey] and r[key] is not None]
    return float(np.median(v)) if v else None


def med_ratio(num, den, skey='success'):
    v = [r[num] / max(r[den], 1e-3) for r in R if r[skey] and r[num] is not None]
    return float(np.median(v)) if v else None


summary = {
    'dataPath': args.dataPath, 'kind': args.kind, 'checkpoint': args.checkpoint,
    'group_conv': False, 'planner': 'mppi' + ('_wrap' if args.wrap else ''), 'wrap': bool(args.wrap), 'seed': args.seed, 'batch': args.batch, 'steps': args.steps,
    'step': STEP, 'temperature': TEMP, 'samples': SAMPLE_NUM, 'horizon': HORIZON,
    'distance': 'chart Euclidean (official); SE(3) criterion recorded alongside',
    'n_cases': n, 'n_endpoints_ok': nok, 'n_crossing': int(sum(r['crosses_seam'] for r in R)),
    'success_rate': rate('success'), 'collision_rate': rate('collision'),
    'nonconv_rate': float(np.mean([not r['converged'] for r in R])) if R else None,
    'success_rate_se3': rate('success_se3'), 'collision_rate_se3': rate('collision_se3'),
    'nonconv_rate_se3': float(np.mean([not r['converged_se3'] for r in R])) if R else None,
    'median_iters_success': med_succ('iters'), 'median_iters_success_se3': med_succ('iters_se3', 'success_se3'),
    'median_trans_ratio_success': med_ratio('trans_len', 'straight_trans'),
    'median_rotgeo_ratio_success': med_ratio('rot_len_geo', 'straight_rot_geo'),
    'median_trans_ratio_success_se3': med_ratio('trans_len_se3', 'straight_trans', 'success_se3'),
    'median_rotgeo_ratio_success_se3': med_ratio('rot_len_geo_se3', 'straight_rot_geo', 'success_se3'),
    'median_rot_len_geo_success': med_succ('rot_len_geo'), 'median_trans_len_success': med_succ('trans_len'),
    'median_min_clearance_success': med_succ('min_clearance'),
    'median_tau_frac_increase': float(np.median([r['tau_frac_increase'] for r in R])) if R else None,
    'seconds': {'filter': t_filter, 'plan_gpu': t_plan, 'collision_cpu': t_check},
}
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w') as fh:
    json.dump({'summary': summary, 'records': R}, fh, indent=1, default=float)
if POOL is not None:
    POOL.close()
print(json.dumps(summary))
