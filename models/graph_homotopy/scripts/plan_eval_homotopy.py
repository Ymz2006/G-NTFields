"""Per-case planning evaluation of the homotopy head: base field vs planning on the double cover.

Same forward MPPI, test sets, bookkeeping and collision check as
models/graph_method_triangle/scripts/plan_eval_graph.py (50 samples, horizon 5, step 0.015, temperature 50,
200 iterations, chart convergence ball 0.01 freezes an episode; SE(3) convergence recorded alongside; collision
= an environment surface point inside the placed shape at any waypoint). Three planners:

  base    the official planner on the base field T(x, y) = min_sigma T~(x~, sigma y~) (identical to
          plan_eval_graph.py --kind graph_native);
  cover   the goal is lifted to the sheet the field prefers at the start, sigma* = argmin_sigma T~(x~_s, sigma y~),
          and the MPPI costs are the cover field T~(x~, sigma* y~) along the way: the pose of every sample is
          lifted continuously from the unwrapped rotation-vector chart (q = exp(pi r)), so the planner descends
          one sheet of the cover and never meets the ridge;
  dual    both lifted goals are planned (two cover runs, sigma = +1 and -1); per case the plan that converged
          (SE(3) criterion, then chart, then fewer iterations) is kept, ties / neither -> sigma*.

Run inside the container with the Lie repo mounted read-only at /lieshape (test sets, dataprocessing) and this
repo at /workspace:

    python /workspace/models/graph_homotopy/scripts/plan_eval_homotopy.py --planner cover \
        --checkpoint /workspace/Experiments/graph_homotopy/<run>/latest.pt --dataPath testing_data/3dshape/Tshape3d_env1 \
        --out /workspace/results/graph_homotopy/plan/<run>/Tshape3d_env1__cover.json --device cuda:0
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
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--dataPath', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--planner', default='base', choices=['base', 'cover', 'dual'])
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--cases', type=int, default=0, help='0 = all')
ap.add_argument('--batch', type=int, default=125)
ap.add_argument('--steps', type=int, default=200)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--lie-root', default='/lieshape')
ap.add_argument('--graph-root', default='/workspace')
ap.add_argument('--temp', type=float, default=50.0)
ap.add_argument('--kind', default='graph', choices=['graph', 'metric'],
                help='graph: this repository\'s heads (graph_native loader); metric: the original metric NTFields '
                     'baseline (models/metric of the self-contained bundle at --metric-root)')
ap.add_argument('--metric-root', default='/export')
ap.add_argument('--save-paths', action='store_true',
                help='also save the chosen plan of every case with its per-waypoint clearance (<out>_chosen.npz)')
ap.add_argument('--camera', default='',
                help='camera json of dataprocessing/make_camera3d.py (tip, base, bird mesh): score every plan by '
                     'the visibility of the bird from the camera (ray casting against the scene mesh) and the '
                     'collision of the bird alone, next to the collision of the whole body')
ap.add_argument('--gpu-clearance', action='store_true',
                help='exact mesh clearance on the GPU (models/graph_homotopy/scripts/gpu_clearance.py: batched '
                     'point-triangle distance + generalized winding number) instead of libigl in a process pool; '
                     'same signs, distances exact where libigl\'s fast-winding path is off by ~1e-4')
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


def _load_pkg(name, pkg_dir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, '__init__.py'),
                                                  submodule_search_locations=[pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gm = _load_pkg('graph_method_triangle', os.path.join(args.graph_root, 'models', 'graph_method_triangle'))
se3 = importlib.import_module('graph_method_triangle.geometry.se3')
hd = importlib.import_module('graph_method_triangle.homotopy_decoder')


def geodesic_angle(r0, r1):
    R0 = se3.so3_exp(TWO_PI * torch.as_tensor(np.asarray(r0), dtype=torch.float64))
    R1 = se3.so3_exp(TWO_PI * torch.as_tensor(np.asarray(r1), dtype=torch.float64))
    return se3.rotation_angle(R0.transpose(-1, -2) @ R1)


def crosses_seam(r0, r1, k=65):
    r0 = torch.as_tensor(np.asarray(r0), dtype=torch.float64)
    r1 = torch.as_tensor(np.asarray(r1), dtype=torch.float64)
    R0 = se3.so3_exp(TWO_PI * r0)
    R1 = se3.so3_exp(TWO_PI * r1)
    w = se3.so3_log(R0.transpose(-1, -2) @ R1)
    lam = torch.linspace(0, 1, k, dtype=torch.float64)
    Rl = R0[:, None] @ se3.so3_exp(lam[None, :, None] * w[:, None, :])
    rl = se3.so3_log(Rl.reshape(-1, 3, 3)).reshape(r0.shape[0], k, 3) / TWO_PI
    jump = (rl[:, 1:] - rl[:, :-1]).norm(dim=-1).max(dim=1).values
    return (jump > 0.25).numpy()


# ----------------------------------------------------------------------------- scene
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))


def _resolve(path):
    cands = [path, os.path.join(args.dataPath, os.path.basename(path)),
             os.path.join(args.lie_root, 'datasets/3dshape', os.path.basename(path)),
             os.path.join(args.graph_root, 'datasets/3dshape/obj', os.path.basename(path))]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(path)


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


# ----------------------------------------------------------------------------- camera (visibility of the bird)
def _rays_blocked(o, d, tri, chunk=4096, eps=1e-6):
    """segments o -> o + d (N, 3) against triangles (T, 3, 3): True where a segment meets a triangle at a
    parameter in (eps, 1 - eps) (Moeller-Trumbore, batched)."""
    a, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    T = len(tri)
    out = []
    for i in range(0, len(o), chunk):
        oc, dc = o[i:i + chunk], d[i:i + chunk]
        n = len(oc)
        p = torch.cross(dc[:, None, :].expand(n, T, 3), e2[None].expand(n, T, 3), dim=-1)
        det = (e1[None] * p).sum(-1)
        ok = det.abs() > 1e-12
        inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
        tv = oc[:, None, :] - a[None]
        u = (tv * p).sum(-1) * inv
        q = torch.cross(tv, e1[None].expand(n, T, 3), dim=-1)
        v = (dc[:, None, :] * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps) & (t < 1 - eps)
        out.append(hit.any(1))
    return torch.cat(out)


CAM = None
if args.camera:
    cj = json.load(open(args.camera))
    cdir = os.path.dirname(os.path.abspath(args.camera))
    centre = np.asarray(cj['frame_centre'], dtype=np.float64)
    assert np.allclose(centre, 0.5 * (V_sh.min(0) + V_sh.max(0)), atol=1e-6), 'the camera frame is not the set\'s shape frame'
    Vb_, Fb_, _ = load_obj(os.path.join(cdir, cj['bird_obj']))
    bird_V = np.ascontiguousarray((Vb_ - centre) / env_scale * float(meta['shape_scale']), dtype=np.float64)
    bird_F = np.ascontiguousarray(Fb_, dtype=np.int64)
    cam_apex = (np.asarray(cj['apex'], dtype=np.float64) - centre) / env_scale
    _st = np.random.get_state()
    np.random.seed(12345)
    S_all = sample_surface_points(bird_V, bird_F, 3000)
    np.random.set_state(_st)
    # samples the camera sees in the body frame (not hidden by the bird itself)
    _o = torch.tensor(np.repeat(cam_apex[None], len(S_all), 0), dtype=torch.float64)
    _d = torch.tensor(S_all, dtype=torch.float64) - _o
    _self = _rays_blocked(_o, _d, torch.tensor(bird_V[bird_F], dtype=torch.float64), eps=1e-4).numpy()
    cam_samples = S_all[~_self][:256]
    env_tri = ((V_env - env_center) / env_scale)[F_env]
    CAM = dict(apex=cam_apex, samples=cam_samples, n_tri=len(env_tri), fov_deg=cj.get('fov_deg'))
    print('camera: tip {} (body frame), {} of {} bird samples in view, {} scene triangles'.format(
        np.round(cam_apex, 4).tolist(), len(cam_samples), len(S_all), len(env_tri)), flush=True)


def path_clearance_bird(P):
    """clearance of the bird alone along a path (the camera's view volume and the camera itself not counted)."""
    P = np.asarray(P, dtype=np.float64)
    out = np.full(len(P), CLEAR_CAP)
    for k, sel in enumerate(env_tree.query_ball_point(P[:, :3], r=shape_radius + CLEAR_CAP)):
        if not sel:
            continue
        R = _rotvec_to_matrix_np(P[k, 3:6] * TWO_PI)
        S = igl.signed_distance(np.ascontiguousarray((env_pts[sel] - P[k, :3]) @ R), bird_V, bird_F, _SDF_SIGN)[0]
        out[k] = min(float(S.min()), CLEAR_CAP)
    return out


import multiprocessing as mp  # noqa: E402
POOL = mp.get_context('fork').Pool(args.workers) if args.workers > 1 and not args.gpu_clearance else None
GPU_CLEAR = GPU_CLEAR_BIRD = None
if args.gpu_clearance:
    sys.path.insert(0, os.path.join(args.graph_root, 'models', 'graph_homotopy', 'scripts'))
    from gpu_clearance import GpuClearance  # noqa: E402
    GPU_CLEAR = GpuClearance(env_pts, shape_V, shape_F, shape_radius + CLEAR_CAP, CLEAR_CAP, args.device)
    if CAM is not None:
        GPU_CLEAR_BIRD = GpuClearance(env_pts, bird_V, bird_F, shape_radius + CLEAR_CAP, CLEAR_CAP, args.device)


def clear_paths(L):
    """capped clearance of the whole body along each path of L."""
    if GPU_CLEAR is not None:
        return GPU_CLEAR.paths(L)
    return POOL.map(path_clearance, L) if POOL is not None else [path_clearance(P) for P in L]


def clear_paths_bird(L):
    if GPU_CLEAR_BIRD is not None:
        return GPU_CLEAR_BIRD.paths(L)
    return POOL.map(path_clearance_bird, L) if POOL is not None else [path_clearance_bird(P) for P in L]


# ----------------------------------------------------------------------------- model
if args.kind == 'graph':
    gn = _load_pkg('graph_native', os.path.join(args.graph_root, 'models', 'graph_native'))
    md = importlib.import_module('graph_native.model_train_metric')
    m = md.Model('/tmp/analysis_model', 'datasets/3dshape/data/Tshape3d_env1/', DIM, [0.0] * DIM, device=args.device)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
    head = m.network.net
else:
    # the original metric NTFields (Lie-Shape-NTFields metric_baseline = the bundle's models/metric, alpha 1.0)
    import pickle
    sys.modules.setdefault('pickle5', pickle)
    sys.modules.pop('models', None)
    sys.path.insert(0, args.metric_root)
    md = importlib.import_module('models.metric.model_train_metric')
    m = md.Model('/tmp/metric_eval', 'datasets/3dshape/data/Tshape3d_env1/', DIM, [0.0] * DIM, device=args.device)
    m.load(args.checkpoint)
    m.network.eval()
    fn = m.function
    head = None
assert args.planner == 'base' or hasattr(head, 'value_cover'), 'cover planning needs the homotopy head'

if CAM is not None:
    CAM_T = dict(apex=torch.tensor(CAM['apex'], dtype=torch.float32, device=args.device),
                 samples=torch.tensor(CAM['samples'], dtype=torch.float32, device=args.device),
                 tri=torch.tensor(env_tri, dtype=torch.float32, device=args.device))


def view_fraction(P):
    """(M,) fraction of the bird's visible samples whose line of sight from the camera is free of the scene, per
    pose of the chart path P (M, 6)."""
    Pt = torch.as_tensor(np.asarray(P, dtype=np.float64), dtype=torch.float32, device=args.device)
    R = se3.so3_exp(TWO_PI * Pt[:, 3:6])
    t = Pt[:, :3]
    o = t + torch.einsum('mij,j->mi', R, CAM_T['apex'])
    sw = t[:, None, :] + torch.einsum('mij,nj->mni', R, CAM_T['samples'])
    M, N = sw.shape[0], sw.shape[1]
    d = (sw - o[:, None, :]).reshape(-1, 3)
    blocked = _rays_blocked(o[:, None, :].expand(M, N, 3).reshape(-1, 3), d, CAM_T['tri']).reshape(M, N)
    return (~blocked).float().mean(1).cpu().numpy()


def base_cost(X, G):
    """T(x, goal) on the base field; X (N, 6) chart states, G (N, 6) chart goals."""
    return fn.TravelTimes(torch.cat([X, G], 1))


def cover_cost(X, tg, qg, chunk=65536):
    """T~(x~, goal~): X (N, 6) chart states lifted continuously, (tg, qg) the lifted goals (N, 3), (N, 4)."""
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            x = X[i:i + chunk]
            out.append(head.value_cover(x[:, :3], hd.quat_exp_turns(x[:, 3:6]), tg[i:i + chunk], qg[i:i + chunk]))
    return torch.cat(out)


# ----------------------------------------------------------------------------- planner (verbatim MPPI)
def pose_dist_chart(XP):
    return torch.norm(XP[:, DIM:2 * DIM] - XP[:, 0:DIM], dim=1)


def pose_dist_se3(XP):
    dp = XP[:, DIM:DIM + 3] - XP[:, 0:3]
    q0, q1 = hd.quat_exp_turns(XP[:, 3:6]), hd.quat_exp_turns(XP[:, DIM + 3:DIM + 6])
    c = (q0 * q1).sum(1).abs().clamp(max=1.0)
    theta = 2.0 * torch.arccos(c)
    return torch.sqrt((dp ** 2).sum(1) + (theta / TWO_PI) ** 2)


def MPPI_batched(XP, steps, sigma=None):
    """Official MPPI; sigma None = base field, else (B,) relative sign of the lifted goal (cover field)."""
    B = XP.shape[0]
    dev = XP.device
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    conv_step = torch.full((B,), steps - 1, dtype=torch.long, device=dev)
    conv_se3 = torch.full((B,), -1, dtype=torch.long, device=dev)
    recorded = [XP[:, 0:DIM].clone()]
    if sigma is not None:
        tg = XP[:, DIM:DIM + 3]
        qg = sigma[:, None] * hd.quat_exp_turns(XP[:, DIM + 3:2 * DIM])
    for it in range(steps):
        XP_tmp = XP.clone()[:, None, None, :].repeat(1, SAMPLE_NUM, HORIZON, 1)
        r = torch.full((B,), STEP, device=dev)[:, None, None, None]
        dP = STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, 1, DIM), dtype=torch.float32, device=dev) \
            + STEP * torch.normal(0, 1, size=(B, SAMPLE_NUM, HORIZON, DIM), dtype=torch.float32, device=dev)
        dP_norm = torch.norm(dP, dim=3, keepdim=True)
        dP = dP / (torch.clamp(dP_norm, min=r) / r)
        XP_tmp[..., 0:DIM] = XP_tmp[..., 0:DIM] + torch.cumsum(dP, dim=2)
        endpoints = XP_tmp[:, :, [0, -1], :].reshape(-1, DIM * 2)
        with torch.no_grad():
            if sigma is None:
                cost = base_cost(endpoints[:, :DIM], endpoints[:, DIM:])
            else:
                rep = SAMPLE_NUM * 2
                cost = cover_cost(endpoints[:, :DIM], tg.repeat_interleave(rep, 0), qg.repeat_interleave(rep, 0))
            cost = cost.reshape(B, SAMPLE_NUM, 2)
        cost = 10 * cost[:, :, 0] + cost[:, :, 1]
        weight = torch.softmax(-TEMP * cost, dim=1)
        step_prior = torch.bmm(weight.unsqueeze(1), dP[:, :, 0, :]).squeeze(1)
        XP[:, 0:DIM] = XP[:, 0:DIM] + step_prior * (~done).unsqueeze(1)
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
    full = torch.stack(recorded, 1).cpu().numpy()
    return dict(full=full, iters=conv_step.cpu().numpy(), succ=done.cpu().numpy(), iters3=conv_se3.cpu().numpy(),
                goal=XP[:, DIM:2 * DIM].cpu().numpy())


# ----------------------------------------------------------------------------- run
arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float32)
n = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
arr = arr[:n]

t0 = time.time()
ends = [arr[i].reshape(2, 6) for i in range(n)]
end_clear = clear_paths(ends)
ok_idx = [i for i in range(n) if (end_clear[i] >= 0.0).all()]
t_filter = time.time() - t0
cross_all = crosses_seam(arr[:, 3:6], arr[:, 9:12])

records = []
t_plan = t_check = 0.0
chosen_paths = {'case': [], 'path': [], 'clear': [], 'iters3': []}
saved = {'case': [], 'full_plus': [], 'full_minus': [], 'iters3_plus': [], 'iters3_minus': [], 'succ_plus': [], 'succ_minus': []}
for s in range(0, len(ok_idx), args.batch):
    idx = ok_idx[s:s + args.batch]
    XP = torch.tensor(arr[idx]).to(args.device)
    ta = time.time()
    # the field's own choice of homotopy class at the start (relative to the exp lifts of start and goal)
    with torch.no_grad():
        if hasattr(head, 'value_cover'):
            tg, qg = XP[:, DIM:DIM + 3], hd.quat_exp_turns(XP[:, DIM + 3:2 * DIM])
            tp = cover_cost(XP[:, :DIM], tg, qg)
            tm = cover_cost(XP[:, :DIM], tg, -qg)
            sig_star = torch.where(tp <= tm, 1.0, -1.0)
            gap = ((torch.maximum(tp, tm) - torch.minimum(tp, tm)) / torch.minimum(tp, tm).clamp_min(1e-6)).cpu().numpy()
            sig_star_np = sig_star.cpu().numpy()
            t_start = torch.minimum(tp, tm).cpu().numpy()
        else:
            sig_star_np = np.ones(len(idx)); gap = np.full(len(idx), np.nan); t_start = np.full(len(idx), np.nan)
    if args.planner == 'base':
        res = MPPI_batched(XP.clone(), args.steps)
        chosen = np.zeros(len(idx), int)
        runs = [res]
    elif args.planner == 'cover':
        res = MPPI_batched(XP.clone(), args.steps, sigma=sig_star.to(XP.device))
        chosen = np.zeros(len(idx), int)
        runs = [res]
    else:
        ones = torch.ones(len(idx), device=XP.device)
        runs = [MPPI_batched(XP.clone(), args.steps, sigma=ones), MPPI_batched(XP.clone(), args.steps, sigma=-ones)]
        chosen = np.zeros(len(idx), int)
        for b in range(len(idx)):
            key = []
            for r_ in runs:
                key.append((int(r_['iters3'][b] >= 0), int(r_['succ'][b]),
                            -int(r_['iters3'][b] if r_['iters3'][b] >= 0 else args.steps)))
            if key[0] == key[1]:
                chosen[b] = 0 if sig_star_np[b] > 0 else 1
            else:
                chosen[b] = int(np.argmax([k[0] * 1e6 + k[1] * 1e3 + k[2] for k in key]))
    if 'cuda' in args.device:
        torch.cuda.synchronize(args.device)
    t_plan += time.time() - ta
    if args.planner == 'dual':
        saved['case'] += [int(i) for i in idx]
        for k_, r_ in (('plus', runs[0]), ('minus', runs[1])):
            fp = np.full((len(idx), args.steps + 1, DIM), np.nan, np.float32)
            fp[:, :r_['full'].shape[1]] = r_['full']
            saved['full_' + k_].append(fp)
            saved['iters3_' + k_] += [int(v) for v in r_['iters3']]
            saved['succ_' + k_] += [bool(v) for v in r_['succ']]
    tb = time.time()
    full = np.stack([runs[chosen[b]]['full'][b] for b in range(len(idx))])
    iters = [runs[chosen[b]]['iters'][b] for b in range(len(idx))]
    succ = [runs[chosen[b]]['succ'][b] for b in range(len(idx))]
    iters3 = [runs[chosen[b]]['iters3'][b] for b in range(len(idx))]
    goal = runs[0]['goal']
    paths = []
    for b in range(len(idx)):
        end = int(iters[b]) + 2 if succ[b] else full.shape[1]
        paths.append(np.concatenate([full[b, :end], goal[b:b + 1]], 0))
    fulls = [full[b] for b in range(len(idx))]
    clears_full = clear_paths(fulls)
    if CAM is not None:
        clears_bird = clear_paths_bird(fulls)
        views_full = [view_fraction(P) for P in fulls]
    if args.save_paths:
        for b, i in enumerate(idx):
            fp = np.full((args.steps + 1, DIM), np.nan, np.float32)
            fp[:len(fulls[b])] = fulls[b]
            cp = np.full(args.steps + 1, np.nan, np.float32)
            cp[:len(clears_full[b])] = clears_full[b]
            chosen_paths['case'].append(int(i))
            chosen_paths['path'].append(fp)
            chosen_paths['clear'].append(cp)
            chosen_paths['iters3'].append(int(iters3[b]))
            if CAM is not None:
                vp = np.full(args.steps + 1, np.nan, np.float32)
                vp[:len(views_full[b])] = views_full[b]
                bp = np.full(args.steps + 1, np.nan, np.float32)
                bp[:len(clears_bird[b])] = clears_bird[b]
                chosen_paths.setdefault('view', []).append(vp)
                chosen_paths.setdefault('bird_clear', []).append(bp)
    sheet_ok = None
    if args.planner == 'dual':
        # SE(3) success of each sheet's own plan (collision-checked up to its first SE(3)-converged waypoint)
        sheet_ok = []
        for r_ in runs:
            segs = []
            for b in range(len(idx)):
                k3 = int(r_['iters3'][b])
                segs.append(r_['full'][b, :k3 + 2] if k3 >= 0 else r_['full'][b])
            cl = clear_paths(segs)
            sheet_ok.append([bool(r_['iters3'][b] >= 0) and not bool((cl[b] < 0).any()) for b in range(len(idx))])
    goal_clear = clear_paths([goal[b:b + 1] for b in range(len(idx))])
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
        rec = {
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
            'sigma_star': float(sig_star_np[b]), 'sheet_gap': float(gap[b]), 'tau_start_min': float(t_start[b]),
            'planner': args.planner,
        }
        if args.planner == 'dual':
            rec['chosen_sigma'] = 1.0 if chosen[b] == 0 else -1.0
            rec['conv_se3_plus'] = bool(runs[0]['iters3'][b] >= 0)
            rec['conv_se3_minus'] = bool(runs[1]['iters3'][b] >= 0)
            rec['success_se3_plus'] = sheet_ok[0][b]
            rec['success_se3_minus'] = sheet_ok[1][b]
            rec['success_se3_cover'] = sheet_ok[0][b] if sig_star_np[b] > 0 else sheet_ok[1][b]
            rec['success_se3_either'] = sheet_ok[0][b] or sheet_ok[1][b]
            # alternative planner-side rule: converged, then the shorter translational path (SE(3) segment)
            lens, convs = [], []
            for r_ in runs:
                k3_ = int(r_['iters3'][b])
                seg = r_['full'][b, :k3_ + 2] if k3_ >= 0 else r_['full'][b]
                lens.append(float(np.linalg.norm(np.diff(seg[:, :3], axis=0), axis=1).sum()))
                convs.append(k3_ >= 0)
            if convs[0] != convs[1]:
                pick = 0 if convs[0] else 1
            elif lens[0] != lens[1]:
                pick = int(np.argmin(lens))
            else:
                pick = 0 if sig_star_np[b] > 0 else 1
            rec['success_se3_shorter'] = sheet_ok[pick][b]
        if CAM is not None:
            # the camera scores on the SE(3) segment the collision test uses (the whole plan if it never converged)
            seg = slice(0, k3 + 2) if conv3 else slice(None)
            v3, cb3 = views_full[b][seg], clears_bird[b][seg]
            rec['bird_collision_se3'] = bool((cb3 < 0).any())
            rec['bird_min_clearance_se3'] = float(cb3.min())
            rec['view_min'] = float(v3.min())
            rec['view_mean'] = float(v3.mean())
            rec['view_full_frac'] = float((v3 >= 0.99).mean())
            rec['cam_view_ok'] = bool(v3.min() >= 0.99)
            rec['follow_success_se3'] = bool(conv3 and not rec['bird_collision_se3'] and rec['cam_view_ok'])
        rec['_path'] = P
        records.append(rec)
    t_check += time.time() - tb

# tau along each (official) path for monotonicity, one batched pass (base field)
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
    'dataPath': args.dataPath, 'kind': 'graph_native' if args.kind == 'graph' else 'metric_baseline', 'checkpoint': args.checkpoint, 'planner': args.planner,
    'seed': args.seed, 'batch': args.batch, 'steps': args.steps,
    'step': STEP, 'temperature': TEMP, 'samples': SAMPLE_NUM, 'horizon': HORIZON,
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
    'frac_sigma_minus': float(np.mean([r['sigma_star'] < 0 for r in R])) if R else None,
    'median_sheet_gap': float(np.nanmedian([r['sheet_gap'] for r in R])) if R else None,
    'seconds': {'filter': t_filter, 'plan_gpu': t_plan, 'collision_cpu': t_check, 'clearance_on': 'gpu' if args.gpu_clearance else 'cpu'},
}
if args.planner == 'dual':
    summary['frac_chosen_minus'] = float(np.mean([r['chosen_sigma'] < 0 for r in R]))
    summary['conv_se3_plus'] = float(np.mean([r['conv_se3_plus'] for r in R]))
    summary['conv_se3_minus'] = float(np.mean([r['conv_se3_minus'] for r in R]))
    summary['conv_se3_either'] = float(np.mean([r['conv_se3_plus'] or r['conv_se3_minus'] for r in R]))
    for k in ('success_se3_plus', 'success_se3_minus', 'success_se3_cover', 'success_se3_either', 'success_se3_shorter'):
        summary[k] = float(np.mean([r[k] for r in R]))
if CAM is not None:
    for k in ('bird_collision_se3', 'cam_view_ok', 'follow_success_se3'):
        summary[k + '_rate'] = rate(k)
    summary['mean_view_mean'] = float(np.mean([r['view_mean'] for r in R])) if R else None
    summary['median_view_min'] = float(np.median([r['view_min'] for r in R])) if R else None
    summary['camera'] = dict(file=args.camera, samples=int(len(CAM['samples'])), scene_triangles=int(CAM['n_tri']),
                             fov_deg=CAM['fov_deg'])
os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w') as fh:
    json.dump({'summary': summary, 'records': R}, fh, indent=1, default=float)
if args.planner == 'dual' and saved['case']:
    np.savez_compressed(args.out[:-5] + '_paths.npz', case=np.array(saved['case']),
                        full_plus=np.concatenate(saved['full_plus']), full_minus=np.concatenate(saved['full_minus']),
                        iters3_plus=np.array(saved['iters3_plus']), iters3_minus=np.array(saved['iters3_minus']),
                        succ_plus=np.array(saved['succ_plus']), succ_minus=np.array(saved['succ_minus']),
                        sheet_ok_plus=np.array([r['success_se3_plus'] for r in R]), sheet_ok_minus=np.array([r['success_se3_minus'] for r in R]),
                        sigma_star=np.array([r['sigma_star'] for r in R]))
if args.save_paths:
    np.savez_compressed(args.out[:-5] + '_chosen.npz', case=np.array(chosen_paths['case']),
                        path=np.stack(chosen_paths['path']), clear=np.stack(chosen_paths['clear']),
                        iters3=np.array(chosen_paths['iters3']), goal=arr[np.array(chosen_paths['case']), 6:],
                        env_pts=env_pts.astype(np.float32), shape_V=shape_V.astype(np.float32), shape_F=shape_F,
                        **({'view': np.stack(chosen_paths['view']), 'bird_clear': np.stack(chosen_paths['bird_clear'])}
                           if CAM is not None else {}))
if POOL is not None:
    POOL.close()
print(json.dumps(summary))
