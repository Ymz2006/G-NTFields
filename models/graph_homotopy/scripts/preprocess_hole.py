"""Training data for the wall-with-hole cells: the Lie repo's preprocess_obj.py sampler (same placement
distribution, same GPU clearance / normal / rotational-clearance evaluation, same output arrays and meta) with an
analytic collision pre-filter. These environments consist of boxes only (the room and the four pieces of the slab),
so a placement whose shape-surface sample points penetrate a box by more than --delta (0.004 normalized units) is
rejected before the expensive evaluation; everything that passes the pre-filter goes through the original
evaluate_placements, whose collision test and clearances decide as before. Without the pre-filter ~97 % of the
drawn placements are evaluated only to be rejected (the shapes have radius 0.2 in a room of half-height 0.32), and
one cell took ~2 GPU-hours per 200k pairs on the RTX 5070.

    python /workspace/models/graph_homotopy/scripts/preprocess_hole.py --env /workspace/datasets/3dshape/obj/hole90.obj \
        --shape /workspace/datasets/3dshape/obj/Tshape3d.obj --out /workspace/datasets/3dshape/data/Tshape3d_hole90 \
        --num_samples 400000 --device cuda

At the end 300 pre-filter-rejected placements are re-checked with the original test and the agreement is printed
and stored in meta.json ('prefilter').
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--env', required=True)
ap.add_argument('--shape', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--num_samples', type=int, default=400000)
ap.add_argument('--margin', type=float, default=0.05)
ap.add_argument('--offset', type=float, default=0.001)
ap.add_argument('--shape_scale', type=float, default=1.0)
ap.add_argument('--num_env_points', type=int, default=10000)
ap.add_argument('--num_radius_points', type=int, default=1000)
ap.add_argument('--radius_bins', type=int, default=10)
ap.add_argument('--batch_size', type=int, default=2000, help='placements drawn per round (before the pre-filter)')
ap.add_argument('--eval_chunk', type=int, default=120, help='placements per evaluate_placements call (GPU memory)')
ap.add_argument('--delta', type=float, default=0.004, help='penetration depth tolerated by the pre-filter')
ap.add_argument('--shape_points', type=int, default=1000)
ap.add_argument('--device', default='cuda')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--lie-root', default='/lieshape')
args = ap.parse_args()

sys.path.insert(0, args.lie_root)
os.chdir(args.lie_root)
from dataprocessing.preprocess_obj import (load_obj, sample_surface_points, tetrahedralize_shape,  # noqa: E402
                                           generate_radius_surface_points, evaluate_placements, _sample_configs,
                                           _min_center_dist, wrap_rotvec, rotvec_to_matrix, DEFAULT_TET_SWITCHES, EPS)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
dev = args.device
t_start = time.time()

# ----------------------------------------------------------------------------- scene (as preprocess_obj.main)
V_env, F_env, names_env = load_obj(args.env)
bb_min, bb_max = V_env.min(axis=0), V_env.max(axis=0)
center_env = 0.5 * (bb_min + bb_max)
scale = float((bb_max - bb_min).max())
V_env_n = (V_env - center_env) / scale
sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
env_points = sample_surface_points(V_env_n, F_env[sample_mask], args.num_env_points).astype(np.float32)
ranges = (bb_max - bb_min) / scale
half_extent = ranges * 0.5 - 0.01
V_sh, F_sh, _ = load_obj(args.shape)
shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
V_sh_local = (V_sh - shape_center) / scale * args.shape_scale
TV, TT, TF = tetrahedralize_shape(V_sh_local, F_sh, switches=DEFAULT_TET_SWITCHES)
tet_verts_local = torch.tensor(TV[TT], dtype=torch.float32).to(dev)
face_verts_local = torch.tensor(TV[TF], dtype=torch.float32).to(dev)
rad_points, rad_bins = generate_radius_surface_points(V_sh_local, F_sh, args.num_radius_points, args.radius_bins)
rad_points, rad_bins = rad_points.to(dev), rad_bins.to(dev)
env_t = torch.tensor(env_points, dtype=torch.float32).to(dev)
R_shape = float(face_verts_local.reshape(-1, 3).norm(dim=1).max())
hx, hy, hz = [float(h) for h in half_extent]
print('env {}: scale {:.3f}, half extent {}, {} env points; shape {}: {} tets, radius {:.4f}'.format(
    os.path.basename(args.env), scale, np.round(half_extent, 3).tolist(), len(env_points), os.path.basename(args.shape),
    TT.shape[0], R_shape), flush=True)

# ----------------------------------------------------------------------------- analytic boxes of the environment
groups = {}
for i, n in enumerate(names_env):
    groups.setdefault(str(n), []).append(i)
boxes_obst, room = [], None
for n, fidx in groups.items():
    vid = np.unique(F_env[fidx].reshape(-1))
    lo, hi = V_env_n[vid].min(0), V_env_n[vid].max(0)
    if 'wall' in n.lower():
        room = (lo, hi)
    else:
        boxes_obst.append((lo, hi))
assert room is not None and boxes_obst, 'expected a wall group and box obstacles'
room_lo = torch.tensor(room[0], dtype=torch.float32, device=dev) - args.delta
room_hi = torch.tensor(room[1], dtype=torch.float32, device=dev) + args.delta
obst_lo = torch.tensor(np.stack([b[0] for b in boxes_obst]), dtype=torch.float32, device=dev) + args.delta   # (nb, 3)
obst_hi = torch.tensor(np.stack([b[1] for b in boxes_obst]), dtype=torch.float32, device=dev) - args.delta
S_local = np.concatenate([V_sh_local, sample_surface_points(V_sh_local, F_sh, args.shape_points)], 0)
S_local = torch.tensor(S_local, dtype=torch.float32, device=dev)                                            # (P, 3)
print('pre-filter: room {} .. {}, {} obstacle boxes, {} shape points, delta {}'.format(
    np.round(room[0], 3).tolist(), np.round(room[1], 3).tolist(), len(boxes_obst), len(S_local), args.delta), flush=True)


def gate_reject(cfg):
    """(B, 6) raw configs on device -> (B,) bool: True = surely in collision (penetration > delta)."""
    R = rotvec_to_matrix(cfg[:, 3:6])                                       # (B, 3, 3)
    W = torch.einsum('bij,pj->bpi', R, S_local) + cfg[:, None, 0:3]         # (B, P, 3)
    out_room = ((W < room_lo) | (W > room_hi)).any(-1).any(-1)
    inb = ((W[:, :, None, :] > obst_lo) & (W[:, :, None, :] < obst_hi)).all(-1)   # (B, P, nb)
    return out_room | inb.any(-1).any(-1)


def evaluate(cfg):
    outs = [[] for _ in range(6)]
    for i in range(0, cfg.shape[0], args.eval_chunk):
        r = evaluate_placements(cfg[i:i + args.eval_chunk], tet_verts_local, face_verts_local, env_t, rad_points, rad_bins, dev)
        for k in range(6):
            outs[k].append(r[k])
    return [torch.cat(o) for o in outs]


def sample_disp(n):
    d = torch.rand(n, 6) - 0.5
    d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + EPS)
    return d * (torch.rand(n, 1) * float(np.sqrt(6.0)))


# ----------------------------------------------------------------------------- sampling loop (training mode)
N = args.num_samples
pairs = torch.zeros(N, 12); dists = torch.zeros(N, 2); angles = torch.zeros(N, 2)
normals = torch.zeros(N, 12); trans_normals = torch.zeros(N, 12); rot_normals = torch.zeros(N, 12)
count = 0
stats = dict(drawn=0, gate_pass0=0, eval0=0, band0=0, x1_drawn=0, gate_pass1=0, accepted=0)
gate_rejected_samples = []
t0 = time.time()
last = t0
while count < N:
    x0 = _sample_configs(args.batch_size, hx, hy, hz)
    stats['drawn'] += len(x0)
    m0 = _min_center_dist(x0[:, 0:3], env_t, dev)
    feas0 = (m0 - R_shape <= args.margin) & (m0 + R_shape >= args.offset)
    x0 = x0[feas0]
    if len(x0) == 0:
        continue
    rej = gate_reject(x0.to(dev)).cpu()
    if len(gate_rejected_samples) < 300 and bool(rej.any()):
        gate_rejected_samples.append(x0[rej][:300 - len(gate_rejected_samples)])
    x0 = x0[~rej]
    stats['gate_pass0'] += len(x0)
    if len(x0) == 0:
        continue
    free0, dist0, angle0, normal0, trans0, rot0 = evaluate(x0)
    stats['eval0'] += len(x0)
    keep0 = free0 & (dist0 > args.offset) & (dist0 < args.margin)
    if int(keep0.sum()) == 0:
        continue
    x0, dist0, angle0, normal0, trans0, rot0 = [t[keep0] for t in (x0, dist0, angle0, normal0, trans0, rot0)]
    stats['band0'] += len(x0)
    delta = sample_disp(len(x0))
    x1 = x0.clone()
    x1[:, 0:3] = x0[:, 0:3] + delta[:, 0:3]
    x1[:, 3:6] = wrap_rotvec(x0[:, 3:6] + delta[:, 3:6] * 2 * np.pi)
    stats['x1_drawn'] += len(x1)
    ok = (x1[:, 0].abs() <= hx) & (x1[:, 1].abs() <= hy) & (x1[:, 2].abs() <= hz)
    ok &= (_min_center_dist(x1[:, 0:3], env_t, dev) + R_shape >= args.offset)
    if int(ok.sum()) == 0:
        continue
    ok[ok.clone()] &= ~gate_reject(x1[ok].to(dev)).cpu()
    if int(ok.sum()) == 0:
        continue
    x0, dist0, angle0, normal0, trans0, rot0, x1 = [t[ok] for t in (x0, dist0, angle0, normal0, trans0, rot0, x1)]
    stats['gate_pass1'] += len(x1)
    free1, dist1, angle1, normal1, trans1, rot1 = evaluate(x1)
    keep = free1 & (dist1 > args.offset)
    nk = int(keep.sum())
    if nk == 0:
        continue
    take = min(nk, N - count)
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[:take]
    sl = slice(count, count + take)
    pairs[sl, 0:6] = x0[idx]; pairs[sl, 6:12] = x1[idx]
    dists[sl, 0] = dist0[idx]; dists[sl, 1] = dist1[idx]
    angles[sl, 0] = angle0[idx]; angles[sl, 1] = angle1[idx]
    normals[sl, 0:6] = normal0[idx]; normals[sl, 6:12] = normal1[idx]
    trans_normals[sl, 0:6] = trans0[idx]; trans_normals[sl, 6:12] = trans1[idx]
    rot_normals[sl, 0:6] = rot0[idx]; rot_normals[sl, 6:12] = rot1[idx]
    count += take
    stats['accepted'] = count
    if time.time() - last > 30:
        last = time.time()
        print('generated: {} / {}  ({:.0f} pairs/s; drawn {}, pre-filter passed {} ({:.1%}), in band {}, x1 passed {})'.format(
            count, N, count / (time.time() - t0), stats['drawn'], stats['gate_pass0'], stats['gate_pass0'] / max(stats['drawn'], 1),
            stats['band0'], stats['gate_pass1']), flush=True)
t_sample = time.time() - t0
print('Sampling done in {:.1f}s ({:.0f} pairs/s)'.format(t_sample, N / t_sample), flush=True)

# pre-filter audit: the original test on pre-filter-rejected placements
audit = None
if gate_rejected_samples:
    xs = torch.cat(gate_rejected_samples)[:300]
    free_r, dist_r = evaluate(xs)[:2]
    audit = dict(n=int(len(xs)), free_by_original_test=int(free_r.sum()),
                 free_with_dist_gt_offset=int((free_r & (dist_r > args.offset)).sum()))
    print('pre-filter audit: of {} rejected placements the original test calls {} free ({} with clearance > offset)'.format(
        audit['n'], audit['free_by_original_test'], audit['free_with_dist_gt_offset']), flush=True)

# ----------------------------------------------------------------------------- outputs (as preprocess_obj.main)
pairs = pairs.numpy(); dists = dists.numpy(); angles = angles.numpy()
normals = normals.numpy(); trans_normals = trans_normals.numpy(); rot_normals = rot_normals.numpy()
speed_pairs = (np.clip(dists / args.margin, a_min=args.offset / args.margin, a_max=1.0) + angles / np.pi) / 2
speed_angles = angles / np.pi
speed_dists = np.clip(dists / args.margin, a_min=args.offset / args.margin, a_max=1.0)
pairs[:, 3:6] /= (2 * np.pi)
pairs[:, 9:12] /= (2 * np.pi)
np.save(os.path.join(args.out, 'speed'), speed_pairs)
np.save(os.path.join(args.out, 'speed_angles'), speed_angles)
np.save(os.path.join(args.out, 'speed_dists'), speed_dists)
np.save(os.path.join(args.out, 'normal'), normals)
np.save(os.path.join(args.out, 'trans_n'), trans_normals)
np.save(os.path.join(args.out, 'rot_n'), rot_normals)
np.save(os.path.join(args.out, 'env'), env_points)
meta = {
    'shape_scale': float(args.shape_scale), 'shape_obj': os.path.abspath(args.shape), 'env_obj': os.path.abspath(args.env),
    'margin': float(args.margin), 'offset': float(args.offset), 'rot_norm': float(2 * np.pi), 'testing_data': False,
    'yrot': False, 'two_d': False, 'shape_z_thickness': None, 'env_scale': scale, 'env_center': center_env.tolist(),
    'num_tets': int(TT.shape[0]),
    'prefilter': dict(script='models/graph_homotopy/scripts/preprocess_hole.py', delta=args.delta, shape_points=int(len(S_local)),
                      stats=stats, audit=audit, seconds=t_sample, num_samples=N),
}
with open(os.path.join(args.out, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)
np.save(os.path.join(args.out, 'sampled_points'), pairs)        # last: the queue's completion marker
print('Wrote {} training pairs to {} in {:.0f}s; speed x0 [{:.3f}, {:.3f}], x1 [{:.3f}, {:.3f}]'.format(
    N, args.out, time.time() - t_start, speed_pairs[:, 0].min(), speed_pairs[:, 0].max(), speed_pairs[:, 1].min(),
    speed_pairs[:, 1].max()), flush=True)
