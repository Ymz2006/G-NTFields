"""Training pairs for the stretchable link in one scene: preprocess_obj.py's sampler and speeds in 7 DOF.

A pose is (t, R, L): drone a's position, the link's orientation (R e_z points from drone a to drone b) and its length
in [L_min, L_max]; the chart the models read is x = (t, rotvec / 2 pi, L) (models/drone_link/geometry.py).

    x0 = (t uniform in the room, R uniform on SO(3), L uniform in [L_min, L_max]), kept in the narrow band
         offset < c(x0) < margin, c = min(scene clearance, L - L_min, L_max - L)  (exact, geometry.clearance);
         --band scene: offset < scene clearance < margin and length slack > offset instead (every x0 near an
         obstacle, none bound by a length limit; the speeds are the same functions of c)
    x1 = x0's SE(3) part displaced as in preprocess_obj.py (a uniform direction of the 6-d chart (t, rotvec / 2 pi)
         times a length uniform in [0, sqrt(6)], rotation vector wrapped) with a new length L1 uniform in
         [L_min, L_max]; kept if drone a is in the room and c(x1) > offset; --x1-uniform p: for a share p of the
         pairs x1 is a uniform pose in the room instead (long-range pairs)
    speed_dists  = clip(c / margin, offset / margin, 1)
    speed_angles = angle / pi   (geometry.LinkAngle: about drone a, shells of the box of that length)
    speed        = (speed_dists + speed_angles) / 2      (the speed the metric NTFields trains on)
    normal       = unit 7-d gradient of c (geometry.normal7)

    python models/drone_link/scripts/preprocess_link.py --scene forest --out datasets/3dshape/data/link_forest
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.drone_link import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--scene', required=True, help='scene name (datasets/3dshape/obj/<scene>.obj + _boxes.json)')
ap.add_argument('--out', required=True)
ap.add_argument('--width', type=float, default=0.02)
ap.add_argument('--L-min', type=float, default=0.06)
ap.add_argument('--L-max', type=float, default=0.30)
ap.add_argument('--num_samples', type=int, default=800000, help='pairs')
ap.add_argument('--margin', type=float, default=0.05)
ap.add_argument('--offset', type=float, default=0.001)
ap.add_argument('--env-spacing', type=float, default=0.01, help='surface sampling of the scene for the angle')
ap.add_argument('--batch', type=int, default=65536)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--chunk', type=int, default=8192, help='poses per exact-clearance batch (GPU memory)')
ap.add_argument('--x1-uniform', type=float, default=0.0,
                help='share of the pairs whose x1 is a uniform pose in the room (t, R, L uniform) instead of the '
                     'displacement of x0: long-range pairs (the displacement keeps 97 %% of the drone-a moves under 0.7)')
ap.add_argument('--band', default='total', choices=['total', 'scene'],
                help='total: x0 in the band of c (scene and length walls); scene: x0 in the band of the scene '
                     'clearance with the length slack > offset (every x0 near an obstacle; speeds unchanged)')
args = ap.parse_args()

TWO_PI = 2 * math.pi
dev = args.device
G.CHUNK = args.chunk
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
env_rel = 'datasets/3dshape/obj/{}.obj'.format(args.scene)
scene = G.Scene(os.path.join(REPO, env_rel))
link = G.Link(args.width, args.L_min, args.L_max)
t_start = time.time()
env_pts = scene.surface_points(args.env_spacing, seed=args.seed)
angle = G.LinkAngle(env_pts, args.width, dev, seed=args.seed)
lo = torch.as_tensor(scene.room_lo, dtype=torch.float32, device=dev)
hi = torch.as_tensor(scene.room_hi, dtype=torch.float32, device=dev)
print('scene {} ({} boxes), link w {} L [{}, {}], {} surface points'.format(
    args.scene, scene.n_boxes, args.width, args.L_min, args.L_max, len(env_pts)), flush=True)


def f32(a):
    return torch.as_tensor(a, dtype=torch.float32, device=dev)


N = args.num_samples
keep = {k: [] for k in ('t0', 'r0', 'L0', 't1', 'r1', 'L1')}
count, drawn, n_band, n_band_len = 0, 0, 0, 0
while count < N:
    B = args.batch
    drawn += B
    t0 = lo + (hi - lo) * torch.rand(B, 3, device=dev)
    r0 = f32(G.random_rotvecs(B, rng))
    L0 = f32(rng.uniform(args.L_min, args.L_max, B))
    c0, cs0, sl0 = G.clearance(scene, link, t0, G.so3_exp(r0), L0, parts=True)
    if args.band == 'total':
        band = (c0 > args.offset) & (c0 < args.margin)
    else:
        band = (cs0 > args.offset) & (cs0 < args.margin) & (sl0 > args.offset)
    n_band += int(band.sum())
    n_band_len += int((band & (sl0 < cs0)).sum())
    t0, r0, L0 = t0[band], r0[band], L0[band]
    n = t0.shape[0]
    d = torch.randn(n, 6, device=dev)
    d = d / d.norm(dim=1, keepdim=True) * (torch.rand(n, 1, device=dev) * math.sqrt(6.0))
    t1 = t0 + d[:, :3]
    r1 = G.wrap_rotvec(r0 + TWO_PI * d[:, 3:])
    L1 = f32(rng.uniform(args.L_min, args.L_max, n))
    if args.x1_uniform > 0:
        u = torch.rand(n, device=dev) < args.x1_uniform
        t1 = torch.where(u[:, None], lo + (hi - lo) * torch.rand(n, 3, device=dev), t1)
        r1 = torch.where(u[:, None], f32(G.random_rotvecs(n, rng)), r1)
    inside = ((t1 >= lo) & (t1 <= hi)).all(1)
    t0, r0, L0, t1, r1, L1 = t0[inside], r0[inside], L0[inside], t1[inside], r1[inside], L1[inside]
    ok = G.clearance(scene, link, t1, G.so3_exp(r1), L1) > args.offset
    take = min(int(ok.sum()), N - count)
    for k, v in (('t0', t0), ('r0', r0), ('L0', L0), ('t1', t1), ('r1', r1), ('L1', L1)):
        keep[k].append(v[ok][:take].cpu())
    count += take
    print('pairs {} / {} (drawn {}, band {:.3f}, length-bound share of the band {:.3f}, {:.0f}s)'.format(
        count, N, drawn, n_band / drawn, n_band_len / max(n_band, 1), time.time() - t_start), flush=True)

P = {k: torch.cat(v) for k, v in keep.items()}
print('sampling done in {:.0f}s; speeds ...'.format(time.time() - t_start), flush=True)


def endpoint_terms(t, r, L):
    """clearance, scene clearance, angle, 7-d unit normal for one endpoint set (CPU tensors)."""
    cl, cs_, an, nm = [], [], [], []
    for i in range(0, t.shape[0], 65536):
        tt, rr, LL = t[i:i + 65536].to(dev), r[i:i + 65536].to(dev), L[i:i + 65536].to(dev)
        R = G.so3_exp(rr)
        cs, s, w = G.scene_clearance(scene, link, tt, R, LL, witness=True)
        sl = link.slack(LL)
        c = torch.minimum(cs, sl)
        bound = torch.where(sl < cs, torch.where(LL - link.L_min < link.L_max - LL, -1, 1), 0)
        cl.append(c.cpu())
        cs_.append(cs.cpu())
        an.append(angle(tt, R, LL).cpu())
        nm.append(G.normal7(link, tt, R, LL, s, w, bound).cpu())
        print('  speeds {} / {} ({:.0f}s)'.format(min(i + 65536, t.shape[0]), t.shape[0], time.time() - t_start),
              flush=True)
    return torch.cat(cl), torch.cat(cs_), torch.cat(an), torch.cat(nm)


c0, cs0, a0, n0 = endpoint_terms(P['t0'], P['r0'], P['L0'])
c1, cs1, a1, n1 = endpoint_terms(P['t1'], P['r1'], P['L1'])
assert float(c0.min()) > args.offset and float(c1.min()) > args.offset
dists = torch.stack([c0, c1], 1).numpy()
angles = torch.stack([a0, a1], 1).numpy()
speed_dists = np.clip(dists / args.margin, args.offset / args.margin, 1.0)
speed_angles = angles / np.pi
speed = (speed_dists + speed_angles) / 2
x0 = G.to_chart(P['t0'].numpy(), P['r0'].numpy(), P['L0'].numpy())
x1 = G.to_chart(P['t1'].numpy(), P['r1'].numpy(), P['L1'].numpy())
pts = np.concatenate([x0, x1], 1).astype(np.float32)
assert np.isfinite(pts).all() and np.isfinite(speed).all()

out = args.out
os.makedirs(out, exist_ok=True)
np.save(os.path.join(out, 'sampled_points.npy'), pts)
np.save(os.path.join(out, 'speed.npy'), speed.astype(np.float32))
np.save(os.path.join(out, 'speed_dists.npy'), speed_dists.astype(np.float32))
np.save(os.path.join(out, 'speed_angles.npy'), speed_angles.astype(np.float32))
np.save(os.path.join(out, 'normal.npy'), torch.cat([n0, n1], 1).numpy().astype(np.float32))
np.save(os.path.join(out, 'env.npy'), env_pts[rng.permutation(len(env_pts))[:50000]])
np.save(os.path.join(out, 'scene_clearance.npy'), torch.stack([cs0, cs1], 1).numpy().astype(np.float32))
meta = {
    'shape_scale': 1.0, 'shape_obj': None, 'env_obj': '/workspace/' + env_rel, 'env_obj_rel': env_rel,
    'margin': args.margin, 'offset': args.offset, 'rot_norm': TWO_PI, 'testing_data': False,
    'env_scale': scene.env_scale, 'env_center': scene.env_center.tolist(), 'num_pairs': int(len(pts)), 'dim': 7,
    'link': {'width': args.width, 'L_min': args.L_min, 'L_max': args.L_max, 'scene': args.scene,
             'chart': '(t, rotvec / 2 pi, L): t = drone a, R e_z = (b - a) / |b - a|, L = |b - a|',
             'box': '[-w/2, w/2]^2 x [0, L] in the body frame (drone a at the origin, drone b at (0, 0, L))',
             'clearance': 'min(scene clearance, L - L_min, L_max - L)'},
    'sampling': {'x0': 'uniform t in the room, uniform SO(3), uniform L; offset < clearance < margin' + (
                     '' if args.band == 'total' else ' (scene clearance; length slack > offset)'), 'band': args.band,
                 'x1': 'x0 + 6-d chart displacement (uniform direction, |.| ~ U[0, sqrt 6]), L1 uniform' + (
                     '; a share {} uniform in the room'.format(args.x1_uniform) if args.x1_uniform else ''),
                 'x1_uniform': args.x1_uniform,
                 'drawn_x0': drawn, 'band_rate': n_band / drawn, 'band_length_bound_share': n_band_len / max(n_band, 1),
                 'env_spacing': args.env_spacing, 'n_env_points': int(len(env_pts)), 'seed': args.seed},
    'stats': {'speed_dists_mean': float(speed_dists.mean()), 'speed_angles_mean': float(speed_angles.mean()),
              'speed_mean': float(speed.mean()), 'clearance_x1_median': float(np.median(dists[:, 1])),
              'pair_dist_median': float(np.median(np.linalg.norm(pts[:, 7:] - pts[:, :7], axis=1))),
              'seconds': time.time() - t_start},
}
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
print('wrote', out, pts.shape, json.dumps(meta['stats']), flush=True)
