"""Training pairs on the path-constrained pose space M = [0, L] x SO(3) (one cell = scene + path + prism).

The sampling and speeds of dataprocessing/preprocess_obj.py, restricted to M:

    x0 = (l0, R0), l0 uniform on [0, L], R0 uniform on SO(3), kept in the narrow band offset < clearance < margin
    x1 = x0 displaced along M: (l0 + dl, R0 exp(2 pi dr)), (dl, dr) a uniform direction of the 4-d tangent space
         (dr in turns) times a length uniform in [0, 2] (= sqrt(4), as sqrt(6) in SE(3)); kept if l1 is on the
         path and clearance > offset
    speed_dists  = clip(clearance / margin, offset / margin, 1)        (clearance exact, geometry.prism_clearance)
    speed_angles = angle / pi                                            (geometry.AngleClearance, about the anchor)
    speed        = (speed_dists + speed_angles) / 2
    normal       = unit(d clearance / d(l, rotvec / 2 pi)) from the exact witness pair (translation block = the
                   workspace normal's component along the path, rotation block = 2 pi x the moment about the anchor)

Two cells are written from the same poses: <out> in the arclength chart (t = (l - L / 2, 0, 0)) and <out>_world in
the world chart (t = p(l)) (models/path_constrained/path.py).

    python models/path_constrained/scripts/preprocess_path.py --path datasets/3dshape/paths/env1_a.npz \
        --out datasets/3dshape/data/prism_env1_a --num_samples 800000
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.path_constrained import geometry as G  # noqa: E402
from models.path_constrained.path import Path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--path', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--width', type=float, default=0.02)
ap.add_argument('--length', type=float, default=0.2)
ap.add_argument('--num_samples', type=int, default=800000, help='pairs')
ap.add_argument('--margin', type=float, default=0.05)
ap.add_argument('--offset', type=float, default=0.001)
ap.add_argument('--env-spacing', type=float, default=0.005, help='surface sampling of the scene for the angle')
ap.add_argument('--batch', type=int, default=32768)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--no-world', action='store_true')
args = ap.parse_args()

TWO_PI = 2 * math.pi
dev = args.device
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
path = Path.load(args.path)
env = str(path.extra['env'])
env_rel = 'datasets/3dshape/obj/{}.obj'.format(env)
scene = G.BoxScene(os.path.join(REPO, env_rel))
prism = G.Prism(args.width, args.length)
L = path.L
t_start = time.time()
env_pts = scene.surface_points(args.env_spacing, seed=args.seed)
angle = G.AngleClearance(env_pts, prism, path.pts, dev)
print('scene {} ({} boxes), path L = {:.3f}, prism {} x {}, {} surface points ({} max near the path)'.format(
    env, scene.n_boxes, L, args.width, args.length, len(env_pts), angle.max_pts), flush=True)


def clear(ell, R, witness=False):
    return G.prism_clearance(scene, prism, path.pos(ell), R, witness=witness)


def rand_rot(n):
    return G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng), dtype=torch.float32, device=dev))


N = args.num_samples
keep = {k: [] for k in ('l0', 'l1', 'R0', 'R1')}
count, drawn = 0, 0
while count < N:
    B = args.batch
    drawn += B
    l0 = torch.as_tensor(rng.uniform(0, L, B), dtype=torch.float32, device=dev)
    R0 = rand_rot(B)
    c0 = clear(l0, R0)
    band = (c0 > args.offset) & (c0 < args.margin)
    l0, R0 = l0[band], R0[band]
    n = l0.shape[0]
    d = torch.as_tensor(rng.normal(size=(n, 4)), dtype=torch.float32, device=dev)
    d = d / d.norm(dim=1, keepdim=True) * torch.as_tensor(rng.uniform(0, 2.0, (n, 1)), dtype=torch.float32, device=dev)
    l1 = l0 + d[:, 0]
    R1 = R0 @ G.so3_exp(TWO_PI * d[:, 1:])
    on = (l1 >= 0) & (l1 <= L)
    l0, R0, l1, R1 = l0[on], R0[on], l1[on], R1[on]
    c1 = clear(l1, R1)
    ok = c1 > args.offset
    take = min(int(ok.sum()), N - count)
    for k, v in (('l0', l0), ('l1', l1), ('R0', R0), ('R1', R1)):
        keep[k].append(v[ok][:take].cpu())
    count += take
    print('pairs {} / {} (drawn {}, band {:.3f}, {:.0f}s)'.format(count, N, drawn, float(band.float().mean()),
                                                                  time.time() - t_start), flush=True)

l0, l1 = torch.cat(keep['l0']), torch.cat(keep['l1'])
R0, R1 = torch.cat(keep['R0']), torch.cat(keep['R1'])
print('sampling done in {:.0f}s; speeds ...'.format(time.time() - t_start), flush=True)


def endpoint_terms(ell, R):
    """clearance, angle, unit normal in (l, rotvec / 2 pi) and in world coordinates (6,) for one endpoint set."""
    cl, an, nl, nw = [], [], [], []
    for i in range(0, ell.shape[0], 65536):
        e = ell[i:i + 65536].to(dev)
        r = R[i:i + 65536].to(dev)
        t = path.pos(e)
        c, s, w = G.prism_clearance(scene, prism, t, r, witness=True)
        a = angle(t, r, path.sample_index(e))
        n_ws = (s - w) / (s - w).norm(dim=1, keepdim=True).clamp_min(1e-12)
        mom = torch.cross(s - t, n_ws, dim=1)
        tau = path.tangent(e)
        nt = (n_ws * tau).sum(1, keepdim=True)
        v = torch.cat([nt, torch.zeros_like(nt), torch.zeros_like(nt), TWO_PI * mom], 1)
        vw = torch.cat([nt * tau, TWO_PI * mom], 1)
        cl.append(c.cpu())
        an.append(a.cpu())
        nl.append((v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu())
        nw.append((vw / vw.norm(dim=1, keepdim=True).clamp_min(1e-12)).cpu())
        print('  speeds {} / {} ({:.0f}s)'.format(min(i + 65536, ell.shape[0]), ell.shape[0], time.time() - t_start),
              flush=True)
    return torch.cat(cl), torch.cat(an), torch.cat(nl), torch.cat(nw)


c0, a0, n0, nw0 = endpoint_terms(l0, R0)
c1, a1, n1, nw1 = endpoint_terms(l1, R1)
dists = torch.stack([c0, c1], 1).numpy()
angles = torch.stack([a0, a1], 1).numpy()
speed_dists = np.clip(dists / args.margin, args.offset / args.margin, 1.0)
speed_angles = angles / np.pi
speed = (speed_dists + speed_angles) / 2
rv0 = G.so3_log(R0.double()).numpy() / TWO_PI
rv1 = G.so3_log(R1.double()).numpy() / TWO_PI
assert np.isfinite(rv0).all() and np.isfinite(rv1).all()


def write(out, chart):
    os.makedirs(out, exist_ok=True)
    if chart == 'arclength':
        t0 = np.zeros((len(l0), 3))
        t1 = np.zeros((len(l1), 3))
        t0[:, 0] = l0.numpy() - 0.5 * L
        t1[:, 0] = l1.numpy() - 0.5 * L
        nrm = np.concatenate([n0.numpy(), n1.numpy()], 1)
    else:
        t0 = path.pos(l0.double()).numpy()
        t1 = path.pos(l1.double()).numpy()
        nrm = np.concatenate([nw0.numpy(), nw1.numpy()], 1)
    pts = np.concatenate([t0, rv0, t1, rv1], 1).astype(np.float32)
    np.save(os.path.join(out, 'sampled_points.npy'), pts)
    np.save(os.path.join(out, 'speed.npy'), speed.astype(np.float32))
    np.save(os.path.join(out, 'speed_dists.npy'), speed_dists.astype(np.float32))
    np.save(os.path.join(out, 'speed_angles.npy'), speed_angles.astype(np.float32))
    np.save(os.path.join(out, 'normal.npy'), nrm.astype(np.float32))
    np.save(os.path.join(out, 'env.npy'), env_pts[rng.permutation(len(env_pts))[:50000]])
    np.save(os.path.join(out, 'path_ell.npy'), np.stack([l0.numpy(), l1.numpy()], 1).astype(np.float32))
    shutil.copy2(args.path, os.path.join(out, 'path.npz'))
    meta = {
        'shape_scale': 1.0, 'shape_obj': None, 'env_obj': '/workspace/' + env_rel, 'env_obj_rel': env_rel,
        'margin': args.margin, 'offset': args.offset, 'rot_norm': TWO_PI, 'testing_data': False,
        'env_scale': scene.env_scale, 'env_center': scene.env_center.tolist(), 'num_pairs': int(len(pts)),
        'constraint': {'type': 'path', 'chart': chart, 'path_file': 'path.npz', 'path_source': args.path,
                       'L': L, 'prism_width': args.width, 'prism_length': args.length,
                       'anchor': 'centre of the end face z = 0', 'axis': 'body +z', 'env': env},
        'sampling': {'x0': 'uniform l, uniform SO(3), offset < clearance < margin',
                     'x1': 'x0 exp(4-d tangent displacement, |.| ~ U[0, 2])', 'drawn_x0': drawn,
                     'env_spacing': args.env_spacing, 'seed': args.seed},
        'stats': {'speed_dists_mean': float(speed_dists.mean()), 'speed_angles_mean': float(speed_angles.mean()),
                  'speed_mean': float(speed.mean()), 'clearance_x1_median': float(np.median(dists[:, 1]))},
    }
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    print('wrote', out, pts.shape, flush=True)


write(args.out, 'arclength')
if not args.no_world:
    write(args.out.rstrip('/') + '_world', 'world')
print('total {:.0f}s'.format(time.time() - t_start))
