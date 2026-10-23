"""Point paths for the path-constrained prism (the anchor's trajectory), one scene at a time.

A candidate is a centripetal Catmull-Rom spline through a start, a goal and 1-3 via points drawn uniformly in the
room (each with point clearance >= --via-clear); it is kept if the whole curve keeps point clearance >= --clear
(exact, box scene), its length is in [--min-len, --max-len], its start and goal are >= --min-span apart and its
turning radius is >= --min-radius. For each kept candidate and prism length the free fraction of random rotations
(uniform on SO(3); exact prism clearance > 0) is measured at --stations points along the path: where it is small the
anchored prism has little room to turn and the rotation has to be planned.

    python models/path_constrained/scripts/make_paths.py --env env1 --out datasets/3dshape/paths --seed 0

Writes <out>/<env>_cand.json (every kept candidate's statistics) and <out>/<env>_cand.npz (their samples); the
chosen paths are exported by --export <index> ... as <out>/<env>_<letter>.npz.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.path_constrained import geometry as G  # noqa: E402
from models.path_constrained.path import Path, max_turn_rate, resample, smooth_curve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--env', required=True)
ap.add_argument('--obj-dir', default='datasets/3dshape/obj')
ap.add_argument('--out', default='datasets/3dshape/paths')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--n-candidates', type=int, default=200)
ap.add_argument('--max-tries', type=int, default=200000)
ap.add_argument('--clear', type=float, default=0.04, help='point clearance of the whole path')
ap.add_argument('--via-clear', type=float, default=0.06)
ap.add_argument('--min-len', type=float, default=0.9)
ap.add_argument('--max-len', type=float, default=1.6)
ap.add_argument('--min-span', type=float, default=0.6)
ap.add_argument('--min-radius', type=float, default=0.06)
ap.add_argument('--width', type=float, default=0.02)
ap.add_argument('--lengths', default='0.1,0.15,0.2', help='prism lengths to score')
ap.add_argument('--stations', type=int, default=64)
ap.add_argument('--rots', type=int, default=1024)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--export', nargs='*', type=int, default=None, help='candidate indices to export as paths')
args = ap.parse_args()

dev = args.device if torch.cuda.is_available() else 'cpu'
scene = G.BoxScene(os.path.join(args.obj_dir, args.env + '.obj'))
os.makedirs(args.out, exist_ok=True)
cand_npz = os.path.join(args.out, '{}_cand.npz'.format(args.env))
cand_json = os.path.join(args.out, '{}_cand.json'.format(args.env))

if args.export is not None:
    z = np.load(cand_npz, allow_pickle=True)
    stats = json.load(open(cand_json))
    for j, k in enumerate(args.export):
        name = '{}_{}'.format(args.env, 'abcdefgh'[j])
        p = Path(z['path_{}'.format(k)])
        p.save(os.path.join(args.out, name + '.npz'), via=z['via_{}'.format(k)], env=args.env,
               candidate=k, clear=args.clear)
        print('exported candidate', k, '->', name, 'L = {:.3f}'.format(p.L), stats['candidates'][k]['free'])
    sys.exit(0)

rng = np.random.default_rng(args.seed)
lo, hi = scene.room_lo, scene.room_hi
lengths = [float(x) for x in args.lengths.split(',')]
prisms = {h: G.Prism(args.width, h) for h in lengths}


def pclear(P):
    return scene.point_clearance(torch.as_tensor(P, dtype=torch.float32, device=dev)).cpu().numpy()


def draw_points(n):
    out = []
    while len(out) < n:
        P = rng.uniform(lo, hi, (4096, 3))
        out += list(P[pclear(P) >= args.via_clear])
    return np.asarray(out[:n])


def free_profile(path, prism):
    ell = np.linspace(0, path.L, args.stations)
    t = path.pos(torch.as_tensor(ell, dtype=torch.float32, device=dev))
    rv = torch.as_tensor(G.random_rotvecs(args.rots, rng), dtype=torch.float32, device=dev)
    R = G.so3_exp(rv)
    T = t[:, None, :].expand(-1, args.rots, -1).reshape(-1, 3)
    RR = R[None].expand(args.stations, -1, -1, -1).reshape(-1, 3, 3)
    c = G.prism_clearance(scene, prism, T, RR).reshape(args.stations, args.rots)
    return (c > 0).float().mean(1).cpu().numpy(), (c > 0.02).float().mean(1).cpu().numpy()


cands, paths, vias = [], [], []
t0 = time.time()
tries = 0
while len(cands) < args.n_candidates and tries < args.max_tries:
    tries += 1
    m = int(rng.integers(1, 4))
    pts = draw_points(m + 2)
    if np.linalg.norm(pts[-1] - pts[0]) < args.min_span:
        continue
    # order the via points along the start -> goal direction (fewer loops)
    d = pts[-1] - pts[0]
    mid = pts[1:-1][np.argsort((pts[1:-1] - pts[0]) @ d)]
    via = np.concatenate([pts[:1], mid, pts[-1:]])
    curve = resample(smooth_curve(via), 1e-3)
    p = Path(curve)
    if not (args.min_len <= p.L <= args.max_len):
        continue
    if pclear(p.pts).min() < args.clear:
        continue
    if max_turn_rate(Path(resample(curve, 5e-3))) > 1.0 / args.min_radius:
        continue
    st = {'index': len(cands), 'L': p.L, 'n_via': m, 'min_point_clear': float(pclear(p.pts).min()), 'free': {}}
    for h, pr in prisms.items():
        f0, f2 = free_profile(p, pr)
        st['free'][str(h)] = {'mean': float(f0.mean()), 'min': float(f0.min()), 'q10': float(np.quantile(f0, 0.1)),
                              'frac_below_0.3': float((f0 < 0.3).mean()), 'start': float(f0[0]), 'end': float(f0[-1]),
                              'mean_clear02': float(f2.mean()), 'min_clear02': float(f2.min()), 'profile': f0.round(3).tolist()}
    cands.append(st)
    paths.append(p.pts)
    vias.append(via)
    if len(cands) % 20 == 0:
        print('{} candidates ({} tries, {:.0f}s)'.format(len(cands), tries, time.time() - t0), flush=True)

np.savez(cand_npz, **{'path_{}'.format(i): x for i, x in enumerate(paths)}, **{'via_{}'.format(i): v for i, v in enumerate(vias)})
json.dump({'env': args.env, 'args': vars(args), 'tries': tries, 'room_lo': lo.tolist(), 'room_hi': hi.tolist(),
           'candidates': cands}, open(cand_json, 'w'), indent=1)
print('{} candidates from {} tries in {:.0f}s -> {}'.format(len(cands), tries, time.time() - t0, cand_json))
for h in lengths:
    k = str(h)
    mean = np.array([c['free'][k]['mean'] for c in cands])
    mn = np.array([c['free'][k]['min'] for c in cands])
    print('prism length {}: free fraction mean over path  q10/50/90 {:.3f} {:.3f} {:.3f}; min over path q10/50/90 '
          '{:.3f} {:.3f} {:.3f}'.format(h, *np.quantile(mean, [.1, .5, .9]), *np.quantile(mn, [.1, .5, .9])))
