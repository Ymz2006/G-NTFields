#!/usr/bin/env python3
"""Export the camera + object demo for the dashboard: per scene, one case where the camera-aware body plan keeps
the bird in view and the rigid follower does not, with the meshes, the scene boxes and every plan's poses.

Reads the --save-paths runs of plan_eval_homotopy.py --camera (results/graph_homotopy/plan_camera/<run>/
birdcam3d_<env>__<planner>.json + _chosen.npz) and writes one JSON (normalized units, y up):

  body / bird        meshes in the body frame (V, F); camera: tip and base rectangle (body frame)
  scenes[env]        room (inner faces), obstacles (boxes), the chosen case: start, goal, and per plan the poses
                     up to SE(3) convergence (+ goal), per-waypoint view fraction, bird and body clearance, outcome

    python3 models/graph_homotopy/scripts/export_camera_showcase.py --out results/graph_homotopy/camera/showcase.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(HOST, 'dataprocessing'))
import make_shapes3d as ms  # noqa: E402
import make_camera3d as mc  # noqa: E402

RES = 'results/graph_homotopy/plan_camera'
SETS = 'testing_data/3dshape'
OBJ = 'datasets/3dshape/obj'
# (key, body trained on, model, planner, label)
PLANS = [('lifted_body', 'birdcam3d', 'hlift', 'dual', 'lifted encoder, camera-aware body'),
         ('homotopy_body', 'birdcam3d', 'hminL', 'dual', 'homotopy head, camera-aware body'),
         ('ntf_body', 'birdcam3d', 'mbtd', 'base', 'full NTFields, camera-aware body'),
         ('lifted_follower', 'bird3d', 'hlift', 'dual', 'lifted encoder, bird only (rigid follower)'),
         ('homotopy_follower', 'bird3d', 'hminL', 'dual', 'homotopy head, bird only (rigid follower)'),
         ('ntf_follower', 'bird3d', 'mbtd', 'base', 'full NTFields, bird only (rigid follower)')]


def load_obj_simple(path):
    V, F = [], []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def rel_angle_deg(r0, r1):
    def quat(r):
        v = 2 * math.pi * np.asarray(r, np.float64)
        th = np.linalg.norm(v)
        ax = v / th if th > 1e-12 else np.zeros(3)
        return np.concatenate([[math.cos(th / 2)], ax * math.sin(th / 2)])
    return math.degrees(2 * math.acos(min(1.0, abs(float(quat(r0) @ quat(r1))))))


def load_run(base):
    j = json.load(open(base + '.json'))
    z = np.load(base + '_chosen.npz')
    recs = {r['case']: r for r in j['records']}
    return recs, z, j['summary']


def segment(z, i, goal):
    """poses of plan i up to SE(3) convergence (+ goal) or the whole plan, with view / clearances cut alike."""
    P, C, V, B, k3 = z['path'][i], z['clear'][i], z['view'][i], z['bird_clear'][i], int(z['iters3'][i])
    ok = ~np.isnan(P[:, 0])
    P, C, V, B = P[ok], C[ok], V[ok], B[ok]
    conv = k3 >= 0
    if conv:
        P, C, V, B = P[:k3 + 2], C[:k3 + 2], V[:k3 + 2], B[:k3 + 2]
        P = np.concatenate([P, goal[None]], 0)
        C, V, B = (np.concatenate([x, x[-1:]]) for x in (C, V, B))
    return P, C, V, B, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_homotopy/camera/showcase.json')
    ap.add_argument('--envs', nargs='*', default=['forest', 'city'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', default=RES)
    ap.add_argument('--sets', default=SETS)
    ap.add_argument('--n-ghosts', type=int, default=7, help='poses drawn along each plan')
    a = ap.parse_args()
    os.chdir(HOST)
    cam = json.load(open(os.path.join(OBJ, 'birdcam3d_camera.json')))
    centre = np.asarray(cam['frame_centre'])
    meta0 = json.load(open(os.path.join(a.sets, 'birdcam3d_{}'.format(a.envs[0]), 'meta.json')))
    sc, ec = float(meta0['env_scale']), np.asarray(meta0['env_center'])
    nb = lambda p: ((np.asarray(p, np.float64) - centre) / sc)          # body frame  # noqa: E731
    ne = lambda p: ((np.asarray(p, np.float64) - ec) / sc)              # scene frame  # noqa: E731
    Vb, Fb = load_obj_simple(os.path.join(OBJ, 'birdcam3d.obj'))
    Vi, Fi = load_obj_simple(os.path.join(OBJ, 'bird3d_only.obj'))
    out = dict(env_scale=sc, env_center=ec.tolist(), seed=a.seed,
               body=dict(V=np.round(nb(Vb), 5).tolist(), F=Fb.tolist()), bird=dict(V=np.round(nb(Vi), 5).tolist(), F=Fi.tolist()),
               camera=dict(apex=np.round(nb(cam['apex']), 5).tolist(), base=np.round(nb(cam['base']), 5).tolist(),
                           fov_deg=cam['fov_deg'], follow_distance=mc.CAM_L / sc),
               plans={k: dict(body=b, model=m, planner=p, label=lab) for k, b, m, p, lab in PLANS}, scenes={})
    rng = np.random.default_rng(7)
    boxes_of = {'forest': [(n, bs) for n, bs in mc.trees(rng)], 'city': mc.buildings()}
    for env in a.envs:
        setname = 'birdcam3d_' + env
        x = np.load(os.path.join(a.sets, setname, 'sampled_points.npy'))
        runs = {}
        for key, body, model, planner, _ in PLANS:
            base = os.path.join(a.res, '{}_{}_{}_s{}'.format(body, env, model, a.seed), '{}__{}'.format(setname, planner))
            if os.path.exists(base + '.json') and os.path.exists(base + '_chosen.npz'):
                runs[key] = load_run(base)
        if 'lifted_body' not in runs:
            print(env, 'no lifted body run; skipped')
            continue
        R = {k: v[0] for k, v in runs.items()}
        cases = sorted(set.intersection(*[set(r) for r in R.values()]))

        def good(k, fail):
            def f(c):
                r = R[k][c]
                return (not r['follow_success_se3']) if fail else r['follow_success_se3']
            return f
        # the picture: camera-aware body plans keep the view, the rigid follower loses it, NTFields' body plan collides
        tiers = [lambda c: good('lifted_body', False)(c) and good('homotopy_body', False)(c) and good('lifted_follower', True)(c)
                 and good('ntf_body', True)(c),
                 lambda c: good('lifted_body', False)(c) and good('lifted_follower', True)(c) and good('ntf_body', True)(c),
                 lambda c: good('lifted_body', False)(c) and good('lifted_follower', True)(c),
                 lambda c: good('lifted_body', False)(c),
                 lambda c: True]
        tiers = [t for t in tiers if all(k in R for k in ('lifted_body', 'homotopy_body', 'lifted_follower', 'ntf_body'))] or tiers[-2:]
        cand, tier = [], None
        for ti, t in enumerate(tiers):
            cand = [c for c in cases if t(c)]
            if cand:
                tier = ti
                break
        if not cand:
            print(env, 'no case; skipped')
            continue
        # the clearest picture: a long move with a big rotation whose follower loses the view early
        score = lambda c: R['lifted_body'][c]['straight_trans'] + 0.25 * R['lifted_body'][c]['straight_rot_geo']  # noqa: E731
        k = max(cand, key=score)
        plans = {}
        for key, (recs, z, summ) in runs.items():
            i = list(z['case']).index(k)
            P, C, V, B, conv = segment(z, i, x[k, 6:])
            r = recs[k]
            # first waypoint where the camera loses the bird (view < 99 %) or the bird / body collides
            lost = int(np.argmax(V < 0.99)) if (V < 0.99).any() else None
            bird_hit = int(np.argmax(B < 0)) if (B < 0).any() else None
            body_hit = int(np.argmax(C < 0)) if (C < 0).any() else None
            idx = np.unique(np.round(np.linspace(0, len(P) - 1, a.n_ghosts)).astype(int)).tolist()
            plans[key] = dict(poses=np.round(P, 5).tolist(), view=np.round(V, 3).tolist(), bird_clear=np.round(B, 4).tolist(),
                              body_clear=np.round(C, 4).tolist(), converged=bool(conv), ghosts=idx, view_lost_at=lost,
                              bird_hit_at=bird_hit, body_hit_at=body_hit,
                              outcome=dict(success_se3=r['success_se3'], follow_success_se3=r['follow_success_se3'],
                                           bird_collision_se3=r['bird_collision_se3'], cam_view_ok=r['cam_view_ok'],
                                           view_min=round(r['view_min'], 3), view_mean=round(r['view_mean'], 3),
                                           steps=int(len(P) - 1)))
        obstacles = [dict(name=n, boxes=[[np.round(ne([b[0], b[2], b[4]]), 5).tolist(), np.round(ne([b[1], b[3], b[5]]), 5).tolist()]
                                         for b in bs]) for n, bs in boxes_of[env]]
        out['scenes'][env] = dict(
            set=setname, case=int(k), tier=tier, candidates=len(cand), start=np.round(x[k, :6], 5).tolist(), goal=np.round(x[k, 6:], 5).tolist(),
            trans_dist=round(float(np.linalg.norm(x[k, 6:9] - x[k, :3])), 3), rot_deg=round(rel_angle_deg(x[k, 3:6], x[k, 9:12]), 1),
            room=dict(lo=np.round(ne([ms.X0, ms.Y0, ms.Z0]), 5).tolist(), hi=np.round(ne([ms.X1, ms.Y1, ms.Z1]), 5).tolist()),
            obstacles=obstacles, plans=plans)
        print(env, 'case', k, 'tier', tier, 'of', len(cand), 'candidates;',
              {kk: (v['outcome']['follow_success_se3'], v['outcome']['success_se3'], v['view_lost_at']) for kk, v in plans.items()})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    # The dashboard reads JSON files as rows: one row per part, the part's object in `data`.
    scenes = out.pop('scenes')
    rows = [dict(part='common', data=out)] + [dict(part=env, data=S) for env, S in scenes.items()]
    with open(a.out, 'w') as fh:
        json.dump({'rows': rows}, fh, separators=(',', ':'))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'kB')


if __name__ == '__main__':
    main()
