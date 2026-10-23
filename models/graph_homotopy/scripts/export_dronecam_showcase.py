#!/usr/bin/env python3
"""Export the drone + chase-camera demo for its dashboard: per scene, one case (of one camera set-up) where the
camera-aware body plan keeps the drone in view and the rigid follower does not, with the meshes, the scene boxes and
every plan's poses.

Reads the --save-paths runs of plan_eval_homotopy.py --camera (results/graph_homotopy/plan_dronecam/<run>/
dronecam<cam>_<env>__<planner>.json + _chosen.npz) and writes one JSON of rows (the Dashboard type reads JSON as rows):
row `common` = env scale / centre, the bodies (drone, and per camera set-up the body mesh, apex, base, field of view),
the plan keys; one row per scene = room (inner faces), obstacles (boxes + part), the chosen case (start, goal, camera
set-up) and per plan the poses up to SE(3) convergence (+ goal), per-waypoint view fraction, drone and body clearance,
outcome.  Normalized units, y up.

    python3 models/graph_homotopy/scripts/export_dronecam_showcase.py --out results/graph_homotopy/dronecam/showcase.json
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

RES = 'results/graph_homotopy/plan_dronecam'
SETS = 'testing_data/3dshape'
OBJ = 'datasets/3dshape/obj'
ENVS = ['meadow', 'suburb', 'campus', 'downtown', 'overgrown']
CAMS = ['30t', '30w', '55t', '55w', '80t', '80w']
# (key, trained as, model, planner, label)
PLANS = [('lifted_body', 'body', 'hlift', 'dual', 'lifted encoder, camera-aware body'),
         ('ntf_body', 'body', 'mbtd', 'base', 'full NTFields, camera-aware body'),
         ('lifted_follower', 'follower', 'hlift', 'dual', 'lifted encoder, drone only (rigid follower)'),
         ('ntf_follower', 'follower', 'mbtd', 'base', 'full NTFields, drone only (rigid follower)')]


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


def body_name(kind, cam):
    return 'dronecam' + cam if kind == 'body' else 'drone' + cam[:-1]


def load_run(base):
    j = json.load(open(base + '.json'))
    z = np.load(base + '_chosen.npz')
    return {r['case']: r for r in j['records']}, z, j['summary']


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
    ap.add_argument('--out', default='results/graph_homotopy/dronecam/showcase.json')
    ap.add_argument('--envs', nargs='*', default=ENVS)
    ap.add_argument('--cams', nargs='*', default=['55w', '80w', '80t', '55t', '30w', '30t'], help='camera set-ups tried per scene, in order')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--res', default=RES)
    ap.add_argument('--sets', default=SETS)
    ap.add_argument('--n-ghosts', type=int, default=7, help='poses drawn along each plan')
    a = ap.parse_args()
    os.chdir(HOST)
    meta0 = None
    for env in a.envs:
        for cam in a.cams:
            p = os.path.join(a.sets, 'dronecam{}_{}'.format(cam, env), 'meta.json')
            if os.path.exists(p):
                meta0 = json.load(open(p))
                break
        if meta0:
            break
    if meta0 is None:
        sys.exit('no test set found')
    sc, ec = float(meta0['env_scale']), np.asarray(meta0['env_center'])
    ne = lambda p: ((np.asarray(p, np.float64) - ec) / sc)              # scene frame  # noqa: E731
    bodies = {}
    for cam in CAMS:
        cj = json.load(open(os.path.join(OBJ, 'dronecam{}_camera.json'.format(cam))))
        centre = np.asarray(cj['frame_centre'])
        nb = lambda p, c=centre: ((np.asarray(p, np.float64) - c) / sc)  # noqa: E731
        Vb, Fb = load_obj_simple(os.path.join(OBJ, cj['body_obj']))
        Vi, Fi = load_obj_simple(os.path.join(OBJ, cj['bird_obj']))
        bodies[cam] = dict(body=dict(V=np.round(nb(Vb), 5).tolist(), F=Fb.tolist()),
                           drone=dict(V=np.round(nb(Vi), 5).tolist(), F=Fi.tolist()),
                           camera=dict(apex=np.round(nb(cj['apex']), 5).tolist(), base=np.round(nb(cj['base']), 5).tolist(),
                                       fov_deg=cj['fov_deg'], distance=cj['distance'], fov=cj['fov'],
                                       follow_distance=(cj['base'][0][0] - cj['apex'][0]) / sc),
                           label='{} units behind, {} field of view ({:.0f} x {:.0f} deg)'.format(
                               cj['distance'], 'tight' if cj['fov'] == 't' else 'wide', *cj['fov_deg']))
    out = dict(env_scale=sc, env_center=ec.tolist(), seed=a.seed, bodies=bodies,
               plans={k: dict(kind=b, model=m, planner=p, label=lab) for k, b, m, p, lab in PLANS})
    scenes = {}
    for env in a.envs:
        picked = None
        for cam in a.cams:
            setname = 'dronecam{}_{}'.format(cam, env)
            xp = os.path.join(a.sets, setname, 'sampled_points.npy')
            if not os.path.exists(xp):
                continue
            x = np.load(xp)
            runs = {}
            for key, kind, model, planner, _ in PLANS:
                base = os.path.join(a.res, '{}_{}_{}_s{}'.format(body_name(kind, cam), env, model, a.seed), '{}__{}'.format(setname, planner))
                if os.path.exists(base + '.json') and os.path.exists(base + '_chosen.npz'):
                    try:
                        runs[key] = load_run(base)
                    except (ValueError, KeyError):
                        pass
            if 'lifted_body' not in runs or 'lifted_follower' not in runs:
                continue
            picked = (cam, setname, x, runs)
            break
        if picked is None:
            print(env, 'no lifted body + follower run yet; skipped')
            continue
        cam, setname, x, runs = picked
        R = {k: v[0] for k, v in runs.items()}
        cases = sorted(set.intersection(*[set(r) for r in R.values()]))

        def ok(k, c):
            return R[k][c]['follow_success_se3']
        # the picture: camera-aware body plans keep the view, the rigid follower loses it, NTFields' body plan fails
        tiers = [lambda c: ok('lifted_body', c) and not ok('lifted_follower', c) and 'ntf_body' in R and not ok('ntf_body', c)
                 and 'ntf_follower' in R and not ok('ntf_follower', c),
                 lambda c: ok('lifted_body', c) and not ok('lifted_follower', c) and 'ntf_follower' in R and not ok('ntf_follower', c),
                 lambda c: ok('lifted_body', c) and not ok('lifted_follower', c),
                 lambda c: ok('lifted_body', c),
                 lambda c: True]
        cand, tier = [], None
        for ti, t in enumerate(tiers):
            cand = [c for c in cases if t(c)]
            if cand:
                tier = ti
                break
        # the clearest picture: a long move with a big rotation whose follower loses a large part of the view
        score = lambda c: (R['lifted_body'][c]['straight_trans'] + 0.25 * R['lifted_body'][c]['straight_rot_geo']  # noqa: E731
                           + 1.5 * (1.0 - min(1.0, R['lifted_follower'][c]['view_min'])))
        k = max(cand, key=score)
        plans = {}
        for key, (recs, z, summ) in runs.items():
            i = list(z['case']).index(k)
            P, C, V, B, conv = segment(z, i, x[k, 6:])
            r = recs[k]
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
        J = json.load(open(os.path.join(OBJ, env + '_boxes.json')))
        obstacles = [dict(name=g['name'], kind=g['kind'], parts=g['parts'],
                          boxes=[[np.round(ne([b[0], b[2], b[4]]), 5).tolist(), np.round(ne([b[1], b[3], b[5]]), 5).tolist()]
                                 for b in g['boxes']]) for g in J['groups']]
        scenes[env] = dict(
            set=setname, camera=cam, case=int(k), tier=tier, candidates=len(cand), note=J['note'],
            start=np.round(x[k, :6], 5).tolist(), goal=np.round(x[k, 6:], 5).tolist(),
            trans_dist=round(float(np.linalg.norm(x[k, 6:9] - x[k, :3])), 3), rot_deg=round(rel_angle_deg(x[k, 3:6], x[k, 9:12]), 1),
            room=dict(lo=np.round(ne([ms.X0, ms.Y0, ms.Z0]), 5).tolist(), hi=np.round(ne([ms.X1, ms.Y1, ms.Z1]), 5).tolist()),
            obstacles=obstacles, plans=plans)
        print(env, 'camera', cam, 'case', k, 'tier', tier, 'of', len(cand), 'candidates;',
              {kk: (v['outcome']['follow_success_se3'], v['outcome']['success_se3'], v['view_lost_at']) for kk, v in plans.items()})
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    rows = [dict(part='common', data=out)] + [dict(part=env, data=S) for env, S in scenes.items()]
    with open(a.out, 'w') as fh:
        json.dump({'rows': rows}, fh, separators=(',', ':'))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'kB')


if __name__ == '__main__':
    main()
