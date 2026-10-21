#!/usr/bin/env python3
"""Pick one case per showcase set that the homotopy head (dual planner) solves and the full metric NTFields (TD and
normal terms on) fails by colliding, and export it for the dashboard's scene views.

Reads the --save-paths re-runs of plan_eval_homotopy.py in results/graph_homotopy/showcase/<run>/ (seed 0, the same
evaluator and planner seed as the campaign) and writes four CSV files to --out:

  showcase_cases.csv   one row per set: the case, both outcomes, its distance / rotation, walls crossed, up axis
  showcase_paths.csv   both plans (homotopy up to SE(3) convergence, NTFields up to its first colliding waypoint)
  showcase_env.csv     environment surface points near the plans (the evaluator's own 50k samples, cropped)
  showcase_shape.csv   surface samples of the moving shape in its own frame (world point = p + R(r) s)

    python3 models/graph_homotopy/scripts/export_showcase.py --out <dir>
"""
import argparse
import csv
import json
import math
import os

import numpy as np

SHOW = 'results/graph_homotopy/showcase'
SETS = 'testing_data/3dshape'
# (scene id, family, code, set, up axis, walls: list of (axis, env-unit plane) as aggregate_shapes.PLANES)
SCENES = [('tla', 'tla', 'T1', 'Tshape3d_env1_hard', 1, []),
          ('shapes', 'shapes', 'tripod3d_env9', 'tripod3d_env9', 1, [(2, -90.0), (0, 100.0)])]
ENV_CAP = 6000
SHAPE_N = 260
MARGIN, UP_MARGIN = 0.14, 0.07


def load_run(run, setname, planner):
    base = os.path.join(SHOW, run, '{}__{}'.format(setname, planner))
    recs = {r['case']: r for r in json.load(open(base + '.json'))['records']}
    z = np.load(base + '_chosen.npz')
    return recs, z


def rel_angle_deg(r0, r1):
    def quat(r):
        v = 2 * math.pi * np.asarray(r, np.float64)
        th = np.linalg.norm(v)
        ax = v / th if th > 1e-12 else np.zeros(3)
        return np.concatenate([[math.cos(th / 2)], ax * math.sin(th / 2)])
    return math.degrees(2 * math.acos(min(1.0, abs(float(quat(r0) @ quat(r1))))))


def sample_surface(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', nargs='*', default=None, help='scene ids to export (default: all)')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cases, paths, envs, shapes = [], [], [], []
    for sid, fam, code, sn, up, walls in SCENES:
        if a.only and sid not in a.only:
            continue
        hr, hz = load_run('{}_hminL_s0'.format(code), sn, 'dual')
        nr, nz = load_run('{}_mbtd_s0'.format(code), sn, 'base')
        x = np.load(os.path.join(SETS, sn, 'sampled_points.npy'))
        meta = json.load(open(os.path.join(SETS, sn, 'meta.json')))
        c, sc = np.array(meta['env_center']), meta['env_scale']
        crossed = np.zeros(len(x), int)
        for axis, v in walls:
            pl = (v - c[axis]) / sc
            crossed += (x[:, axis] > pl) != (x[:, 6 + axis] > pl)
        cand = [k for k in hr if k in nr and hr[k]['success_se3'] and not nr[k]['success_se3'] and nr[k]['collision_se3']]
        if walls:
            most = max(crossed[k] for k in cand)
            cand = [k for k in cand if crossed[k] == most]
        # the clearest picture: a long move with a large rotation
        k = max(cand, key=lambda q: hr[q]['straight_trans'] + 0.25 * hr[q]['straight_rot_geo'])
        hi, ni = list(hz['case']).index(k), list(nz['case']).index(k)
        k3 = int(hz['iters3'][hi])
        hp = hz['path'][hi][:k3 + 2]
        hc = hz['clear'][hi][:k3 + 2]
        npth, ncl = nz['path'][ni], nz['clear'][ni]
        ok = ~np.isnan(npth[:, 0])
        npth, ncl = npth[ok], ncl[ok]
        hit = int(np.argmax(ncl < 0))
        npth, ncl = npth[:hit + 1], ncl[:hit + 1]
        goal = x[k, 6:]
        hp = np.concatenate([hp, goal[None]], 0)
        hc = np.concatenate([hc, [np.nan]])
        cases.append(dict(scene=sid, family=fam, set=sn, case=k, seed=0, hom_outcome='reached the goal',
                          ntf_outcome='collided', hom_steps=k3 + 1, ntf_collide_step=hit,
                          trans_dist=round(float(np.linalg.norm(x[k, 6:9] - x[k, :3])), 3),
                          rot_deg=round(rel_angle_deg(x[k, 3:6], x[k, 9:12]), 1), walls_crossed=int(crossed[k]),
                          up='xyz'[up], hom_min_clear=round(float(np.nanmin(hc)), 4), candidates=len(cand)))
        for name, P, C in (('homotopy', hp, hc), ('ntf', npth, ncl)):
            for i, (p, cl) in enumerate(zip(P, C)):
                paths.append(dict(scene=sid, method=name, step=i, x=p[0], y=p[1], z=p[2],
                                  rx=2 * math.pi * p[3], ry=2 * math.pi * p[4], rz=2 * math.pi * p[5],
                                  clear='' if np.isnan(cl) else round(float(cl), 4),
                                  collide=int(name == 'ntf' and i == len(P) - 1)))
        E = hz['env_pts']
        allp = np.concatenate([hp[:, :3], npth[:, :3]])
        lo, hi_ = allp.min(0) - MARGIN, allp.max(0) + MARGIN
        lo[up], hi_[up] = allp[:, up].min() - UP_MARGIN, allp[:, up].max() + UP_MARGIN
        E = E[((E >= lo) & (E <= hi_)).all(1)]
        if len(E) > ENV_CAP:
            E = E[rng.choice(len(E), ENV_CAP, replace=False)]
        envs += [dict(scene=sid, x=p[0], y=p[1], z=p[2]) for p in E]
        S = sample_surface(hz['shape_V'].astype(np.float64), hz['shape_F'], SHAPE_N, rng)
        shapes += [dict(scene=sid, x=p[0], y=p[1], z=p[2]) for p in S]
        print(sid, 'case', k, 'of', len(cand), 'candidates; homotopy', k3 + 1, 'steps, NTFields collides at', hit,
              '; env points', len(E))
    os.makedirs(a.out, exist_ok=True)
    r4 = lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else v  # noqa: E731
    for name, rows in (('showcase_cases.csv', cases), ('showcase_paths.csv', paths), ('showcase_env.csv', envs),
                       ('showcase_shape.csv', shapes)):
        with open(os.path.join(a.out, name), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: r4(v) for k, v in r.items()})


if __name__ == '__main__':
    main()
