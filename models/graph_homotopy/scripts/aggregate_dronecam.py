#!/usr/bin/env python3
"""Summarize the drone + chase-camera test (run_dronecam_queue.py): results/graph_homotopy/plan_dronecam/<run>/<set>__<planner>.json.

Per scene (meadow .. overgrown), camera set-up (30t, 30w, 55t, 55w, 80t, 80w) and model (hlift dual, mbtd base), two
planners of the same 1000 pairs:

  body       the model trained on dronecam<L><f> (drone + camera + view pyramid): the planner knows the field of view
  follower   the model trained on drone<L> (drone + camera point): the rigid follower that plans the drone alone

Outcomes per case (SE(3) segment): success_se3 = converged and the whole body collision-free (a *clean frame*: nothing
of the scene inside the field of view all along); follow_success_se3 = converged, the drone alone collision-free and the
camera sees >= 99 % of its visible drone samples at every waypoint; bird_collision_se3 (the drone hits), cam_view_ok,
view_mean.  Paired tests: exact McNemar on seed-matched cases.

    python3 models/graph_homotopy/scripts/aggregate_dronecam.py
writes results/graph_homotopy/summary/dronecam_summary.{json,md} and the dashboard tables
results/graph_homotopy/dronecam/dash/dronecam_results.csv (one row per scene x camera x model x trained-on) and
dronecam_paired.csv.
"""
import csv
import glob
import json
import math
import os
import re
import statistics

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.environ.get('DRONECAM_RES', os.path.join(HOST, 'results/graph_homotopy/plan_dronecam'))
OUT = os.environ.get('DRONECAM_OUT', os.path.join(HOST, 'results/graph_homotopy/summary'))
DASH = os.environ.get('DRONECAM_DASH', os.path.join(HOST, 'results/graph_homotopy/dronecam/dash'))
OBJ = os.path.join(HOST, 'datasets/3dshape/obj')
ENVS = ('meadow', 'suburb', 'campus', 'downtown', 'overgrown')
CAMS = ('30t', '30w', '55t', '55w', '80t', '80w')
MODELS = (('hlift', 'dual', 'lifted encoder'), ('mbtd', 'base', 'full NTFields'))
KEYS = ('success_se3', 'follow_success_se3', 'bird_collision_se3', 'collision_se3', 'cam_view_ok', 'converged_se3')
MODEL_LABEL = {'hlift': 'Lifted encoder', 'mbtd': 'Full NTFields'}


def binom_two_sided(k, n):
    if n == 0:
        return None
    lo = min(k, n - k)
    # log space: 2^n overflows a float beyond n = 1024
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(0, lo + 1)]
    m = max(logs)
    p = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * p)


def load(run, setname, planner):
    f = os.path.join(RES, run, '{}__{}.json'.format(setname, planner))
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except ValueError:
        return None


def body_of(kind, cam):
    return 'dronecam' + cam if kind == 'body' else 'drone' + cam[:-1]


def runs(kind, env, cam, model, planner):
    """{seed: records} of the model trained on <body>_<env>, scored on dronecam<cam>_<env> with <planner>."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, '{}_{}_{}_s*'.format(body_of(kind, cam), env, model)))):
        seed = int(re.search(r'_s(\d+)$', d).group(1))
        j = load(os.path.basename(d), 'dronecam{}_{}'.format(cam, env), planner)
        if j is not None:
            out[seed] = j['records']
    return out


def rate(recs, key):
    return sum(bool(r.get(key)) for r in recs) / float(len(recs))


def mean_key(recs, key):
    return statistics.mean(float(r[key]) for r in recs)


def paired(A, B, key):
    """seed-matched cases: (A wins, B wins, p) on outcome key."""
    w = l = 0
    for s in set(A) & set(B):
        bd = {r['case']: bool(r.get(key)) for r in B[s]}
        for r in A[s]:
            if r['case'] in bd:
                a = bool(r.get(key))
                w += a and not bd[r['case']]
                l += bd[r['case']] and not a
    return w, l, binom_two_sided(min(w, l), w + l) if w + l else None


def msd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, 0
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0), len(v)


def fmt(m, sd=None, n=None):
    if m is None:
        return '-'
    s = '{:.1f}'.format(100 * m)
    if sd is not None and n and n > 1:
        s += ' +- {:.1f}'.format(100 * sd)
    return s


def pc(v):
    return '' if v is None else round(100 * v, 1)


def cam_label(cam):
    try:
        cj = json.load(open(os.path.join(OBJ, 'dronecam{}_camera.json'.format(cam))))
        return '{} behind, {} ({:.0f} x {:.0f} deg)'.format(cj['distance'], 'tight' if cam.endswith('t') else 'wide', *cj['fov_deg'])
    except (OSError, KeyError):
        return cam


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DASH, exist_ok=True)
    out = {'cells': {}, 'paired': {}}
    rows, prow = [], []
    L = ['# Drone + chase camera: five scenes, six camera set-ups', '',
         'Scenes meadow / suburb / campus / downtown / overgrown (dataprocessing/make_dronecam3d.py), 1000 RRT-verified pairs '
         'of the camera-aware body per scene and camera set-up. Every model is scored on the same pairs; body = trained on '
         'dronecam<L><f> (the planner knows the field of view), follower = trained on drone<L> (plans the drone + camera '
         'point, no view model). Clean frame = converged and the whole body collision-free (nothing enters the field of '
         'view); follow success = converged, the drone collision-free and the camera sees >= 99 % of its visible drone '
         'samples at every waypoint (ray casting against the scene mesh). Mean +- sd over seeds. Paired: seed-matched '
         'cases, exact McNemar p.', '']
    for env in ENVS:
        cells, P = {}, {}
        for cam in CAMS:
            have = False
            for model, planner, label in MODELS:
                for kind in ('body', 'follower'):
                    R = runs(kind, env, cam, model, planner)
                    if not R:
                        continue
                    if not have:
                        L += ['## {} / camera {} ({})'.format(env, cam, cam_label(cam)), '',
                              '| model | planned as | clean frame | follow success | drone collision | view blocked | '
                              'not converged | mean view | seeds |', '|' + '---|' * 9]
                        have = True
                    per_seed = {k: [rate(recs, k) for recs in R.values()] for k in KEYS}
                    per_seed['view_mean'] = [mean_key(recs, 'view_mean') for recs in R.values()]
                    row = {k: msd(v) for k, v in per_seed.items()}
                    row['seeds'] = sorted(R)
                    key = '{}:{}:{}'.format(cam, model, kind)
                    cells[key] = {k: v if k == 'seeds' else list(v) for k, v in row.items()}
                    L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                        label, kind, fmt(*row['success_se3']), fmt(*row['follow_success_se3']), fmt(*row['bird_collision_se3']),
                        fmt(1 - row['cam_view_ok'][0], row['cam_view_ok'][1], row['cam_view_ok'][2]),
                        fmt(1 - row['converged_se3'][0], row['converged_se3'][1], row['converged_se3'][2]),
                        fmt(*row['view_mean']), len(R)))
                    rows.append(dict(key='{}:{}'.format(env, key), scene=env, scene_order=ENVS.index(env), camera=cam,
                                     camera_label=cam_label(cam), distance=int(cam[:-1]), fov='tight' if cam.endswith('t') else 'wide',
                                     model=MODEL_LABEL[model], model_code=model, planner=planner,
                                     trained_on='camera-aware body' if kind == 'body' else 'rigid follower', kind=kind,
                                     success_se3=pc(row['success_se3'][0]), success_sd=pc(row['success_se3'][1]),
                                     follow_success=pc(row['follow_success_se3'][0]), follow_sd=pc(row['follow_success_se3'][1]),
                                     drone_collision=pc(row['bird_collision_se3'][0]), body_collision=pc(row['collision_se3'][0]),
                                     view_blocked=pc(1 - row['cam_view_ok'][0]), not_converged=pc(1 - row['converged_se3'][0]),
                                     mean_view=pc(row['view_mean'][0]), seeds=len(R)))
            if not have:
                continue
            L += ['', 'Paired (seed-matched cases; wins / losses / p):', '']
            for model, planner, label in MODELS:
                for key, what in (('follow_success_se3', 'follow success'), ('success_se3', 'clean frame')):
                    A, B = runs('body', env, cam, model, planner), runs('follower', env, cam, model, planner)
                    if A and B:
                        w, l, p = paired(A, B, key)
                        name = '{} body vs follower ({})'.format(label, what)
                        P['{}:{}'.format(cam, name)] = dict(a='body', b='follower', model=model, key=key, wins=w, losses=l, p=p)
                        L.append('- camera {}: {}: +{} / -{}, p {}'.format(cam, name, w, l, '-' if p is None else '{:.1e}'.format(p)))
                        prow.append(dict(key='{}:{}:{}'.format(env, cam, name), scene=env, camera=cam, comparison=name, outcome=key,
                                         wins=w, losses=l, p='' if p is None else '{:.1e}'.format(p),
                                         verdict=('body ahead' if w > l else 'follower ahead') if (p is not None and p < 0.01) else 'no clear difference'))
            A, B = runs('body', env, cam, 'hlift', 'dual'), runs('body', env, cam, 'mbtd', 'base')
            if A and B:
                for key, what in (('follow_success_se3', 'follow success'), ('success_se3', 'clean frame')):
                    w, l, p = paired(A, B, key)
                    name = 'lifted vs NTFields (body, {})'.format(what)
                    P['{}:{}'.format(cam, name)] = dict(a='hlift', b='mbtd', key=key, wins=w, losses=l, p=p)
                    L.append('- camera {}: {}: +{} / -{}, p {}'.format(cam, name, w, l, '-' if p is None else '{:.1e}'.format(p)))
                    prow.append(dict(key='{}:{}:{}'.format(env, cam, name), scene=env, camera=cam, comparison=name, outcome=key,
                                     wins=w, losses=l, p='' if p is None else '{:.1e}'.format(p),
                                     verdict=('lifted ahead' if w > l else 'NTFields ahead') if (p is not None and p < 0.01) else 'no clear difference'))
            L.append('')
        out['cells'][env] = cells
        out['paired'][env] = P
    with open(os.path.join(OUT, 'dronecam_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(OUT, 'dronecam_summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    rows.sort(key=lambda r: (r['scene_order'], r['distance'], r['fov'], r['model_code'], r['kind'] != 'body'))
    if rows:
        with open(os.path.join(DASH, 'dronecam_results.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(DASH, 'dronecam_paired.csv'), 'w', newline='') as fh:   # header even when empty: the dashboard reads it
        w = csv.DictWriter(fh, fieldnames=list(prow[0]) if prow else ['key', 'scene', 'camera', 'comparison', 'outcome', 'wins', 'losses', 'p', 'verdict'])
        w.writeheader()
        w.writerows(prow)
    print('\n'.join(L))
    print(len(rows), 'result rows,', len(prow), 'paired rows ->', DASH)


if __name__ == '__main__':
    main()
