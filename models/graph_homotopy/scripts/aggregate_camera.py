#!/usr/bin/env python3
"""Summarize the camera + object test (run_camera_queue.py): results/graph_homotopy/plan_camera/<run>/<set>__<planner>.json.

Per scene (forest, city) and model (hminL, hlift, mbtd), two planners of the same pairs:

  body       the model trained on birdcam3d (bird + camera + view pyramid): the planner knows the view volume
  follower   the model trained on bird3d (bird + camera point): the rigid follower that plans the bird alone

scored on the birdcam3d set of the scene by plan_eval_homotopy.py --camera.  Outcomes per case (SE(3) segment):
success_se3 = converged and the whole body (bird, camera, view pyramid) collision-free; follow_success_se3 =
converged, the bird alone collision-free and the camera sees >= 99 % of its visible bird samples at every
waypoint; bird_collision_se3; cam_view_ok; view_mean.  Paired tests: exact McNemar on seed-matched cases.

    python3 models/graph_homotopy/scripts/aggregate_camera.py
writes results/graph_homotopy/summary/camera_summary.{json,md}
"""
import glob
import json
import math
import os
import re
import statistics

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.environ.get('CAMERA_RES', os.path.join(HOST, 'results/graph_homotopy/plan_camera'))
OUT = os.environ.get('CAMERA_OUT', os.path.join(HOST, 'results/graph_homotopy/summary'))
ENVS = ('forest', 'city')
MODELS = (('hlift', 'dual', 'lifted encoder (dual)'), ('hminL', 'dual', 'homotopy (dual)'), ('mbtd', 'base', 'full NTFields'),
          ('hlift', 'base', 'lifted encoder (base)'), ('hminL', 'base', 'homotopy (base)'))
KEYS = ('success_se3', 'follow_success_se3', 'bird_collision_se3', 'collision_se3', 'cam_view_ok', 'converged_se3')


def binom_two_sided(k, n):
    """two-sided exact binomial p of k successes in n at p = 0.5."""
    if n == 0:
        return None
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2.0 ** n
    return min(1.0, 2.0 * p)


def load(run, setname, planner):
    f = os.path.join(RES, run, '{}__{}.json'.format(setname, planner))
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except ValueError:
        return None


def runs(body, env, model, planner):
    """{seed: records} of the model trained on <body>_<env>, scored on birdcam3d_<env> with <planner>."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, '{}_{}_{}_s*'.format(body, env, model)))):
        seed = int(re.search(r'_s(\d+)$', d).group(1))
        j = load(os.path.basename(d), 'birdcam3d_' + env, planner)
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


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'cells': {}, 'paired': {}}
    L = ['# Camera + object test: a bird followed by a camera that must keep it in view', '',
         'Scenes forest / city (dataprocessing/make_camera3d.py), 1000 RRT-verified pairs of the body birdcam3d each. '
         'Every model is scored on the same pairs; body = trained on birdcam3d (the planner knows the camera\'s view '
         'volume), follower = trained on bird3d (plans the bird + camera point, no view model). SE(3) success = '
         'converged and the whole body collision-free; follow success = converged, the bird collision-free and '
         'the camera sees >= 99 % of its visible bird samples at every waypoint (ray casting against the scene mesh). '
         'Mean +- sd over seeds (count in parentheses). Paired: seed-matched cases, exact McNemar p.', '']
    for env in ENVS:
        L += ['## {}'.format(env), '',
              '| model | planner | trained on | SE(3) success | follow success | bird collision | body collision | '
              'view blocked | mean view | seeds |', '|' + '---|' * 10]
        cell = {}
        for model, planner, label in MODELS:
            for body in ('birdcam3d', 'bird3d'):
                R = runs(body, env, model, planner)
                if not R:
                    continue
                per_seed = {k: [rate(recs, k) for recs in R.values()] for k in KEYS}
                per_seed['view_mean'] = [mean_key(recs, 'view_mean') for recs in R.values()]
                row = {k: msd(v) for k, v in per_seed.items()}
                row['seeds'] = sorted(R)
                cell['{}:{}:{}'.format(model, planner, body)] = {k: v if k == 'seeds' else list(v) for k, v in row.items()}
                L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    label.split(' (')[0], planner, 'body' if body == 'birdcam3d' else 'follower',
                    fmt(*row['success_se3']), fmt(*row['follow_success_se3']), fmt(*row['bird_collision_se3']),
                    fmt(*row['collision_se3']), fmt(1 - row['cam_view_ok'][0], row['cam_view_ok'][1], row['cam_view_ok'][2]),
                    fmt(*row['view_mean']), len(R)))
        out['cells'][env] = cell
        # paired comparisons
        P = {}
        L += ['', 'Paired (seed-matched cases; wins / losses / p):', '']
        comps = [('hlift:dual:birdcam3d', 'hminL:dual:birdcam3d', 'success_se3', 'lifted vs homotopy (body)'),
                 ('hlift:dual:birdcam3d', 'mbtd:base:birdcam3d', 'success_se3', 'lifted vs NTFields (body)'),
                 ('hminL:dual:birdcam3d', 'mbtd:base:birdcam3d', 'success_se3', 'homotopy vs NTFields (body)'),
                 ('hlift:dual:birdcam3d', 'hlift:dual:bird3d', 'follow_success_se3', 'lifted body vs lifted follower (follow success)'),
                 ('hminL:dual:birdcam3d', 'hminL:dual:bird3d', 'follow_success_se3', 'homotopy body vs homotopy follower (follow success)'),
                 ('mbtd:base:birdcam3d', 'mbtd:base:bird3d', 'follow_success_se3', 'NTFields body vs NTFields follower (follow success)'),
                 ('hlift:dual:birdcam3d', 'hminL:dual:birdcam3d', 'follow_success_se3', 'lifted vs homotopy (body, follow success)'),
                 ('hlift:dual:birdcam3d', 'mbtd:base:birdcam3d', 'follow_success_se3', 'lifted vs NTFields (body, follow success)'),
                 ('hminL:dual:birdcam3d', 'mbtd:base:birdcam3d', 'follow_success_se3', 'homotopy vs NTFields (body, follow success)')]
        for a, b, key, label in comps:
            ma, pa_, ba = a.split(':')
            mb, pb_, bb = b.split(':')
            A, B = runs(ba, env, ma, pa_), runs(bb, env, mb, pb_)
            if not A or not B:
                continue
            w, l, p = paired(A, B, key)
            P[label] = dict(a=a, b=b, key=key, wins=w, losses=l, p=p)
            L.append('- {}: +{} / -{}, p {}'.format(label, w, l, '-' if p is None else '{:.1e}'.format(p)))
        out['paired'][env] = P
        L.append('')
    with open(os.path.join(OUT, 'camera_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(OUT, 'camera_summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
