#!/usr/bin/env python3
"""CSV tables of the camera + object test for its dashboard, from aggregate_camera.py's camera_summary.json.

    python3 models/graph_homotopy/scripts/export_camera_dash.py --out results/graph_homotopy/camera/dash

camera_results.csv   one row per scene x model x planner x trained-on: SE(3) success, follow success, bird / body
                     collision, view blocked, mean view (percent, mean +- sd over seeds)
camera_paired.csv    the seed-matched paired tests (wins / losses / p / verdict)
"""
import argparse
import csv
import json
import os

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL = {'hlift': 'Lifted encoder', 'hminL': 'Homotopy head', 'mbtd': 'Full NTFields'}
TRAINED = {'birdcam3d': 'camera-aware body', 'bird3d': 'rigid follower'}
ORDER = {'hlift': 0, 'hminL': 1, 'mbtd': 2}


def pc(v):
    return '' if v is None else round(100 * v, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default=os.path.join(HOST, 'results/graph_homotopy/summary/camera_summary.json'))
    ap.add_argument('--out', default=os.path.join(HOST, 'results/graph_homotopy/camera/dash'))
    a = ap.parse_args()
    S = json.load(open(a.summary))
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for env, cell in S['cells'].items():
        for key, r in cell.items():
            model, planner, body = key.split(':')
            rows.append(dict(key='{}:{}'.format(env, key), scene=env, model=MODEL[model], model_code=model, planner=planner,
                             trained_on=TRAINED[body], body=body, order=ORDER[model] * 2 + (body == 'bird3d'),
                             success_se3=pc(r['success_se3'][0]), success_sd=pc(r['success_se3'][1]),
                             follow_success=pc(r['follow_success_se3'][0]), follow_sd=pc(r['follow_success_se3'][1]),
                             bird_collision=pc(r['bird_collision_se3'][0]), body_collision=pc(r['collision_se3'][0]),
                             view_blocked=pc(None if r['cam_view_ok'][0] is None else 1 - r['cam_view_ok'][0]),
                             not_converged=pc(None if r['converged_se3'][0] is None else 1 - r['converged_se3'][0]),
                             mean_view=pc(r['view_mean'][0]), seeds=len(r['seeds'])))
    rows.sort(key=lambda r: (r['scene'], r['order'], r['planner']))
    with open(os.path.join(a.out, 'camera_results.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    P = []
    for env, comps in S['paired'].items():
        for label, c in comps.items():
            p = c['p']
            sig = p is not None and p < 0.01
            P.append(dict(key='{}:{}'.format(env, label), scene=env, comparison=label, outcome=c['key'], wins=c['wins'], losses=c['losses'],
                          p='' if p is None else '{:.1e}'.format(p),
                          verdict=('first ahead' if c['wins'] > c['losses'] else 'second ahead') if sig else 'no clear difference'))
    with open(os.path.join(a.out, 'camera_paired.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(P[0]))
        w.writeheader()
        w.writerows(P)
    print(len(rows), 'result rows,', len(P), 'paired rows ->', a.out)


if __name__ == '__main__':
    main()
