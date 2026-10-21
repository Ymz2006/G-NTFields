#!/usr/bin/env python3
"""Convert plan_eval_homotopy.py --save-paths runs (<out>_chosen.npz + <out>.json) to the path files of
dataprocessing/shapes3d_viser.py (<cell>.npz: case, path, clear, converged, collision, success).

Each plan is cut at its first SE(3)-converged waypoint and the goal appended (as the evaluator's success test);
a plan that never converges is shown in full. Outcomes are the SE(3) ones (success_se3 / collision_se3).

    python3 models/graph_homotopy/scripts/export_viser_paths.py \
        --run results/graph_homotopy/showcase/tripod3d_env9_hminL_s0/tripod3d_env9__dual \
        --cell tripod3d_env9 --out results/graph_homotopy/viser_paths/homotopy
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='evaluator output without extension')
    ap.add_argument('--cell', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    z = np.load(a.run + '_chosen.npz')
    recs = {r['case']: r for r in json.load(open(a.run + '.json'))['records']}
    paths, clears = [], []
    for i, case in enumerate(z['case']):
        P, C, k3 = z['path'][i], z['clear'][i], int(z['iters3'][i])
        ok = ~np.isnan(P[:, 0])
        P, C = P[ok], C[ok]
        if k3 >= 0:
            P = np.concatenate([P[:k3 + 2], z['goal'][i][None]], 0)
            C = np.concatenate([C[:k3 + 2], C[k3 + 1:k3 + 2]])
        paths.append(P.astype(np.float32))
        clears.append(C.astype(np.float32))
    obj = lambda xs: np.array(xs + [None], dtype=object)[:-1]  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, a.cell + '.npz'), case=z['case'], path=obj(paths), clear=obj(clears),
                        converged=np.array([recs[int(c)]['converged_se3'] for c in z['case']]),
                        collision=np.array([recs[int(c)]['collision_se3'] for c in z['case']]),
                        success=np.array([recs[int(c)]['success_se3'] for c in z['case']]))
    print(a.cell, len(paths), 'paths, success', np.mean([recs[int(c)]['success_se3'] for c in z['case']]))


if __name__ == '__main__':
    main()
