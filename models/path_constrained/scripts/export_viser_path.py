"""Export the forward-mode plans of the path-constrained runs for path_viser.py (numpy only, no torch there).

One file per test set, results/path_constrained/viser/<set>.npz:
    scene      room_lo, room_hi (3,), box_lo, box_hi (k, 3), env, prism_width, prism_length, L, path_pts (n, 3)
    cases      case, l0, rv0, l1, rv1 (radians), start_clear, goal_clear
    per model  <m>_case, <m>_len, <m>_flat (sum len, 9: l, anchor xyz, quaternion wxyz, clearance),
               <m>_seg (sum len: smallest clearance on the motion from the waypoint to the next, 4 sub-steps),
               <m>_success, <m>_collision, <m>_converged, <m>_label
m = hom (homotopy, dual planner) and ntf (metric NTFields), seed 0. A plan is cut at its first SE(3)-converged
waypoint and the goal appended (the evaluator's success test); a plan that never converges is shown in full.

    python models/path_constrained/scripts/export_viser_path.py          (container, repo root)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.path_constrained import geometry as G  # noqa: E402
from models.path_constrained.path import Path  # noqa: E402

RES = os.path.join(REPO, 'results/path_constrained')
SETS = os.path.join(REPO, 'testing_data/3dshape')
TWO_PI = 2 * math.pi
# (run code, test sets)
GROUPS = [('e1a', ['prism_env1_a_' + v for v in ('full', 'tight', 'bigrot', 'hard', 'seg')]),
          ('e3a', ['prism_env3_a_' + v for v in ('full', 'tight', 'bigrot', 'hard', 'seg')]),
          ('e4a', ['prism_env4_a_' + v for v in ('full', 'tight', 'bigrot', 'hard', 'seg')]),
          ('e1b', ['prism_env1_b_full', 'prism_env1_b_seg']), ('e3b', ['prism_env3_b_full', 'prism_env3_b_seg']),
          ('e4b', ['prism_env4_b_full', 'prism_env4_b_seg']),
          ('e1s', ['prism30_env1_a_full', 'prism30_env1_a_hard']), ('e3s', ['prism30_env3_a_full', 'prism30_env3_a_hard']),
          ('e4s', ['prism30_env4_a_full', 'prism30_env4_a_hard'])]
MODELS = [('hom', 'hom_s0', 'dual', 'homotopy (dual planner)'), ('ntf', 'ntf_s0', 'base', 'metric NTFields')]


def plan_poses(Y, k3, goal):
    """Waypoints (l, r = rotvec / 2 pi) cut at convergence, goal appended."""
    Y = Y[~np.isnan(Y[:, 0])].astype(np.float64)
    if k3 >= 0:
        Y = np.concatenate([Y[:k3 + 2], goal[None]], 0)
    return Y


def export_set(code, s, out):
    d = os.path.join(SETS, s)
    meta = json.load(open(os.path.join(d, 'meta.json')))
    con = meta['constraint']
    path = Path.load(os.path.join(d, con['path_file']))
    scene = G.BoxScene(os.path.join(REPO, meta['env_obj_rel']))
    prism = G.Prism(con['prism_width'], con['prism_length'])
    cz = np.load(os.path.join(d, 'cases.npz'))
    rec = dict(room_lo=scene.room_lo, room_hi=scene.room_hi, box_lo=scene.box_lo, box_hi=scene.box_hi,
               env=con['env'], prism_width=prism.w, prism_length=prism.h, L=con['L'], path_pts=path.pts,
               case=np.arange(len(cz['l0'])), l0=cz['l0'], rv0=cz['rv0'], l1=cz['l1'], rv1=cz['rv1'])
    for nm, pose in (('start', (cz['l0'], cz['rv0'])), ('goal', (cz['l1'], cz['rv1']))):
        l = torch.as_tensor(pose[0])
        rec[nm + '_clear'] = G.prism_clearance(scene, prism, path.pos(l), G.so3_exp(torch.as_tensor(pose[1]))).numpy()
    for m, run, planner, label in MODELS:
        base = os.path.join(RES, 'plan', '{}_{}'.format(code, run), '{}__{}_forward'.format(s, planner))
        if not (os.path.exists(base + '.json') and os.path.exists(base + '_paths.npz')):
            print('  missing', base)
            continue
        recs = {r['case']: r for r in json.load(open(base + '.json'))['records']}
        z = np.load(base + '_paths.npz')
        polys, cases = [], []
        for i, c in enumerate(z['case']):
            c = int(c)
            goal = np.concatenate([[cz['l1'][c]], cz['rv1'][c] / TWO_PI])
            polys.append(plan_poses(z['path'][i], int(z['iters3'][i]), goal))
            cases.append(c)
        Y = torch.as_tensor(np.concatenate(polys))
        lens = np.array([len(p) for p in polys])
        R = G.so3_exp(TWO_PI * Y[:, 1:])
        t = path.pos(Y[:, 0])
        clear = G.prism_clearance(scene, prism, t, R).numpy()
        # motion to the next waypoint of the same plan: l linear, R geodesic, 4 sub-steps
        nxt = np.arange(1, len(Y) + 1)
        ends = np.cumsum(lens) - 1
        nxt[ends] = ends
        nxt = torch.as_tensor(nxt)
        rel = G.so3_log(R.transpose(-1, -2) @ R[nxt])
        seg = torch.as_tensor(clear).clone()
        for u in (0.2, 0.4, 0.6, 0.8):
            Ru = R @ G.so3_exp(u * rel)
            lu = (1 - u) * Y[:, 0] + u * Y[nxt, 0]
            seg = torch.minimum(seg, G.prism_clearance(scene, prism, path.pos(lu), Ru))
        seg = torch.minimum(seg, torch.as_tensor(clear)[nxt])
        quat = G.quat_from_rotvec(TWO_PI * Y[:, 1:].numpy())
        rec[m + '_flat'] = np.concatenate([Y[:, :1].numpy(), t.numpy(), quat, clear[:, None]], 1).astype(np.float32)
        rec[m + '_seg'] = seg.numpy().astype(np.float32)
        rec[m + '_case'] = np.array(cases)
        rec[m + '_len'] = lens
        rec[m + '_label'] = label
        for k in ('success', 'collision', 'converged'):
            rec['{}_{}'.format(m, k)] = np.array([bool(recs[c][k + '_se3']) for c in cases])
        print('  {} {}: {} plans, success {:.1f} %'.format(s, m, len(cases), 100 * rec[m + '_success'].mean()), flush=True)
    np.savez_compressed(os.path.join(out, s + '.npz'), **rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(RES, 'viser'))
    ap.add_argument('--only', nargs='*', default=None, help='test set names')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(8)
    for code, sets in GROUPS:
        for s in sets:
            if a.only and s not in a.only:
                continue
            print(s, flush=True)
            export_set(code, s, a.out)


if __name__ == '__main__':
    main()
