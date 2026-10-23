"""Export the constrained-movement plans for cmove_viser.py (numpy only on the viewer's side, no torch there).

One file per test set, results/constrained_movement/viser/<run>_<variant>.npz, run = forest, city, corridor2, env1,
env4 (corridor2 = the corridor model trained with pairs up to 2.75 apart, on the cm_corridor_a sets), variant = full
(the whole path) or seg (sub-paths):
    scene      env, run, room_lo, room_hi (3,), box_lo, box_hi, box_rgb (k, 3), L, path_pts (n, 3)
    bodies     shape_lo, shape_hi (parts, 3) around the shape's centre, pyr_W, pyr_H, pyr_D
    cases      case, l0, rv0, l1, rv1 (radians), start_clear, goal_clear, angle_deg, straight_ok
    per plan   <m>_case, <m>_len, <m>_flat (sum len, 9: t, centre xyz, quaternion wxyz, clearance),
               <m>_seg (sum len: smallest clearance from the waypoint to the next one, both included),
               <m>_success, <m>_collision, <m>_converged, <m>_first_block (t, nan if never blocked), <m>_label
Plans, all in the forward mode (the camera-follow query):
    ntf0, ntf1  metric NTFields seeds 0 and 1 as evaluated (plan_eval_cmove.py --save-paths): the MPPI waypoints up
                to the first SE(3)-converged one (all of them for a plan that never converges), the clearance at the
                waypoints and at the 4 chart-linear sub-steps of every motion, as the evaluator checks it
    rrt         the test set's verifying RRT-Connect on M (t never decreases), sampled at the generator's resolution
                0.003 and shown at every 5th sample (0.015 of M per waypoint, the MPPI step)
    line        the straight motion (t linear, R slerp the short way), sampled the same way; success = the test
                generator's check (verification.json direct_short)

    python models/constrained_movement/scripts/export_viser_cmove.py          (container, repo root, GPU optional)
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
from models.constrained_movement import geometry as G  # noqa: E402
from models.constrained_movement import render as RD  # noqa: E402

RES = os.path.join(REPO, 'results/constrained_movement')
SETS = os.path.join(REPO, 'testing_data/3dshape')
TWO_PI = 2 * math.pi
RUNS = [('forest', 'cm_forest_a'), ('city', 'cm_city_a'), ('corridor2', 'cm_corridor_a'), ('env1', 'cm_env1_a'),
        ('env4', 'cm_env4_a')]
VARIANTS = ('full', 'seg')
NTF = [('ntf0', 0, 'NTFields, seed 0'), ('ntf1', 1, 'NTFields, seed 1')]
FINE = 0.003          # make_cmove_tests.py --res
EVERY = 5             # shown waypoint every EVERY fine samples
SUB = 4               # plan_eval_cmove.py --substeps


class Exporter:
    def __init__(self, setdir, dev):
        self.dev = dev
        meta = json.load(open(os.path.join(setdir, 'meta.json')))
        con = meta['constraint']
        self.scene = G.scene_from_meta(meta, REPO)
        self.pyr = G.Pyramid(con['pyramid']['W'], con['pyramid']['H'], con['pyramid']['D'])
        self.path = G.Path.load(os.path.join(setdir, con['path_file']))
        self.shape = G.Shape(con['shape_parts_raw'], self.scene.env_scale)

    def clearance(self, t, R):
        """Exact pyramid clearance (N,) at arc lengths t (N,), rotations R (N, 3, 3), float32 as the evaluator."""
        out = []
        for i in range(0, len(t), 262144):
            tt = torch.as_tensor(t[i:i + 262144], dtype=torch.float32, device=self.dev)
            RR = torch.as_tensor(R[i:i + 262144], dtype=torch.float32, device=self.dev)
            out.append(G.pyramid_clearance(self.scene, self.pyr, self.path.pos(tt), RR).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def centres(self, t):
        return self.path.pos(torch.as_tensor(t, dtype=torch.float64)).numpy()

    def ntf(self, base):
        """Plans of one evaluation (base = .../<set>__base_forward): (fields, cases)."""
        recs = {r['case']: r for r in json.load(open(base + '.json'))['records']}
        z = np.load(base + '_paths.npz')
        polys, cases = [], []
        for i, c in enumerate(z['case']):
            Y = z['path'][i]
            Y = Y[~np.isnan(Y[:, 0])]
            k3 = int(z['iters3'][i])
            if k3 >= 0:
                Y = Y[:k3 + 2]
            polys.append(Y.astype(np.float32))
            cases.append(int(c))
        lens = np.array([len(p) for p in polys])
        Y = np.concatenate(polys)
        nxt = np.arange(1, len(Y) + 1)
        ends = np.cumsum(lens) - 1
        nxt[ends] = ends
        u = (np.arange(SUB + 1) / SUB).astype(np.float32)          # 0 = the waypoint, 1 = the next one
        Yi = Y[:, None, :] + (Y[nxt] - Y)[:, None, :] * u[None, :, None]
        Ri = G.so3_exp(TWO_PI * torch.as_tensor(Yi[..., 1:].reshape(-1, 3))).numpy()
        c = self.clearance(Yi[..., 0].ravel(), Ri).reshape(len(Y), SUB + 1)
        rec = dict(flat=np.concatenate([Y[:, :1], self.centres(Y[:, 0]), G.quat_from_rotvec(TWO_PI * Y[:, 1:].astype(
                       np.float64)), c[:, :1]], 1).astype(np.float32),
                   seg=c.min(1).astype(np.float32), case=np.array(cases), len=lens)
        rec['success'] = np.array([bool(recs[k]['success_se3']) for k in cases])
        rec['collision'] = np.array([bool(recs[k]['collision_se3']) for k in cases])
        rec['converged'] = np.array([bool(recs[k]['converged_se3']) for k in cases])
        rec['first_block'] = np.array([np.nan if recs[k].get('first_block_t') is None else recs[k]['first_block_t']
                                       for k in cases])
        starts = np.concatenate([[0], ends[:-1] + 1])
        mine = np.array([(rec['seg'][a:a + n] < 0).any() for a, n in zip(starts, lens)])
        print('    collision flags: {} of {} differ from the evaluator'.format(int((mine != rec['collision']).sum()),
                                                                             len(cases)), flush=True)
        return rec

    def sampled(self, polys, cases):
        """Plans given as node lists (k, 5) (t, wxyz) joined by t linear + slerp the short way: sampled at FINE,
        shown at every EVERY-th sample."""
        fines, shown = [], []
        for P in polys:
            out = [P[:1]]
            for a, b in zip(P[:-1], P[1:]):
                d = math.hypot(b[0] - a[0], float(G.quat_angle(a[1:], b[1:])) / TWO_PI)
                k = max(1, int(math.ceil(d / FINE)))
                u = np.arange(1, k + 1) / k
                out.append(np.concatenate([(a[0] + u * (b[0] - a[0]))[:, None], G.slerp(a[1:], b[1:], u)], 1))
            F = np.concatenate(out)
            idx = list(range(0, len(F), EVERY))
            if idx[-1] != len(F) - 1:
                idx.append(len(F) - 1)
            fines.append(F)
            shown.append(np.array(idx))
        F = np.concatenate(fines)
        c = self.clearance(F[:, 0], G.quat_matrix(F[:, 1:]))
        flat, seg, lens = [], [], []
        a = 0
        for Fk, idx in zip(fines, shown):
            ck = c[a:a + len(Fk)]
            a += len(Fk)
            flat.append(np.concatenate([Fk[idx, :1], self.centres(Fk[idx, 0]), Fk[idx, 1:], ck[idx, None]], 1))
            seg.append(np.array([ck[i:j + 1].min() for i, j in zip(idx[:-1], idx[1:])] + [ck[idx[-1]]]))
            lens.append(len(idx))
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        rec = dict(flat=np.concatenate(flat).astype(np.float32), seg=np.concatenate(seg).astype(np.float32),
                   case=np.array(cases), len=np.array(lens))
        S = rec['seg']
        rec['collision'] = np.array([(S[s:s + n] < 0).any() for s, n in zip(starts, lens)])
        rec['first_block'] = np.full(len(cases), np.nan)
        for i, (s, n) in enumerate(zip(starts, lens)):
            bad = np.nonzero(S[s:s + n] < 0)[0]
            if len(bad):
                rec['first_block'][i] = rec['flat'][s + bad[0], 0]
        rec['converged'] = np.ones(len(cases), bool)
        rec['success'] = ~rec['collision']
        return rec


def export_set(run, setname, out, dev):
    d = os.path.join(SETS, setname)
    ex = Exporter(d, dev)
    sc, pyr = ex.scene, ex.pyr
    cz = np.load(os.path.join(d, 'cases.npz'))
    ver = json.load(open(os.path.join(d, 'verification.json')))['cases']
    n = len(cz['l0'])
    q0, q1 = G.quat_from_rotvec(cz['rv0']), G.quat_from_rotvec(cz['rv1'])
    rec = dict(env=sc.name, run=run, room_lo=sc.room_lo, room_hi=sc.room_hi, box_lo=sc.box_lo, box_hi=sc.box_hi,
               box_rgb=np.asarray(RD.obstacle_colors(sc)), L=ex.path.L, path_pts=ex.path.pts,
               shape_lo=ex.shape.lo, shape_hi=ex.shape.hi, pyr_W=pyr.W, pyr_H=pyr.H, pyr_D=pyr.D,
               case=np.arange(n), l0=cz['l0'], rv0=cz['rv0'], l1=cz['l1'], rv1=cz['rv1'],
               start_clear=np.array([v['clear0'] for v in ver]), goal_clear=np.array([v['clear1'] for v in ver]),
               angle_deg=np.array([v['angle_deg'] for v in ver]),
               straight_ok=np.array([bool(v['direct_short']) for v in ver]))
    plans = {}
    for m, seed, label in NTF:
        base = os.path.join(RES, 'plan', '{}_ntf_s{}'.format(run, seed), '{}__base_forward'.format(setname))
        if os.path.exists(base + '.json') and os.path.exists(base + '_paths.npz'):
            print('  ' + m, flush=True)
            plans[m] = (ex.ntf(base), label)
    ends = np.cumsum(cz['rrt_len'])
    rrt = [cz['rrt_flat'][e - k:e] for e, k in zip(ends, cz['rrt_len'])]
    print('  rrt', flush=True)
    plans['rrt'] = (ex.sampled(rrt, list(range(n))), 'RRT reference (test set)')
    print('  line', flush=True)
    line = ex.sampled([np.stack([np.concatenate([[cz['l0'][k]], q0[k]]), np.concatenate([[cz['l1'][k]], q1[k]])])
                       for k in range(n)], list(range(n)))
    mine = ~line['collision']
    line['success'] = rec['straight_ok'].copy()
    line['collision'] = ~rec['straight_ok']
    print('    straight motion: {} of {} cases differ from the generator\'s check'.format(
        int((mine != rec['straight_ok']).sum()), n), flush=True)
    plans['line'] = (line, 'straight motion')
    for m, (p, label) in plans.items():
        for k, v in p.items():
            rec['{}_{}'.format(m, k)] = v
        rec[m + '_label'] = label
        print('  {:5s} {:4d} plans, success {:5.1f} %, waypoints {}'.format(
            m, len(p['case']), 100 * p['success'].mean(), len(p['flat'])), flush=True)
    rrt_min = plans['rrt'][0]['seg'].min()
    print('  rrt smallest clearance {:.4f}'.format(rrt_min), flush=True)
    np.savez_compressed(os.path.join(out, '{}_{}.npz'.format(run, setname.rsplit('_', 1)[1])), **rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(RES, 'viser'))
    ap.add_argument('--runs', default=','.join(r for r, _ in RUNS))
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(8)
    for run, cell in RUNS:
        if run not in a.runs.split(','):
            continue
        for v in VARIANTS:
            s = '{}_{}'.format(cell, v)
            print(run, s, flush=True)
            export_set(run, s, a.out, a.device)


if __name__ == '__main__':
    main()
