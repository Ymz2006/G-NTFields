#!/usr/bin/env python3
"""Summary of the constrained-movement evaluations (numpy + json only; runs on the host).

Reads results/constrained_movement/plan/<scene>_ntf_s<seed>/<set>__base_<mode>.json, the test sets'
verification.json and the trainings' train_summary.json; writes results/constrained_movement/summary/summary.json
and summary.md: success / collision / stall rates per scene, set and mode (mean +- std over seeds), the trivial
straight motion of every case, plan statistics of the successes and where along the path the failures happen.

    python3 models/constrained_movement/scripts/aggregate_cmove.py [--plan plan_e2000 --out summary_e2000]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(HOST, 'results/constrained_movement')
SETS = os.path.join(HOST, 'testing_data/3dshape')
EXP = os.path.join(HOST, 'Experiments/constrained_movement')
SCENES = ('forest', 'city', 'corridor', 'corridor2', 'corridor3', 'env1', 'env4')


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def first_bad(paths, case, path_rows, L):
    """l / L of the last waypoint of a failed plan (where it stalled or was when it ended)."""
    i = np.nonzero(case == path_rows)[0]
    if not len(i):
        return None
    P = paths[i[0]]
    P = P[np.isfinite(P[:, 0])]
    return float(P[-1, 0] / L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', default='plan', help='results/constrained_movement/<plan>/<run>/*.json')
    ap.add_argument('--out', default='summary', help='results/constrained_movement/<out>/summary.{json,md}')
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(RES, a.plan, '*_ntf_s*', '*.json'))):
        m = re.match(r'(.+)_ntf_s(\d+)$', os.path.basename(os.path.dirname(f)))
        n = re.match(r'(cm_(.+)_a_(\w+))__base_(\w+)\.json$', os.path.basename(f))
        if not (m and n):
            continue
        J = load(f)
        if J is None:
            continue
        runs.setdefault((m.group(1), n.group(3), n.group(4)), {})[int(m.group(2))] = (f, J, n.group(1))
    out = {'cells': {}, 'train': {}, 'sets': {}}
    for d in sorted(glob.glob(os.path.join(EXP, '*_ntf_s*', 'train_summary.json'))):
        out['train'][os.path.basename(os.path.dirname(d))] = load(d)
    for s in sorted(glob.glob(os.path.join(SETS, 'cm_*_a_*', 'verification.json'))):
        V = load(s)
        meta = load(os.path.join(os.path.dirname(s), 'meta.json')) or {}
        out['sets'][os.path.basename(os.path.dirname(s))] = dict(V['summary'], L=meta.get('constraint', {}).get('L'))
    for (scene, variant, mode), by_seed in sorted(runs.items()):
        recs = {}
        for seed, (f, J, set_name) in sorted(by_seed.items()):
            S = J['summary']
            R = J['records']
            succ = [r for r in R if r['success_se3']]
            fails = [r for r in R if not r['success_se3']]
            pz = f[:-5] + '_paths.npz'
            L = out['sets'].get(set_name, {}).get('L') or 1.0
            where, where_c, where_s = [], [], []
            if fails and 'end_t' in fails[0]:
                # collisions: where the pyramid is first blocked; stalls: where the plan ended
                where_c = [r['first_block_t'] / L for r in fails if r['first_block_t'] is not None]
                where_s = [r['end_t'] / L for r in fails if r['first_block_t'] is None]
                where = where_c + where_s
            elif fails and os.path.exists(pz):
                z = np.load(pz)
                for r in fails:
                    w = first_bad(z['path'], r['case'], z['case'], L)
                    if w is not None:
                        where.append(w)
            recs[seed] = {
                'n': S['n_cases'], 'success': S['success_rate_se3'], 'collision': S['collision_rate_se3'],
                'stall': S['nonconv_rate_se3'], 'success_monotone': S['success_rate_se3_monotone'],
                'monotone': S['monotone_rate'], 'median_iters': S['median_iters_se3'],
                'median_rot_turns': S['median_rot_len_success'], 'median_min_clear': S['median_min_clearance_success'],
                'collision_and_stall': float(np.mean([r['collision_se3'] and not r['converged_se3'] for r in R])),
                'stall_clean': float(np.mean([not r['collision_se3'] and not r['converged_se3'] for r in R])),
                'fail_where_median': float(np.median(where)) if where else None,
                'fail_where_first_10pct': float(np.mean(np.asarray(where) < 0.1)) if where else None,
                'collision_where': [round(w, 4) for w in where_c], 'stall_where': [round(w, 4) for w in where_s],
                'seconds': S['seconds']['total'],
            }
        agg = {}
        for k in ('success', 'collision', 'stall', 'stall_clean', 'success_monotone', 'monotone'):
            v = np.array([r[k] for r in recs.values()], float)
            agg[k] = {'mean': float(v.mean()), 'std': float(v.std()), 'n_seeds': len(v)}
        out['cells']['{}|{}|{}'.format(scene, variant, mode)] = {'seeds': recs, 'agg': agg, 'set': set_name,
                                                               'steps': J['summary']['steps']}
    out['plan_dir'] = a.plan
    os.makedirs(os.path.join(RES, a.out), exist_ok=True)
    with open(os.path.join(RES, a.out, 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)

    def pct(cell, k):
        a = cell['agg'][k]
        return '{:.1f}'.format(100 * a['mean']) + (' +- {:.1f}'.format(100 * a['std']) if a['n_seeds'] > 1 else '')

    lines = ['# Constrained movement: metric NTFields (camera pyramid anchored at the followed bird)', '',
             'SE(3) success: converged to within 0.01 in sqrt(dt^2 + (theta / 2 pi)^2) and exact pyramid clearance '
             '>= 0 at every waypoint and 4 sub-steps between waypoints. "straight" = the direct motion (t linear, R '
             'slerp the short way) is collision-free. Rates in %, mean +- std over seeds.', '']
    for mode in ('forward', 'free'):
        lines += ['## {} mode'.format(mode), '',
                  '| scene | set | L | straight | NTFields success | collision | stall | seeds | median iters | '
                  'fail at l/L (median) |', '|---|---|---|---|---|---|---|---|---|---|']
        for scene in SCENES:
            for variant in ('full', 'seg'):
                c = out['cells'].get('{}|{}|{}'.format(scene, variant, mode))
                if c is None:
                    continue
                st = out['sets'].get(c['set'], {})
                it = [r['median_iters'] for r in c['seeds'].values() if r['median_iters'] is not None]
                fw = [r['fail_where_median'] for r in c['seeds'].values() if r['fail_where_median'] is not None]
                lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    scene, variant, '{:.2f}'.format(st['L']) if st.get('L') else '-',
                    '{:.1f}'.format(100 * st['direct_short_ok']) if st else '-', pct(c, 'success'),
                    pct(c, 'collision'), pct(c, 'stall'), c['agg']['success']['n_seeds'],
                    '{:.0f}'.format(np.median(it)) if it else '-', '{:.2f}'.format(np.median(fw)) if fw else '-'))
        lines.append('')
    with open(os.path.join(RES, a.out, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
