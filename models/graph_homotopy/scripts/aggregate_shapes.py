#!/usr/bin/env python3
"""Aggregate the third comparison (scripts/run_shapes_queue.py): the 8000-epoch homotopy head (base planner / dual
cover planner) against the original metric NTFields (8000 epochs and its epoch-4000 checkpoint) on

  shapes   tripod3d, crank3d, loop3d, twistI3d, chair3d x env4..env9 (standard 1000-pair sets), with the success by
           the number of partitions between start and goal (env5 / 7 / 8: one wall with a 50 / 60 / 70 window; env6 /
           env9: two crossing walls, 50 / 70 windows) and the old metric runs of G-Shape-NTFields-old
           (results/shapes3d, 4000 epochs, seed 1, Lie analysis/plan_eval.py, chart criterion) for reference;
  tla      T / L / A x env1 / env3 on the standard sets and the targeted variants long / bigrot / tight / hard.

    python3 models/graph_homotopy/scripts/aggregate_shapes.py

Reads results/graph_homotopy/plan_shapes/<run>/<set>__<planner>.json and writes
results/graph_homotopy/summary/shapes_summary.{md,json}. Seed means +- population std of SE(3) success /
collision / non-convergence; seed-matched paired McNemar tests (dual vs NTFields 8k, pooled over the seeds).
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

import numpy as np

SHAPES = ('tripod3d', 'crank3d', 'loop3d', 'twistI3d', 'chair3d')
ENVS = ('env4', 'env8', 'env7', 'env5', 'env9', 'env6')            # widening -> tighter, as results/shapes3d
ENV_NOTE = {'env4': 'open room', 'env8': '1 wall, 70 window', 'env7': '1 wall, 60 window', 'env5': '1 wall, 50 window',
            'env9': '2 crossing walls, 70 windows', 'env6': '2 crossing walls, 50 windows'}
TLA = {'T': 'Tshape3d', 'L': 'Lshape3d', 'A': 'Ashape3d'}
TLA_ENVS = ('env1', 'env3')
VARIANTS = ('std', 'long', 'bigrot', 'tight', 'hard')
# partition mid-planes (env units; dataprocessing/shapes3d_report.py)
PLANES = {'env5': [(2, -90.0)], 'env7': [(2, -90.0)], 'env8': [(2, -90.0)], 'env6': [(2, -90.0), (0, 100.0)],
          'env9': [(2, -90.0), (0, 100.0)]}
ANG_BINS = [(0, 60), (60, 120), (120, 150), (150, 180.01)]
OLD = '/media/jeffrey/G-Shape-NTFields-old/results/shapes3d/summary.json'
GIBSON = ('cube_eastville', 'rect_small_eastville')
OLD_GIBSON = '/media/jeffrey/G-Shape-NTFields-old/results/gibson_metric/plan/{}_metric_s{}.json'   # seeds 1-6, 5000 epochs
MODELS = [('dual', 'hminL', 'dual', 'success_rate_se3'), ('base', 'hminL', 'base', 'success_rate_se3'),
          ('ntf8k', 'mb', 'base', 'success_rate_se3'), ('ntf4k', 'mb4k', 'base', 'success_rate_se3')]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def ms(vals, d=1, n=True):
    v = [x for x in vals if x is not None]
    if not v:
        return '-'
    if len(v) == 1:
        return '{:.{}f}'.format(100 * v[0], d) + (' (1)' if n else '')
    return '{:.{}f} +- {:.{}f}'.format(100 * statistics.mean(v), d, 100 * statistics.pstdev(v), d) + \
        (' ({})'.format(len(v)) if n else '')


def mean(vals):
    v = [x for x in vals if x is not None]
    return statistics.mean(v) if v else None


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (1 << n)
    return min(1.0, 2.0 * p)


def fmt_p(p):
    return '{:.0e}'.format(p) if p < 1e-3 else '{:.3f}'.format(p)


def rel_angle_deg(x):
    """Relative rotation (deg) of every pair of a test set; chart rotation = rotvec / 2 pi."""
    def quat(r):
        v = 2 * math.pi * r
        th = np.linalg.norm(v, axis=1, keepdims=True)
        ax = np.where(th > 1e-12, v / np.maximum(th, 1e-12), 0.0)
        return np.concatenate([np.cos(th / 2), ax * np.sin(th / 2)], 1)
    q0, q1 = quat(x[:, 3:6].astype(np.float64)), quat(x[:, 9:12].astype(np.float64))
    return np.degrees(2 * np.arccos(np.clip(np.abs((q0 * q1).sum(1)), 0, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/graph_homotopy/plan_shapes')
    ap.add_argument('--sets', default='testing_data/3dshape')
    ap.add_argument('--out', default='results/graph_homotopy/summary')
    a = ap.parse_args()

    # R[(code, model, seed, set, planner)] = (summary, records)
    R = {}
    for f in glob.glob(os.path.join(a.res, '*', '*__*.json')):
        m = re.match(r'^(.+)_(hminL|hlift|mb|mb4k|mbtd|mbtd4k|mbtd5k|oldntf)_s(\d+)$', os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        setname, planner = os.path.basename(f)[:-5].split('__')
        d = load(f)
        if d and 'summary' in d:
            R[(m.group(1), m.group(2), int(m.group(3)), setname, planner)] = (d['summary'], d['records'])

    def runs(code, model, setname, planner):
        return sorted((k[2], v) for k, v in R.items() if k[0] == code and k[1] == model and k[3] == setname and k[4] == planner)

    def vals(code, model, setname, planner, key):
        return [s.get(key) for _, (s, _) in runs(code, model, setname, planner)]

    def paired(code, setname, ref='mb', model='hminL', ref_planner='base'):
        """model's dual vs NTFields 8k (run tag ref; or any run / planner), seed-matched: (wins, losses, p)."""
        dual = {s: r for s, (_, r) in runs(code, model, setname, 'dual')}
        base = {s: r for s, (_, r) in runs(code, ref, setname, ref_planner)}
        w = l = 0
        for s in set(dual) & set(base):
            bd = {r['case']: r['success_se3'] for r in base[s]}
            for r in dual[s]:
                if r['case'] in bd:
                    w += r['success_se3'] and not bd[r['case']]
                    l += bd[r['case']] and not r['success_se3']
        return w, l, binom_two_sided(min(w, l), w + l) if w + l else None

    out = {'shapes': {}, 'tla': {}}
    old = (load(OLD) or {}).get('cells', {})
    L = ['# Homotopy head vs the original metric NTFields: new shapes x env4-9, T / L / A x env1 / env3', '',
         'SE(3) success % (converged to within 0.01 in the product metric without a collision; mean +- std over seeds, '
         'number of seeds in parentheses). homotopy = graph_homotopy, 8000 epochs (`--lift min`); dual = the cover '
         'planner (both lifted goals planned, converged plan kept); base = the official MPPI on the base field. '
         'NTFields = the original metric NTFields (Lie-Shape-NTFields metric_baseline, alpha 1.0) at 8000 epochs and '
         'its epoch-4000 checkpoint. Same evaluator for all (`plan_eval_homotopy.py`: forward MPPI, 50 samples, horizon 5, '
         'step 0.015, temperature 50, 200 iterations, 1000 cases, collisions against 50k environment surface points). '
         'Paired: cases the dual planner solves and NTFields 8k does not / the reverse, seed-matched, exact McNemar p.', '']

    # ------------------------------------------------------------------ shapes grid
    L += ['## 1. New shapes x env4-9 (standard sets)', '',
          '| cell | env | homotopy dual | homotopy base | NTFields 8k | NTFields 4k | old NTFields 4k (chart) '
          '| coll. dual / NTF 8k | non-conv. dual / NTF 8k | path ratio dual / NTF 8k | paired | p |',
          '|' + '---|' * 12]
    piv = {}
    for env in ENVS:
        for shape in SHAPES:
            cell = '{}_{}'.format(shape, env)
            row = {k: vals(cell, mdl, cell, pl, key) for k, mdl, pl, key in MODELS}
            row['coll_dual'] = vals(cell, 'hminL', cell, 'dual', 'collision_rate_se3')
            row['coll_ntf'] = vals(cell, 'mb', cell, 'base', 'collision_rate_se3')
            row['nc_dual'] = vals(cell, 'hminL', cell, 'dual', 'nonconv_rate_se3')
            row['nc_ntf'] = vals(cell, 'mb', cell, 'base', 'nonconv_rate_se3')
            row['pr_dual'] = vals(cell, 'hminL', cell, 'dual', 'median_trans_ratio_success_se3')
            row['pr_ntf'] = vals(cell, 'mb', cell, 'base', 'median_trans_ratio_success_se3')
            o = old.get(cell, {}).get('plan')
            row['old_ntf4k_chart'] = o.get('success_rate') if o else None
            w, l, p = paired(cell, cell)
            row['paired'] = [w, l, p]
            out['shapes'][cell] = row
            piv[cell] = row
            pr = lambda v: '{:.2f}'.format(mean(v)) if mean(v) else '-'  # noqa: E731
            L.append('| {} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | {} / {} | +{} / -{} | {} |'.format(
                shape, env, ms(row['dual']), ms(row['base'], n=False), ms(row['ntf8k']), ms(row['ntf4k'], n=False),
                '-' if row['old_ntf4k_chart'] is None else '{:.1f}'.format(100 * row['old_ntf4k_chart']),
                ms(row['coll_dual'], n=False), ms(row['coll_ntf'], n=False),
                ms(row['nc_dual'], n=False), ms(row['nc_ntf'], n=False), pr(row['pr_dual']), pr(row['pr_ntf']),
                w, l, '-' if p is None else fmt_p(p)))

    L += ['', '### Success by shape and environment (homotopy dual / NTFields 8k, seed means)', '',
          '| env | ' + ' | '.join(SHAPES) + ' | mean |', '|' + '---|' * (len(SHAPES) + 2)]
    for env in ENVS:
        cells_ = []
        d_all, n_all = [], []
        for shape in SHAPES:
            r = piv['{}_{}'.format(shape, env)]
            d, n = mean(r['dual']), mean(r['ntf8k'])
            d_all.append(d)
            n_all.append(n)
            cells_.append('-' if d is None or n is None else '**{:.1f}** / {:.1f}'.format(100 * d, 100 * n))
        dm, nm = mean(d_all), mean(n_all)
        L.append('| {} ({}) | {} | {} |'.format(env, ENV_NOTE[env], ' | '.join(cells_),
                                                '-' if dm is None or nm is None else '**{:.1f}** / {:.1f}'.format(100 * dm, 100 * nm)))

    # partitions crossed
    L += ['', '### By partitions between start and goal (pooled over shapes and seeds)', '',
          'success % (n cases) for 0, 1, 2 partitions crossed; only (cell, seed) pairs that both the homotopy head and '
          'NTFields 8k have, so the rows are matched.', '',
          '| env | model | 0 | 1 | 2 |', '|---|---|---|---|---|']
    out['partitions'] = {}
    for env in ENVS:
        if env not in PLANES:
            continue
        acc = {'dual': {}, 'ntf8k': {}, 'base': {}}
        for shape in SHAPES:
            cell = '{}_{}'.format(shape, env)
            sp = os.path.join(a.sets, cell)
            if not os.path.exists(os.path.join(sp, 'sampled_points.npy')):
                continue
            x = np.load(os.path.join(sp, 'sampled_points.npy'))
            meta = load(os.path.join(sp, 'meta.json'))
            c, sc = np.array(meta['env_center']), meta['env_scale']
            k = np.zeros(len(x), dtype=int)
            for axis, v in PLANES[env]:
                pl = (v - c[axis]) / sc
                k += (x[:, axis] > pl) != (x[:, 6 + axis] > pl)
            seeds = set(s for s, _ in runs(cell, 'hminL', cell, 'dual')) & set(s for s, _ in runs(cell, 'mb', cell, 'base'))
            for key, mdl, pln in (('dual', 'hminL', 'dual'), ('base', 'hminL', 'base'), ('ntf8k', 'mb', 'base')):
                for s, (_, recs) in runs(cell, mdl, cell, pln):
                    if s not in seeds:
                        continue
                    for r in recs:
                        acc[key].setdefault(int(k[r['case']]), []).append(r['success_se3'])
        out['partitions'][env] = {m: {kk: [len(v), float(np.mean(v))] for kk, v in d.items()} for m, d in acc.items()}
        for key, label in (('dual', 'homotopy dual'), ('base', 'homotopy base'), ('ntf8k', 'NTFields 8k')):
            row = []
            for kk in range(len(PLANES[env]) + 1):
                v = acc[key].get(kk)
                row.append('{:.1f} ({})'.format(100 * np.mean(v), len(v)) if v else '-')
            row += ['-'] * (3 - len(row))
            L.append('| {} | {} | {} |'.format(env, label, ' | '.join(row)))

    # relative rotation bins
    L += ['', '### By relative rotation (pooled over shape cells and seeds; matched (cell, seed) pairs only)', '',
          '| rotation | homotopy dual | homotopy base | NTFields 8k | n |', '|---|---|---|---|---|']
    ang_acc = {m: [[] for _ in ANG_BINS] for m in ('dual', 'base', 'ntf8k')}
    for env in ENVS:
        for shape in SHAPES:
            cell = '{}_{}'.format(shape, env)
            sp = os.path.join(a.sets, cell, 'sampled_points.npy')
            if not os.path.exists(sp):
                continue
            ang = rel_angle_deg(np.load(sp))
            seeds = set(s for s, _ in runs(cell, 'hminL', cell, 'dual')) & set(s for s, _ in runs(cell, 'mb', cell, 'base'))
            for key, mdl, pln in (('dual', 'hminL', 'dual'), ('base', 'hminL', 'base'), ('ntf8k', 'mb', 'base')):
                for s, (_, recs) in runs(cell, mdl, cell, pln):
                    if s not in seeds:
                        continue
                    for r in recs:
                        for bi, (lo, hi) in enumerate(ANG_BINS):
                            if lo <= ang[r['case']] < hi:
                                ang_acc[key][bi].append(r['success_se3'])
    for bi, (lo, hi) in enumerate(ANG_BINS):
        f = lambda m: '{:.1f}'.format(100 * np.mean(ang_acc[m][bi])) if ang_acc[m][bi] else '-'  # noqa: E731
        L.append('| {}-{} deg | {} | {} | {} | {} |'.format(lo, int(hi), f('dual'), f('base'), f('ntf8k'), len(ang_acc['ntf8k'][bi])))

    # ------------------------------------------------------------------ T / L / A grid
    L += ['', '## 2. T / L / A x env1 / env3 (standard and targeted sets)', '',
          'Sets: std = the standard 1000-case set (Lie-Shape-NTFields); long (|dp| >= 0.6), bigrot (relative rotation >= '
          '150 deg), tight (endpoint clearance 0.005-0.02), hard (clearance 0.005-0.02, |dp| >= 0.5, rotation >= 135 deg): '
          '1000 RRT-verified cases each (`make_test_sets.py`).', '',
          '| scene | set | homotopy dual | homotopy base | NTFields 8k | NTFields 4k | coll. dual / NTF 8k | non-conv. dual / NTF 8k | paired | p |',
          '|' + '---|' * 10]
    for env in TLA_ENVS:
        for k, shape in TLA.items():
            code, cell = '{}{}'.format(k, env[3:]), '{}_{}'.format(shape, env)
            for v in VARIANTS:
                sn = cell if v == 'std' else '{}_{}'.format(cell, v)
                row = {kk: vals(code, mdl, sn, pl, key) for kk, mdl, pl, key in MODELS}
                row['coll_dual'] = vals(code, 'hminL', sn, 'dual', 'collision_rate_se3')
                row['coll_ntf'] = vals(code, 'mb', sn, 'base', 'collision_rate_se3')
                row['nc_dual'] = vals(code, 'hminL', sn, 'dual', 'nonconv_rate_se3')
                row['nc_ntf'] = vals(code, 'mb', sn, 'base', 'nonconv_rate_se3')
                w, l, p = paired(code, sn)
                row['paired'] = [w, l, p]
                out['tla'][sn] = row
                L.append('| {} | {} | {} | {} | {} | {} | {} / {} | {} / {} | +{} / -{} | {} |'.format(
                    cell, v, ms(row['dual']), ms(row['base'], n=False), ms(row['ntf8k']), ms(row['ntf4k'], n=False),
                    ms(row['coll_dual'], n=False), ms(row['coll_ntf'], n=False), ms(row['nc_dual'], n=False),
                    ms(row['nc_ntf'], n=False), w, l, '-' if p is None else fmt_p(p)))

    # ------------------------------------------------------------------ overall
    def pooled(rows):
        d = [mean(r['dual']) for r in rows if mean(r['dual']) is not None and mean(r['ntf8k']) is not None]
        n = [mean(r['ntf8k']) for r in rows if mean(r['dual']) is not None and mean(r['ntf8k']) is not None]
        return (statistics.mean(d) if d else None, statistics.mean(n) if n else None, len(d))
    L += ['', '## 3. Overall (mean of the cell means)', '', '| group | cells | homotopy dual | NTFields 8k |', '|---|---|---|---|']
    groups = [('new shapes, all envs', list(out['shapes'].values()))]
    groups += [('new shapes, ' + e, [out['shapes']['{}_{}'.format(s, e)] for s in SHAPES]) for e in ENVS]
    groups += [('T / L / A, ' + v, [out['tla'][('{}_{}'.format(TLA[k], e)) + ('' if v == 'std' else '_' + v)]
                                    for e in TLA_ENVS for k in TLA]) for v in VARIANTS]
    out['overall'] = {}
    for g, rows in groups:
        d, n, c = pooled(rows)
        out['overall'][g] = [d, n, c]
        L.append('| {} | {} | {} | {} |'.format(g, c, '-' if d is None else '**{:.1f}**'.format(100 * d),
                                               '-' if n is None else '{:.1f}'.format(100 * n)))
    # ------------------------------------------------------------------ Gibson
    if any(k[0] in GIBSON for k in R):
        L += ['', '## 4. Gibson scene eastville (cube, small rectangle; standard sets, 6 seeds)', '',
              'Two NTFields configurations. NTF = the bundle used everywhere above (Lie-Shape-NTFields metric_baseline: '
              'TD and normal-alignment terms off), as the old shapes runs. NTF+TD = the original metric NTFields with the '
              'TD and normal-alignment terms on (1e-3 each; G-Shape-NTFields-old models/metric), as the old Gibson runs '
              '(results/gibson_metric). old NTF+TD 5k = those runs\' own checkpoints (5000 epochs, seeds 1-6) scored here; '
              'old (chart) = their scores from the old evaluator (Lie analysis/plan_eval.py).', '',
              '| cell | homotopy dual | homotopy base | NTF+TD 8k | NTF+TD 5k | old NTF+TD 5k | old NTF+TD 5k (chart) | '
              'NTF 8k | NTF 4k | paired dual / NTF+TD 8k | p | paired dual / NTF 8k | p |', '|' + '---|' * 13]
        L2 = ['', '| cell | coll. dual / NTF+TD 8k / NTF 8k | non-conv. dual / NTF+TD 8k / NTF 8k | '
              'path ratio dual / NTF+TD 8k / NTF 8k |', '|---|---|---|---|']
        out['gibson'] = {}
        for cell in GIBSON:
            row = {k: vals(cell, mdl, cell, pl, key) for k, mdl, pl, key in MODELS}
            for k, mdl in (('td8k', 'mbtd'), ('td5k', 'mbtd5k'), ('old5k', 'oldntf')):
                row[k] = vals(cell, mdl, cell, 'base', 'success_rate_se3')
            for k, key in (('coll', 'collision_rate_se3'), ('nc', 'nonconv_rate_se3'), ('pr', 'median_trans_ratio_success_se3')):
                for m, mdl, pl in (('dual', 'hminL', 'dual'), ('td', 'mbtd', 'base'), ('ntf', 'mb', 'base')):
                    row['{}_{}'.format(k, m)] = vals(cell, mdl, cell, pl, key)
            row['old5k_chart'] = [(load(OLD_GIBSON.format(cell, sd)) or {}).get('summary', {}).get('success_rate')
                                  for sd in range(1, 7)]
            w, l, p = paired(cell, cell)
            wt, lt, pt = paired(cell, cell, 'mbtd')
            row['paired'], row['paired_td'] = [w, l, p], [wt, lt, pt]
            out['gibson'][cell] = row
            pr = lambda v: '{:.2f}'.format(mean(v)) if mean(v) else '-'  # noqa: E731
            fp = lambda q: '-' if q is None else fmt_p(q)  # noqa: E731
            L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | +{} / -{} | {} | +{} / -{} | {} |'.format(
                cell, ms(row['dual']), ms(row['base'], n=False), ms(row['td8k']), ms(row['td5k'], n=False),
                ms(row['old5k']), ms(row['old5k_chart']), ms(row['ntf8k']), ms(row['ntf4k'], n=False),
                wt, lt, fp(pt), w, l, fp(p)))
            L2.append('| {} | {} / {} / {} | {} / {} / {} | {} / {} / {} |'.format(
                cell, *[ms(row['{}_{}'.format(k, m)], n=False) for k in ('coll', 'nc') for m in ('dual', 'td', 'ntf')],
                *[pr(row['pr_' + m]) for m in ('dual', 'td', 'ntf')]))
        L += L2
    # ------------------------------------------------------------------ full NTFields (TD + normal) on shapes / T L A
    if any(k[1] == 'mbtd' and k[0] not in GIBSON for k in R):
        L += ['', '## 5. Homotopy head vs the full metric NTFields (TD and normal-alignment terms on)', '',
              'NTF+TD = metric NTFields with the TD and normal-alignment terms at 1e-3 (G-Shape-NTFields-old models/metric, '
              'as its Gibson runs), 8000 epochs and its epoch-4000 checkpoint; NTF = the bundle of sections 1-3 (both off). '
              'Paired: dual vs NTF+TD 8k, seed-matched.', '',
              '| cell / set | homotopy dual | NTF+TD 8k | NTF+TD 4k | NTF 8k | coll. dual / NTF+TD 8k | paired | p |',
              '|---|---|---|---|---|---|---|---|']
        full = {}
        items = [('{}_{}'.format(sh, e), '{}_{}'.format(sh, e), 'shapes ' + e) for e in ENVS for sh in SHAPES]
        items += [('{}{}'.format(k, e[3:]), '{}_{}'.format(TLA[k], e) + ('' if v == 'std' else '_' + v), 'tla ' + v)
                  for e in TLA_ENVS for k in TLA for v in VARIANTS]
        for code, sn, grp in items:
            row = {'dual': vals(code, 'hminL', sn, 'dual', 'success_rate_se3'),
                   'td8k': vals(code, 'mbtd', sn, 'base', 'success_rate_se3'),
                   'td4k': vals(code, 'mbtd4k', sn, 'base', 'success_rate_se3'),
                   'ntf8k': vals(code, 'mb', sn, 'base', 'success_rate_se3'),
                   'coll_dual': vals(code, 'hminL', sn, 'dual', 'collision_rate_se3'),
                   'coll_td': vals(code, 'mbtd', sn, 'base', 'collision_rate_se3')}
            w, l, p = paired(code, sn, 'mbtd')
            row['paired_td'], row['group'] = [w, l, p], grp
            full[sn] = row
            L.append('| {} | {} | {} | {} | {} | {} / {} | +{} / -{} | {} |'.format(
                sn, ms(row['dual']), ms(row['td8k']), ms(row['td4k'], n=False), ms(row['ntf8k']),
                ms(row['coll_dual'], n=False), ms(row['coll_td'], n=False), w, l, '-' if p is None else fmt_p(p)))
        L += ['', '| group | cells | homotopy dual | NTF+TD 8k | NTF 8k | cells dual ahead / behind (p < 0.01) |',
              '|---|---|---|---|---|---|']
        groups = [('shapes, all envs', lambda g: g.startswith('shapes'))]
        groups += [('shapes, ' + e, (lambda g, e=e: g == 'shapes ' + e)) for e in ENVS]
        groups += [('T / L / A, ' + v, (lambda g, v=v: g == 'tla ' + v)) for v in VARIANTS]
        out['full_ntf'] = {'cells': full, 'groups': {}}
        for gname, f in groups:
            rows = [r for r in full.values() if f(r['group']) and mean(r['dual']) is not None and mean(r['td8k']) is not None]
            if not rows:
                continue
            d = statistics.mean(mean(r['dual']) for r in rows)
            t = statistics.mean(mean(r['td8k']) for r in rows)
            nn = [mean(r['ntf8k']) for r in rows if mean(r['ntf8k']) is not None]
            ahead = sum(1 for r in rows if r['paired_td'][2] is not None and r['paired_td'][2] < 0.01 and r['paired_td'][0] > r['paired_td'][1])
            behind = sum(1 for r in rows if r['paired_td'][2] is not None and r['paired_td'][2] < 0.01 and r['paired_td'][0] < r['paired_td'][1])
            out['full_ntf']['groups'][gname] = [d, t, len(rows), ahead, behind]
            L.append('| {} | {} | {:.1f} | {:.1f} | {} | {} / {} |'.format(
                gname, len(rows), 100 * d, 100 * t, '{:.1f}'.format(100 * statistics.mean(nn)) if nn else '-', ahead, behind))
    # ------------------------------------------------------------------ lifted encoder (hlift) vs the homotopy head
    if any(k[1] == 'hlift' for k in R):
        L += ['', '## 6. Lifted-encoder homotopy head (graph_homotopy_lift) vs the homotopy head', '',
              'lifted = the homotopy head whose encoder reads the lifted pose (t, q) on the double cover, both lifts '
              '+-q of every pose, split into the q-even part (E, rho, lam) and the q-odd part (Q) instead of encoding '
              'G(x) = (t, vec R) and forming Q = sum_i q_i W_i e; same heads, sheet formula, loss and training. '
              'Paired: lifted dual vs homotopy dual, and lifted dual vs NTF+TD 8k, seed-matched.', '',
              '| cell / set | lifted dual | lifted base | homotopy dual | homotopy base | NTF+TD 8k | NTF 8k | '
              'coll. lifted / homotopy | paired vs homotopy | p | paired vs NTF+TD | p |', '|' + '---|' * 12]
        lift = {}
        items = [('{}_{}'.format(sh, e), '{}_{}'.format(sh, e), 'shapes ' + e) for e in ENVS for sh in SHAPES]
        items += [('{}{}'.format(k, e[3:]), '{}_{}'.format(TLA[k], e) + ('' if v == 'std' else '_' + v), 'tla ' + v)
                  for e in TLA_ENVS for k in TLA for v in VARIANTS]
        items += [(c, c, 'gibson') for c in GIBSON]
        fp = lambda q: '-' if q is None else fmt_p(q)  # noqa: E731
        for code, sn, grp in items:
            row = {'lift_dual': vals(code, 'hlift', sn, 'dual', 'success_rate_se3'),
                   'lift_base': vals(code, 'hlift', sn, 'base', 'success_rate_se3'),
                   'dual': vals(code, 'hminL', sn, 'dual', 'success_rate_se3'),
                   'base': vals(code, 'hminL', sn, 'base', 'success_rate_se3'),
                   'td8k': vals(code, 'mbtd', sn, 'base', 'success_rate_se3'),
                   'ntf8k': vals(code, 'mb', sn, 'base', 'success_rate_se3'),
                   'coll_lift': vals(code, 'hlift', sn, 'dual', 'collision_rate_se3'),
                   'coll_dual': vals(code, 'hminL', sn, 'dual', 'collision_rate_se3'),
                   'nc_lift': vals(code, 'hlift', sn, 'dual', 'nonconv_rate_se3'),
                   'nc_dual': vals(code, 'hminL', sn, 'dual', 'nonconv_rate_se3'),
                   'group': grp}
            row['paired_hom'] = list(paired(code, sn, 'hminL', 'hlift', 'dual'))
            row['paired_td'] = list(paired(code, sn, 'mbtd', 'hlift'))
            lift[sn] = row
            if not row['lift_dual']:
                continue
            (w, l, p), (wt, lt, pt) = row['paired_hom'], row['paired_td']
            L.append('| {} | {} | {} | {} | {} | {} | {} | {} / {} | +{} / -{} | {} | +{} / -{} | {} |'.format(
                sn, ms(row['lift_dual']), ms(row['lift_base'], n=False), ms(row['dual']), ms(row['base'], n=False),
                ms(row['td8k']), ms(row['ntf8k']), ms(row['coll_lift'], n=False), ms(row['coll_dual'], n=False),
                w, l, fp(p), wt, lt, fp(pt)))
        L += ['', '| group | cells | lifted dual | homotopy dual | NTF+TD 8k | NTF 8k | lifted vs homotopy ahead / '
              'behind (p < 0.01) | lifted vs NTF+TD ahead / behind |', '|---|---|---|---|---|---|---|---|']
        groups = [('shapes + T / L / A, all sets', lambda g: g != 'gibson'), ('shapes, all envs', lambda g: g.startswith('shapes'))]
        groups += [('shapes, ' + e, (lambda g, e=e: g == 'shapes ' + e)) for e in ENVS]
        groups += [('T / L / A, ' + v, (lambda g, v=v: g == 'tla ' + v)) for v in VARIANTS]
        groups += [('Gibson eastville', lambda g: g == 'gibson')]
        out['lift'] = {'cells': lift, 'groups': {}}

        def tally(rows, key):
            sig = [r[key] for r in rows if r[key][2] is not None and r[key][2] < 0.01]
            return sum(1 for w, l, _ in sig if w > l), sum(1 for w, l, _ in sig if w < l)
        for gname, f in groups:
            rows = [r for r in lift.values() if f(r['group']) and all(mean(r[k]) is not None for k in ('lift_dual', 'dual', 'td8k'))]
            if not rows:
                continue
            gm = {k: statistics.mean(mean(r[k]) for r in rows) for k in ('lift_dual', 'dual', 'td8k')}
            nn = [mean(r['ntf8k']) for r in rows if mean(r['ntf8k']) is not None]
            gm['ntf8k'] = statistics.mean(nn) if nn else None
            ah, bh = tally(rows, 'paired_hom')
            at, bt = tally(rows, 'paired_td')
            out['lift']['groups'][gname] = dict(gm, cells=len(rows), ahead_hom=ah, behind_hom=bh, ahead_td=at, behind_td=bt)
            L.append('| {} | {} | **{:.1f}** | {:.1f} | {:.1f} | {} | {} / {} | {} / {} |'.format(
                gname, len(rows), 100 * gm['lift_dual'], 100 * gm['dual'], 100 * gm['td8k'],
                '-' if gm['ntf8k'] is None else '{:.1f}'.format(100 * gm['ntf8k']), ah, bh, at, bt))
    L += ['', 'Evaluations read: {}.'.format(len(R))]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'shapes_summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    with open(os.path.join(a.out, 'shapes_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
