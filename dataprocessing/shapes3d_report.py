"""Summarize the shapes3d pipeline: data, training and metric-baseline planning per cell.

Reads datasets/3dshape/data/<cell>/, testing_data/3dshape/<cell>/,
Experiments/shapes3d_metric_baseline/<cell>_s1/ and results/shapes3d/plan/,
plus the T/L/A metric-baseline runs of Lie-Shape-NTFields (same evaluator, same
4000-epoch recipe) for reference.  Writes results/shapes3d/summary.{md,json}.

Also the moving-sofa cells (make_sofa2d.py, planar --2d pipeline), where
"crossing" means start and goal in different legs of the corridor.

    python3 dataprocessing/shapes3d_report.py
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIE_PLAN = '/media/jeffrey/Lie-Shape-NTFields/results/compare/plan'
OUT = os.path.join(ROOT, 'results', 'shapes3d')
SHAPES = ['tripod3d', 'crank3d', 'loop3d', 'twistI3d', 'chair3d']
ENVS = ['env4', 'env8', 'env7', 'env5', 'env9', 'env6']     # widening -> tighter
SOFA_CELLS = ['Usofa2d_2dhall44', 'Usofa2d_2dhall34', 'Usofa2d_2dhall30']     # easy -> hard
SOFA_NOTE = {'2dhall44': 'W 44, easy: turns either way round (clearance 8.9 notch-in / 5.5 out)',
             '2dhall34': 'W 34, medium: turns only notch-in (clearance 3.45)',
             '2dhall30': 'W 30, hard: turns only notch-in (clearance 1.41)'}
REFS = [('Tshape3d_env1', 'metric_s{}'), ('Lshape3d_env3', 'Lshape3d_env3_metric_s{}'),
        ('Ashape3d_env4', 'Ashape3d_env4_metric_s{}')]
# Partition mid-planes of the tight envs (env units, as in make_shapes3d.py): a test
# pair whose endpoints lie on opposite sides of one has to pass through a window.
PLANES = {'env5': [(2, -90.0)], 'env7': [(2, -90.0)], 'env8': [(2, -90.0)], 'env6': [(2, -90.0), (0, 100.0)],
          'env9': [(2, -90.0), (0, 100.0)]}


def plan_summary(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))['summary']


def final_loss(run_dir):
    cks = sorted(glob.glob(os.path.join(run_dir, 'Model_Epoch_*_ValLoss_*.pt')))
    if not cks:
        return None, None
    m = re.search(r'Model_Epoch_(\d+)_ValLoss_([0-9.e+-]+)\.pt', os.path.basename(cks[-1]))
    return int(m.group(1)), float(m.group(2))


def data_stats(cell):
    d = os.path.join(ROOT, 'datasets', '3dshape', 'data', cell)
    if not os.path.exists(os.path.join(d, 'speed.npy')):
        return None
    sd = np.load(os.path.join(d, 'speed_dists.npy'), mmap_mode='r')
    sp = np.load(os.path.join(d, 'speed.npy'), mmap_mode='r')
    meta = json.load(open(os.path.join(d, 'meta.json')))
    return {'pairs': int(len(sp)), 'num_tets': meta['num_tets'],
            'speed_dist_x0_mean': float(np.mean(sd[:, 0])), 'speed_dist_x1_mean': float(np.mean(sd[:, 1])),
            'frac_x1_saturated': float(np.mean(np.asarray(sd[:, 1]) >= 1.0)), 'shards': meta.get('shards')}


def crossings(env, x, meta):
    """(per-pair crossings, max) for test pairs x (N, 12): partitions between start and goal,
    or for a sofa hall whether start and goal lie in different legs (sides of the diagonal x = y)."""
    c, sc = np.array(meta['env_center']), meta['env_scale']
    if env.startswith('2d'):
        w0, w1 = c[:2] + sc * x[:, 0:2], c[:2] + sc * x[:, 6:8]
        return ((w0[:, 0] > w0[:, 1]) != (w1[:, 0] > w1[:, 1])).astype(int), 1
    n = np.zeros(len(x), dtype=int)
    for axis, v in PLANES[env]:
        p = (v - c[axis]) / sc
        n += (x[:, axis] > p) != (x[:, 6 + axis] > p)
    return n, len(PLANES[env])


def crossing_breakdown(cell, plan_path):
    """Success / collision / not-converged by the number of partitions (legs) between start and goal."""
    env = cell.split('_')[1]
    if (env not in PLANES and not env.startswith('2d')) or not os.path.exists(plan_path):
        return None
    d = os.path.join(ROOT, 'testing_data', '3dshape', cell)
    x = np.load(os.path.join(d, 'sampled_points.npy'))
    n, kmax = crossings(env, x, json.load(open(os.path.join(d, 'meta.json'))))
    out = {}
    recs = json.load(open(plan_path))['records']
    for k in range(kmax + 1):
        sel = [r for r in recs if n[r['case']] == k]
        if sel:
            out[k] = {'n': len(sel), 'success_rate': float(np.mean([r['success'] for r in sel])),
                      'collision_rate': float(np.mean([r['collision'] for r in sel])),
                      'nonconv_rate': float(np.mean([not r['converged'] for r in sel]))}
    return out


def seam_breakdown(cell, plan_path):
    """Planar cells: pairs whose rz differ by more than half a turn in the chart (a chart-space
    planner has to spin the long way round, through the +-pi seam) vs the rest."""
    if not os.path.exists(plan_path):
        return None
    x = np.load(os.path.join(ROOT, 'testing_data', '3dshape', cell, 'sampled_points.npy'))
    seam = np.abs(x[:, 11] - x[:, 5]) > 0.5
    recs = json.load(open(plan_path))['records']
    out = {}
    for k, name in ((0, 'no_seam'), (1, 'seam')):
        sel = [r for r in recs if seam[r['case']] == k]
        if sel:
            out[name] = {'n': len(sel), 'success_rate': float(np.mean([r['success'] for r in sel]))}
    return out


def row(name, s):
    if s is None:
        return '| {} | - | - | - | - | - |'.format(name)
    return '| {} | {} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
        name, s['n_endpoints_ok'], 100 * s['success_rate'], 100 * s['collision_rate'],
        100 * s['nonconv_rate'],
        '{:.2f}'.format(s['median_trans_ratio_success']) if s.get('median_trans_ratio_success') else '-')


def main():
    res = {'cells': {}, 'reference': {}}
    lines = ['# Metric NTFields baseline on the new 3-D shapes', '',
             'Planner: Lie-Shape-NTFields analysis/plan_eval.py --kind metric (MPPI, 200 steps, chart '
             'convergence 0.01), collisions against 50k env points. Columns: test pairs with collision-free '
             'endpoints, success %, collision %, not-converged %, median path / straight-line translation '
             '(successes).', '',
             '| cell | pairs | success | collision | not conv. | path ratio |', '|---|---|---|---|---|---|']
    for cell in ['{}_{}'.format(shape, env) for env in ENVS for shape in SHAPES] + SOFA_CELLS:
        plan_path = os.path.join(OUT, 'plan', '{}_metric_s1.json'.format(cell))
        s = plan_summary(plan_path)
        ep, loss = final_loss(os.path.join(ROOT, 'Experiments', 'shapes3d_metric_baseline', cell + '_s1'))
        if not os.path.exists(os.path.join(ROOT, 'testing_data', '3dshape', cell, 'meta.json')):
            continue
        res['cells'][cell] = {'plan': s, 'epoch': ep, 'val_loss': loss, 'data': data_stats(cell),
                              'by_partitions_crossed': crossing_breakdown(cell, plan_path)}
        if cell in SOFA_CELLS:
            res['cells'][cell]['by_angle_seam'] = seam_breakdown(cell, plan_path)
        lines.append(row(cell, s))
    lines += ['', 'Reference (same evaluator and recipe, 3 seeds each):', '',
              '| cell | pairs | success | collision | not conv. | path ratio |', '|---|---|---|---|---|---|']
    for cell, tag in REFS:
        ss = [plan_summary(os.path.join(LIE_PLAN, tag.format(k), cell + '.json')) for k in (1, 2, 3)]
        ss = [s for s in ss if s]
        res['reference'][cell] = ss
        for k, s in enumerate(ss, 1):
            lines.append(row('{} s{}'.format(cell, k), s))
    lines += ['', '## By partitions between start and goal (env5/7/8: 1 partition, 50/60/70 window; env6 / env9: 2 crossing, 50 / 70 windows)', '',
              'success % / collision % (n pairs) for 0, 1, 2 partitions crossed.', '',
              '| cell | 0 | 1 | 2 |', '|---|---|---|---|']
    for cell, c in res['cells'].items():
        b = c['by_partitions_crossed']
        if b:
            lines.append('| {} | {} |'.format(cell, ' | '.join(
                '{:.1f} / {:.1f} ({})'.format(100 * b[k]['success_rate'], 100 * b[k]['collision_rate'], b[k]['n'])
                if k in b else '-' for k in range(3))))
    lines += ['', '## Moving sofa (planar, --2d): U sofa 60 x 20, notch 24 x 10, 90-degree corridor turn', '',
              'Test pairs: 4000 drawn, kept the first 1000 whose start and goal are connected '
              '(check_sofa2d.py filter). Corner = start and goal in different legs. Seam = start and goal rz '
              'differ by more than half a turn in the chart, so a chart-space planner must spin the long way '
              'round (impossible in a corridor although the pair is connected).', '',
              '| cell | corridor | connected / drawn | success | collision | not conv. | same leg | around the corner '
              '| seam pairs | success w/o seam |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for cell in SOFA_CELLS:
        c = res['cells'].get(cell)
        if not c:
            continue
        ff = json.load(open(os.path.join(ROOT, 'testing_data', '3dshape', cell, 'meta.json'))).get('feasible_filter', {})
        s, b = c['plan'], c['by_partitions_crossed'] or {}
        fmt = lambda k: '{:.1f} ({})'.format(100 * b[k]['success_rate'], b[k]['n']) if k in b else '-'
        sm = c.get('by_angle_seam') or {}
        lines.append('| {} | {} | {} / {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            cell, SOFA_NOTE[cell.split('_')[1]], ff.get('connected', '-'), ff.get('generated', '-'),
            *(['{:.1f}'.format(100 * s[k]) for k in ('success_rate', 'collision_rate', 'nonconv_rate')]
              if s else ['-'] * 3), fmt(0), fmt(1),
            sm['seam']['n'] if 'seam' in sm else '-',
            '{:.1f}'.format(100 * sm['no_seam']['success_rate']) if 'no_seam' in sm else '-'))
    lines += ['', '## Training data', '', '| cell | pairs | tets | mean speed_dist x0 | x1 | x1 saturated | final loss |',
              '|---|---|---|---|---|---|---|']
    for cell, c in res['cells'].items():
        d = c['data']
        if d is None:
            lines.append('| {} | - | - | - | - | - | - |'.format(cell))
            continue
        lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.1f}% | {} |'.format(
            cell, d['pairs'], d['num_tets'], d['speed_dist_x0_mean'], d['speed_dist_x1_mean'],
            100 * d['frac_x1_saturated'], '{:.3e}'.format(c['val_loss']) if c['val_loss'] else '-'))
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(OUT, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
