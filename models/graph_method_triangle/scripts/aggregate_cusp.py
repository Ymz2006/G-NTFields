"""Aggregate the per-case MPPI planning results of plan_eval_graph.py.

    python3 models/graph_method_triangle/scripts/aggregate_cusp.py [--root results/graph_method_cusp/plan]
        [--lie-root /media/jeffrey/Lie-Shape-NTFields/results/compare/plan] [--out results/graph_method_cusp/summary]

Tags <group>_s<seed> are pooled per group. For every (group, set) it reports, as mean +- sd
over seeds: success under the official chart convergence and under SE(3) convergence,
non-convergence and collision rates, median iterations and path-length ratios of successful
cases; then the same success rates split by the start -> goal relative rotation
(0-90 / 90-150 / 150-180 deg) and by whether the shortest rotation crosses the chart seam.
Records of the Lie repo (original pipeline) lack the SE(3) fields; for the chart-input
baseline the two criteria coincide (its planner never crosses the seam), which is what the
re-scored native runs in --root confirm. Needs only numpy.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/graph_method_cusp/plan')
ap.add_argument('--lie-root', default='/media/jeffrey/Lie-Shape-NTFields/results/compare/plan')
ap.add_argument('--lie-groups', default='metric_s*,lie_rsqrtN_s*,Lshape3d_env3_metric_s*,Ashape3d_env4_metric_s*,'
                'Lshape3d_env3_lie_rsqrtN_s*,Ashape3d_env4_lie_rsqrtN_s*')
ap.add_argument('--out', default='results/graph_method_cusp/summary')
ap.add_argument('--sets', default='Tshape3d_env1,Tshape3d_env1_bigangle,Tshape3d_env1_rotheavy,Tshape3d_env1_transonly,'
                'Tshape3d_env1_tight_3d,Lshape3d_env3,Ashape3d_env4')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
BINS = [(0, 90), (90, 150), (150, 181)]


def load_runs():
    runs = {}  # (group, set) -> list of record lists
    for d in sorted(glob.glob(os.path.join(args.root, '*'))):
        tag = os.path.basename(d)
        if tag.startswith('test_'):
            continue
        g = re.sub(r'_s\d+$', '', tag)
        for p in glob.glob(os.path.join(d, '*.json')):
            s = os.path.basename(p)[:-5]
            runs.setdefault((g, s), []).append(json.load(open(p))['records'])
    for pat in args.lie_groups.split(','):
        for d in sorted(glob.glob(os.path.join(args.lie_root, pat))):
            tag = os.path.basename(d)
            if tag.endswith('_g') or tag.endswith('_mt') or tag.endswith('_tw'):
                continue
            g = 'LIE_' + re.sub(r'_s\d+$', '', tag)
            for p in glob.glob(os.path.join(d, '*.json')):
                s = os.path.basename(p)[:-5]
                runs.setdefault((g, s), []).append(json.load(open(p))['records'])
    return runs


def ms(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return float(v.mean()), float(v.std(ddof=0)), int(v.size)


def fmt(x, scale=100.0, nd=1):
    if x is None:
        return '-'
    m, s, n = x
    return '{:.{nd}f} ± {:.{nd}f}'.format(m * scale, s * scale, nd=nd) if n > 1 else '{:.{nd}f}'.format(m * scale, nd=nd)


def per_seed(recs, key, cond=None, complement=False):
    r = [x for x in recs if (cond is None or cond(x))]
    if not r:
        return None
    v = [x[key] for x in r if x.get(key) is not None]
    if not v:
        return None
    m = float(np.mean(v))
    return 1.0 - m if complement else m


def med_seed(recs, key, cond):
    v = [x[key] for x in recs if cond(x) and x.get(key) is not None]
    return float(np.median(v)) if v else None


runs = load_runs()
groups = sorted({g for g, _ in runs})
sets = [s for s in args.sets.split(',') if any((g, s) in runs for g in groups)]
lines = []
summary = {}
for s in sets:
    lines.append('\n## {}\n'.format(s))
    lines.append('| model | seeds | n cases | crossing | success chart (%) | success SE(3) (%) | non-conv chart (%) | non-conv SE(3) (%) | collision (%) | iters (median) | trans ratio | rot ratio (geo) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for g in groups:
        if (g, s) not in runs:
            continue
        R = runs[(g, s)]
        has3 = all('success_se3' in r[0] for r in R)
        row = {
            'seeds': len(R), 'n': len(R[0]), 'crossing': ms([per_seed(r, 'crosses_seam') for r in R]) if 'crosses_seam' in R[0][0] else None,
            'succ': ms([per_seed(r, 'success') for r in R]),
            'succ3': ms([per_seed(r, 'success_se3') for r in R]) if has3 else None,
            'nc': ms([per_seed(r, 'converged', complement=True) for r in R]),
            'nc3': ms([per_seed(r, 'converged_se3', complement=True) for r in R]) if has3 else None,
            'coll': ms([per_seed(r, 'collision') for r in R]),
            'iters': ms([med_seed(r, 'iters', lambda x: x['success']) for r in R]),
            'tr': ms([np.median([x['trans_len'] / max(x['straight_trans'], 1e-3) for x in r if x['success']]) if any(x['success'] for x in r) else None for r in R]),
            'rr': ms([np.median([x['rot_len_geo'] / max(x['straight_rot_geo'], 1e-3) for x in r if x['success']]) if any(x['success'] for x in r) else None for r in R]),
        }
        # by angle bin and by crossing
        for lo, hi in BINS:
            c = lambda x, lo=lo, hi=hi: lo <= x['straight_rot_geo'] * 360 < hi
            row['succ_{}-{}'.format(lo, min(hi, 180))] = ms([per_seed(r, 'success', c) for r in R])
            row['succ3_{}-{}'.format(lo, min(hi, 180))] = ms([per_seed(r, 'success_se3', c) for r in R]) if has3 else None
            row['n_{}-{}'.format(lo, min(hi, 180))] = int(sum(c(x) for x in R[0]))
        if 'crosses_seam' in R[0][0]:
            for name, val in (('cross', True), ('nocross', False)):
                c = lambda x, val=val: x['crosses_seam'] == val
                row['succ_' + name] = ms([per_seed(r, 'success', c) for r in R])
                row['succ3_' + name] = ms([per_seed(r, 'success_se3', c) for r in R]) if has3 else None
                row['nc3_' + name] = ms([per_seed(r, 'converged_se3', c, complement=True) for r in R]) if has3 else None
                row['coll_' + name] = ms([per_seed(r, 'collision', c) for r in R])
                row['n_' + name] = int(sum(c(x) for x in R[0]))
        summary.setdefault(s, {})[g] = row
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            g, row['seeds'], row['n'], fmt(row['crossing']), fmt(row['succ']), fmt(row['succ3']), fmt(row['nc']),
            fmt(row['nc3']), fmt(row['coll']), fmt(row['iters'], 1.0, 0), fmt(row['tr'], 1.0, 2), fmt(row['rr'], 1.0, 2)))
    lines.append('\nBy relative rotation angle (success chart / SE(3), %) and by seam crossing:\n')
    lines.append('| model | 0-90 | 90-150 | 150-180 | crossing (n) | non-crossing (n) | non-conv SE(3) cross / no | collision cross / no |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for g in groups:
        if (g, s) not in runs:
            continue
        row = summary[s][g]
        cells = []
        for lo, hi in BINS:
            k = '{}-{}'.format(lo, min(hi, 180))
            cells.append('{} / {} (n={})'.format(fmt(row['succ_' + k]), fmt(row['succ3_' + k]), row['n_' + k]))
        if 'succ_cross' in row:
            cells.append('{} / {} ({})'.format(fmt(row['succ_cross']), fmt(row['succ3_cross']), row['n_cross']))
            cells.append('{} / {} ({})'.format(fmt(row['succ_nocross']), fmt(row['succ3_nocross']), row['n_nocross']))
            cells.append('{} / {}'.format(fmt(row['nc3_cross']), fmt(row['nc3_nocross'])))
            cells.append('{} / {}'.format(fmt(row['coll_cross']), fmt(row['coll_nocross'])))
        else:
            cells += ['-', '-', '-', '-']
        lines.append('| {} | {} |'.format(g, ' | '.join(cells)))
md = '\n'.join(lines)
with open(os.path.join(args.out, 'plan_summary.md'), 'w') as fh:
    fh.write(md)
with open(os.path.join(args.out, 'plan_summary.json'), 'w') as fh:
    json.dump(summary, fh, indent=1)
print(md)
