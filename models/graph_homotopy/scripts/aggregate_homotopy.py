#!/usr/bin/env python3
"""Aggregate the per-case evaluations of the homotopy campaign into one table (markdown + JSON).

    python3 models/graph_homotopy/scripts/aggregate_homotopy.py [--res results/graph_homotopy/plan]

Columns (SE(3) success %, the primary metric): base = official planner on the base field; cover = one plan on the
sheet the field prefers at the start (sigma*); dual = both sheets planned, converged one kept; either = at least
one of the two sheet plans succeeded (an upper bound for any selection rule); +, - = each sheet alone.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import statistics


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)['summary']
    except Exception:
        return None


def pct(x):
    return '-' if x is None else '{:.1f}'.format(100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/graph_homotopy/plan')
    ap.add_argument('--out', default='results/graph_homotopy/summary')
    a = ap.parse_args()
    rows = {}
    for run_dir in sorted(glob.glob(os.path.join(a.res, '*'))):
        run = os.path.basename(run_dir)
        if run == 'test':
            continue
        for f in sorted(glob.glob(os.path.join(run_dir, '*__*.json'))):
            s = load(f)
            if s is None:
                continue
            scene, planner = os.path.basename(f)[:-5].split('__')
            d = rows.setdefault((run, scene), {})
            d[planner] = s
    lines = ['| run | set | base SE(3) | base coll | base nonconv | cover | dual | dual-short | either | + | - | sigma- % | gap med | iters base/dual | rot ratio base/dual |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    table = []
    for (run, scene), d in sorted(rows.items()):
        b, du = d.get('base'), d.get('dual')
        rec = dict(run=run, scene=scene,
                   base=b and b['success_rate_se3'], base_coll=b and b['collision_rate_se3'], base_nonconv=b and b['nonconv_rate_se3'],
                   cover=du and du.get('success_se3_cover'), dual=du and du['success_rate_se3'], either=du and du.get('success_se3_either'),
                   plus=du and du.get('success_se3_plus'), minus=du and du.get('success_se3_minus'), shorter=du and du.get('success_se3_shorter'),
                   sigma_minus=(du or b or {}).get('frac_sigma_minus'), gap=(du or b or {}).get('median_sheet_gap'),
                   iters_base=b and b['median_iters_success_se3'], iters_dual=du and du['median_iters_success_se3'],
                   rot_base=b and b['median_rotgeo_ratio_success_se3'], rot_dual=du and du['median_rotgeo_ratio_success_se3'])
        table.append(rec)
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} |'.format(
            run, scene, pct(rec['base']), pct(rec['base_coll']), pct(rec['base_nonconv']), pct(rec['cover']), pct(rec['dual']),
            pct(rec['shorter']), pct(rec['either']), pct(rec['plus']), pct(rec['minus']), pct(rec['sigma_minus']),
            '-' if rec['gap'] is None else '{:.2f}'.format(rec['gap']),
            rec['iters_base'], rec['iters_dual'],
            '-' if rec['rot_base'] is None else '{:.2f}'.format(rec['rot_base']),
            '-' if rec['rot_dual'] is None else '{:.2f}'.format(rec['rot_dual'])))
    # seed means per (tag, scene)
    groups = defaultdict(list)
    for rec in table:
        m = re.match(r'([TLA])_(.+)_s(\d+)$', rec['run'])
        if m:
            groups[(m.group(1), m.group(2))].append(rec)
    lines += ['', '| scene | variant | n | base SE(3) | cover | dual | dual-short | either |', '|---|---|---|---|---|---|---|---|']
    for (sc, tag), recs in sorted(groups.items()):
        def ms(k):
            v = [r[k] for r in recs if r[k] is not None]
            if not v:
                return '-'
            if len(v) == 1:
                return '{:.1f}'.format(100 * v[0])
            return '{:.1f} +- {:.1f}'.format(100 * statistics.mean(v), 100 * statistics.pstdev(v))
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(sc, tag, len(recs), ms('base'), ms('cover'), ms('dual'), ms('shorter'), ms('either')))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'plan_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(a.out, 'plan_summary.json'), 'w') as fh:
        json.dump(table, fh, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
