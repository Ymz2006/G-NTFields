"""Aggregate plan_by_angle_free.py results: per model group (tags <group>_s<seed> pooled) and
angle bin, success / cost ratio / steps, mean +- sd over seeds.

    python3 models/graph_method_triangle/scripts/aggregate_free.py [--root results/graph_method_cusp/free]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/graph_method_cusp/free')
ap.add_argument('--out', default='results/graph_method_cusp/summary')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

runs = {}
for p in sorted(glob.glob(os.path.join(args.root, '*.json'))):
    tag = os.path.basename(p)[:-5]
    g = re.sub(r'_s\d+$', '', tag)
    runs.setdefault(g, []).append(json.load(open(p)))
bins = None
for g, L in runs.items():
    bins = list(L[0]['bins'].keys())
    break


def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return (float(v.mean()), float(v.std()), v.size) if v.size else None


def fmt(x, scale=1.0, nd=3):
    if x is None:
        return '-'
    m, s, n = x
    return '{:.{nd}f} ± {:.{nd}f}'.format(m * scale, s * scale, nd=nd) if n > 1 else '{:.{nd}f}'.format(m * scale, nd=nd)


lines = []
summary = {}
for mode in ('gradient', 'sampling'):
    for key, label, scale, nd in (('success', 'success (%)', 100, 1), ('cost_ratio_p50', 'cost ratio median', 1, 4),
                                  ('cost_ratio_p95', 'cost ratio p95', 1, 3), ('steps_mean', 'steps (mean)', 1, 0),
                                  ('fallback_frac', 'fallback fraction', 1, 3)):
        lines.append('\n### {} planner: {}\n'.format(mode, label))
        lines.append('| model | seeds | ' + ' | '.join(bins) + ' |')
        lines.append('|---|---|' + '---|' * len(bins))
        for g in sorted(runs):
            L = runs[g]
            cells = []
            for b in bins:
                v = ms([r['bins'][b][mode][key] for r in L if mode in r['bins'][b]])
                summary.setdefault(g, {}).setdefault(mode, {}).setdefault(b, {})[key] = v
                cells.append(fmt(v, scale, nd))
            lines.append('| {} | {} | {} |'.format(g, len(L), ' | '.join(cells)))
md = '\n'.join(lines)
open(os.path.join(args.out, 'free_summary.md'), 'w').write(md)
json.dump(summary, open(os.path.join(args.out, 'free_summary.json'), 'w'), indent=1)
print(md)
