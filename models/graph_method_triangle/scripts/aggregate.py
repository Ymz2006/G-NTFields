"""Aggregate evaluation metrics across runs/seeds into a markdown table.

    python models/graph_method_triangle/scripts/aggregate.py results/graph_method/final \
        --groups graph_gram=gram_s* metric=metric_s* --out results/graph_method/final/summary.md

A pattern containing '/' is a path glob (e.g. results/graph_method_final/graph_gram_s*),
so rows from other result roots can be put in the same table.
"""

import argparse
import fnmatch
import glob
import json
import math
import os

KEYS = [
    ('uniform_rel_p50', 'uni rel p50', 100), ('uniform_rel_p95', 'uni rel p95', 100),
    ('mixture_rel_p50', 'mix rel p50', 100), ('mixture_rel_p95', 'mix rel p95', 100),
    ('uniform_rmse', 'uni RMSE', 1),
    ('angle_0-90_rel_p50', '0-90 rel p50', 100), ('angle_150-170_rel_p50', '150-170 rel p50', 100),
    ('angle_170-179_rel_p50', '170-179 rel p50', 100), ('angle_179-180_rel_p50', '179-180 rel p50', 100),
    ('exact_pi_abs_mean', 'exact-pi abs', 1), ('near_diag_abs_mean', 'near-diag abs', 1),
    ('goal_T_mean', 'T(x,x)', 1),
    ('uniform_eik_x_regular_p50', 'eik res p50', 1), ('uniform_eik_x_regular_p95', 'eik res p95', 1),
    ('triangle_viol_frac', 'triangle viol', 100),
    ('plan_gradient_success', 'grad succ', 100), ('plan_gradient_cost_ratio_p50', 'grad cost p50', 1),
    ('plan_gradient_cost_ratio_p95', 'grad cost p95', 1), ('plan_gradient_fallback_frac', 'grad fallback', 100),
    ('plan_sampling_success', 'samp succ', 100), ('plan_sampling_cost_ratio_p50', 'samp cost p50', 1),
    ('plan_sampling_cost_ratio_p95', 'samp cost p95', 1),
]


def mean_std(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float('nan'), float('nan'), 0
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s, len(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('--groups', nargs='+', required=True, help='label=glob over run dirs')
    p.add_argument('--out', default=None)
    p.add_argument('--keys', nargs='*', default=None)
    a = p.parse_args()
    runs = sorted(d for d in os.listdir(a.root) if os.path.exists(os.path.join(a.root, d, 'metrics.json')))
    rows = []
    keys = [k for k in KEYS if a.keys is None or k[0] in a.keys]
    for g in a.groups:
        label, pat = g.split('=', 1)
        if '/' in pat:
            members = sorted(d for d in glob.glob(pat) if os.path.exists(os.path.join(d, 'metrics.json')))
        else:
            members = [os.path.join(a.root, r) for r in runs if fnmatch.fnmatch(r, pat)]
        ms = [json.load(open(os.path.join(r, 'metrics.json'))) for r in members]
        cells = []
        for k, _, scale in keys:
            m, s, n = mean_std([x.get(k) for x in ms])
            cells.append('{:.3g} ± {:.2g}'.format(m * scale, s * scale) if n > 1 else '{:.3g}'.format(m * scale))
        rows.append((label, len(members), cells))
    lines = ['| model | n | ' + ' | '.join(k[1] + (' (%)' if k[2] == 100 else '') for k in keys) + ' |',
             '|' + '---|' * (len(keys) + 2)]
    for label, n, cells in rows:
        lines.append('| {} | {} | {} |'.format(label, n, ' | '.join(cells)))
    text = '\n'.join(lines)
    print(text)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
