#!/usr/bin/env python3
"""Tables for the multi-branch learned-cut campaign (models/graph_mb): per-case MPPI results (results/graph_branch/plan/<run>/<set>.json) grouped by
variant over seeds, plus the reference graph_metric iso / Lie runs of models/graph_lie (results/graph_lie/plan) and the
original metric NTFields baseline (Lie-Shape-NTFields metric_baseline checkpoints, seeds 1-3, scored by the same
evaluator in results/graph_method_cusp/plan/<scene>_native_s<k>).

    python3 models/graph_branch/scripts/aggregate_mb.py            # -> results/graph_mb/summary/plan_summary.{md,json}

Columns: SE(3) success (primary: same pose on SE(3)), chart success (official criterion), collision, SE(3)
non-convergence, median iterations to SE(3) convergence, median translation / geodesic-rotation path ratios of
successful cases, median fraction of path steps along which tau increases. mean +- sd over seeds (sd only with
>= 2 seeds); n = seeds. Paired differences against graph_gram use the same seeds. Python 3 standard library only.
"""
import json
import math
import os
import re
import statistics as st
import sys

HOST = '/media/jeffrey/G-Shape-NTFields'
RES = os.path.join(HOST, 'results/graph_mb/plan')
PREV = os.path.join(HOST, 'results/graph_branch/plan')   # previous campaign (graph_gram, branch conformal heads)
REF = os.path.join(HOST, 'results/graph_lie/plan')
ORIG = os.path.join(HOST, 'results/graph_method_cusp/plan')
OUT = os.path.join(HOST, 'results/graph_mb/summary')
SETS = ['Tshape3d_env1', 'Lshape3d_env3', 'Ashape3d_env4', 'Tshape3d_env1_bigangle', 'Tshape3d_env1_rotheavy',
        'Tshape3d_env1_transonly', 'Tshape3d_env1_tight_3d']
LABEL = {
    'orig': 'metric baseline (original NTFields)',
    'ggram': 'graph_gram (fixed decoder)',
    'gmetric': 'graph_metric (graph_lie runs)',
    'bc4': 'branch K=4 conformal (previous campaign, winner-only training)',
    'bc4e': 'branch K=4 conformal + end (previous campaign)',
    'mb4': 'learned cut: folded conformal K=4, spec defaults',
    'mb4e': 'learned cut: folded conformal K=4, spec defaults + end',
    'mb6': 'learned cut: folded conformal K=6, spec defaults',
    'mb4F': 'learned cut: spec free heads K=4',
    'mb4A': 'ablation: losses on the active branch only',
    'mb4nd': 'ablation: no distinctness loss',
    'mb4D': 'ablation: band Delta = 0.1 absolute',
    'mb4z': 'round 2: no initial offsets, revival without raise',
    'mb4r': 'round 2: revival raised inside the band',
    'mb4s': 'round 3: split revival (gradient-sign halves), K=4',
    'mb2s': 'round 3: split revival, K=2',
    'mb4se': 'round 3: split revival, K=4 + end',
    'mb4p': 'round 3b: proactive gradient-sign splits every 400 epochs, K=4',
    'mb2p': 'round 3b: proactive gradient-sign splits every 400 epochs, K=2',
    'mb4pe': 'round 3b: proactive splits, K=4 + end',
}
PREV_KEEP = ('ggram', 'bc4', 'bc4e')
KEYS = [('success_rate_se3', 'SE(3) succ.', 100), ('success_rate', 'chart succ.', 100),
        ('collision_rate_se3', 'collision', 100), ('nonconv_rate_se3', 'non-conv.', 100),
        ('median_iters_success_se3', 'iters', 1), ('median_trans_ratio_success_se3', 'trans. ratio', 1),
        ('median_rotgeo_ratio_success_se3', 'rot. ratio', 1), ('median_tau_frac_increase', 'tau incr.', 1)]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)['summary']
    except (OSError, ValueError, KeyError):
        return None


def collect():
    rows = {}   # (variant, set) -> {seed: summary}
    for root, keep in ((RES, None), (PREV, PREV_KEEP)):
        if not os.path.isdir(root):
            continue
        for run in sorted(os.listdir(root)):
            m = re.match(r'^([TLA])_(.+)_s(\d+)$', run)
            if not m:
                continue
            _, var, seed = m.group(1), m.group(2), int(m.group(3))
            if keep is not None and var not in keep:
                continue
            for s in SETS:
                summ = load(os.path.join(root, run, s + '.json'))
                if summ:
                    rows.setdefault((var, s), {})[seed] = summ
    if os.path.isdir(REF):  # graph_metric, iso speed, Lie gradient (the S-series baseline)
        for run in sorted(os.listdir(REF)):
            m = re.match(r'^([TLA])_gmetric_iso_lie_s(\d+)$', run)
            if not m:
                continue
            for s in SETS:
                summ = load(os.path.join(REF, run, s + '.json'))
                if summ and summ.get('batch') == 125:
                    rows.setdefault(('gmetric', s), {})[int(m.group(2))] = summ
    if os.path.isdir(ORIG):  # the original metric NTFields checkpoints (its own trainer), same evaluator and settings
        for run in sorted(os.listdir(ORIG)):
            m = re.match(r'^([TLA])_native_s(\d+)$', run)
            if not m:
                continue
            for s in SETS:
                summ = load(os.path.join(ORIG, run, s + '.json'))
                if summ and summ.get('batch') == 125 and summ.get('seed') == 0:
                    rows.setdefault(('orig', s), {})[int(m.group(2))] = summ
    return rows


def fmt(vals, scale):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return '-'
    if len(vals) == 1:
        return '{:.1f}'.format(vals[0]) if scale == 100 else '{:.3g}'.format(vals[0])
    m, sd = st.mean(vals), st.stdev(vals)
    return '{:.1f} +- {:.1f}'.format(m, sd) if scale == 100 else '{:.3g} +- {:.2g}'.format(m, sd)


def main():
    rows = collect()
    os.makedirs(OUT, exist_ok=True)
    lines = ['# Branch-decoder campaign: per-case MPPI planning (forward MPPI, batch 125, seed 0)', '',
             'Generated by models/graph_mb/scripts/aggregate_mb.py from results/graph_branch/plan and the',
             'graph_metric iso / Lie runs of results/graph_lie/plan. mean +- sd over seeds; n = seeds.', '']
    out_json = {}
    order = list(LABEL)
    for s in SETS:
        vs = [v for v in order if (v, s) in rows] + sorted({v for v, ss in rows if ss == s and v not in LABEL})
        if not vs:
            continue
        lines += ['## ' + s, '', '| variant | n | ' + ' | '.join(k[1] for k in KEYS) + ' |',
                  '|' + '---|' * (len(KEYS) + 2)]
        for v in vs:
            d = rows[(v, s)]
            seeds = sorted(d)
            cells = [fmt([d[sd].get(k) for sd in seeds], sc) for k, _, sc in KEYS]
            lines.append('| {} | {} | {} |'.format(LABEL.get(v, v), len(seeds), ' | '.join(cells)))
            out_json.setdefault(s, {})[v] = {'seeds': seeds, **{k: [d[sd].get(k) for sd in seeds] for k, _, _ in KEYS}}
        # paired differences vs graph_gram (SE(3) success, collision) on common seeds
        if ('bc4', s) in rows:
            g = rows[('bc4', s)]
            lines += ['', 'Paired difference to branch K=4 conformal (previous campaign) on common seeds (points; SE(3) success / collision):', '']
            for v in vs:
                if v in ('bc4', 'orig'):   # orig: other trainer, seeds 1-3 do not pair with ours
                    continue
                common = sorted(set(g) & set(rows[(v, s)]))
                if not common:
                    continue
                ds = [100 * (rows[(v, s)][c]['success_rate_se3'] - g[c]['success_rate_se3']) for c in common]
                dc = [100 * (rows[(v, s)][c]['collision_rate_se3'] - g[c]['collision_rate_se3']) for c in common]
                lines.append('* {}: {:+.1f} / {:+.1f} (seeds {})'.format(LABEL.get(v, v), st.mean(ds), st.mean(dc), common))
        if ('gmetric', s) in rows:
            g = rows[('gmetric', s)]
            lines += ['', 'Paired difference to graph_metric on common seeds (points; SE(3) success / collision):', '']
            for v in vs:
                if v in ('gmetric', 'orig'):
                    continue
                common = sorted(set(g) & set(rows[(v, s)]))
                if not common:
                    continue
                ds = [100 * (rows[(v, s)][c]['success_rate_se3'] - g[c]['success_rate_se3']) for c in common]
                dc = [100 * (rows[(v, s)][c]['collision_rate_se3'] - g[c]['collision_rate_se3']) for c in common]
                lines.append('* {}: {:+.1f} / {:+.1f} (seeds {})'.format(LABEL.get(v, v), st.mean(ds), st.mean(dc), common))
        lines.append('')
    with open(os.path.join(OUT, 'plan_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'plan_summary.json'), 'w') as fh:
        json.dump(out_json, fh, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
