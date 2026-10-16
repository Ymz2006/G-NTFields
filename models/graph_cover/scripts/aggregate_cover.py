#!/usr/bin/env python3
"""Tables for the cover-decoder campaign from the per-case planning JSONs (plan_eval_graph.py summaries).

    python3 models/graph_cover/scripts/aggregate_cover.py [--out results/graph_cover/summary]

Groups runs by (scene, tag) over seeds. Includes the reference runs of models/graph_lie (graph_metric and
the triangle head, iso speed, Lie gradient) and, when present, the min-of-branches runs of
models/graph_branch (another campaign, read only). Standard library only.
"""
import argparse
import glob
import json
import os
import re
import statistics

HOST = '/media/jeffrey/G-Shape-NTFields'
SOURCES = [
    ('results/graph_cover/plan', 'cover'),
    ('results/graph_lie/plan', 'lie'),
    ('results/graph_branch/plan', 'branch'),
]
KEEP_LIE = re.compile(r'^[TLA]_(gmetric|trieq)_iso_lie_s\d$')
LABEL = {
    'lp2': 'cover lp, 2 cuts (metric)', 'lp0': 'cover lp, q routes only (metric)', 'lp4': 'cover lp, 4 cuts (metric)',
    'eu2': 'cover euclid, 2 cuts (metric)', 'lse2': 'cover lse, 2 cuts (non-metric ablation)',
    'lp2nf': 'cover lp, 2 cuts, no T_free (metric)', 'lp2soft': 'cover lp, 2 cuts, soft min 0.02',
    'lp2p2': 'cover lp p=2, 2 cuts (metric)', 'eu0': 'cover euclid, q routes only (metric)',
    'eu2w': 'cover euclid, 2 cuts, wide (metric)', 'eu2r': 'cover euclid, 2 cuts, res 0.1 (metric)',
    'eu2c': 'cover euclid, 2 cuts + conformal cone (non-metric)', 'eu2L': 'cover euclid, 2 cuts, lip cap 4 (metric)',
    'lp2p2c': 'cover lp p=2, 2 cuts + conformal cone (non-metric)', 'gmetric_iso_lie': 'graph_metric (reference)',
    'trieq_iso_lie': 'graph_tri_metric eq. V (reference, metric)',
    'ggram': 'graph_gram (branch campaign)', 'br1': 'branch K1', 'br4': 'branch K4', 'bc1e': 'branch conf K1 +end',
    'bc4e': 'branch conf K4 +end', 'bc4': 'branch conf K4', 'br4L': 'branch K4 lse', 'bc4Le': 'branch conf K4 lse +end',
}


def complete(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def collect():
    rows = {}
    for rel, src in SOURCES:
        for f in glob.glob(os.path.join(HOST, rel, '*', '*.json')):
            run = os.path.basename(os.path.dirname(f))
            if src == 'lie' and not KEEP_LIE.match(run):
                continue
            if not complete(f):
                continue
            try:
                s = json.load(open(f))['summary']
            except Exception:  # noqa: BLE001
                continue
            m = re.match(r'^([TLA])_(.+)_s(\d+)$', run)
            if not m:
                continue
            scene, tag, seed = m.group(1), m.group(2), int(m.group(3))
            tset = os.path.basename(f)[:-5]
            rows.setdefault((scene, tag, src), {}).setdefault(tset, {})[seed] = s
    return rows


def ms(vals, pct=True):
    if not vals:
        return '-'
    f = 100.0 if pct else 1.0
    if len(vals) == 1:
        return '{:.1f}'.format(vals[0] * f) if pct else '{:.2f}'.format(vals[0])
    return '{:.1f} ± {:.1f}'.format(statistics.mean(vals) * f, statistics.stdev(vals) * f) if pct else \
        '{:.2f} ± {:.2f}'.format(statistics.mean(vals), statistics.stdev(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_cover/summary')
    a = ap.parse_args()
    rows = collect()
    out = ['# Cover-quotient decoder campaign: per-case forward MPPI (plan_eval_graph.py, 1000 cases, seed 0)', '',
           'SE(3) = goal reached on SE(3) (primary); chart = official criterion; mean ± sd over seeds (n).', '']
    for scene in 'TLA':
        std = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}[scene]
        keys = sorted([k for k in rows if k[0] == scene], key=lambda k: (k[2] != 'cover', k[1]))
        if not keys:
            continue
        out += ['## {} standard set (`{}`)'.format(scene, std), '',
                '| head | src | n | SE(3) success | chart success | collision | SE(3) non-conv | iters | trans ratio | rot ratio |',
                '|---|---|---|---|---|---|---|---|---|---|']
        for k in keys:
            d = rows[k].get(std, {})
            if not d:
                continue
            v = list(d.values())
            out.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                LABEL.get(k[1], k[1]), k[2], len(v), ms([s['success_rate_se3'] for s in v]), ms([s['success_rate'] for s in v]),
                ms([s['collision_rate_se3'] for s in v]), ms([s['nonconv_rate_se3'] for s in v]),
                ms([s['median_iters_success_se3'] for s in v], False), ms([s['median_trans_ratio_success_se3'] for s in v], False),
                ms([s['median_rotgeo_ratio_success_se3'] for s in v], False)))
        extra = sorted({t for k in keys for t in rows[k] if t != std})
        if extra:
            out += ['', '### {} additional sets (SE(3) success / collision)'.format(scene), '',
                    '| head | src | ' + ' | '.join('`{}`'.format(t.replace(std + '_', '')) for t in extra) + ' |',
                    '|---|---|' + '---|' * len(extra)]
            for k in keys:
                cells = []
                for t in extra:
                    v = list(rows[k].get(t, {}).values())
                    cells.append('{} / {} ({})'.format(ms([s['success_rate_se3'] for s in v]), ms([s['collision_rate_se3'] for s in v]), len(v)) if v else '-')
                out.append('| {} | {} | '.format(LABEL.get(k[1], k[1]), k[2]) + ' | '.join(cells) + ' |')
        out.append('')
    os.makedirs(os.path.join(HOST, a.out), exist_ok=True)
    with open(os.path.join(HOST, a.out, 'plan_summary.md'), 'w') as fh:
        fh.write('\n'.join(out))
    with open(os.path.join(HOST, a.out, 'plan_summary.json'), 'w') as fh:
        json.dump({'{}|{}|{}'.format(*k): v for k, v in rows.items()}, fh, indent=1)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
