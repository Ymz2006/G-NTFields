#!/usr/bin/env python3
"""Branch-population summary of every run (Experiments/graph_mb/<run>/branch_log.json + train_summary.json):
final alive units, usage per unit, events (revive / split / prune / merge), near-tie and duplicate fractions
at the last check. Standard library only.

    python3 models/graph_mb/scripts/branch_summary.py            # -> results/graph_mb/summary/branch_summary.md
"""
import json
import os
import sys

HOST = '/media/jeffrey/G-Shape-NTFields'
EXP = os.path.join(HOST, 'Experiments/graph_mb')
OUT = os.path.join(HOST, 'results/graph_mb/summary')


def main():
    rows = []
    for run in sorted(os.listdir(EXP)):
        p = os.path.join(EXP, run, 'branch_log.json')
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            log = json.load(fh)
        if not log:
            continue
        last = log[-1]
        ev = {}
        for rec in log:
            for e in rec['events']:
                ev[e['event']] = ev.get(e['event'], 0) + 1
        done = os.path.exists(os.path.join(EXP, run, 'train_summary.json'))
        rows.append((run, last['epoch'], 'done' if done else 'running', sum(last['alive']), len(last['alive']),
                     ' '.join('{:.3f}'.format(u) for u in last['usage_unit']), last['rows_per_pair'],
                     last['near_tie_frac'], last['near_tie_dup_frac'],
                     ' '.join('{}x{}'.format(v, k) for k, v in sorted(ev.items())) or '-'))
    lines = ['# Branch population per run (last check)', '',
             '| run | epoch | status | alive / K | usage per unit | rows/pair | near-tie frac | dup frac at ties | events |',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {} | {} | {} | {} / {} | {} | {:.2f} | {:.3f} | {:.2f} | {} |'.format(*r))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'branch_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    sys.exit(main())
