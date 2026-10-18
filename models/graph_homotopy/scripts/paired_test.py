#!/usr/bin/env python3
"""Paired per-case comparison of the base planner and the cover planners on the same fields (McNemar counts and
exact binomial p-value on the discordant pairs; standard library only).

    python3 models/graph_homotopy/scripts/paired_test.py --runs T_hminL_s0,T_hminL_s1,T_hminL_s2 --set Tshape3d_env1
"""
import argparse
import json
import math
import os


def binom_two_sided(k, n):
    """Two-sided exact binomial test p-value for k successes of n at p = 0.5."""
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--set', default='Tshape3d_env1')
    ap.add_argument('--res', default='results/graph_homotopy/plan')
    a = ap.parse_args()
    pairs = []
    for run in a.runs.split(','):
        fb = os.path.join(a.res, run, a.set + '__base.json')
        fd = os.path.join(a.res, run, a.set + '__dual.json')
        if not (os.path.exists(fb) and os.path.exists(fd)):
            continue
        B = {r['case']: r for r in json.load(open(fb))['records']}
        for r in json.load(open(fd))['records']:
            pairs.append((B[r['case']]['success_se3'], r))
    n = len(pairs)
    print('runs {} set {}: {} paired cases'.format(a.runs, a.set, n))
    print('| planner | success % | base fail -> success | base success -> fail | net | p (exact McNemar) |')
    print('|---|---|---|---|---|---|')
    for key, name in (('success_se3_cover', 'cover (sigma*)'), ('success_se3', 'dual'), ('success_se3_shorter', 'dual, shorter-path rule'),
                      ('success_se3_either', 'either (oracle)')):
        if any(key not in r for _, r in pairs):
            continue
        gain = sum(1 for b, r in pairs if (not b) and r[key])
        loss = sum(1 for b, r in pairs if b and not r[key])
        succ = 100.0 * sum(r[key] for _, r in pairs) / n
        print('| {} | {:.1f} | {} | {} | {:+d} | {:.2g} |'.format(name, succ, gain, loss, gain - loss, binom_two_sided(loss, gain + loss)))
    print('base success %: {:.1f}'.format(100.0 * sum(b for b, _ in pairs) / n))


if __name__ == '__main__':
    main()
