#!/usr/bin/env python3
"""SE(3) success by relative rotation angle (start -> goal) for the base and dual planners, pooled over the given
runs (seeds). Standard library only.

    python3 models/graph_homotopy/scripts/angle_table.py --runs T_hmin_s0,T_hmin_s1,T_hmin_s2 --set Tshape3d_env1
"""
import argparse
import json
import os

BINS = [(0, 60), (60, 120), (120, 150), (150, 180)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--set', default='Tshape3d_env1')
    ap.add_argument('--res', default='results/graph_homotopy/plan')
    a = ap.parse_args()
    base, dual = [], []
    for run in a.runs.split(','):
        fb = os.path.join(a.res, run, a.set + '__base.json')
        fd = os.path.join(a.res, run, a.set + '__dual.json')
        if os.path.exists(fb) and os.path.exists(fd):
            base += json.load(open(fb))['records']
            dual += json.load(open(fd))['records']
    print('| angle (deg) | n | base | cover (sigma*) | dual | either | + | - | sigma* = - |')
    print('|---|---|---|---|---|---|---|---|---|')
    rows = BINS + [(0, 181)]
    for lo, hi in rows:
        sb = [r for r in base if lo <= r['straight_rot_geo'] * 360 < hi]
        sd = [r for r in dual if lo <= r['straight_rot_geo'] * 360 < hi]
        if not sd:
            continue
        f = lambda rows_, k: 100.0 * sum(r[k] for r in rows_) / len(rows_)
        print('| {}-{} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |'.format(
            lo, hi if hi <= 180 else 180, len(sd), f(sb, 'success_se3'), f(sd, 'success_se3_cover'), f(sd, 'success_se3'),
            f(sd, 'success_se3_either'), f(sd, 'success_se3_plus'), f(sd, 'success_se3_minus'),
            100.0 * sum(r['sigma_star'] < 0 for r in sd) / len(sd)))


if __name__ == '__main__':
    main()
