#!/usr/bin/env python
"""Branch-value plots of the stage-0 circle runs: every alive branch along delta, the decoded minimum, the truth,
the winner's region colour bar and the ridge / hand-over ties (spec section 7, "plot at checkpoints").

    python models/graph_mb/scripts/stage0_plot.py results/graph_mb/stage0 --tags 0a_K4_s0 0a_K1_s0 0b_K4_s0
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402

COL = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    with open(os.path.join(a.folder, 'stage0_results.json')) as fh:
        res = json.load(fh)['results']
    tags = a.tags or sorted(res)
    fig, axes = plt.subplots(len(tags), 1, figsize=(9, 3.2 * len(tags)), squeeze=False)
    for ax, tag in zip(axes[:, 0], tags):
        z = np.load(os.path.join(a.folder, tag + '_curves.npz'))
        d, V, win, T = np.degrees(z['d']), z['V'], z['win'], z['T']
        r = res[tag]
        for k in range(V.shape[1]):
            if np.isfinite(V[:, k]).any():
                ax.plot(d, V[:, k], color=COL[k % len(COL)], lw=1.0, alpha=0.8, label='branch {}'.format(k))
        ax.plot(d, T, 'k--', lw=1.0, label='truth')
        ax.plot(d, V.min(1), color='k', lw=2.0, alpha=0.35, label='decoded min')
        for c in r.get('ridge_cuts_deg', []):
            ax.axvline(c, color='k', lw=1.2)
        for c in r.get('handovers_deg', []):
            ax.axvline(c, color='grey', lw=0.8, ls=':')
        ymax = min(np.nanmax(np.where(np.isfinite(V), V, np.nan)), 2.0 * T.max())
        ax.set_ylim(0, ymax * 1.05)
        for k in range(V.shape[1]):
            m = win == k
            if m.any():
                ax.fill_between(d, 0, ymax * 0.03, where=m, color=COL[k % len(COL)], alpha=0.9, step='mid')
        ax.set_title('{}: surviving {}  median err {:.2f} %  ridge cuts {}  hand-overs {}'.format(
            tag, r['surviving'], r['median_rel_err_pct'], r.get('ridge_cuts_deg', []), r.get('handovers_deg', [])),
            fontsize=9)
        ax.set_xlabel('delta (deg)')
        ax.set_ylabel('T')
        ax.legend(fontsize=7, ncol=4, loc='upper left')
        ax.set_xlim(0, 360)
    fig.tight_layout()
    out = a.out or os.path.join(a.folder, 'stage0_branches.png')
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    sys.exit(main())
