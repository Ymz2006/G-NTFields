"""Figures for the cover-decoder campaign (run inside the container: matplotlib).

    python models/graph_cover/scripts/make_figures.py --out results/graph_cover/figures

* success_vs_dex.png   SE(3) planning success vs speed-calibration error (dex) for every diagnosed head
* ridge_free.png       copied from results/graph_cover/ridge_free (detector curves)
* slices_*.png         copied slice figures
"""
import argparse
import glob
import json
import os
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOST = '.'
RES_DIRS = [('results/graph_cover/plan', 'cover'), ('results/graph_lie/plan', 'lie'), ('results/graph_branch/plan', 'branch')]
METRIC_TAGS = ('lp2', 'lp0', 'lp4', 'lp2nf', 'lp2p2', 'eu2', 'eu0', 'eu2w', 'eu2r', 'eu2L', 'trieq_iso_lie')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_cover/figures')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pts = []
    for f in glob.glob('results/graph_cover/diag/*.json'):
        run = os.path.basename(f)[:-5]
        d = json.load(open(f))
        succ = None
        for rel, src in RES_DIRS:
            p = os.path.join(rel, run, 'Tshape3d_env1.json')
            if os.path.exists(p):
                try:
                    succ = json.load(open(p))['summary']['success_rate_se3']
                except Exception:
                    pass
        if succ is None:
            continue
        m = re.match(r'^T_(.+)_s\d+$', run)
        tag = m.group(1) if m else run
        pts.append((d['dex_bins'], 100 * succ, tag, tag in METRIC_TAGS))
    fig, ax = plt.subplots(figsize=(7, 5))
    for dex, s, tag, metric in pts:
        ax.scatter(dex, s, c='tab:blue' if metric else 'tab:red', marker='o' if metric else 's', s=40)
        ax.annotate(tag, (dex, s), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.scatter([], [], c='tab:blue', label='pseudometric by construction')
    ax.scatter([], [], c='tab:red', marker='s', label='no guarantee')
    ax.set_xlabel('speed calibration error dex = mean |log10(predicted / true speed)| over clearance decades')
    ax.set_ylabel('SE(3) planning success (%), Tshape3d_env1, 1000 cases')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'success_vs_dex.png'), dpi=130)
    for src in ['results/graph_cover/ridge_free/ridge_free.png']:
        if os.path.exists(src):
            shutil.copy(src, a.out)
    for run in ('T_eu2_s0', 'T_lp2_s0', 'T_gmetric_iso_lie_s0', 'T_lp2soft_s0'):
        for g in (0, 2):
            src = 'results/graph_cover/slices/{}/slice_goal{}.png'.format(run, g)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(a.out, 'slice_{}_goal{}.png'.format(run, g)))
    print('figures written to', a.out, len(pts), 'points')


if __name__ == '__main__':
    main()
