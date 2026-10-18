"""Figures for the campaign report (run inside the container: matplotlib).

    python models/graph_homotopy/scripts/make_figures.py --out results/graph_homotopy/figures/report

* success_by_angle.png: SE(3) success vs relative rotation angle, base vs cover (dual) planner, pooled seeds of the
  main variant on T;
* planners.png: per-scene bars of base / cover / dual / either for the main variants (seed means).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/graph_homotopy/plan')
    ap.add_argument('--out', default='results/graph_homotopy/figures/report')
    ap.add_argument('--variants', default='hmin,hminP')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    variants = a.variants.split(',')
    # ---- success by angle (T, pooled seeds, first variant)
    base, dual = [], []
    for f in sorted(glob.glob(os.path.join(a.res, 'T_{}_s*'.format(variants[0]), 'Tshape3d_env1__dual.json'))):
        fb = f.replace('__dual', '__base')
        if os.path.exists(fb):
            base += json.load(open(fb))['records']
            dual += json.load(open(f))['records']
    if dual:
        edges = np.linspace(0, 180, 7)
        ang_b = np.array([r['straight_rot_geo'] * 360 for r in base])
        ang_d = np.array([r['straight_rot_geo'] * 360 for r in dual])
        sb = np.array([r['success_se3'] for r in base], float)
        sd = np.array([r['success_se3'] for r in dual], float)
        se = np.array([r['success_se3_either'] for r in dual], float)
        sc = np.array([r['success_se3_cover'] for r in dual], float)
        xs, yb, yd, ye, yc = [], [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mb, md_ = (ang_b >= lo) & (ang_b < hi + (hi >= 180)), (ang_d >= lo) & (ang_d < hi + (hi >= 180))
            if md_.sum() == 0:
                continue
            xs.append(0.5 * (lo + hi)); yb.append(100 * sb[mb].mean()); yd.append(100 * sd[md_].mean())
            ye.append(100 * se[md_].mean()); yc.append(100 * sc[md_].mean())
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(xs, yb, 'o-', label='base field, official planner')
        ax.plot(xs, yc, 's-', label='cover, one plan (sigma*)')
        ax.plot(xs, yd, 'd-', label='cover, both classes (dual)')
        ax.plot(xs, ye, '^--', color='gray', label='either sheet (oracle bound)')
        ax.set_xlabel('relative rotation start -> goal (deg)')
        ax.set_ylabel('SE(3) success (%)')
        ax.set_title('T scene, {} ({} cases, {} seeds)'.format(variants[0], len(dual), len(dual) // 1000))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'success_by_angle.png'), dpi=130)
        plt.close(fig)
    # ---- planners per scene / variant
    agg = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(glob.glob(os.path.join(a.res, '*_s*'))):
        m = re.match(r'([TLA])_(.+)_s(\d+)$', os.path.basename(run_dir))
        if not m or m.group(2) not in variants:
            continue
        for f in glob.glob(os.path.join(run_dir, '*__dual.json')):
            du = json.load(open(f))['summary']
            fb = f.replace('__dual', '__base')
            if not os.path.exists(fb):
                continue
            b = json.load(open(fb))['summary']
            key = (m.group(1), m.group(2))
            agg[key]['base'].append(100 * b['success_rate_se3'])
            agg[key]['cover'].append(100 * du['success_se3_cover'])
            agg[key]['dual'].append(100 * du['success_rate_se3'])
            agg[key]['either'].append(100 * du['success_se3_either'])
    if agg:
        keys = sorted(agg)
        fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(keys)), 3.6))
        w = 0.2
        for j, pl in enumerate(('base', 'cover', 'dual', 'either')):
            means = [np.mean(agg[k][pl]) for k in keys]
            errs = [np.std(agg[k][pl]) for k in keys]
            ax.bar(np.arange(len(keys)) + (j - 1.5) * w, means, w, yerr=errs, capsize=2, label=pl)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(['{} {}'.format(*k) for k in keys], fontsize=8)
        ax.set_ylim(80, 100)
        ax.set_ylabel('SE(3) success (%)')
        ax.legend(fontsize=7, ncol=4)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'planners.png'), dpi=130)
        plt.close(fig)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
