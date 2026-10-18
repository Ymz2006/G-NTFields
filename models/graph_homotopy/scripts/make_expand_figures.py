"""Figures of the expanded comparison (run inside the container: matplotlib, numpy).

    python /workspace/models/graph_homotopy/scripts/make_expand_figures.py --out results/graph_homotopy/figures/expand

* sets.png     SE(3) success per scene and set (std, long, bigrot, tight, hard): ours base / ours dual / NTFields 8k / 4k
* holes.png    success vs hole side on the through-the-hole sets, per shape, both models (and collision / non-convergence)
* bins.png     success by relative rotation, distance and endpoint clearance (targeted sets pooled), ours dual vs NTFields 8k
Reads results/graph_homotopy/summary/expand_summary.json (aggregate_expand.py).
"""
import argparse
import json
import os

import numpy as np


def mean_or_nan(v):
    v = [x for x in v if x is not None]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/graph_homotopy/summary/expand_summary.json')
    ap.add_argument('--out', default='results/graph_homotopy/figures/expand')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.summary))
    cols = [('ours base', '#9bb3c9'), ('ours dual', '#1f4e79'), ('NTFields 8k', '#e0a040'), ('NTFields 4k', '#f0d4a0')]
    sets = ['std', 'long', 'bigrot', 'tight', 'hard']
    scenes = ['T', 'L', 'A']

    # ---- sets.png
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, sc in zip(axes, scenes):
        rows = {r['set']: r for r in S['targeted'] if r['scene'] == sc}
        x = np.arange(len(sets))
        w = 0.2
        for i, (label, color) in enumerate(cols):
            m = [mean_or_nan(rows[s][label])[0] * 100 if s in rows else np.nan for s in sets]
            e = [mean_or_nan(rows[s][label])[1] * 100 if s in rows else 0 for s in sets]
            ax.bar(x + (i - 1.5) * w, m, w, yerr=e, color=color, label=label, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sets)
        ax.set_title('{} scene'.format(sc))
        ax.set_ylim(50, 100)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('SE(3) success %')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'sets.png'), dpi=150)

    # ---- holes.png
    holes = ['90', '75', '66']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, (key, title) in enumerate((('succ', 'SE(3) success %'), ('coll', 'collision %'), ('nonconv', 'non-convergence %'))):
        ax = axes[k]
        for sc, ls in zip(scenes, ('-', '--', ':')):
            rows = {r['hole']: r for r in S['wall'] if r['scene'] == sc}
            for label, color, field in (('ours dual', '#1f4e79', ('ours dual', 'coll_ours', 'nonconv_ours')),
                                        ('NTFields 8k', '#e0a040', ('NTFields 8k', 'coll_mb', 'nonconv_mb'))):
                f = {'succ': field[0], 'coll': field[1], 'nonconv': field[2]}[key]
                y = [mean_or_nan(rows[h][f])[0] * 100 if h in rows else np.nan for h in holes]
                ax.plot([90, 75, 66], y, ls, marker='o', color=color, label='{} {}'.format(sc, label))
        ax.set_xlabel('hole side (units; shapes are 60 long)')
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'holes.png'), dpi=150)

    # ---- bins.png
    kinds = [('ang', 'relative rotation (deg)', {0: '0-60', 60: '60-120', 120: '120-150', 150: '150-180'}),
             ('dp', 'start-goal distance', {0: '0-0.3', 0.3: '0.3-0.5', 0.5: '0.5-0.7', 0.7: '>0.7'}),
             ('cl', 'min endpoint clearance', {0: '<0.01', 0.01: '0.01-0.02', 0.02: '0.02-0.03', 0.03: '>0.03'})]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (kind, xlabel, labels) in zip(axes, kinds):
        los = sorted(labels)
        x = np.arange(len(los))
        w = 0.25
        for i, (label, color, key) in enumerate((('ours base', '#9bb3c9', 'ours_base'), ('ours dual', '#1f4e79', 'ours_dual'), ('NTFields 8k', '#e0a040', 'ntfields8k'))):
            y = []
            for lo in los:
                s_ = n_ = 0
                for b in S['bins']:
                    if b['kind'] == kind and abs(b['lo'] - lo) < 1e-9:
                        s_ += b[key][0]
                        n_ += b[key][1]
                y.append(100.0 * s_ / n_ if n_ else np.nan)
            ax.bar(x + (i - 1) * w, y, w, color=color, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[lo] for lo in los])
        ax.set_xlabel(xlabel)
        ax.set_ylim(50, 100)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('SE(3) success % (targeted sets pooled)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'bins.png'), dpi=150)
    print('wrote figures to', a.out)


if __name__ == '__main__':
    main()
