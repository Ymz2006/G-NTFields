"""Figures for models/graph_lie from results/graph_lie/summary/summary.json (run aggregate.py first).

    python models/graph_lie/scripts/make_figures.py      (inside pytorchserver, from the repo root)

Writes results/graph_lie/figures/{success_dumbbell, paired_delta, eikonal_by_rotvec, free_by_rotvec}.png.
Colours: reference categorical slots 1 / 2 (blue = Lie gradient, orange = chart gradient).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = 'results/graph_lie'
OUT = os.path.join(ROOT, 'figures')
LIE, CHART = '#2a78d6', '#eb6834'
SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
HEADS = {'gmetric': 'graph_metric', 'trieq': 'triangle head'}
SPEEDS = {'comb': 'combined', 'iso': 'isotropic', 'aniso': 'anisotropic'}
SCENES = {'T': 'T-shape', 'L': 'L-shape', 'A': 'A-shape'}
STD = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def configs():
    return [(h, s) for h in HEADS for s in SPEEDS]


def runs_metric(S, env, head, speed, grad, sset, key='succ_se3'):
    out = []
    for k, m in S['runs'].items():
        name, st = k.split('|')
        e, h, sp, g, seed = name.split('_')
        if (e, h, sp, g) == (env, head, speed, grad) and st == sset and m.get(key) is not None:
            out.append(m[key])
    return out


def success_dumbbell(S):
    """Rows = heads (own x-range), columns = scenes; y = speed formulation; chart vs Lie gradient."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.6), sharey=True)
    y = np.arange(len(SPEEDS))[::-1]
    for i, h in enumerate(HEADS):
        lo, hi = 100.0, 0.0
        for ax, env in zip(axes[i], SCENES):
            for yi, s in zip(y, SPEEDS):
                c = runs_metric(S, env, h, s, 'chart', STD[env])
                l = runs_metric(S, env, h, s, 'lie', STD[env])
                if c and l:
                    ax.plot([np.mean(c), np.mean(l)], [yi, yi], color=BASE, lw=2, zorder=1, solid_capstyle='round')
                for vals, col in ((c, CHART), (l, LIE)):
                    if vals:
                        ax.scatter(np.mean(vals), yi, s=46, color=col, edgecolor=SURF, linewidth=2, zorder=3)
                        lo, hi = min(lo, np.mean(vals)), max(hi, np.mean(vals))
            ax.set_title('{} · {}'.format(SCENES[env], HEADS[h]), color=INK, fontsize=9.5, loc='left')
            ax.grid(axis='y', visible=False)
        pad = max(1.0, 0.08 * (hi - lo))
        for ax in axes[i]:
            ax.set_xlim(lo - pad, min(100.0, hi + pad))
        axes[i][0].set_yticks(y)
        axes[i][0].set_yticklabels([SPEEDS[s] for s in SPEEDS], color=INK2)
    fig.supxlabel('SE(3) success (%), official MPPI, mean over seeds', color=INK2, fontsize=9)
    h1 = axes[0][0].scatter([], [], s=46, color=CHART, label='chart gradient (partial derivatives)')
    h2 = axes[0][0].scatter([], [], s=46, color=LIE, label='Lie gradient')
    fig.legend(handles=[h1, h2], loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'success_dumbbell.png'), dpi=160)
    plt.close(fig)


def paired_delta(S):
    """Rows = planner (official chart MPPI, Lie MPPI), columns = scenes; dots = Lie - chart per seed."""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True, sharex=True)
    labels = ['{} · {}'.format(HEADS[h], SPEEDS[s]) for h, s in configs()]
    y = np.arange(len(labels))[::-1]
    for i, (key, planner) in enumerate((('paired', 'official MPPI (chart samples)'),
                                        ('paired_liemppi', 'Lie MPPI (body-twist samples)'))):
        for ax, env in zip(axes[i], SCENES):
            ax.axvline(0, color=INK2, lw=1.3, zorder=2.5)
            for yi, (h, s) in zip(y, configs()):
                P = S.get(key, {}).get('{}|{}|{}|{}'.format(env, h, s, STD[env]))
                if not P:
                    continue
                d = P['se3']['diffs']
                ax.scatter(d, [yi] * len(d), s=18, color=LIE, alpha=0.45, edgecolor='none', zorder=2)
                ax.scatter(np.mean(d), yi, s=52, color=LIE, edgecolor=SURF, linewidth=2, zorder=3)
            ax.set_title('{} · {}'.format(SCENES[env], planner), color=INK, fontsize=9.5, loc='left')
            ax.grid(axis='y', visible=False)
        axes[i][0].set_yticks(y)
        axes[i][0].set_yticklabels(labels, color=INK2)
    fig.supxlabel('Δ SE(3) success, Lie − chart gradient (points)', color=INK2, fontsize=9)
    fig.suptitle('Paired by seed (same initialisation and data order): small dots = seeds, large dot = mean',
                 x=0.6, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'paired_delta.png'), dpi=160)
    plt.close(fig)


def eikonal_by_rotvec(S, env='T', head='gmetric'):
    D = S.get('diag', {})
    if not D:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, sp in zip(axes, SPEEDS):
        for g, col, lab in (('chart', CHART, 'trained with chart gradient'), ('lie', LIE, 'trained with Lie gradient')):
            d = D.get('{}|{}|{}|{}'.format(env, head, sp, g))
            if not d:
                continue
            x = [0.5 * (lo + hi) for lo, hi in d['bins']]
            ax.plot(x, d['N_lie'], color=col, lw=2, marker='o', ms=5, mec=SURF, mew=1.5, label=lab)
        ax.axhline(1.0, color=BASE, lw=1, zorder=0)
        ax.set_title('{} formulation'.format(SPEEDS[sp]), color=INK, fontsize=10, loc='left')
        ax.set_xlabel('endpoint rotation angle |rotvec| (deg)')
    axes[0].set_ylabel('median Lie-Eikonal ratio N (1 = exact)')
    axes[0].legend(loc='upper left')
    fig.suptitle('{} on {}: the Eikonal equation in the physical (left-invariant) metric'.format(HEADS[head], SCENES[env]),
                 x=0.5, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, 'eikonal_by_rotvec.png'), dpi=160)
    plt.close(fig)


def free_by_rotvec(S):
    F = S.get('free', {})
    if not F:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, h in zip(axes, HEADS):
        for g, col, lab in (('chart', CHART, 'chart gradient'), ('lie', LIE, 'Lie gradient')):
            d = F.get('F2|{}|{}'.format(h, g))
            if not d:
                continue
            ax.plot(range(len(d['by_rotvec'])), d['by_rotvec'], color=col, lw=2, marker='o', ms=5, mec=SURF, mew=1.5,
                    label=lab)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['0-90', '90-135', '135-160', '160-180'])
        ax.set_title(HEADS[h], color=INK, fontsize=10, loc='left')
        ax.set_xlabel('larger endpoint |rotvec| (deg)')
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('median relative error vs exact T* (%)')
    axes[0].legend(loc='upper left')
    fig.suptitle('Obstacle-free SE(3): error against the exact geodesic travel time', x=0.5, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, 'free_by_rotvec.png'), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(os.path.join(ROOT, 'summary', 'summary.json')))
    success_dumbbell(S)
    paired_delta(S)
    eikonal_by_rotvec(S)
    free_by_rotvec(S)
    print('figures in', OUT)


if __name__ == '__main__':
    main()
