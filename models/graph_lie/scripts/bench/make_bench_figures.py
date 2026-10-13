"""Figures for the chart-vs-Lie benchmark, from results/graph_lie/bench/summary/bench_summary.json.

    python models/graph_lie/scripts/bench/make_bench_figures.py   (inside pytorchserver, repo root)

fig_success_subsets.png   SE(3) success per targeted subset, chart- vs Lie-trained fields, both samplers, per formulation
fig_N_by_theta.png        Eikonal value N (own formulation) with the Lie gradient on training endpoints, by |rotvec|
fig_free_tau_<S>.png      tau / T_geo on saturated-clearance pairs by start |rotvec|, per scene
fig_slope_by_theta.png    local slope -dtau/dcost (Lie metric) on rotation-dominated MPPI steps, by |rotvec|
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = 'results/graph_lie/bench'
J = json.load(open(os.path.join(ROOT, 'summary', 'bench_summary.json')))
OUT = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True)
C = {'chart': '#2a78d6', 'lie': '#eb6834'}          # categorical slots 1 and 2
TXT, TXT2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
SPEEDS = [('comb', 'combined'), ('iso', 'iso'), ('aniso', 'aniso')]
SUBSETS = ['nearid', 'bothfar', 'startfar', 'goalfar', 'perp', 'par', 'longtrans_near', 'longtrans_far', 'tightstart',
           'standard300', 'bigangle', 'rotheavy', 'transonly']
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': TXT2, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'text.color': TXT, 'axes.titlecolor': TXT, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})


def style(ax):
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- success per subset
for scene in ('T', 'L', 'A'):
    samplers = [s for s in ('chart', 'lie') if any(k.startswith('{}|{}|'.format(scene, s)) for k in J['plan'])]
    if not samplers:
        continue
    fig, axes = plt.subplots(len(SPEEDS), len(samplers), figsize=(6.2 * len(samplers), 2.6 * len(SPEEDS)), squeeze=False, sharey=True)
    for i, (sp, lab) in enumerate(SPEEDS):
        for j, sa in enumerate(samplers):
            ax = axes[i, j]
            style(ax)
            subs = [s for s in SUBSETS if '{}|{}|{}|{}|chart'.format(scene, sa, s, sp) in J['plan']]
            x = np.arange(len(subs))
            for k, g in enumerate(('chart', 'lie')):
                vals, seeds = [], []
                for s in subs:
                    e = J['plan'].get('{}|{}|{}|{}|{}'.format(scene, sa, s, sp, g))
                    vals.append(e['succ'] if e else np.nan)
                    seeds.append([p['succ'] for p in e['per_seed']] if e else [])
                ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=C[g], label='{} gradient'.format(g), zorder=2)
                for xi, sd in zip(x, seeds):
                    ax.scatter(np.full(len(sd), xi + (k - 0.5) * 0.38), sd, s=9, color='#fcfcfb', edgecolor=TXT2, linewidth=0.5, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(subs, rotation=35, ha='right')
            ax.set_ylim(40, 100)
            ax.set_title('{} scene, {} MPPI, {} formulation'.format(scene, 'official chart' if sa == 'chart' else 'Lie', lab), loc='left', fontsize=9)
            if j == 0:
                ax.set_ylabel('SE(3) success (%)')
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_success_subsets_{}.png'.format(scene)), dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------- N by theta (training endpoints)
ts = J.get('train_subsets', {})
if ts:
    bins = ['theta_0_45', 'theta_45_90', 'theta_90_135', 'theta_135_160', 'theta_160_180']
    xl = ['0-45', '45-90', '90-135', '135-160', '160-180']
    scenes = sorted({k.split('|')[0] for k in ts}, key='TLA'.index)
    fig, axes = plt.subplots(len(scenes), 3, figsize=(11, 2.7 * len(scenes)), squeeze=False, sharey=True)
    for r, scene in enumerate(scenes):
        for i, (sp, lab) in enumerate(SPEEDS):
            ax = axes[r, i]
            style(ax)
            for g in ('chart', 'lie'):
                yl = [ts.get('{}|{}|{}|{}'.format(scene, b, sp, g), {}).get('N_lie', np.nan) for b in bins]
                yc = [ts.get('{}|{}|{}|{}'.format(scene, b, sp, g), {}).get('N_chart', np.nan) for b in bins]
                ax.plot(xl, yl, '-o', color=C[g], lw=2, ms=5, label='{}-trained: N with Lie gradient'.format(g))
                ax.plot(xl, yc, '--s', color=C[g], lw=1.2, ms=4, alpha=0.7, label='{}-trained: N with chart gradient'.format(g))
            ax.axhline(1.0, color=TXT2, lw=0.8, ls=':')
            ax.set_title('{} scene, {} formulation'.format(scene, lab), loc='left', fontsize=9)
            if i == 0:
                ax.set_ylabel('N = S |grad tau|  (1 = Eikonal)')
            if r == 0 and i == 0:
                ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle('Median Eikonal value N on training endpoints, by |rotvec| (deg); solid = Lie gradient, dashed = chart gradient', x=0.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, 'fig_N_by_theta.png'), dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------- free pairs
for pf, F in J.get('free', {}).items():
    sc = pf[0]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
    for i, (sp, lab) in enumerate(SPEEDS):
        ax = axes[i]
        style(ax)
        for g in ('chart', 'lie'):
            e = F.get('{}|{}'.format(sp, g))
            if not e:
                continue
            ax.plot(F['labels'], e['tau_over_Tgeo'], '-o', color=C[g], lw=2, ms=5, label='{}-trained field, tau / T_geo'.format(g))
        ax.plot(F['labels'], F['T_chart_over_T_geo'], ':', color=TXT2, lw=1.2, label='chart segment length / T_geo')
        ax.axhline(1.0, color=TXT2, lw=0.8, ls=':')
        ax.set_title('{} scene, {} formulation'.format(sc, lab), loc='left', fontsize=9)
        if i == 0:
            ax.set_ylabel('tau / exact geodesic time')
            ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle('Saturated-clearance pairs (S_d = 1 along the whole geodesic): predicted time / exact time, by start |rotvec| (deg)', x=0.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'fig_free_tau_{}.png'.format(sc)), dpi=160)
    plt.close(fig)

# ---------------------------------------------------------------- local slope by theta
ab = J.get('audit_by_theta', {})
if ab:
    labels = ['0-45', '45-90', '90-135', '135-160', '160-180', '>180']
    for sa in ('chart', 'lie'):
        keys = [k for k in ab if k.split('|')[1] == sa]
        if not keys:
            continue
        scenes = sorted({k.split('|')[0] for k in keys}, key='TLA'.index)
        fig, axes = plt.subplots(len(scenes), 3, figsize=(11, 2.7 * len(scenes)), squeeze=False, sharey=True)
        for r, scene in enumerate(scenes):
            for i, (sp, lab) in enumerate(SPEEDS):
                ax = axes[r, i]
                style(ax)
                for g in ('chart', 'lie'):
                    e = ab.get('{}|{}|{}|{}'.format(scene, sa, sp, g))
                    if not e:
                        continue
                    n = len(e['slope Lie, rot steps'])
                    ax.plot(labels[:n], e['slope Lie, rot steps'], '-o', color=C[g], lw=2, ms=5, label='{}-trained: Lie metric'.format(g))
                    ax.plot(labels[:n], e['slope chart, rot steps'], '--s', color=C[g], lw=1.2, ms=4, alpha=0.7, label='{}-trained: chart metric'.format(g))
                ax.axhline(1.0, color=TXT2, lw=0.8, ls=':')
                ax.set_title('{} scene, {} formulation'.format(scene, lab), loc='left', fontsize=9)
                if i == 0:
                    ax.set_ylabel('local slope (1 = tau measures that time)')
                if r == 0 and i == 0:
                    ax.legend(frameon=False, fontsize=7.5)
        fig.suptitle('{} MPPI, rotation-dominated steps of executed paths: local slope -dtau/dcost by |rotvec| (deg); solid = Lie metric, dashed = chart metric'.format('Official chart' if sa == 'chart' else 'Lie'), x=0.01, ha='left', fontsize=9.5)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(OUT, 'fig_slope_by_theta_{}mppi.png'.format(sa)), dpi=160)
        plt.close(fig)
print('figures written to', OUT)
