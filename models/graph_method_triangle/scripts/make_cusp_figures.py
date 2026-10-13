"""Figures for the cusp / seam study (reads results/graph_method_cusp/summary/*.json and slices/).

    python models/graph_method_triangle/scripts/make_cusp_figures.py [--root results/graph_method_cusp]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='results/graph_method_cusp')
args = ap.parse_args()
S = json.load(open(os.path.join(args.root, 'summary', 'plan_summary.json')))
out = os.path.join(args.root, 'figures')
os.makedirs(out, exist_ok=True)

# fixed categorical assignment (validated palette, light mode)
COL = {'original': '#eb6834', 'metric_head': '#2a78d6', 'graph_metric': '#1baf7a', 'graph_tri_eq': '#6250d6',
       'graph_gram': '#eda100', 'native_graph_tri_eq': '#008300', 'native_graph_metric': '#e87ba4'}
LABEL = {'original': 'original metric NTFields (chart input)', 'metric_head': 'ours: metric head on G(x)',
         'graph_metric': 'ours: graph_metric (Gram b, decoder)', 'graph_tri_eq': 'ours: graph_tri_metric, equivariant V (triangle)',
         'graph_gram': 'ours: graph_gram', 'native_graph_tri_eq': 'graph_tri_metric eq. V in the ORIGINAL trainer',
         'native_graph_metric': 'graph_metric in the ORIGINAL trainer'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True})


def group_key(prefix, name):
    return {'original': prefix + '_native', 'metric_head': prefix + '_metric', 'graph_metric': prefix + '_gmetric',
            'graph_tri_eq': prefix + '_trieq', 'graph_gram': prefix + '_ggram',
            'native_graph_tri_eq': 'GN_' + prefix + '_trieq_chart', 'native_graph_metric': 'GN_' + prefix + '_gmetric_chart'}[name]


def val(row, key):
    v = row.get(key)
    return (float('nan'), 0.0) if v is None else (100 * v[0], 100 * v[1])


# ---- Figure 1: success by relative-rotation bin, chart vs SE(3), three environments
sets = [('Tshape3d_env1', 'T'), ('Lshape3d_env3', 'L'), ('Ashape3d_env4', 'A')]
bins = ['0-90', '90-150', '150-180']
models = ['original', 'metric_head', 'graph_metric', 'graph_tri_eq', 'native_graph_tri_eq']
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (s, p) in zip(axes, sets):
    rows = S.get(s, {})
    present = [m for m in models if group_key(p, m) in rows]
    w = 0.8 / max(len(present), 1)
    for i, m in enumerate(present):
        row = rows[group_key(p, m)]
        x = np.arange(len(bins)) + (i - (len(present) - 1) / 2) * w
        m3 = [val(row, 'succ3_' + b) for b in bins]
        mc = [val(row, 'succ_' + b) for b in bins]
        if m == 'original':
            ax.bar(x, [v[0] for v in mc], w * 0.9, color=COL[m], label=LABEL[m], yerr=[v[1] for v in mc], capsize=2, error_kw={'lw': 0.8})
        else:
            ax.bar(x, [v[0] for v in m3], w * 0.9, color=COL[m], label=LABEL[m] + ' [SE(3)]', yerr=[v[1] for v in m3], capsize=2, error_kw={'lw': 0.8})
            ax.plot(x, [v[0] for v in mc], marker='_', ms=12, ls='none', color='#0b0b0b', mew=1.4)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([b + ' deg' for b in bins])
    ax.set_title('{} ({} cases)'.format(s, rows[group_key(p, 'original')]['n'] if group_key(p, 'original') in rows else '?'))
    ax.set_ylim(0, 100)
axes[0].set_ylabel('forward MPPI success (%)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h + [plt.Line2D([], [], marker='_', ms=12, ls='none', color='#0b0b0b', mew=1.4)], l + ['same field, official chart criterion'],
           loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
fig.suptitle('Success by start-goal relative rotation: original vs smooth-graph fields (mean ± sd over seeds)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, 'success_by_angle.png'), dpi=140, bbox_inches='tight')
plt.close(fig)

# ---- Figure 2: special T sets, overall success, SE(3) for ours vs chart for original; non-convergence
special = [('Tshape3d_env1', 'standard'), ('Tshape3d_env1_bigangle', 'bigangle\n(both ends 135-180 deg)'),
           ('Tshape3d_env1_rotheavy', 'rotation-heavy'), ('Tshape3d_env1_transonly', 'translation only'),
           ('Tshape3d_env1_tight_3d', 'tight clearance')]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, (key3, keyc, title) in zip(axes, [('succ3', 'succ', 'success (%)'), ('nc3', 'nc', 'non-convergence (%)')]):
    present = [m for m in models if any(group_key('T', m) in S.get(s, {}) for s, _ in special)]
    w = 0.8 / max(len(present), 1)
    for i, m in enumerate(present):
        xs, ys, es = [], [], []
        for j, (s, _) in enumerate(special):
            row = S.get(s, {}).get(group_key('T', m))
            if row is None:
                continue
            v = val(row, keyc if m == 'original' else key3)
            xs.append(j + (i - (len(present) - 1) / 2) * w)
            ys.append(v[0])
            es.append(v[1])
        ax.bar(xs, ys, w * 0.9, color=COL[m], yerr=es, capsize=2, error_kw={'lw': 0.8}, label=LABEL[m])
    ax.set_xticks(range(len(special)))
    ax.set_xticklabels([t for _, t in special], fontsize=8)
    ax.set_ylabel(title)
axes[0].legend(fontsize=7, frameon=False, loc='lower left')
fig.suptitle('T-shape test suites: original (chart criterion) vs ours (SE(3) criterion); mean ± sd over seeds', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, 'special_sets.png'), dpi=140, bbox_inches='tight')
plt.close(fig)

# ---- Figure 3: free-space planning by angle (gradient planner): success and cost ratio
fp = os.path.join(args.root, 'summary', 'free_summary.json')
if os.path.exists(fp):
    F = json.load(open(fp))
    fbins = ['0-90', '90-150', '150-170', '170-179', '179-180', 'pure_rot_170-180']
    FC = {'graph_gram': ('#eda100', 'graph_gram'), 'tri_eq': ('#6250d6', 'graph_tri, equivariant V'),
          'trimetric_eq': ('#9085e9', 'graph_tri_metric, equivariant V'), 'gram_proj': ('#c98500', 'graph_gram read by the exact head'),
          'metric_best_recipe': ('#eb6834', 'metric head, best recipe'), 'metric_ntfields_loss': ('#d95926', 'metric head, NTFields loss'),
          'scalar': ('#52514e', 'direct scalar network')}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, lab, scale) in zip(axes, [('success', 'success (%)', 100), ('cost_ratio_p50', 'path cost / optimal (median)', 1),
                                           ('cost_ratio_p95', 'path cost / optimal (95th pct)', 1)]):
        for g, (c, l) in FC.items():
            if g not in F:
                continue
            ys = [F[g]['gradient'][b][key] for b in fbins]
            ax.errorbar(range(len(fbins)), [scale * y[0] if y else np.nan for y in ys], yerr=[scale * y[1] if y else 0 for y in ys],
                        color=c, marker='o', ms=4, lw=1.6, capsize=2, label=l)
        ax.set_xticks(range(len(fbins)))
        ax.set_xticklabels(['0-90', '90-150', '150-170', '170-179', '179-180', 'pure rot\n170-180'], fontsize=8)
        ax.set_ylabel(lab)
        ax.set_xlabel('relative rotation (deg)')
    axes[0].legend(fontsize=7, frameon=False, loc='lower left')
    fig.suptitle('Obstacle-free SE(3), gradient planner on the decoded time, 500 pairs per bin, mean ± sd over seeds', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'free_by_angle.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

# ---- Figure 4: tau slices through the seam / cut locus
sp = os.path.join(args.root, 'slices', 'slices.json')
if os.path.exists(sp):
    J = json.load(open(sp))
    phi = np.degrees(np.array(J['phi']))
    SC = {'original_metric': ('#eb6834', 'original metric NTFields'), 'ours_metric_head': ('#2a78d6', 'ours: metric head on G(x)'),
          'ours_graph_metric': ('#1baf7a', 'ours: graph_metric'), 'ours_graph_tri_eq': ('#6250d6', 'ours: graph_tri_metric eq. V'),
          'ours_graph_gram': ('#eda100', 'ours: graph_gram')}
    recs = J['anchors'][:4]
    fig, axes = plt.subplots(len(recs), 1, figsize=(8, 2.4 * len(recs)), sharex=True)
    for ax, rec in zip(np.atleast_1d(axes), recs):
        for k, (c, l) in SC.items():
            if k in rec['tau']:
                ax.plot(phi, rec['tau'][k], color=c, lw=1.4, label=l)
        for s in rec['seam_phi_deg']:
            ax.axvline(s, color='#52514e', ls=':', lw=1)
        ax.axvline(180, color='#0b0b0b', lw=0.6)
        ax.set_ylabel('tau')
        ax.set_title('test case {}: goal rotation angle {:.0f} deg (dotted: chart seam; solid: half-turn)'.format(rec['case'], rec['goal_theta_deg']), fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel('rotation of the start about a body axis, from the goal (deg)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'slices.png'), dpi=140)
    plt.close(fig)
print('figures in', out)
