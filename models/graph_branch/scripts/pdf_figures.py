"""Vector figures of the methodology report (results/graph_branch/report/latex): planning success, failure
partition, cone calibration at the goal, and an exact illustration of the decoders on the slow-sector problem.

    python models/graph_branch/scripts/pdf_figures.py --out results/graph_branch/report/latex/figs

Reads results/graph_branch/summary/{plan_summary,cone_calibration_T}.json (aggregate_branch.py,
cone_calibration.py) and the per-case planning records of every run (for the success / collision / non-convergence
partition). Every head is drawn in the colours of style.py, the two metric baselines included.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append('.')

from models.graph_branch.scripts import style  # noqa: E402

SUM = 'results/graph_branch/summary'
SETS = [('Tshape3d_env1', 'T-shape'), ('Lshape3d_env3', 'L-shape'), ('Ashape3d_env4', 'A-shape'),
        ('Tshape3d_env1_bigangle', 'T big-angle'), ('Tshape3d_env1_rotheavy', 'T rotation-heavy')]
MAIN = ['orig', 'ggram', 'gmetric', 'bc4', 'bc4e']


def run_dirs(head, scene_tag):
    """Per-case result directories of one head on one scene tag (T, L, A)."""
    if head == 'orig':
        return sorted(glob.glob('results/graph_method_cusp/plan/{}_native_s[0-9]'.format(scene_tag)))
    if head == 'gmetric':
        return sorted(glob.glob('results/graph_lie/plan/{}_gmetric_iso_lie_s[0-9]'.format(scene_tag)))
    return sorted(glob.glob('results/graph_branch/plan/{}_{}_s[0-9]'.format(scene_tag, head)))


def partition(head, s):
    """Mean over seeds of (success, collision, non-converged without collision) in % (they sum to 100)."""
    tag = s[0]
    out = []
    for d in run_dirs(head, tag):
        p = os.path.join(d, s + '.json')
        if not os.path.exists(p):
            continue
        J = json.load(open(p))
        if J['summary'].get('batch') != 125:
            continue
        R = J['records']
        n = len(R)
        succ = sum(r['success_se3'] for r in R) / n
        coll = sum(r['collision_se3'] for r in R) / n
        nonc = sum((not r['converged_se3']) and (not r['collision_se3']) for r in R) / n
        out.append((100 * succ, 100 * coll, 100 * nonc))
    if not out:
        return None
    return [st.mean(x[i] for x in out) for i in range(3)], len(out)


def fig_success(plan, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    W = 0.16
    for gi, (s, lab) in enumerate(SETS):
        for hi, h in enumerate(MAIN):
            d = plan.get(s, {}).get(h)
            if not d:
                continue
            v = [100 * x for x in d['success_rate_se3'] if x is not None]
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else 0.0
            x = gi + (hi - (len(MAIN) - 1) / 2) * W
            ax.bar(x, m, W * 0.92, color=style.COLOR[h], label=style.LABEL[h] if gi == 0 else None, zorder=2)
            ax.errorbar(x, m, yerr=sd, color='k', lw=0.7, capsize=1.5, zorder=3)
            ax.text(x, m + sd + 1.0, '{:.0f}'.format(m), ha='center', va='bottom', fontsize=6)
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([lab for _, lab in SETS])
    ax.set_ylim(25, 104)
    ax.set_ylabel('SE(3) success (%)')
    ax.grid(axis='y', lw=0.4, color='0.85', zorder=0)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.30))
    fig.savefig(os.path.join(out, 'fig_success.pdf'))
    plt.close(fig)


def fig_failures(out):
    heads = ['orig', 'ggram', 'gmetric', 'ggc', 'br1', 'br4', 'bc4', 'bc4e']
    sets = [('Tshape3d_env1', 'T-shape (1000 cases)'), ('Tshape3d_env1_bigangle', 'T big-angle (191)'),
            ('Tshape3d_env1_rotheavy', 'T rotation-heavy (196)')]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, (s, lab) in zip(axes, sets):
        for i, h in enumerate(heads):
            r = partition(h, s)
            if r is None:
                continue
            (sc, cc, nc), n = r
            y = len(heads) - 1 - i
            ax.barh(y, sc, color=style.COLOR[h], height=0.72)
            ax.barh(y, cc, left=sc, color='#4d4d4d', height=0.72)
            ax.barh(y, nc, left=sc + cc, color='#bdbdbd', height=0.72)
            ax.text(sc - 1.5, y, '{:.0f}'.format(sc), ha='right', va='center', fontsize=6, color='white')
        ax.set_xlim(0, 100)
        ax.set_title(lab)
        ax.set_xlabel('% of cases')
    axes[0].set_yticks(range(len(heads)))
    axes[0].set_yticklabels([style.SHORT[h] for h in reversed(heads)])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='0.55', label='success (bar colour = head)'), Patch(color='#4d4d4d', label='collision'),
                        Patch(color='#bdbdbd', label='not converged, no collision')],
               ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, 'fig_failures.pdf'))
    plt.close(fig)


def fig_cone(out):
    p = os.path.join(SUM, 'cone_calibration_T.json')
    if not os.path.exists(p):
        return
    C = json.load(open(p))
    heads = ['orig', 'ggram', 'gmetric', 'br4', 'bc4', 'bc4e']
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, eps in zip(axes, ('0.01', '0.0001')):
        for h in heads:
            runs = [v for k, v in C['models'].items() if k.rsplit('_s', 1)[0] == h]
            if not runs:
                continue
            nb = len(C['edges']) - 1
            xs, ys, lo, hi = [], [], [], []
            for b in range(nb):
                bins = [r['eps_' + eps]['bins'][b] for r in runs if r['eps_' + eps]['bins'][b]]
                if len(bins) < len(runs):
                    continue
                xs.append(st.mean(x['S_median'] for x in bins))
                ys.append(st.mean(x['median'] for x in bins))
                lo.append(st.mean(x['q25'] for x in bins))
                hi.append(st.mean(x['q75'] for x in bins))
            ax.plot(xs, ys, marker='o', ms=2.8, color=style.COLOR[h], ls=style.DASH.get(h, '-'), label=style.LABEL[h])
            ax.fill_between(xs, lo, hi, color=style.COLOR[h], alpha=0.12, lw=0)
        ax.axhline(1.0, color='k', lw=0.6, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('local speed S at the goal (own speed model)')
        ax.set_title('slope measured at distance {} from the goal'.format('$10^{-2}$' if eps == '0.01' else '$10^{-4}$'))
        ax.grid(lw=0.4, color='0.88', which='both')
    axes[0].set_ylabel('S x cone slope  (Eikonal limit = 1)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_cone.pdf'))
    plt.close(fig)


def fig_decoders(out):
    """Exact pure-rotation slice of the slow-sector problem (A = 4, start yaw 90 deg) and the conformal term."""
    import torch
    from models.graph_branch import synth
    prob = synth.Problem('sector', A=4.0)
    TWO_PI = 2 * math.pi
    ps = 0.5 * math.pi
    rel = torch.linspace(0, TWO_PI, 1441, dtype=torch.float64)          # goal yaw relative to the start
    psi = ps + rel
    H = prob.H
    Hs = H(torch.tensor([ps], dtype=torch.float64))[0]
    Hp = torch.where(psi <= TWO_PI, H(torch.clamp(psi, max=TWO_PI)), prob.H_tot + H(torch.clamp(psi - TWO_PI, min=0)))
    ccw = (Hp - Hs) / TWO_PI                                            # time along the counter-clockwise route
    cw = prob.H_tot / TWO_PI - ccw                                      # the other route
    tf_ccw, tf_cw = rel / TWO_PI, 1.0 - rel / TWO_PI
    deg = rel * 180 / math.pi
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    ax.plot(deg, torch.minimum(tf_ccw, tf_cw), color='k', ls=':', lw=1.0, label='T_free = min over the two routes (free space)')
    ax.plot(deg, ccw, color='#0e7a71', ls='--', lw=1.0, label='counter-clockwise route: its T_free + residual')
    ax.plot(deg, cw, color='#3b63ad', ls='--', lw=1.0, label='clockwise route: its T_free + residual')
    ax.plot(deg, torch.minimum(ccw, cw), color='k', lw=1.8, label='exact T* = min over routes')
    ax.axhline(0.5, color=style.COLOR['ggram'], lw=0.9, label='graph_gram: largest pure-rotation time (cap)')
    r = (float(prob.ridge_psi(torch.tensor([ps], dtype=torch.float64))[0]) - ps) * 180 / math.pi % 360
    ax.axvline(r, color='k', lw=0.5, ls='-.')
    ax.axvline(180, color='grey', lw=0.5, ls='-.')
    ax.annotate('ridge of T*\n(routes tie)', xy=(r, 0.9), xytext=(r - 70, 1.25), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', lw=0.5))
    ax.annotate('half turn: the only\npossible kink of\ngraph_gram', xy=(180, 0.5), xytext=(205, 0.95), fontsize=6.5,
                arrowprops=dict(arrowstyle='->', lw=0.5))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 1.85)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('goal yaw relative to the start (deg); start yaw 90 deg')
    ax.set_ylabel('pure-rotation travel time')
    ax.set_title('(a) slow rotation sector (A = 4): routes and ridge')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    tf = np.linspace(0, 0.35, 400)
    for rho, lam, c in ((1.5, 0.05, '#0e7a71'), (1.5, 0.15, '#3fb8ab'), (0.3, 0.05, '#8fa8db')):
        ax.plot(tf, tf + rho * lam * np.tanh(tf / lam), color=c, lw=1.2,
                label='rho = {}, lambda = {}'.format(rho, lam))
        ax.plot(tf[:60], (1 + rho) * tf[:60], color=c, lw=0.6, ls=':')
    ax.plot(tf, tf, color='k', lw=0.8, ls='--', label='strict cone: T = T_free')
    ax.set_xlabel('T_free')
    ax.set_ylabel('T_free + C')
    ax.set_title('(b) conformal term (slope 1 + rho at the goal)')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_decoders.pdf'))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/graph_branch/report/latex/figs')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style.rc()
    plan = json.load(open(os.path.join(SUM, 'plan_summary.json')))
    fig_success(plan, a.out)
    fig_failures(a.out)
    fig_cone(a.out)
    fig_decoders(a.out)
    print('wrote figures to', a.out)


if __name__ == '__main__':
    main()
