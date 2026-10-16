"""Pure-rotation slices of the slow-sector problems (A = 4, 20) for the methodology report: every head against
the exact travel time T*, label-fitted (synth_fit.py) or trained by the Eikonal loss only (synth_dataset.py runs).

    python models/graph_branch/scripts/pdf_synth_slices.py --mode fit  --out results/graph_branch/report/latex/figs/fig_synth_fit.pdf
    python models/graph_branch/scripts/pdf_synth_slices.py --mode phys --out results/graph_branch/report/latex/figs/fig_synth_phys.pdf
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.append('.')

from models.graph_branch import chart_metric  # noqa: E402,F401  (arch 'metric_chart')
from models.graph_branch import synth  # noqa: E402
from models.graph_branch.scripts import style  # noqa: E402
from models.graph_branch.scripts.synth_figures import load_net, load_phys  # noqa: E402
from models.graph_branch.scripts.synth_fit import value  # noqa: E402

TWO_PI = 2.0 * math.pi
FIT = ['morig', 'ggram', 'gmetric', 'br4', 'bc4']
PHYS = ['morig05', 'ggram', 'gmet05', 'br4s', 'br4Hs']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fit', 'phys'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    style.rc()
    probs = [4.0, 20.0]
    starts = [0.0, 0.5 * math.pi]
    fig, axes = plt.subplots(len(probs), len(starts), figsize=(7.0, 4.7))
    rel = torch.linspace(0, TWO_PI, 721, dtype=torch.float64)
    for i, A in enumerate(probs):
        prob = synth.Problem('sector', A=A)
        nets = {}
        for h in (FIT if a.mode == 'fit' else PHYS):
            if a.mode == 'fit':
                p = 'results/graph_branch/synth/sector_A{:g}/{}_s{}.pt'.format(A, h, a.seed)
                if os.path.exists(p):
                    nets[h] = load_net(p, a.device)
            else:
                p = 'Experiments/graph_branch/S{:g}_{}_aniso_s{}/latest.pt'.format(A, h, a.seed)
                if os.path.exists(p):
                    nets[h] = load_phys(p, a.device)
        for j, ps in enumerate(starts):
            ax = axes[i, j]
            st_ = torch.tensor([[0.3, -0.35, ps]], dtype=torch.float64).repeat(len(rel), 1)
            go = torch.cat([st_[:, :2], torch.remainder(ps + rel, TWO_PI)[:, None]], 1)
            deg = rel * 180 / math.pi
            with torch.no_grad():
                ax.plot(deg, prob.T_star(st_, go), color='k', lw=2.0, label=style.LABEL['exact'])
                for h, net in nets.items():
                    ax.plot(deg, value(net, st_, go, a.device), color=style.COLOR[h], lw=1.1,
                            ls=style.DASH.get(h, '-'), label=style.LABEL[h])
            r = float(prob.ridge_psi(torch.tensor([ps], dtype=torch.float64))[0])
            ax.axvline(((r - ps) % TWO_PI) * 180 / math.pi, color='k', lw=0.5, ls=':')
            ax.axvline(180, color='grey', lw=0.5, ls='-.')
            ax.axhline(0.5, color=style.COLOR['ggram'], lw=0.5, ls=':')
            ax.set_xlim(0, 360)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_title('A = {:g}, start yaw {:.0f} deg'.format(A, ps * 180 / math.pi))
            if i == len(probs) - 1:
                ax.set_xlabel('goal yaw relative to the start (deg)')
            if j == 0:
                ax.set_ylabel('pure-rotation travel time')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(os.path.splitext(a.out)[0] + '.png', dpi=150)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
