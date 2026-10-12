"""Figures for the obstacle-free comparison.

    python models/graph_method/scripts/make_figures.py --exp Experiments/graph_method_final \
        --res results/graph_method_final --out results/graph_method_final/figures
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from models.graph_method import model_train_graph as mt
from models.graph_method.fields import oracle
from models.graph_method.geometry import se3

MODELS = [('graph_gram', 'graph_gram (ours)', '#1f77b4'), ('graph_sqdist', 'graph_sqdist (ours, variant)', '#17becf'),
          ('metric_best_recipe', 'G-Shape metric head', '#d62728'), ('scalar', 'direct scalar T', '#7f7f7f')]
ANGLE_BINS = ['0-90', '90-150', '150-170', '170-179', '179-180']


def slice_values(f, device):
    n = 721
    th = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    tx = torch.zeros(n, 3, dtype=torch.float64)
    ty = torch.full((n, 3), 0.3, dtype=torch.float64)
    Rx = se3.so3_exp(torch.tensor([[0.3, -0.5, 0.2]], dtype=torch.float64)).expand(n, 3, 3)
    Ry = Rx @ se3.so3_exp(th[:, None] * torch.tensor([0.0, 0.6, 0.8], dtype=torch.float64))
    Ts = oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0)
    c = lambda v: v.to(device=device, dtype=f.dtype)
    with torch.no_grad():
        T = f.network.value(se3.features(c(tx), c(Rx)), se3.features(c(ty), c(Ry))).double().cpu()
    return th, Ts, T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', required=True)
    p.add_argument('--res', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1) cut-locus slice, seed 0 of each model
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    th = Ts = None
    for key, label, col in MODELS:
        ck = os.path.join(a.exp, key + '_s0', 'best.pt')
        if not os.path.exists(ck):
            continue
        f, _ = mt.load_function(ck, device=a.device)
        th, Ts, T = slice_values(f, a.device)
        ax[0].plot(th.numpy(), T.numpy(), color=col, lw=1.4, label=label)
        ax[1].plot(th.numpy(), (T - Ts).numpy(), color=col, lw=1.2, label=label)
    ax[0].plot(th.numpy(), Ts.numpy(), 'k--', lw=1.0, label='oracle T*')
    for x in ax:
        x.axvline(math.pi, color='gray', lw=0.5)
        x.set_xlabel('rotation of the goal about a fixed axis (rad)')
    ax[0].set_ylabel('travel time T')
    ax[1].set_ylabel('T - T*')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Slice crossing the rotational cut locus (pi)')
    ax[1].set_title('Error along the slice')
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'cut_locus_slice_comparison.png'), dpi=130)
    plt.close(fig)

    # 2) error by relative angle (mean +- std over seeds of the per-bin median relative error)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.8 / len(MODELS)
    for i, (key, label, col) in enumerate(MODELS):
        vals = []
        for s in (0, 1, 2):
            mpath = os.path.join(a.res, '{}_s{}'.format(key, s), 'metrics.json')
            if os.path.exists(mpath):
                m = json.load(open(mpath))
                vals.append([m['angle_{}_rel_p50'.format(b)] * 100 for b in ANGLE_BINS])
        if not vals:
            continue
        v = torch.tensor(vals)
        mean, std = v.mean(0), (v.std(0) if v.shape[0] > 1 else torch.zeros(v.shape[1]))
        xs = [j + (i - len(MODELS) / 2 + 0.5) * w for j in range(len(ANGLE_BINS))]
        ax.bar(xs, mean.tolist(), w, yerr=std.tolist(), color=col, label=label, capsize=2)
    ax.set_yscale('log')
    ax.set_xticks(range(len(ANGLE_BINS)))
    ax.set_xticklabels([b + ' deg' for b in ANGLE_BINS])
    ax.set_ylabel('median relative error (%)')
    ax.set_title('Held-out error by relative rotation angle (3 seeds)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'error_by_angle.png'), dpi=130)
    plt.close(fig)
    print('figures written to', a.out)


if __name__ == '__main__':
    main()
