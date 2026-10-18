"""Slices through trained obstacle fields: the start is rotated at the goal's position about a fixed axis through one
full turn (rotation slice), or moved along a straight line through the scene (translation slice), and tau(start, goal)
of several heads is plotted together with the speed-limit time T_free (a lower bound for the true time).

    python models/graph_branch/scripts/obstacle_slices.py --scene Tshape3d_env1 --out results/graph_branch/figures/T_slices.png \
        ggram=Experiments/graph_branch/T_ggram_s0/latest.pt bc4=Experiments/graph_branch/T_bc4_s0/latest.pt \
        orig=orig:/lieshape/Experiments/3dshape_metric_baseline/<run>/Model_Epoch_04000_<...>.pt

Labels are keys of style.py (orig, ggram, gmetric, br4, bc4, bc4e, ...); 'orig:' marks a checkpoint of the original
metric trainer (chart input, loaded with chart_metric.load_original). Writes the PNG and a vector PDF next to it.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.append('.')

from models.graph_branch import chart_metric  # noqa: E402
from models.graph_branch.scripts import style  # noqa: E402
from models.graph_method_triangle import branch_decoder as bd  # noqa: E402
from models.graph_native import _sib  # noqa: E402
from models.graph_native import model_train_metric as gn_train  # noqa: E402

TWO_PI = 2.0 * math.pi
se3 = _sib.se3


def load(ck, dev):
    if ck.startswith('orig:'):
        return chart_metric.load_original(ck[5:], dev)
    m = gn_train.Model('/tmp/sl', 'datasets/3dshape/data/Tshape3d_env1', 6, [0.0] * 6, device=dev)
    m.load(ck)
    m.network.eval()
    return m.network


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--width', type=float, default=7.0, help='figure width in inches')
    ap.add_argument('models', nargs='+', help='label=checkpoint')
    a = ap.parse_args()
    style.rc()
    nets = [(s.split('=', 1)[0], load(s.split('=', 1)[1], a.device)) for s in a.models]
    test = np.load(os.path.join('/media/jeffrey/Lie-Shape-NTFields/testing_data/3dshape', a.scene, 'sampled_points.npy')
                   if os.path.exists('/media/jeffrey/Lie-Shape-NTFields') else
                   os.path.join('/lieshape/testing_data/3dshape', a.scene, 'sampled_points.npy'))
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(test), generator=g)[:a.n].tolist()
    phi = torch.linspace(0, TWO_PI, 721, device=a.device)
    fig, axes = plt.subplots(2, a.n, figsize=(a.width, a.width * 0.64))
    for j, i in enumerate(idx):
        goal = torch.tensor(test[i, 6:], dtype=torch.float32, device=a.device)
        start = torch.tensor(test[i, :6], dtype=torch.float32, device=a.device)
        tg, Rg = goal[None, :3], se3.so3_exp(TWO_PI * goal[None, 3:6])
        ax_dir = torch.randn(3, generator=g).to(a.device)
        ax_dir = ax_dir / ax_dir.norm()
        Rs = Rg @ se3.so3_exp(phi[:, None] * ax_dir[None])
        gs = se3.features(tg.expand(len(phi), 3), Rs)
        gg = se3.features(tg, Rg).expand(len(phi), 12)
        ax = axes[0, j]
        ax.plot(phi.cpu() * 180 / math.pi, bd.free_time(gs, gg, 1.0, TWO_PI).cpu(), color='k', ls=':', lw=1.0,
                label=style.LABEL['tfree'])
        for lab, net in nets:
            ax.plot(phi.cpu() * 180 / math.pi, net.net.value(gs, gg).cpu(), color=style.COLOR.get(lab), lw=1.1,
                    ls=style.DASH.get(lab, '-'), label=style.LABEL.get(lab, lab))
        ax.axvline(180, color='grey', lw=0.5, ls='-.')
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel('start rotation (deg)')
        ax.set_title('case {}, rotation slice'.format(i))
        # translation slice: start position moved along start -> goal and beyond, start rotation fixed
        lam = torch.linspace(-0.3, 1.3, 641, device=a.device)
        ts = start[None, :3] + lam[:, None] * (goal[None, :3] - start[None, :3])
        Rst = se3.so3_exp(TWO_PI * start[None, 3:6]).expand(len(lam), 3, 3)
        gs2 = se3.features(ts, Rst)
        gg2 = gg[:1].expand(len(lam), 12)
        ax = axes[1, j]
        ax.plot(lam.cpu(), bd.free_time(gs2, gg2, 1.0, TWO_PI).cpu(), color='k', ls=':', lw=1.0)
        for lab, net in nets:
            ax.plot(lam.cpu(), net.net.value(gs2, gg2).cpu(), color=style.COLOR.get(lab), lw=1.1,
                    ls=style.DASH.get(lab, '-'))
        ax.set_xlabel('start position s')
        ax.set_title('case {}, translation slice'.format(i))
    axes[0, 0].set_ylabel('T(start, goal)')
    axes[1, 0].set_ylabel('T(start, goal)')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=min(3, len(l)), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 * ((len(l) + 2) // 3)), h_pad=1.2, w_pad=0.6)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches='tight')
    fig.savefig(os.path.splitext(a.out)[0] + '.pdf', bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
