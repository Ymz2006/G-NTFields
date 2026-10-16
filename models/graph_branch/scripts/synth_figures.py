"""Figures for the synthetic exact-truth benchmark: rotation slices (goal yaw over one turn, start fixed, same
position) of every fitted head against the exact T*, and the translational slice behind the disc.

    python models/graph_branch/scripts/synth_figures.py --dir results/graph_branch/synth/sector_A4 --seed 0
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.append('.')

from models.graph_branch import synth  # noqa: E402
from models.graph_branch import chart_metric  # noqa: E402,F401  (arch 'metric_chart')
from models.graph_branch.scripts.synth_fit import HEADS, value  # noqa: E402
from models.graph_native import model_network_metric as gn_network  # noqa: E402

TWO_PI = 2.0 * math.pi
COL = {'ggram': 'tab:red', 'gmetric': 'tab:orange', 'br1': 'tab:blue', 'br4': 'tab:green', 'br4P': 'tab:purple', 'br4H': 'tab:blue',
       'br4L': 'tab:brown', 'bc4': 'tab:cyan'}


def load_net(path, device):
    ck = torch.load(path, map_location=device)
    net = gn_network.NN(device, 6, None, graph=ck['graph'])
    net.load_state_dict(ck['state'])
    return net.to(device).float().eval()


def load_phys(ck, device):
    from models.graph_native import model_train_metric as gn_train
    m = gn_train.Model('/tmp/sf', 'datasets/3dshape/Tshape3d_env1', 6, [0.0] * 6, device=device)
    m.load(ck)
    return m.network.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='supervised results dir (results/graph_branch/synth/<kind>_A<A>)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--phys', nargs='*', default=None,
                    help='label=checkpoint of physics-trained runs: plot these instead of the supervised fits')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    tag = os.path.basename(a.dir.rstrip('/'))
    kind, A = tag.split('_A')
    prob = synth.Problem(kind, A=float(A))
    nets = {}
    if a.phys:
        for s in a.phys:
            lab, ck = s.split('=', 1)
            nets[lab] = load_phys(ck, a.device)
    else:
        for p in sorted(glob.glob(os.path.join(a.dir, '*_s{}.pt'.format(a.seed)))):
            h = os.path.basename(p).rsplit('_s', 1)[0]
            nets[h] = load_net(p, a.device)
    psi = torch.linspace(0, TWO_PI, 721, dtype=torch.float64)
    starts = [0.0, 0.5 * math.pi, 0.25 * math.pi] if prob.A > 0 else [0.0, 0.5 * math.pi]
    ncol = len(starts) + (1 if prob.r0 > 0 else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.0))
    with torch.no_grad():
        for ax, ps in zip(axes, starts):
            st = torch.tensor([[0.3, -0.35, ps]], dtype=torch.float64).repeat(len(psi), 1)
            go = torch.cat([st[:, :2], psi[:, None]], 1)
            Ts = prob.T_star(st, go)
            ax.plot(psi * 180 / math.pi, Ts, 'k-', lw=2.2, label='exact T*')
            for i, (h, net) in enumerate(nets.items()):
                ax.plot(psi * 180 / math.pi, value(net, st, go, a.device), '--', color=COL.get(h, 'C{}'.format(i)), lw=1.3,
                        label=h)
            r = prob.ridge_psi(torch.tensor([ps], dtype=torch.float64))[0] if prob.A > 0 else ps + math.pi
            ax.axvline(float(torch.remainder(r, TWO_PI)) * 180 / math.pi, color='k', lw=0.6, ls=':')
            ax.axvline(((ps + math.pi) % TWO_PI) * 180 / math.pi, color='grey', lw=0.6, ls='-.')
            ax.axhline(0.5, color='tab:red', lw=0.5, ls=':')
            ax.set_xlabel('goal yaw (deg); start yaw {:.0f} deg'.format(ps * 180 / math.pi))
            ax.set_ylabel('travel time')
            ax.set_title('pure rotation slice (dotted: true ridge, dash-dot: half turn)', fontsize=8)
        if prob.r0 > 0:
            ax = axes[-1]
            ang = torch.linspace(-math.pi, math.pi, 721, dtype=torch.float64)
            st = torch.tensor([[0.4, 0.05, 1.0]], dtype=torch.float64).repeat(len(ang), 1)
            go = torch.stack([0.4 * torch.cos(ang), 0.4 * torch.sin(ang), torch.full_like(ang, 1.0)], 1)
            ax.plot(ang * 180 / math.pi, prob.T_star(st, go), 'k-', lw=2.2, label='exact T*')
            for h, net in nets.items():
                ax.plot(ang * 180 / math.pi, value(net, st, go, a.device), '--', color=COL.get(h), lw=1.3, label=h)
            ax.set_xlabel('goal angle on the circle r = 0.4 around the disc (deg)')
            ax.set_title('translation slice behind the disc', fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle('{} (A = {}): {} vs exact travel time, seed {}'.format(
        kind, A, 'physics-trained fields (Eikonal only)' if a.phys else 'fitted heads', a.seed))
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, 'slices_s{}.png'.format(a.seed))
    fig.savefig(out, dpi=120)
    print('wrote', out)


if __name__ == '__main__':
    main()
