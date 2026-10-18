"""Rotation slices of a trained homotopy field: the two sheets, the base minimum and the ridge.

    python models/graph_homotopy/scripts/ridge_slice.py --ckpt Experiments/graph_homotopy/T_hrand_s0/latest.pt \
        --dataPath /lieshape/testing_data/3dshape/Tshape3d_env1 --out results/graph_homotopy/figures/T_hrand_s0

For a few test cases (start s, goal g): (1) along the rotation slice theta in [0, 2 pi] about the axis of the
relative rotation, at the start's translation, T~(x(theta), +g~), T~(x(theta), -g~), their minimum and T_free;
(2) a 2-D slice (translation along the start -> goal line x rotation angle): the base field with the ridge
(sheet tie) drawn as a contour. The free-space ridge is the half turn (vertical line at theta = pi); the trained
ridge moves where obstacles make one homotopy class slower.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_method_triangle import homotopy_decoder as hd  # noqa: E402
from models.graph_method_triangle.geometry import se3  # noqa: E402
from models.graph_native import model_train_metric as md  # noqa: E402

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataPath', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cases', default='0,1,2,3,4,5')
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = md.Model('/tmp/slice_model', 'datasets/3dshape/Tshape3d_env1', 6, [0.0] * 6, device=a.device)
    m.load(a.ckpt)
    head = m.network.net
    arr = np.load(os.path.join(a.dataPath, 'sampled_points.npy')).astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    cases = [int(c) for c in a.cases.split(',')]
    stats = {}
    nth, nx = 181, 121
    for ci in cases:
        s, g = torch.tensor(arr[ci, :6]), torch.tensor(arr[ci, 6:])
        qs, qg = hd.quat_exp_turns(s[None, 3:]), hd.quat_exp_turns(g[None, 3:])
        rel = hd.relative_quat(qs, qg)[0]
        axis = rel[1:] / rel[1:].norm().clamp_min(1e-9)
        # rotation slice about the relative axis, starting at the start orientation, at the start translation
        th = torch.linspace(0, TWO_PI, nth)
        dq = torch.cat([torch.cos(th / 2)[:, None], torch.sin(th / 2)[:, None] * axis[None]], 1)
        qx = hd.quat_mul(qs.expand(nth, 4), dq)
        tx = s[None, :3].expand(nth, 3)
        tg = g[None, :3].expand(nth, 3)
        with torch.no_grad():
            dev = a.device
            tp = head.value_cover(tx.to(dev), qx.to(dev), tg.to(dev), qg.expand(nth, 4).to(dev)).cpu()
            tm = head.value_cover(tx.to(dev), qx.to(dev), tg.to(dev), -qg.expand(nth, 4).to(dev)).cpu()
            tf = head.free_time(se3.features(tx, se3.quat_to_matrix(qx)).to(dev), se3.features(tg, se3.quat_to_matrix(qg.expand(nth, 4))).to(dev)).cpu()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(th * 180 / math.pi, tp, label='sheet +g~')
        ax[0].plot(th * 180 / math.pi, tm, label='sheet -g~')
        ax[0].plot(th * 180 / math.pi, torch.minimum(tp, tm), 'k--', lw=1, label='min (base field)')
        ax[0].plot(th * 180 / math.pi, tf, ':', color='gray', label='T_free')
        sw = torch.nonzero((tp[1:] <= tm[1:]) != (tp[:-1] <= tm[:-1])).flatten()
        for k in sw.tolist():
            ax[0].axvline(float(th[k]) * 180 / math.pi, color='r', lw=0.8)
        ax[0].set_xlabel('rotation about the relative axis from the start (deg)')
        ax[0].set_ylabel('time')
        ax[0].set_title('case {}: rotation slice (red: ridge)'.format(ci))
        ax[0].legend(fontsize=7)
        # 2-D slice: translation along the start -> goal line (lambda in [-0.25, 1.25]) x rotation angle
        lam = torch.linspace(-0.25, 1.25, nx)
        T2 = torch.zeros(nx, nth)
        R2 = torch.zeros(nx, nth)
        with torch.no_grad():
            for i, l in enumerate(lam):
                t_i = (s[:3] + l * (g[:3] - s[:3]))[None].expand(nth, 3)
                p_ = head.value_cover(t_i.to(dev), qx.to(dev), tg.to(dev), qg.expand(nth, 4).to(dev)).cpu()
                m_ = head.value_cover(t_i.to(dev), qx.to(dev), tg.to(dev), -qg.expand(nth, 4).to(dev)).cpu()
                T2[i] = torch.minimum(p_, m_)
                R2[i] = p_ - m_
        im = ax[1].imshow(T2.numpy(), origin='lower', aspect='auto', extent=[0, 360, -0.25, 1.25], cmap='viridis')
        ax[1].contour(th.numpy() * 180 / math.pi, lam.numpy(), R2.numpy(), levels=[0.0], colors='r', linewidths=1)
        ax[1].axhline(1.0, color='w', lw=0.5, ls='--')
        ax[1].set_xlabel('rotation (deg)')
        ax[1].set_ylabel('position along start -> goal (1 = goal)')
        ax[1].set_title('base field, ridge in red (free-space ridge: half turn)')
        plt.colorbar(im, ax=ax[1])
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, 'case{}.png'.format(ci)), dpi=110)
        plt.close(fig)
        ridge_deg = [float(th[k]) * 180 / math.pi for k in sw.tolist()]
        # on this slice the goal orientation sits at theta = psi (the relative angle) and the free-space ridge at
        # theta = psi + 180 (the orientation antipodal to the goal about the axis)
        psi_deg = float(2.0 * torch.atan2(rel[1:].norm(), rel[0])) * 180 / math.pi
        free_ridge = (psi_deg + 180.0) % 360.0
        off = [min(abs(r - free_ridge), 360.0 - abs(r - free_ridge)) for r in ridge_deg]
        ax[0].axvline(free_ridge, color='gray', lw=0.8, ls='--')
        fig.savefig(os.path.join(a.out, 'case{}.png'.format(ci)), dpi=110)
        stats[ci] = {'ridge_deg_on_slice': ridge_deg, 'n_switch': len(ridge_deg), 'max_sheet_diff': float((tp - tm).abs().max()),
                     'psi_deg': psi_deg, 'free_ridge_deg': free_ridge, 'ridge_offset_from_free_ridge_deg': off}
    with open(os.path.join(a.out, 'ridge_stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
