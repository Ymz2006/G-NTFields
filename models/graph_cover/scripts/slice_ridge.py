"""Ridges on a 2-D translation slice of an obstacle field (stage 1 among obstacles).

For a goal pose y = (p_g, R_g) of the test set, the start moves on the plane {z = p_g,z + dz} with the goal's
rotation: x(u, v) = (u, v, p_g,z + dz; R_g). On the grid the script records T(x, y), the Lie-gradient speed
S |c| (S from the exact clearance predictor of the scene), the detector's lambda_min and cross derivative r,
the detector mask, and -- for cover / branch heads -- the active group element (its boundaries are the
field's exact ridges). A smooth head's learned ridge is the zero set of r inside the band; the figure overlays
it on T and on the speed map.

    python models/graph_cover/scripts/slice_ridge.py --out results/graph_cover/slices/T_lp2_s0 \
        --ckpt Experiments/graph_cover/T_lp2_s0/latest.pt --clearance Experiments/graph_lie/speed_pred/Tshape3d_env1_clearance.pt \
        --goals 0,1,2 --testPath /lieshape/testing_data/3dshape/Tshape3d_env1 [--res 121]
"""
import argparse
import json
import math
import os
import sys

sys.path.append('.')

import numpy as np
import torch

from models.graph_cover import ridge
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_train_metric as gnm

TWO_PI = 2 * math.pi


def load_clearance(path, device):
    from models.graph_method_triangle import clearance as cl
    ck = torch.load(path, map_location=device)
    kw = ck.get('kw', {})
    pred = cl.ClearancePredictor(**kw)
    pred.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck, strict=True)
    return pred.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--clearance', default='')
    ap.add_argument('--testPath', default='/lieshape/testing_data/3dshape/Tshape3d_env1')
    ap.add_argument('--goals', default='0,1,2,3')
    ap.add_argument('--res', type=int, default=121)
    ap.add_argument('--half', type=float, default=0.5)
    ap.add_argument('--dz', type=float, default=0.0)
    ap.add_argument('--band', type=float, default=0.9)
    ap.add_argument('--kappa', type=float, default=1.0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    m = gnm.Model('.', '.', 6, [0.0] * 6, device=dev)
    m.load(a.ckpt)
    net = m.network
    head = net.net
    pred = load_clearance(a.clearance, dev) if a.clearance else None
    pts = np.load(os.path.join(a.testPath, 'sampled_points.npy'))
    lin = torch.linspace(-a.half, a.half, a.res, device=dev)
    U, V = torch.meshgrid(lin, lin, indexing='ij')
    summary = {}
    for gi in [int(s) for s in a.goals.split(',')]:
        row = torch.from_numpy(pts[gi]).float().to(dev)
        pg, Rg = row[6:9], se3.so3_exp(TWO_PI * row[9:12][None])[0]
        n = a.res * a.res
        tx = torch.stack([U.reshape(-1), V.reshape(-1), torch.full((n,), float(pg[2]) + a.dz, device=dev)], 1)
        Rx = Rg[None].expand(n, -1, -1)
        ty, Ry = pg[None].expand(n, -1), Rg[None].expand(n, -1, -1)
        T, c, H, S, elem = [], [], [], [], []
        for i in range(0, n, 2048):
            Ti, ci, Hi = ridge.body_derivatives(head.value, tx[i:i + 2048], Rx[i:i + 2048], ty[i:i + 2048],
                                                Ry[i:i + 2048], 1.0, TWO_PI)
            T.append(Ti); c.append(ci); H.append(Hi)
            with torch.no_grad():
                g = se3.features(tx[i:i + 2048], Rx[i:i + 2048])
                S.append(pred(g) if pred is not None else torch.ones(Ti.shape[0], device=dev))
                if hasattr(head, 'all_candidates'):
                    elem.append(head.all_candidates(g, se3.features(ty[i:i + 2048], Ry[i:i + 2048])).argmin(1))
                elif hasattr(head, 'branch_values'):
                    bv = head.branch_values(g, se3.features(ty[i:i + 2048], Ry[i:i + 2048]))
                    elem.append(bv.reshape(bv.shape[0], -1).argmin(1))
        T, c, H, S = torch.cat(T), torch.cat(c), torch.cat(H), torch.cat(S)
        hit, f = ridge.detect(c, H, speed=S, band=a.band, kappa=a.kappa)
        rec = {'T': T.reshape(a.res, a.res).cpu().numpy(), 'cn': f['cn'].reshape(a.res, a.res).cpu().numpy(),
               'lam': f['lam_min'].reshape(a.res, a.res).cpu().numpy(), 'r': f['r'].reshape(a.res, a.res).cpu().numpy(),
               'S': S.reshape(a.res, a.res).cpu().numpy(), 'hit': hit.reshape(a.res, a.res).cpu().numpy(),
               'elem': torch.cat(elem).reshape(a.res, a.res).cpu().numpy() if elem else None,
               'goal': pg.cpu().numpy()}
        np.savez_compressed(os.path.join(a.out, 'slice_goal{}.npz'.format(gi)), **{k: v for k, v in rec.items() if v is not None})
        free = rec['S'] > 0.05
        summary[gi] = {'band_frac_free': float(((rec['cn'] < a.band) & free).mean() / max(free.mean(), 1e-9)),
                       'hit_frac_free': float((rec['hit'] & free).mean() / max(free.mean(), 1e-9)),
                       'n_elements_used': int(len(np.unique(rec['elem'][free]))) if rec['elem'] is not None else None,
                       'T_range': [float(rec['T'][free].min()), float(rec['T'][free].max())]}
        print(gi, summary[gi], flush=True)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
            ext = [-a.half, a.half, -a.half, a.half]
            Tm = np.where(free, rec['T'], np.nan)
            axs[0].imshow(Tm.T, origin='lower', extent=ext, cmap='viridis'); axs[0].set_title('T (free region), goal *')
            axs[0].contour(lin.cpu(), lin.cpu(), Tm.T, levels=20, colors='w', linewidths=0.4)
            axs[0].plot(rec['goal'][0], rec['goal'][1], 'r*')
            axs[1].imshow(np.where(free, rec['cn'], np.nan).T, origin='lower', extent=ext, vmin=0, vmax=1.5, cmap='coolwarm'); axs[1].set_title('S|c| (1 = Eikonal)')
            axs[2].imshow(np.where(free, np.clip(rec['lam'], -50, 10), np.nan).T, origin='lower', extent=ext, cmap='RdBu'); axs[2].set_title('lambda_min (clipped)')
            if rec['elem'] is not None:
                axs[3].imshow(np.where(free, rec['elem'], np.nan).T, origin='lower', extent=ext, cmap='tab20', interpolation='nearest'); axs[3].set_title('active group element')
            else:
                axs[3].imshow(np.where(free, rec['hit'], np.nan).T, origin='lower', extent=ext, cmap='gray'); axs[3].set_title('detector band')
            for ax in axs[:3]:
                ax.contour(lin.cpu(), lin.cpu(), np.where(free, rec['r'], np.nan).T, levels=[0.0], colors='m', linewidths=0.8)
                ax.contour(lin.cpu(), lin.cpu(), (rec['hit'] & free).T.astype(float), levels=[0.5], colors='k', linewidths=0.6)
            for ax in axs:
                ax.contour(lin.cpu(), lin.cpu(), free.T.astype(float), levels=[0.5], colors='k', linewidths=1.0)
            fig.suptitle('{} goal {} (magenta: r = 0, black: detector band)'.format(os.path.basename(os.path.dirname(a.ckpt)), gi))
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, 'slice_goal{}.png'.format(gi)), dpi=110)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print('figure skipped:', e)
    with open(os.path.join(a.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
