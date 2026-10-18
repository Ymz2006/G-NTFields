"""Measure the nine decoder properties of "Decoder properties that hold with or without obstacles" on trained
fields (graph_native-contract checkpoints: graph_gram, graph_metric, branch heads, ...).

    python models/graph_branch/scripts/decoder_properties.py --scene Tshape3d_env1 \
        --out results/graph_branch/properties/T_br4_s0.json Experiments/graph_branch/T_br4_s0/latest.pt

Poses are endpoints of the scene's training pairs (collision-free, with stored clearance speeds). Per property:

 P1  max |T(p, p)|; fraction of pairs with T < 1e-4 (x != y)
 P2  max and median |T(x, y) - T(y, x)| / T
 P3  triangle violations on random triples: rate of T(x, z) > T(x, y) + T(y, z) + 1e-4 and the largest
     violation relative to T(x, z)
 P4  fraction of pairs with T < T_free - 1e-4 (T_free the exact speed-limit time) and min T / T_free
 P5  cone at the goal, both endpoints: slope s(eps) = T(p exp(eps xi), p) / eps for unit directions xi at
     eps = 1e-2, 1e-3, 1e-4; convergence |s(1e-3) - s(1e-4)| / s(1e-4)
 P6  metric shape: anisotropy max_xi s / min_xi s at eps = 1e-4 (1 = the metric's shape, any scale), and the
     scale against the Eikonal limit at the goal: median S_d(p) s(p) (1 = the cone matches the local speed)
 P7  V-shaped kinks along random lines x(s) = x exp(s xi) (goal fixed, not on the line): h-scaled second
     differences that do not shrink under 8x refinement, positive (V) vs negative (ridge); counted per line
 P8  rotational ridge: the start rotated at the goal's position about a random axis through one turn (1441
     samples): angle of the global maximum of T, its distance from the half turn, and the number of interior
     ridges (concave kinks) per slice
 P9  pure-rotation times (t_x = t_y): max T and the share above the free half-turn time 0.5; for graph_gram /
     graph_metric also the maximum of the rotation term phi(b) / s_r^2 (capped at 1/4)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_branch import chart_metric  # noqa: E402
from models.graph_lie.model_train_lie import warp_speeds  # noqa: E402
from models.graph_method_triangle import branch_decoder as bd  # noqa: E402
from models.graph_method_triangle.fields import decoder as abdec  # noqa: E402
from models.graph_native import _sib  # noqa: E402
from models.graph_native import model_train_metric as gn_train  # noqa: E402

TWO_PI = 2.0 * math.pi
se3 = _sib.se3


def load(ckpt, device):
    m = gn_train.Model('/tmp/props', 'datasets/3dshape/data/Tshape3d_env1', 6, [0.0] * 6, device=device)
    m.load(ckpt)
    m.network.eval()
    return m.network


def poses_from_chart(c):
    t = c[:, :3]
    R = se3.so3_exp(TWO_PI * c[:, 3:6])
    return t, R


def F(t, R):
    return se3.features(t, R)


@torch.no_grad()
def V(net, gx, gy, chunk=32768):
    out = []
    for i in range(0, gx.shape[0], chunk):
        out.append(net.net.value(gx[i:i + chunk], gy[i:i + chunk]))
    return torch.cat(out)


def unit_twists(n, g, device):
    u = torch.randn(n, 6, generator=g).to(device)
    u = u / u.norm(dim=1, keepdim=True)
    return torch.cat([u[:, :3], TWO_PI * u[:, 3:]], 1)  # g(xi, xi) = |v|^2 + |w|^2 / (2 pi)^2 = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--orig', action='store_true',
                    help='checkpoint of the original metric trainer (Lie-Shape-NTFields metric_baseline, chart input)')
    ap.add_argument('ckpt')
    a = ap.parse_args()
    dev = a.device
    torch.set_grad_enabled(False)
    net = chart_metric.load_original(a.ckpt, dev) if a.orig else load(a.ckpt, dev)
    arch = net.graph['arch']
    g = torch.Generator().manual_seed(0)
    X = np.load(os.path.join('datasets/3dshape/data', a.scene, 'sampled_points.npy'), mmap_mode='r')
    rows = torch.from_numpy(np.array(X[:400000:7][: 3 * a.n])).float()
    d = torch.from_numpy(np.concatenate([np.asarray(np.load(os.path.join('datasets/3dshape/data', a.scene, f + '.npy'),
                                                            mmap_mode='r')[:400000:7][: 3 * a.n])
                                         for f in ('sampled_points', 'speed', 'normal', 'speed_dists', 'speed_angles')],
                                        1)).float()
    P = torch.cat([rows[:, :6], rows[:, 6:]], 0).to(dev)   # endpoint poses (chart)
    perm = torch.randperm(P.shape[0], generator=g)
    P = P[perm.to(dev)]
    tP, RP = poses_from_chart(P)
    GP = F(tP, RP)
    n = a.n
    res = {'ckpt': a.ckpt, 'arch': arch, 'scene': a.scene, 'graph': {k: v for k, v in net.graph.items()
                                                                     if isinstance(v, (int, float, str, bool))}}
    x, y, z = GP[:n], GP[n:2 * n], GP[2 * n:3 * n]
    Txy, Tyx, Txz, Tyz = V(net, x, y), V(net, y, x), V(net, x, z), V(net, y, z)
    Tf = bd.free_time(x, y, 1.0, TWO_PI)
    # P1
    res['P1_goal_max_abs'] = float(V(net, x[:4000], x[:4000]).abs().max())
    res['P1_frac_T_below_1e-4'] = float((Txy < 1e-4).double().mean())
    # P2
    sym = (Txy - Tyx).abs() / Txy.clamp_min(1e-6)
    res['P2_sym_rel_max'], res['P2_sym_rel_median'] = float(sym.max()), float(sym.median())
    # P3
    viol = Txz - Txy - Tyz
    res['P3_triangle_viol_rate'] = float((viol > 1e-4).double().mean())
    res['P3_triangle_viol_max_rel'] = float((viol / Txz.clamp_min(1e-6)).max())
    # P4
    res['P4_frac_below_Tfree'] = float((Txy < Tf - 1e-4).double().mean())
    res['P4_min_T_over_Tfree'] = float((Txy / Tf.clamp_min(1e-6)).min())
    res['T_over_Tfree_median'] = float((Txy / Tf.clamp_min(1e-6)).median())
    # P5 / P6: cone at the goal (both endpoints)
    m = 1000
    gp = GP[:m]
    xi = unit_twists(12, g, dev)
    slopes = {}
    off = V(net, gp, gp)   # T(p, p): exactly 0 for the graph and branch heads, 3.2e-3 for the original metric head
    for eps in (1e-2, 1e-3, 1e-4):
        S0, S1 = [], []
        for j in range(12):
            tq, Rq = se3.right_perturb(tP[:m], RP[:m], (eps * xi[j]).expand(m, 6))
            gq = F(tq, Rq)
            S0.append((V(net, gq, gp) - off) / eps)
            S1.append((V(net, gp, gq) - off) / eps)
        slopes[eps] = (torch.stack(S0, 1), torch.stack(S1, 1))     # (m, 12) each
    for end in (0, 1):
        s2, s3, s4 = slopes[1e-2][end], slopes[1e-3][end], slopes[1e-4][end]
        res['P5_cone_conv_e{}'.format(end)] = float(((s3 - s4).abs() / s4).median())
        res['P5_cone_conv_1e-2_e{}'.format(end)] = float(((s2 - s4).abs() / s4).median())
        res['P6_anisotropy_median_e{}'.format(end)] = float((s4.max(1).values / s4.min(1).values).median())
        res['P6_anisotropy_p90_e{}'.format(end)] = float((s4.max(1).values / s4.min(1).values).quantile(0.9))
    # S_d per endpoint pose: P stacks the start rows (speed column 0) then the goal rows (column 1), then permutes
    W = warp_speeds(d)['Sd']
    Sd_all = torch.cat([W[:, 0], W[:, 1]], 0).to(dev)[perm.to(dev)][:m]
    Wc = warp_speeds(d)['S']   # combined speed: the speed model of the original metric trainer
    Sc_all = torch.cat([Wc[:, 0], Wc[:, 1]], 0).to(dev)[perm.to(dev)][:m]
    for end in (0, 1):
        s4 = slopes[1e-4][end]
        sm = s4.mean(1)
        res['P6_eikonal_at_goal_median_e{}'.format(end)] = float((sm * Sd_all).median())
        res['P6_eikonal_at_goal_dex_e{}'.format(end)] = float((torch.log10(sm * Sd_all)).abs().median())
        res['P6_eikonal_at_goal_comb_median_e{}'.format(end)] = float((sm * Sc_all).median())
        lo = Sd_all < 0.05
        if lo.any():
            res['P6_eikonal_at_goal_median_lowS_e{}'.format(end)] = float((sm * Sd_all)[lo].median())
    # P7: kinks along random lines
    n_lines = 200
    nv = nr = nj = 0
    for li in range(n_lines):
        x0t, x0R = tP[li:li + 1], RP[li:li + 1]
        y0 = GP[2 * n + li:2 * n + li + 1]
        xi1 = unit_twists(1, g, dev)[0]
        d2s, dmax = [], []
        for npt in (400, 3200):
            s = torch.linspace(-0.6, 0.6, npt + 1, device=dev)
            tt, RR = se3.right_perturb(x0t.expand(len(s), 3), x0R.expand(len(s), 3, 3), s[:, None] * xi1[None])
            T = V(net, F(tt, RR), y0.expand(len(s), 12))
            dmax.append(float((T[1:] - T[:-1]).abs().max()))
            hstep = (s[1] - s[0]).item()
            d2s.append(((T[2:] - 2 * T[1:-1] + T[:-2]) / hstep).double())
        c, f_ = d2s
        if f_.max() > max(0.5 * c.max().item(), 0.05):
            nv += 1
        if c.min() < -0.05 and f_.min() < 0.5 * c.min().item():
            nr += 1
        # a jump (discontinuity): the largest step between neighbouring samples does not shrink under 8x refinement
        if dmax[1] > max(0.5 * dmax[0], 0.05):
            nj += 1
    res['P7_lines'], res['P7_lines_with_V_kink'], res['P7_lines_with_ridge'] = n_lines, nv, nr
    res['P7_lines_with_jump'] = nj
    # P8 / P9: rotation slices at the goal's position
    n_sl, n_phi = 200, 1441
    phi = torch.linspace(0, TWO_PI, n_phi, device=dev)
    ax = torch.randn(n_sl, 3, generator=g).to(dev)
    ax = ax / ax.norm(dim=1, keepdim=True)
    peaks, n_ridges, Tmax, share = [], [], [], []
    rotmax, kink_off, switch_off, n_switch, bmax = [], [], [], [], []
    for k in range(n_sl):
        tg, Rg = tP[2 * n + k:2 * n + k + 1], RP[2 * n + k:2 * n + k + 1]
        Rs = Rg @ se3.so3_exp(phi[:, None] * ax[k][None])
        gs = F(tg.expand(n_phi, 3), Rs)
        gg = F(tg, Rg).expand(n_phi, 12)
        T = V(net, gs, gg)
        peaks.append(phi[T.argmax()].item())
        d2 = (T[2:] - 2 * T[1:-1] + T[:-2]) / (phi[1] - phi[0]).item()
        loc = (T[1:-1] >= T[:-2]) & (T[1:-1] >= T[2:]) & (d2 < -0.02)
        n_ridges.append(int(loc.sum()))
        Tmax.append(T.max().item())
        share.append(float((T > 0.5).double().mean()))
        if hasattr(net.net, 'predict_latent'):
            # the fixed decoder's only rotational kink is at the maximum of b (b -> 1): where is it on the slice?
            a_, b_ = net.net.predict_latent(gs, gg)
            rotmax.append(float((abdec.phi(b_) / net.net.s_r ** 2).max()))
            kink_off.append(abs(phi[b_.argmax()].item() - math.pi) * 180 / math.pi)
            bmax.append(float(b_.max()))   # the kink exists only where b reaches 1 (rank-1 lift); below: rounded
        if arch.startswith('graph_branch'):
            # route switches along the slice: where the selected rotation route changes (the learned ridges)
            # the slice rotates the start by phi about ax: the selected route either turns back by phi or goes on
            # by 2 pi - phi; the free solution switches once, at phi = pi; a learned ridge switches elsewhere
            BV = net.net.branch_values(gs, gg)
            K = BV.shape[2]
            sig = 1.0 - 2.0 * (BV.reshape(n_phi, -1).argmin(1) // K).to(phi.dtype)
            qs = bd.quat_from_matrix(Rs)
            qg = bd.quat_from_matrix(Rg).expand(n_phi, 4)
            c = (qs * qg).sum(1)
            psi_sel = 2.0 * torch.atan2(torch.sqrt(bd.route_s2(qs, qg).clamp_min(0)), sig * c)
            back = (psi_sel - phi).abs() < (psi_sel - (TWO_PI - phi)).abs()
            sw = torch.nonzero(back[1:] != back[:-1])[:, 0].tolist()
            # merge switches within 2 deg (numerical flips where the two routes tie to rounding) into one ridge
            clusters = []
            for i in sw:
                if clusters and (phi[i] - phi[clusters[-1][-1]]).item() < 2.0 * math.pi / 180:
                    clusters[-1].append(i)
                else:
                    clusters.append([i])
            n_switch.append(len(clusters))
            for cl in clusters:
                c_ang = 0.5 * (phi[cl[0]] + phi[cl[-1]]).item()
                switch_off.append(abs(c_ang - math.pi) * 180 / math.pi)
    pk = torch.tensor(peaks)
    off = (torch.remainder(pk - math.pi + math.pi, TWO_PI) - math.pi).abs() * 180 / math.pi
    res['P8_rot_ridge_offset_from_halfturn_deg_median'] = float(off.median())
    res['P8_rot_ridge_offset_deg_p90'] = float(off.quantile(0.9))
    res['P8_frac_slices_ridge_beyond_10deg'] = float((off > 10).double().mean())
    res['P8_ridges_per_slice_mean'] = float(np.mean(n_ridges))
    res['P9_rot_slice_Tmax_median'] = float(np.median(Tmax))
    res['P9_rot_slice_share_above_halfturn_time'] = float(np.mean(share))
    if rotmax:
        res['P9_rotation_term_max'] = float(max(rotmax))
    if kink_off:
        ko = torch.tensor(kink_off)
        res['P8_decoder_kink_offset_from_halfturn_deg_median'] = float(ko.median())
        res['P8_decoder_kink_offset_deg_p90'] = float(ko.quantile(0.9))
        res['P8_frac_slices_kink_beyond_10deg'] = float((ko > 10).double().mean())
        bm = torch.tensor(bmax)
        res['P8_bmax_median'] = float(bm.median())
        res['P8_frac_slices_with_kink_b_gt_0.999'] = float((bm > 0.999).double().mean())
    if n_switch:
        so = torch.tensor(switch_off) if switch_off else torch.zeros(1)
        res['P8_route_switches_per_slice_mean'] = float(np.mean(n_switch))
        res['P8_route_switch_offset_from_halfturn_deg_median'] = float(so.median())
        res['P8_route_switch_offset_deg_p90'] = float(so.quantile(0.9))
        res['P8_frac_switches_beyond_10deg'] = float((so > 10).double().mean())
    if arch.startswith('graph_branch'):
        BV = []
        for i in range(0, n, 8192):
            BV.append(net.net.branch_values(x[i:i + 8192], y[i:i + 8192]))
        BV = torch.cat(BV)                                     # (n, 2, K)
        best = BV.reshape(n, -1).argmin(1)
        K = BV.shape[2]
        qx = bd.quat_from_matrix(x[:, 3:].reshape(-1, 3, 3))
        qy = bd.quat_from_matrix(y[:, 3:].reshape(-1, 3, 3))
        c = (qx * qy).sum(1)
        short_is_plus = c >= 0
        route_plus = (best // K) == 0
        res['branch_long_way_frac'] = float((route_plus != short_is_plus).double().mean())
        res['branch_k_usage'] = [float(((best % K) == k).double().mean()) for k in range(K)]
        if getattr(net.net, 'conf', False):
            e = net.net.enc(GP[:20000])
            rho = torch.nn.functional.softplus(net.net.rho_head(e)[:, 0])
            lam = net.net.lam_min + torch.nn.functional.softplus(net.net.lam_head(e)[:, 0])
            res['conf_rho_median'], res['conf_rho_p90'] = float(rho.median()), float(rho.quantile(0.9))
            res['conf_lam_median'] = float(lam.median())
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == '__main__':
    main()
