"""Stage E (separate research branch): graph fields among obstacles.

Training data are the preprocessed 3dshape datasets (`datasets/3dshape/data/<shape>_<env>`):
pairs of poses stored as (t, rotvec / 2 pi) with per-endpoint clearance speeds.
The speed model is the *declared position-dependent* ellipsoidal model

    |v|^2 / (S_d(x) s_t)^2 + |w|^2 / (S_a(x) s_r)^2 <= 1,   s_t = 1, s_r = 2 pi,

i.e. rotations are measured in turns as in the metric baseline, and the
per-pose speed multipliers come from the dataset after the baseline's warp
S = s^2 (2 - s)^2. speed_model='iso' uses the combined speed for both blocks
(the baseline's isotropic choice); 'aniso' uses speed_dists / speed_angles.

Losses: separate endpoint Eikonal  |c|^2 = 1 with
c = (S_d s_t R^T dT/dt, S_a s_r dT/dw) on pairs that are regular in free-space
geometry, plus an optional one-step Bellman/TD term along the greedy direction
using the local speed at x (assumed constant over the step, as in NTFields;
there is no swept collision check, so this term is approximate).
No oracle exists here: no anchors, no label losses.

The two-scalar decoder no longer implies separate physical meanings for a and b;
this branch tests the hypothesis empirically and makes no smoothness claim.
"""

import math
import os

import numpy as np
import torch

from .geometry import se3, tangents
from .losses import bellman
from . import model_function_graph as mf
from . import model_network_graph as nets


def warp(s):
    return s * s * (2 - s) * (2 - s)


def coords_to_features(xp):
    """(N, 6) (t, rotvec / 2 pi) -> (N, 12) G = (t, vec R)."""
    t = xp[:, :3]
    R = se3.so3_exp(2.0 * math.pi * xp[:, 3:6])
    return se3.features(t, R)


def shape_config(**overrides):
    base = dict(data_path='datasets/3dshape/data/Tshape3d_env1', speed_model='aniso', eik_scale_weight=False,
                exp_weight=0.5, exp_weight_grad=False, td_weight_T=None, n_holdout=20000, w_lb=0.0,
                coords=False, ntf_rollback=False, ntf_epoch=5, ntf_ratio=1.2)
    extra = {k: overrides.pop(k) for k in list(overrides) if k in base}
    base.update(extra)
    if base['coords']:
        overrides['feature_dim'] = 6
    cfg = mf.default_config(**{'s_t': 1.0, 's_r': 2 * math.pi, 'L_ref': 1.0, 'mode': 'shape',
                               'n_anchor': 0, 'w_anchor': 0.0, 'w_goal': 0.0, **overrides})
    cfg.update(base)
    return cfg


class ShapeFunction(mf.Function):
    def __init__(self, cfg, network=None):
        super().__init__(cfg, network)
        d = cfg['data_path']
        pts = torch.from_numpy(np.load(os.path.join(d, 'sampled_points.npy'))).float()
        S = torch.from_numpy(np.load(os.path.join(d, 'speed.npy'))).float()
        Sd = torch.from_numpy(np.load(os.path.join(d, 'speed_dists.npy'))).float()
        Sa = torch.from_numpy(np.load(os.path.join(d, 'speed_angles.npy'))).float()
        if cfg['speed_model'] == 'iso':
            Sd, Sa = S, S
        Sd, Sa = warp(Sd).clamp_min(1e-3), warp(Sa).clamp_min(1e-3)
        G0 = coords_to_features(pts[:, :6].double())
        G1 = coords_to_features(pts[:, 6:].double())
        if cfg['coords']:  # baseline convention: raw (t, rotvec / 2 pi) inputs
            g0, g1 = pts[:, :6], pts[:, 6:]
        else:
            g0, g1 = G0.float(), G1.float()
        dev = self.device
        self.data = {'g0': g0.to(dev), 'g1': g1.to(dev), 'Sd': Sd.to(dev), 'Sa': Sa.to(dev)}
        # free-space geometry for the regular mask / curriculum
        t0, R0 = se3.split_features(G0)
        t1, R1 = se3.split_features(G1)
        ang = se3.rotation_angle(R0.transpose(-1, -2) @ R1)
        Tfree = torch.sqrt(((t0 - t1) ** 2).sum(-1) + (ang / (2 * math.pi)) ** 2)
        b = (3.0 - (R0 * R1).sum((-1, -2))) / 4.0
        self.data.update(ang=ang.float().to(dev), Tfree=Tfree.float().to(dev), bfree=b.float().to(dev))
        n = pts.shape[0]
        g = torch.Generator().manual_seed(cfg['seed'])
        perm = torch.randperm(n, generator=g)
        self.hold_idx = perm[:cfg['n_holdout']].to(dev)
        self.train_idx = perm[cfg['n_holdout']:].to(dev)
        self._order, self._opos = None, 0

    def _indices(self, n, step):
        th_max = self.max_angle(step)
        out, have = [], 0
        while have < n:
            if self._order is None or self._opos >= self._order.shape[0]:
                self._order = self.train_idx[torch.randperm(self.train_idx.shape[0], device=self.device,
                                                            generator=self.gen)]
                self._opos = 0
            idx = self._order[self._opos:self._opos + 2 * n]
            self._opos += 2 * n
            if th_max < math.pi:
                idx = idx[self.data['ang'][idx] <= th_max]
            out.append(idx)
            have += idx.shape[0]
        return torch.cat(out)[:n]

    def tangent_c(self, gx, gy, Sd_x, Sa_x, Sd_y, Sa_y, create_graph=True):
        if self.cfg['coords']:
            # Euclidean gradient w.r.t. (t, rotvec / 2 pi), as in the metric baseline
            gx = gx.detach().requires_grad_(True)
            gy = gy.detach().requires_grad_(True)
            T = self.network.value(gx, gy)
            cx, cy = torch.autograd.grad(T.sum(), (gx, gy), create_graph=create_graph)
        else:
            T, cx, cy = tangents.tangent_gradients(self.network.value, gx, gy, self.s_t, self.s_r,
                                                   create_graph=create_graph)
        mx = torch.cat([Sd_x[:, None].expand(-1, 3), Sa_x[:, None].expand(-1, 3)], -1)
        my = torch.cat([Sd_y[:, None].expand(-1, 3), Sa_y[:, None].expand(-1, 3)], -1)
        return T, cx * mx, cy * my

    def loss(self, step):
        self.set_phase(step)
        c = self.cfg
        D = self.data
        idx = self._indices(c['batch_size'], step)
        reg = (D['Tfree'][idx] >= c['diag_band'] * self.tau_ref) & (1 - D['bfree'][idx] >= c['pi_band'])
        idx = idx[reg]
        gx, gy = D['g0'][idx], D['g1'][idx]
        Sd, Sa = D['Sd'][idx], D['Sa'][idx]
        T, cx, cy = self.tangent_c(gx, gy, Sd[:, 0], Sa[:, 0], Sd[:, 1], Sa[:, 1])
        from .losses import eikonal
        res = eikonal.endpoint_residual(cx, cy, c['eik_form'])
        # NTFields weights the residual by exp(-0.5 T) WITHOUT detaching T, which
        # also pushes T up wherever the residual is large (exp_weight_grad=True)
        Tw = T if c['exp_weight_grad'] else T.detach()
        w = torch.exp(-c['exp_weight'] * Tw) if c['exp_weight'] else torch.ones_like(T)
        LE = (res * w).mean()
        total = c['w_eik'] * LE
        logs = {'L_E': LE.detach(), 'T_mean': T.detach().mean()}
        if c['w_lb'] > 0:
            # speed bound: all speeds <= 1, so no pair can be traversed faster than
            # the obstacle-free time T_free (pose geometry only, no labels). Rules
            # out the degenerate projection-metric solutions of the Eikonal (E10).
            Lb = (torch.relu(D['Tfree'][idx] - T) ** 2).mean() / self.tau_ref ** 2
            total = total + c['w_lb'] * Lb
            logs['L_lb'] = Lb.detach()
        if c['w_bell'] > 0:
            h = c['bell_h']
            cxd = cx.detach()
            u = -cxd / torch.linalg.norm(cxd, dim=-1, keepdim=True).clamp_min(1e-12)
            # local speed at x scales the executed twist; elapsed time is h
            sc = torch.cat([Sd[:, :1].expand(-1, 3) * self.s_t, Sa[:, :1].expand(-1, 3) * self.s_r], -1)
            tx, Rx = se3.split_features(gx)
            t2, R2 = se3.right_perturb(tx, Rx, h * u * sc)
            with torch.no_grad():
                tgt = h + self.target.value(se3.features(t2, R2), gy)
            far = T.detach() > h
            LB = (((T - tgt) ** 2) * far.to(T.dtype) * w).mean() / self.tau_ref ** 2
            total = total + c['w_bell'] * LB
            logs['L_B'] = LB.detach()
        return total, logs

    @torch.no_grad()
    def holdout_residual(self, n=8192):
        """Held-out Eikonal residual (no oracle exists among obstacles)."""
        D = self.data
        idx = self.hold_idx[:n]
        reg = (D['Tfree'][idx] >= self.cfg['diag_band']) & (1 - D['bfree'][idx] >= self.cfg['pi_band'])
        idx = idx[reg]
        with torch.enable_grad():
            T, cx, cy = self.tangent_c(D['g0'][idx], D['g1'][idx], D['Sd'][idx, 0], D['Sa'][idx, 0],
                                       D['Sd'][idx, 1], D['Sa'][idx, 1], create_graph=False)
        r = ((cx ** 2).sum(-1) - 1).abs()
        return {'hold_eik_p50': r.median().item(), 'hold_eik_mean': r.mean().item(),
                'hold_T_mean': T.mean().item()}
