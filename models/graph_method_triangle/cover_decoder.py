"""Cover-quotient metric decoder: triangle inequality AND movable ridges by construction (2026-10-13).

"Decoder properties that hold with or without obstacles" lists the triangle inequality (property 3) as
guaranteed in two cases only: (a) one distance between embeddings (no ridges), or (b) branches that are
copies of one metric under a symmetry group, T = min_g d(x, g . y). Case (b) is more general than the
+-q rotation routes: *any* free isometric action of a finite group on a covering space gives a quotient
pseudometric, and the group need not come from the geometry. This head builds such a cover with learned
branch points, so the ridges (ties between group elements) are learned, while every parameter value gives a
pseudometric.

Per pose x the shared G-Shape encoder gives e(x); from it, features on a cover of the free space:

    even      E(x)   = U e(x)                                   (a function on the base)
    q-odd     Q(x~)  = q_x (x) v(x),   v = W e(x)                (odd under q -> -q: the rotation double cover)
    cut-odd   O_i(x~) = w(z_i(x)) (x) A_i e(x),  i = 1..m        (odd under the deck map of the i-th learned cover)

where z_i(x) = Z_i e(x) in R^2 and w(z) = |z|^{1/2} sqrt(z) (complex square root, scaled so that |w| = |z|,
Lipschitz at the branch point z = 0). A loop around a zero of z_i flips the sign of w, so O_i is a function on
a double cover of the free space branched at {z_i = 0}: a learned branch point, meant to sit inside an obstacle.
The group G = Z_2^{1 + m} acts on the product cover by flipping the signs of the odd blocks (and swapping the
two quaternion routes); the action is free and isometric for every l_p norm (coordinate sign flips). With

    T_free,sigma(x, y)^2 = |t_x - t_y|^2 / s_t^2 + psi_sigma^2 / s_r^2,     psi_+ = 2 atan2(|v|, q_x . q_y), psi_- = 2 pi - psi_+
    rho(d) = |d|^2 ('euclid') | (metric_scale sum_g |d_g|_p)^2 ('lp')      a squared seminorm of the feature difference

the field is

    T(x, y)^2 = min_{g in G} [ T_free,sigma(g)(x, y)^2 + rho( F(x~) - g . F(y~) ) ],   F = [E; Q; O_1; ...; O_m].

sqrt(T_free^2 + rho) is a pseudometric on the cover (l_2 combination of two pseudometrics; T_free is the
product metric of R^3 x S^3, rho^(1/2) a seminorm of a difference), invariant under G, so the min over G is a
pseudometric on the base (d(x, g y) <= d(x, h z) + d(z, h^-1 g y), minimise over h then g). Hence for every
parameter value:

 1  T >= 0, T(x, x) = 0 (g = identity: psi_+ = 0, F_x - F_x = 0), T(x, y) >= T_free(x, y) > 0 for x != y
 2  T(x, y) = T(y, x)               (every term is a symmetric comparison; the group is closed under inverses)
 3  T(x, z) <= T(x, y) + T(y, z)    (quotient of an invariant pseudometric)
 4  T >= T_free                      (rho >= 0, min over the two rotation routes of T_free,sigma is T_free)
 5  T = sqrt(Q) near the goal with Q smooth, Q(x, x) = 0, grad Q(x, x) = 0 (the identity branch is active there)
 7  peaks only: the only non-smooth operations on the output path are the min, the goal's square root and the
    far point of a rotation route (psi -> 2 pi, a concave corner)
 8  movable ridges: ties between group elements, e.g. between s_i = +1 and s_i = -1 at <O_i,x, O_i,y> = 0, i.e.
    where z_i(x) and z_i(y) are antiparallel in R^2 -- a learned surface emanating from the branch point, the
    shape of an obstacle's shadow. NOTE (report, 2026-10-13): with the euclid residual the sigma tie always
    contains the half-turn set q_x . q_y = 0 (the cross term carries the factor q_x . q_y), so the rotation
    ridge is pinned there and only the cut ridges are learned; the lp seminorm does not pin it
 9  no cap: rho is unbounded.
 6  (cone shape) holds for the free metric's cone only up to the residual's first-order term, as for any
    embedding distance; it is not claimed.

'lse' replaces the seminorm by the G-Shape logsumexp group max (graph_metric's): same branch structure, no
triangle guarantee (an ablation of the metric constraint). `tfree=False` drops T_free (pure quotient
seminorm; properties 4 and 6 lost, the rest kept). `conf=True` adds branch_decoder's conformal cone term
T_free -> T_free + rho_bar lam_bar tanh(T_free / lam_bar) (per-pose softplus heads): an ablation that scales
the goal cone by 1 + rho(p) without Lipschitz capacity; it is not a pseudometric (measured, not guaranteed).

Branch selection is done without autograd; only the selected group element is recomputed with gradients, so
the unused routes' singular points (psi -> 2 pi, where d psi / d|v| is infinite) cannot inject NaN into the
double backward. quat_from_matrix / route helpers follow branch_decoder.py (same conventions), copied here so
that this module has no runtime dependency on it.
"""
import itertools
import math

import torch
from torch.nn import Linear

from . import model_network_graph as mng
from .interface import PairValueModel

TWO_PI = 2.0 * math.pi
_SERIES_S2 = 1e-6
RES_KINDS = ('euclid', 'lp', 'lse')


def quat_from_matrix(R):
    """Unit quaternion (w, x, y, z) of R (..., 3, 3), Shepperd's method (largest radicand as pivot; the unused
    candidates are evaluated on a safe pivot and dropped by the gather, so they get exactly zero gradient)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    r = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    k = r.detach().argmax(-1)
    sel = torch.nn.functional.one_hot(k, 4).bool()
    s = 0.5 * torch.sqrt(torch.where(sel, r, torch.ones_like(r)).clamp_min(1e-6))
    d = 0.25 / s
    a, b_, c_ = m21 - m12, m02 - m20, m10 - m01
    p01, p02, p12 = m01 + m10, m02 + m20, m12 + m21
    s0, s1, s2, s3 = s.unbind(-1)
    d0, d1, d2, d3 = d.unbind(-1)
    cand = torch.stack([
        torch.stack([s0, a * d0, b_ * d0, c_ * d0], -1),
        torch.stack([a * d1, s1, p01 * d1, p02 * d1], -1),
        torch.stack([b_ * d2, p01 * d2, s2, p12 * d2], -1),
        torch.stack([c_ * d3, p02 * d3, p12 * d3, s3], -1),
    ], -2)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def route_s2(qx, qy):
    """|v|^2 = 1 - c^2 of the relative quaternion, cancellation-free."""
    return ((qx - qy) ** 2).sum(-1) * ((qx + qy) ** 2).sum(-1) * 0.25


def route_angle_sq(s2, cc):
    """psi^2 = 4 atan2(sqrt(s2), cc)^2; series in s2 at the route's own goal (cc > 0, s2 -> 0)."""
    small = (s2 < _SERIES_S2) & (cc > 0)
    z2 = s2 / torch.where(small, cc, torch.ones_like(cc)) ** 2
    series = 4.0 * z2 * (1.0 + z2 * (-2.0 / 3.0 + z2 * (23.0 / 45.0 - z2 * 44.0 / 105.0)))
    s_safe = torch.where(small, torch.ones_like(s2), s2).clamp_min(1e-20)
    exact = 4.0 * torch.atan2(torch.sqrt(s_safe), cc) ** 2
    return torch.where(small, series, exact)


def free_routes_sq(a, s2, c, s_r):
    """(B, 2) squared free times of the routes (+, -), forward values only."""
    psi_p = 2.0 * torch.atan2(torch.sqrt(s2.clamp_min(0.0)), c)
    psi = torch.stack([psi_p, TWO_PI - psi_p], 1)
    return a[:, None] + psi ** 2 / (s_r * s_r)


def free_time(gx, gy, s_t, s_r):
    """Exact obstacle-free time (the speed-limit lower bound) from the embeddings."""
    qx = quat_from_matrix(gx[:, 3:12].reshape(-1, 3, 3))
    qy = quat_from_matrix(gy[:, 3:12].reshape(-1, 3, 3))
    a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1) / (s_t * s_t)
    return torch.sqrt(free_routes_sq(a, route_s2(qx, qy), (qx * qy).sum(-1).clamp(-1, 1), s_r).amin(1))


def half_angle(z, eps=1e-6):
    """w(z) = |z|^{1/2} sqrt(z) for z (..., 2) read as a complex number: |w| = |z| and w^2 / |w| = z.

    Cancellation-free in both half planes (z1 >= 0: w ~ (a, z2 / a); z1 < 0: w ~ (|z2| / a, sign(z2) a) with
    a = sqrt(r + |z1|)), so the derivative is finite and smooth on each side of the cut {z1 < 0, z2 = 0}; the
    cut itself is where the sign flips, which the min over the deck group absorbs. eps regularises r at the
    branch point, where w -> 0 anyway.
    """
    z1, z2 = z[..., 0], z[..., 1]
    r = torch.sqrt(z1 * z1 + z2 * z2 + eps * eps)
    a = torch.sqrt(r + z1.abs())
    pos = z1 >= 0
    sgn = torch.where(z2 >= 0, torch.ones_like(z2), -torch.ones_like(z2))
    w1 = torch.where(pos, a, z2.abs() / a)
    w2 = torch.where(pos, z2 / a, sgn * a)
    scale = torch.sqrt(r / 2.0)
    return torch.stack([scale * w1, scale * w2], -1)


class CoverNN(PairValueModel, mng._Siamese):
    """Cover-quotient metric head (module doc). Options:

    n_cut         m, number of learned double covers (branch points); 0 = rotation routes only
    me, mq, mo    even dim, q-odd dim (x4 coordinates), per-cut odd dim (x2 coordinates)
    res_kind      'lp' (exact seminorm, guarantee) | 'euclid' (guarantee) | 'lse' (graph_metric's, no guarantee)
    res_scale     multiplies F (initial residual size)
    tfree         include the free-space lower bound T_free,sigma (default True)
    lp_p, group, metric_scale, lse_beta   seminorm options (as graph_tri_metric / graph_metric)
    soft_tau      > 0: soft minimum (width tau, in units of T) over the learned-cut elements of the selected
                  rotation route, T = -tau log sum_s exp(-T_s / tau); the rotation route itself stays a hard
                  min (its far point is singular). Differentiable everywhere; T(x, x) deviates from 0 by at most
                  tau log 2^m and the triangle inequality is no longer exact (deviation O(tau)).
    """
    has_latent = False

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=False, skip_input=False, lip_cap=1.0,
                 n_cut=2, me=64, mq=16, mo=16, res_kind='lp', res_scale=1.0, tfree=True, lp_p=8.0, group=16,
                 metric_scale=0.2, lse_beta=10.0, cut_eps=1e-6, soft_tau=0.0, conf=False, rho_init=0.5, lam_init=0.05,
                 **extra):
        PairValueModel.__init__(self, s_t, s_r)
        assert B.shape[1] == mng.FEATURE_DIM, 'the cover head reads G(x) = (t, vec R)'
        assert res_kind in RES_KINDS, res_kind
        self.enc = self.ENCODER(B.shape[1], B, h_size, nl1, normalize, skip_input, lip_cap,
                                **self.encoder_extra(extra))
        self.m, self.me, self.mq, self.mo = int(n_cut), int(me), int(mq), int(mo)
        self.res_kind, self.res_scale, self.tfree = res_kind, float(res_scale), bool(tfree)
        self.lp_p, self.group, self.metric_scale, self.lse_beta = float(lp_p), int(group), float(metric_scale), float(lse_beta)
        self.cut_eps = float(cut_eps)
        self.soft_tau = float(soft_tau)
        self.conf = bool(conf)
        D = self.enc.out_dim
        if self.conf:   # conformal cone term of branch_decoder (ablation: not a pseudometric)
            self.rho_head = Linear(D, 1)
            self.lam_head = Linear(D, 1)
        self.U = Linear(D, self.me, bias=False) if self.me > 0 else None
        self.W = Linear(D, self.mq, bias=False) if self.mq > 0 else None
        self.Z = Linear(D, 2 * self.m) if self.m > 0 else None
        self.A = Linear(D, self.m * self.mo, bias=False) if self.m > 0 else None
        self.dim_F = self.me + 4 * self.mq + 2 * self.mo * self.m
        if res_kind in ('lp', 'lse'):
            assert self.dim_F % self.group == 0, 'feature blocks must tile the seminorm groups'
        # sign patterns of the group elements: (G, dim_F), first bit = quaternion route sigma
        blocks = [(1.0, self.me)] + [(None, 4 * self.mq)] + [(None, 2 * self.mo)] * self.m
        pats, sigmas = [], []
        for bits in itertools.product([1.0, -1.0], repeat=1 + self.m):
            row, j = [], 0
            for fixed, width in blocks:
                if fixed is not None:
                    row += [fixed] * width
                else:
                    row += [bits[j]] * width
                    j += 1
            pats.append(row)
            sigmas.append(bits[0])
        self.register_buffer('signs', torch.tensor(pats))
        self.register_buffer('sigmas', torch.tensor(sigmas))
        self.register_buffer('_rows_plus', (self.sigmas > 0).nonzero().squeeze(-1), persistent=False)
        self.register_buffer('_rows_minus', (self.sigmas < 0).nonzero().squeeze(-1), persistent=False)
        mng._init(self, self.enc)
        if self.conf:
            with torch.no_grad():
                self.rho_head.weight.zero_()
                self.rho_head.bias.fill_(math.log(math.expm1(max(rho_init, 1e-4))))
                self.lam_head.weight.zero_()
                self.lam_head.bias.fill_(math.log(math.expm1(max(lam_init - 1e-3, 1e-6))))
        self._last = {}

    def conf_terms(self, ex, ey):
        """(rho_bar, lam_bar) of the conformal cone term T_free -> T_free + rho_bar lam_bar tanh(T_free / lam_bar)."""
        sp = torch.nn.functional.softplus
        rb = 0.5 * (sp(self.rho_head(ex)[:, 0]) + sp(self.rho_head(ey)[:, 0]))
        lb = 1e-3 + 0.5 * (sp(self.lam_head(ex)[:, 0]) + sp(self.lam_head(ey)[:, 0]))
        return rb, lb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # derived index buffers were briefly persistent; ignore them in old checkpoints
        for k in ('_rows_plus', '_rows_minus'):
            state_dict.pop(prefix + k, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def n_elements(self):
        return int(self.signs.shape[0])

    # ------------------------------------------------------------------ per-pose features
    def features_cover(self, g, e):
        """q (N, 4), F (N, dim_F), z (N, m, 2) for poses with embeddings g, e."""
        h = self.enc.h_size
        es = torch.cat([mng.standardize(e[:, :h]), e[:, h:]], -1)
        q = quat_from_matrix(g[:, 3:12].reshape(-1, 3, 3))
        parts = []
        if self.U is not None:
            parts.append(self.U(es))
        if self.W is not None:
            parts.append((q[:, :, None] * self.W(es)[:, None, :]).reshape(g.shape[0], -1))
        z = None
        if self.m > 0:
            z = self.Z(es).view(-1, self.m, 2)
            w = half_angle(z, self.cut_eps)                                           # (N, m, 2)
            a = self.A(es).view(-1, self.m, self.mo)
            parts.append((w[:, :, :, None] * a[:, :, None, :]).reshape(g.shape[0], -1))
        F = self.res_scale * torch.cat(parts, -1)
        return q, F, z

    def rho(self, d):
        """Squared seminorm of feature differences d (..., dim_F)."""
        if self.res_kind == 'euclid':
            return (d ** 2).sum(-1)
        shp = d.shape[:-1]
        d2 = d.reshape(-1, d.shape[-1])
        if self.res_kind == 'lp':
            m = self.metric_scale * mng.group_lp_norm(d2, self.group, self.lp_p)
        else:
            m = self.metric_scale * (mng.MetricNN._group_max(d2, self.group, self.lse_beta)
                                     - (d.shape[-1] // self.group) * 1e-3)
            m = m.clamp_min(0.0)
        return (m ** 2).reshape(shp)

    # ------------------------------------------------------------------ value
    def _parts(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        qx, Fx, zx = self.features_cover(gx, ex)
        qy, Fy, zy = self.features_cover(gy, ey)
        a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1) / (self.s_t * self.s_t)
        c = (qx * qy).sum(-1)
        s2 = route_s2(qx, qy)
        self._conf = self.conf_terms(ex, ey) if self.conf else None
        return a, c, s2, Fx, Fy, zx, zy

    def _free_sq(self, tf2):
        """Squared free time(s) of the route(s), with the conformal term when enabled."""
        if self._conf is None:
            return tf2
        rb, lb = self._conf
        tf = torch.sqrt(tf2.clamp_min(1e-30))
        if tf.dim() == 2:
            rb, lb = rb[:, None], lb[:, None]
        tf = tf + rb * lb * torch.tanh(tf / lb)
        return tf * tf

    def all_candidates(self, gx, gy):
        """Squared value of every group element, (B, G), forward only (ridge diagnostics)."""
        a, c, s2, Fx, Fy, _, _ = self._parts(gx, gy)
        rho = self.rho(Fx[:, None, :] - self.signs[None] * Fy[:, None, :])        # (B, G)
        if self.tfree:
            tf2 = self._free_sq(free_routes_sq(a, s2, c.clamp(-1, 1), self.s_r))   # (B, 2)
            route = (self.sigmas < 0).long()                                         # 0 for +, 1 for -
            rho = rho + tf2[:, route]
        return rho

    def value_sq(self, gx, gy):
        a, c, s2, Fx, Fy, zx, zy = self._parts(gx, gy)
        with torch.no_grad():
            cand = self.rho(Fx[:, None, :] - self.signs[None] * Fy[:, None, :])
            if self.tfree:
                route = (self.sigmas < 0).long()
                cand = cand + self._free_sq(free_routes_sq(a, s2, c.clamp(-1, 1), self.s_r))[:, route]
            idx = cand.argmin(1)
        sigma = self.sigmas[idx]
        if self.soft_tau > 0 and self.m > 0:
            # all learned-cut elements of the selected rotation route, soft minimum on T
            rows = torch.where(sigma[:, None] > 0, self._rows_plus[None], self._rows_minus[None])   # (B, G/2)
            signs = self.signs[rows]                                                                # (B, G/2, D)
            Q = self.rho(Fx[:, None, :] - signs * Fy[:, None, :])
            if self.tfree:
                Q = Q + self._free_sq(a + route_angle_sq(s2, sigma * c) / (self.s_r * self.s_r))[:, None]
            T = -self.soft_tau * torch.logsumexp(-torch.sqrt(Q.clamp_min(1e-30)) / self.soft_tau, dim=1)
            self._last = {'idx': idx, 'sigma': sigma, 'rho': Q.detach().amin(1)}
            return T * T.abs()                                                                      # sign-preserving square
        signs = self.signs[idx]                                                      # (B, dim_F)
        Q = self.rho(Fx - signs * Fy)
        if self.tfree:
            Q = Q + self._free_sq(a + route_angle_sq(s2, sigma * c) / (self.s_r * self.s_r))
        self._last = {'idx': idx, 'sigma': sigma, 'rho': Q.detach()}
        return Q

    def value(self, gx, gy):
        Q = self.value_sq(gx, gy)
        if self.soft_tau > 0 and self.m > 0:
            return torch.sign(Q) * torch.sqrt(Q.abs())
        return torch.sqrt(Q)

    def free_time(self, gx, gy):
        return free_time(gx, gy, self.s_t, self.s_r)


ARCHITECTURES = {
    'graph_cover': (CoverNN, mng.FEATURE_DIM),
}
