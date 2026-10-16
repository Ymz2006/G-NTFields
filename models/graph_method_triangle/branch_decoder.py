"""Min-of-branches decoder for the SE(3) travel time ("Decoder properties that hold with or without
obstacles", 2026-10-13).

The fixed (a, b) decoder T = sqrt(a / s_t^2 + phi(b) / s_r^2) of graph_gram puts its only ridge where the
learned b reaches 1 (the half-turn of its Gram lift) and caps the rotation term at pi^2 / s_r^2; extra time
can only go into the Euclidean a-latent, which has no ridges. This module replaces it by the reference
design of the properties note, a minimum over smooth branches:

    T(x, y) = min_{sigma in {+, -}} min_{k = 1..K}  T_{sigma,k}(x, y)
    T_{sigma,k} = T_free,sigma(x, y) + C_sigma(x, y) + delta_{sigma,k}(x, y)

* T_free,sigma: the obstacle-free time along rotation route sigma. With unit quaternions q_x, q_y of R_x, R_y
  (either sign) and c = q_x . q_y, route + turns by psi_+ = 2 atan2(|v|, c) in [0, 2 pi] (v = vector part of
  q_x^* q_y) and route - by psi_- = 2 pi - psi_+, so
      T_free,sigma = sqrt(|t_x - t_y|^2 / s_t^2 + psi_sigma^2 / s_r^2),   min_sigma T_free,sigma = T_free,
  the exact speed-limit time. The two routes are the copies of the R^3 x S^3 product metric under the deck
  transformation q -> -q (property 3b); they are continuous through the half-turn, so a route keeps its
  identity (and its residual) when it becomes the long way round.
* delta_{sigma,k} >= 0: residual on route sigma, branch k, from a per-pose feature map on the double cover
      F_k(x~) = [U_k e(x),  sum_i q_i W_{k,i} e(x)]       (even part | odd part in q),
  e(x) the G-Shape embedding of G(x) = (t, vec R) (even in q). sigma = + compares F_k(x~) with F_k(y~),
  sigma = - with F_k(-y~), i.e. the odd part enters with the opposite sign. Flipping the sign of q_y swaps
  the two routes and their residuals, so T is a function on SE(3) x SE(3) whatever signs the quaternion
  extraction picks. 'sq': delta = |dF|^2; 'lp': delta = m(dF)^2 with m = metric_scale sum_g |dF_g|_p, a sum of
  group l_p norms (p = 8: close to the l_inf structure of graph_metric's distance, but smooth wherever a whole
  group is non-zero, i.e. everywhere except at the goal along generic curves); 'lse': m = the G-Shape logsumexp
  group distance (graph_metric's). All vanish quadratically at the goal and are unbounded. 'lse' smooths |d_k|
  as sqrt(d_k^2 + 1e-6): a near-V kink of width ~1e-3 wherever a residual coordinate crosses 0, i.e. it meets
  property 7 only above that scale (tests/test_branch_decoder.py measures it). Optional pseudo-Huber growth
  (res_huber = eps > 0, eps learned): delta = sqrt(eps^2 + D) - eps with D the squared residual above, i.e.
  D / (2 eps) near the goal (still quadratic: properties 5, 6 unchanged) and sqrt(D) - eps far away, a
  metric-like linear growth (like graph_metric's T ~ m) instead of a quadratic one; smooth and unbounded.
* C_sigma (conf=True): rho_bar lam_bar tanh(T_free,sigma / lam_bar), rho_bar = (rho_x + rho_y) / 2 >= 0,
  lam_bar = (lam_x + lam_y) / 2 > 0 (per-pose softplus heads). Near the goal C = rho(p) T_free + O(T_free^3),
  so the cone keeps the metric's shape scaled by 1 + rho(p): the cone of the speed-weighted metric g / S(p)^2
  when 1 + rho(p) = 1 / S(p) (isotropic speed model). Far away it saturates at rho_bar lam_bar (an escape-time
  offset). conf=False is the strict form of the note (the cone is the free metric's, valid where S = 1).

Properties guaranteed for every parameter value (tests/test_branch_decoder.py checks each):
 1 T >= 0; T(x, x) = 0 exactly (route + with q_y = q_x: psi = 0, C = 0, delta = 0); T > 0 for x != y (T >= T_free).
 2 T(x, y) = T(y, x) (every term is a symmetric comparison of per-pose quantities).
 4 T >= T_free (C, delta >= 0).
 5 T = sqrt(Q) near the goal with Q smooth to second order, Q(x, x) = 0, grad Q(x, x) = 0, in both endpoints.
 6 T(p exp(xi), p) = (1 + rho(p)) sqrt(g(xi, xi)) + O(|xi|^2); Hess_xi Q = 2 (1 + rho(p))^2 diag(s_t^-2 I, s_r^-2 I)
   (rho = 0 for conf=False: exactly the note's formula).
 7 the only non-smooth operations are the min (peaks only), the goal cone and the far point of a route
   (psi -> 2 pi, a concave peak); no max / abs / ReLU on the output path.
 8 the ridge is wherever two branches tie: it moves with the learned residuals (route + vs route -, branch k vs j).
 9 delta and C are unbounded, so any travel time is representable.
 3 (triangle inequality) is not guaranteed: T_free + a squared residual is not a metric. It is measured.

The minimum is hard (a soft minimum would break property 1 and 4). The branch is selected without autograd
and only the selected branch is recomputed with gradients, so the unused branches' singular points (the far
point of a route, where d psi / d|v| is infinite) cannot inject inf * 0 = NaN into double backward.
"""
import math

import torch
from torch.nn import Linear

from . import model_network_graph as mng
from .interface import PairValueModel

TWO_PI = 2.0 * math.pi
_SERIES_S2 = 1e-6  # psi^2 series switch on |v|^2


def quat_from_matrix(R):
    """Unit quaternion (w, x, y, z) of R (..., 3, 3) by Shepperd's method.

    The pivot is the largest of the four radicands 4 q_i^2 (>= 1, since they sum to 4), so the selected
    formula is well conditioned; the three unused candidates are evaluated on a safe pivot (1) and dropped by
    the gather, whose backward gives them exactly zero gradient. The sign is arbitrary (fixed per pivot
    region). Same convention as se3.quat_to_matrix.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    r = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    k = r.detach().argmax(-1)
    sel = torch.nn.functional.one_hot(k, 4).bool()
    s = 0.5 * torch.sqrt(torch.where(sel, r, torch.ones_like(r)).clamp_min(1e-6))   # (..., 4) pivot values
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
    ], -2)                                                                      # (..., 4 pivots, 4)
    q = torch.gather(cand, -2, k[..., None, None].expand(*k.shape, 1, 4)).squeeze(-2)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def route_s2(qx, qy):
    """|v|^2 = 1 - c^2 for the relative quaternion, cancellation-free: (1 - c)(1 + c) = |qx - qy|^2 |qx + qy|^2 / 4."""
    return ((qx - qy) ** 2).sum(-1) * ((qx + qy) ** 2).sum(-1) * 0.25


def route_angle_sq(s2, cc):
    """psi^2 = 4 atan2(sqrt(s2), cc)^2 for one route (cc = sigma c). Analytic in s2 at the route's own goal
    (cc > 0, s2 -> 0) through the series of atan(z)^2 = z^2 - 2/3 z^4 + 23/45 z^6 - 44/105 z^8."""
    small = (s2 < _SERIES_S2) & (cc > 0)
    z2 = s2 / torch.where(small, cc, torch.ones_like(cc)) ** 2
    series = 4.0 * z2 * (1.0 + z2 * (-2.0 / 3.0 + z2 * (23.0 / 45.0 - z2 * 44.0 / 105.0)))
    s_safe = torch.where(small, torch.ones_like(s2), s2).clamp_min(1e-20)
    exact = 4.0 * torch.atan2(torch.sqrt(s_safe), cc) ** 2
    return torch.where(small, series, exact)


def free_routes(tx, qx, ty, qy, s_t, s_r):
    """(T_free,+, T_free,-), each (B,), forward values (no singular-point care: selection / diagnostics)."""
    a = ((tx - ty) ** 2).sum(-1) / (s_t * s_t)
    c = (qx * qy).sum(-1).clamp(-1.0, 1.0)
    s = torch.sqrt(route_s2(qx, qy).clamp_min(0.0))
    psi_p = 2.0 * torch.atan2(s, c)
    psi_m = TWO_PI - psi_p
    return torch.sqrt(a + psi_p ** 2 / (s_r * s_r)), torch.sqrt(a + psi_m ** 2 / (s_r * s_r))


def free_time(gx, gy, s_t, s_r):
    """Exact obstacle-free time T_free(x, y) from the embeddings (the speed-limit lower bound, property 4)."""
    qx = quat_from_matrix(gx[:, 3:12].reshape(-1, 3, 3))
    qy = quat_from_matrix(gy[:, 3:12].reshape(-1, 3, 3))
    tp, tm = free_routes(gx[:, :3], qx, gy[:, :3], qy, s_t, s_r)
    return torch.minimum(tp, tm)


def _inv_softplus(y):
    return math.log(math.expm1(y))


def group_lp(d, group, p):
    """sum_g |d_g|_p over groups of the last axis, exactly 0 where a group is 0 (property 1).

    Same scaling trick as model_network_graph.group_lp_norm (a detached per-group max M, value exact for every
    M > 0) without its 1e-30 floor, which would leave T(x, x) ~ 1e-61 instead of 0."""
    x = d.abs().reshape(*d.shape[:-1], -1, group)
    M = x.amax(-1, keepdim=True).detach()
    Ms = M.clamp_min(1e-30)
    s = ((x / Ms) ** p).sum(-1).clamp_min(1.0)
    val = Ms[..., 0] * s ** (1.0 / p)
    return torch.where(M[..., 0] > 1e-30, val, torch.zeros_like(val)).sum(-1)


class BranchNN(PairValueModel, mng._Siamese):
    """Min-of-branches head (module doc). Options:

    n_branch   K residual branches per rotation route (K = 1: only the two +-q routes)
    res_even   dimension of the even residual coordinates U_k e(x)
    res_odd    dimension of the odd residual coordinates sum_i q_i W_{k,i} e(x)
    res_kind   'sq' (|dF|^2), 'lp' (m(dF)^2, m a sum of group l_p norms) or 'lse' (G-Shape logsumexp groups)
    res_scale  multiplies F (init scale of the residual)
    conf       add the conformal cone term C (rho, lam per-pose heads)
    res_huber  > 0: pseudo-Huber residual sqrt(eps^2 + D) - eps with learned eps (initialised at this value)
    """
    has_latent = False

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=False, skip_input=False, lip_cap=1.0,
                 n_branch=1, res_even=16, res_odd=16, res_kind='sq', res_scale=1.0, conf=False, rho_init=0.5,
                 lam_init=0.05, lam_min=1e-3, metric_scale=0.2, group=16, lse_beta=10.0, lp_p=8.0, res_huber=0.0,
                 **extra):
        PairValueModel.__init__(self, s_t, s_r)
        assert B.shape[1] == mng.FEATURE_DIM, 'branch heads read G(x) = (t, vec R)'
        assert res_kind in ('sq', 'lp', 'lse'), res_kind
        self.enc = self.ENCODER(B.shape[1], B, h_size, nl1, normalize, skip_input, lip_cap,
                                **self.encoder_extra(extra))
        self.K, self.me, self.mo = int(n_branch), int(res_even), int(res_odd)
        self.res_kind, self.res_scale, self.conf = res_kind, float(res_scale), bool(conf)
        self.metric_scale, self.group, self.lse_beta = float(metric_scale), int(group), float(lse_beta)
        self.lp_p = float(lp_p)
        if res_kind in ('lp', 'lse'):
            assert self.me % self.group == 0 and self.mo % self.group == 0, 'lse groups must tile both parts'
        self.U = Linear(self.enc.out_dim, self.K * self.me, bias=False)
        self.W = Linear(self.enc.out_dim, self.K * self.mo * 4, bias=False)
        self.lam_min = float(lam_min)
        self.res_huber = float(res_huber)
        if self.res_huber > 0:
            self.log_eps = torch.nn.Parameter(torch.tensor(math.log(self.res_huber)))
        if self.conf:
            self.rho_head = Linear(self.enc.out_dim, 1)
            self.lam_head = Linear(self.enc.out_dim, 1)
        mng._init(self, self.enc)
        if self.conf:
            with torch.no_grad():
                self.rho_head.weight.zero_()
                self.rho_head.bias.fill_(_inv_softplus(max(rho_init, 1e-4)))
                self.lam_head.weight.zero_()
                self.lam_head.bias.fill_(_inv_softplus(max(lam_init - lam_min, 1e-6)))
        self._last = {}

    # ------------------------------------------------------------------ per-pose quantities
    def pose_terms(self, g, e):
        """q (N, 4), even (N, K, me), odd (N, K, mo) residual coordinates, rho (N,), lam (N,)."""
        q = quat_from_matrix(g[:, 3:12].reshape(-1, 3, 3))
        n = g.shape[0]
        ev = self.res_scale * self.U(e).view(n, self.K, self.me)
        od = self.res_scale * torch.einsum('nkmi,ni->nkm', self.W(e).view(n, self.K, self.mo, 4), q)
        if self.conf:
            rho = torch.nn.functional.softplus(self.rho_head(e)[:, 0])
            lam = self.lam_min + torch.nn.functional.softplus(self.lam_head(e)[:, 0])
        else:
            rho = lam = None
        return q, ev, od, rho, lam

    def endpoint_slope(self, g):
        """Cone slope factor 1 + rho(p) at a goal p: lim T(p exp(eps xi), p) / (eps |xi|_g) (property 6)."""
        if not self.conf:
            return torch.ones(g.shape[0], dtype=g.dtype, device=g.device)
        e = self.enc(g)
        return 1.0 + torch.nn.functional.softplus(self.rho_head(e)[:, 0])

    def _metric(self, d):
        """Group distance of d (..., m): 'lp' sum_g |d_g|_p, 'lse' G-Shape logsumexp groups (offset removed)."""
        if self.res_kind == 'lp':
            return self.metric_scale * group_lp(d, self.group, self.lp_p)
        x = torch.sqrt(d ** 2 + 1e-6)
        x = x.reshape(*d.shape[:-1], -1, self.group)
        m = ((torch.logsumexp(self.lse_beta * x, -1) - math.log(self.group)) / self.lse_beta).sum(-1)
        return self.metric_scale * (m - (d.shape[-1] // self.group) * 1e-3).clamp_min(0.0)

    def residuals(self, evx, odx, evy, ody):
        """delta (B, 2, K) for routes (+, -) and branches k."""
        de = evx - evy
        dp, dm = odx - ody, odx + ody
        if self.res_kind == 'sq':
            e2 = (de ** 2).sum(-1)
            D = torch.stack([e2 + (dp ** 2).sum(-1), e2 + (dm ** 2).sum(-1)], 1)
        else:
            D = torch.stack([self._metric(torch.cat([de, dp], -1)) ** 2,
                             self._metric(torch.cat([de, dm], -1)) ** 2], 1)
        if self.res_huber > 0:
            eps = torch.exp(self.log_eps)
            return D / (torch.sqrt(eps * eps + D) + eps)   # = sqrt(eps^2 + D) - eps, no cancellation near D = 0
        return D

    # ------------------------------------------------------------------ value
    def _parts(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        qx, evx, odx, rhox, lamx = self.pose_terms(gx, ex)
        qy, evy, ody, rhoy, lamy = self.pose_terms(gy, ey)
        return qx, qy, evx, odx, evy, ody, rhox, rhoy, lamx, lamy

    def branch_values(self, gx, gy):
        """All branch values (B, 2, K), forward only (diagnostics: active branch, ridge detection)."""
        qx, qy, evx, odx, evy, ody, rhox, rhoy, lamx, lamy = self._parts(gx, gy)
        tp, tm = free_routes(gx[:, :3], qx, gy[:, :3], qy, self.s_t, self.s_r)
        tf = torch.stack([tp, tm], 1)                                        # (B, 2)
        if self.conf:
            rb, lb = 0.5 * (rhox + rhoy), 0.5 * (lamx + lamy)
            tf = tf + (rb * lb)[:, None] * torch.tanh(tf / lb[:, None])
        return tf[:, :, None] + self.residuals(evx, odx, evy, ody)

    def value(self, gx, gy):
        qx, qy, evx, odx, evy, ody, rhox, rhoy, lamx, lamy = self._parts(gx, gy)
        delta = self.residuals(evx, odx, evy, ody)                           # (B, 2, K), smooth everywhere
        a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1) / (self.s_t * self.s_t)
        c = (qx * qy).sum(-1)
        s2 = route_s2(qx, qy)
        if self.conf:
            rb, lb = 0.5 * (rhox + rhoy), 0.5 * (lamx + lamy)
        with torch.no_grad():                                                # branch selection (piecewise constant)
            psi_p = 2.0 * torch.atan2(torch.sqrt(s2.clamp_min(0.0)), c)
            tf = torch.sqrt(a[:, None] + torch.stack([psi_p, TWO_PI - psi_p], 1) ** 2 / (self.s_r * self.s_r))
            if self.conf:
                tf = tf + (rb * lb)[:, None] * torch.tanh(tf / lb[:, None])
            idx = (tf[:, :, None] + delta).reshape(gx.shape[0], -1).argmin(1)
            sigma = 1.0 - 2.0 * (idx // self.K).to(c.dtype)                  # +1 / -1
            k = idx % self.K
        # recompute the selected route with gradients (only its own singular points are touched)
        Tf = torch.sqrt(a + route_angle_sq(s2, sigma * c) / (self.s_r * self.s_r))
        if self.conf:
            Tf = Tf + rb * lb * torch.tanh(Tf / lb)
        d_sel = delta.reshape(gx.shape[0], -1).gather(1, idx[:, None])[:, 0]
        # long_way: the selected route turns by more than a half turn (sigma c < 0), i.e. the decoder sends this
        # pair the other way round the rotation; the sign of sigma alone depends on the arbitrary lifts
        self._last = {'rho_x': rhox, 'rho_y': rhoy, 'sigma': sigma, 'k': k, 'long_way': (sigma * c.detach()) < 0}
        return Tf + d_sel

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2

    def free_time(self, gx, gy):
        return free_time(gx, gy, self.s_t, self.s_r)


class BranchConfNN(BranchNN):
    """Branch head with the conformal cone term on by default (cone = speed-weighted metric)."""

    def __init__(self, B, conf=True, **kw):
        super().__init__(B, conf=conf, **kw)


class GramConfNN(mng.GraphGramNN):
    """Ablation: graph_gram's fixed decoder D(a, b) plus the branch head's conformal term,
    T = D + rho_bar lam_bar tanh(D / lam_bar). Tests whether the conformal / escape-time term alone explains the
    branch heads' planning gains (it inherits graph_gram's decoder: capped rotation term, ridge only at b = 1)."""

    def __init__(self, B, rho_init=0.5, lam_init=0.05, lam_min=1e-3, **kw):
        super().__init__(B, **kw)
        self.lam_min = float(lam_min)
        self.rho_head = Linear(self.enc.out_dim, 1)
        self.lam_head = Linear(self.enc.out_dim, 1)
        with torch.no_grad():
            self.rho_head.weight.zero_()
            self.rho_head.bias.fill_(_inv_softplus(max(rho_init, 1e-4)))
            self.lam_head.weight.zero_()
            self.lam_head.bias.fill_(_inv_softplus(max(lam_init - lam_min, 1e-6)))
        self._last = {}

    def value(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        a = self.a_scale * (self.A(ex - ey) ** 2).sum(-1)
        Px, Py = self._gram(ex), self._gram(ey)
        b = (1.0 - mng.B_EPS) * (0.5 * ((Px - Py) ** 2).sum((-1, -2))).clamp(max=1.0)
        D = self.decode(a, b)
        rho = lambda e: torch.nn.functional.softplus(self.rho_head(e)[:, 0])
        lam = lambda e: self.lam_min + torch.nn.functional.softplus(self.lam_head(e)[:, 0])
        rb, lb = 0.5 * (rho(ex) + rho(ey)), 0.5 * (lam(ex) + lam(ey))
        return D + rb * lb * torch.tanh(D / lb)

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2


ARCHITECTURES = {
    'graph_branch': (BranchNN, mng.FEATURE_DIM),
    'graph_branch_conf': (BranchConfNN, mng.FEATURE_DIM),
    'graph_gram_conf': (GramConfNN, mng.FEATURE_DIM),
}
