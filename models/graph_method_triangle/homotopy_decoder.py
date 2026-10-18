"""Homotopy (double-cover) decoder for the SE(3) travel time: one cone, one ridge.

The field is first defined on the double cover of the pose space,

    T~ : (R^3 x S^3) x (R^3 x S^3) -> R,      x~ = (t_x, q_x),  y~ = (t_y, q_y),

with unit quaternions q (both signs are distinct points of S^3). The deck transformation q -> -q acts on
both arguments at once; flipping both quaternions is the same pair of poses and must give the same time,

    T~(-x~, -y~) = T~(x~, y~)                                                                  (deck invariance)

which the architecture builds in by only reading invariants of the diagonal action: the even per-pose
embedding G(x) = (t, vec R) (R = R(q) = R(-q)), the relative lifted rotation q_x^* q_y = (c, v) (c = q_x . q_y;
invariant under the simultaneous flip, negated under a single flip) and the q-odd coordinates
Q(x~) = sum_i q_i W_i e(x) compared through |Q(x~) - Q(y~)|^2 (a quadratic form in q_x (x) q_y). Paths in the
cover from x~ to y~ are the paths in SE(3) from x to y in one homotopy class (pi_1(SO(3)) = Z_2); the
cover is simply connected, so T~(., y~) is the smooth-in-the-sense-of-Eikonal time to the lifted goal y~ and
has no rotational ridge of its own. The time between poses is the minimum over the two classes,

    T(x, y) = min( T~(x~, y~), T~(x~, -y~) )                                                   (the ridge)

the only min of the decoder: its ridge is where the two homotopy classes tie. Any lift of x and y may be
used (deck invariance makes the set {T~(x~, +-y~)} independent of the lifts).

Cover field (per sheet, all terms >= 0, every term symmetric):

    T~ = T_free (1 + P) + C + delta
    T_free = sqrt(|t_x - t_y|^2 / s_t^2 + psi^2 / s_r^2),  psi = 2 atan2(|v|, c) in [0, 2 pi]   (the cone)
    C      = rho_bar lam_bar tanh(T_free / lam_bar)            conformal cone scale 1 + rho(p) (per-pose heads)
    delta  = |E(x) - E(y)|^2 + |Q(x~) - Q(y~)|^2                smooth deviation (even | q-odd coordinates)
    P      = softplus(h), h = (h'(x~, y~) + h'(y~, x~)) / 2     optional pair term (pair_dim > 0), h' an MLP on
                                                               the invariants (p(e_x), p(e_y), c, v, T_free)

Properties for every parameter value (tests/test_graph_homotopy.py): T >= T_free >= 0, T(x, x) = 0 exactly,
T symmetric, T = sqrt(Q) with Q smooth to second order at the goal, cone = the metric's shape scaled by
1 + rho(p) + P(p, p) (conformal), the only non-smooth operations on the output path are the goal cone, the
far point of a sheet (psi -> 2 pi, a concave peak) and the one min (peaks only), the ridge moves with the
learned terms, no cap, deck invariance exact (bitwise up to the lifts' rounding), lifting consistency
value(gx, gy) = min_sigma sheet_value(gx, gy, sigma) = min_sigma value_cover(t, q_x, t, sigma q_y).

Training may use the base min (lift 'min', the sheet selected without autograd and recomputed with gradients,
as in the branch decoder) or lift the training pairs to the cover (lift 'random' / 'both', models/graph_homotopy),
which trains both sheets everywhere: T~ is then the Eikonal solution on the cover, and planning can descend
one sheet (plan to +q_goal or -q_goal) without ever meeting the ridge.
"""
import math

import torch
from torch.nn import Linear

from . import model_network_graph as mng
from .branch_decoder import TWO_PI, _inv_softplus, free_time, quat_from_matrix, route_angle_sq, route_s2
from .geometry import se3
from .interface import PairValueModel


# --------------------------------------------------------------------------- quaternion helpers (w, x, y, z)
def quat_mul(p, q):
    """Hamilton product p q, (..., 4) x (..., 4) -> (..., 4)."""
    pw, pv = p[..., :1], p[..., 1:]
    qw, qv = q[..., :1], q[..., 1:]
    w = pw * qw - (pv * qv).sum(-1, keepdim=True)
    v = pw * qv + qw * pv + torch.cross(pv, qv, dim=-1)
    return torch.cat([w, v], -1)


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def relative_quat(qx, qy):
    """q_x^* q_y = (c, v): the lifted relative rotation (R_x^T R_y = R(q_x^* q_y)); invariant under (q_x, q_y) ->
    (-q_x, -q_y), negated under a single flip; conjugated (v -> -v) under x <-> y."""
    return quat_mul(quat_conj(qx), qy)


def quat_exp_turns(r):
    """Continuous lift of the rotation-vector chart: r (..., 3) in turns -> exp(pi r) = (cos(pi |r|), pi sinc(|r|) r).
    A path in the unwrapped chart lifts to a continuous path on S^3 (a full turn arrives at -q)."""
    n = torch.linalg.norm(r, dim=-1, keepdim=True)
    return torch.cat([torch.cos(math.pi * n), math.pi * torch.sinc(n) * r], -1)


class HomotopyNN(PairValueModel, mng._Siamese):
    """Cover-field head (module doc). Options:

    res_even / res_odd   dimensions of the even E(x) = U e(x) and odd Q(x~) = sum_i q_i W_i e(x) coordinates
    res_scale            multiplies E, Q (initial size of delta)
    conf                 conformal cone term C (per-pose rho, lam heads); False = strict free-metric cone
    pair_dim             > 0: pair term P from an MLP on the invariants (projection width pair_dim); 0 = off
    pair_hidden / pair_layers / pair_init   MLP width / depth / initial value of P
    """
    has_latent = False

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=False, skip_input=False, lip_cap=1.0,
                 res_even=16, res_odd=16, res_scale=1.0, conf=True, rho_init=0.5, lam_init=0.05, lam_min=1e-3,
                 pair_dim=0, pair_hidden=128, pair_layers=2, pair_init=0.05, **extra):
        PairValueModel.__init__(self, s_t, s_r)
        assert B.shape[1] == mng.FEATURE_DIM, 'the homotopy head reads G(x) = (t, vec R)'
        self.enc = self.ENCODER(B.shape[1], B, h_size, nl1, normalize, skip_input, lip_cap,
                                **self.encoder_extra(extra))
        self.me, self.mo = int(res_even), int(res_odd)
        self.res_scale, self.conf = float(res_scale), bool(conf)
        self.lam_min = float(lam_min)
        self.pair_dim = int(pair_dim)
        self.U = Linear(self.enc.out_dim, self.me, bias=False)
        self.W = Linear(self.enc.out_dim, self.mo * 4, bias=False)
        if self.conf:
            self.rho_head = Linear(self.enc.out_dim, 1)
            self.lam_head = Linear(self.enc.out_dim, 1)
        if self.pair_dim:
            self.pair_proj = Linear(self.enc.out_dim, self.pair_dim)
            layers, d = [], 2 * self.pair_dim + 5
            for _ in range(int(pair_layers)):
                layers += [Linear(d, int(pair_hidden)), torch.nn.SiLU()]
                d = int(pair_hidden)
            layers.append(Linear(d, 1))
            self.pair_mlp = torch.nn.Sequential(*layers)
        mng._init(self, self.enc)
        with torch.no_grad():
            if self.conf:
                self.rho_head.weight.zero_()
                self.rho_head.bias.fill_(_inv_softplus(max(rho_init, 1e-4)))
                self.lam_head.weight.zero_()
                self.lam_head.bias.fill_(_inv_softplus(max(lam_init - lam_min, 1e-6)))
            if self.pair_dim:
                self.pair_mlp[-1].weight.zero_()
                self.pair_mlp[-1].bias.fill_(_inv_softplus(max(pair_init, 1e-4)))
        self._last = {}

    # ------------------------------------------------------------------ per-pose quantities
    def pose_terms(self, e, q):
        """Per-pose terms of the lifted pose (e = enc(G(x)), q its lift): dict with ev (N, me), od (N, mo),
        rho, lam (N,) or None, p (N, pair_dim) or None."""
        n = e.shape[0]
        ev = self.res_scale * self.U(e)
        od = self.res_scale * torch.einsum('nmi,ni->nm', self.W(e).view(n, self.mo, 4), q)
        out = {'ev': ev, 'od': od, 'rho': None, 'lam': None, 'p': None}
        if self.conf:
            out['rho'] = torch.nn.functional.softplus(self.rho_head(e)[:, 0])
            out['lam'] = self.lam_min + torch.nn.functional.softplus(self.lam_head(e)[:, 0])
        if self.pair_dim:
            out['p'] = self.pair_proj(e)
        return out

    @staticmethod
    def flip(terms, sigma):
        """Terms of the other lift -q (sigma (N,) in {+1, -1}): only the odd coordinates change sign."""
        return dict(terms, od=terms['od'] * sigma[:, None])

    @staticmethod
    def geometry(tx, qx, ty, qy):
        """a = |dt|^2 (unscaled), c = q_x . q_y, s2 = |v|^2 (cancellation-free), v = vec(q_x^* q_y)."""
        a = ((tx - ty) ** 2).sum(-1)
        c = (qx * qy).sum(-1)
        s2 = route_s2(qx, qy)
        v = relative_quat(qx, qy)[:, 1:]
        return a, c, s2, v

    def pair_factor(self, px, py, c, v, tf):
        """P = softplus(h) >= 0, h symmetrised over x <-> y (p_x <-> p_y, v -> -v)."""
        n = c.shape[0]
        f = torch.cat([torch.cat([px, py, c[:, None], v, tf[:, None]], -1),
                       torch.cat([py, px, c[:, None], -v, tf[:, None]], -1)], 0)
        h = self.pair_mlp(f)[:, 0]
        return torch.nn.functional.softplus(0.5 * (h[:n] + h[n:]))

    def sheet(self, a, c, s2, v, tx_terms, ty_terms):
        """T~ for the sheet given by the geometry (a, c, s2, v) of the lifted pair and its per-pose terms."""
        tf = torch.sqrt(a / (self.s_t * self.s_t) + route_angle_sq(s2, c) / (self.s_r * self.s_r))
        T = tf
        if self.pair_dim:
            T = tf * (1.0 + self.pair_factor(tx_terms['p'], ty_terms['p'], c, v, tf))
        if self.conf:
            rb = 0.5 * (tx_terms['rho'] + ty_terms['rho'])
            lb = 0.5 * (tx_terms['lam'] + ty_terms['lam'])
            T = T + rb * lb * torch.tanh(tf / lb)
        delta = ((tx_terms['ev'] - ty_terms['ev']) ** 2).sum(-1) + ((tx_terms['od'] - ty_terms['od']) ** 2).sum(-1)
        return T + delta

    # ------------------------------------------------------------------ values
    def _lifted(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        qx = quat_from_matrix(gx[:, 3:12].reshape(-1, 3, 3))
        qy = quat_from_matrix(gy[:, 3:12].reshape(-1, 3, 3))
        return qx, qy, self.pose_terms(ex, qx), self.pose_terms(ey, qy)

    def sheet_value(self, gx, gy, sigma):
        """T~(x~, sigma y~) for the Shepperd lifts of gx, gy and the relative sign sigma (B,) in {+1, -1}."""
        qx, qy, Tx, Ty = self._lifted(gx, gy)
        a, c, s2, v = self.geometry(gx[:, :3], qx, gy[:, :3], qy)
        return self.sheet(a, sigma * c, s2, sigma[:, None] * v, Tx, self.flip(Ty, sigma))

    def sheet_values(self, gx, gy):
        """Both sheets (B, 2): columns (+, -) relative to the Shepperd lifts. Forward only (diagnostics)."""
        with torch.no_grad():
            qx, qy, Tx, Ty = self._lifted(gx, gy)
            a, c, s2, v = self.geometry(gx[:, :3], qx, gy[:, :3], qy)
            ones = torch.ones_like(c)
            tp = self.sheet(a, c, s2, v, Tx, Ty)
            tm = self.sheet(a, -c, s2, -v, Tx, self.flip(Ty, -ones))
        return torch.stack([tp, tm], 1)

    def value(self, gx, gy):
        """T(x, y) = min over the two homotopy classes. The class is selected without autograd and only the
        selected sheet is recomputed with gradients (the other sheet's far point cannot inject NaN)."""
        qx, qy, Tx, Ty = self._lifted(gx, gy)
        a, c, s2, v = self.geometry(gx[:, :3], qx, gy[:, :3], qy)
        with torch.no_grad():
            ones = torch.ones_like(c)
            tp = self.sheet(a, c, s2, v, Tx, Ty)
            tm = self.sheet(a, -c, s2, -v, Tx, self.flip(Ty, -ones))
            sigma = torch.where(tp <= tm, ones, -ones)
        T = self.sheet(a, sigma * c, s2, sigma[:, None] * v, Tx, self.flip(Ty, sigma))
        self._last = {'rho_x': Tx['rho'], 'rho_y': Ty['rho'], 'sigma': sigma,
                      'long_way': (sigma * c.detach()) < 0, 'gap': (tp - tm).abs()}
        return T

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2

    def value_cover(self, tx, qx, ty, qy):
        """The cover field T~(x~, y~) from lifted poses (t (B, 3), unit quaternion q (B, 4) of either sign)."""
        gx = se3.features(tx, se3.quat_to_matrix(qx))
        gy = se3.features(ty, se3.quat_to_matrix(qy))
        ex, ey = self.embed_pair(gx, gy)
        Tx, Ty = self.pose_terms(ex, qx), self.pose_terms(ey, qy)
        a, c, s2, v = self.geometry(tx, qx, ty, qy)
        return self.sheet(a, c, s2, v, Tx, Ty)

    def endpoint_slope(self, g):
        """Cone slope factor at a goal p: lim T(p exp(eps xi), p) / (eps |xi|_g) = 1 + rho(p) + P(p, p)."""
        e = self.enc(g)
        s = torch.ones(g.shape[0], dtype=g.dtype, device=g.device)
        if self.conf:
            s = s + torch.nn.functional.softplus(self.rho_head(e)[:, 0])
        if self.pair_dim:
            p = self.pair_proj(e)
            z = torch.zeros_like(s)
            s = s + self.pair_factor(p, p, torch.ones_like(s), torch.zeros(g.shape[0], 3, dtype=g.dtype, device=g.device), z)
        return s

    def free_time(self, gx, gy):
        return free_time(gx, gy, self.s_t, self.s_r)


ARCHITECTURES = {
    'graph_homotopy': (HomotopyNN, mng.FEATURE_DIM),
}
