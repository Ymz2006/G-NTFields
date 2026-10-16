"""Multi-branch decoder that learns the cut ("Spec: multi-branch decoder that learns the cut", 2026-10-13).

    T(x, y) = min_b T_b(x, y)    over the alive branches b = 0 .. n_branches - 1,

each T_b a smooth function of the pair. The ridge set (the cut) is wherever two branches tie, so its position
and the number of competing route families are learned. The spec's training procedure (models/graph_mb) trains
every branch on its competitive band B_b = {T_b <= T + Delta}, not only the winner, keeps duplicate branches
apart at ties and manages the branch population (usage, revival, pruning, merging). This module holds the two
head forms and the per-branch API the trainer needs:

    all_values(gx, gy)         -> (B, nb)  every branch's value (dead branches = +inf), forward only
    value_branch(gx, gy, b)    -> (B,)     the value of branch b[i] for pair i, differentiable (one pass)
    value(gx, gy)              -> (B,)     the hard minimum (evaluation / planning)
    n_units, unit_of_branch(b), direct_units, alive (buffer), copy_unit(dst, src, noise, raise_by),
    unit_param_slices(k), branch_stats()

'folded' form (``graph_mb``): the min-of-branches head of branch_decoder.py (the "branch K = 4 conformal" head,
the best decoder of the previous campaign) with detour offsets. Branch b = sigma K + k (route sigma in {+, -},
unit k):

    T_{sigma,k} = T_free,sigma + C_sigma + delta_{sigma,k} + o_k,       o_0 = 0,  o_k = softplus(beta_k)  (k >= 1)

T_free,sigma the exact free-space time along rotation route sigma (the +-q lifts), C_sigma the conformal cone term
rho_bar lam_bar tanh(T_free / lam_bar) (cone scale 1 + rho(p), learned per pose; ``conf``), delta the squared
residual |F_k(x~) - sigma F_k(y~)|^2 of unit k. Unit 0 on route + is the direct head: it is exactly 0 at x = y and
carries the goal cone (properties 1, 5, 6 of the properties note hold as before). Units k >= 1 are detours: their
offset o_k > 0 keeps them away from the goal, and the different initial heights o_k = k offset_init separate the
branches at the start (spec section 6, step 1). Route - of every unit is positive at the goal as well (psi = 2 pi).
A unit's two routes share its parameters, so the unit (not the route) is what revival, pruning and merging act on.

'free' form (``graph_mb_free``): the spec's heads as written, without the fold:

    Q_0 = (|A_0 (e_x - e_y)|^2 + rho_0 |G(x) - G(y)|^2) exp(2 tanh(g_0(s, d))),     T_0 = sqrt(Q_0)
    Q_k = softplus(f_k(s, d))^2,                                                    T_k = softplus(f_k)   (k >= 1)

with exchange-symmetric features s = e_x + e_y, d = (e_x - e_y) (.) (e_x - e_y), small per-branch MLPs g_0, f_k.
T_0 vanishes quadratically at x = y and only there (rho_0 > 0); the detours are positive everywhere. No free-space
lower bound, no conformal cone: the ablation the spec asks for (stages 0-2 default, stage 3a).

Both forms: the hard minimum selects the branch without autograd and recomputes only that branch with gradients,
so the other branches' singular points (the far point of a rotation route) cannot produce NaN in the double
backward of the Eikonal loss. Dead units (``alive[k] = 0``) are excluded from every minimum; ``alive`` is a buffer,
so a checkpoint carries the learned branch count.
"""
import math

import torch
from torch.nn import Linear

from . import branch_decoder as bd
from . import model_network_graph as mng
from .interface import PairValueModel

TWO_PI = 2.0 * math.pi
INF = float('inf')


def softmin(V, tau):
    """Soft minimum over the last axis, -tau log sum exp(-V / tau) (spec section 4); tau <= 0 = hard minimum."""
    if tau <= 0:
        return V.min(-1).values
    return -tau * torch.logsumexp(-V / tau, -1)


def _reset_adam_rows(optimizer, param, rows):
    """Zero the AdamW moments of `param` on the given row index tensor (None = whole tensor)."""
    st = optimizer.state.get(param)
    if not st:
        return
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if key in st:
            if rows is None:
                st[key].zero_()
            else:
                st[key][rows] = 0.0


class _BranchAPI:
    """Shared management helpers (both forms)."""
    direct_units = (0,)

    def alive_units(self):
        return [k for k in range(self.n_units) if self.alive[k] > 0.5]

    def branch_alive(self):
        """(nb,) bool: branch alive <=> its unit is alive."""
        return self.alive[self.unit_index] > 0.5

    def mask_dead(self, V):
        return torch.where(self.branch_alive()[None, :], V, torch.full_like(V, INF))

    def reset_optimizer_unit(self, optimizer, k):
        for p, rows in self.unit_param_slices(k):
            _reset_adam_rows(optimizer, p, rows)

    def softmin_value(self, gx, gy, tau):
        """Soft-min decoded value over alive branches (diagnostics / the soft-min test), forward only."""
        with torch.no_grad():
            return softmin(self.all_values(gx, gy), tau)


class MultiBranchNN(_BranchAPI, bd.BranchNN):
    """Folded form (module doc). Options of bd.BranchNN plus:

    n_branch     K units (route +/- each): 2K branches
    offset_init  initial detour height o_k = k * offset_init (k >= 1)
    conf         conformal cone term (default True: the previous campaign's best head)
    """
    FORM = 'folded'

    def __init__(self, B, n_branch=4, offset_init=0.015, conf=True, **kw):
        bd.BranchNN.__init__(self, B, n_branch=n_branch, conf=conf, **kw)
        K = self.K
        self.n_units, self.n_branches = K, 2 * K
        self.register_buffer('alive', torch.ones(K))
        self.register_buffer('unit_index', torch.arange(2 * K) % K)
        self.offset_init = float(offset_init)
        if K > 1:
            init = torch.tensor([bd._inv_softplus(max(offset_init * k, 1e-4)) for k in range(1, K)])
            self.beta = torch.nn.Parameter(init)

    # ------------------------------------------------------------------ branch bookkeeping
    def unit_of_branch(self, b):
        return b % self.K

    def offsets(self):
        """(K,) detour offsets, o_0 = 0."""
        z = self.alive.new_zeros(1)
        if self.K == 1:
            return z
        return torch.cat([z, torch.nn.functional.softplus(self.beta)])

    def unit_param_slices(self, k):
        out = [(self.U.weight, torch.arange(k * self.me, (k + 1) * self.me)),
               (self.W.weight, torch.arange(k * self.mo * 4, (k + 1) * self.mo * 4))]
        if k >= 1:
            out.append((self.beta, torch.tensor([k - 1])))
        return out

    @torch.no_grad()
    def copy_unit(self, dst, src, noise=0.1, raise_by=0.0):
        """Revival (spec section 6, step 6): unit dst <- unit src's residual map with relative weight noise
        and its offset raised by `raise_by`. dst must be a detour unit (>= 1)."""
        assert dst >= 1 and dst != src
        for (pd, rd), (ps, rs) in zip(self.unit_param_slices(dst), self.unit_param_slices(src)):
            if pd is self.beta and src == 0:
                continue
            pd[rd] = ps[rs]
            if pd is not self.beta:
                pd[rd] += noise * pd[rd].std().nan_to_num(0.0) * torch.randn_like(pd[rd])
        o_src = self.offsets()[src].item()
        self.beta[dst - 1] = bd._inv_softplus(max(o_src + raise_by, 1e-4))
        self.alive[dst] = 1.0

    # ------------------------------------------------------------------ values
    def _route_free(self, a, s2, c, sigma, rb, lb):
        Tf = torch.sqrt(a + bd.route_angle_sq(s2, sigma * c) / (self.s_r * self.s_r))
        if self.conf:
            Tf = Tf + rb * lb * torch.tanh(Tf / lb)
        return Tf

    def all_values(self, gx, gy):
        with torch.no_grad():
            V = self.branch_values(gx, gy) + self.offsets()[None, None, :]       # (B, 2, K)
            return self.mask_dead(V.reshape(gx.shape[0], -1))

    def _value_at(self, gx, gy, b=None):
        qx, qy, evx, odx, evy, ody, rhox, rhoy, lamx, lamy = self._parts(gx, gy)
        delta = self.residuals(evx, odx, evy, ody)                           # (B, 2, K)
        a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1) / (self.s_t * self.s_t)
        c = (qx * qy).sum(-1)
        s2 = bd.route_s2(qx, qy)
        rb = lb = None
        if self.conf:
            rb, lb = 0.5 * (rhox + rhoy), 0.5 * (lamx + lamy)
        off = self.offsets()
        if b is None:
            with torch.no_grad():
                psi_p = 2.0 * torch.atan2(torch.sqrt(s2.clamp_min(0.0)), c)
                tf = torch.sqrt(a[:, None] + torch.stack([psi_p, TWO_PI - psi_p], 1) ** 2 / (self.s_r * self.s_r))
                if self.conf:
                    tf = tf + (rb * lb)[:, None] * torch.tanh(tf / lb[:, None])
                V = (tf[:, :, None] + delta + off[None, None, :]).reshape(gx.shape[0], -1)
                b = self.mask_dead(V).argmin(1)
        sigma = 1.0 - 2.0 * (b // self.K).to(c.dtype)
        k = b % self.K
        Tf = self._route_free(a, s2, c, sigma, rb, lb)
        d_sel = delta.reshape(gx.shape[0], -1).gather(1, b[:, None])[:, 0]
        self._last = {'rho_x': rhox, 'rho_y': rhoy, 'sigma': sigma, 'k': k, 'b': b,
                      'long_way': (sigma * c.detach()) < 0}
        return Tf + d_sel + off[k]

    def value_branch(self, gx, gy, b):
        return self._value_at(gx, gy, b)

    def value(self, gx, gy):
        return self._value_at(gx, gy, None)

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2


class _Head(torch.nn.Module):
    """Small MLP on the symmetric pair features (s, d) -> scalar; bias_init sets the initial height."""

    def __init__(self, in_dim, hidden=64, bias_init=0.0):
        super().__init__()
        self.l1 = Linear(in_dim, hidden)
        self.l2 = Linear(hidden, hidden)
        self.l3 = Linear(hidden, 1)
        self.act = torch.nn.Softplus(beta=10)
        with torch.no_grad():
            self.l3.weight.mul_(0.1)
            self.l3.bias.fill_(bias_init)

    def forward(self, z):
        return self.l3(self.act(self.l2(self.act(self.l1(z)))))[:, 0]


class MultiBranchFreeNN(_BranchAPI, PairValueModel, mng._Siamese):
    """Free form (module doc): direct head T_0 = sqrt(Q_0), detour heads T_k = softplus(f_k); nb = n_units = K.

    n_branch   K branches
    rho0       weight of |G(x) - G(y)|^2 in Q_0 (spec default 0.01)
    k_a        rows of A_0
    head_hidden  MLP width of g_0 / f_k
    offset_init  initial detour height: f_k's output bias = inverse softplus(k * offset_init + d0) with d0 the
                 typical direct value (0.3)
    """
    FORM = 'free'
    has_latent = False

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=False, skip_input=False, lip_cap=1.0,
                 n_branch=4, rho0=0.01, k_a=16, head_hidden=64, offset_init=0.015, direct_level=0.3, **extra):
        PairValueModel.__init__(self, s_t, s_r)
        assert B.shape[1] == mng.FEATURE_DIM
        self.enc = self.ENCODER(B.shape[1], B, h_size, nl1, normalize, skip_input, lip_cap,
                                **self.encoder_extra(extra))
        K = int(n_branch)
        self.K = self.n_units = self.n_branches = K
        self.rho0 = float(rho0)
        self.register_buffer('alive', torch.ones(K))
        self.register_buffer('unit_index', torch.arange(K))
        self.A0 = Linear(self.enc.out_dim, k_a, bias=False)
        self.heads = torch.nn.ModuleList(
            [_Head(2 * self.enc.out_dim, head_hidden, 0.0)] +
            [_Head(2 * self.enc.out_dim, head_hidden, bd._inv_softplus(direct_level + offset_init * k))
             for k in range(1, K)])
        self.enc.apply(self.enc.init_weights)
        self.enc.reset_gates()
        self.A0.apply(self.enc.init_weights)
        self._last = {}

    def unit_of_branch(self, b):
        return b

    def unit_param_slices(self, k):
        return [(p, None) for p in self.heads[k].parameters()]

    @torch.no_grad()
    def copy_unit(self, dst, src, noise=0.1, raise_by=0.0):
        assert dst >= 1 and dst != src
        if src == 0:   # the direct head has a different parametrisation: restart the detour at a raised height
            for p in self.heads[dst].parameters():
                p.normal_(0, 0.02)
            self.heads[dst].l3.bias.fill_(bd._inv_softplus(0.3 + raise_by))
        else:
            for pd, ps in zip(self.heads[dst].parameters(), self.heads[src].parameters()):
                pd.copy_(ps)
                scale = pd.std() if pd.numel() > 1 else pd.abs()
                pd.add_(noise * scale * torch.randn_like(pd))
            # softplus(f + c) >= softplus(f) + ...: raise the pre-activation so the value rises by ~raise_by
            self.heads[dst].l3.bias.add_(raise_by)
        self.alive[dst] = 1.0

    # ------------------------------------------------------------------ values
    def _feats(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        z = torch.cat([ex + ey, (ex - ey) ** 2], -1)
        base = (self.A0(ex - ey) ** 2).sum(-1) + self.rho0 * ((gx - gy) ** 2).sum(-1)
        return z, base

    def _branch(self, z, base, k):
        if k == 0:
            return torch.sqrt(base.clamp_min(0.0)) * torch.exp(torch.tanh(self.heads[0](z)))
        return torch.nn.functional.softplus(self.heads[k](z))

    def all_values(self, gx, gy):
        with torch.no_grad():
            z, base = self._feats(gx, gy)
            V = torch.stack([self._branch(z, base, k) for k in range(self.K)], 1)
            return self.mask_dead(V)

    def value_branch(self, gx, gy, b):
        z, base = self._feats(gx, gy)
        out = torch.zeros(gx.shape[0], dtype=gx.dtype, device=gx.device)
        for k in range(self.K):
            m = b == k
            if m.any():
                out = out + torch.where(m, self._branch(z, base, k), torch.zeros_like(out))
        self._last = {'b': b}
        return out

    def value(self, gx, gy):
        b = self.all_values(gx, gy).argmin(1)
        return self.value_branch(gx, gy, b)

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2


ARCHITECTURES = {
    'graph_mb': (MultiBranchNN, mng.FEATURE_DIM),
    'graph_mb_free': (MultiBranchFreeNN, mng.FEATURE_DIM),
}
