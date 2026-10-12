"""Networks for the SE(3) smooth-graph travel-time field.

All variants share the G-Shape-NTFields encoder from
``models/metric/model_network_metric.py``: random Fourier features, Lipschitz
row-normalized linear layers, sine-modulated (u, v) gated residual blocks and
per-sample normalization.  They differ only in the pair head:

graph_gram   siamese encoder -> a = |A (e_x - e_y)|^2,
             b = 1/2 |P_x - P_y|_F^2 with P = L L^T / |L L^T|_F, L = 4x4 from e
             (b in [0, 1], exact zeros on the diagonal; the analytic
             solution P = q q^T is representable)
graph_sqdist siamese encoder -> a = |A Δe|^2, b = smooth_min(|B Δe|^2, 1)
graph_joint  joint encoder on (G(x), G(y)); raw outputs averaged under endpoint
             exchange; a = softplus(raw_a), b = sigmoid(raw_b)  (spec's
             optional adapter; not boundary-exact)
metric       siamese encoder + the original G-Shape logsumexp metric head
             (latent-metric baseline inspired by [1])
scalar       joint encoder, exchange-averaged softplus T (direct scalar baseline)

Triangle-inequality heads (this package; see GraphTriNN): a shared per-pose
encoder f(x) = (u(x), V(x)), V(x) in SO(3), and Z(x, y) built only from
f(x) and f(y), so that T is a pseudometric for *every* parameter value.

graph_tri         a = |u_x - u_y|^2,  u = A e            (the proposal as stated)
graph_tri_metric  a = m(x, y)^2, m a sum of group l_p norms of embedding
                  differences (exact-seminorm version of graph_metric's
                  logsumexp distance, which is not subadditive)
graph_tri_mix     a = |u_x - u_y|^2 + m(x, y)^2
all three         b = |V_x - V_y|_F^2 / 8 = (3 - tr(V_x^T V_y)) / 4
graph_gram_tri    graph_gram (relaxed, contractible Gram space) until tri_switch,
                  then V = rot6d(R(P)) from the same Gram matrix (exact)

Graph models output (a, b) and decode with the fixed analytic decoder D.
Every learned range adapter keeps b <= 1 - B_EPS strictly: phi'(b) is
genuinely infinite at b = 1, and a saturated adapter would otherwise produce
inf * 0 = NaN gradients.  Consequently exact half-turns are represented only
to within 2 sqrt(B_EPS) = 2e-3 rad (documented approximation, spec page 3).
"""

import math

import numpy as np
import torch
from torch.nn import Linear

from .interface import PairLatentModel, PairValueModel

FEATURE_DIM = 12
B_EPS = 1e-6


def make_fourier_matrix(n_freq, in_dim, sigma, seed, sigma_t=None, sigma_r=None, trunc='clamp'):
    """Random Fourier matrix; optional per-block std for the translation (t) and
    rotation columns of each pose block. trunc='clamp' clips N(0,1) at +-2 (used
    for the obstacle-free runs); 'resample' is a true truncated normal as in
    the metric baseline's trainer (trunc_normal_)."""
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(n_freq, in_dim, generator=g)
    if trunc == 'resample':
        bad = B.abs() > 2.0
        while bad.any():
            B[bad] = torch.randn(int(bad.sum()), generator=g)
            bad = B.abs() > 2.0
    else:
        B = torch.clamp(B, -2.0, 2.0)
    scale = torch.full((in_dim,), float(sigma))
    fd = FEATURE_DIM if in_dim % FEATURE_DIM == 0 else 6  # 12-d G(x) or 6-d (t, rotvec) blocks
    for blk in range(in_dim // fd):
        o = blk * fd
        if sigma_t is not None:
            scale[o:o + 3] = sigma_t
        if sigma_r is not None:
            scale[o + 3:o + fd] = sigma_r
    return B * scale


class GShapeEncoder(torch.nn.Module):
    """Port of the encoder in models/metric/model_network_metric.NN.out()."""

    def __init__(self, in_dim, B, h_size=256, nl1=2, normalize=True, skip_input=False):
        super().__init__()
        assert B.shape[0] * 2 == h_size, 'Fourier features must match h_size'
        self.register_buffer('B', B.t().contiguous())  # (in_dim, n_freq)
        self.in_dim = in_dim
        self.h_size = h_size
        self.nl1 = nl1
        self.normalize = normalize
        self.skip_input = skip_input
        self.act = torch.nn.Softplus(beta=10)
        self.pe_gate = torch.nn.ModuleList([Linear(h_size, h_size), Linear(h_size, h_size)])
        self.encoder = torch.nn.ModuleList([Linear(h_size, h_size) for _ in range(3 * nl1 + 1)])
        self.gate = torch.nn.ModuleList([Linear(1, 1) for _ in range(nl1)])
        self.out_dim = h_size + (in_dim if skip_input else 0)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            stdv = np.sqrt(2.0 / (m.weight.size(0) + m.weight.size(1)))
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stdv, a=-2.0 * stdv, b=2.0 * stdv)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def reset_gates(self):
        for g in self.gate:
            g.weight.data.fill_(0)
            g.bias.data.fill_(0)

    def input_mapping(self, x):
        x_proj = x @ (2.0 * np.pi * self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def lip_norm(self, w):
        absrowsum = torch.sqrt(torch.sum(w ** 2, dim=1)).detach()
        scale = 1 + 1e-5 - self.act(1 - 1 / absrowsum)
        return w * scale.unsqueeze(1)

    def _lin(self, layer, x):
        return x @ self.lip_norm(layer.weight).t() + layer.bias

    def forward(self, g):
        x = self.input_mapping(g)
        u = torch.sin(self._lin(self.pe_gate[0], x))
        v = torch.sin(self._lin(self.pe_gate[1], x))
        for ii in range(self.nl1):
            x_tmp = x
            y = self._lin(self.encoder[3 * ii], x)
            x = u * torch.sin(y) + v * (1 - torch.sin(y))
            y = self._lin(self.encoder[3 * ii + 1], x)
            x = u * torch.sin(y) + v * (1 - torch.sin(y))
            y = self._lin(self.encoder[3 * ii + 2], x)
            weight = torch.sigmoid(0.1 * self.gate[ii].weight)
            x = (1 - weight) * x_tmp + weight * torch.sin(y)
        y = self._lin(self.encoder[-1], x)
        if self.normalize:
            # InstanceNorm1d on a (N, h) tensor == per-row standardization.
            mu = y.mean(-1, keepdim=True)
            var = ((y - mu) ** 2).mean(-1, keepdim=True)
            y = (y - mu) / torch.sqrt(var + 1e-5)
        if self.skip_input:
            y = torch.cat([y, g], -1)
        return y


def _init(module, encoder):
    module.apply(encoder.init_weights)
    encoder.reset_gates()


class _Siamese(torch.nn.Module):
    """Shared helper: run both endpoints through one encoder in one batch."""

    def embed_pair(self, gx, gy):
        e = self.enc(torch.cat([gx, gy], 0))
        return e[: gx.shape[0]], e[gx.shape[0]:]


class GraphGramNN(PairLatentModel, _Siamese):
    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, k_a=16, gram_k=4,
                 normalize=False, skip_input=False, a_scale=1.0, gram_power=1, **_):
        PairLatentModel.__init__(self, s_t, s_r)
        self.enc = GShapeEncoder(B.shape[1], B, h_size, nl1, normalize, skip_input)
        self.A = Linear(self.enc.out_dim, k_a, bias=False)
        self.L = Linear(self.enc.out_dim, gram_k * gram_k)
        self.gram_k = gram_k
        self.gram_power = gram_power
        self.a_scale = a_scale
        _init(self, self.enc)

    def _gram(self, e):
        L = self.L(e).view(-1, self.gram_k, self.gram_k)
        M = L @ L.transpose(-1, -2)
        M = M / torch.sqrt((M * M).sum((-1, -2), keepdim=True) + 1e-12)
        for _ in range(self.gram_power - 1):  # spectral sharpening: favours rank-1 P
            M = M @ M
            M = M / torch.sqrt((M * M).sum((-1, -2), keepdim=True) + 1e-12)
        return M

    def predict_raw(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        a = self.a_scale * (self.A(ex - ey) ** 2).sum(-1)
        Px, Py = self._gram(ex), self._gram(ey)
        b = 0.5 * ((Px - Py) ** 2).sum((-1, -2))
        return torch.stack([a, b], -1)

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        # 0 <= b_raw <= 1 holds in exact arithmetic (PSD Gram inner products
        # are >= 0); the clamp only removes float rounding above 1.
        return raw[:, 0], (1.0 - B_EPS) * raw[:, 1].clamp(max=1.0)


class GraphMetricNN(GraphGramNN):
    """Stage-E hypothesis ("richer latent"): a = m(x, y)^2 where m is the
    G-Shape logsumexp metric distance between normalised embeddings (an
    L-infinity-type metric, not Euclidean), b from the PSD Gram head, same
    analytic decoder. Symmetric and exactly zero on the diagonal up to the
    metric head's 1e-3 smoothing offset, which is subtracted."""

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, gram_k=4, skip_input=False,
                 metric_scale=0.2, group=16, lse_beta=10.0, **kw):
        kw.pop('normalize', None)  # the metric branch normalises its own embedding
        super().__init__(B, s_t=s_t, s_r=s_r, h_size=h_size, nl1=nl1, gram_k=gram_k, normalize=False,
                         skip_input=skip_input, **kw)
        self.metric_scale, self.group, self.lse_beta = metric_scale, group, float(lse_beta)

    def predict_raw(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        h = self.enc.h_size

        def norm(e):
            mu = e.mean(-1, keepdim=True)
            return (e - mu) / torch.sqrt(((e - mu) ** 2).mean(-1, keepdim=True) + 1e-5)
        m = self.metric_scale * (MetricNN._group_max(norm(ex[:, :h]) - norm(ey[:, :h]), self.group, self.lse_beta)
                                 - (h // self.group) * 1e-3)
        Px, Py = self._gram(ex), self._gram(ey)
        b = 0.5 * ((Px - Py) ** 2).sum((-1, -2))
        return torch.stack([m.clamp_min(0) ** 2, b], -1)


def rot6d_to_matrix(r6):
    """Gram-Schmidt map (a1, a2) in R^6 -> V in SO(3), columns (b1, b2, b1 x b2).

    The 6-d representation of Zhou et al. (CVPR 2019): it has a continuous
    right inverse (V -> its first two columns), so a smooth network can output
    V(x) = R_x. A normalised 4-vector cannot: a continuous lift SO(3) -> S^3 of
    the identity does not exist (pi_1(SO(3)) = Z/2), so a quaternion head must
    vanish or jump somewhere on SO(3). Orthonormal up to rounding wherever
    a1 != 0 and a2 is not parallel to a1.
    """
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = a1 / torch.linalg.norm(a1, dim=-1, keepdim=True).clamp_min(1e-12)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / torch.linalg.norm(a2, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def standardize(e):
    """Per-row standardisation (the metric head's InstanceNorm on an (N, h) tensor)."""
    mu = e.mean(-1, keepdim=True)
    return (e - mu) / torch.sqrt(((e - mu) ** 2).mean(-1, keepdim=True) + 1e-5)


def group_lp_norm(d, group, p):
    """sum_g |d_g|_p over consecutive groups of `group` coordinates: a seminorm of d.

    It replaces the metric head's per-group logsumexp, which is a smooth max
    but not subadditive (so it is not a norm). Each group is scaled by its
    max |d_i| so |d_i|^p cannot underflow; the scale is detached because the
    value equals |d_g|_p for every positive scale, so all derivatives are
    exact. s >= 1 whenever the max is above 1e-30 (the max term is exactly 1);
    the clamp only affects d_g ~ 0, where it gives value ~0 and zero gradient
    instead of the NaN of pow(0, 1/p)'.
    """
    x = d.abs().reshape(d.shape[0], -1, group)
    M = x.amax(-1, keepdim=True).detach().clamp_min(1e-30)
    s = ((x / M) ** p).sum(-1).clamp_min(1.0)
    return (M[..., 0] * s ** (1.0 / p)).sum(-1)


class GraphTriNN(PairLatentModel, _Siamese):
    """Triangle inequality by construction.

    One shared encoder gives every pose an embedding e(x) and a rotation
    V(x) = rot6d_to_matrix(W e(x)) in SO(3); the latents depend on the pair
    only through these per-pose quantities:

        a = a_scale |A (e_x - e_y)|^2     ('euclid': sqrt(a) is a Euclidean distance)
          = m(x, y)^2                     ('lp':  m = metric_scale sum_g |n(e_x)_g - n(e_y)_g|_p)
          = both summed                   ('mix': sqrt(a) = |(d_euclid, m)|_2)
        b = |V_x - V_y|_F^2 / 8 = (3 - tr(V_x^T V_y)) / 4

    With d_a = sqrt(a) / s_t a pseudometric and theta = 2 asin(sqrt b) =
    angle(V_x^T V_y) the geodesic distance on SO(3),
    T = |(d_a, theta / s_r)|_2 is a pseudometric for every parameter value:
    T(x, y) = T(y, x), T(x, x) = 0 and T(x, z) <= T(x, y) + T(y, z)
    (componentwise triangle inequalities, then Minkowski in R^2).

    b is computed as |V_x - V_y|^2 / 8 rather than (3 - tr) / 4: identical
    for rotations but free of the cancellation that makes 3 - tr useless in
    float32 for small angles. The range adapter b -> (1 - B_EPS) b maps theta
    to f(theta) = 2 asin(sqrt(1 - B_EPS) sin(theta / 2)), which is increasing
    and concave with f(0) = 0, hence subadditive: the guarantee survives the
    adapter. Away from the learned cut locus (b -> 1) everything is smooth.
    """
    A_KIND = 'euclid'

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, k_a=16, normalize=False, skip_input=False,
                 a_scale=1.0, metric_scale=0.2, group=16, lp_p=8.0, rot_gain=1.0, rot_bias=0.0, rot_mode='free',
                 **_):
        PairLatentModel.__init__(self, s_t, s_r)
        assert rot_mode in ('free', 'equivariant'), rot_mode
        assert rot_mode == 'free' or B.shape[1] == FEATURE_DIM, 'equivariant V needs G(x) = (t, vec R) inputs'
        self.enc = GShapeEncoder(B.shape[1], B, h_size, nl1, normalize, skip_input)
        self.A = Linear(self.enc.out_dim, k_a, bias=False) if self.A_KIND in ('euclid', 'mix') else None
        self.W = Linear(self.enc.out_dim, 6)
        self.a_scale, self.metric_scale, self.group, self.lp_p = a_scale, metric_scale, group, float(lp_p)
        self.rot_mode = rot_mode
        _init(self, self.enc)
        with torch.no_grad():
            # optional near-constant start Q(x) ~ I: the random 6-d output has |a1| ~ 0.5
            # and rotates fast wherever |a1| is small (Gram-Schmidt divides by it)
            self.W.weight.mul_(rot_gain)
            self.W.bias.copy_(rot_bias * torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def rotation(self, e, g):
        """V(x) in SO(3). 'free': V = Q(x) = rot6d(W e(x)). 'equivariant': V = R_x Q(x).

        Maps into SO(3) fall into homotopy classes, and gradient training cannot
        move a free V(x) from a constant / random field into the class of
        x -> R_x without passing through singular poses (|a1| -> 0), where the
        Eikonal loss explodes; in practice the free field collapses to a constant
        (b -> 0) and the a-latent absorbs rotation (EXPERIMENTS.md T2). With
        V = R_x Q(x) the class is fixed by construction while V stays an exact
        rotation for every parameter value. Q = const gives b = b* exactly, so in
        the obstacle-free benchmark this is a structural prior, not a learned b.
        """
        Q = rot6d_to_matrix(self.W(e))
        if self.rot_mode == 'equivariant':
            return g[:, 3:].reshape(-1, 3, 3) @ Q
        return Q

    def metric(self, ex, ey):
        h = self.enc.h_size
        return self.metric_scale * group_lp_norm(standardize(ex[:, :h]) - standardize(ey[:, :h]),
                                                 self.group, self.lp_p)

    def predict_raw(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        a = torch.zeros_like(ex[:, 0])
        if self.A is not None:
            a = a + self.a_scale * (self.A(ex - ey) ** 2).sum(-1)
        if self.A_KIND in ('lp', 'mix'):
            a = a + self.metric(ex, ey) ** 2
        b = ((self.rotation(ex, gx) - self.rotation(ey, gy)) ** 2).sum((-1, -2)) / 8.0
        return torch.stack([a, b], -1)

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        # b <= 1 holds for exact rotations; the clamp only removes rounding above 1
        return raw[:, 0], (1.0 - B_EPS) * raw[:, 1].clamp(0.0, 1.0)


class GraphTriMetricNN(GraphTriNN):
    A_KIND = 'lp'


class GraphTriMixNN(GraphTriNN):
    A_KIND = 'mix'


def gram_to_matrix(P):
    """Linear map with qq^T -> |q|^2 R(q) for quaternions q = (w, x, y, z), applied to any 4x4 P.

    Every entry of R(q) is a quadratic form in q, i.e. linear in qq^T, so on
    rank-1 Gram matrices this is exactly the quaternion-to-rotation map.
    """
    p = lambda i, j: P[..., i, j]
    return torch.stack([
        torch.stack([p(0, 0) + p(1, 1) - p(2, 2) - p(3, 3), 2 * (p(1, 2) - p(0, 3)), 2 * (p(1, 3) + p(0, 2))], -1),
        torch.stack([2 * (p(1, 2) + p(0, 3)), p(0, 0) - p(1, 1) + p(2, 2) - p(3, 3), 2 * (p(2, 3) - p(0, 1))], -1),
        torch.stack([2 * (p(1, 3) - p(0, 2)), 2 * (p(2, 3) + p(0, 1)), p(0, 0) - p(1, 1) - p(2, 2) + p(3, 3)], -1),
    ], -2)


class GraphGramTriNN(GraphGramNN):
    """graph_gram, then the exact triangle head on the same parameters.

    Relaxed phase (exact = False): graph_gram's b = 1 - <P_x, P_y> on unit
    Frobenius-norm PSD Gram matrices -- a contractible representation space,
    so training can deform it freely (no homotopy-class obstruction), but b is
    not a metric unless P is rank 1. Exact phase: V(x) = rot6d(columns of
    gram_to_matrix(P(x))) in SO(3) and b = |V_x - V_y|^2 / 8 as in GraphTriNN,
    so T is a pseudometric for every parameter value. On rank-1 P both phases
    give the same b. The training Function switches at step `tri_switch`;
    `exact` defaults to True, so loaded checkpoints are always the exact model.
    """

    def __init__(self, B, tri_switch=0, **kw):
        super().__init__(B, **kw)
        self.tri_switch = int(tri_switch)
        self.exact = True

    def rotation(self, e):
        M = gram_to_matrix(self._gram(e))
        return rot6d_to_matrix(torch.cat([M[..., 0], M[..., 1]], -1))

    def predict_raw(self, gx, gy):
        if not self.exact:
            return super().predict_raw(gx, gy)
        ex, ey = self.embed_pair(gx, gy)
        a = self.a_scale * (self.A(ex - ey) ** 2).sum(-1)
        b = ((self.rotation(ex) - self.rotation(ey)) ** 2).sum((-1, -2)) / 8.0
        return torch.stack([a, b], -1)

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        return raw[:, 0], (1.0 - B_EPS) * raw[:, 1].clamp(0.0, 1.0)


class GraphSqDistNN(PairLatentModel, _Siamese):
    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, k_a=16, k_b=16,
                 normalize=False, skip_input=False, a_scale=1.0, b_soft=0.02, **_):
        PairLatentModel.__init__(self, s_t, s_r)
        self.enc = GShapeEncoder(B.shape[1], B, h_size, nl1, normalize, skip_input)
        self.A = Linear(self.enc.out_dim, k_a, bias=False)
        self.Bp = Linear(self.enc.out_dim, k_b, bias=False)
        self.a_scale = a_scale
        self.b_soft = b_soft
        _init(self, self.enc)

    def predict_raw(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        d = ex - ey
        return torch.stack([self.a_scale * (self.A(d) ** 2).sum(-1), (self.Bp(d) ** 2).sum(-1)], -1)

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        z, dl = raw[:, 1], self.b_soft
        # smooth min(z, 1), shifted so that b(0) = 0 exactly
        b = z - dl * torch.nn.functional.softplus((z - 1.0) / dl) + dl * math.log1p(math.exp(-1.0 / dl))
        return raw[:, 0], (1.0 - B_EPS) * b.clamp(0.0, 1.0)


class GraphJointNN(PairLatentModel):
    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=True, **_):
        PairLatentModel.__init__(self, s_t, s_r)
        self.enc = GShapeEncoder(2 * FEATURE_DIM, B, h_size, nl1, normalize)
        self.head = Linear(self.enc.out_dim, 2)
        _init(self, self.enc)

    def predict_raw(self, gx, gy):
        n = gx.shape[0]
        e = self.enc(torch.cat([torch.cat([gx, gy], -1), torch.cat([gy, gx], -1)], 0))
        r = self.head(e)
        return 0.5 * (r[:n] + r[n:])

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        return torch.nn.functional.softplus(raw[:, 0]), (1.0 - B_EPS) * torch.sigmoid(raw[:, 1])


class MetricNN(PairValueModel, _Siamese):
    """G-Shape-NTFields metric head (models/metric/model_network_metric.py)."""

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, metric_scale=0.2,
                 group=16, skip_input=False, **_):
        PairValueModel.__init__(self, s_t, s_r)
        self.enc = GShapeEncoder(B.shape[1], B, h_size, nl1, True, skip_input)
        self.metric_scale = metric_scale
        self.group = group
        _init(self, self.enc)

    @staticmethod
    def _group_max(d, group, beta=10.0):
        # logsumexp soft max with offset log(group) / beta: a triangle violation of up to
        # 2 log(group) / beta per group is possible (it is not subadditive); beta = 10 is G-Shape's
        x = torch.sqrt(d ** 2 + 1e-6).view(d.shape[0], -1, group)
        return ((torch.logsumexp(beta * x, 2) - np.log(group)) / beta).sum(1)

    def value(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        h = self.enc.h_size
        T = self._group_max(ex[:, :h] - ey[:, :h], self.group)
        if self.enc.skip_input:  # raw pose features form one extra metric group
            T = T + self._group_max(ex[:, h:] - ey[:, h:], ex.shape[1] - h)
        return self.metric_scale * T


class ScalarNN(PairValueModel):
    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, normalize=True, **_):
        PairValueModel.__init__(self, s_t, s_r)
        self.enc = GShapeEncoder(2 * FEATURE_DIM, B, h_size, nl1, normalize)
        self.head = Linear(self.enc.out_dim, 1)
        _init(self, self.enc)

    def value(self, gx, gy):
        n = gx.shape[0]
        e = self.enc(torch.cat([torch.cat([gx, gy], -1), torch.cat([gy, gx], -1)], 0))
        r = self.head(e)[:, 0]
        return torch.nn.functional.softplus(0.5 * (r[:n] + r[n:]))


class AnalyticGraph(PairLatentModel):
    """Known smooth latent H with the exact decoder (Stage A, no learning).

    a = |t_x - t_y|^2, b = (3 - <vec R_x, vec R_y>) / 4.  Feature inputs off
    SO(3) (tangent probes) can push b marginally outside [0, 1]; those tiny
    excursions are clamped and counted in `n_clamped`.
    """

    def __init__(self, s_t=1.0, s_r=1.0, **_):
        super().__init__(s_t, s_r)
        self.n_clamped = 0

    def predict_raw(self, gx, gy):
        a = ((gx[:, :3] - gy[:, :3]) ** 2).sum(-1)
        b = (3.0 - (gx[:, 3:] * gy[:, 3:]).sum(-1)) / 4.0
        return torch.stack([a, b], -1)

    def predict_latent(self, gx, gy):
        raw = self.predict_raw(gx, gy)
        b = raw[:, 1]
        self.n_clamped += int(((b < 0) | (b > 1)).sum())
        return raw[:, 0], b.clamp(0.0, 1.0)


ARCHITECTURES = {
    'graph_gram': (GraphGramNN, FEATURE_DIM),
    'graph_metric': (GraphMetricNN, FEATURE_DIM),
    'graph_tri': (GraphTriNN, FEATURE_DIM),
    'graph_tri_metric': (GraphTriMetricNN, FEATURE_DIM),
    'graph_tri_mix': (GraphTriMixNN, FEATURE_DIM),
    'graph_gram_tri': (GraphGramTriNN, FEATURE_DIM),
    'graph_sqdist': (GraphSqDistNN, FEATURE_DIM),
    'graph_joint': (GraphJointNN, 2 * FEATURE_DIM),
    'metric': (MetricNN, FEATURE_DIM),
    'scalar': (ScalarNN, 2 * FEATURE_DIM),
}


def build_network(arch, s_t=1.0, s_r=1.0, h_size=256, fourier_sigma=0.5, fourier_seed=0,
                  fourier_sigma_t=None, fourier_sigma_r=None, feature_dim=FEATURE_DIM, fourier_trunc='clamp', **kw):
    if arch == 'analytic':
        return AnalyticGraph(s_t=s_t, s_r=s_r)
    cls, in_dim = ARCHITECTURES[arch]
    in_dim = in_dim // FEATURE_DIM * feature_dim
    B = make_fourier_matrix(h_size // 2, in_dim, fourier_sigma, fourier_seed, fourier_sigma_t, fourier_sigma_r,
                            fourier_trunc)
    return cls(B, s_t=s_t, s_r=s_r, h_size=h_size, **kw)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
