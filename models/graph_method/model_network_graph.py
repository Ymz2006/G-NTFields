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
                 metric_scale=0.2, group=16, **kw):
        kw.pop('normalize', None)  # the metric branch normalises its own embedding
        super().__init__(B, s_t=s_t, s_r=s_r, h_size=h_size, nl1=nl1, gram_k=gram_k, normalize=False,
                         skip_input=skip_input, **kw)
        self.metric_scale, self.group = metric_scale, group

    def predict_raw(self, gx, gy):
        ex, ey = self.embed_pair(gx, gy)
        h = self.enc.h_size

        def norm(e):
            mu = e.mean(-1, keepdim=True)
            return (e - mu) / torch.sqrt(((e - mu) ** 2).mean(-1, keepdim=True) + 1e-5)
        m = self.metric_scale * (MetricNN._group_max(norm(ex[:, :h]) - norm(ey[:, :h]), self.group)
                                 - (h // self.group) * 1e-3)
        Px, Py = self._gram(ex), self._gram(ey)
        b = 0.5 * ((Px - Py) ** 2).sum((-1, -2))
        return torch.stack([m.clamp_min(0) ** 2, b], -1)


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
    def _group_max(d, group):
        x = torch.sqrt(d ** 2 + 1e-6).view(d.shape[0], -1, group)
        return ((torch.logsumexp(10 * x, 2) - np.log(group)) / 10).sum(1)

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
