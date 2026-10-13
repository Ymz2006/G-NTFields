"""Speed-conditioned Fourier embedding for the graph heads.

Motivation. With the fixed random Fourier matrix (sigma_t = 0.5, sigma_r = 0.1) and the Lipschitz
row-normalised encoder, |grad tau| at a pose is bounded by (2 pi sigma |B|) x (layer Lipschitz
constants) x metric_scale: a few tens at most. The isotropic Eikonal equation asks for
|grad tau| = 1 / S_d, up to 1000 where the warped clearance speed is 0.001. The trained fields
therefore predict speeds ~0.05 where the truth is 0.002-0.01 and, as the global scale compensates,
~0.8 where the truth is 0.9-1 (results/graph_lie/iso_report/speed_calibration.json).

Architecture. A small, separately trained **speed predictor** s_hat(x) (G(x) -> S_d, log-space
regression on the dataset's own clearance speeds, `SpeedPredictor`) is part of the network and
controls the Fourier frequencies of the pose embedding *per pose*:

    scale        gamma(x) = [sin(2 pi f(s_hat(x)) B x), cos(...)],  f(s) = s^(-alpha) capped at f_max
                 (every frequency is multiplied; the chain-rule term (B x) grad f is part of the field)
    bands        the n_freq rows of B are split into K octave bands (sigma, 2 sigma, ..., 2^(K-1) sigma);
                 band k is gated by w_k(s_hat) = softmax_k(-(k* - k)^2 / 2 tau^2), k* = alpha log2(1 / s_hat)
                 clipped to [0, K - 1]: low speed selects the high-frequency bands
    fixed_bands  the same K bands with w_k = 1 (no speed predictor; multi-scale control)
    none         the original single-sigma embedding (so the class reproduces GraphMetricNN)

The predictor's parameters are frozen by default (`sf_train_pred=False`); gradients with respect
to the pose always flow through it, so the Eikonal residual and the planner see the same field.
Heads: `graph_metric_sf` (GraphMetricNN with this encoder), `graph_tri_metric_sf`.
"""
import math

import torch
from torch.nn import Linear

from . import model_network_graph as mng

SF_MODES = ('none', 'scale', 'bands', 'fixed_bands')
LOG_S_MIN = math.log(1e-3)  # warped clearance speeds are clamped at 1e-3 by the trainers


class SpeedPredictor(torch.nn.Module):
    """G(x) in R^12 -> warped translational clearance speed S_d(x) in [1e-3, 1] (smooth, twice differentiable).

    Fourier features of G(x) (own matrix), SiLU MLP, output log S_d = LOG_S_MIN * sigmoid(z).
    Trained by models/graph_lie/scripts/train_speed_predictor.py (MSE in log space).
    """

    def __init__(self, n_freq=128, sigma_t=1.0, sigma_r=0.3, width=256, depth=3, seed=1234):
        super().__init__()
        B = mng.make_fourier_matrix(n_freq, mng.FEATURE_DIM, 1.0, seed, sigma_t, sigma_r, 'resample')
        self.register_buffer('B', B.t().contiguous())
        layers, d = [], 2 * n_freq
        for _ in range(depth):
            layers += [Linear(d, width), torch.nn.SiLU()]
            d = width
        layers.append(Linear(d, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def log_speed(self, g):
        p = g @ (2.0 * math.pi * self.B)
        z = self.mlp(torch.cat([torch.sin(p), torch.cos(p)], -1))[:, 0]
        return LOG_S_MIN * torch.sigmoid(z)

    def forward(self, g):
        return torch.exp(self.log_speed(g))


class SpeedFourierEncoder(mng.GShapeEncoder):
    """GShapeEncoder whose input mapping is conditioned on the predicted speed (see module doc)."""

    def __init__(self, in_dim, B, h_size=256, nl1=2, normalize=True, skip_input=False, lip_cap=1.0, sf_mode='scale',
                 sf_alpha=0.5, sf_fmax=16.0, sf_bands=4, sf_tau=0.75, sf_cols='all', sf_detach=False,
                 sf_pred=None, sf_train_pred=False, sf_pred_kw=None, sf_psi='none', sf_psi_rho=0.2,
                 sf_psi_margin=0.05, sf_pred_kind='mlp', sf_psi_in=0.0):
        assert sf_mode in SF_MODES, sf_mode
        assert in_dim == mng.FEATURE_DIM, 'the speed predictor reads G(x) = (t, vec R)'
        super().__init__(in_dim, B, h_size, nl1, normalize, skip_input, lip_cap)
        self.sf_mode, self.sf_alpha, self.sf_fmax = sf_mode, float(sf_alpha), float(sf_fmax)
        self.sf_bands, self.sf_tau, self.sf_cols, self.sf_detach = int(sf_bands), float(sf_tau), sf_cols, sf_detach
        n_freq = B.shape[0]
        if sf_mode in ('bands', 'fixed_bands'):
            assert n_freq % self.sf_bands == 0
            # band k multiplies its rows of B by 2^k; the column selection (t only / all) applies too
            mult = torch.ones(n_freq, in_dim)
            per = n_freq // self.sf_bands
            for k in range(self.sf_bands):
                mult[k * per:(k + 1) * per, self._cols(in_dim)] = 2.0 ** k
            self.register_buffer('B', (B * mult).t().contiguous())
            self.register_buffer('band_index', torch.arange(self.sf_bands, dtype=torch.float32))
        self.sf_psi_in, self.sf_psi_margin = float(sf_psi_in), float(sf_psi_margin)
        if self.sf_psi_in > 0:  # escape time Psi(x) as a 13th Fourier input with its own std
            g = torch.Generator().manual_seed(int(1e6) + 7)
            self.register_buffer('B_psi', torch.randn(n_freq, generator=g).clamp(-2.0, 2.0) * self.sf_psi_in)
        self.speed_pred = None
        if sf_mode in ('scale', 'bands') or sf_psi != 'none' or self.sf_psi_in > 0:
            if sf_pred is None:
                if sf_pred_kind == 'clearance':  # structured: env sdf grid + shape points (clearance.py)
                    from .clearance import ClearancePredictor
                    sf_pred = ClearancePredictor(**(sf_pred_kw or {}))
                else:
                    sf_pred = SpeedPredictor(**(sf_pred_kw or {}))
            self.speed_pred = sf_pred
            for p in self.speed_pred.parameters():
                p.requires_grad_(bool(sf_train_pred))

    def _cols(self, in_dim):
        return slice(0, 3) if self.sf_cols == 't' else slice(0, in_dim)

    def predicted_speed(self, g):
        s = self.speed_pred(g)
        return s.detach() if self.sf_detach else s

    def frequency_factor(self, s):
        """scale mode: f(s) = s^(-alpha), capped at f_max (hard cap, zero slope beyond)."""
        s_min = self.sf_fmax ** (-1.0 / self.sf_alpha)
        return s.clamp_min(s_min) ** (-self.sf_alpha)

    def band_gates(self, s):
        """bands mode: (N, K) gates, soft one-hot on k* = alpha log2(1 / s) clipped to [0, K - 1]."""
        k_star = (self.sf_alpha * torch.log2(1.0 / s)).clamp(0.0, self.sf_bands - 1.0)
        logits = -((k_star[:, None] - self.band_index[None, :]) ** 2) / (2.0 * self.sf_tau ** 2)
        return torch.softmax(logits, -1)

    def escape_time(self, g):
        pred = self.speed_pred
        if hasattr(pred, 'raw_speed'):
            return escape_time_raw(pred.raw_speed(g), self.sf_psi_margin)
        return escape_time(pred(g), self.sf_psi_margin)

    def input_mapping(self, x):
        if self.sf_mode == 'none' or self.sf_mode == 'fixed_bands':
            x_proj = x @ (2.0 * math.pi * self.B)
            if self.sf_psi_in > 0:
                x_proj = x_proj + 2.0 * math.pi * self.escape_time(x)[:, None] * self.B_psi[None, :]
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
        s = self.predicted_speed(x)
        if self.sf_mode == 'scale':
            f = self.frequency_factor(s)[:, None]
            if self.sf_cols == 't':
                x = torch.cat([x[:, :3] * f, x[:, 3:]], -1)
                x_proj = x @ (2.0 * math.pi * self.B)
            else:
                x_proj = f * (x @ (2.0 * math.pi * self.B))
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
        # bands
        x_proj = x @ (2.0 * math.pi * self.B)
        w = self.band_gates(s)                                   # (N, K)
        n_freq = x_proj.shape[1]
        gate = w.repeat_interleave(n_freq // self.sf_bands, dim=1)  # (N, n_freq), band-major rows of B
        feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)
        return feats * torch.cat([gate, gate], -1)


def _sf_extra(kw):
    """The sf_* options of a head config that the encoder takes (speed_pred_ckpt is consumed by the trainer)."""
    enc_keys = ('sf_mode', 'sf_alpha', 'sf_fmax', 'sf_bands', 'sf_tau', 'sf_cols', 'sf_detach', 'sf_train_pred',
                'sf_pred_kw', 'sf_psi', 'sf_pred_kind', 'sf_psi_in', 'sf_psi_margin')
    return {k: v for k, v in kw.items() if k in enc_keys}


def escape_time_raw(s, margin=0.05):
    """Psi(s): time to leave the clearance band from raw speed s = d / margin at the warped speed
    S = s^2 (2 - s)^2:  Psi = int_d^margin dd' / S(d') = (margin / 4) [1/s - 1/(2 - s) + ln((2 - s) / s)]
    (exact antiderivative), so d Psi / dd = -1 / S exactly: an endpoint term with this gradient satisfies
    the isotropic Eikonal equation inside the band by construction. Psi(1) = 0."""
    s = s.clamp(1e-4, 1.0)
    return 0.25 * margin * (1.0 / s - 1.0 / (2.0 - s) + torch.log((2.0 - s) / s))


def raw_from_warped(s_hat):
    """Inverse of the warp S = s^2 (2 - s)^2 on [0, 1] (clamped away from the sqrt singularity at S = 1)."""
    return 1.0 - torch.sqrt((1.0 - torch.sqrt(s_hat.clamp(1e-6, 1.0))).clamp_min(1e-6))


def escape_time(s_hat, margin=0.05):
    """Psi as a function of the warped speed s_hat (MLP predictors); see escape_time_raw."""
    return escape_time_raw(raw_from_warped(s_hat), margin)


class _PsiMixin:
    """Escape-time potential on top of a pair head (sf_psi != 'none'):

        T(x, y) = D(a, b) + (psi_x + psi_y) (1 - kappa(x, y)),   kappa = exp(-a / rho^2),
        psi(x)  = gain * Psi(s_hat(x))   [+ softplus(w e_x + b) when sf_psi = 'learned'].

    T(x, x) = 0 and T is symmetric; for endpoints in different clearance bands (kappa ~ 0) the
    gradient at x is grad D + grad psi_x, and grad Psi = -1 / S by construction, whatever the Fourier
    frequencies of the encoder. (Not a metric: graph_metric's logsumexp distance is not one either.)
    """

    def _init_psi(self, kw):
        self.sf_psi = kw.get('sf_psi', 'none')
        assert self.sf_psi in ('none', 'fixed', 'learned', 'coord'), self.sf_psi
        self.sf_psi_margin = float(kw.get('sf_psi_margin', 0.05))
        if self.sf_psi != 'none':
            self.psi_gain = torch.nn.Parameter(torch.tensor(float(kw.get('sf_psi_gain', 1.0))))
            self.psi_log_rho = torch.nn.Parameter(torch.tensor(math.log(float(kw.get('sf_psi_rho', 0.2)))))
            if self.sf_psi == 'learned':
                self.psi_corr = Linear(self.enc.out_dim, 1)
                with torch.no_grad():
                    self.psi_corr.weight.zero_()
                    self.psi_corr.bias.fill_(-5.0)

    def psi(self, g, e=None):
        pred = self.enc.speed_pred
        if hasattr(pred, 'raw_speed'):  # clearance predictor: raw s = clip(d / margin), no inverse warp needed
            p = self.psi_gain * escape_time_raw(pred.raw_speed(g), self.sf_psi_margin)
        else:
            p = self.psi_gain * escape_time(pred(g), self.sf_psi_margin)
        if self.sf_psi == 'learned' and e is not None:
            p = p + torch.nn.functional.softplus(self.psi_corr(e)[:, 0])
        return p

    def predict_latent(self, gx, gy):
        a, b = super().predict_latent(gx, gy)
        if self.sf_psi == 'coord':
            # Psi as an extra Euclidean coordinate of the latent distance: a = m^2 + (gain (Psi_x - Psi_y))^2.
            # Local pairs in the band get |Psi_x - Psi_y| ~ |x - y|_normal / S (the conformal stretch), far
            # pairs get the escape time; symmetric, zero on the diagonal, still a (pseudo)metric.
            a = a + (self.psi_gain * (self.enc.escape_time(gx) - self.enc.escape_time(gy))) ** 2
        return a, b

    def value(self, gx, gy):
        if self.sf_psi in ('none', 'coord'):
            return super().value(gx, gy)
        ex, ey = self.embed_pair(gx, gy)
        a, b = self.latent_from_embeddings(ex, ey, gx, gy)
        D = self.decode(a, b)
        kappa = torch.exp(-a / torch.exp(2.0 * self.psi_log_rho))
        return D + (self.psi(gx, ex) + self.psi(gy, ey)) * (1.0 - kappa)

    def value_sq(self, gx, gy):
        if self.sf_psi in ('none', 'coord'):
            return super().value_sq(gx, gy)
        return self.value(gx, gy) ** 2


class GraphMetricSFNN(_PsiMixin, mng.GraphMetricNN):
    """graph_metric with the speed-conditioned Fourier encoder (same init stream: sf_mode='none'
    reproduces graph_metric bit for bit) and the optional escape-time potential."""
    ENCODER = SpeedFourierEncoder
    encoder_extra = staticmethod(_sf_extra)

    def __init__(self, B, **kw):
        super().__init__(B, **kw)
        self._init_psi(kw)

    def latent_from_embeddings(self, ex, ey, gx, gy):
        h = self.enc.h_size
        m = self.metric_scale * (mng.MetricNN._group_max(mng.standardize(ex[:, :h]) - mng.standardize(ey[:, :h]),
                                                         self.group, self.lse_beta) - (h // self.group) * 1e-3)
        Px, Py = self._gram(ex), self._gram(ey)
        b = 0.5 * ((Px - Py) ** 2).sum((-1, -2))
        return m.clamp_min(0) ** 2, (1.0 - mng.B_EPS) * b.clamp(max=1.0)


class GraphTriMetricSFNN(mng.GraphTriMetricNN):
    """graph_tri_metric (triangle head) with the speed-conditioned Fourier encoder."""
    ENCODER = SpeedFourierEncoder
    encoder_extra = staticmethod(_sf_extra)


ARCHITECTURES = {
    'graph_metric_sf': (GraphMetricSFNN, mng.FEATURE_DIM),
    'graph_tri_metric_sf': (GraphTriMetricSFNN, mng.FEATURE_DIM),
}


def load_speed_predictor(net, path, device='cpu'):
    """Load a trained SpeedPredictor state dict (train_speed_predictor.py) into net.enc.speed_pred."""
    enc = getattr(net, 'enc', None)
    if enc is None or getattr(enc, 'speed_pred', None) is None:
        return False
    ck = torch.load(path, map_location=device)
    enc.speed_pred.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck, strict=True)
    enc.speed_pred.to(device)
    return True
