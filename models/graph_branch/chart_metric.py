"""The original metric NTFields network (Lie-Shape-NTFields models/metric_baseline) as a pair head of this
repository, so the analysis scripts of models/graph_branch can evaluate it next to the graph heads.

The original network reads the rotation-vector chart c = (t, rotvec / 2 pi) of each endpoint: random Fourier
features of c (B in R^{128 x 6}, truncated N(0, 1)), the G-Shape gated sine encoder, per-row standardisation
(InstanceNorm1d on an (N, 256) tensor) and the logsumexp group distance

    tau(x, y) = 0.2 sum_{g=1}^{16} [logsumexp_{i in g}(10 sqrt((e_i(x) - e_i(y))^2 + 1e-6)) - log 16] / 10.

`ChartMetricNN` is that network (models/graph_method_triangle's MetricNN with a 6-d chart encoder), with
`value(gx, gy)` taking the fixed embeddings G = (t, vec R) like every other head: the chart is recovered with the
canonical logarithm (|rotvec| <= pi). The chart has a seam at |rotvec| = pi, where the field is discontinuous; the
map below is differentiable everywhere else (the identity included), so gradients through it are the chart
network's own. `value_chart` evaluates raw chart rows (any |rotvec|, as the planner feeds them).

Registered as arch 'metric_chart' in build_network's table on import (the shared module is not edited).
`load_original(ckpt)` loads a checkpoint written by the original trainer (keys model_state_dict / B_state_dict)
into a graph_native NN wrapper (`.net.value`, `.graph`), as used by the analysis scripts.
"""
import math

import torch

from models.graph_native import _sib
from models.graph_native import model_network_metric as gn_network
from models.graph_method_triangle import branch_decoder as bd

TWO_PI = 2.0 * math.pi
mng = _sib.nets
_SMALL_V2 = 1e-8


def chart_from_rotation(R):
    """Canonical rotation vector / 2 pi (|rotvec| <= pi) of R (..., 3, 3), differentiable off the seam.

    q = (w, v) from Shepperd's method with w >= 0; rotvec = 2 atan2(|v|, w) v / |v| (series of atan(z) / z in
    z^2 = |v|^2 / w^2 near the identity, so the map is smooth there)."""
    q = bd.quat_from_matrix(R)
    q = q * torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    w, v = q[..., 0], q[..., 1:]
    v2 = (v * v).sum(-1)
    small = v2 < _SMALL_V2
    w_safe = torch.where(small, w, torch.ones_like(w)).clamp_min(1e-12)
    z2 = v2 / (w_safe * w_safe)
    series = (1.0 - z2 / 3.0 + z2 * z2 / 5.0) / w_safe
    vn = torch.sqrt(torch.where(small, torch.ones_like(v2), v2))
    exact = torch.atan2(vn, w) / vn
    f = torch.where(small, series, exact)
    return (2.0 * f)[..., None] * v / TWO_PI


def chart_from_features(g):
    return torch.cat([g[..., :3], chart_from_rotation(g[..., 3:12].reshape(*g.shape[:-1], 3, 3))], -1)


class ChartMetricNN(mng.MetricNN):
    """Original metric head on the 6-d chart (module doc). The 12-d Fourier matrix passed by build_network is
    ignored; B is drawn from `fourier_seed` like the original (128 x 6, truncated N(0, 1), sigma 1)."""

    def __init__(self, B, s_t=1.0, s_r=1.0, h_size=256, nl1=2, metric_scale=0.2, group=16, fourier_seed=0, **_):
        B6 = mng.make_fourier_matrix(h_size // 2, 6, 1.0, fourier_seed, trunc='resample')
        super().__init__(B6, s_t=s_t, s_r=s_r, h_size=h_size, nl1=nl1, metric_scale=metric_scale, group=group,
                         skip_input=False)

    def value_chart(self, cx, cy):
        return mng.MetricNN.value(self, cx, cy)

    def value(self, gx, gy):
        return self.value_chart(chart_from_features(gx), chart_from_features(gy))


mng.ARCHITECTURES.setdefault('metric_chart', (ChartMetricNN, mng.FEATURE_DIM))


def load_original(ckpt, device):
    """graph_native NN wrapper around a checkpoint of the original trainer (Lie-Shape-NTFields metric_baseline)."""
    ck = torch.load(ckpt, map_location='cpu')
    net = gn_network.NN(device, 6, None, graph=dict(arch='metric_chart', fourier_seed=0))
    sd = {}
    for k, v in ck['model_state_dict'].items():
        p = k.split('.')
        if p[0] == 'encoder':
            if p[1] == '0':
                continue                     # Linear(252, 256) of the original is never used in its forward
            sd['enc.encoder.{}.{}'.format(int(p[1]) - 1, p[2])] = v
        elif p[0] in ('gate', 'pe_gate'):
            sd['enc.' + k] = v
    B = ck['B_state_dict']                   # (128, 6); the encoder stores B^T
    sd['enc.B'] = B.t().contiguous().float()
    missing, unexpected = net.net.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    net.graph = dict(net.graph, arch='metric_chart', source='original metric_baseline', ckpt=ckpt)
    return net.to(device).eval()
