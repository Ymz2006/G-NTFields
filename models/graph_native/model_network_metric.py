"""Graph-head network with the original ``NN.out(coords) -> (tau, w, coords)`` contract.

``coords`` are (N, 12) rows [t_x, rotvec_x / 2 pi, t_y, rotvec_y / 2 pi]. They are mapped
(differentiably) to the fixed embeddings G(x), G(y) = (t, vec R) and fed to a siamese graph
head from models/graph_method_triangle/model_network_graph.py with the analytic decoder.
tau is a function on SE(3) x SE(3): it is periodic in the rotation vector, so it has no
seam at |rotvec| = pi (the original chart network does, see EXPERIMENTS.md C1).

The constructor keeps the original signature ``NN(device, dim, B)``; ``B`` (the original
6-d Fourier matrix) is ignored, the head builds its own 12-d one from ``fourier_seed`` /
``fourier_sigma_t`` / ``fourier_sigma_r`` (stage-E values of graph_method: 0.5 / 0.1,
resampled truncated normal). All head options are passed through ``graph`` (a dict) and
stored in the checkpoint, so ``Model.load`` can rebuild the network.
"""
import math

import torch

from . import _sib

DEFAULT_GRAPH = dict(arch='graph_tri_metric', rot_mode='equivariant', rot_bias=1.0, rot_gain=1.0,
                     skip_input=True, fourier_sigma=0.3, fourier_sigma_t=0.5, fourier_sigma_r=0.1,
                     fourier_trunc='resample', fourier_seed=0, h_size=256, nl1=2, metric_scale=0.2, group=16,
                     lp_p=8.0, k_a=16, gram_k=4, lse_beta=10.0, a_scale=1.0)


class NN(torch.nn.Module):
    def __init__(self, device, dim, B, graph=None):
        super().__init__()
        assert dim == 6, 'SE(3) only'
        self.dim = dim
        g = dict(DEFAULT_GRAPH)
        g.update(graph or {})
        self.graph = g
        kw = {k: v for k, v in g.items() if k not in ('arch',)}
        # s_r = 2 pi: one unit of rotation = one turn, as the chart coordinates rotvec / 2 pi
        self.net = _sib.nets.build_network(g['arch'], s_t=1.0, s_r=2.0 * math.pi, feature_dim=12, **kw)
        self.net.to(device)

    def features(self, x):
        """(N, 6) chart -> (N, 12) G = (t, vec R), differentiable through the SO(3) exponential."""
        return _sib.se3.features(x[:, :3], _sib.se3.so3_exp(2.0 * math.pi * x[:, 3:6]))

    def value_features(self, g0, g1):
        return self.net.value(g0, g1)

    def out(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        g0 = self.features(coords[:, :self.dim])
        g1 = self.features(coords[:, self.dim:])
        tau = self.net.value(g0, g1)[:, None]
        return tau, None, coords
