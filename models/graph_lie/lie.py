"""Endpoint derivatives of tau(x, y) on SE(3) x SE(3): chart (partial) gradient vs Lie gradient.

Poses come in the chart of the datasets, x = (t, r) with r = rotvec / 2 pi (turns). The graph
network maps them to the fixed embedding G(x) = (t, vec R), R = exp(2 pi [r]_x), so tau is a
function on SE(3) x SE(3) and both derivatives below exist.

chart   g = d tau / d(t, r)
        The partial derivatives of the original method. The Eikonal equation |g| = 1 / S then
        measures speed in the flat metric |dt|^2 + |dr|^2 of the rotation-vector ball. That is
        not a metric on SO(3): it depends on where the chart origin is, and the two rotation
        vectors of one pose near |rotvec| = pi see different metrics.

lie     c_i = d/de tau(x exp(e E_i)),  E_1..3 body translations (metres),
                                        E_4..6 body rotations by one turn (2 pi rad)
        One autograd call w.r.t. G followed by the exact tangent projection
            c_v = R^T d tau / d t,   c_w = 2 pi vee(A - A^T),   A = R^T d tau / d R
        (models/graph_method_triangle/geometry/tangents.py, s_t = 1, s_r = 2 pi).
        |c|^2 = |c_v|^2 + |c_w|^2 is the left-invariant product metric |v|^2 + |w / 2 pi|^2,
        identical at every pose; no chart is involved, so there is no seam.

Relation (chain rule through R = exp(2 pi [r]_x), right Jacobian Jr):
        c_v = R^T g_v,   c_w = Jr^{-1}(2 pi r)^T g_w.
So |c_v| = |g_v|, and only the rotation block changes: in the plane orthogonal to r the factor
is (theta / 2) / sin(theta / 2), theta = 2 pi |r|, from 1 at the identity to pi / 2 at a half
turn. Along r it is 1.
"""
import math

import torch

from models.graph_native import _sib

TWO_PI = 2.0 * math.pi
GRAD_MODES = ('chart', 'lie')


def endpoint_gradients(network, coords, mode, create_graph=True):
    """Return tau (N,) and the start / goal endpoint gradients gx, gy (N, 6) in `mode`.

    network: a models/graph_native NN (features, net.value, out). coords: (N, 12) chart rows
    (t_x, r_x, t_y, r_y). Gradients are ordered (translation, rotation) in both modes.
    """
    if mode == 'chart':
        tau, _, Xp = network.out(coords)
        d = torch.autograd.grad(tau, Xp, torch.ones_like(tau), only_inputs=True, retain_graph=True,
                                create_graph=create_graph)[0]
        return tau[:, 0], d[:, :6], d[:, 6:]
    if mode == 'lie':
        g0 = network.features(coords[:, :6]).detach()
        g1 = network.features(coords[:, 6:]).detach()
        return _sib.tangents.tangent_gradients(network.net.value, g0, g1, 1.0, TWO_PI, create_graph=create_graph)
    raise ValueError('grad mode must be one of {}, got {!r}'.format(GRAD_MODES, mode))


def lie_gradients_perturb(network, coords, create_graph=False):
    """Reference Lie gradient: autograd through x exp(xi) at xi = 0 (series-safe SE(3) exponential)."""
    se3 = _sib.se3
    tx, Rx = coords[:, :3], se3.so3_exp(TWO_PI * coords[:, 3:6])
    ty, Ry = coords[:, 6:9], se3.so3_exp(TWO_PI * coords[:, 9:12])
    return _sib.tangents.tangent_gradients_perturb(network.net.value, tx, Rx, ty, Ry, 1.0, TWO_PI,
                                                   create_graph=create_graph)


def so3_right_jacobian_inv(phi):
    """Jr^{-1}(phi) for rotation vectors phi (..., 3) in radians:

        log(exp(phi) exp(d)) = phi + Jr^{-1}(phi) d + O(|d|^2),
        Jr^{-1} = I + 1/2 [phi]_x + (1 / theta^2 - (1 + cos theta) / (2 theta sin theta)) [phi]_x^2.

    Series below theta = 1e-2. Singular at theta = pi (the chart seam): verification and
    diagnostics only; training uses the tangent projection, which needs no chart.
    """
    theta = phi.norm(dim=-1)
    small = theta < 1e-2
    th = torch.where(small, torch.ones_like(theta), theta)
    c = 1.0 / th ** 2 - (1.0 + torch.cos(th)) / (2.0 * th * torch.sin(th))
    c = torch.where(small, 1.0 / 12.0 + theta ** 2 / 720.0, c)
    K = _sib.se3.hat(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(K)
    return eye + 0.5 * K + c[..., None, None] * (K @ K)


def chart_to_lie(r, g):
    """Chart endpoint gradient g = (g_v, g_w) (..., 6) at r (turns) -> Lie gradient (c_v, c_w)."""
    R = _sib.se3.so3_exp(TWO_PI * r)
    c_v = (R.transpose(-1, -2) @ g[..., :3, None])[..., 0]
    c_w = (so3_right_jacobian_inv(TWO_PI * r).transpose(-1, -2) @ g[..., 3:, None])[..., 0]
    return torch.cat([c_v, c_w], -1)


def perpendicular_factor(r):
    """(theta / 2) / sin(theta / 2), theta = 2 pi |r|: |c_w| / |g_w| for g_w orthogonal to r."""
    half = math.pi * r.norm(dim=-1)
    return torch.where(half > 1e-6, half / torch.sin(half.clamp_min(1e-6)), torch.ones_like(half))
