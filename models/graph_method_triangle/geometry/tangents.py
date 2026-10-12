"""Scaled endpoint tangent derivatives (spec page 2).

    E_1..E_3 : body translations scaled by s_t
    E_4..E_6 : body rotations scaled by s_r
    c_x[i] = d/de T(x exp(e E_i), y) |_{e=0}
    c_y[i] = d/de T(x, y exp(e E_i)) |_{e=0}

Two equivalent implementations are provided:

* ``tangent_gradients`` (training path): one autograd call w.r.t. the fixed
  embedding G = (t, vec R) of both endpoints, followed by the analytic tangent
  projection  c_t = s_t R^T dT/dt,  c_w = s_r vee(A - A^T),  A = R^T dT/dR.
* ``tangent_gradients_perturb`` (verification path): differentiates through
  x exp(hat(s * xi)) at xi = 0 using the series-safe SE(3) exponential.

The projection only uses the physical tangent directions, never redundant
matrix entries as independent motions.
"""

import torch

from . import se3


def project_feature_gradient(dT_dg, R, s_t, s_r):
    """(B, 12) feature gradient at pose with rotation R -> (B, 6) scaled tangent."""
    g_t = dT_dg[..., :3]
    G_R = dT_dg[..., 3:].reshape(*dT_dg.shape[:-1], 3, 3)
    Rt = R.transpose(-1, -2)
    c_t = s_t * (Rt @ g_t.unsqueeze(-1)).squeeze(-1)
    A = Rt @ G_R
    c_w = s_r * se3.vee(A - A.transpose(-1, -2))
    return torch.cat([c_t, c_w], -1)


def tangent_gradients(value_fn, gx, gy, s_t, s_r, create_graph=True):
    """Return (T, c_x, c_y) for value_fn(gx, gy) -> (B,).

    gx, gy are fixed embeddings (B, 12). They are detached and re-leafed here,
    so the caller's tensors are never modified.
    """
    gx = gx.detach().requires_grad_(True)
    gy = gy.detach().requires_grad_(True)
    T = value_fn(gx, gy)
    dgx, dgy = torch.autograd.grad(T.sum(), (gx, gy), create_graph=create_graph)
    Rx = gx.detach()[..., 3:].reshape(-1, 3, 3)
    Ry = gy.detach()[..., 3:].reshape(-1, 3, 3)
    c_x = project_feature_gradient(dgx, Rx, s_t, s_r)
    c_y = project_feature_gradient(dgy, Ry, s_t, s_r)
    return T, c_x, c_y


def tangent_gradients_perturb(value_fn, tx, Rx, ty, Ry, s_t, s_r, create_graph=False):
    """Reference implementation via zero local twists through se3_exp."""
    B = tx.shape[0]
    scale = torch.tensor([s_t] * 3 + [s_r] * 3, dtype=tx.dtype, device=tx.device)
    xi_x = torch.zeros(B, 6, dtype=tx.dtype, device=tx.device, requires_grad=True)
    xi_y = torch.zeros(B, 6, dtype=tx.dtype, device=tx.device, requires_grad=True)
    tx2, Rx2 = se3.right_perturb(tx, Rx, xi_x * scale)
    ty2, Ry2 = se3.right_perturb(ty, Ry, xi_y * scale)
    T = value_fn(se3.features(tx2, Rx2), se3.features(ty2, Ry2))
    c_x, c_y = torch.autograd.grad(T.sum(), (xi_x, xi_y), create_graph=create_graph)
    return T, c_x, c_y
