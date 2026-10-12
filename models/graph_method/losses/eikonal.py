"""Separate endpoint Eikonal residuals (spec page 4) and regular-sample policy (page 5).

    c_x, c_y      scaled tangent gradients of the decoded time T
    L_E = mean[(|c_x|^2 - 1)^2 + (|c_y|^2 - 1)^2]          form='sq'   (spec)
          mean[(|c_x| - 1)^2 + (|c_y| - 1)^2]              form='abs'
          mean[log(|c_x|^2)^2 + log(|c_y|^2)^2]             form='log'
          mean[(sqrt|c_x| - 1)^2 + (sqrt|c_y| - 1)^2]        form='sqrt' (NTFields)

Regular samples are selected from the *input geometry* (oracle distance and
relative angle of the pair), never from predicted latents, and the Eikonal
batch is built only from regular pairs so no singular derivative is ever
evaluated (spec page 8 correction).
"""

import torch

from ..fields import oracle
from ..geometry import tangents


def regular_mask(tx, Rx, ty, Ry, s_t, s_r, tau_ref, diag_band=1e-3, pi_band=1e-4):
    _, b = oracle.latent_star(tx, Rx, ty, Ry)
    T = oracle.time_star(tx, Rx, ty, Ry, s_t, s_r)
    return (T / tau_ref >= diag_band) & (1.0 - b >= pi_band)


def endpoint_residual(c_x, c_y, form='sq'):
    nx = (c_x * c_x).sum(-1)
    ny = (c_y * c_y).sum(-1)
    if form == 'sq':
        return (nx - 1.0) ** 2 + (ny - 1.0) ** 2
    if form == 'abs':
        return (torch.sqrt(nx + 1e-12) - 1.0) ** 2 + (torch.sqrt(ny + 1e-12) - 1.0) ** 2
    if form == 'sqrt':  # NTFields metric form (sqrt(S |grad T|) - 1)^2 with c = S grad T
        return (nx.clamp_min(1e-12) ** 0.25 - 1.0) ** 2 + (ny.clamp_min(1e-12) ** 0.25 - 1.0) ** 2
    if form == 'log':  # scale-symmetric: penalizes too-flat fields as strongly as too-steep ones
        return torch.log(nx + 1e-12) ** 2 + torch.log(ny + 1e-12) ** 2
    raise ValueError(form)


def eikonal_terms(model, gx, gy, create_graph=True):
    """Returns (T, c_x, c_y) on (already regular) feature batches."""
    return tangents.tangent_gradients(model.value, gx, gy, model.s_t, model.s_r,
                                      create_graph=create_graph)
