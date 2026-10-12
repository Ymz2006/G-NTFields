"""Explicit latent-to-time decoder D(a, b) for the ellipsoidal speed model.

    phi(b) = 4 asin(sqrt(b))^2            (= theta^2, theta the relative angle)
    Q      = a / s_t^2 + phi(b) / s_r^2   (squared time)
    D(a,b) = sqrt(Q)

Small b uses the series 4b + 4/3 b^2 + 32/45 b^3 + 16/35 b^4 + 512/1575 b^5,
which removes the removable 0/0 singularity of phi'(b) at b = 0.  The b = 1
singularity of phi' is genuine (cut locus) and is *not* removed.
"""

import torch

PHI_SERIES_MAX = 1e-3


def phi(b):
    small = b < PHI_SERIES_MAX
    b_safe = torch.where(small, torch.full_like(b, 0.5), b)
    exact = 4.0 * torch.asin(torch.sqrt(b_safe)) ** 2
    series = b * (4.0 + b * (4.0 / 3.0 + b * (32.0 / 45.0 + b * (16.0 / 35.0 + b * 512.0 / 1575.0))))
    return torch.where(small, series, exact)


def phi_prime(b):
    """Analytic derivative, for verification only (singular at b = 1)."""
    small = b < PHI_SERIES_MAX
    b_safe = torch.where(small, torch.full_like(b, 0.5), b)
    exact = 4.0 * torch.asin(torch.sqrt(b_safe)) / torch.sqrt(b_safe * (1.0 - b_safe))
    series = 4.0 + b * (8.0 / 3.0 + b * (32.0 / 15.0 + b * (64.0 / 35.0 + b * 512.0 / 315.0)))
    return torch.where(small, series, exact)


def theta_from_b(b):
    """Forward-only angle 2 atan2(sqrt b, sqrt(1-b)); does not cure phi' at b=1."""
    return 2.0 * torch.atan2(torch.sqrt(b), torch.sqrt(1.0 - b))


def decode_Q(a, b, s_t, s_r):
    return a / (s_t * s_t) + phi(b) / (s_r * s_r)


def decode(a, b, s_t, s_r):
    return torch.sqrt(decode_Q(a, b, s_t, s_r))
