"""Deterministic pose-pair samplers and fixed held-out suites (spec page 6).

Training batches are a declared mixture, not a uniform distribution:
    60 %  uniform pairs: t ~ U[-box, box]^3, R Haar (normalized Gaussian quaternion)
    20 %  small relative perturbations y = x exp(r * u), u on the unit 6-sphere
          in the scaled twist frame, r log-uniform in [small_min, small_max]
    20 %  near-half-turn pairs: R_y = R_x exp((pi - delta) n), n random axis,
          delta log-uniform in [pi_delta_min, pi_delta_max], t_y uniform
A separate diagonal batch (x, x) is drawn for the goal loss. Exact half-turns
are only used by the held-out value suites.
"""

import math

import torch

from ..geometry import se3


class PoseSampler:
    def __init__(self, seed=0, device='cpu', dtype=torch.float32, box=1.0,
                 mix=(0.6, 0.2, 0.2), s_t=1.0, s_r=1.0,
                 small_min=1e-3, small_max=0.3,
                 pi_delta_min=1e-3, pi_delta_max=0.5):
        self.device = torch.device(device)
        self.dtype = dtype
        self.box = box
        self.mix = mix
        self.s_t, self.s_r = s_t, s_r
        self.small_min, self.small_max = small_min, small_max
        self.pi_delta_min, self.pi_delta_max = pi_delta_min, pi_delta_max
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    # -- reproducibility -------------------------------------------------
    def state_dict(self):
        return {'gen': self.gen.get_state()}

    def load_state_dict(self, state):
        self.gen.set_state(state['gen'])

    # -- primitives --------------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _log_uniform(self, n, lo, hi):
        u = self._rand(n)
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))

    def unit_vectors(self, n, d):
        v = self._randn(n, d)
        return v / torch.linalg.norm(v, dim=-1, keepdim=True)

    def translations(self, n):
        return (2.0 * self._rand(n, 3) - 1.0) * self.box

    def rotations(self, n):
        return se3.quat_to_matrix(self._randn(n, 4))

    def uniform_poses(self, n):
        return self.translations(n), self.rotations(n)

    # -- pair families -----------------------------------------------------
    def uniform_pairs(self, n):
        tx, Rx = self.uniform_poses(n)
        ty, Ry = self.uniform_poses(n)
        return tx, Rx, ty, Ry

    def small_pairs(self, n, r_min=None, r_max=None):
        tx, Rx = self.uniform_poses(n)
        r = self._log_uniform(n, r_min or self.small_min, r_max or self.small_max)
        u = self.unit_vectors(n, 6)
        scale = torch.tensor([self.s_t] * 3 + [self.s_r] * 3, dtype=self.dtype, device=self.device)
        ty, Ry = se3.right_perturb(tx, Rx, r[:, None] * u * scale)
        return tx, Rx, ty, Ry

    def near_pi_pairs(self, n, d_min=None, d_max=None):
        tx, Rx = self.uniform_poses(n)
        ty = self.translations(n)
        delta = self._log_uniform(n, d_min or self.pi_delta_min, d_max or self.pi_delta_max)
        axis = self.unit_vectors(n, 3)
        Ry = Rx @ se3.so3_exp((math.pi - delta)[:, None] * axis)
        return tx, Rx, ty, Ry

    def exact_pi_pairs(self, n):
        tx, Rx = self.uniform_poses(n)
        ty = self.translations(n)
        axis = self.unit_vectors(n, 3)
        Ry = Rx @ se3.so3_exp(math.pi * axis)
        return tx, Rx, ty, Ry

    def pure_translation_pairs(self, n):
        tx, Rx = self.uniform_poses(n)
        return tx, Rx, self.translations(n), Rx.clone()

    def pure_rotation_pairs(self, n):
        tx, Rx = self.uniform_poses(n)
        return tx, Rx, tx.clone(), self.rotations(n)

    def diagonal_pairs(self, n):
        tx, Rx = self.uniform_poses(n)
        return tx, Rx, tx.clone(), Rx.clone()

    def training_pairs(self, n):
        """Declared 60/20/20 mixture; returns (tx, Rx, ty, Ry, family) with
        family = 0 uniform, 1 small, 2 near-pi."""
        n_u = int(round(self.mix[0] * n))
        n_s = int(round(self.mix[1] * n))
        n_p = n - n_u - n_s
        parts = [self.uniform_pairs(n_u), self.small_pairs(n_s), self.near_pi_pairs(n_p)]
        out = [torch.cat([p[i] for p in parts], 0) for i in range(4)]
        fam = torch.cat([torch.full((m,), k, device=self.device, dtype=torch.long)
                         for k, m in enumerate((n_u, n_s, n_p))])
        return (*out, fam)


def pack(tx, Rx, ty, Ry):
    return {'tx': tx, 'Rx': Rx, 'ty': ty, 'Ry': Ry}


def build_heldout_suites(seed, n_main=10000, n_suite=1000, device='cpu',
                         dtype=torch.float64, **sampler_kw):
    """Fixed evaluation suites; generated in float64 from a dedicated seed."""
    s = PoseSampler(seed=seed, device=device, dtype=dtype, **sampler_kw)
    suites = {
        'uniform': pack(*s.uniform_pairs(n_main)),
        'mixture': pack(*s.training_pairs(n_main)[:4]),
        'near_pi': pack(*s.near_pi_pairs(n_suite, 1e-6, 1e-1)),
        'exact_pi': pack(*s.exact_pi_pairs(n_suite)),
        'near_diag': pack(*s.small_pairs(n_suite, 1e-5, 1e-2)),
        'pure_translation': pack(*s.pure_translation_pairs(n_suite)),
        'pure_rotation': pack(*s.pure_rotation_pairs(n_suite)),
        'diagonal': pack(*s.diagonal_pairs(n_suite)),
    }
    return suites


def suites_to(suites, device, dtype):
    return {k: {kk: vv.to(device=device, dtype=dtype) for kk, vv in v.items()}
            for k, v in suites.items()}
