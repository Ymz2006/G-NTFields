"""Mandatory tests for models/graph_method (spec page 8). Run from the repo root:

    python -m unittest tests.test_graph_method -v
"""

import math
import sys
import unittest

sys.path.append('.')

import torch

from models.graph_method.data import poses
from models.graph_method.fields import decoder, oracle
from models.graph_method.geometry import se3, tangents
from models.graph_method.losses import bellman, eikonal
from models.graph_method import model_network_graph as nets
from models.graph_method.planning import rollout

DT = torch.float64


def sampler(seed=0, **kw):
    return poses.PoseSampler(seed=seed, dtype=DT, **kw)


class TestSE3(unittest.TestCase):
    def test_rotation_validity(self):
        R = sampler().rotations(500)
        eye = torch.eye(3, dtype=DT)
        self.assertLess((R.transpose(-1, -2) @ R - eye).abs().max().item(), 1e-12)
        self.assertLess((torch.det(R) - 1).abs().max().item(), 1e-12)
        w = torch.randn(500, 3, dtype=DT) * 2
        Rw = se3.so3_exp(w)
        self.assertLess((Rw.transpose(-1, -2) @ Rw - eye).abs().max().item(), 1e-12)

    def test_quaternion_sign_invariance(self):
        q = torch.randn(200, 4, dtype=DT)
        self.assertLess((se3.quat_to_matrix(q) - se3.quat_to_matrix(-q)).abs().max().item(), 1e-15)

    def test_exp_log_roundtrip(self):
        s = sampler(1)
        axis = s.unit_vectors(400, 3)
        ang = torch.cat([torch.logspace(-9, -1, 100, dtype=DT), torch.rand(200, dtype=DT) * 3.0,
                         math.pi - torch.logspace(-7, -2, 100, dtype=DT)])
        w = axis * ang[:, None]
        self.assertLess((se3.so3_log(se3.so3_exp(w)) - w).abs().max().item(), 1e-7)
        xi = torch.cat([torch.randn(400, 3, dtype=DT), w], -1)
        t, R = se3.se3_exp(xi)
        self.assertLess((se3.se3_log(t, R) - xi).abs().max().item(), 1e-6)
        # regular angles: tight
        m = ang < 3.0
        self.assertLess((se3.se3_log(t[m], R[m]) - xi[m]).abs().max().item(), 1e-10)

    def test_compose_inverse(self):
        tx, Rx = sampler(2).uniform_poses(100)
        ti, Ri = se3.inverse(tx, Rx)
        t, R = se3.compose(tx, Rx, ti, Ri)
        self.assertLess(t.abs().max().item(), 1e-12)
        self.assertLess((R - torch.eye(3, dtype=DT)).abs().max().item(), 1e-12)
        ty, Ry = sampler(3).uniform_poses(100)
        tz, Rz = sampler(4).uniform_poses(100)
        a = se3.compose(*se3.compose(tx, Rx, ty, Ry), tz, Rz)
        b = se3.compose(tx, Rx, *se3.compose(ty, Ry, tz, Rz))
        self.assertLess((a[0] - b[0]).abs().max().item(), 1e-12)
        self.assertLess((a[1] - b[1]).abs().max().item(), 1e-12)

    def test_exp_autodiff_finite_at_zero(self):
        xi = torch.zeros(5, 6, dtype=DT, requires_grad=True)
        t, R = se3.se3_exp(xi)
        g = torch.autograd.grad((t.sum() + R.sum()), xi, create_graph=True)[0]
        self.assertTrue(torch.isfinite(g).all())
        gg = torch.autograd.grad(g.sum(), xi)[0]
        self.assertTrue(torch.isfinite(gg).all())


class TestOracleDecoder(unittest.TestCase):
    def test_decoder_matches_independent_oracle(self):
        s = sampler(5)
        tx, Rx, ty, Ry, _ = s.training_pairs(5000)
        a, b = oracle.latent_star(tx, Rx, ty, Ry)
        b, n_bad = oracle.clamp_b(b)
        T_dec = decoder.decode(a, b, 1.3, 0.7)
        T_or = oracle.time_star(tx, Rx, ty, Ry, 1.3, 0.7)
        scale = T_or.max().item()
        self.assertLess((T_dec - T_or).abs().max().item(), 1e-8 * scale)

    def test_exact_and_near_pi(self):
        s = sampler(6)
        tx, Rx, ty, Ry = s.exact_pi_pairs(300)
        th = oracle.relative_angle(Rx, Ry)
        self.assertLess((th - math.pi).abs().max().item(), 1e-7)
        a, b = oracle.latent_star(tx, Rx, ty, Ry)
        self.assertLess((b - 1).abs().max().item(), 1e-12)
        b, _ = oracle.clamp_b(b)
        T = decoder.decode(a, b, 1.0, 1.0)
        self.assertLess((T - oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0)).abs().max().item(), 1e-6)
        tx, Rx, ty, Ry = s.near_pi_pairs(300, 1e-6, 1e-2)
        a, b = oracle.latent_star(tx, Rx, ty, Ry)
        b, _ = oracle.clamp_b(b)
        # Near pi, b* = 1 - O(delta^2) loses precision in float64: looser tolerance
        self.assertLess((decoder.decode(a, b, 1.0, 1.0) - oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0)).abs().max().item(), 1e-6)

    def test_pure_motions_and_diagonal(self):
        s = sampler(7)
        tx, Rx, ty, Ry = s.pure_translation_pairs(200)
        T = oracle.time_star(tx, Rx, ty, Ry, 2.0, 1.0)
        self.assertLess((T - torch.linalg.norm(tx - ty, dim=-1) / 2.0).abs().max().item(), 1e-12)
        tx, Rx, ty, Ry = s.pure_rotation_pairs(200)
        T = oracle.time_star(tx, Rx, ty, Ry, 1.0, 0.5)
        w = se3.so3_log(Rx.transpose(-1, -2) @ Ry)
        self.assertLess((T - torch.linalg.norm(w, dim=-1) / 0.5).abs().max().item(), 1e-9)
        tx, Rx, ty, Ry = s.diagonal_pairs(100)
        self.assertEqual(oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0).abs().max().item(), 0.0)

    def test_symmetry_and_left_invariance(self):
        s = sampler(8)
        tx, Rx, ty, Ry = s.uniform_pairs(500)
        T = oracle.time_star(tx, Rx, ty, Ry, 1.0, 2.0)
        self.assertLess((T - oracle.time_star(ty, Ry, tx, Rx, 1.0, 2.0)).abs().max().item(), 1e-12)
        tg, Rg = s.uniform_poses(500)
        T2 = oracle.time_star(*se3.compose(tg, Rg, tx, Rx), *se3.compose(tg, Rg, ty, Ry), 1.0, 2.0)
        self.assertLess((T - T2).abs().max().item(), 1e-10)

    def test_phi_series_and_derivative(self):
        b = torch.cat([torch.logspace(-8, -1, 300, dtype=DT), torch.linspace(0.1, 0.999, 300, dtype=DT)])
        exact = 4 * torch.asin(torch.sqrt(b)) ** 2
        self.assertLess(((decoder.phi(b) - exact) / exact).abs().max().item(), 1e-12)
        bb = b.clone().requires_grad_(True)
        g = torch.autograd.grad(decoder.phi(bb).sum(), bb)[0]
        self.assertLess(((g - decoder.phi_prime(b)) / decoder.phi_prime(b)).abs().max().item(), 1e-10)
        z = torch.zeros(1, dtype=DT, requires_grad=True)
        g0 = torch.autograd.grad(decoder.phi(z).sum(), z)[0]
        self.assertAlmostEqual(g0.item(), 4.0, places=12)
        # continuity across the series switch
        e = torch.tensor([decoder.PHI_SERIES_MAX * (1 - 1e-9), decoder.PHI_SERIES_MAX * (1 + 1e-9)], dtype=DT)
        self.assertLess((decoder.phi(e)[0] - decoder.phi(e)[1]).abs().item(), 1e-10)


class TestTangentsEikonal(unittest.TestCase):
    def setUp(self):
        self.model = nets.build_network('analytic', s_t=1.7, s_r=0.6)
        s = sampler(9)
        tx, Rx, ty, Ry, _ = s.training_pairs(3000)
        m = eikonal.regular_mask(tx, Rx, ty, Ry, 1.7, 0.6, 1.0 / 1.7, 1e-3, 1e-4)
        self.p = (tx[m], Rx[m], ty[m], Ry[m])

    def test_projection_matches_exp_perturbation_and_fd(self):
        tx, Rx, ty, Ry = self.p
        gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
        _, cx, cy = tangents.tangent_gradients(self.model.value, gx, gy, 1.7, 0.6, create_graph=False)
        _, px, py = tangents.tangent_gradients_perturb(self.model.value, tx, Rx, ty, Ry, 1.7, 0.6)
        self.assertLess((cx - px).abs().max().item(), 1e-9)
        self.assertLess((cy - py).abs().max().item(), 1e-9)
        # central finite differences along exp(e E_i), with a step sweep
        scale = torch.tensor([1.7] * 3 + [0.6] * 3, dtype=DT)
        errs = []
        for eps in (1e-3, 1e-4, 1e-5):
            fd = torch.zeros_like(cx)
            for i in range(6):
                e = torch.zeros(6, dtype=DT)
                e[i] = eps
                tp, Rp = se3.right_perturb(tx, Rx, (e * scale).expand(tx.shape[0], 6))
                tm, Rm = se3.right_perturb(tx, Rx, (-e * scale).expand(tx.shape[0], 6))
                fd[:, i] = (oracle.time_star(tp, Rp, ty, Ry, 1.7, 0.6)
                            - oracle.time_star(tm, Rm, ty, Ry, 1.7, 0.6)) / (2 * eps)
            errs.append((fd - cx).abs().median().item())
        self.assertLess(min(errs), 1e-7)
        self.assertLess(errs[1], errs[0])  # converging before round-off

    def test_both_endpoint_eikonal(self):
        tx, Rx, ty, Ry = self.p
        _, cx, cy = tangents.tangent_gradients(self.model.value, se3.features(tx, Rx),
                                               se3.features(ty, Ry), 1.7, 0.6, create_graph=False)
        self.assertLess(((cx ** 2).sum(-1) - 1).abs().max().item(), 1e-6)
        self.assertLess(((cy ** 2).sum(-1) - 1).abs().max().item(), 1e-6)

    def test_bellman_consistency_of_oracle(self):
        tx, Rx, ty, Ry = [v[:500] for v in self.p]
        h = 0.01
        u = oracle.optimal_twist_direction(tx, Rx, ty, Ry, 1.7, 0.6)
        U = bellman.candidate_controls(u, 16, 0, 0.0)
        tgt, idx, _ = bellman.bellman_target(self.model, tx, Rx, ty, Ry, U, h, 1.7, 0.6)
        T = oracle.time_star(tx, Rx, ty, Ry, 1.7, 0.6)
        gap = tgt - T
        self.assertGreater(gap.min().item(), -1e-9)  # finite steps are admissible
        self.assertLess(gap.median().item(), 1e-3 * h)  # exact action nearly optimal


class TestNetworks(unittest.TestCase):
    def _pairs(self, n=64):
        tx, Rx, ty, Ry, _ = poses.PoseSampler(seed=11).training_pairs(n)
        return se3.features(tx, Rx), se3.features(ty, Ry)

    def test_symmetry_diag_and_finiteness(self):
        gx, gy = self._pairs()
        for arch in nets.ARCHITECTURES:
            net = nets.build_network(arch, fourier_seed=1).eval()
            T1, T2 = net.value(gx, gy), net.value(gy, gx)
            self.assertLess((T1 - T2).abs().max().item(), 1e-5, arch)
            self.assertTrue(torch.isfinite(T1).all(), arch)
            if arch in ('graph_gram', 'graph_sqdist'):
                self.assertLess(net.value(gx, gx).abs().max().item(), 1e-6, arch)

    def test_quaternion_sign_invariance_of_predictions(self):
        q = torch.randn(32, 4)
        t = torch.rand(32, 3)
        g1 = se3.features(t, se3.quat_to_matrix(q))
        g2 = se3.features(t, se3.quat_to_matrix(-q))
        gy, _ = self._pairs(32)
        for arch in nets.ARCHITECTURES:
            net = nets.build_network(arch).eval()
            self.assertEqual((net.value(g1, gy) - net.value(g2, gy)).abs().max().item(), 0.0, arch)

    def test_mixed_derivatives_finite_through_optimizer_step(self):
        gx, gy = self._pairs(128)
        for arch in nets.ARCHITECTURES:
            net = nets.build_network(arch)
            opt = torch.optim.Adam(net.parameters(), 1e-3)
            T, cx, cy = eikonal.eikonal_terms(net, gx, gy, create_graph=True)
            loss = eikonal.endpoint_residual(cx, cy).mean() + T.mean()
            opt.zero_grad()
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), arch)
            opt.step()
            T, cx, cy = eikonal.eikonal_terms(net, gx, gy, create_graph=False)
            self.assertTrue(torch.isfinite(cx).all() and torch.isfinite(cy).all(), arch)


class TestSamplingAndLeakage(unittest.TestCase):
    def test_reproducible_sampling(self):
        a = poses.PoseSampler(seed=3).training_pairs(100)
        b = poses.PoseSampler(seed=3).training_pairs(100)
        for u, v in zip(a, b):
            self.assertTrue(torch.equal(u, v))
        s = poses.PoseSampler(seed=4)
        s.training_pairs(10)
        st = s.state_dict()
        x1 = s.training_pairs(10)
        s.load_state_dict(st)
        x2 = s.training_pairs(10)
        self.assertTrue(torch.equal(x1[0], x2[0]))

    def test_no_oracle_label_leakage_in_physics_loss(self):
        from models.graph_method import model_function_graph as mf
        cfg = mf.default_config(arch='graph_gram', mode='physics', w_anchor=0.0, n_anchor=0,
                                batch_size=64, device='cpu')
        func = mf.Function(cfg)
        calls = {'n': 0}
        real = oracle.time_star

        def spy(*a, **k):
            calls['n'] += 1
            return real(*a, **k)

        # Masks are allowed to use input geometry; everything else must not.
        oracle.time_star = spy
        try:
            mask_calls = 0
            orig_mask = eikonal.regular_mask

            def counted_mask(*a, **k):
                nonlocal mask_calls
                mask_calls += 1
                return orig_mask(*a, **k)

            eikonal.regular_mask = counted_mask
            try:
                func.loss(step=0)
            finally:
                eikonal.regular_mask = orig_mask
        finally:
            oracle.time_star = real
        self.assertEqual(calls['n'], mask_calls)


class TestRollout(unittest.TestCase):
    def test_full_rollout_to_goal_with_analytic_field(self):
        model = nets.build_network('analytic')
        tx, Rx, ty, Ry = sampler(12).uniform_pairs(40)
        T = oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0)
        for mode in ('gradient', 'sampling'):
            res = rollout.rollout(model, tx, Rx, ty, Ry, mode=mode, h=0.02, max_steps=600)
            self.assertTrue(res['success'].all(), mode)
            ratio = res['cost'] / T
            self.assertLess(ratio.max().item(), 1.25 if mode == 'sampling' else 1.02, mode)
            self.assertGreater(ratio.min().item(), 1 - 1e-6, mode)


if __name__ == '__main__':
    unittest.main()
