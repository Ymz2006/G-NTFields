"""Architectural property tests for the cover-quotient metric head (models/graph_method_triangle/cover_decoder.py).

    python -m unittest tests.test_graph_cover -v

Random (perturbed) parameters in float64: the tests check the architecture, not a trained field. Properties
follow the numbering of "Decoder properties that hold with or without obstacles".
"""
import math
import sys
import unittest

sys.path.append('.')

import torch

from models.graph_method_triangle import cover_decoder as cd
from models.graph_method_triangle import triangle
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_network_metric as gn_network

DT = torch.float64
TWO_PI = 2.0 * math.pi
HEADS = {
    'cover_lp2': dict(n_cut=2, res_kind='lp'),
    'cover_lp0': dict(n_cut=0, res_kind='lp'),
    'cover_eu2': dict(n_cut=2, res_kind='euclid'),
    'cover_lp3_nofree': dict(n_cut=3, res_kind='lp', tfree=False),
    'cover_lse2': dict(n_cut=2, res_kind='lse'),
}
METRIC = ('cover_lp2', 'cover_lp0', 'cover_eu2', 'cover_lp3_nofree')


def make_net(extra, seed=0, perturb=0.05):
    torch.manual_seed(seed)
    net = gn_network.NN('cpu', 6, None, graph=dict(arch='graph_cover', **extra))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(perturb * torch.randn_like(p))
    return net.to(DT)


def random_poses(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9
    q = torch.randn(n, 4, generator=g, dtype=DT)
    q = q / q.norm(dim=1, keepdim=True)
    return t, se3.quat_to_matrix(q)


def feats(t, R):
    return se3.features(t, R)


class TestHalfAngle(unittest.TestCase):
    def test_square_root_identity_and_norm(self):
        z = torch.randn(5000, 2, dtype=DT) * 3
        w = cd.half_angle(z, eps=0.0)
        # w = |z|^{1/2} sqrt(z): |w| = |z| and (w / |w|^{1/2})^2 = z as complex numbers
        r = z.norm(dim=1)
        self.assertLess((w.norm(dim=1) - r).abs().max().item(), 1e-12)
        u = w / r.sqrt()[:, None]
        sq = torch.stack([u[:, 0] ** 2 - u[:, 1] ** 2, 2 * u[:, 0] * u[:, 1]], 1)
        self.assertLess((sq - z).abs().max().item(), 1e-10)

    def test_gradient_finite_near_cut_and_branch_point(self):
        z = torch.tensor([[-1.0, 1e-9], [-1.0, -1e-9], [1e-9, 0.0], [0.0, 0.0], [-1e-7, 1e-9], [2.0, 0.0]],
                         dtype=DT, requires_grad=True)
        w = cd.half_angle(z, eps=1e-6)
        J = torch.autograd.functional.jacobian(lambda x: cd.half_angle(x, 1e-6), z.detach())
        self.assertTrue(torch.isfinite(w).all())
        self.assertTrue(torch.isfinite(J).all())
        self.assertLess(J.abs().max().item(), 10.0)  # |z|-scaling keeps the derivative bounded at the branch point

    def test_sign_flip_across_cut_only(self):
        # crossing the cut {z1 < 0, z2 = 0} flips the sign; crossing the positive real axis does not
        a = cd.half_angle(torch.tensor([[-1.0, 1e-6]], dtype=DT), 0.0)
        b = cd.half_angle(torch.tensor([[-1.0, -1e-6]], dtype=DT), 0.0)
        self.assertLess((a + b).abs().max().item(), 1e-5)
        a = cd.half_angle(torch.tensor([[1.0, 1e-6]], dtype=DT), 0.0)
        b = cd.half_angle(torch.tensor([[1.0, -1e-6]], dtype=DT), 0.0)
        self.assertLess((a - b).abs().max().item(), 1e-5)


class TestCoverHead(unittest.TestCase):
    def test_symmetry_zero_and_lower_bound(self):
        for name, extra in HEADS.items():
            net = make_net(extra)
            t, R = random_poses(300, 1)
            t2, R2 = random_poses(300, 2)
            gx, gy = feats(t, R), feats(t2, R2)
            Txy = net.net.value(gx, gy)
            Tyx = net.net.value(gy, gx)
            self.assertLess((Txy - Tyx).abs().max().item(), 1e-10, name)        # property 2
            self.assertLess(net.net.value_sq(gx, gx).abs().max().item(), 1e-20, name)  # property 1
            self.assertTrue((Txy > 0).all(), name)
            if extra.get('tfree', True):
                self.assertTrue((Txy >= net.net.free_time(gx, gy) - 1e-12).all(), name)  # property 4

    def test_residual_off_is_exact_free_space(self):
        net = make_net(dict(n_cut=2, res_kind='lp', res_scale=0.0))
        t, R = random_poses(2000, 3)
        t2, R2 = random_poses(2000, 4)
        T = net.net.value(feats(t, R), feats(t2, R2))
        Tstar = oracle.time_star(t, R, t2, R2, 1.0, TWO_PI)
        self.assertLess((T - Tstar).abs().max().item(), 1e-9)

    def test_triangle_inequality_adversarial(self):
        tx, Rx = random_poses(256, 5)
        tz, Rz = random_poses(256, 6)
        for name, extra in HEADS.items():
            net = make_net(extra)
            viol, Txz, _ = triangle.attack(net.net, tx, Rx, tz, Rz, steps=40, lr=0.02)
            rel = (viol / Txz.clamp_min(1e-12)).max().item()
            if name in METRIC:
                self.assertLess(rel, 1e-9, '{}: violation {}'.format(name, rel))     # property 3
            else:
                self.assertGreater(rel, 1e-4, 'lse ablation should be violable')

    def test_min_matches_candidates_and_ridges_exist(self):
        net = make_net(dict(n_cut=2, res_kind='lp', res_scale=1.0), perturb=0.3)
        t, R = random_poses(4000, 7)
        t2, R2 = random_poses(4000, 8)
        gx, gy = feats(t, R), feats(t2, R2)
        Q = net.net.value_sq(gx, gy)
        cand = net.net.all_candidates(gx, gy)
        self.assertEqual(cand.shape[1], 8)
        self.assertLess((Q - cand.amin(1)).abs().max().item(), 1e-10)
        used = torch.bincount(cand.argmin(1), minlength=8)
        self.assertGreater(int((used > 0).sum()), 2, 'several group elements should be active (movable ridges)')

    def test_double_backward_finite(self):
        net = make_net(dict(n_cut=2, res_kind='lp'))
        t, R = random_poses(64, 9)
        t2, R2 = random_poses(64, 10)
        gx = feats(t, R).requires_grad_(True)
        gy = feats(t2, R2).requires_grad_(True)
        T = net.net.value(gx, gy)
        dgx, = torch.autograd.grad(T.sum(), gx, create_graph=True)
        loss = (dgx ** 2).sum()
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_cone_at_goal_is_sqrt_of_smooth_Q(self):
        # Q(p exp(eps xi), p) = eps^2 (quadratic form) + O(eps^3) in both endpoints (property 5)
        net = make_net(dict(n_cut=2, res_kind='euclid', res_scale=0.3))
        t, R = random_poses(50, 11)
        g = torch.Generator().manual_seed(12)
        xi = torch.randn(50, 6, generator=g, dtype=DT)
        xi = xi / xi.norm(dim=1, keepdim=True)
        base = feats(t, R)
        vals = {}
        for eps in (1e-3, 2e-3, 4e-3):
            t2, R2 = se3.right_perturb(t, R, eps * xi)
            vals[eps] = (net.net.value_sq(feats(t2, R2), base), net.net.value_sq(base, feats(t2, R2)))
        for k in range(2):
            q1, q2, q4 = vals[1e-3][k], vals[2e-3][k], vals[4e-3][k]
            self.assertLess(((q2 / q1) - 4).abs().max().item(), 2e-2)
            self.assertLess(((q4 / q2) - 4).abs().max().item(), 4e-2)


if __name__ == '__main__':
    unittest.main()
