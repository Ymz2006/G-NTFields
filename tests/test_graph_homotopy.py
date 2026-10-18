"""Architectural tests for the double-cover (homotopy) head, models/graph_method_triangle/homotopy_decoder.py.
Random perturbed parameters in float64: the architecture is tested, not a trained field.

    python -m unittest tests.test_graph_homotopy -v
"""
import math
import sys
import unittest

sys.path.append('.')

import torch

from models.graph_method_triangle import branch_decoder as bd
from models.graph_method_triangle import homotopy_decoder as hd
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_network_metric as gn_network

DT = torch.float64
TWO_PI = 2.0 * math.pi
VARIANTS = {
    'hm': dict(),
    'hm_pair': dict(pair_dim=32, pair_hidden=64),
    'hm_strict': dict(conf=False),
    'hm_wide_pair': dict(res_even=64, res_odd=64, pair_dim=64),
}


def make_net(extra=None, seed=0, perturb=0.05):
    torch.manual_seed(seed)
    net = gn_network.NN('cpu', 6, None, graph=dict(arch='graph_homotopy', **(extra or {})))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(perturb * torch.randn_like(p))
        if net.net.conf:
            net.net.rho_head.weight.normal_(0, 0.2)
            net.net.lam_head.weight.normal_(0, 0.2)
    return net.to(DT)


def random_poses(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9
    q = torch.randn(n, 4, generator=g, dtype=DT)
    return t, q / q.norm(dim=1, keepdim=True)


class TestCoverField(unittest.TestCase):
    def test_deck_invariance_and_sheet_swap(self):
        for name, extra in VARIANTS.items():
            net = make_net(extra).net
            tx, qx = random_poses(64, 1)
            ty, qy = random_poses(64, 2)
            tp = net.value_cover(tx, qx, ty, qy)
            self.assertTrue(torch.allclose(tp, net.value_cover(tx, -qx, ty, -qy), atol=1e-12), name)
            tm = net.value_cover(tx, qx, ty, -qy)
            self.assertTrue(torch.allclose(tm, net.value_cover(tx, -qx, ty, qy), atol=1e-12), name)
            self.assertGreater((tp - tm).abs().max().item(), 1e-3, name)   # the sheets differ
            # base value = min over the sheets, whatever lifts the matrices give
            gx, gy = se3.features(tx, se3.quat_to_matrix(qx)), se3.features(ty, se3.quat_to_matrix(qy))
            T = net.value(gx, gy)
            self.assertTrue(torch.allclose(T, torch.minimum(tp, tm), atol=1e-10), name)
            sv = net.sheet_values(gx, gy)
            self.assertTrue(torch.allclose(sv.min(1).values, T, atol=1e-10), name)
            sig = torch.ones(64, dtype=DT)
            s1 = net.sheet_value(gx, gy, sig)
            s2 = net.sheet_value(gx, gy, -sig)
            self.assertTrue(torch.allclose(torch.minimum(s1, s2), T, atol=1e-10), name)
            self.assertTrue(torch.allclose(torch.maximum(s1, s2), torch.maximum(tp, tm), atol=1e-10), name)

    def test_symmetry_zero_lower_bound(self):
        for name, extra in VARIANTS.items():
            net = make_net(extra).net
            tx, qx = random_poses(128, 3)
            ty, qy = random_poses(128, 4)
            gx, gy = se3.features(tx, se3.quat_to_matrix(qx)), se3.features(ty, se3.quat_to_matrix(qy))
            T, Tt = net.value(gx, gy), net.value(gy, gx)
            self.assertTrue(torch.allclose(T, Tt, atol=1e-10), name)
            self.assertTrue(torch.allclose(net.value_cover(tx, qx, ty, qy), net.value_cover(ty, qy, tx, qx), atol=1e-10), name)
            self.assertEqual(net.value(gx, gx).abs().max().item(), 0.0, name)
            self.assertTrue(bool((T >= net.free_time(gx, gy) - 1e-12).all()), name)
            self.assertTrue(bool((T > 0).all()), name)

    def test_cone_conformal(self):
        """T(p exp(eps xi), p) / (eps |xi|_g) -> 1 + rho(p) + P(p, p) with a quadratic correction."""
        for name, extra in VARIANTS.items():
            net = make_net(extra).net
            t, q = random_poses(16, 5)
            R = se3.quat_to_matrix(q)
            g = torch.Generator().manual_seed(6)
            xi = torch.randn(16, 6, generator=g, dtype=DT)
            xi = xi / torch.sqrt((xi[:, :3] ** 2).sum(1) + (xi[:, 3:] ** 2).sum(1) / TWO_PI ** 2)[:, None]
            slope = net.endpoint_slope(se3.features(t, R))
            errs = []
            for eps in (1e-4, 1e-6):
                t2, R2 = se3.right_perturb(t, R, eps * xi)   # xi = (v metres, w radians), |xi|_g = 1
                T = net.value(se3.features(t2, R2), se3.features(t, R))
                errs.append(((T / eps) / slope - 1).abs().max().item())
            self.assertLess(errs[1], 1e-3, (name, errs))
            self.assertLess(errs[1], errs[0], (name, errs))

    def test_lift_exp(self):
        g = torch.Generator().manual_seed(7)
        r = torch.randn(32, 3, generator=g, dtype=DT) * 0.6
        q = hd.quat_exp_turns(r)
        self.assertTrue(torch.allclose(q.norm(dim=1), torch.ones(32, dtype=DT), atol=1e-12))
        self.assertTrue(torch.allclose(se3.quat_to_matrix(q), se3.so3_exp(TWO_PI * r), atol=1e-10))
        n = r / r.norm(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(hd.quat_exp_turns(n), -hd.quat_exp_turns(torch.zeros_like(n)), atol=1e-12))  # one turn -> -1
        # relative quaternion: scalar part = dot, R(q_x^* q_y) = R_x^T R_y
        _, qx = random_poses(32, 8)
        _, qy = random_poses(32, 9)
        rel = hd.relative_quat(qx, qy)
        self.assertTrue(torch.allclose(rel[:, 0], (qx * qy).sum(1), atol=1e-12))
        self.assertTrue(torch.allclose(se3.quat_to_matrix(rel), se3.quat_to_matrix(qx).transpose(-1, -2) @ se3.quat_to_matrix(qy), atol=1e-10))

    def test_loss_finite_all_lifts(self):
        from models.graph_homotopy import model_function_homotopy as mfh
        g = torch.Generator().manual_seed(10)
        for name, extra in (('hm', {}), ('hm_pair', dict(pair_dim=16, pair_hidden=32))):
            for lift in ('min', 'random', 'both'):
                net = make_net(extra)
                fn = mfh.Function('/tmp', 'cpu', net, 6, lift=lift, lift_far=0.5)
                pts = torch.cat([(torch.rand(64, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                                 (torch.rand(64, 3, generator=g, dtype=DT) - 0.5) * 1.2,
                                 (torch.rand(64, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                                 (torch.rand(64, 3, generator=g, dtype=DT) - 0.5) * 1.2], 1)
                pts[:8, 6:] = pts[:8, :6] + 1e-3  # near-coincident pairs (the cone; the exact tip has no gradient)
                sp = {k: 0.2 + 0.8 * torch.rand(64, 2, generator=g, dtype=DT) for k in ('S', 'Sd', 'Sa')}
                loss, loss_n, _ = fn.Loss(pts, sp, 1.0, 0.0, 1)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item(), (name, lift))
                for p in net.parameters():
                    self.assertTrue(torch.isfinite(p.grad).all().item() if p.grad is not None else True, (name, lift))
                if lift == 'both':
                    self.assertAlmostEqual(fn.last_terms['far'], 0.5)


if __name__ == '__main__':
    unittest.main()
