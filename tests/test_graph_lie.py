"""Tests for models/graph_lie (Lie-gradient Eikonal training of the graph heads). From the repo root:

    python -m unittest tests.test_graph_lie -v
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

sys.path.append('.')

import numpy as np
import torch

from models.graph_lie import lie, model_function_lie as mfl, model_train_lie as mtl
from models.graph_native import model_function_metric as gn_function, model_network_metric as gn_network
from models.graph_native import model_train_metric as gn_train
from models.graph_method_triangle import model_network_graph as nets

DT = torch.float64
TWO_PI = 2.0 * math.pi
HEADS = {'graph_metric': {}, 'graph_tri_metric': {'rot_mode': 'equivariant'}}


def make_net(arch, dtype=DT, seed=0, perturb=0.05):
    torch.manual_seed(seed)
    net = gn_network.NN('cpu', 6, None, graph=dict(arch=arch, **HEADS.get(arch, {})))
    with torch.no_grad():  # move away from the symmetric init (zero gates) so all paths are active
        for p in net.parameters():
            p.add_(perturb * torch.randn_like(p))
    return net.to(dtype)


def random_coords(n, seed=0, max_turns=0.49, dtype=DT):
    """(n, 12) chart rows; rotation vectors with angle up to 2 pi max_turns."""
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(n, 2, 3, generator=g, dtype=dtype) - 0.5) * 0.9
    axis = torch.randn(n, 2, 3, generator=g, dtype=dtype)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    ang = torch.rand(n, 2, 1, generator=g, dtype=dtype) * max_turns
    return torch.cat([t[:, 0], axis[:, 0] * ang[:, 0], t[:, 1], axis[:, 1] * ang[:, 1]], -1)


class TestLieGradient(unittest.TestCase):
    def test_lie_gradient_equals_exp_perturbation(self):
        """Tangent projection == d/de tau(x exp(e E_i)) through the SE(3) exponential, both heads."""
        X = random_coords(256, seed=1)
        for arch in HEADS:
            net = make_net(arch)
            T, cx, cy = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
            T2, px, py = lie.lie_gradients_perturb(net, X)
            self.assertLess((T - T2).abs().max().item(), 1e-12, arch)
            scale = cx.abs().max().item()
            self.assertLess((cx - px).abs().max().item() / scale, 1e-9, arch)
            self.assertLess((cy - py).abs().max().item() / scale, 1e-9, arch)

    def test_lie_equals_jacobian_mapped_chart_gradient(self):
        """c_v = R^T g_v and c_w = Jr^{-1}(2 pi r)^T g_w (chain rule), up to 175 degrees."""
        X = random_coords(256, seed=2, max_turns=175.0 / 360.0)
        for arch in HEADS:
            net = make_net(arch)
            _, gx, gy = lie.endpoint_gradients(net, X, 'chart', create_graph=False)
            _, cx, cy = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
            for g, c, r in ((gx, cx, X[:, 3:6]), (gy, cy, X[:, 9:12])):
                mapped = lie.chart_to_lie(r, g)
                self.assertLess((mapped - c).abs().max().item() / c.abs().max().item(), 1e-8, arch)
                self.assertLess(((g[:, :3].norm(dim=1) - c[:, :3].norm(dim=1)).abs()).max().item(), 1e-10)

    def test_chart_gradient_matches_finite_differences(self):
        X = random_coords(32, seed=3)
        net = make_net('graph_metric')
        _, gx, gy = lie.endpoint_gradients(net, X, 'chart', create_graph=False)
        g = torch.cat([gx, gy], 1)
        h = 1e-6
        fd = torch.zeros_like(g)
        with torch.no_grad():
            for k in range(12):
                e = torch.zeros_like(X)
                e[:, k] = h
                fd[:, k] = (net.out(X + e)[0][:, 0] - net.out(X - e)[0][:, 0]) / (2 * h)
        self.assertLess((fd - g).abs().max().item() / g.abs().max().item(), 1e-6)

    def test_gradients_coincide_at_identity_rotation(self):
        X = random_coords(64, seed=4)
        X[:, 3:6] = 0.0
        net = make_net('graph_tri_metric')
        _, gx, _ = lie.endpoint_gradients(net, X, 'chart', create_graph=False)
        _, cx, _ = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
        self.assertLess((gx - cx).abs().max().item(), 1e-10)

    def test_rotation_block_factor(self):
        """|c_w| / |g_w| = (theta/2) / sin(theta/2) orthogonal to r and 1 along r."""
        n = 200
        g = torch.Generator().manual_seed(5)
        r = torch.randn(n, 3, generator=g, dtype=DT)
        r = r / r.norm(dim=1, keepdim=True) * torch.rand(n, 1, generator=g, dtype=DT) * 0.49
        u = torch.randn(n, 3, generator=g, dtype=DT)
        perp = u - (u * r).sum(1, keepdim=True) * r / (r * r).sum(1, keepdim=True)
        for gw, expect in ((perp, lie.perpendicular_factor(r)), (r.clone(), torch.ones(n, dtype=DT))):
            G = torch.cat([torch.zeros(n, 3, dtype=DT), gw], 1)
            ratio = lie.chart_to_lie(r, G)[:, 3:].norm(dim=1) / gw.norm(dim=1)
            self.assertLess((ratio - expect).abs().max().item(), 1e-9)
        self.assertAlmostEqual(lie.perpendicular_factor(torch.tensor([[0.5, 0, 0]], dtype=DT)).item(),
                               math.pi / 2, places=6)

    def test_free_space_geodesic_solves_lie_eikonal_but_not_chart(self):
        """The exact free-space time T = sqrt(|dt|^2 + (theta / 2 pi)^2) on SE(3) has |c_x| = 1 for every
        pose (the Lie Eikonal with S = 1), while the chart gradient's norm depends on |rotvec_x|: the
        chart equation is not satisfied by the true geodesic distance."""
        X = random_coords(4000, seed=6, max_turns=0.49)
        net = gn_network.NN('cpu', 6, None, graph=dict(arch='graph_metric'))
        net.net = nets.AnalyticGraph(s_t=1.0, s_r=TWO_PI)
        net = net.to(DT)
        T, cx, _ = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
        _, gx, _ = lie.endpoint_gradients(net, X, 'chart', create_graph=False)
        b = net.net.predict_raw(net.features(X[:, :6]), net.features(X[:, 6:]))[:, 1]
        reg = (T > 1e-2) & (b < 1 - 1e-3)  # away from the diagonal and the cut locus
        lie_res = (cx[reg].norm(dim=1) - 1).abs()
        chart_res = (gx[reg].norm(dim=1) - 1).abs()
        self.assertLess(lie_res.max().item(), 1e-6)
        big = (X[reg][:, 3:6].norm(dim=1) > 0.35)  # start rotation beyond 126 degrees
        self.assertGreater(chart_res[big].median().item(), 1e-2)
        self.assertGreater(chart_res[big].max().item(), 0.1)


class TestLoss(unittest.TestCase):
    def batch(self, n=512, seed=7, dtype=torch.float32):
        X = random_coords(n, seed=seed, dtype=dtype)
        g = torch.Generator().manual_seed(seed)
        raw = torch.rand(n, 6, generator=g, dtype=dtype)
        d = torch.zeros(n, 30, dtype=dtype)
        d[:, :12] = X
        d[:, 12:14] = 0.01 + 0.99 * raw[:, :2]
        d[:, 26:28] = 0.02 + 0.98 * raw[:, 2:4]
        d[:, 28:30] = 0.3 * raw[:, 4:6]
        return d

    def test_combined_loss_equals_graph_native(self):
        """speed 'combined' reproduces models/graph_native's loss exactly, in both gradient modes."""
        d = self.batch()
        for arch in HEADS:
            for mode in ('chart', 'lie'):
                net = make_net(arch, dtype=torch.float32)
                ref = gn_function.Function('', 'cpu', net, 6, grad_mode=mode)
                new = mfl.Function('', 'cpu', net, 6, grad_mode=mode, speed_model='combined')
                S = d[:, 12:14]
                S = S * S * (2 - S) * (2 - S)
                S = 1.0 * S + 1 - 1.0
                l_ref, n_ref, _ = ref.Loss(d[:, :12], S, d[:, 14:26], 3.0, 1e-3, 1)
                l_new, n_new, _ = new.Loss(d[:, :12], mtl.warp_speeds(d), 3.0, 1e-3, 1)
                self.assertEqual(l_ref.item(), l_new.item(), (arch, mode))
                g_ref = torch.autograd.grad(l_ref, list(net.parameters()), allow_unused=True)
                g_new = torch.autograd.grad(l_new, list(net.parameters()), allow_unused=True)
                for a, b in zip(g_ref, g_new):
                    if a is not None:
                        self.assertTrue(torch.equal(a, b), (arch, mode))

    def test_speed_model_formulas(self):
        g = torch.randn(100, 6, dtype=DT)
        Sd, Sa = torch.rand(100, 2, dtype=DT) + 0.1, torch.rand(100, 2, dtype=DT) + 0.1
        sp = {'S': Sd, 'Sd': Sd, 'Sa': Sa}
        iso = mfl.eikonal_N(g, sp, 0, 'iso')
        self.assertLess((iso - Sd[:, 0] * torch.sqrt((g ** 2).sum(1) + 1e-8)).abs().max().item(), 1e-14)
        aniso_eq = mfl.eikonal_N(g, {'Sd': Sd, 'Sa': Sd}, 0, 'aniso')
        # S_a = S_d: aniso == iso up to where the 1e-8 regulariser sits (S sqrt(q + eps) vs sqrt(S^2 q + eps),
        # both as in the Lie-Shape-NTFields residuals)
        self.assertLess((aniso_eq - iso).abs().max().item(), 1e-7)
        self.assertLess((mfl.eikonal_N(g, {'Sd': Sd, 'Sa': Sd}, 0, 'aniso', eps=0.0)
                         - mfl.eikonal_N(g, sp, 0, 'iso', eps=0.0)).abs().max().item(), 1e-14)
        aniso = mfl.eikonal_N(g, sp, 1, 'aniso')
        ref = torch.sqrt(Sd[:, 1] ** 2 * (g[:, :3] ** 2).sum(1) + Sa[:, 1] ** 2 * (g[:, 3:] ** 2).sum(1) + 1e-8)
        self.assertLess((aniso - ref).abs().max().item(), 1e-14)
        self.assertLess((mfl.eikonal_N(g, sp, 0, 'combined') - iso).abs().max().item(), 1e-14)

    def test_warp_speeds_matches_original_trainers(self):
        """Verbatim speed preprocessing of graph_native (S) and Lie-Shape-NTFields lie_metric (S_d, S_a)."""
        d = self.batch(2000)
        d[:5, 28:30] = 0.0       # zero rotational clearance -> clamped
        d[5:10, 28:30] = 0.9     # beyond pi / 6 -> 1
        out = mtl.warp_speeds(d)
        speed = d[:, 12:14]
        speed = speed * speed * (2 - speed) * (2 - speed)
        speed = 1.0 * speed + 1 - 1.0
        speed_dist = d[:, 26:28]
        speed_angle = d[:, 28:30]
        angle_max = 1 / 6
        angle_min = 0.0001
        speed_angle = torch.clamp(speed_angle / angle_max, min=angle_min / angle_max, max=1)
        speed_dist = speed_dist ** 2 * (2 - speed_dist) ** 2
        speed_angle = speed_angle ** 2 * (2 - speed_angle) ** 2
        speed_dist = torch.clamp(speed_dist, min=0.001)
        speed_angle = torch.clamp(speed_angle, min=0.001)
        self.assertTrue(torch.equal(out['S'], speed))
        self.assertTrue(torch.equal(out['Sd'], speed_dist))
        self.assertTrue(torch.equal(out['Sa'], speed_angle))
        self.assertTrue(bool((out['Sa'][5:10] == 1).all()))

    def test_lie_loss_is_finite_at_the_chart_seam(self):
        """The Lie gradient needs no chart Jacobian, so rotations at exactly pi are harmless."""
        d = self.batch(64)
        axis = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
        d[:, 3:6] = 0.5 * axis
        net = make_net('graph_tri_metric', dtype=torch.float32)
        for speed_model in mfl.SPEED_MODELS:
            f = mfl.Function('', 'cpu', net, 6, grad_mode='lie', speed_model=speed_model)
            loss, _, _ = f.Loss(d[:, :12], mtl.warp_speeds(d), 1.0, 1e-3, 1)
            grads = torch.autograd.grad(loss, list(net.parameters()), allow_unused=True)
            self.assertTrue(math.isfinite(loss.item()))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads if g is not None))


class TestTrainerContract(unittest.TestCase):
    def test_train_save_and_load_with_graph_native_loader(self):
        """A short run writes checkpoints that the evaluator's loader (graph_native Model.load) reads
        back to the identical field."""
        tmp = tempfile.mkdtemp()
        try:
            data = os.path.join(tmp, 'data')
            os.makedirs(data)
            n = 3000
            X = random_coords(n, seed=8, dtype=torch.float32).numpy()
            rng = np.random.default_rng(0)
            np.save(os.path.join(data, 'sampled_points.npy'), X)
            np.save(os.path.join(data, 'speed.npy'), rng.uniform(0.01, 1, (n, 2)).astype(np.float32))
            np.save(os.path.join(data, 'normal.npy'), np.zeros((n, 12), np.float32))
            np.save(os.path.join(data, 'speed_dists.npy'), rng.uniform(0.02, 1, (n, 2)).astype(np.float32))
            np.save(os.path.join(data, 'speed_angles.npy'), rng.uniform(0, 0.3, (n, 2)).astype(np.float32))
            for speed_model in ('iso', 'aniso'):
                m = mtl.Model(tmp, data, 6, [0.0] * 6, device='cpu')
                m.folder = os.path.join(tmp, 'run_' + speed_model)
                m.Params['Training'].update({'Number of Epochs': 3, 'Grad mode': 'lie', 'Speed model': speed_model,
                                             'Print Every * Epoch': 1})
                m.Params['Network']['Graph'] = {'arch': 'graph_tri_metric', 'rot_mode': 'equivariant'}
                m.train()
                ck = torch.load(os.path.join(m.folder, 'latest.pt'))
                self.assertEqual((ck['grad_mode'], ck['speed_model'], ck['epoch']), ('lie', speed_model, 3))
                ref = gn_train.Model(tmp, data, 6, [0.0] * 6, device='cpu')
                ref.load(os.path.join(m.folder, 'latest.pt'))
                again = mtl.Model(tmp, data, 6, [0.0] * 6, device='cpu')
                again.load(os.path.join(m.folder, 'latest.pt'))
                self.assertEqual(again.function.speed_model, speed_model)
                Xq = torch.from_numpy(X[:500])
                self.assertTrue(torch.equal(ref.function.TravelTimes(Xq), m.function.TravelTimes(Xq)))
                self.assertTrue(torch.equal(again.function.TravelTimes(Xq), m.function.TravelTimes(Xq)))
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
