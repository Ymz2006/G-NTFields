"""Tests for the speed-conditioned Fourier heads (models/graph_method_triangle/speed_fourier.py).

    python -m unittest tests.test_speed_fourier
"""
import math
import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph_lie import lie  # noqa: E402
from models.graph_native import model_network_metric as mn  # noqa: E402
from models.graph_method_triangle import speed_fourier as sf  # noqa: E402

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def net(**graph):
    torch.manual_seed(0)
    return mn.NN(DEV, 6, None, graph=graph)


def rows(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 12, generator=g) - 0.5) * torch.tensor([0.8, 0.6, 0.9] + [0.6] * 3 + [0.8, 0.6, 0.9] + [0.6] * 3)


class SpeedFourierTests(unittest.TestCase):
    def test_sf_none_reproduces_graph_metric(self):
        a, b = net(arch='graph_metric'), net(arch='graph_metric_sf', sf_mode='none')
        x = rows().to(DEV)
        self.assertTrue(torch.equal(a.out(x)[0], b.out(x)[0]))

    def test_predictor_is_frozen_and_in_the_state_dict(self):
        n = net(arch='graph_metric_sf', sf_mode='scale')
        frozen = [k for k, p in n.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen and all('speed_pred' in k for k in frozen))
        self.assertTrue(any('speed_pred' in k for k in n.state_dict()))
        n2 = net(arch='graph_metric_sf', sf_mode='scale', sf_train_pred=True)
        self.assertTrue(all(p.requires_grad for p in n2.parameters()))

    def test_lie_gradient_includes_the_predictor_path(self):
        """The autograd Lie gradient equals the exp-perturbation reference, i.e. it differentiates
        through s_hat(x) (no detach by default); with sf_detach it does not, so the two differ."""
        for mode in ('scale', 'bands'):
            n = net(arch='graph_metric_sf', sf_mode=mode, sf_alpha=1.0, sf_fmax=32.0).double()
            with torch.no_grad():  # an untrained predictor is nearly constant: make s_hat(x) vary strongly
                n.net.enc.speed_pred.mlp[-1].weight.mul_(50.0)
            x = rows().double().to(DEV)
            _, cx, cy = lie.endpoint_gradients(n, x, 'lie', create_graph=False)
            _, rx, ry = lie.lie_gradients_perturb(n, x)
            self.assertTrue(torch.allclose(cx, rx, rtol=1e-6, atol=1e-8), mode)
            self.assertTrue(torch.allclose(cy, ry, rtol=1e-6, atol=1e-8), mode)
            n.net.enc.sf_detach = True
            _, dx, _ = lie.endpoint_gradients(n, x, 'lie', create_graph=False)
            self.assertFalse(torch.allclose(cx, dx, rtol=1e-3), mode)

    def test_band_gates_select_high_bands_at_low_speed(self):
        n = net(arch='graph_metric_sf', sf_mode='bands', sf_bands=4, sf_alpha=0.34)
        enc = n.net.enc
        w = enc.band_gates(torch.tensor([1.0, 0.1, 0.01, 0.001], device=DEV))
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(4, device=DEV)))
        self.assertEqual(w.argmax(1).tolist(), [0, 1, 2, 3])
        # band k of B carries 2^k times the base frequencies
        B = enc.B.t()  # (n_freq, 12)
        per = B.shape[0] // 4
        self.assertAlmostEqual((B[per:2 * per].abs().mean() / B[:per].abs().mean()).item(), 2.0, delta=0.4)

    def test_scale_mode_frequency_factor(self):
        n = net(arch='graph_metric_sf', sf_mode='scale', sf_alpha=0.5, sf_fmax=16.0)
        f = n.net.enc.frequency_factor(torch.tensor([1.0, 0.04, 0.001], device=DEV))
        self.assertTrue(torch.allclose(f, torch.tensor([1.0, 5.0, 16.0], device=DEV), rtol=1e-4))

    def test_lip_cap_bounds_row_norms(self):
        n = net(arch='graph_metric', lip_cap=4.0)
        enc = n.net.enc
        w = torch.randn(8, 256, device=DEV) * 3.0
        norms = enc.lip_norm(w).norm(dim=1)
        self.assertTrue((norms <= 4.0 + 1e-3).all() and norms.max() > 3.9)
        enc.lip_cap = 1.0
        self.assertTrue((enc.lip_norm(w).norm(dim=1) <= 1.0 + 1e-3).all())

    def test_predictor_checkpoint_loads_and_survives_native_loader(self):
        from models.graph_lie import model_train_lie as mtl
        pred = sf.SpeedPredictor()
        with torch.no_grad():
            for p in pred.parameters():
                p.add_(0.1)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'pred.pt')
            torch.save({'state_dict': pred.state_dict(), 'kw': {}}, path)
            n = net(arch='graph_metric_sf', sf_mode='scale', speed_pred_ckpt=path)
            self.assertTrue(sf.load_speed_predictor(n.net, path, DEV))
            x = rows().to(DEV)
            g = n.features(x[:, :6])
            self.assertTrue(torch.allclose(n.net.enc.speed_pred(g), pred.to(DEV)(g)))
            m = mtl.Model(d, 'datasets/3dshape/x', 6, [0.0] * 6, device=DEV)
            m.folder = d
            m.network, m.B = n, torch.zeros(128, 6)
            m.Params['Network']['Graph'] = n.graph
            m.save(name='latest.pt')
            m2 = mtl.Model(d, 'datasets/3dshape/x', 6, [0.0] * 6, device=DEV)
            m2.load(os.path.join(d, 'latest.pt'))
            self.assertTrue(torch.equal(m2.network.out(x)[0], n.out(x)[0]))

    def test_escape_time_gradient_is_one_over_speed(self):
        margin = 0.05
        d = torch.linspace(0.0015, 0.049, 50, dtype=torch.float64, requires_grad=True)
        s_raw = d / margin
        s_hat = s_raw ** 2 * (2 - s_raw) ** 2
        psi = sf.escape_time(s_hat, margin)
        (dpsi,) = torch.autograd.grad(psi.sum(), d)
        self.assertTrue(torch.allclose(dpsi, -1.0 / s_hat.detach(), rtol=1e-4))
        self.assertAlmostEqual(sf.escape_time(torch.tensor([1.0], dtype=torch.float64), margin).item(), 0.0, places=3)

    def test_psi_head_is_symmetric_zero_on_diagonal_and_differentiable(self):
        for psi in ('fixed', 'learned'):
            n = net(arch='graph_metric_sf', sf_mode='none', sf_psi=psi).double()
            with torch.no_grad():
                n.net.enc.speed_pred.mlp[-1].weight.mul_(50.0)
            x = rows().double().to(DEV)
            gx, gy = n.features(x[:, :6]), n.features(x[:, 6:])
            self.assertTrue(torch.allclose(n.net.value(gx, gy), n.net.value(gy, gx)))
            self.assertTrue(torch.allclose(n.net.value(gx, gx), torch.zeros(6, dtype=torch.float64, device=DEV), atol=1e-6))
            _, cx, cy = lie.endpoint_gradients(n, x, 'lie', create_graph=False)
            _, rx, ry = lie.lie_gradients_perturb(n, x)
            self.assertTrue(torch.allclose(cx, rx, rtol=1e-6, atol=1e-8), psi)
            plain = net(arch='graph_metric_sf', sf_mode='none').double()
            self.assertFalse(torch.allclose(n.net.value(gx, gy), plain.net.value(gx, gy)))

    def test_clearance_predictor_selection_equals_full_min_and_is_finite(self):
        from models.graph_method_triangle import clearance as cl
        pred = cl.ClearancePredictor(n_points=600, mode='boxes', n_boxes=2, d_bias=0.002).to(DEV).double()
        with torch.no_grad():
            pred.box_center.copy_(torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]))
            pred.box_half.copy_(torch.tensor([[0.05, 0.05, 0.05], [0.5, 0.4, 0.5]]))
            pred.box_sign.copy_(torch.tensor([1.0, -1.0]))  # a block and the room (container)
            pred.points.copy_(torch.randn(600, 3) * 0.05)
        x = rows(8).double().to(DEV)
        g = _features(x[:, :6]).requires_grad_(True)
        d_sel = pred.clearance(g)
        t, R = cl.se3.split_features(g)
        d_full = pred.sdf(torch.einsum('nij,kj->nki', R, pred.points) + t[:, None, :]).min(dim=1).values
        self.assertTrue(torch.allclose(d_sel, d_full))
        (gr,) = torch.autograd.grad(d_sel.sum(), g, create_graph=True)
        self.assertTrue(torch.isfinite(gr).all())
        # psi with the clearance predictor: finite value and finite double backward at s = 1 (free space)
        n = net(arch='graph_metric_sf', sf_mode='none', sf_psi='fixed', sf_pred_kind='clearance',
                sf_pred_kw=dict(n_points=600, mode='boxes', n_boxes=2, d_bias=0.002)).double()
        n.net.enc.speed_pred.load_state_dict(pred.state_dict())
        _, cx, cy = lie.endpoint_gradients(n, x, 'lie', create_graph=True)
        loss = ((cx.norm(dim=1) - 1) ** 2).sum() + ((cy.norm(dim=1) - 1) ** 2).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in n.parameters() if p.grad is not None))
        self.assertTrue((n.net.enc.speed_pred.raw_speed(_features(x[:, :6])) == 1.0).any())  # free-space poses present

    def test_psi_coord_and_psi_input(self):
        for kw in (dict(sf_psi='coord'), dict(sf_psi_in=0.3), dict(sf_psi='coord', sf_psi_in=0.3)):
            n = net(arch='graph_metric_sf', sf_mode='none', **kw).double()
            with torch.no_grad():
                n.net.enc.speed_pred.mlp[-1].weight.mul_(50.0)
            x = rows().double().to(DEV)
            gx, gy = n.features(x[:, :6]), n.features(x[:, 6:])
            self.assertTrue(torch.allclose(n.net.value(gx, gy), n.net.value(gy, gx)))
            self.assertTrue(torch.allclose(n.net.value(gx, gx), torch.zeros(6, dtype=torch.float64, device=DEV), atol=1e-6))
            _, cx, _ = lie.endpoint_gradients(n, x, 'lie', create_graph=False)
            _, rx, _ = lie.lie_gradients_perturb(n, x)
            self.assertTrue(torch.allclose(cx, rx, rtol=1e-6, atol=1e-8), kw)


def _features(x):
    return mn._sib.se3.features(x[:, :3], mn._sib.se3.so3_exp(2.0 * math.pi * x[:, 3:6]))


if __name__ == '__main__':
    unittest.main()
