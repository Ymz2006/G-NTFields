"""Tests of models/path_constrained: charts, the projected Eikonal gradients, the exact prism geometry.

    python -m unittest tests.test_path_constrained -v        (in the container, repo root)
"""
import math
import os
import unittest

import numpy as np
import torch

from models.path_constrained import geometry as G
from models.path_constrained import losses as LS
from models.path_constrained.path import Path, resample, smooth_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWO_PI = 2 * math.pi


def curved_path():
    via = np.array([[-0.3, -0.1, 0.2], [-0.1, 0.15, 0.0], [0.15, -0.05, -0.2], [0.35, 0.1, 0.1]])
    return Path(resample(smooth_curve(via), 1e-3))


class _Tangent:
    def __init__(self, path, chart):
        self.path, self.chart = path, chart

    def __call__(self, t):
        return self.path.chart_tangent(t, self.chart)


def smooth_T(x, y):
    """A smooth non-trivial function of two chart poses (B, 6) (translation, rotvec / 2 pi)."""
    Rx = G.so3_exp(TWO_PI * x[:, 3:])
    Ry = G.so3_exp(TWO_PI * y[:, 3:])
    a = torch.tensor([0.3, -1.2, 0.7], dtype=x.dtype)
    return (torch.sin(3 * x[:, :3] @ a) + ((x[:, :3] - y[:, :3]) ** 2).sum(1) + (Rx * Ry).sum((1, 2))
            + (Rx[:, :, 2] * x[:, :3]).sum(1) * 2.0)


class TestPath(unittest.TestCase):
    def test_arclength_and_charts(self):
        p = curved_path()
        ell = torch.linspace(0, p.L, 257, dtype=torch.float64)
        seg = (p.pos(ell[1:]) - p.pos(ell[:-1])).norm(dim=1)
        self.assertAlmostEqual(float(seg.sum()), p.L, delta=2e-4)          # chord sum -> arc length
        for chart in ('arclength', 'world'):
            t = p.chart_t(ell, chart)
            back = p.chart_ell(t, chart)
            self.assertLess(float((back - ell).abs().max()), 1e-6, chart)
        tau = p.tangent(ell)
        self.assertLess(float((tau.norm(dim=1) - 1).abs().max()), 1e-9)
        fd = (p.pos(ell[1:-1] + 1e-4) - p.pos(ell[1:-1] - 1e-4)) / 2e-4
        self.assertLess(float((fd - tau[1:-1]).norm(dim=1).max()), 2e-2)   # tangent = d p / d l


class TestProjection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.p = curved_path()

    def _poses(self, chart, B=64):
        ell = torch.rand(B, dtype=torch.float64) * (self.p.L - 0.2) + 0.1
        r = (torch.rand(B, 3, dtype=torch.float64) - 0.5) * 0.6
        x = torch.cat([self.p.chart_t(ell, chart), r], 1)
        ell2 = torch.rand(B, dtype=torch.float64) * self.p.L
        y = torch.cat([self.p.chart_t(ell2, chart), (torch.rand(B, 3, dtype=torch.float64) - 0.5) * 0.6], 1)
        return ell, x, y

    def _dT_dl(self, ell, x, y, chart, h=1e-5):
        """Derivative of T along the chart tangent tau at x (the direction M allows; dp / dl = tau is checked
        in TestPath)."""
        tau = self.p.chart_tangent(x[:, :3], chart)
        xp, xm = x.clone(), x.clone()
        xp[:, :3] = x[:, :3] + h * tau
        xm[:, :3] = x[:, :3] - h * tau
        return (smooth_T(xp, y) - smooth_T(xm, y)) / (2 * h)

    def test_chart_projection_is_the_path_derivative(self):
        for chart in ('arclength', 'world'):
            ell, x, y = self._poses(chart)
            xg = x.clone().requires_grad_(True)
            g = torch.autograd.grad(smooth_T(xg, y).sum(), xg)[0]
            gp = LS.project_chart(g, x, _Tangent(self.p, chart))
            fd = self._dT_dl(ell, x, y, chart)
            self.assertLess(float((gp[:, :3].norm(dim=1) - fd.abs()).abs().max()), 1e-4, chart)
            self.assertTrue(torch.equal(gp[:, 3:], g[:, 3:]))

    def test_lie_projection_is_the_path_derivative(self):
        """project_lie on the body-frame Lie gradient c_v = R^T dT/dt gives |dT/dl| as well."""
        for chart in ('arclength', 'world'):
            ell, x, y = self._poses(chart)
            xg = x.clone().requires_grad_(True)
            g = torch.autograd.grad(smooth_T(xg, y).sum(), xg)[0]
            R = G.so3_exp(TWO_PI * x[:, 3:])
            c = torch.cat([torch.einsum('bji,bj->bi', R, g[:, :3]), torch.randn(len(x), 3, dtype=x.dtype)], 1)
            cp = LS.project_lie(c, x, _Tangent(self.p, chart))
            fd = self._dT_dl(ell, x, y, chart)
            self.assertLess(float((cp[:, :3].norm(dim=1) - fd.abs()).abs().max()), 1e-4, chart)
            self.assertTrue(torch.equal(cp[:, 3:], c[:, 3:]))

    def test_free_cone_is_eikonal_on_the_unrolled_path(self):
        """In the arclength chart the free cone sqrt(dl^2 + (psi / 2 pi)^2) is the distance of M: the projected
        Lie gradient has unit norm at both ends (speed 1), the world-chart chord cone does not."""
        from models.graph_method_triangle.geometry import se3, tangents

        def cone(gx, gy):
            Rx, Ry = gx[:, 3:].reshape(-1, 3, 3), gy[:, 3:].reshape(-1, 3, 3)
            tr = (Rx * Ry).sum((1, 2))
            psi = torch.arccos(((tr - 1) / 2).clamp(-1 + 1e-12, 1 - 1e-12))
            return torch.sqrt(((gx[:, :3] - gy[:, :3]) ** 2).sum(1) + (psi / TWO_PI) ** 2)

        res = {}
        for chart in ('arclength', 'world'):
            _, x, y = self._poses(chart)
            gx = se3.features(x[:, :3], G.so3_exp(TWO_PI * x[:, 3:]))
            gy = se3.features(y[:, :3], G.so3_exp(TWO_PI * y[:, 3:]))
            T, cx, cy = tangents.tangent_gradients(cone, gx, gy, 1.0, TWO_PI, create_graph=False)
            tan = _Tangent(self.p, chart)
            nx = LS.project_lie(cx, x, tan).norm(dim=1)
            ny = LS.project_lie(cy, y, tan).norm(dim=1)
            res[chart] = (nx, ny)
        nx, ny = res['arclength']
        self.assertLess(float((nx - 1).abs().max()), 1e-6)
        self.assertLess(float((ny - 1).abs().max()), 1e-6)
        self.assertGreater(float((res['world'][0] - 1).abs().max()), 1e-2)


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = G.BoxScene(os.path.join(REPO, 'datasets/3dshape/obj/env1.obj'))
        cls.prism = G.Prism(0.02, 0.2)

    def test_scene(self):
        s = self.scene
        self.assertEqual(s.n_boxes, 4)
        np.testing.assert_allclose(s.room_hi, [0.475, 0.3329, 0.5], atol=2e-4)
        self.assertAlmostEqual(s.env_scale, 210.252, places=3)

    def test_clearance_matches_dense_sampling(self):
        rng = np.random.default_rng(1)
        n = 600
        t = torch.as_tensor(rng.uniform(self.scene.room_lo + 0.02, self.scene.room_hi - 0.02, (n, 3)))
        R = G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng)))
        c, s, e = G.prism_clearance(self.scene, self.prism, t, R, witness=True)
        u = np.linspace(-0.01, 0.01, 5)
        z = np.linspace(0, 0.2, 81)
        pts = np.array([[a, b, h] for a in u for b in u for h in z])
        W = t[:, None, :] + torch.einsum('bij,pj->bpi', R, torch.as_tensor(pts))
        bf = self.scene.point_clearance(W).min(1).values
        free = (c > 0) & (bf > 0)
        self.assertGreater(int(free.sum()), 200)
        self.assertGreaterEqual(float((bf[free] - c[free]).min()), -1e-9)    # exact <= any sampled bound
        self.assertLess(float((bf[free] - c[free]).max()), 2e-3)              # and close to it
        self.assertLess(float(((s - e).norm(dim=1)[free] - c[free]).abs().max()), 1e-9)
        agree = ((c < 0) == (bf < 0)).double().mean()
        self.assertGreater(float(agree), 0.99)

    def test_known_poses(self):
        s = self.scene
        # standing up in the middle of the room, far from the bars: clearance = distance to the nearest bar
        t = torch.tensor([[0.0, 0.05, -0.05]], dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)[None]
        c = float(G.prism_clearance(s, self.prism, t, R)[0])
        bf = float(s.point_clearance(t[:, None] + torch.tensor([[[0, 0, z]] for z in np.linspace(0, 0.2, 201)]
                                                                ).reshape(1, -1, 3).double()).min())
        self.assertAlmostEqual(c, bf - 0.01, delta=2e-3)
        # pointing through a bar: intersection
        t = torch.tensor([[0.0, -0.08, 0.05]], dtype=torch.float64)
        self.assertLess(float(G.prism_clearance(s, self.prism, t, R)[0]), 0)

    def test_angle_clearance_free_space(self):
        """No surface point within reach of the anchor: every shell is empty, the angle is pi."""
        p = Path(np.array([[0.0, 0.0, -0.4], [0.0, 0.0, -0.39]]))
        far = np.array([[1.0, 1.0, 1.0]], np.float32)
        ac = G.AngleClearance(far, self.prism, p.pts, 'cpu')
        a = ac(torch.tensor([[0.0, 0.0, -0.4]]), torch.eye(3)[None], torch.tensor([0]))
        self.assertAlmostEqual(float(a[0]), math.pi, places=5)

    def test_so3_log_exp(self):
        rng = np.random.default_rng(3)
        rv = torch.as_tensor(G.random_rotvecs(2000, rng))
        rv[:5] = rv[:5] / rv[:5].norm(dim=1, keepdim=True) * (math.pi - 1e-7)
        back = G.so3_log(G.so3_exp(rv))
        same = torch.minimum((back - rv).norm(dim=1), (back + rv).norm(dim=1) * (rv.norm(dim=1) > 3.14))
        self.assertLess(float(((G.so3_exp(back) - G.so3_exp(rv)).abs()).max()), 1e-6)
        self.assertLess(float(same[5:].max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
