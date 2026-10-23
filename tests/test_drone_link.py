"""Tests of models/drone_link: the exact geometry of the stretchable link, its 7-d normal and angular clearance.

    python -m unittest tests.test_drone_link -v        (in the container, repo root)
"""
import math
import os
import unittest

import numpy as np
import torch

from models.drone_link import geometry as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWO_PI = 2 * math.pi


def scene(name):
    return G.Scene(os.path.join(REPO, 'datasets', '3dshape', 'obj', name + '.obj'))


def random_poses(sc, link, n, rng, dtype=torch.float64):
    t = torch.as_tensor(rng.uniform(sc.room_lo, sc.room_hi, (n, 3)), dtype=dtype)
    R = G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng), dtype=dtype))
    L = torch.as_tensor(rng.uniform(link.L_min, link.L_max, n), dtype=dtype)
    return t, R, L


class TestClearance(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.link = G.Link(0.02, 0.06, 0.30)

    def test_fixed_length_is_the_prism(self):
        sc = scene('scaffold')
        t, R, _ = random_poses(sc, self.link, 2000, self.rng)
        for h in (0.08, 0.27):
            L = torch.full((2000,), h, dtype=torch.float64)
            a = G.scene_clearance(sc, self.link, t, R, L)
            b = G.PG.prism_clearance(sc, G.PG.Prism(0.02, h), t, R)
            self.assertLess(float((a - b).abs().max()), 1e-12)

    def test_dense_sampling_bound(self):
        """The exact clearance never exceeds the clearance of dense surface samples of the box, and is close to it."""
        sc = scene('windows')
        t, R, L = random_poses(sc, self.link, 1500, self.rng)
        c, s, e = G.scene_clearance(sc, self.link, t, R, L, witness=True)
        u = torch.linspace(-0.01, 0.01, 9, dtype=torch.float64)
        z = torch.linspace(0, 1, 241, dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(u, u, z, indexing='ij')
        P = torch.stack([gx, gy, gz], -1).reshape(-1, 3)
        P = P[(P[:, 0].abs() > 0.0099) | (P[:, 1].abs() > 0.0099) | (P[:, 2] < 1e-9) | (P[:, 2] > 1 - 1e-9)]
        Pl = P[None].repeat(len(t), 1, 1)
        Pl[..., 2] = Pl[..., 2] * L[:, None]
        dens = sc.point_clearance(t[:, None] + torch.einsum('bij,bnj->bni', R, Pl)).min(1).values
        free = c > 0
        self.assertTrue(bool((c[free] <= dens[free] + 1e-9).all()))
        self.assertLess(float((dens[free] - c[free]).max()), 1e-3)
        self.assertLessEqual(int(((c > 0) != (dens > 0)).sum()), 3)          # grazing contacts only
        self.assertLess(float(((s - e).norm(dim=1)[free] - c[free]).abs().max()), 1e-9)

    def test_length_limits(self):
        sc = scene('forest')
        t = torch.zeros(3, 3, dtype=torch.float64)
        t[:, 1] = 0.25                                       # above the crowns, link along +z (free for these L)
        R = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
        L = torch.tensor([0.05, 0.2, 0.31], dtype=torch.float64)
        c, cs, sl = G.clearance(sc, self.link, t, R, L, parts=True)
        self.assertTrue(bool((cs > 0).all()))
        np.testing.assert_allclose(sl.numpy(), [-0.01, 0.1, -0.01], atol=1e-12)
        np.testing.assert_allclose(c.numpy(), np.minimum(cs.numpy(), sl.numpy()))


class TestNormal(unittest.TestCase):
    def test_normal_is_the_clearance_gradient(self):
        """normal7 against central differences of the scene clearance: translation and length in the chart,
        rotation as a world-frame perturbation exp(2 pi h e_k) R (preprocess_obj's moment convention)."""
        rng = np.random.default_rng(1)
        sc = scene('scaffold')
        link = G.Link(0.02, 0.06, 0.30)
        t, R, L = random_poses(sc, link, 4000, rng)
        c, s, e = G.scene_clearance(sc, link, t, R, L, witness=True)
        ok = (c > 0.005) & (link.slack(L) > c + 0.01)
        t, R, L, c, s, e = t[ok], R[ok], L[ok], c[ok], s[ok], e[ok]
        n = G.normal7(link, t, R, L, s, e, torch.zeros(len(t), dtype=torch.long))
        h = 1e-6
        g = torch.zeros(len(t), 7, dtype=torch.float64)
        for k in range(3):
            d = torch.zeros(3, dtype=torch.float64)
            d[k] = h
            g[:, k] = (G.scene_clearance(sc, link, t + d, R, L) - G.scene_clearance(sc, link, t - d, R, L)) / (2 * h)
            Rp, Rm = G.so3_exp(TWO_PI * d)[None] @ R, G.so3_exp(-TWO_PI * d)[None] @ R
            g[:, 3 + k] = (G.scene_clearance(sc, link, t, Rp, L) - G.scene_clearance(sc, link, t, Rm, L)) / (2 * h)
        g[:, 6] = (G.scene_clearance(sc, link, t, R, L + h) - G.scene_clearance(sc, link, t, R, L - h)) / (2 * h)
        gu = g / g.norm(dim=1, keepdim=True)
        cos = (gu * n).sum(1)
        # the clearance is smooth where the witness pair is unique; ties (a face parallel to a face) are rare
        self.assertGreater(float((cos > 0.999).double().mean()), 0.97)
        self.assertGreater(float((n[:, 6].abs() > 1e-3).double().mean()), 0.05)   # the length matters

    def test_length_bound_normal(self):
        link = G.Link(0.02, 0.06, 0.30)
        z = torch.zeros(2, 3, dtype=torch.float64)
        n = G.normal7(link, z, torch.eye(3, dtype=torch.float64).expand(2, 3, 3), torch.tensor([0.07, 0.29]),
                      z + 1.0, z, torch.tensor([-1, 1]))
        np.testing.assert_allclose(n.numpy(), [[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, -1.0]])


class TestAngle(unittest.TestCase):
    def test_against_brute_force(self):
        """LinkAngle equals the definition computed point by point (same link and scene samples)."""
        rng = np.random.default_rng(2)
        sc = scene('forest')
        link = G.Link(0.02, 0.06, 0.30)
        env = sc.surface_points(0.02, seed=0)
        ang = G.LinkAngle(env, 0.02, 'cpu', num=200, bins=5, seed=3)
        t, R, L = random_poses(sc, link, 12, rng, dtype=torch.float32)
        got = ang(t, R, L).numpy()
        P = ang.shape_points(L).numpy()
        for b in range(len(t)):
            p = P[b]
            r = np.linalg.norm(p, axis=1)
            o = np.argsort(r, kind='stable')
            p, r = p[o], r[o]
            per = len(p) // 5
            edges = list(r[::per][:5]) + [r[-1]]
            best = -1.0
            ev = env - t[b].numpy()
            d = np.linalg.norm(ev, axis=1)
            su = (R[b].numpy() @ p.T).T
            su /= np.linalg.norm(su, axis=1, keepdims=True)
            for i in range(5):
                sel = (d > edges[i]) & (d <= edges[i + 1])
                if not sel.any():
                    continue
                eu = ev[sel] / d[sel, None]
                best = max(best, float((su[i * per:(i + 1) * per] @ eu.T).max()))
            self.assertAlmostEqual(float(got[b]), math.acos(max(-1.0, min(1.0, best))), delta=2e-3)

    def test_shape_points_on_the_surface(self):
        ang = G.LinkAngle(np.zeros((1, 3), np.float32), 0.02, 'cpu', num=1000)
        L = torch.tensor([0.06, 0.3])
        P = ang.shape_points(L)
        for b in range(2):
            p = P[b]
            on_side = (p[:, 0].abs() - 0.01).abs() < 1e-6
            on_side |= (p[:, 1].abs() - 0.01).abs() < 1e-6
            on_cap = (p[:, 2].abs() < 1e-6) | ((p[:, 2] - L[b]).abs() < 1e-6)
            self.assertTrue(bool((on_side | on_cap).all()))
            self.assertTrue(bool(((p[:, :2].abs() <= 0.01 + 1e-6).all(1) & (p[:, 2] >= -1e-6)
                                  & (p[:, 2] <= L[b] + 1e-6)).all()))
            caps = float(on_cap.double().mean())
            expect = 2 * 0.02 ** 2 / (2 * 0.02 ** 2 + 4 * 0.02 * float(L[b]))
            self.assertAlmostEqual(caps, expect, delta=0.03)


class TestChart(unittest.TestCase):
    def test_round_trip_and_distance(self):
        rng = np.random.default_rng(4)
        rv = G.random_rotvecs(100, rng)
        x = torch.as_tensor(G.to_chart(rng.uniform(-0.4, 0.4, (100, 3)), rv, rng.uniform(0.06, 0.3, 100)))
        t, R, L = G.from_chart(x)
        back = G.so3_log(R) / TWO_PI
        self.assertLess(float((back - x[:, 3:6]).abs().max()), 1e-9)
        self.assertLess(float(G.pose_dist(x, x).max()), 1e-6)
        y = x.clone()
        y[:, 6] += 0.05
        np.testing.assert_allclose(G.pose_dist(x, y).numpy(), 0.05, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
