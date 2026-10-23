"""Tests of models/constrained_movement: the exact pyramid and shape clearances, the scenes.

    python -m unittest tests.test_constrained_movement -v        (in the container, repo root)
"""
import math
import os
import unittest

import numpy as np
import torch

from models.constrained_movement import geometry as G

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = os.path.join(REPO, 'datasets', '3dshape', 'obj')


def tri_bruteforce(p, a, b, c, n=400):
    """Closest point of one triangle to p by dense barycentric sampling (float64)."""
    u = torch.linspace(0, 1, n, dtype=torch.float64)
    U, V = torch.meshgrid(u, u, indexing='ij')
    m = (U + V) <= 1
    pts = a + (b - a) * U[m][:, None] + (c - a) * V[m][:, None]
    return float((pts - p).norm(dim=1).min())


def pyramid_samples(pyr, R, t, spacing=0.004, interior=6):
    """Dense surface points and interior points (scaled copies of the surface towards the centroid)."""
    S = torch.as_tensor(pyr.surface_points(spacing), dtype=torch.float64)
    cen = S.mean(0)
    inner = [cen + (S - cen) * f for f in np.linspace(0.0, 1.0, interior + 1)[:-1]]
    return t + S @ R.T, t + torch.cat(inner) @ R.T


class TestPyramid(unittest.TestCase):
    def test_closest_point_triangle(self):
        g = torch.Generator().manual_seed(0)
        for _ in range(40):
            a, b, c, p = (torch.rand(3, generator=g, dtype=torch.float64) * 2 - 1 for _ in range(4))
            q = G.closest_point_triangle(p, a, b, c)
            self.assertAlmostEqual(float((p - q).norm()), tri_bruteforce(p, a, b, c), delta=4e-3)

    def test_shape(self):
        pyr = G.Pyramid(0.2, 0.1, 0.25)
        V, F = pyr.mesh()
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0          # divergence theorem, outward
        self.assertAlmostEqual(vol, 0.2 * 0.1 * 0.25 / 3, places=12)
        self.assertEqual(pyr.normals.shape, (5, 3))
        self.assertAlmostEqual(pyr.fov_deg[0], 2 * math.degrees(math.atan(0.1 / 0.25)), places=9)


class TestClearance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = {s: G.Scene(os.path.join(OBJ, s + '.obj')) for s in ('forest', 'city', 'env4')}

    def test_scenes(self):
        self.assertEqual(self.scenes['forest'].n_boxes, 8)
        self.assertEqual(self.scenes['city'].n_boxes, 6)
        f = self.scenes['forest']
        self.assertTrue(np.allclose(f.boxes_raw[:, 1] - f.boxes_raw[:, 0], [12, 30] * 4, atol=1e-6) or
                        np.allclose(np.sort(f.boxes_raw[:, 1] - f.boxes_raw[:, 0]), [12] * 4 + [30] * 4, atol=1e-6))

    def test_pyramid_matches_dense_sampling(self):
        """Exact clearance vs the sampled distance of dense pyramid points: never above the samples, close to
        them, the same sign (interior samples catch a box inside the pyramid)."""
        rng = np.random.default_rng(0)
        for name, sc in self.scenes.items():
            pyr = G.default_pyramid(sc.env_scale)
            n = 120
            t = torch.as_tensor(rng.uniform(sc.room_lo + 0.05, sc.room_hi - 0.05, (n, 3)))
            R = G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng)))
            c = G.pyramid_clearance(sc, pyr, t, R)
            n_neg = 0
            for i in range(n):
                surf, inner = pyramid_samples(pyr, R[i], t[i])
                ds = float(sc.point_clearance(surf).min())
                di = float(sc.point_clearance(inner).min())
                if c[i] > 0:
                    self.assertGreater(di, 0, name)
                    self.assertLessEqual(float(c[i]), ds + 1e-9, name)
                    self.assertLess(ds - float(c[i]), 4e-3, name)
                else:
                    n_neg += 1
                    self.assertLess(min(ds, di), 4e-3, name)
            self.assertGreater(n_neg, 5, name)

    def test_witness_and_translation_lipschitz(self):
        sc = self.scenes['forest']
        pyr = G.default_pyramid(sc.env_scale)
        rng = np.random.default_rng(1)
        n = 400
        t = torch.as_tensor(rng.uniform(sc.room_lo + 0.05, sc.room_hi - 0.05, (n, 3)))
        R = G.so3_exp(torch.as_tensor(G.random_rotvecs(n, rng)))
        c, s, e = G.pyramid_clearance(sc, pyr, t, R, witness=True)
        pos = c > 1e-6
        self.assertTrue(torch.allclose((s - e).norm(dim=1)[pos], c[pos], atol=1e-9))
        self.assertLess(float(sc.point_clearance(e[pos]).abs().max()), 1e-9)       # e* on the scene boundary
        # a translation by d changes the clearance by at most |d| (contact crossings included)
        d = torch.as_tensor(rng.normal(size=(n, 3))) * 1e-3
        c2 = G.pyramid_clearance(sc, pyr, t + d, R)
        self.assertLessEqual(float(((c2 - c).abs() - d.norm(dim=1)).max()), 1e-9)

    def test_shape_clearance(self):
        rng = np.random.default_rng(2)
        for name, sc in self.scenes.items():
            shp = G.Shape(G.BIRD, sc.env_scale)
            c = torch.as_tensor(rng.uniform(sc.room_lo, sc.room_hi, (300, 3)))
            d = G.Shape.clearance(shp, sc, c)
            for i in range(0, 300, 7):
                pts = []
                for lo, hi in shp.boxes(c[i].numpy()):
                    g = [np.linspace(lo[k], hi[k], 25) for k in range(3)]
                    P = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
                    pts.append(P)
                ds = float(sc.point_clearance(torch.as_tensor(np.concatenate(pts))).min())
                if d[i] > 0:
                    self.assertLessEqual(float(d[i]), ds + 1e-9)
                    self.assertLess(ds - float(d[i]), 3e-3)
                else:
                    self.assertLess(ds, 3e-3)


if __name__ == '__main__':
    unittest.main()
