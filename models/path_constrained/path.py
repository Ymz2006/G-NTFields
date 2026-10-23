"""The point path the prism's anchor is glued to, and the two charts of the constrained pose space.

The path is stored as dense samples p_k at arc lengths l_k (spacing ~1e-3, l_0 = 0, l_n = L) with unit tangents
tau_k. The anchored prism's configuration space is M = [0, L] x SO(3), a 4-d submanifold of SE(3) with the induced
product metric dl^2 + |omega / 2 pi|^2 (the metric of every SE(3) cell: translation in normalized units, rotation in
turns). Two charts put M into the 6-d (t, rotvec / 2 pi) layout the models read:

    arclength   t = (l - L / 2, 0, 0): the path unrolled onto the x axis. M is then literally a slice of SE(3)
                (the poses whose translation lies on a segment of the x axis) with the same metric, so the
                Euclidean translation distance of the chart is the arc length.
    world       t = p(l): the real anchor position (the path embedded in the room).

In both charts the tangent space of M at a pose is span(tau) + so(3), tau = e_x (arclength) or tau(l) (world).
"""
import json
import math

import numpy as np
import torch

CHARTS = ('arclength', 'world')


class Path:
    def __init__(self, pts):
        pts = np.asarray(pts, np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-12])
        pts = pts[keep]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.pts = pts
        self.ell = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = float(self.ell[-1])
        tan = np.gradient(pts, self.ell, axis=0)
        self.tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
        self._dev = {}

    # ------------------------------------------------------------------ io
    def save(self, fname, **extra):
        np.savez(fname, pts=self.pts, ell=self.ell, tan=self.tan, **extra)

    @classmethod
    def load(cls, fname):
        z = np.load(fname, allow_pickle=True)
        p = cls(z['pts'])
        p.extra = {k: z[k] for k in z.files if k not in ('pts', 'ell', 'tan')}
        return p

    # ------------------------------------------------------------------ torch evaluation
    def tensors(self, device, dtype=torch.float32):
        key = (str(device), dtype)
        if key not in self._dev:
            f = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
            self._dev[key] = dict(pts=f(self.pts), ell=f(self.ell), tan=f(self.tan))
        return self._dev[key]

    def _interp(self, ell, which):
        T = self.tensors(ell.device, ell.dtype)
        e = ell.clamp(0.0, self.L)
        i = torch.searchsorted(T['ell'], e.contiguous(), right=True).clamp(1, len(self.ell) - 1)
        e0, e1 = T['ell'][i - 1], T['ell'][i]
        w = ((e - e0) / (e1 - e0).clamp_min(1e-12))[..., None]
        v = T[which][i - 1] * (1 - w) + T[which][i] * w
        return v, i

    def pos(self, ell):
        return self._interp(ell, 'pts')[0]

    def tangent(self, ell):
        v, _ = self._interp(ell, 'tan')
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def sample_index(self, ell):
        """Nearest dense-sample index of arc lengths (for per-sample precomputations)."""
        T = self.tensors(ell.device, ell.dtype)
        i = torch.searchsorted(T['ell'], ell.clamp(0, self.L).contiguous()).clamp(0, len(self.ell) - 1)
        im = (i - 1).clamp(min=0)
        closer = (ell - T['ell'][im]).abs() < (T['ell'][i] - ell).abs()
        return torch.where(closer, im, i)

    def nearest_ell(self, p):
        """Arc length of the path point nearest to positions p (N, 3) (world chart -> l); projects onto the
        nearest segment, so it is exact for points on the path."""
        T = self.tensors(p.device, p.dtype)
        k = torch.cdist(p, T['pts']).argmin(-1)
        best = None
        for j in (k - 1, k):
            j = j.clamp(0, len(self.ell) - 2)
            a, b = T['pts'][j], T['pts'][j + 1]
            ab = b - a
            u = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-18)).clamp(0, 1)
            q = a + u[:, None] * ab
            d = (p - q).norm(dim=-1)
            e = T['ell'][j] + u * (T['ell'][j + 1] - T['ell'][j])
            if best is None:
                best = (d, e)
            else:
                m = d < best[0]
                best = (torch.where(m, d, best[0]), torch.where(m, e, best[1]))
        return best[1]

    # ------------------------------------------------------------------ charts
    def chart_t(self, ell, chart):
        """Chart translation (N, 3) of arc lengths ell (N,)."""
        if chart == 'arclength':
            t = torch.zeros(*ell.shape, 3, dtype=ell.dtype, device=ell.device)
            t[..., 0] = ell - 0.5 * self.L
            return t
        if chart == 'world':
            return self.pos(ell)
        raise ValueError(chart)

    def chart_ell(self, t, chart):
        """Arc length of chart translations t (N, 3)."""
        if chart == 'arclength':
            return t[..., 0] + 0.5 * self.L
        if chart == 'world':
            return self.nearest_ell(t)
        raise ValueError(chart)

    def chart_tangent(self, t, chart):
        """Unit tangent of M's translation direction in chart coordinates at chart translations t (N, 3)."""
        if chart == 'arclength':
            tau = torch.zeros_like(t)
            tau[..., 0] = 1.0
            return tau
        return self.tangent(self.nearest_ell(t))


def load_constraint(data_dir):
    """(meta, Path, chart) of a path-constrained cell or test set directory."""
    import os
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    c = meta['constraint']
    return meta, Path.load(os.path.join(data_dir, c['path_file'])), c['chart']


def max_turn_rate(path):
    """Largest curvature (1 / radius) of the dense samples."""
    dt = np.linalg.norm(np.diff(path.tan, axis=0), axis=1)
    ds = np.diff(path.ell)
    return float((dt / np.maximum(ds, 1e-12)).max())


def smooth_curve(via, n=4000, alpha=0.5):
    """Centripetal Catmull-Rom spline through via points (m, 3), the end tangents extrapolated; (n, 3) samples."""
    P = np.asarray(via, np.float64)
    P = np.concatenate([2 * P[:1] - P[1:2], P, 2 * P[-1:] - P[-2:-1]])
    out = []
    segs = len(P) - 3
    per = max(2, n // segs)
    for i in range(segs):
        p0, p1, p2, p3 = P[i:i + 4]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        ts = np.linspace(t1, t2, per, endpoint=(i == segs - 1))[:, None]
        A1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        A2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        A3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        B1 = (t2 - ts) / (t2 - t0) * A1 + (ts - t0) / (t2 - t0) * A2
        B2 = (t3 - ts) / (t3 - t1) * A2 + (ts - t1) / (t3 - t1) * A3
        out.append((t2 - ts) / (t2 - t1) * B1 + (ts - t1) / (t2 - t1) * B2)
    return np.concatenate(out)


def resample(pts, spacing):
    """Uniform arc-length resampling of a polyline (n, 3) at about ``spacing``."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    ell = np.concatenate([[0.0], np.cumsum(seg)])
    m = max(2, int(math.ceil(ell[-1] / spacing)) + 1)
    e = np.linspace(0, ell[-1], m)
    return np.stack([np.interp(e, ell, pts[:, k]) for k in range(3)], 1)
