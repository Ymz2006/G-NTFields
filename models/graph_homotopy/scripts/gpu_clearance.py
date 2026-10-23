"""Exact mesh clearance of a rigid body along paths, on the GPU.

The same quantity as the evaluators' CPU check (libigl signed distance of the scene's surface samples to the body
mesh, fast-winding-number sign, capped at CLEAR_CAP, min over the samples within shape_radius + CLEAR_CAP of the
body centre): per waypoint, the signed distance of the nearest scene sample to the body, negative = in collision.
Point-triangle distance (Ericson) and the generalized winding number (Van Oosterom-Strackee solid angles) are
batched over (sample, triangle) pairs; all paths of a call are checked in one go.
"""
import math

import numpy as np
import torch


def _so3_exp(r):
    """Rodrigues, (M, 3) rotation vectors -> (M, 3, 3)."""
    th = r.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = r / th
    K = torch.zeros(r.shape[0], 3, 3, dtype=r.dtype, device=r.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, dtype=r.dtype, device=r.device)[None]
    return I + s * K + (1 - c) * (K @ K)


def _point_tri_dist2(p, a, b, c):
    """squared distance of points p (N, 1, 3) to triangles a, b, c (1, T, 3) -> (N, T)  [Ericson, RTCD 5.1.5]."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    denom = 1.0 / (va + vb + vc).where((va + vb + vc).abs() > 1e-300, torch.full_like(va, 1e-300))
    v, w = vb * denom, vc * denom
    q = a + ab * v[..., None] + ac * w[..., None]                                   # interior
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(1e-300)
    r6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    q = torch.where(r6[..., None], b + (c - b) * t[..., None], q)
    t = d2 / (d2 - d6).where((d2 - d6).abs() > 1e-300, torch.full_like(d2, 1e-300))
    r5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    q = torch.where(r5[..., None], a + ac * t[..., None], q)
    r4 = (d6 >= 0) & (d5 <= d6)
    q = torch.where(r4[..., None], c.expand_as(q), q)
    t = d1 / (d1 - d3).where((d1 - d3).abs() > 1e-300, torch.full_like(d1, 1e-300))
    r3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    q = torch.where(r3[..., None], a + ab * t[..., None], q)
    r2 = (d3 >= 0) & (d4 <= d3)
    q = torch.where(r2[..., None], b.expand_as(q), q)
    r1 = (d1 <= 0) & (d2 <= 0)
    q = torch.where(r1[..., None], a.expand_as(q), q)
    return ((p - q) ** 2).sum(-1)


def _winding(p, a, b, c):
    """generalized winding number of the closed triangle soup at points p (N, 1, 3) -> (N,)."""
    A, B, C = a - p, b - p, c - p
    la, lb, lc = A.norm(dim=-1), B.norm(dim=-1), C.norm(dim=-1)
    num = (A * torch.cross(B, C, dim=-1)).sum(-1)
    den = la * lb * lc + (A * B).sum(-1) * lc + (B * C).sum(-1) * la + (C * A).sum(-1) * lb
    return torch.atan2(num, den).sum(-1) / (2.0 * math.pi)


class GpuClearance:
    def __init__(self, env_pts, shape_V, shape_F, radius, cap, device, dtype=torch.float64, pair_chunk=30000, wp_chunk=256):
        self.env = torch.as_tensor(np.asarray(env_pts), dtype=dtype, device=device)
        tri = torch.as_tensor(np.asarray(shape_V)[np.asarray(shape_F)], dtype=dtype, device=device)      # (T, 3, 3)
        self.a, self.b, self.c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
        self.lo, self.hi = tri.reshape(-1, 3).min(0).values, tri.reshape(-1, 3).max(0).values   # body box (body frame)
        self.radius, self.cap, self.device, self.dtype, self.chunk, self.wp_chunk = float(radius), float(cap), device, dtype, pair_chunk, wp_chunk

    @torch.no_grad()
    def signed_distance_body(self, q):
        """signed distance of body-frame points q (N, 3) to the body mesh (negative inside)."""
        out = torch.empty(len(q), dtype=self.dtype, device=self.device)
        for i in range(0, len(q), self.chunk):
            p = q[i:i + self.chunk, None, :]
            d = _point_tri_dist2(p, self.a, self.b, self.c).min(1).values.clamp_min(0).sqrt()
            inside = _winding(p, self.a, self.b, self.c) > 0.5
            out[i:i + self.chunk] = torch.where(inside, -d, d)
        return out

    @torch.no_grad()
    def __call__(self, P):
        """(M, 6) chart poses (t, rotation vector / 2 pi) -> (M,) capped clearance."""
        P = torch.as_tensor(np.asarray(P, dtype=np.float64), dtype=self.dtype, device=self.device)
        M = len(P)
        out = torch.full((M,), self.cap, dtype=self.dtype, device=self.device)
        if M == 0:
            return out.cpu().numpy()
        t, R = P[:, :3], _so3_exp(2.0 * math.pi * P[:, 3:6])
        for i in range(0, M, self.wp_chunk):          # (wp_chunk x 50k) distance table, then its pairs
            near = torch.cdist(t[i:i + self.wp_chunk], self.env) < self.radius
            nz = near.nonzero()
            if len(nz) == 0:
                continue
            wi, pi = nz[:, 0] + i, nz[:, 1]
            q = torch.einsum('nji,nj->ni', R[wi], self.env[pi] - t[wi])            # R^T (p - t): body frame
            # a sample farther than cap from the body's box is farther than cap from the body: capped anyway
            keep = torch.maximum(self.lo - q, q - self.hi).clamp_min(0).norm(dim=-1) < self.cap
            if bool(keep.any()):
                out.scatter_reduce_(0, wi[keep], self.signed_distance_body(q[keep]), reduce='amin')
        return out.clamp_max(self.cap).cpu().numpy()

    def paths(self, paths):
        """list of (M_i, 6) -> list of (M_i,), one GPU pass."""
        lens = [len(p) for p in paths]
        if sum(lens) == 0:
            return [np.full(n, self.cap) for n in lens]
        flat = self(np.concatenate([np.asarray(p, np.float64).reshape(-1, 6) for p in paths], 0))
        return list(np.split(flat, np.cumsum(lens)[:-1]))
