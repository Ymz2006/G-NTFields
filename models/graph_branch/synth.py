"""Synthetic SE(2)-in-SE(3) problems with exact travel times, for testing where a decoder can put its ridges
(property 8) and how large a travel time it can express (property 9).

Poses x = (t, R) with t = (t1, t2, 0) and R = Rz(psi) (yaw); the network sees the usual SE(3) embedding
G(x) = (t, vec R) and chart rows (t, rotvec / 2 pi). The metric is a product of a planar translation part and
a yaw part, so the exact travel time is the l2 combination of the two distances (product of length spaces):

    T*(x, y) = sqrt(D_t(t_x, t_y)^2 + D_r(psi_x, psi_y)^2)

D_r  yaw distance with conformal factor h(psi) = 1 + A exp(-(psi - psi0)^2 / (2 w^2)) (a slow-rotation sector;
     rotation speed S_a = 1 / h), in time units with s_r = 2 pi:
         H(psi) = int_0^psi h,  H_ccw = (H(psi_y) - H(psi_x)) mod H_tot,  D_r = min(H_ccw, H_tot - H_ccw) / (2 pi).
     The ridge (the two yaw routes tie) is where H_ccw = H_tot / 2: away from the half turn unless psi_x is
     symmetric to the sector, and D_r reaches H_tot / (4 pi) > 1/2, graph_gram's cap on its rotation term.
D_t  planar Euclidean distance, or the geodesic distance around a hard disc of radius r0 at the origin
     (tangent segments + arc, closed form); the ridge is the ray behind the disc opposite to the start.

Kinds: 'sector' (A > 0, no disc), 'disc' (A = 0, disc), 'both'.
"""
import math

import numpy as np
import torch

TWO_PI = 2.0 * math.pi


class Problem:
    def __init__(self, kind='sector', A=4.0, w=0.5, psi0=math.pi, r0=0.15, half=0.5):
        assert kind in ('sector', 'disc', 'both', 'free')
        self.kind = kind
        self.A = A if kind in ('sector', 'both') else 0.0
        self.w, self.psi0 = w, psi0
        self.r0 = r0 if kind in ('disc', 'both') else 0.0
        self.half = half
        self.H_tot = float(self.H(torch.tensor([TWO_PI], dtype=torch.float64))[0])

    # ------------------------------------------------------------------ rotation part
    def h(self, psi):
        d = torch.remainder(psi - self.psi0 + math.pi, TWO_PI) - math.pi
        return 1.0 + self.A * torch.exp(-d ** 2 / (2 * self.w ** 2))

    def H(self, psi):
        """int_0^psi h for psi in [0, 2 pi] (the bump is centred inside (0, 2 pi) with negligible tails)."""
        c = self.w * math.sqrt(math.pi / 2.0)
        s = math.sqrt(2.0) * self.w
        return psi + self.A * c * (torch.erf((psi - self.psi0) / s) - math.erf((0.0 - self.psi0) / s))

    def D_r(self, px, py):
        px = torch.remainder(px, TWO_PI)
        py = torch.remainder(py, TWO_PI)
        ccw = torch.remainder(self.H(py) - self.H(px), self.H_tot)
        return torch.minimum(ccw, self.H_tot - ccw) / TWO_PI

    def ridge_psi(self, px, n=20001):
        """Yaw of the rotational ridge for start yaw px (the goal yaw where the two routes tie), by bisection on H."""
        target = torch.remainder(self.H(torch.remainder(px, TWO_PI)) + 0.5 * self.H_tot, self.H_tot)
        lo = torch.zeros_like(px)
        hi = torch.full_like(px, TWO_PI)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            go = self.H(mid) < target
            lo = torch.where(go, mid, lo)
            hi = torch.where(go, hi, mid)
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ translation part
    def D_t(self, p, q):
        d = (p - q).norm(dim=-1)
        if self.r0 <= 0:
            return d
        r0 = self.r0
        # does the segment p -> q pass through the open disc?
        pq = q - p
        L2 = (pq ** 2).sum(-1).clamp_min(1e-300)
        s = (-(p * pq).sum(-1) / L2).clamp(0, 1)
        closest = (p + s[..., None] * pq).norm(dim=-1)
        blocked = closest < r0
        rp, rq = p.norm(dim=-1).clamp_min(r0), q.norm(dim=-1).clamp_min(r0)
        cosg = ((p * q).sum(-1) / (rp * rq)).clamp(-1, 1)
        gam = torch.arccos(cosg)
        wrap = torch.sqrt(rp ** 2 - r0 ** 2) + torch.sqrt(rq ** 2 - r0 ** 2) \
            + r0 * (gam - torch.arccos(r0 / rp) - torch.arccos(r0 / rq))
        return torch.where(blocked, wrap, d)

    # ------------------------------------------------------------------ sampling
    def sample_t(self, n, g):
        out = torch.empty(0, 2, dtype=torch.float64)
        while out.shape[0] < n:
            c = (torch.rand(2 * n, 2, generator=g, dtype=torch.float64) * 2 - 1) * self.half
            if self.r0 > 0:
                c = c[c.norm(dim=1) > self.r0 + 0.01]
            out = torch.cat([out, c], 0)
        return out[:n]

    def sample_pairs(self, n, g, mix=(0.5, 0.25, 0.25)):
        """(n, 3) start (t1, t2, psi) and goal rows: uniform pairs, near-ridge pairs, local pairs."""
        n_u, n_r = int(mix[0] * n), int(mix[1] * n)
        n_l = n - n_u - n_r
        p0 = torch.cat([self.sample_t(n, g), torch.rand(n, 1, generator=g, dtype=torch.float64) * TWO_PI], 1)
        p1 = torch.cat([self.sample_t(n, g), torch.rand(n, 1, generator=g, dtype=torch.float64) * TWO_PI], 1)
        # near-ridge: goal yaw within +-0.3 rad of the start's rotational ridge; with a disc, goal behind it
        i = slice(n_u, n_u + n_r)
        rp = self.ridge_psi(p0[i, 2]) if self.A > 0 else p0[i, 2] + math.pi
        p1[i, 2] = rp + (torch.rand(n_r, generator=g, dtype=torch.float64) - 0.5) * 0.6
        if self.r0 > 0:
            u = -p0[i, :2] / p0[i, :2].norm(dim=1, keepdim=True)
            perp = torch.stack([-u[:, 1], u[:, 0]], 1)
            rad = self.r0 + 0.02 + torch.rand(n_r, 1, generator=g, dtype=torch.float64) * (self.half - self.r0)
            off = (torch.rand(n_r, 1, generator=g, dtype=torch.float64) - 0.5) * 0.2
            q = u * rad + perp * off
            ok = (q.abs() <= self.half).all(1) & (q.norm(dim=1) > self.r0 + 0.01)
            p1[i, :2] = torch.where(ok[:, None], q, p1[i, :2])
        # local pairs
        j = slice(n_u + n_r, n)
        loc = torch.cat([(torch.rand(n_l, 2, generator=g, dtype=torch.float64) - 0.5) * 0.3,
                         torch.randn(n_l, 1, generator=g, dtype=torch.float64) * 0.5], 1)
        q = p0[j] + loc
        ok = (q[:, :2].abs() <= self.half).all(1)
        if self.r0 > 0:
            ok = ok & (q[:, :2].norm(dim=1) > self.r0 + 0.01)
        p1[j] = torch.where(ok[:, None], q, p1[j])
        p1[:, 2] = torch.remainder(p1[:, 2], TWO_PI)
        return p0, p1

    def T_star(self, p0, p1):
        return torch.sqrt(self.D_t(p0[:, :2], p1[:, :2]) ** 2 + self.D_r(p0[:, 2], p1[:, 2]) ** 2)

    def T_free(self, p0, p1):
        th = torch.remainder(p1[:, 2] - p0[:, 2] + math.pi, TWO_PI) - math.pi
        return torch.sqrt(((p0[:, :2] - p1[:, :2]) ** 2).sum(1) + (th / TWO_PI) ** 2)


def to_chart(p):
    """(N, 3) planar rows -> (N, 6) chart rows (t1, t2, 0, 0, 0, psi_canonical / 2 pi)."""
    psi = torch.remainder(p[:, 2] + math.pi, TWO_PI) - math.pi
    z = torch.zeros_like(psi)
    return torch.stack([p[:, 0], p[:, 1], z, z, z, psi / TWO_PI], 1)


def to_features(p):
    """(N, 3) planar rows -> (N, 12) G = (t, vec Rz(psi))."""
    c, s = torch.cos(p[:, 2]), torch.sin(p[:, 2])
    z, o = torch.zeros_like(c), torch.ones_like(c)
    R = torch.stack([c, -s, z, s, c, z, z, z, o], 1)
    return torch.cat([p[:, 0:1], p[:, 1:2], z[:, None], R], 1)
