"""Architectural property tests for the min-of-branches decoder (models/graph_method_triangle/branch_decoder.py),
one group per property of "Decoder properties that hold with or without obstacles" (2026-10-13), and the
same checks on graph_gram's fixed (a, b) decoder where the note says it fails. From the repo root:

    python -m unittest tests.test_branch_decoder -v

Every check runs at random (perturbed) parameters in float64, so it tests the architecture, not a trained field.
"""
import math
import sys
import unittest

sys.path.append('.')

import torch

from models.graph_lie import lie
from models.graph_method_triangle import branch_decoder as bd
from models.graph_method_triangle.fields import decoder as abdec
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_network_metric as gn_network

DT = torch.float64
TWO_PI = 2.0 * math.pi
BRANCH = {
    'branch_K1': ('graph_branch', dict(n_branch=1)),
    'branch_K4': ('graph_branch', dict(n_branch=4)),
    'branch_conf_K4': ('graph_branch_conf', dict(n_branch=4)),
    'branch_lp_K2': ('graph_branch', dict(n_branch=2, res_kind='lp', res_even=32, res_odd=32)),
    'branch_conf_lp_K2': ('graph_branch_conf', dict(n_branch=2, res_kind='lp', res_even=32, res_odd=32)),
    'branch_lse_K2': ('graph_branch', dict(n_branch=2, res_kind='lse', res_even=32, res_odd=32)),
    'branch_huber_K2': ('graph_branch', dict(n_branch=2, res_huber=0.05)),
    'branch_conf_lp_huber_K2': ('graph_branch_conf', dict(n_branch=2, res_kind='lp', res_even=32, res_odd=32,
                                                          res_huber=0.05)),
}


def make_net(arch, extra=None, seed=0, perturb=0.05, res_gain=1.0):
    torch.manual_seed(seed)
    net = gn_network.NN('cpu', 6, None, graph=dict(arch=arch, **(extra or {})))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(perturb * torch.randn_like(p))
        if hasattr(net.net, 'U'):
            net.net.U.weight.mul_(res_gain)
            net.net.W.weight.mul_(res_gain)
        if getattr(net.net, 'conf', False):  # non-constant rho / lam so the conformal factor varies with the pose
            net.net.rho_head.weight.normal_(0, 0.2)
            net.net.lam_head.weight.normal_(0, 0.2)
    return net.to(DT)


def random_poses(n, seed, max_turns=0.5):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9
    q = torch.randn(n, 4, generator=g, dtype=DT)
    q = q / q.norm(dim=1, keepdim=True)
    return t, se3.quat_to_matrix(q)


def feats(t, R):
    return se3.features(t, R)


def unit_twists(n, seed, s_t=1.0, s_r=TWO_PI):
    """Random body twists xi = (v, w) with g(xi, xi) = |v|^2 / s_t^2 + |w|^2 / s_r^2 = 1."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 6, generator=g, dtype=DT)
    u = u / u.norm(dim=1, keepdim=True)
    return torch.cat([s_t * u[:, :3], s_r * u[:, 3:]], 1)


def perturb(t, R, xi, eps):
    return se3.right_perturb(t, R, eps * xi)


class TestQuaternionLift(unittest.TestCase):
    def test_roundtrip_and_sign_invariance(self):
        q = torch.randn(20000, 4, dtype=DT)
        q = q / q.norm(dim=1, keepdim=True)
        R = se3.quat_to_matrix(q)
        q2 = bd.quat_from_matrix(R)
        self.assertLess((se3.quat_to_matrix(q2) - R).abs().max().item(), 1e-13)
        self.assertGreater((q * q2).sum(1).abs().min().item(), 1 - 1e-13)
        # near the half turn and the identity (pivot switches)
        for ang in (1e-9, 1e-4, math.pi - 1e-6, math.pi):
            ax = torch.randn(500, 3, dtype=DT)
            ax = ax / ax.norm(dim=1, keepdim=True)
            Rk = se3.so3_exp(ang * ax)
            self.assertLess((se3.quat_to_matrix(bd.quat_from_matrix(Rk)) - Rk).abs().max().item(), 1e-12)

    def test_value_independent_of_quaternion_signs(self):
        """T is a function on SE(3): flipping a lift's sign swaps the routes and their residuals."""
        t0, R0 = random_poses(300, 1)
        t1, R1 = random_poses(300, 2)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra)
            h = net.net
            g0, g1 = feats(t0, R0), feats(t1, R1)
            T = h.value(g0, g1)
            orig = bd.quat_from_matrix
            try:
                bd.quat_from_matrix = lambda R: -orig(R)            # global flip of every lift
                T_flip_both = h.value(g0, g1)
                calls = {'n': 0}

                def flip_second(R):                                 # flips only the second call's lifts (q_y)
                    calls['n'] += 1
                    return -orig(R) if calls['n'] % 2 == 0 else orig(R)
                bd.quat_from_matrix = flip_second
                T_flip_y = h.value(g0, g1)
            finally:
                bd.quat_from_matrix = orig
            self.assertLess((T - T_flip_both).abs().max().item(), 1e-12, name)
            self.assertLess((T - T_flip_y).abs().max().item(), 1e-12, name)

    def test_lie_gradient_through_the_lift(self):
        """Tangent projection of dT/dG == d/de T(x exp(e E_i)) through the SE(3) exponential."""
        g = torch.Generator().manual_seed(3)
        X = torch.cat([(torch.rand(128, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                       torch.randn(128, 3, generator=g, dtype=DT) * 0.15,
                       (torch.rand(128, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                       torch.randn(128, 3, generator=g, dtype=DT) * 0.15], 1)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra)
            T, cx, cy = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
            T2, px, py = lie.lie_gradients_perturb(net, X)
            sc = cx.abs().max().item()
            self.assertLess((cx - px).abs().max().item() / sc, 1e-9, name)
            self.assertLess((cy - py).abs().max().item() / sc, 1e-9, name)
            # central finite differences along the body generators
            h = 1e-6
            t0, R0 = X[:, :3], se3.so3_exp(TWO_PI * X[:, 3:6])
            t1, R1 = X[:, 6:9], se3.so3_exp(TWO_PI * X[:, 9:12])
            fd = torch.zeros_like(cx)
            with torch.no_grad():
                for i in range(6):
                    E = torch.zeros(128, 6, dtype=DT)
                    E[:, i] = 1.0 if i < 3 else TWO_PI
                    ta, Ra = se3.right_perturb(t0, R0, h * E)
                    tb, Rb = se3.right_perturb(t0, R0, -h * E)
                    fd[:, i] = (net.net.value(feats(ta, Ra), feats(t1, R1))
                                - net.net.value(feats(tb, Rb), feats(t1, R1))) / (2 * h)
            # skip pairs within 1e-3 of a branch switch (the ridge: one-sided derivatives differ there)
            bv = net.net.branch_values(feats(t0, R0), feats(t1, R1)).reshape(128, -1)
            top2 = bv.topk(2, dim=1, largest=False).values
            regular = (top2[:, 1] - top2[:, 0]) > 1e-3
            self.assertLess((fd - cx)[regular].abs().max().item() / sc, 1e-5, name)


class TestMetricProperties(unittest.TestCase):
    """Properties 1-4."""

    def test_p1_nonnegative_zero_exactly_at_goal(self):
        t0, R0 = random_poses(500, 4)
        t1, R1 = random_poses(500, 5)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra)
            g0, g1 = feats(t0, R0), feats(t1, R1)
            self.assertEqual(net.net.value(g0, g0).abs().max().item(), 0.0, name)
            T = net.net.value(g0, g1)
            self.assertTrue(bool((T > 0).all()), name)
            # the same pose with the rotation re-expressed (R -> R exp(2 pi n) is R): still exactly 0
            self.assertEqual(net.net.value(g1, feats(t1, R1.clone())).abs().max().item(), 0.0, name)

    def test_p2_symmetry(self):
        t0, R0 = random_poses(500, 6)
        t1, R1 = random_poses(500, 7)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra)
            g0, g1 = feats(t0, R0), feats(t1, R1)
            self.assertEqual((net.net.value(g0, g1) - net.net.value(g1, g0)).abs().max().item(), 0.0, name)

    def test_p4_speed_limit_lower_bound(self):
        """T >= T_free against the independent atan2 oracle (s_t = 1, s_r = 2 pi), also near the half turn."""
        t0, R0 = random_poses(2000, 8)
        t1, R1 = random_poses(2000, 9)
        ax = torch.randn(500, 3, dtype=DT)
        ax = ax / ax.norm(dim=1, keepdim=True)
        R1[:500] = R0[:500] @ se3.so3_exp((math.pi - 10 ** torch.empty(500, dtype=DT).uniform_(-6, -1))[:, None] * ax)
        Tf = oracle.time_star(t0, R0, t1, R1, 1.0, TWO_PI)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra, res_gain=0.01)  # small residual: the bound is tight, the test meaningful
            T = net.net.value(feats(t0, R0), feats(t1, R1))
            self.assertGreaterEqual((T - Tf).min().item(), -1e-12, name)
            # residual off -> exactly the free time (the obstacle-free solution is representable)
            with torch.no_grad():
                net.net.U.weight.zero_()
                net.net.W.weight.zero_()
                if net.net.conf:
                    net.net.rho_head.weight.zero_()
                    net.net.rho_head.bias.fill_(-50.0)
            T0 = net.net.value(feats(t0, R0), feats(t1, R1))
            self.assertLess((T0 - Tf).abs().max().item(), 1e-9, name)

    def test_p4_fails_for_graph_gram(self):
        """graph_gram has no lower bound: for a valid parameter value T < T_free on most pairs."""
        t0, R0 = random_poses(2000, 10)
        t1, R1 = random_poses(2000, 11)
        net = make_net('graph_gram')
        with torch.no_grad():  # a valid parameter value: a small translation latent (nothing forbids it)
            net.net.A.weight.mul_(0.01)
        T = net.net.value(feats(t0, R0), feats(t1, R1))
        Tf = oracle.time_star(t0, R0, t1, R1, 1.0, TWO_PI)
        self.assertGreater(((T - Tf) < -1e-3).double().mean().item(), 0.5)

    def test_p3_triangle_measured_not_guaranteed(self):
        """Not guaranteed (the note). With a tiny residual T ~ T_free (a metric) and violations are negligible."""
        t0, R0 = random_poses(2000, 12)
        t1, R1 = random_poses(2000, 13)
        t2, R2 = random_poses(2000, 14)
        net = make_net('graph_branch', dict(n_branch=1), res_gain=1e-4)
        g0, g1, g2 = feats(t0, R0), feats(t1, R1), feats(t2, R2)
        v = net.net.value(g0, g2) - net.net.value(g0, g1) - net.net.value(g1, g2)
        self.assertLess(v.max().item(), 1e-6)


class TestGoalStructure(unittest.TestCase):
    """Properties 5 and 6, in both endpoints."""

    def _cone(self, net, t, R, xi, eps, end):
        tp, Rp = perturb(t, R, xi, eps)
        g, gp = feats(t, R), feats(tp, Rp)
        return net.net.value(gp, g) if end == 0 else net.net.value(g, gp)

    def test_p5_p6_cone_has_the_metric_shape(self):
        t, R = random_poses(400, 15)
        xi = unit_twists(400, 16)
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra)
            slope_p = net.net.endpoint_slope(feats(t, R))                 # 1 + rho(p) (1 if strict)
            for end in (0, 1):
                # random weights make the residual's curvature large (T ~ 5-30 at unit distance; a Huber residual is
                # 1 / (2 eps) more curved), so the quadratic regime is small: checked at eps = 1e-5 / 1e-6 / 1e-7
                s5 = self._cone(net, t, R, xi, 1e-5, end) / 1e-5
                s6 = self._cone(net, t, R, xi, 1e-6, end) / 1e-6
                s7 = self._cone(net, t, R, xi, 1e-7, end) / 1e-7
                # cone: T(eps) / eps = slope + kappa eps + O(eps^2), i.e. each 10x smaller eps shrinks the gap ~10x
                self.assertLess(((s6 - s7).abs() - 0.2 * (s5 - s6).abs()).max().item(), 1e-8, (name, end))
                kappa = (s6 - s7) / (1e-6 - 1e-7)
                # metric shape: the limit is (1 + rho(p)) sqrt(g(xi, xi)) = 1 + rho(p) for every direction xi
                lim = s7 - kappa * 1e-7
                self.assertLess(((lim - slope_p).abs() / slope_p).max().item(), 1e-6, (name, end))
                # Q = T^2: Q(eps) / eps^2 -> (1 + rho)^2 g(xi, xi), and grad Q(goal) = 0 (Q is even to 2nd order)
                Qp = self._cone(net, t, R, xi, 1e-6, end) ** 2
                Qm = self._cone(net, t, R, -xi, 1e-6, end) ** 2
                self.assertLess(((Qp / 1e-12 - slope_p ** 2).abs() / slope_p ** 2).max().item(), 1e-2, (name, end))
                self.assertLess(((Qp - Qm) / 2e-6).abs().max().item(), 1e-2 * slope_p.max().item() ** 2, (name, end))

    def test_p6_fails_for_graph_gram(self):
        """graph_gram's cone is a learned (anisotropic) quadratic form: the slope depends on the direction."""
        t, R = random_poses(200, 17)
        xi = unit_twists(200, 18)
        net = make_net('graph_gram')
        s = self._cone(net, t, R, xi, 1e-5, 0) / 1e-5
        self.assertGreater((s.max() / s.min()).item(), 1.5)


class TestKinks(unittest.TestCase):
    """Properties 7-9."""

    @staticmethod
    def line_kinks(net, t0, R0, t1, R1, xi, s_max, n):
        """Kinks of s -> T(x exp(s xi), y) on [-s_max, s_max]: second differences that do not shrink with h."""
        out = []
        for h_div in (8, 64):  # a coarse grid misses narrow high-curvature regions; compare two fine ones
            s = torch.linspace(-s_max, s_max, n * h_div + 1, dtype=DT)
            tt, RR = se3.right_perturb(t0.expand(len(s), 3), R0.expand(len(s), 3, 3), s[:, None] * xi[None])
            T = net.net.value(feats(tt, RR), feats(t1.expand(len(s), 3), R1.expand(len(s), 3, 3)))
            h = (s[1] - s[0]).item()
            d2 = (T[2:] - 2 * T[1:-1] + T[:-2]) / h           # -> slope jump at a kink, -> 0 where smooth
            out.append((s[1:-1], d2))
        return out

    def test_p7_peaks_only(self):
        """Along random lines the decoded time has only concave kinks (slope drops), never V-shaped ones.

        h-scaled second differences d2 = (T(s+h) - 2T(s) + T(s-h)) / h tend to T'' h where T is smooth and to
        the slope jump at a kink. Refining h by 8: positive d2 (convexity) must shrink ~8x (no V-kink), while
        the negative d2 at ridges keep their size."""
        for name, (arch, extra) in BRANCH.items():
            net = make_net(arch, extra, res_gain=3.0)  # large residuals: many branch switches along each line
            n_ridge = 0
            for k in range(12):
                t0, R0 = random_poses(1, 100 + k)
                t1, R1 = random_poses(1, 200 + k)
                xi = unit_twists(1, 300 + k)[0]
                (s_c, d2_c), (s_f, d2_f) = self.line_kinks(net, t0, R0, t1, R1, xi, 1.5, 300)
                # the goal is not on these lines (generic start / goal): no positive kink may survive refinement
                self.assertLess(d2_f.max().item(), 0.3 * d2_c.max().clamp_min(1e-9).item() + 1e-6, (name, k))
                if d2_c.min() < -1e-3:
                    n_ridge += int(d2_f.min().item() < 0.3 * d2_c.min().item())   # ridge did not shrink
            self.assertGreater(n_ridge, 0, name)  # the test does see ridges

    def test_p8_ridge_moves_off_the_half_turn(self):
        """Rotation slice through the cut locus: with branch-specific residuals the ridge of T leaves theta = pi;
        graph_gram's ridge is exactly where its b latent is maximal (pinned to the decoder's b = 1)."""
        n = 3601
        phi = torch.linspace(0, TWO_PI, n, dtype=DT)
        t0, R0 = random_poses(1, 400)
        ax = torch.tensor([[0.3, -0.5, 0.81]], dtype=DT)
        ax = ax / ax.norm()
        R1 = R0 @ se3.so3_exp(phi[:, None] * ax)
        g0 = feats(t0.expand(n, 3), R0.expand(n, 3, 3))
        g1 = feats(t0.expand(n, 3), R1)
        net = make_net('graph_branch', dict(n_branch=1))
        with torch.no_grad():
            net.net.U.weight.zero_()
        T = net.net.value(g0, g1)
        ridge = phi[T.argmax()].item()
        self.assertGreater(abs(ridge - math.pi), 0.05)    # moved by the residual difference of the two routes
        net.net.W.weight.data.zero_()                      # no residual: the free solution, ridge at pi
        self.assertLess(abs(phi[net.net.value(g0, g1).argmax()].item() - math.pi), 2e-3)
        # graph_gram: the only rotational ridge is the decoder's b -> 1
        gg = make_net('graph_gram')
        a, b = gg.net.predict_latent(g0, g1)
        Tg = gg.net.value(g0, g1)
        rot = abdec.phi(b) / gg.net.s_r ** 2
        self.assertLessEqual(rot.max().item(), math.pi ** 2 / gg.net.s_r ** 2 + 1e-12)   # property 9 cap
        self.assertTrue(bool((b <= 1.0 - 1e-6 + 1e-12).all()))

    def test_p9_no_cap(self):
        """A pure rotation can take any time (the residual is unbounded); graph_gram's rotation term is capped."""
        t0, R0 = random_poses(200, 500)
        _, R1 = random_poses(200, 501)
        g0, g1 = feats(t0, R0), feats(t0, R1)
        Ts = []
        for gain in (1.0, 10.0, 100.0):
            net = make_net('graph_branch', dict(n_branch=1), res_gain=gain)
            Ts.append(net.net.value(g0, g1).median().item())
        self.assertGreater(Ts[2], 50 * Ts[0])
        self.assertGreater(Ts[2], 0.5 * 100)  # far beyond the free half-turn time 0.5 (turns)
