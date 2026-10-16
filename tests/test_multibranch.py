"""Unit tests of the multi-branch decoder that learns the cut (spec section 10):

  * T is unchanged when branches are permuted;
  * T(x, x) = 0 exactly and every detour branch is positive at x = y;
  * T(x, y) = T(y, x) to 1e-6;
  * the soft minimum approaches the hard minimum as tau -> 0;
  * per-branch c_x matches central finite differences;
  * band and gap masks use detached values (the loss runs, every mask is gradient-free, gradients are finite);
  * revival copies a unit, raises it and resets its optimiser state;
  * dead units are excluded from the minimum.

    python -m unittest tests.test_multibranch -v
"""
import math
import sys
import unittest

sys.path.append('.')

import torch

from models.graph_lie import lie
from models.graph_lie.model_train_lie import warp_speeds
from models.graph_mb import model_function_mb as mfm
from models.graph_method_triangle import multibranch as mb
from models.graph_method_triangle.geometry import se3
from models.graph_native import model_network_metric as gn_network
from models.graph_native import _sib

DT = torch.float64
TWO_PI = 2.0 * math.pi
HEADS = {
    'folded_K4': ('graph_mb', dict(n_branch=4)),
    'folded_K1': ('graph_mb', dict(n_branch=1)),
    'folded_K3_strict': ('graph_mb', dict(n_branch=3, conf=False)),
    'free_K4': ('graph_mb_free', dict(n_branch=4)),
    'free_K2': ('graph_mb_free', dict(n_branch=2)),
}


def make_net(arch, extra=None, seed=0, perturb=0.05):
    torch.manual_seed(seed)
    net = gn_network.NN('cpu', 6, None, graph=dict(arch=arch, **(extra or {})))
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith('beta'):
                continue
            p.add_(perturb * torch.randn_like(p))
        if getattr(net.net, 'conf', False):
            net.net.rho_head.weight.normal_(0, 0.2)
            net.net.lam_head.weight.normal_(0, 0.2)
    return net.to(DT)


def random_poses(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9
    q = torch.randn(n, 4, generator=g, dtype=DT)
    q = q / q.norm(dim=1, keepdim=True)
    return t, se3.quat_to_matrix(q)


def random_chart(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                      torch.randn(n, 3, generator=g, dtype=DT) * 0.15,
                      (torch.rand(n, 3, generator=g, dtype=DT) - 0.5) * 0.9,
                      torch.randn(n, 3, generator=g, dtype=DT) * 0.15], 1)


class TestValues(unittest.TestCase):
    def test_goal_zero_and_detours_positive(self):
        t, R = random_poses(200, 1)
        for name, (arch, extra) in HEADS.items():
            h = make_net(arch, extra).net
            g = se3.features(t, R)
            V = h.all_values(g, g)
            self.assertEqual(V[:, 0].abs().max().item(), 0.0, name)             # direct branch exactly 0
            self.assertEqual(h.value(g, g).abs().max().item(), 0.0, name)
            if V.shape[1] > 1:
                self.assertGreater(V[:, 1:].min().item(), 0.0, name)             # every other branch positive

    def test_symmetry(self):
        t0, R0 = random_poses(300, 2)
        t1, R1 = random_poses(300, 3)
        for name, (arch, extra) in HEADS.items():
            h = make_net(arch, extra).net
            g0, g1 = se3.features(t0, R0), se3.features(t1, R1)
            self.assertLess((h.value(g0, g1) - h.value(g1, g0)).abs().max().item(), 1e-6, name)
            V01, V10 = h.all_values(g0, g1), h.all_values(g1, g0)
            self.assertLess((V01 - V10).abs().max().item(), 1e-6, name)

    def test_permutation_invariance(self):
        """Permuting the detour units (their parameters) leaves T unchanged."""
        t0, R0 = random_poses(300, 4)
        t1, R1 = random_poses(300, 5)
        g0, g1 = se3.features(t0, R0), se3.features(t1, R1)
        for name in ('folded_K4', 'free_K4'):
            arch, extra = HEADS[name]
            h = make_net(arch, extra).net
            T = h.value(g0, g1).clone()
            perm = [0, 3, 1, 2]
            with torch.no_grad():
                if h.FORM == 'folded':
                    U = h.U.weight.view(h.K, h.me, -1).clone()
                    W = h.W.weight.view(h.K, h.mo * 4, -1).clone()
                    beta = torch.cat([torch.zeros(1, dtype=DT), h.beta.clone()])
                    h.U.weight.copy_(U[perm].reshape_as(h.U.weight))
                    h.W.weight.copy_(W[perm].reshape_as(h.W.weight))
                    h.beta.copy_(beta[perm][1:])
                else:
                    heads = list(h.heads)
                    for i, p in enumerate(perm):
                        h.heads[i] = heads[p]
            T2 = h.value(g0, g1)
            self.assertLess((T - T2).abs().max().item(), 1e-12, name)

    def test_softmin_to_hardmin(self):
        t0, R0 = random_poses(300, 6)
        t1, R1 = random_poses(300, 7)
        g0, g1 = se3.features(t0, R0), se3.features(t1, R1)
        for name, (arch, extra) in HEADS.items():
            h = make_net(arch, extra).net
            hard = h.value(g0, g1)
            prev = None
            for tau in (1e-1, 1e-2, 1e-3, 1e-4):
                soft = h.softmin_value(g0, g1, tau)
                self.assertTrue((soft <= hard + 1e-12).all(), name)             # soft min <= hard min
                err = (hard - soft).max().item()
                if prev is not None:
                    self.assertLessEqual(err, prev + 1e-12, name)
                prev = err
            self.assertLess(prev, 1e-3, name)

    def test_dead_units_excluded(self):
        t0, R0 = random_poses(200, 8)
        t1, R1 = random_poses(200, 9)
        g0, g1 = se3.features(t0, R0), se3.features(t1, R1)
        for name in ('folded_K4', 'free_K4'):
            arch, extra = HEADS[name]
            h = make_net(arch, extra).net
            V = h.all_values(g0, g1).clone()
            h.alive[2] = 0.0
            V2 = h.all_values(g0, g1)
            dead = (h.unit_index == 2)
            self.assertTrue(torch.isinf(V2[:, dead]).all(), name)
            self.assertTrue(torch.allclose(V2[:, ~dead], V[:, ~dead]), name)
            Tm = h.value(g0, g1)
            self.assertTrue(torch.allclose(Tm, V[:, ~dead].min(1).values), name)
            self.assertTrue((h._last['b'] != 2).all() if h.FORM == 'free' else (h._last['k'] != 2).all(), name)


class TestGradients(unittest.TestCase):
    def test_branch_gradient_finite_differences(self):
        """Per-branch Lie gradient (value_branch + tangent projection) vs central differences along the body
        generators, for a fixed branch per pair (not the minimum)."""
        X = random_chart(96, 10)
        h_fd = 1e-6
        for name, (arch, extra) in HEADS.items():
            net = make_net(arch, extra)
            head = net.net
            nb = head.n_branches
            b = torch.arange(96) % nb
            g0 = net.features(X[:, :6]).detach().requires_grad_(True)
            g1 = net.features(X[:, 6:]).detach().requires_grad_(True)
            T = head.value_branch(g0, g1, b)
            dgx, dgy = torch.autograd.grad(T.sum(), (g0, g1))
            cx = _sib.tangents.project_feature_gradient(dgx, g0.detach()[:, 3:].reshape(-1, 3, 3), 1.0, TWO_PI)
            cy = _sib.tangents.project_feature_gradient(dgy, g1.detach()[:, 3:].reshape(-1, 3, 3), 1.0, TWO_PI)
            t0, R0 = X[:, :3], se3.so3_exp(TWO_PI * X[:, 3:6])
            t1, R1 = X[:, 6:9], se3.so3_exp(TWO_PI * X[:, 9:12])
            fdx, fdy = torch.zeros_like(cx), torch.zeros_like(cy)
            with torch.no_grad():
                for i in range(6):
                    E = torch.zeros(96, 6, dtype=DT)
                    E[:, i] = 1.0 if i < 3 else TWO_PI
                    tp, Rp = se3.right_perturb(t0, R0, h_fd * E)
                    tm, Rm = se3.right_perturb(t0, R0, -h_fd * E)
                    fdx[:, i] = (head.value_branch(se3.features(tp, Rp), se3.features(t1, R1), b)
                                 - head.value_branch(se3.features(tm, Rm), se3.features(t1, R1), b)) / (2 * h_fd)
                    tp, Rp = se3.right_perturb(t1, R1, h_fd * E)
                    tm, Rm = se3.right_perturb(t1, R1, -h_fd * E)
                    fdy[:, i] = (head.value_branch(se3.features(t0, R0), se3.features(tp, Rp), b)
                                 - head.value_branch(se3.features(t0, R0), se3.features(tm, Rm), b)) / (2 * h_fd)
            sc = cx.abs().max().item()
            self.assertLess((cx - fdx).abs().max().item() / sc, 1e-6, name)
            self.assertLess((cy - fdy).abs().max().item() / sc, 1e-6, name)

    def test_min_gradient_matches_lie_module(self):
        """The hard-min value's Lie gradient through lie.endpoint_gradients equals the perturbation reference."""
        X = random_chart(64, 11)
        for name, (arch, extra) in HEADS.items():
            net = make_net(arch, extra)
            T, cx, cy = lie.endpoint_gradients(net, X, 'lie', create_graph=False)
            T2, px, py = lie.lie_gradients_perturb(net, X)
            sc = cx.abs().max().item()
            self.assertLess((cx - px).abs().max().item() / sc, 1e-9, name)
            self.assertLess((T - T2).abs().max().item(), 1e-12, name)


class _FakeNet(torch.nn.Module):
    pass


class TestLoss(unittest.TestCase):
    def _function(self, arch, extra, spec=None):
        net = make_net(arch, extra).float()
        return mfm.Function('/tmp', 'cpu', net, 6, spec=spec, epochs=100), net

    def _batch(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        X = random_chart(n, seed).float()
        sp = torch.rand(n, 2, generator=g) * 0.8 + 0.2
        d = torch.cat([X, sp, torch.zeros(n, 12), sp, torch.ones(n, 2)], 1)
        return X, warp_speeds(d, 6, 1.0)

    def test_masks_detached_and_loss_finite(self):
        X, speeds = self._batch(64, 12)
        for name, (arch, extra) in HEADS.items():
            f, net = self._function(arch, extra, dict(delta=100.0, eta=0.5))   # band = everything: rows = branches
            g0, g1, V, srt, order = f.branch_table(X)
            for t in (g0, g1, V, srt):
                self.assertFalse(t.requires_grad, name)
            loss, loss_n, r = f.Loss(X, speeds, 1.0, 0.0, 1)
            self.assertTrue(torch.isfinite(loss).item(), name)
            loss.backward()
            for n_, p in net.named_parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), (name, n_))
            lt = f.last_terms
            self.assertGreaterEqual(lt['rows'], 1.0, name)
            self.assertGreater(lt['Sheet'], 0.0, name)
            self.assertEqual(f.stats['n_pairs'], 64)
            self.assertEqual(lt['rows'], float(net.net.n_branches), name)   # band holds every alive branch

    def test_active_only_ablation_has_one_row_per_pair(self):
        X, speeds = self._batch(64, 13)
        f, net = self._function('graph_mb', dict(n_branch=4), dict(band_only_active=True))
        f.Loss(X, speeds, 1.0, 0.0, 1)
        self.assertEqual(f.last_terms['rows'], 1.0)
        self.assertEqual(f.last_terms['Dist'], 0.0)

    def test_temperature_schedule(self):
        f, _ = self._function('graph_mb', dict(n_branch=2))
        s = f.spec
        self.assertAlmostEqual(f.tau(1), s['tau_ref'] * s['tau0'])
        self.assertAlmostEqual(f.tau(61), s['tau_ref'] * s['tau_end'])      # end of the first 60 % (epochs 1..60)
        self.assertAlmostEqual(f.tau(100), s['tau_ref'] * s['tau_end'])
        self.assertLess(f.tau(30), f.tau(1))
        self.assertGreater(f.tau(30), f.tau(60))

    def test_near_cut_indices(self):
        X, _ = self._batch(256, 14)
        f, _ = self._function('graph_mb', dict(n_branch=4))
        idx = f.near_cut_indices(X, 32)
        _, _, _, srt, _ = f.branch_table(X)
        gap = srt[:, 1] - srt[:, 0]
        self.assertLessEqual(gap[idx].max().item(), gap.sort().values[31].item() + 1e-7)


class TestRevival(unittest.TestCase):
    def test_copy_unit_and_optimizer_reset(self):
        t0, R0 = random_poses(200, 15)
        t1, R1 = random_poses(200, 16)
        g0, g1 = se3.features(t0, R0), se3.features(t1, R1)
        for name in ('folded_K4', 'free_K4'):
            arch, extra = HEADS[name]
            h = make_net(arch, extra).net
            opt = torch.optim.AdamW(h.parameters(), lr=1e-3)
            h.value(g0, g1).sum().backward()
            opt.step()
            opt.zero_grad()
            moments_before = {id(p): opt.state[p]['exp_avg_sq'].clone() for p in h.parameters() if p in opt.state}
            src, dst = 1, 3
            raise_by = 0.2
            torch.manual_seed(0)
            h.copy_unit(dst, src, noise=0.0, raise_by=raise_by)
            h.reset_optimizer_unit(opt, dst)
            V = h.all_values(g0, g1)
            if h.FORM == 'folded':
                d = V[:, dst] - V[:, src]                     # route +, same residual map: differ by the offset
                self.assertTrue(torch.allclose(d, torch.full_like(d, raise_by), atol=1e-9), name)
                for p, rows in h.unit_param_slices(dst):
                    self.assertEqual(opt.state[p]['exp_avg_sq'][rows].abs().max().item(), 0.0, name)
                for p, rows in h.unit_param_slices(src):
                    self.assertGreater(opt.state[p]['exp_avg_sq'][rows].abs().max().item(), 0.0, name)
            else:
                d = V[:, dst] - V[:, src]
                self.assertGreater(d.min().item(), 0.0, name)   # raised
                for p, _ in h.unit_param_slices(dst):
                    self.assertEqual(opt.state[p]['exp_avg_sq'].abs().max().item(), 0.0, name)
                for p, _ in h.unit_param_slices(src):
                    self.assertGreater(opt.state[p]['exp_avg_sq'].abs().max().item(), 0.0, name)
            # the other units' moments are untouched
            for p in h.parameters():
                if p in opt.state and id(p) in moments_before and not any(p is q for q, _ in h.unit_param_slices(dst)):
                    self.assertTrue(torch.equal(opt.state[p]['exp_avg_sq'], moments_before[id(p)]), name)


if __name__ == '__main__':
    unittest.main()
