"""graph_lie's Eikonal loss on the double cover.

The head's cover field T~(x~, y~) is a function of lifted poses; the Eikonal equation S(x) |c| = 1 holds on
the cover (the lifted speed is S~(x~) = S(x), the Lie gradient on S^3 is the Lie gradient on SO(3)). Each
training pair (x, y) with its endpoint speeds is lifted to the cover before the residual is formed:

    lift 'min'     the sheet with the smaller time (the base field T = min_sigma T~, as the branch decoder);
                   each sheet is trained only where it is the minimum
    lift 'random'  with probability lift_far the other sheet (lift_far = 0.5: a uniformly random lift);
                   both sheets are trained everywhere, T~ becomes the Eikonal solution on the whole cover
    lift 'both'    both sheets of every pair (batch doubled)

The residual is models/graph_lie/model_function_lie.Function's, 1e-2 (sqrt(N) - 1)^2 per endpoint with the
exp(-tau / 2) causality weight (tau = the lifted time), grad mode x speed model unchanged. Logged extras:
the fraction of pairs trained on their far sheet ('far'), of selected routes turning by more than a half
turn ('long_way'), and the median relative gap between the sheets ('gap_med').
"""
import math

import torch

from models.graph_lie import model_function_lie as mfl
from models.graph_native import _sib

TWO_PI = 2.0 * math.pi
LIFTS = ('min', 'random', 'both')


class Function(mfl.Function):
    def __init__(self, path, device, network, dim, grad_mode='lie', speed_model='iso', weight_mode='exp',
                 lift='min', lift_far=0.5, seed=0):
        super().__init__(path, device, network, dim, grad_mode, speed_model, weight_mode)
        assert lift in LIFTS, lift
        assert grad_mode == 'lie', 'the cover loss uses the Lie gradient'
        self.lift = lift
        self.lift_far = float(lift_far)
        self.gen = torch.Generator(device=device).manual_seed(int(seed) + 12345)

    def lifted_gradients(self, points, create_graph=True):
        """(T, c_x, c_y, sigma, far): lifted time and Lie endpoint gradients of the chosen sheet(s)."""
        net = self.network
        g0 = net.features(points[:, :self.dim]).detach()
        g1 = net.features(points[:, self.dim:]).detach()
        head = net.net
        if self.lift == 'min':
            T, cx, cy = _sib.tangents.tangent_gradients(head.value, g0, g1, 1.0, TWO_PI, create_graph)
            return T, cx, cy, head._last['sigma'], torch.zeros_like(T, dtype=torch.bool)
        n = g0.shape[0]
        if self.lift == 'both':
            g0, g1 = torch.cat([g0, g0]), torch.cat([g1, g1])
            sigma = torch.cat([torch.ones(n, device=g0.device), -torch.ones(n, device=g0.device)])
            with torch.no_grad():
                tv = head.sheet_values(g0[:n], g1[:n])
                near = torch.where(tv[:, 0] <= tv[:, 1], 1.0, -1.0)
            far = sigma != torch.cat([near, near])
        else:
            with torch.no_grad():
                tv = head.sheet_values(g0, g1)
                near = torch.where(tv[:, 0] <= tv[:, 1], 1.0, -1.0)
                far = torch.rand(n, generator=self.gen, device=g0.device) < self.lift_far
                sigma = torch.where(far, -near, near)
        T, cx, cy = _sib.tangents.tangent_gradients(lambda a, b: head.sheet_value(a, b, sigma), g0, g1, 1.0,
                                                    TWO_PI, create_graph)
        return T, cx, cy, sigma, far

    def Loss(self, points, speeds, beta, gamma, epoch):
        T, gx, gy, sigma, far = self.lifted_gradients(points)
        if self.lift == 'both':
            speeds = {k: torch.cat([v, v]) for k, v in speeds.items()}
        N0 = mfl.eikonal_N(gx, speeds, 0, self.speed_model)
        N1 = mfl.eikonal_N(gy, speeds, 1, self.speed_model)
        loss_weight = 1e-2
        diff = loss_weight * (torch.sqrt(N0) - 1) ** 2 + loss_weight * (torch.sqrt(N1) - 1) ** 2
        if self.weight_mode == 'exp':
            w = torch.exp(-0.5 * T)
        elif self.weight_mode == 'exp_detached':
            w = torch.exp(-0.5 * T.detach())
        else:
            w = torch.ones_like(T)
        loss_n = torch.sum(diff * w) / T.shape[0]
        with torch.no_grad():
            self.last_terms = {'Eik': loss_n.item(), 'T_mean': T.mean().item(),
                               'N_med': torch.cat([N0, N1]).median().item(), 'far': far.float().mean().item()}
            last = getattr(self.network.net, '_last', {})
            if 'long_way' in last and self.lift == 'min':
                self.last_terms['long_way'] = last['long_way'].float().mean().item()
                self.last_terms['gap_med'] = (last['gap'] / T.detach().clamp_min(1e-6)).median().item()
        loss = beta * loss_n
        return loss, loss_n, diff
