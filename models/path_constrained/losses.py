"""The Eikonal equation on the path-constrained pose space M = [0, L] x SO(3) for both models.

On M the travel time solves S(x) |grad_M T(x, y)| = 1 at both endpoints, grad_M = the gradient of the induced
metric, i.e. the SE(3) gradient with its translation block projected onto the path tangent tau (the only
translation M allows) and the rotation block unchanged (every rotation about the anchor is allowed):

    |grad_M T|^2 = (tau . dT/dt)^2 + |rotation block|^2.

The speed S is the clearance speed of the prism at x, computed as for every SE(3) cell; it multiplies the whole
tangent space of M (motion along the path and rotation). With S on the rotation alone the obstacles would be
transparent to motion along the path (the configuration-space obstacle could be crossed at unit speed by sliding
along it).

Per model (everything else of each trainer unchanged):

    homotopy (models/graph_lie pipeline, Lie gradient c = (c_v, c_w), c_v = R^T dT/dt body frame):
        c_v -> u (u . c_v),  u = R^T tau (the tangent in the body frame)
    metric NTFields (the bundle's chart gradient g = (dT/dt, dT/d(rotvec / 2 pi))):
        g_t -> tau (tau . g_t)
    the TD and normal-alignment terms of NTFields (off unless their weights are set) use the projected gradient
    as well, so the TD step stays on M.

tau comes from the chart (models/path_constrained/path.py): e_x in the arclength chart, the path tangent at the
nearest path point in the world chart.
"""
import math
import os

import torch

from .geometry import so3_exp
from .path import load_constraint

TWO_PI = 2.0 * math.pi


class ChartTangent:
    """tau(t) for chart translations t (N, 3) of one cell (data_dir/meta.json 'constraint')."""

    def __init__(self, data_dir):
        self.meta, self.path, self.chart = load_constraint(data_dir)

    def __call__(self, t):
        return self.path.chart_tangent(t, self.chart)


def project_lie(c, x, tangent):
    """Lie endpoint gradient c (B, 6) at chart poses x (B, 6) -> its projection onto M (B, 6)."""
    R = so3_exp(TWO_PI * x[:, 3:6].detach())
    tau = tangent(x[:, :3].detach())
    u = torch.einsum('bji,bj->bi', R, tau)                         # R^T tau
    cv = u * (u * c[:, :3]).sum(-1, keepdim=True)
    return torch.cat([cv, c[:, 3:]], 1)


def project_chart(g, x, tangent):
    """Chart endpoint gradient g (B, 6) at chart poses x (B, 6) -> its projection onto M (B, 6)."""
    tau = tangent(x[:, :3].detach())
    gt = tau * (tau * g[:, :3]).sum(-1, keepdim=True)
    return torch.cat([gt, g[:, 3:]], 1)


# ----------------------------------------------------------------------------- homotopy head
def homotopy_function_class():
    from models.graph_homotopy import model_function_homotopy as mfh

    class HomotopyPathFunction(mfh.Function):
        """model_function_homotopy.Function with the endpoint gradients projected onto M."""
        tangent = None

        def lifted_gradients(self, points, create_graph=True):
            T, cx, cy, sigma, far = super().lifted_gradients(points, create_graph)
            if self.lift == 'both':
                points = torch.cat([points, points])
            cx = project_lie(cx, points[:, :self.dim], self.tangent)
            cy = project_lie(cy, points[:, self.dim:], self.tangent)
            return T, cx, cy, sigma, far

    return HomotopyPathFunction


# ----------------------------------------------------------------------------- metric NTFields (bundle)
def metric_function_class(base):
    """Subclass of the bundle's models.metric.model_function_metric.Function whose Loss is the original one
    with the endpoint gradients projected onto M (operation order otherwise kept)."""

    class MetricPathFunction(base):
        tangent = None

        def Loss(self, points, Yobs, normal, beta, gamma, epoch):
            tau, w, Xp = self.network.out(points)
            dtau = self.gradient(tau, Xp)
            DT0 = project_chart(dtau[:, :self.dim], Xp[:, :self.dim], self.tangent)
            DT1 = project_chart(dtau[:, self.dim:], Xp[:, self.dim:], self.tangent)
            S0 = torch.einsum('ij,ij->i', DT0, DT0)
            S1 = torch.einsum('ij,ij->i', DT1, DT1)
            td_weight = self.td_weight
            if td_weight:
                with torch.no_grad():
                    length0 = 0.03 / Yobs[:, 0].unsqueeze(1)
                    Dir0 = length0 * (DT0 * Yobs[:, 0].unsqueeze(1) ** 2).clone().detach()
                    Xp_new0 = Xp.clone().detach()
                    Xp_new0[:, :self.dim] = Xp_new0[:, :self.dim] - Dir0
                    tau_new0, _, _ = self.network.out(Xp_new0)
                tau_loss0 = td_weight * ((tau - (tau_new0 + length0)) ** 2).squeeze()
                with torch.no_grad():
                    length1 = 0.03 / Yobs[:, 1].unsqueeze(1)
                    Dir1 = length1 * (DT1 * Yobs[:, 1].unsqueeze(1) ** 2).clone().detach()
                    Xp_new1 = Xp.clone().detach()
                    Xp_new1[:, self.dim:] = Xp_new1[:, self.dim:] - Dir1
                    tau_new1, _, _ = self.network.out(Xp_new1)
                tau_loss1 = td_weight * ((tau - (tau_new1 + length1)) ** 2).squeeze()
                tau_loss0[tau[:, 0] < length0.squeeze()] = 0
                tau_loss1[tau[:, 0] < length1.squeeze()] = 0
                tau_loss = tau_loss0 + tau_loss1
            else:
                tau_loss = torch.zeros_like(tau[:, 0])
            Ypred0 = torch.sqrt(S0 + 1e-8)
            Ypred1 = torch.sqrt(S1 + 1e-8)
            l0 = Yobs[:, 0] * Ypred0
            l1 = Yobs[:, 1] * Ypred1
            loss_weight = 1e-2
            loss0 = loss_weight * (torch.sqrt(l0) - 1) ** 2
            loss1 = loss_weight * (torch.sqrt(l1) - 1) ** 2
            T = tau[:, 0]
            diff = loss0 + loss1
            normal_weight = self.normal_weight
            if normal_weight:
                n0 = (1.001 - Yobs[:, 0].unsqueeze(1)) * (Yobs[:, 0].unsqueeze(1) * DT0 + normal[:, :self.dim]) ** 2
                n1 = (1.001 - Yobs[:, 1].unsqueeze(1)) * (Yobs[:, 1].unsqueeze(1) * DT1 + normal[:, self.dim:]) ** 2
                n_loss = normal_weight * (torch.sum(n0, dim=1) + torch.sum(n1, dim=1))
            else:
                n_loss = torch.zeros_like(T)
            loss_n = torch.sum((diff + n_loss + tau_loss) * torch.exp(-0.5 * T)) / Yobs.shape[0]
            with torch.no_grad():
                wT = torch.exp(-0.5 * T) / Yobs.shape[0]
                self.last_terms = {'Eik': torch.sum(diff * wT).item(), 'Normal': torch.sum(n_loss * wT).item(),
                                   'TD': torch.sum(tau_loss * wT).item()}
            return beta * loss_n, loss_n, diff

    return MetricPathFunction


def data_dir_of(path):
    return path.rstrip('/') if os.path.isdir(path) else os.path.dirname(path)
