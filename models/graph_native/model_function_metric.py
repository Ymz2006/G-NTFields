"""Original metric NTFields loss (metric_baseline: TD and normal terms off) for the graph heads.

    Loss = beta * mean_i [ (l0_i + l1_i) exp(-tau_i / 2) ],
    l_e  = 1e-2 (sqrt(S_e |grad_e tau|) - 1)^2,   e = start / goal endpoint,

with S_e the warped combined clearance speed of the dataset (``speed.npy``, warp s^2 (2-s)^2
applied by the trainer). ``grad_mode``:

    chart   |grad tau| = Euclidean norm of d tau / d(t, rotvec / 2 pi) at the endpoint
            (the original method's gradient; it is the Eikonal equation in the chart metric)
    lie     |c| with c = (R^T dT/dt, 2 pi vee(A - A^T)), A = R^T dT/dR, the scaled body-frame
            tangent gradient of the report (translation in metres, rotation in turns)

Both are used with the identical residual and weighting, so the two differ only in the metric
in which the Eikonal equation is imposed.
"""
import math

import torch

from . import _sib


class Function:
    def __init__(self, path, device, network, dim, grad_mode='chart'):
        self.path = path
        self.device = device
        self.dim = dim
        self.network = network
        self.grad_mode = grad_mode
        self.td_weight = 0.0
        self.normal_weight = 0.0
        self.last_terms = {}
        self.total_train_loss = []
        self.total_val_loss = []

    def gradient(self, y, x, create_graph=True):
        grad_y = torch.ones_like(y)
        return torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]

    def endpoint_norms(self, points, create_graph=True):
        """Returns (tau (N,), |grad_x tau|^2, |grad_y tau|^2) in the chosen gradient."""
        if self.grad_mode == 'chart':
            tau, _, Xp = self.network.out(points)
            dtau = self.gradient(tau, Xp, create_graph)
            S0 = (dtau[:, :self.dim] ** 2).sum(1)
            S1 = (dtau[:, self.dim:] ** 2).sum(1)
            return tau[:, 0], S0, S1
        g0 = self.network.features(points[:, :self.dim]).detach()
        g1 = self.network.features(points[:, self.dim:]).detach()
        T, cx, cy = _sib.tangents.tangent_gradients(self.network.net.value, g0, g1, 1.0, 2.0 * math.pi,
                                                    create_graph=create_graph)
        return T, (cx ** 2).sum(1), (cy ** 2).sum(1)

    def Loss(self, points, Yobs, normal, beta, gamma, epoch):
        T, S0, S1 = self.endpoint_norms(points)
        Ypred0 = torch.sqrt(S0 + 1e-8)
        Ypred1 = torch.sqrt(S1 + 1e-8)
        l0 = torch.sqrt(Yobs[:, 0] * Ypred0)
        l1 = torch.sqrt(Yobs[:, 1] * Ypred1)
        loss_weight = 1e-2
        diff = loss_weight * (l0 - 1) ** 2 + loss_weight * (l1 - 1) ** 2
        loss_n = torch.sum(diff * torch.exp(-0.5 * T)) / Yobs.shape[0]
        with torch.no_grad():
            self.last_terms = {'Eik': loss_n.item(), 'T_mean': T.mean().item()}
        loss = beta * loss_n
        return loss, loss_n, diff

    def TravelTimes(self, Xp):
        with torch.no_grad():
            Xp = Xp.to(self.device, torch.float32)
            out = []
            for i in range(0, Xp.shape[0], 65536):
                x = Xp[i:i + 65536]
                out.append(self.network.net.value(self.network.features(x[:, :self.dim]),
                                                  self.network.features(x[:, self.dim:])))
            return torch.cat(out)

    def Speed(self, Xp):
        Xp = Xp.to(self.device)
        T, S0, _ = self.endpoint_norms(Xp, create_graph=False)
        return (1 / torch.sqrt(S0 + 1e-12)).detach()

    def Gradient(self, Xp):
        tau, _, coords = self.network.out(Xp.to(self.device))
        return self.gradient(tau, coords, create_graph=False).detach()
