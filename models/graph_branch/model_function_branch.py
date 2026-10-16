"""graph_lie's Eikonal loss plus the endpoint constraint of the branch heads.

The Eikonal residual is models/graph_lie/model_function_lie.Function unchanged (grad mode x speed model,
1e-2 (sqrt(N) - 1)^2 per endpoint, exp(-tau / 2) weight). Two additions:

end constraint (``end_weight`` > 0, heads with an ``endpoint_slope``). At the goal the branch head's cone is
    T(p exp(eps xi), p) = (1 + rho(p)) |xi|_g eps + O(eps^2)            (property 6, conformal form),
so the endpoint gradient tends to |c| = 1 + rho(p) as the start approaches the goal (from either side,
by symmetry). The Eikonal equation S(p) |c| = 1 in that limit is an algebraic condition on the head,
    N_end(p) = S(p) (1 + rho(p)) = 1,
imposed at both endpoints of every training pair with the same residual form and no causality weight
(tau = 0 at the goal):  L_end = 1e-2 [(sqrt(N_end(x)) - 1)^2 + (sqrt(N_end(y)) - 1)^2].
The loss term is added to the logged Eikonal loss (so the trainer's beta normalisation and accept /
reject rule see it). S is the speed of the chosen speed model (combined: S, iso: S_d); the anisotropic
model has a non-conformal cone and is not supported.

planar (``planar=True``): only the in-plane tangent directions (v_x, v_y, w_z) enter the Eikonal norm. Used
by the synthetic SE(2)-in-SE(3) benchmark (models/graph_branch/scripts/synth_*), whose poses all lie in the
plane z = 0 with yaw rotations, so the off-plane derivatives are not defined by the problem.
"""
import torch

from models.graph_lie import model_function_lie as mfl

PLANAR = [0, 1, 5]  # v_x, v_y, w_z


def planar_N(g, speeds, e, speed_model, eps=1e-8):
    gp = g[:, PLANAR]
    if speed_model == 'aniso':
        return torch.sqrt(speeds['Sd'][:, e] ** 2 * (gp[:, :2] ** 2).sum(1)
                          + speeds['Sa'][:, e] ** 2 * gp[:, 2] ** 2 + eps)
    return mfl.eikonal_N(gp, speeds, e, speed_model, eps)


class Function(mfl.Function):
    def __init__(self, path, device, network, dim, grad_mode='lie', speed_model='iso', weight_mode='exp',
                 end_weight=0.0, planar=False):
        super().__init__(path, device, network, dim, grad_mode, speed_model, weight_mode)
        self.end_weight = float(end_weight)
        self.planar = bool(planar)

    def end_speed(self, speeds, e):
        if self.speed_model == 'combined':
            return speeds['S'][:, e]
        if self.speed_model == 'iso':
            return speeds['Sd'][:, e]
        raise ValueError('the end constraint needs a conformal speed model (combined / iso)')

    def Loss(self, points, speeds, beta, gamma, epoch):
        T, gx, gy = self.endpoint_gradients(points)
        if self.planar:
            N0 = planar_N(gx, speeds, 0, self.speed_model)
            N1 = planar_N(gy, speeds, 1, self.speed_model)
        else:
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
        loss_n = torch.sum(diff * w) / points.shape[0]
        terms = {}
        last = getattr(self.network.net, '_last', {})
        if self.end_weight > 0 and last.get('rho_x') is not None:
            s0 = 1.0 + last['rho_x']
            s1 = 1.0 + last['rho_y']
            E0 = self.end_speed(speeds, 0) * s0
            E1 = self.end_speed(speeds, 1) * s1
            l_end = loss_weight * ((torch.sqrt(E0) - 1) ** 2 + (torch.sqrt(E1) - 1) ** 2)
            loss_end = self.end_weight * l_end.mean()
            loss_n = loss_n + loss_end
            with torch.no_grad():
                terms = {'End': loss_end.item(), 'Nend_med': torch.cat([E0, E1]).median().item()}
        with torch.no_grad():
            self.last_terms = {'Eik': loss_n.item(), 'T_mean': T.mean().item(),
                               'N_med': torch.cat([N0, N1]).median().item(), **terms}
            if 'long_way' in last:
                self.last_terms['long_way'] = last['long_way'].float().mean().item()
        loss = beta * loss_n
        return loss, loss_n, diff
