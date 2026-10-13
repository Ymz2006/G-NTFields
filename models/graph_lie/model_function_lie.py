"""Original metric NTFields Eikonal loss with a choice of gradient and speed formulation.

Per endpoint e (start x, goal y), with g_e the endpoint gradient of tau in the chosen mode
(lie.py; 'chart' = partial derivatives d tau / d(t, rotvec / 2 pi), 'lie' = body-frame Lie
gradient) split into its translation block g_v and rotation block g_w:

    combined   N_e = S_e   |g_e|                                metric_baseline, models/graph_native
    iso        N_e = S_d,e sqrt(|g_v|^2 + |g_w|^2)              Lie-Shape-NTFields lie_isotropic
    aniso      N_e = sqrt(S_d,e^2 |g_v|^2 + S_a,e^2 |g_w|^2)    Lie-Shape-NTFields lie_metric

    Loss = beta * mean_i [ (l(N_x,i) + l(N_y,i)) exp(-tau_i / 2) ],   l(N) = 1e-2 (sqrt(N) - 1)^2,

with tau not detached in the causality weight, as in NTFields (``weight_mode='exp'``). The obstacle-free
control uses ``weight_mode='none'`` (w = 1): with unit speeds the NTFields weight has a degenerate
inflate-tau basin (both gradients converge to tau ~ 20 T*), and a detached weight makes the loss rise while
the field deflates, which the trainer's accept / reject rule then rolls back forever (EXPERIMENTS.md L5).
The speeds arrive warped by the trainer (model_train_lie.warp_speeds). With grad 'chart' and speed 'combined' this is
models/graph_native's loss, operation for operation (tests/test_graph_lie.py checks equality).
"""
import torch

from . import lie

SPEED_MODELS = ('combined', 'iso', 'aniso')
WEIGHT_MODES = ('exp', 'exp_detached', 'none')


def eikonal_N(g, speeds, e, speed_model, eps=1e-8):
    """N of one endpoint e (0 = start, 1 = goal); g (B, 6) its gradient, speeds a dict of (B, 2)."""
    if speed_model == 'combined':
        return speeds['S'][:, e] * torch.sqrt((g ** 2).sum(1) + eps)
    if speed_model == 'iso':
        return speeds['Sd'][:, e] * torch.sqrt((g ** 2).sum(1) + eps)
    if speed_model == 'aniso':
        return torch.sqrt(speeds['Sd'][:, e] ** 2 * (g[:, :3] ** 2).sum(1)
                          + speeds['Sa'][:, e] ** 2 * (g[:, 3:] ** 2).sum(1) + eps)
    raise ValueError('speed model must be one of {}, got {!r}'.format(SPEED_MODELS, speed_model))


class Function:
    def __init__(self, path, device, network, dim, grad_mode='lie', speed_model='iso', weight_mode='exp'):
        assert grad_mode in lie.GRAD_MODES, grad_mode
        assert speed_model in SPEED_MODELS, speed_model
        assert weight_mode in WEIGHT_MODES, weight_mode
        self.path = path
        self.device = device
        self.dim = dim
        self.network = network
        self.grad_mode = grad_mode
        self.speed_model = speed_model
        self.weight_mode = weight_mode
        self.last_terms = {}
        self.total_train_loss = []
        self.total_val_loss = []

    def endpoint_gradients(self, points, create_graph=True):
        return lie.endpoint_gradients(self.network, points, self.grad_mode, create_graph)

    def Loss(self, points, speeds, beta, gamma, epoch):
        T, gx, gy = self.endpoint_gradients(points)
        N0 = eikonal_N(gx, speeds, 0, self.speed_model)
        N1 = eikonal_N(gy, speeds, 1, self.speed_model)
        loss_weight = 1e-2
        diff = loss_weight * (torch.sqrt(N0) - 1) ** 2 + loss_weight * (torch.sqrt(N1) - 1) ** 2
        if self.weight_mode == 'exp':
            w = torch.exp(-0.5 * T)
        elif self.weight_mode == 'exp_detached':
            w = torch.exp(-0.5 * T.detach())
        else:
            w = torch.ones_like(T)
        loss_n = torch.sum(diff * w) / points.shape[0]
        with torch.no_grad():
            self.last_terms = {'Eik': loss_n.item(), 'T_mean': T.mean().item(),
                               'N_med': torch.cat([N0, N1]).median().item()}
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
        """1 / |g| at the start endpoint, in this Function's gradient mode."""
        _, gx, _ = self.endpoint_gradients(Xp.to(self.device), create_graph=False)
        return (1 / torch.sqrt((gx ** 2).sum(1) + 1e-12)).detach()

    def Gradient(self, Xp):
        tau, _, coords = self.network.out(Xp.to(self.device))
        return torch.autograd.grad(tau, coords, torch.ones_like(tau))[0].detach()
