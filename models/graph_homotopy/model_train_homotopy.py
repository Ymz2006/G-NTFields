"""Trainer for the homotopy head: models/graph_lie's trainer (the original metric NTFields procedure) with
model_function_homotopy.Function as the loss. Training parameters added: 'Lift' ('min' / 'random' / 'both')
and 'Lift far' (probability of the far sheet for 'random'). Checkpoints are graph_lie's (graph_native
contract), so the per-case evaluator loads them with ``--kind graph_native``."""
from models.graph_lie import model_train_lie as mtl
from models.graph_native import model_network_metric as model_network

from . import model_function_homotopy as model_function


class Model(mtl.Model):
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        super().__init__(ModelPath, DataPath, dim, source, device)
        self.Params['Training']['Lift'] = 'min'
        self.Params['Training']['Lift far'] = 0.5

    def _build(self, B):
        dev = self.Params['Device']
        P = self.Params['Training']
        self.B = B
        self.network = model_network.NN(dev, self.dim, B, graph=self.Params['Network']['Graph'])
        self.network.to(dev)
        self.network.float()
        self.function = model_function.Function(self.folder, dev, self.network, self.dim,
                                                grad_mode=P['Grad mode'], speed_model=P['Speed model'],
                                                weight_mode=P.get('Weight', 'exp'), lift=P.get('Lift', 'min'),
                                                lift_far=P.get('Lift far', 0.5), seed=P.get('Seed', 0))

    def _log(self, msg):
        if msg.startswith('Epoch'):
            lt = getattr(self.function, 'last_terms', {})
            extra = ['{} = {:.4g}'.format(k, lt[k]) for k in ('far', 'long_way', 'gap_med') if k in lt]
            if extra:
                msg = msg + ' -- ' + ' -- '.join(extra)
        super()._log(msg)
