"""Trainer for the branch heads: models/graph_lie's trainer (itself the original metric NTFields training
procedure) with model_function_branch.Function as the loss.

Training parameters added to graph_lie's: 'End weight' (endpoint cone-slope constraint, 0 = off) and
'Planar' (in-plane Eikonal norm, synthetic SE(2) benchmark only). Both are stored in config.json; the
checkpoints are graph_lie's (graph_native contract), so evaluation uses ``--kind graph_native``.
"""
from models.graph_lie import model_train_lie as mtl
from models.graph_native import model_network_metric as model_network

from . import model_function_branch as model_function


class Model(mtl.Model):
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        super().__init__(ModelPath, DataPath, dim, source, device)
        self.Params['Training']['End weight'] = 0.0
        self.Params['Training']['Planar'] = False

    def _build(self, B):
        dev = self.Params['Device']
        P = self.Params['Training']
        self.B = B
        self.network = model_network.NN(dev, self.dim, B, graph=self.Params['Network']['Graph'])
        self.network.to(dev)
        self.network.float()
        self.function = model_function.Function(self.folder, dev, self.network, self.dim,
                                                grad_mode=P['Grad mode'], speed_model=P['Speed model'],
                                                weight_mode=P.get('Weight', 'exp'),
                                                end_weight=P.get('End weight', 0.0), planar=P.get('Planar', False))

    def _log(self, msg):
        if msg.startswith('Epoch'):
            lt = getattr(self.function, 'last_terms', {})
            extra = ['{} = {:.4g}'.format(k, lt[k]) for k in ('End', 'Nend_med', 'long_way') if k in lt]
            if extra:
                msg = msg + ' -- ' + ' -- '.join(extra)
        super()._log(msg)
