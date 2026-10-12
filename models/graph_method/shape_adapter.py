"""Adapter exposing a graph-method checkpoint through the `Model` / `function.TravelTimes`
contract used by Shape-NTFields' evaluate_training_3d_batched.py (stage E).

Configurations arrive as (t, rotvec / 2 pi) rows of length 12 (start | goal);
they are mapped to the fixed embedding G = (t, vec R) before querying the field.
"""

import torch

from .model_function_shape import coords_to_features
from .model_train_graph import load_function


class _Queries:
    def __init__(self, network, device, coords=False):
        self.network = network
        self.device = device
        self.coords = coords

    def TravelTimes(self, Xp, chunk=65536):
        Xp = Xp.to(self.device, torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, Xp.shape[0], chunk):
                x = Xp[i:i + chunk]
                if self.coords:
                    out.append(self.network.value(x[:, :6], x[:, 6:]))
                else:
                    out.append(self.network.value(coords_to_features(x[:, :6]), coords_to_features(x[:, 6:])))
        return torch.cat(out)


class Model:
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        self.device = device
        self.dim = dim
        self.network = None
        self.function = None

    def load(self, filepath):
        f, ck = load_function(filepath, device=self.device)
        self.network = f.network
        self.function = _Queries(self.network, self.device, coords=bool(ck['config'].get('coords', False)))
