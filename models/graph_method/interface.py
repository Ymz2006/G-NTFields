"""Replaceable pair-to-latent model contract (spec page 3).

    predict_raw(gx, gy)    -> (B, 2)   raw network outputs (graph models)
    predict_latent(gx, gy) -> (a, b)   smooth latent Z after the range adapter
    decode(a, b)           -> T        explicit analytic decoder D
    value(gx, gy)          -> T        = decode(predict_latent(gx, gy))
    value_sq(gx, gy)       -> Q = T^2  (goal loss without differentiating sqrt at 0)

gx, gy are the fixed embeddings G(x) = (t, vec R) in R^12. Baseline models
that output T directly (metric, scalar) implement value / value_sq only and
set has_latent = False. All implementations must be deterministic in eval
mode and at least twice differentiable on regular samples.
"""

import torch

from .fields import decoder


class PairValueModel(torch.nn.Module):
    has_latent = False

    def __init__(self, s_t=1.0, s_r=1.0):
        super().__init__()
        self.s_t = float(s_t)
        self.s_r = float(s_r)

    def value(self, gx, gy):
        raise NotImplementedError

    def value_sq(self, gx, gy):
        return self.value(gx, gy) ** 2

    def forward(self, gx, gy):
        return self.value(gx, gy)


class PairLatentModel(PairValueModel):
    has_latent = True

    def predict_raw(self, gx, gy):
        raise NotImplementedError

    def predict_latent(self, gx, gy):
        raise NotImplementedError

    def decode(self, a, b):
        return decoder.decode(a, b, self.s_t, self.s_r)

    def value(self, gx, gy):
        a, b = self.predict_latent(gx, gy)
        return self.decode(a, b)

    def value_sq(self, gx, gy):
        a, b = self.predict_latent(gx, gy)
        return decoder.decode_Q(a, b, self.s_t, self.s_r)
