"""Smooth-graph travel-time heads inside the ORIGINAL metric NTFields pipeline.

This package is a copy of the original 3-D shape trainer and loss
(Lie-Shape-NTFields/models/metric_baseline == this repo's models/metric with the
TD and normal losses off): the same data (datasets/3dshape/data/<scene>), the same
isotropic Eikonal residual 1e-2 (sqrt(S |grad tau|) - 1)^2 weighted by
exp(-tau / 2), the same AdamW (weight decay 0.5, learning rate 5e-4), the same
5 batches of 2000 per epoch with the accept / reject rollback, and the same
checkpoint contract used by the unmodified evaluator. The only replaced part is
the network: ``model_network_metric.NN`` maps the chart coordinates
(t, rotvec / 2 pi) to the fixed embedding G(x) = (t, vec R) and applies one of
the graph heads of models/graph_method_triangle (graph_gram, graph_metric,
graph_tri_metric with equivariant V, ...) with the analytic decoder
T = sqrt(a + phi(b) / s_r^2), s_r = 2 pi (rotations in turns).

``Function.Loss`` offers two gradients: ``chart`` (the original coordinate
gradient d tau / d(t, rotvec/2pi)) and ``lie`` (the intrinsic scaled tangent
gradient c = (R^T dT/dt, 2 pi vee(A - A^T)), i.e. the Eikonal equation of the
report). Everything else is untouched, so a comparison against the original
metric head trained by the same code is head-to-head.
"""
