"""Lie-gradient Eikonal training of the smooth-graph heads (graph_metric, triangle head).

This package trains the graph heads of models/graph_method_triangle inside the original metric
NTFields pipeline (the trainer of models/graph_native: same data, AdamW, 5 batches of 2000 per
epoch, accept / reject rollback, exp(-tau / 2) causality weight, 1e-2 (sqrt(N) - 1)^2 residual,
4000 epochs) and replaces the partial derivatives of tau in the Eikonal residual by the Lie
gradient on SE(3):

    chart (regular)  g = (d tau / d t,  d tau / d r),           r = rotvec / 2 pi
    lie              c = (R^T d tau / d t,  2 pi vee(A - A^T)),  A = R^T d tau / d R

c is the derivative of tau along the body-frame generators, c_i = d/de tau(x exp(e E_i)), with
rotations measured in turns (lie.py). The two differ only in the rotation block,
c_w = Jr^{-1}(2 pi r)^T g_w, by up to a factor pi / 2 at |rotvec| = pi.

Three speed formulations, each with either gradient (model_function_lie.py):

    combined   S   |g|                              = 1   (metric_baseline / models/graph_native)
    iso        S_d sqrt(|g_v|^2 + |g_w|^2)          = 1   (Lie-Shape-NTFields lie_isotropic)
    aniso      sqrt(S_d^2 |g_v|^2 + S_a^2 |g_w|^2)  = 1   (Lie-Shape-NTFields lie_metric)

with S the combined clearance speed, S_d the translational and S_a the rotational clearance
speed, preprocessed exactly as in the respective original trainers (model_train_lie.py).

Checkpoints keep the models/graph_native contract (the network is graph_native's NN), so the
per-case evaluator loads them with ``--kind graph_native``. See README.md and EXPERIMENTS.md.
"""
