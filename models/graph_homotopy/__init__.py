"""Double-cover ("homotopy") travel-time head (models/graph_method_triangle/homotopy_decoder.py, arch
``graph_homotopy``) trained in the models/graph_lie pipeline (original metric NTFields trainer, Lie-gradient
Eikonal residual, isotropic clearance speed), with the training pairs lifted to the cover R^3 x S^3.

See README.md (method, properties, results) and EXPERIMENTS.md (log). Checkpoints keep the
models/graph_native contract (``--kind graph_native`` in the per-case evaluator); the cover planner is
``scripts/plan_eval_homotopy.py``.
"""
