"""Min-of-branches decoder heads (models/graph_method_triangle/branch_decoder.py) trained in the
models/graph_lie pipeline (original metric NTFields trainer, Lie-gradient Eikonal residual), with the
optional endpoint ("end") constraint on the cone slope at the goal.

See README.md (method, properties, results) and EXPERIMENTS.md (log). Checkpoints keep the
models/graph_native contract, so the per-case evaluator loads them with ``--kind graph_native``.
"""
