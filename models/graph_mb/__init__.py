"""Multi-branch decoder that learns the cut (models/graph_method_triangle/multibranch.py: heads graph_mb /
graph_mb_free) trained with the spec's band-limited sheet losses, distinctness at ties, temperature schedule,
near-cut sampling and branch management (model_function_mb.py, model_train_mb.py). See README.md and
EXPERIMENTS.md. Checkpoints keep the graph_native contract (evaluate with ``--kind graph_native``).
"""
