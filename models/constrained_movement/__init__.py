"""Constrained movement: a camera follows a shape that moves along a given path; the camera only turns.

The camera's view volume is a rectangle-based pyramid: the base rectangle centred at the followed shape's centre,
the apex at the camera. The base centre is glued to the shape, so the pyramid can only rotate about it while the
shape moves along its path (an RRT path of the shape through the scene). The configuration space is the 4-d
submanifold M = [0, L] x SO(3) of SE(3) (arc length t along the path x rotation about the anchor), the formulation
of models/path_constrained with the pyramid in place of the prism: the path is unrolled onto the x axis
(arclength chart), the Eikonal residual is formed with the gradient projected onto M, the planner moves along M.
Baseline model: the original metric NTFields (train/train_metric_path.py). See README.md.
"""
