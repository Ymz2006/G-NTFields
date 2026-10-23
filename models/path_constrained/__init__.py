"""Path-constrained SE(3) planning: a square prism with one end glued to a point path, rotation free.

The camera-follow model: the prism is the camera's line of sight to a moving point object (anchor = the object end,
the other end = the camera), so a collision-free prism keeps the object in view. The configuration space is the
4-d submanifold M = [0, L] x SO(3) of SE(3) (arc length along the path x rotation about the anchor). The models are
the SE(3) ones; the path is unrolled onto the x axis (arclength chart, path.py) so M is a slice of SE(3) with the
same product metric, the Eikonal residual is formed with the gradient projected onto M (losses.py) and the planner
moves along M (scripts/plan_eval_path.py). See README.md.
"""
