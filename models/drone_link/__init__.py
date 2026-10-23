"""Two drones joined by their line of sight: a box that stretches along one axis, planned in 7 DOF.

The body is the link box [-w/2, w/2]^2 x [0, L] in its own frame: drone a at the centre of the end face z = 0 (the
body origin), drone b at the centre of the far face z = L. A pose is (a, R, L): the SE(3) pose of drone a with
R e_z = (b - a) / |b - a|, and the length L = |b - a| in [L_min, L_max] (the limits on how far the link may stretch
and shrink). A collision-free box is a pair of drones that never touch the scene and always see each other, so
planning the box keeps the radio link up. The models are the SE(3) ones fed the 7-d chart
(a, rotvec / 2 pi, L). See README.md.
"""
