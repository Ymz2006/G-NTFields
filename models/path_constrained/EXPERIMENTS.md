# Experiment log: path-constrained prism (camera follow), homotopy head vs metric NTFields (2026-10-21)

Request (user, 2026-10-21): plan a camera that follows a moving object and keeps it in view. Model: a small square
prism (the camera's line of sight), one end glued to a point that follows a given path (xyz constrained), the
rotation about that point free; start and end orientations of the prism given. Training data: a valid point
trajectory in one of env 1-4, one end of the prism glued to it, rotations sampled along it, the speed computed the
same way but applied to the rotation; "it should be the same as planning in SE(3), and the formulation could be
improved". Compare graph_homotopy with the metric NTFields.

Machine: four RTX 3090, shared with another user's 15-16 training processes (all GPUs at 100 % throughout) and
another session; image `pytorchserver:latest`. There is no env2 mesh anywhere on the machine, so "env 1-4" = env1,
env3, env4.

## P0: formulation (README section 1)

* The configuration space is the 4-d submanifold M = {(t, R): t = p(l), l in [0, L]} = [0, L] x SO(3) of SE(3), with
  the body frame of the prism at the glued end (rotations about the anchor keep it on the path). Its induced
  metric is the SE(3) product metric restricted: dl^2 + |omega / 2 pi|^2.
* Arclength chart: the path is unrolled onto the x axis, t = (l - L / 2, 0, 0). M is then literally a slice of
  SE(3) (translation on a segment of the x axis) with the same metric, so the models, their losses, the Lie
  gradient, the double cover and the MPPI planner apply unchanged up to the projection below. The world chart
  t = p(l) (the real anchor position) is kept as an ablation.
* Eikonal on M: S(x) |grad_M T| = 1, grad_M = the SE(3) gradient with its translation block projected onto the
  path tangent tau (e_x in the arclength chart): the user's "xyz speed along the direction of the path".
* The speed multiplies the whole tangent space of M (path direction and rotation), not the rotation alone: with S
  on the rotation only, the metric dl^2 + |dtheta|^2 / S^2 lets the planner slide through the configuration-space
  obstacle along l at unit cost (the obstacles become transparent to motion along the path).
* Planning: MPPI on (l, r) with tangent perturbations; free mode (l anywhere on [0, L]) and forward mode (every
  sample dl >= 0, l <= l_goal: the glued object only moves forward, the camera-follow constraint). The fields are
  symmetric, so monotonicity is a planner constraint, measured in the free mode.
* Test (tests/test_path_constrained.py, 9 tests): the analytic free cone sqrt(dl^2 + (psi / 2 pi)^2) has projected
  Lie gradient norm exactly 1 in the arclength chart (it is the distance of M) and not in the world chart, where the
  chord |p(l_x) - p(l_y)| < arc length; the projections equal finite-difference derivatives along the path for both
  gradients (Lie and chart).

## P1: geometry, prism, paths

* Scenes: env1 / env3 / env4 are a room box (env3 / env4: a shell, inner faces found by ray casting) plus 4 / 3 / 6
  axis-aligned boxes. Exact signed clearance of the prism (oriented box): room slack at the corners, box-box
  distance (vertex-box, box-vertex, 12 x 12 edge-edge) when a separating axis exists, minus the smallest of the 15
  separating-axis overlaps otherwise. Against a 2.5e-3 surface grid of the prism on 4000 random poses: never above
  the sampled bound, at most 2.3e-4 below it, 2 / 4000 sign disagreements (grazing overlaps < 4e-4 the grid misses).
  The 10k-point environment sampling of the SE(3) cells (spacing ~0.02) is as wide as the prism, so it is not used.
* Angular clearance (speed_angles): preprocess_obj.evaluate_placements' definition (smallest angle about the anchor
  between a prism surface point and an environment surface point in the same radial shell, 10 shells), with the
  box faces sampled at spacing 0.005 (206k points in env1) and the points within reach pre-gathered per path sample.
  Checked against the original evaluate_placements (Lie repo, same prism, shell points and surface points) on 300
  poses of env1 a: median difference 1e-7, max 3.5e-4 (float32 arccos near 0).
* Prism: w = 0.02, h = 0.2 (4.2 x 4.2 x 42 room units; room ~200 x 140 x 210). Paths (make_paths.py): 200
  centripetal Catmull-Rom candidates per scene through a start, a goal and 1-3 via points, point clearance >= 0.04,
  length 0.9-1.6, start-goal >= 0.6, turning radius >= 0.06. Scored by the straight motion (l linear, R slerp short
  way) between random endpoint rotations: h = 0.15 / 0.2 / 0.25 / 0.3 -> median success 47 / 30 / 20 / 13 % (env1).
  h = 0.2 chosen. Picked per scene (low straight-motion success, free fraction >= 0.35 everywhere, enough endpoint
  room): env1 #63 (L 1.53), env3 #85 (L 1.50), env4 #41 (L 1.20) as paths a; env1 #40, env3 #176, env4 #61 as b.

## P2: data and test sets

* preprocess_path.py, per cell 800k pairs: x0 = (l uniform, R uniform on SO(3)) in the band 0.001 < clearance <
  0.05 (32 % of the draws), x1 = x0 exp(4-d tangent displacement, length U[0, 2]) kept if on the path and clearance
  > 0.001; speeds by the formulas of preprocess_obj.py (speed_dists, speed_angles / pi, combined); normals from the
  exact witness pair. ~8.5 min per cell on a shared GPU. The world-chart cell is written from the same poses.
* make_path_tests.py: full (l 0 -> L) and seg (l1 - l0 >= 0.25) sets, endpoints uniform on SO(3) with exact
  clearance >= 0.02, every case solved by a forward-monotone RRT-Connect on M (start tree forward, goal tree
  backward; step 0.05, motion resolution 0.003, clearance >= 0.002 along the motion). Smoke set (50 full cases on
  env1 a): the straight motion is collision-free in 4 % (short way) / 18 % (either way) of the cases, RRT median 62
  iterations.

## P3: smoke runs (100 epochs, env1 a, 50 full cases)

homotopy base free 62 % / dual forward 56 % (either sheet 90 %); NTFields base forward 26 % (74 % non-converged).
Full-path plans converge in a median 139 of the 400 iterations (step 0.015, L = 1.53). Throughput on the shared
GPUs: ~1.5 s / epoch alone with the preprocessing running, ~0.9 s / epoch with 5 trainings per GPU later.

## P3b: early full-scale check (02:59, intermediate checkpoints, prism_env1_a_full, 1000 cases, forward mode)

| checkpoint | planner | SE(3) success | collision | non-conv. | median iters | median min clearance |
|---|---|---|---|---|---|---|
| homotopy, epoch 2000 | dual | 100.0 (+ sheet 100.0, - sheet 98.9) | 0.0 | 0.0 | 137 | 0.029 |
| NTFields, epoch 2600 | base | 92.4 | 5.6 | 7.3 | 135 | 0.041 |

Independent re-check of the 100 % (125 cases re-planned with saved paths, `verify_plans` in the session scratchpad):
dense prism points (5 x 5 x 101) against the exact point distance of the scene at 20 sub-steps between waypoints
agree with the evaluator on every case; the plans' smallest clearance is 0.020-0.029 (the endpoints' 0.02 bound is
the tightest point), l advances monotonically 0 -> 1.53.
The test set itself: the straight motion is collision-free in 8.4 % (short way) / 20.5 % (either way) of the cases,
RRT median 56 iterations. The homotopy head is at the ceiling of the standard full set already, so targeted
full-path variants were added (tight: both endpoint clearances in [0.005, 0.02]; bigrot: relative rotation >= 150
deg; hard: both), scored in the forward mode for the main cells.

## P4: main queue (run_path_queue.py, started 01:56)

Main grid prism_env{1,3,4}_a x seeds 0-2 x both models (8000 epochs), chart ablation (world, seed 0), second paths
(b, seed 0); 12 test sets; evaluations hom {base, dual} x {free, forward}, NTF base x {free, forward}, epoch-4000
checkpoints in the forward mode.
Additions during the run: the targeted full-path sets tight / bigrot / hard for the main cells (forward mode), and
the stress grid prism30_env{1,3,4}_a (prism length 0.3 on the same paths, seed 0, sets full and hard). Queue
complete 08:48, nothing failed: 36 trainings (homotopy median 4.0 h, NTFields 2.4 h each on the shared GPUs),
27 test sets (every formed case RRT-solved, none dropped), 354 evaluations.

## P5: results (`results/path_constrained/summary/summary.md`, figures in `results/path_constrained/figures/`)

SE(3) success %, forward mode, main cells (3 seeds), homotopy dual / homotopy base / NTFields:

| set | env1 a | env3 a | env4 a | mean |
|---|---|---|---|---|
| full | 100.0 / 100.0 / 92.4 | 100.0 / 100.0 / 92.3 | 99.0 / 92.2 / 89.9 | 99.7 / 97.4 / 91.5 |
| tight | 100.0 / 100.0 / 87.8 | 100.0 / 99.9 / 87.6 | 99.4 / 95.7 / 82.4 | 99.8 / 98.5 / 85.9 |
| bigrot | 100.0 / 100.0 / 92.5 | 100.0 / 100.0 / 92.2 | 99.1 / 92.3 / 90.2 | 99.7 / 97.4 / 91.7 |
| hard | 100.0 / 99.9 / 87.9 | 100.0 / 99.9 / 88.5 | 99.6 / 96.8 / 82.6 | 99.8 / 98.8 / 86.3 |
| seg | 99.9 / 99.6 / 99.4 | 99.7 / 99.2 / 99.1 | 99.3 / 95.4 / 96.9 | 99.7 / 98.1 / 98.5 |

* Failure modes (forward): homotopy dual 0.0 % collisions everywhere, <= 1.0 % stalls; NTFields 2.8-7.2 %
  collisions and 7.5-16.8 % stalls on the whole-path sets. Seed-matched McNemar, homotopy dual vs NTFields on the
  whole-path sets: +225..+514 / -0..-21 of 3000 pairs, p < 1e-58; sub-paths +17 / -2, +22 / -4, +92 / -19 (p <= 1e-3).
* Where NTFields fails (seed 0, full sets, `summary/failure_location.json`): collisions at a median 2 / 19 / 1 % of
  the path (env1 / env3 / env4), stalls end at 4 / 12 / 9 % (final distance 1.1-1.5: no progress). On the sub-path
  sets its failure rate is 1.4 / 2.6 / 7.1 % when l0 < 0.1 L vs 0.2 / 0.5 / 2.5 % for interior starts: the field is
  weakest near the end l = 0 of M, where fewer training displacements stay on the path. The homotopy head shows no
  such effect (0.0-0.5 %).
* Homotopy base planner on env4: 7.8 % stalls, all at l ~ 0.58 L (one place of the path); dual 99.0 % = the
  two-sheet oracle; + sheet alone 91.9 %, - sheet 84.2 %. On env1 / env3 the + sheet alone is 100 %.
* Stress, prism length 0.3 (seed 0), full / hard: homotopy dual 100.0 / 99.4, 100.0 / 98.7, 99.3 / 99.5;
  NTFields 94.0 / 86.6, 83.7 / 74.0, 83.4 / 79.3 (straight motion 1.4-4.9 %).
* Second paths (seed 0, full): env1 b 100.0 vs 95.5, env3 b 98.1 vs 98.1 (tie: the homotopy head's 1.9 % are stalls
  only, NTFields 0.7 % collisions + 1.9 % stalls), env4 b 99.9 vs 92.3.
* Free vs forward mode: same success within 0.8 points; free-mode plans that move the object backwards: 0.2-11 %
  (whole path), 6-19 % (sub-paths).
* Chart ablation (seed 0): homotopy arclength vs world within 0.4 (dual) / 2.3 (base) points; NTFields loses 1.5-7.2
  points on whole paths in the world chart (env1 full 96.3 -> 89.1).
* Epoch 4000 vs 8000: within 0.4 points (homotopy dual) and 1.2 points (NTFields).
* Successful plans (full, forward): median smallest clearance 0.028-0.029 (homotopy) vs 0.039-0.041 (NTFields, whose
  successes are the easier cases), median rotation 0.66-0.73 vs 0.76-0.89 turns, the same number of iterations.

## P6: viewer and report (2026-10-21, second request)

Request: a viser visualization for this planning, like the other viewers, with a camera view and the view as the
pyramid; publish a report. Built `export_viser_path.py` + `path_viser.py` (README section 5). Checks: the exported
per-motion clearance (4 sub-steps) reproduces the evaluator's collision flag on all 2000 plans of env1 a full;
camera views rendered headless (Chrome + CDP, swiftshader). `get_render` blocks forever if the browser leaves
before answering, so it runs in a side thread with a 30 s timeout. Viewer examples: env1 a case 125 (NTFields
blocked at step 22, l/L 0.05, then stalls at 0.11), env3 a case 244 (NTFields camera 0.026 above the ceiling at
step 80, l/L 0.47). Report: https://claude.ai/artifact/1PnNpLTDBQdiyWWekf4jXY.
