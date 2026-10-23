# Experiment log: constrained movement (camera pyramid anchored at a followed shape), metric NTFields (2026-10-22)

Request (user, 2026-10-22): "constrained movement", a camera-following task. A shape moves; the camera is a
rectangle-based pyramid whose base rectangle is centred at the centre of the followed shape. That centre does not
move relative to the shape: the camera can only rotate about it, so it follows the shape around while the shape
navigates. Model it with t + rotation instead of xyz + rotation, t along the specified path; generate the shape's
path first (e.g. with RRT), then sample the pyramid's poses along it. Use the baseline metric NTFields for now, try a
few existing scenarios, show the results.

Machine: four RTX 3090 shared with another user's training and another session's preprocessing / trainings (all
GPUs at 75-100 % and 15-19 GB used when this started); image `pytorchserver:latest`.

## C0: formulation

The path-constrained prism study (`models/path_constrained`, 2026-10-21) already plans a body glued to a path with
t + rotation (M = [0, L] x SO(3), arclength chart, projected Eikonal residual, MPPI on M); its README section 1
applies unchanged. What is new here: the body is the camera's view pyramid anchored at its base centre (not a thin
prism anchored at an end face), and the path is the RRT path of a real shape through the scene (not a spline through
random via points). The NTFields trainer `train/train_metric_path.py` is reused as is (it reads the path and chart
from the cell's meta); the scripts of this module are the prism study's with the pyramid's geometry.

## C1: geometry (`geometry.py`, `tests/test_constrained_movement.py`, 6 tests, all pass)

* Pyramid = the camera + object test's camera: base 48 x 22 units, apex 50 units from the base centre, 51.3 x 24.8
  degree field of view; base centred on the bird's bounding-box centre (24, 19, 20 in the bird's frame).
* Exact clearance against boxes (26 separating axes, vertex-box, box corner-triangle (Ericson's closest point),
  8 x 12 segment pairs). Dense-sampling check on 120 random poses in each of forest, city, env4 (surface spacing
  0.004 plus interior shells): never above the sampled distance, within 4e-3; every negative case has a sample
  within 4e-3 of or inside an obstacle. Witness distance = clearance (1e-9); |c(x + d) - c(x)| <= |d| on 400
  random translations of 1e-3.
* The followed shape's clearance (parts as boxes, exact Minkowski box distances) checked against a 25^3 grid per
  part (within 3e-3).

## C2: the bird's paths (`scripts/make_shape_paths.py`)

* First pass (no height band, 300 shortcuts): valid but often straight over the roofs (city) or with RRT detours
  left in (forest: a climb over the crowns and back). Second pass: endpoints in a height band for forest (30..75:
  below the crowns, between the trunks) and city (30..90; 30..55 had no room: the bird's 48-unit wingspan fits only
  the 56- and 60-wide streets, a 3.6 x 7.6 unit corridor for its centre), 1000 shortcuts + a greedy taut pass,
  corridor allowed up to L = 3.
* Picked (README section 3): forest #5, city #1, corridor #13, env1 #8, env4 #13. Camera room (fraction of random
  rotations free) mean 0.41-0.67, never below 0.15; straight motion between random endpoint rotations collides in
  73-100 % of the pairs.

## C3: smoke run (env4, 20k pairs, 100 epochs, 24 cases)

Preprocessing 20 s (800k pairs: ~10-15 min per cell on the shared GPUs; band acceptance 44 %); test set 24 cases in
24 s with 8 workers (RRT median 55 iterations, straight motion 17 %); training 42 s; evaluation runs. Speed means:
speed_dists 0.53, speed_angles 0.09 (prism cells: 0.65-0.71, 0.11-0.14; the pyramid is wider, so the training poses
sit closer to the scene: median x1 clearance 0.029 vs 0.048-0.054).

## C4: queue (`scripts/run_cmove_queue.py`, started 01:57)

5 cells, 10 test sets (full, seg), NTFields 8000 epochs, evaluations forward + free on both sets. Started with seeds
0, 1, 2; at 02:10 cut to seed 0 (`--seeds 0 --max-train 5`, the two extra city seeds killed after ~150 epochs and
their folders removed): the machine was at load ~110 on 36 cores and every GPU at 100 % from other users' jobs, and
15 trainings at once would have delayed every seed-0 result. Memory estimate per training lowered to 2 GB (the
NTFields trainings use 0.8-1 GB). Whole-path sets ordered before sub-path sets.

* Preprocessing: 7-9 min per cell (city: 3.95M draws for 800k pairs, speed_dists mean 0.46, speed_angles 0.07).
* Test sets: city full 1000 cases in 20 min (1154 tried, 0 unsolved; straight motion 12.5 % short way, 21.4 % either
  way; RRT median 39 iterations); city seg 12 min (straight 47.1 %).
* Training speed on the shared GPUs: 1.2-1.3 s / epoch once all five ran (8000 epochs ~ 2.8 h).
* Sanity check, city epoch 800, 300 whole-path cases, forward: success 88.3 %, collision 6.0 %, stall 11.7 %; 300
  cases plan in 40 s.

## C5: early checkpoints (epoch 2000, seed 0, forward mode, 1000 cases)

| scene | set | straight | success | collision | stall | failures where (t / L) |
|---|---|---|---|---|---|---|
| city | full | 12.5 | 87.0 | 5.9 | 11.9 | collisions median 0.36 (36 % before 0.1); stalls end at 0.25 |
| city | seg | 47.1 | 94.0 | 2.7 | 5.8 | |
| env1 | full | 0.1 | 83.3 | 11.5 | 15.4 | collisions median 0.16 (93 % before 0.25); stalls end at 0.01 |
| corridor | full | 0.1 | 3.6 | 43.6 | 96.4 | half the plans end before t / L = 0.09; collisions at the first U-turn (0.2) |

Test sets: env1 full 37 min (13 of 1154 drawn cases unsolved by the RRT and dropped; RRT median 151 iterations);
corridor full 56 min (77 unsolved; RRT median 312 iterations). Straight motion 0.1 % in both.

**Corridor diagnosis: the training pairs were too short.** The pair displacement bound U[0, 2] (sqrt(4), the prism
study's analogue of SE(3)'s sqrt(6)) covers M only when L <= 1. The corridor path is L = 2.14, so every whole-path
query (start and goal >= 2.14 apart on M) lies outside every training pair; the plans stall at the start. The bound
is now D = sqrt(max(L, 1)^2 + 3) (preprocess_cmove.py --max-disp; 2.0 for the four other scenes, so their cells are
unchanged, 2.75 for the corridor). New cell `cm_corridor_a2`, run `corridor2_ntf_s0`; the first corridor run is kept
as the ablation. Corridor plans get 860 MPPI steps (400 x L), the same budget per unit of path as the other scenes.

## C6: epoch 4000, the second corridor run, and pair coverage

Whole-path success, forward mode, seed 0, epoch 2000 → 4000: forest 91.6 → 86.4, city 87.0 → 89.7, env1 83.3 → 78.7,
env4 74.7 → 76.0, corridor (pairs <= 2) 3.6 → 4.3. Checkpoint-to-checkpoint changes of up to 5 points: a second seed
(seed 1) of forest, city, env1, env4 is queued behind the seed-0 runs (`--seeds 0,1 --extra-seed-scenes ...`).

`corridor2_ntf_s0` (pairs <= 2.75), epoch 2000, 860 MPPI steps: whole path 25.8 % (collision 12.6 %, 58 % of the
cases stall without moving at all), sub-paths 90.7 %. On sub-paths, success falls with the span t1 - t0: 96 / 97 /
92 / 86 / 84 / 71 % for spans 0.25-0.5 / 0.5-0.75 / 0.75-1 / 1-1.25 / 1.25-1.5 / 1.5-2.2, and plans that never
move are rare (<= 2 %). Only the whole-path query, start and goal pinned to the two ends of the 2.14-long path,
stalls. With isotropic displacement directions in (t, r), pairs that span most of a long path are rare (they need
|d| near the bound and a direction close to the t axis).

`corridor3_ntf_s0` tests that: `preprocess_cmove.py --t-stretch` draws the displacement uniformly in (t / max(L, 1),
r) up to length 2, i.e. the t part stretched by max(L, 1) = 2.14 (the same sampler as the main cells for L <= 1).
Cell `cm_corridor_a3`.

## C7: final seed-0 results (epoch 8000) and the corridor runs at equal epochs

Forward mode, 1000 cases per set, whole path / sub-paths (straight motion in brackets):
forest 86.8 / 94.3 (26.6 / 58.4), city 89.9 / 96.3 (12.5 / 47.1), env1 74.6 / 90.1 (0.1 / 18.0), env4 81.8 / 98.6
(4.7 / 48.8). Free mode within 0.4-3.3 points of forward on whole paths (forest 83.5, city 89.5, env1 73.8, env4
79.1): the forward constraint costs nothing. Whole-path success at epochs 2000 / 4000 / 8000: forest 91.6 / 86.4 /
86.8, city 87.0 / 89.7 / 89.9, env1 83.3 / 78.7 / 74.6, env4 74.7 / 76.0 / 81.8.

Failures follow the camera room (fraction of free rotations, by tenth of the path): env4 failures by tenth
[80 34 19 1 1 2 38 6 0 1], room [0.21 0.52 0.85 0.88 0.88 0.88 0.77 0.28 0.21 0.37] (a cluster at the start and one
just before the room collapses at 0.7); env1 [70 169 13 2 0 ...], room 0.22-0.25 over the first 0.4; forest
[41 16 46 12 0 8 7 1 0 1], room 0.32-0.60 over the first 0.4; city spread.

Corridor, whole path, 860 MPPI steps:

| run | pairs | epoch 2000 | epoch 4000 | epoch 8000 |
|---|---|---|---|---|
| corridor | isotropic, <= 2.0 | 4.2 | 8.0 | 33.0 |
| corridor2 | isotropic, <= 2.75 | 25.8 | 49.5 | 63.2 |
| corridor3 | t stretched x 2.14 | 19.6 | 41.8 | 24.0 |

Sub-paths: corridor (epoch 8000, 400 steps) 88.7, corridor2 90.7 / 91.5 (2000 / 4000), corridor3 91.5 / 92.7 (2000 / 4000).
The pair bound matters (8.0 vs 49.5 at equal epochs) and so does training length (corridor2 25.8 -> 49.5); the
t-stretched sampler does not help (19.6 vs 25.8 at epoch 2000), so the remaining whole-path stalls are not a lack of
path-spanning pairs. About half of the failed whole-path plans never leave the start at epoch 2000 in both runs.

corridor2, epoch 8000 (860 steps): whole path 63.2 % forward (collision 12.8 %, stall 24.0 %), 51.3 % free; sub-paths
94.2 % forward, 89.5 % free. The first corridor run at epoch 8000 with 860 steps: 33.0 % / 89.6 %. In the corridor
the forward constraint helps (+11.9 points on whole paths). Its failures: 299 of 368 end in the first tenth of the
path (240 stalls) although 67 % of the camera rotations are free there: the field's long-range accuracy over a
2.14-long path, not the geometry. Report v4 shows corridor2 as "corridor".

corridor3 (t-stretched pairs), epoch 8000 (860 steps): whole path 24.0 % (collision 36.2 %), sub-paths 91.8 %. It got
worse between epochs 4000 and 8000 (41.8 -> 24.0) and ends far below corridor2 (63.2): stretching the pairs along
the path does not help; corridor2 is the corridor result.

## C8: second seed (seed 1 of forest, city, env1, env4), final

Queue finished 09:51, nothing failed (5 cells, 10 sets, 9 trainings, 36 evaluations; plus the corridor2 / corridor3
runs and early-checkpoint evaluations run by hand). Whole path, forward, seeds 0 / 1: forest 86.8 / 92.3, city 89.9 /
93.5, env1 74.6 / 84.9, env4 81.8 / 91.2 (means 89.5, 91.7, 79.8, 86.5); sub-paths 96.0, 96.4, 91.8, 99.0 (seeds within
3.5 points). Free mode 0.6-4.1 points below forward. Collisions in the first quarter of the path, both seeds: forest
90 %, env1 75 %, env4 81 %, city 59 %. Report v5 (final): https://claude.ai/artifact/KDzupgRi3uqnEcp2Wc47Pe.

## C9: viewer

`scripts/cmove_viser.py` on port 8103 (container `cm_viser`), data from `scripts/export_viser_cmove.py`. The export
recomputes the exact pyramid clearance of every saved plan (waypoints + the evaluator's 4 sub-steps, float32 as the
evaluator). Its collision flags agree with the evaluations on all 18 NTFields sets x 1000 cases, the straight
motion agrees with the test generator's check on all 10 sets, and the RRT references keep >= 0.002 everywhere (the
generator's margin). Success rates reproduce summary.json exactly.
