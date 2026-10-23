# Experiment log: a stretchable link between two drones, 7-DOF metric NTFields (2026-10-22)

Request (user, 2026-10-22): in the forest scenario, introduce a new shape, a rectangle that can shrink along one
length, determined by an SE(3) point (the position of end a, with the direction b - a) and one scalar (the length
from a outwards). Try it with regular NTFields first, re-sampling the speed in 7 DOF (6 for SE(3), 1 for the length),
and report the findings. Set limits on how far it can expand and contract. Application: a pair of drones that must
keep their wireless link (signal contact) while navigating; make three more environments to demonstrate it. All 3-d.

Machine: four RTX 3090, 36 cores, shared with another user's jobs (GPU 2 memory, ~10 cores) for the whole run;
image `pytorchserver:latest`, the metric NTFields bundle `/media/4tbssd/jeffrey/export_metric` at `/export`.

## D0: formulation (README section 1)

* Body: the link box `[-w/2, w/2]^2 x [0, L]`, drone a at the body origin, drone b at `(0, 0, L)`; pose `(t, R, L)`
  with `t = a`, `R e_z = (b - a) / |b - a|`, `L = |b - a|`. In 3-d the "rectangle" is a box of square cross-section;
  its roll about the link axis is the one rotation that does not move the drones (it turns the cross-section).
* `w = 0.02` (4.4 room units, a drone's size and the width of the line-of-sight beam), `L in [0.06, 0.30]` (13-66
  room units: the drones keep at least 3 widths apart and stay within radio range). Room 0.90 x 0.63 x 0.99.
* Chart `x = (t, rotvec / 2 pi, L)`; metric `|dt|^2 + |d theta / 2 pi|^2 + dL^2` (the length in room units).
* Clearance `c = min(c_scene, L - L_min, L_max - L)`: the length limits are walls of the seventh coordinate, and the
  planner also clamps L into the limits. Signal contact = `c_scene >= 0` (line of sight) and `L <= L_max`.
* Speed: `preprocess_obj.py`'s formulas with `c` and the angular clearance of the box of that length about drone a.

## D1: geometry, scenes (tests/test_drone_link.py, 8 tests)

* `geometry.scene_clearance` = `path_constrained.geometry.prism_clearance` with a length per pose: identical to it
  (1e-16) at fixed lengths; against dense surface samples of the box (9 x 9 x 241) on 4000 random poses of the scaffold
  never above the sampled bound, at most 2.7e-4 below it, 1 sign disagreement (a grazing contact).
* `geometry.normal7` against central differences of the clearance (translation and length in the chart, rotation as a
  world-frame perturbation, preprocess_obj's moment convention): cosine > 0.999 on > 97 % of 4000 random free poses.
* `geometry.LinkAngle` (shells of the box of any length from common random numbers) equals the definition computed
  point by point.
* Scenes (`dataprocessing/make_link_scenes.py`): forest (the camera test's, regenerated boxes checked against
  `forest.obj`), windows, corridor, scaffold. The corridor's partitions were first named `wall_*`, which the room
  ray caster treats as room walls (the room came out 0.30 deep); renamed `partition_*`.
* Random link poses (a in the room, R uniform, L uniform): free 54 / 54 / 46 / 47 % (forest / windows / corridor /
  scaffold); the direct 7-d motion between two random free poses with clearance >= 0.02 is valid in 48 / 48 / 31 /
  28 % of the pairs.

## D2: data and test sets

* Cells `link_<scene>` (800k pairs, 2.5-5 min each on a GPU): band rate 42 / 37 / 36 / 40 % of the x0 draws; 35 /
  44 / 36 / 28 % of the band samples are bound by a length limit (the rest by the scene); mean speed 0.40 / 0.46 /
  0.41 / 0.34; median pair distance 0.38-0.42.
* Test sets (RRT-Connect in 7-d, step 0.05, motion resolution 0.003, clearance >= 0.002, then 300 random shortcuts):
  `link_corridor_std`: 1000 kept of 1104 tried (1 unsolved; the batch is oversized by 10 %), direct motion valid
  32.7 %, RRT median 192 iterations / 3.6 s, shortcut RRT path 0.93, start-goal distance 0.51, relative rotation
  133 deg. `link_corridor_cross`: 41 of 1104 tried unsolved within the RRT's 120 s (wall clock, on a machine at load
  100-210 from the other sessions, so the hardest crossings are slightly under-represented), direct motion valid
  0.0 %, RRT median 761 iterations / 22.8 s, shortcut RRT path 1.87, start-goal distance 0.77.

## D3: early check (corridor seed 0, epoch 1700 of 8000, 250 cases of link_corridor_std)

Success 67.2 % (direct motion 32.7 %), collisions 26.0 %, non-converged 28.8 %, mean contact 84 %. 30 % of the
successful plans shorten the link below both endpoint lengths (by > 0.02). The collisions are deep wall crossings
(median penetration 0.056, a drone inside the wall in 88 % of them, 85 % never converge): the field has not yet
closed the thin (0.036) partition walls. 28-44 % of the x0 band samples sit at a length limit, which the planner
enforces anyway, so the ablation cells `link_<scene>_sb` put every x0 in the band of the scene clearance
(`preprocess_link.py --band scene`; the speeds are the same functions of `c`). Their band holds 18-19 % of x0 whose
total clearance is still the length slack (both small).

Second check, corridor seed 0 at epoch 5200, 250 cases each: std 77.2 % (collisions 16.4, stalls 19.6, contact
90.7 %), cross 29.2 % (collisions 57.6, stalls 68.0, contact 64.3 %; 48 % of its successes shorten the link). The
serpentine traversal is the hard query: the field still leaks through the partitions on long queries.

## D4: test sets certified by a roadmap on the GPU, baselines on the GPU (03:00-04:30)

The machine filled up with the other sessions' jobs (load 170-210 on 36 cores, ~15 processes per GPU, 19-23 GB of
24 GB used): the RRT workers got ~4 % of a core each, and after 2 h no forest set had finished. Two changes:

* `scripts/make_link_tests_prm.py`: the same cases (`models/drone_link/cases.py`, case k drawn from
  default_rng(1000003 seed + k) whatever the certificate) certified by one probabilistic roadmap per scene on the GPU:
  60k nodes (exact clearance >= 0.002), 12 nearest neighbours in the 13-d embedding (t, L, vec R / (2 pi sqrt 2)),
  each edge checked as the RRT's motions (t linear, R slerp, L linear, clearance >= 0.002 every 0.003), queries joined
  to their 40 nearest nodes, Dijkstra, greedy shortcut. Corridor crossings, first 334 cases: all solved in 292 s on
  one shared GPU (one connected component, 430k valid edges), certificate paths shorter than the RRT's (median 1.68
  vs 1.87); the RRT had dropped 41 of 1104 at its 120 s limit. All eight sets are generated this way; the RRT corridor
  sets are kept as `link_corridor_{std,cross}_rrt` (logs in `testing_data/3dshape/logs/rrt/`).
* `scripts/baseline_link.py` on the GPU: wavefront relaxation of the 26-connected grid from each goal cell (64
  queries per batch), steepest descent from the start cell, string pulling with the exact point clearance. Against
  the CPU Dijkstra version on 40 corridor crossings: drone path lengths within -2.5 / +4.4 % (median ratio 1.002),
  contact fractions within 0.018 on average, every drone >= 0.015 from the scene. A first GPU version stopped its
  descent at the start cell (on an exact distance field the best neighbour has D[nb] + w = D[c], so the strict test
  must be D[nb] < D[c]); fixed before any set was scored.

## D5: first results, seeds 0-1 (05:00-05:40), and why the crossings fail

PRM-certified sets: every drawn case solved (0 unsolved in all eight sets; one connected roadmap component per
scene). Direct motion valid: forest std 47.7 / cross 11.9, corridor std 32.7 / cross 0.0 (same cases as the RRT sets).

* NTFields (8000 epochs, seeds 0-1; mean): forest std 72.4 / cross 13.6, corridor std 73.2 / cross 34.2, scaffold std
  ~64 / cross 16.2, windows cross 49.5 (seed 0). Epoch 4000: within 0-7 points on the random sets, but corridor
  crossings 18.3 (vs 34.2 at 8000): the long queries keep improving with training.
* Independent drones (GPU baseline, all 1000 cases): valid all the way (line of sight + range + separation) forest
  55.0 / 24.5, corridor 38.1 / 2.7; line of sight lost at least once in 31 / 63 % (forest) and 45 / 90 % (corridor) of
  the cases; contact 89-95 % of the time on average. Certified link-keeping paths are longer per drone than the
  independent shortest paths (corridor crossings: drone a 1.43 vs 1.27, drone b 1.66 vs 1.27).
* Failure split (all failures, seeds 0-1): forest mostly pure stalls (63-70 %), corridor mostly a wall crossing
  followed by a stall (63-76 %), windows crossings often converge through a window frame (36 %).
* Successful plans shorten the link below both endpoint lengths (by > 0.02) in 36-54 % of the cases.
* Success falls with drone a's start-goal distance (forest std, seed 0): 94 % below 0.3, 85 % at 0.3-0.5, 65 % at
  0.5-0.7, 43 % at 0.7-0.9, 5 % beyond (21 cases); stalled plans cover a median 2 % of the distance to the goal and
  jitter in place (net displacement 0.004 over their last 100 steps). Stalls do not depend on the endpoint lengths
  (start in the length walls' slow band or not: 24.0 vs 23.4 % stalls) or the start clearance (26 vs 18 %).
* `scripts/field_diag_link.py` (T along each case's certified path, crossing sets, seed 0; results/drone_link/
  diagnostics/field_*.json): T tracks the remaining certified length (correlation 0.95-0.99). Uphill steps along the
  certified path, stalled vs solved: forest 15 vs 5 %, scaffold 12.5 vs 6.5 %, corridor 2.5 vs 2.5 %, windows 2.2 vs
  0 %. Forest / scaffold: bumps on the valid route trap the descent; corridor / windows: the route is fine, the
  descent leaves it through a wall where the field leaks.
* Planner ablation (200 forest crossings, seed 0, results/drone_link/diagnostics/mppi_*.json): 50 samples / horizon 5
  12.0 %, 200 / 5 11.5 %, 200 / 15 14.5 %: not the planner.
* Training pairs: drone a moves a median 0.18-0.19 within a pair and more than 0.7 in only 2.6-3.2 % of the pairs (the
  6-d displacement spends half its length on rotation; long moves leave the room), while the crossings need 0.7-1.2.
  Hence the long-range ablation `link_<scene>_lr` (`preprocess_link.py --x1-uniform 0.5`: half of the x1 uniform in
  the room), seed 0, queued ahead of the scene-band ablation (whose first launch was stopped after 10 min to free the
  slots; the queue now starts it only once the long-range run of its scene has started).

## D6: seed 2 and the ablations (09:10-10:05)

* Seed 2 (8000 epochs): forest std 67.8 / cross 20.0, windows 65.7 / 42.0, corridor 67.7 / 28.5, scaffold 62.4 / 7.1.
  Seeds 0-2 mean +- std: random 63.6-71.4 (std 1.3-2.6), crossings 13.2-43.1 (std 3.1-5.2); 4000 -> 8000 epochs
  crossings 19.0 -> 26.1 %, random 63.9 -> 68.1 %.
* `ntfb` (TD and normal terms off, the bundle's default loss, seed 0): forest 0.2 / 0.0, windows 0.3 / 0.0, corridor
  0.5 / 0.0, scaffold 59.8 / 15.8. Training logs (aggregate_link.py section 7, mean of the last 100 epochs): Eikonal
  residual 4.5 / 5.1 / 5.3e-3 against 1.0-1.3e-3 for the full recipe's Eikonal part; scaffold 1.58e-3 against 1.61e-3.
  The failed runs learn slowly from the start (epoch 800: ~9e-3 vs ~2e-3). Field (`field_diag_link.py` on the random
  sets, results/drone_link/diagnostics/field_*_std.json): T_start per unit of certified path 15-21 (recipe 1.4-2.0),
  32-38 % of the steps along the certified path uphill (recipe 0-7 %), correlation with the remaining length 0.58-0.72
  (recipe 0.96-0.99): a slow, bumpy field; plans do not converge (nonconv 99.6-100 %). Scaffold's ntfb field matches
  the recipe's (1.9 per unit, 0.6-9 % uphill).
* `ntflr` (half of the x1 uniform in the room, seed 0): corridor 84.5 / 72.1 (recipe 72.9 / 38.0; queries with drone
  a moving >= 0.7: 37 -> 73 %), forest 14.0 / 0.3, windows 1.1 / 0.0, scaffold 32.3 / 6.3 (collisions 64-95 %, forest
  and windows mostly not converging). Field: T per unit of certified path forest 16-20, windows 18-47, scaffold 6-7
  (recipe 1.3-2.2), corridor 2.6-2.8 with no uphill steps; the same slow-field collapse as ntfb except in the
  corridor. Training loss: Eikonal residual 1.8-2.0e-3 in forest / windows (recipe 1.0-1.3e-3), 0.87e-3 in the
  corridor (1.1e-3).
* Follow-up queued 09:50 (queue restarted with the new jobs; running containers adopted): `ntflr20`, cells
  `link_<scene>_lr20` (`--x1-uniform 0.2`), seed 0: does a smaller share keep the near field and some of the
  corridor's gain? `ntfsb` (scene band) still training (epoch ~1900 at 09:48).
* Report v2 published (seeds 0-2, ntfb, ntflr); lede and diagnosis now follow the ablations (the long-range
  hypothesis holds for the corridor only).

## D7: viewer (12:20-13:00)

* `scripts/link_viser.py` (viser, port 8110, container `link_viser`): the plan and a link-unaware motion of the same
  case side by side, synchronized in normalized time, link length / contact chart in the panel. Port 8103 was taken by
  the constrained-movement session's viewer a minute before the first launch (viser silently moved to 8104), hence 8110.
* Checks: the numpy separating-axis test agrees with `geometry.scene_clearance < 0` on 20k random poses per scene (0
  disagreements; room bounds and boxes equal `geometry.Scene`'s); on 100 cases of each of the 8 sets the viewer's
  independent-drone and straight-motion statuses reproduce `baseline_link.py`'s los_blocked / valid / contact fractions
  exactly. Every combination of scene, set, planner, seed, comparison, filter, view and time ran without an error
  (one interpolation fraction just above 1 fixed). A frame takes ~40 ms, a case switch ~0.15 s.

## D8: the scene band and a fifth of the pairs long-range (13:05-13:30)

* `ntfsb` (x0 band on the scene clearance, seed 0): forest 75.0 / 13.7, windows 71.9 / 51.7, corridor 74.6 / 41.0,
  scaffold 62.1 / 17.4 (recipe seed 0: 72.9 / 13.1, 69.9 / 49.5, 72.9 / 38.0, 65.4 / 19.7): within seed noise.
* `ntflr20` (`--x1-uniform 0.2`, seed 0): forest 85.2 / 54.3, windows 68.0 / 37.9, corridor 86.0 / 70.1, scaffold
  62.0 / 26.1. Mean crossings 30.1 -> 47.1 %, random 70.3 -> 75.3 % against the recipe's seed 0. Training losses as low
  as the recipe's (Eikonal 0.9-1.4e-3). Field on the random sets (`field_diag_link.py`): no uphill step along the
  certified paths in any scene (recipe 0-7 %), travel time per unit of path forest 1.9, corridor 2.0, scaffold 2.9,
  windows 4.0 (recipe 1.4, 1.5, 1.9, 2.0; the 50 % run 7-20). Windows: crossings collide more (58 % against 44 %),
  stalled cases have a slow field (T_start / certified length 4.6 against 1.7 for solved ones).
* So the long-range share has an optimum below one half: a fifth keeps the near field everywhere but the windows, and
  lifts the crossings where the recipe stalled on long queries (forest, corridor).
* Seeds 1-2 of `ntflr20` queued 13:12 (queue restarted; 8 trainings, then 16 evaluations).
* Report v3 published (lede, ablation notes and next steps follow the fifth-of-pairs result).

## D9: seeds 1-2 of a fifth long-range (13:12-17:30)

* Queue finished 17:05 (8 trainings, 16 evaluations, no failure). Seeds 0 / 1 / 2, random / crossing success %:
  forest 85.2 / 81.7 / 81.9 and 54.3 / 53.9 / 52.8; corridor 86.0 / 84.0 / 80.6 and 70.1 / 67.9 / 62.5; scaffold
  62.0 / 58.9 / 57.5 and 26.1 / 24.0 / 19.9; windows 68.0 / 0.6 / 0.7 and 37.9 / 0.0 / 0.0 (seeds 1-2 stall on 99-100 %
  of the cases). Recipe means for comparison: forest 70.8 / 15.7, corridor 71.4 / 32.3, scaffold 63.6 / 13.2,
  windows 66.7 / 43.1.
* Forest and corridor replicate seed 0: every seed beats every recipe seed on both sets; on their queries where drone
  a travels >= 0.7, 60 % against 25 % (pooled over seeds). Fields (`field_diag_link.py`, seeds 1-2 added) are nearly
  identical across seeds: travel time per unit of path forest 1.88-1.92, corridor 1.99-2.01, scaffold 2.90-2.92, no
  uphill step.
* Scaffold: every seed below every recipe seed on random, above on crossings (barely: 19.9 against 19.7).
* Windows seeds 1-2: field collapsed (T_start / certified length 19.0-19.5, 32.5 % of the steps uphill, correlation
  0.69-0.72), the same state as the half share (19.7, 30.8 %, 0.73). The loss curves show where it goes wrong:
  every run starts at a total loss of 1.6-3.8e-2 and drops by about an order of magnitude once the field forms, fast
  (from above half its starting loss to below 4e-3 in 8-51 epochs; the scaffold recipe in 128-151). First epoch below
  4e-3: recipe 82-193 in all scenes and seeds; a fifth long-range 128-149 outside the windows. Windows with a fifth
  long-range: seed 0 drops at epoch 1340 (1.2e-2 at 1300, 2.9e-3 at 1400), seeds 1-2 are still at 1.1-1.2e-2 at epoch
  8000. Why the seed decides is open. The half share in the windows starts lower (9e-3) and only creeps to 4e-3 by
  epoch 8000, with no sharp drop.
* `aggregate_link.py`: per-seed fifth-long-range numbers (`ntflr20_per_seed`), training table for its seeds, plateau
  exit epochs (`escape`, first epoch below 4e-3). Report v4: lede, why-notes, a per-seed dot plot (recipe vs a fifth
  long-range), three-seed ablation notes, next step "make the share stable" (switch the long-range pairs on after the
  field has formed, or fine-tune a recipe checkpoint). Viewer: Results panel shows the per-seed column; checked on
  seeds 1-2 of every scene and set (no error), restarted.
