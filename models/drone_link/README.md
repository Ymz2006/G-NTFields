# `models/drone_link`: a stretchable link between two drones, planned in 7 DOF with metric NTFields (2026-10-22)

Two drones fly through a scene and must keep their radio link: they must see each other (line of sight) and stay
within range. The pair is modelled as one body, a box that can stretch and shrink along one axis: the link box
from drone a to drone b. Planning that box collision-free keeps the line of sight clear at every moment, and the
length limits keep the drones within range and apart. This first study uses the original metric NTFields (the
bundle at `/media/4tbssd/jeffrey/export_metric`) in the 7-d configuration space SE(3) x [L_min, L_max]. Log:
`EXPERIMENTS.md`; results: `results/drone_link/`.

## 1. Formulation

**The body.** The link box `[-w/2, w/2]^2 x [0, L]` in its own frame: drone a at the body origin (centre of the end
face z = 0), drone b at `(0, 0, L)` (centre of the far face). A pose is `(t, R, L)`: `t` = drone a's position, `R` the
orientation with `R e_z = (b - a) / |b - a|`, `L = |b - a|`. The SE(3) part embeds drone a's position and the link
direction; the seventh coordinate is the length from drone a outwards. Width `w = 0.02` (4.4 room units),
`L in [0.06, 0.30]` (13 to 66 room units; the room is 0.90 x 0.63 x 0.99 in normalized units, 198 x 138 x 218 raw).

**Configuration space and chart.** `C = SE(3) x [L_min, L_max]`, 7-d. The networks read the chart
`x = (t, rotvec(R) / 2 pi, L)`: the SE(3) chart of every cell in this repository plus the length in normalized
room units, so the metric is `|dt|^2 + |d theta / 2 pi|^2 + dL^2` (stretching the link by dL costs as much as moving
drone a by dL).

**Clearance, and the length limits as walls.** The exact signed distance between the link box and the scene
(`geometry.scene_clearance`: oriented box against axis-aligned boxes, separating-axis overlap when they intersect;
the generalization of `path_constrained.geometry.prism_clearance` to a length per pose), and

    c(t, R, L) = min(c_scene(t, R, L), L - L_min, L_max - L).

`c >= 0` means: neither drone touches the scene, the line of sight between them is clear, and the length is
allowed. The limits on how far the link stretches and shrinks are walls of the seventh coordinate, exactly like the
room walls are walls of the translation.

**Speed, re-sampled in 7 DOF.** The formulas of `preprocess_obj.py` with this clearance:
`speed_dists = clip(c / margin, offset / margin, 1)` (margin 0.05, offset 0.001), `speed_angles = angle / pi` with
the angular clearance about drone a (the smallest angle between a link surface point and a scene surface point in
the same radial shell; 10 shells of the box of that length, `geometry.LinkAngle`), and `speed` = their mean (the
speed the metric NTFields trains on). The normal is the unit 7-d gradient of `c` (`geometry.normal7`; translation
= the workspace normal, rotation = 2 pi x its moment about drone a, length = `zeta n . R e_z` for a witness point at
fraction `zeta` of the box; `-+e_L` where a length limit binds), checked against finite differences.

**Model and planner.** The bundle's metric NTFields, unchanged, with `dim = 7` (Fourier features `B in R^{128 x 7}`,
the Eikonal residual over all 7 chart coordinates), trained with the full recipe of the camera test (TD and
normal-alignment terms 1e-3, 8000 epochs; `train/train_metric_link.py`). Planning: the MPPI of every SE(3)
evaluation (50 samples, horizon 5, step 0.015, temperature 50) in the 7-d chart, the length clamped to
`[L_min, L_max]` in every rollout and step (`scripts/plan_eval_link.py`, 600 iterations).

## 2. Scenes, data, test sets, protocol

* Scenes (`dataprocessing/make_link_scenes.py`), all in env3's room shell (the normalization of every shape cell):
  `forest` (the camera test's forest, unchanged: four trees, 12 x 12 trunks to y = 96, 30-unit crowns at y 90..120),
  and three new ones that break the line of sight in different ways: `windows` (a partition wall with two 40 x 40
  windows, one low and one high: two rooms that see each other only through the windows), `corridor` (two
  full-height walls make one serpentine corridor of three 67-unit legs joined by 54-unit turns: line of sight
  around corners), `scaffold` (a lattice of 3 x 3 columns and beams across the room at two heights).
  Obstacle boxes in `datasets/3dshape/obj/<scene>_boxes.json` (the forest's regenerated from `make_camera3d.py` and
  checked against `forest.obj`).
* Training cells `datasets/3dshape/data/link_<scene>` (`scripts/preprocess_link.py`, 800k pairs, 3-5 min each):
  x0 = (t uniform in the room, R uniform on SO(3), L uniform) in the narrow band `0.001 < c < 0.05`; x1 = x0
  displaced in the 6-d SE(3) chart as in `preprocess_obj.py` (uniform direction, length U[0, sqrt 6]) with a new
  uniform length, kept if drone a is in the room and `c > 0.001`. 28-44 % of the band samples are bound by a length
  limit.
* Test sets `testing_data/3dshape/link_<scene>_{std,cross}` (1000 cases each, endpoint clearance >= 0.02; cases
  drawn by `cases.py`): `std` = uniform endpoints; `cross` = the scene's crossing task with both drones in one region
  at the start and the other at the goal (forest: in front of the trees <-> behind them; windows: front room <-> back
  room; corridor: first leg <-> last leg; scaffold: x < 30 <-> x > 170). Every case is certified solvable in the 7-d
  space by a path whose motions keep the exact clearance >= 0.002: `scripts/make_link_tests_prm.py` (a 60k-node
  roadmap per scene on the GPU, Dijkstra, shortcut; used for all eight sets) or `scripts/make_link_tests.py` (one
  RRT-Connect per case on the CPU; the same cases, too slow on the shared machine; its corridor sets are kept as
  `link_corridor_{std,cross}_rrt`).
* Success (primary): converged to within 0.01 in `sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2)` and `c >= 0` at every
  waypoint and 4 sub-steps between waypoints: the drones kept contact (line of sight and range) and separation the
  whole way. Contact fraction: share of the checked poses with the link clear.
* Baselines (`scripts/baseline_link.py`): `straight` = the direct 7-d motion (drone a linear, R slerp, L linear);
  `independent drones` = each drone flies its own shortest collision-free path (point drone with clearance 0.015,
  26-connected grid of spacing 0.01 searched on the GPU, then string pulling), both leaving and arriving together,
  the link ignored: scored by the share of time with line of sight and range, and whether the whole motion is valid.

## 3. Results

Full tables: `results/drone_link/summary/summary.md` (`scripts/aggregate_link.py`); log: `EXPERIMENTS.md`; report page
https://claude.ai/artifact/QWuiacZDu1HVpRb5ioDHiq (`report/build_report.py`). Percent of 1000 certified cases solved
with the link valid the whole way (drones reach their goals, never touch the scene, always see each other, keep an
allowed distance); NTFields = mean +- std of seeds 0-2.

| scene | set | direct 7-d motion | independent drones | NTFields 7 DOF |
|---|---|---|---|---|
| forest | random | 47.3 | 55.0 | 70.8 +- 2.2 |
| forest | crossing | 11.9 | 24.5 | 15.7 +- 3.1 |
| windows | random | 51.1 | 55.3 | 66.7 +- 2.3 |
| windows | crossing | 2.3 | 21.4 | 43.1 +- 4.9 |
| corridor | random | 32.6 | 38.1 | 71.4 +- 2.6 |
| corridor | crossing | 0.0 | 2.7 | 32.3 +- 4.1 |
| scaffold | random | 27.1 | 41.8 | 63.6 +- 1.3 |
| scaffold | crossing | 2.0 | 11.4 | 13.2 +- 5.2 |

* **The 7-DOF formulation works end to end.** The speed re-sampled with the length (exact clearance with the length
  limits as walls, the original speed formulas) trains the metric NTFields at dim 7, and its plans use the seventh
  coordinate: 36-58 % of the solved cases shorten the link below both endpoint lengths to get past an obstacle.
* **Link-aware planning keeps contact that independent drones lose.** Each drone on its own shortest path keeps
  contact 87-96 % of the time on average, but loses the line of sight at least once in 23-90 % of the cases and comes
  too close in 16-46 %; around corners (corridor crossings) its whole motion is valid in 2.7 % of the cases against
  32 % for NTFields. A plan NTFields completes keeps contact 100 % of the time.
* **NTFields fails on long queries.** Success falls with drone a's travel: 94 % below 0.3, 85 % at 0.3-0.5, 65 % at
  0.5-0.7, 43 % at 0.7-0.9 (forest random, seed 0). Most failures are stalls in local minima of the learned field
  (forest) or a wall crossing followed by a stall (corridor). Not the planner (4x samples, 3x horizon: 12 -> 14.5 %);
  the field along the certified paths is right on average (correlation 0.95-0.99) but bumpy where stalls happen; the
  training pairs move drone a more than 0.7 in only 3 % of the pairs. Training longer helps the crossings most (4000 ->
  8000 epochs: 19.0 -> 26.1 % mean). In the open forest, independent drones do better on crossings (24.5 vs 15.7 %).
* **The TD and normal-alignment terms are required in 7 DOF** (seed-0 ablation, the bundle's default loss without
  them): training fails in forest, windows and corridor (0.2-0.5 % random, 0 % crossings; Eikonal residual
  4.5-5.3e-3 against 1.0-1.3e-3). The field comes out slow and bumpy: travel time 15-21 per unit of certified path
  (1.4-2.0 with the terms), 32-38 % of the steps along it uphill. The scaffold trains either way (59.8 vs 65.4 %).
* **A fifth of the training pairs long-range fixes the forest and the corridor but is unstable in the windows**
  (`--x1-uniform 0.2`, cells `link_<scene>_lr20`, seeds 0-2 against the recipe's seeds 0-2, means). Forest crossings
  15.7 -> 53.7 %, random 70.8 -> 82.9 %; corridor crossings 32.3 -> 66.8 %, random 71.4 -> 83.5 %: every seed beats
  every recipe seed, the fields come out alike in all seeds and clean (no uphill step along the certified paths of
  random cases; travel time 1.9-2.0 per unit of path against 1.4-1.5). Scaffold: every seed trades random queries for
  crossings (63.6 -> 59.5 %, 13.2 -> 23.3 %; collisions 26.7 -> 36.0 %). Windows: seed 0 is in the recipe's range
  (68.0 / 37.9 %), seeds 1 and 2 solve 0.6-0.7 % / 0 %. Every recipe and fifth-long-range run starts on a loss plateau
  (1.6-3.8e-2) and drops by an order of magnitude once the field forms: the recipe by epoch 82-193, a fifth long-range
  by epoch 128-149 outside the windows; windows seed 0 only at epoch 1340, seeds 1-2 never (1.2e-2 at epoch 8000).
  Their fields are the half share's collapsed field (19.0-19.5 per unit of path, 32.5 % of the steps uphill). Next:
  switch the long-range pairs on after the field has formed, or fine-tune a recipe checkpoint with them.
* **Half of the pairs long-range breaks the field outside the corridor** (`--x1-uniform 0.5`): corridor random 72.9 ->
  84.5 %, crossings 38.0 -> 72.1 % (queries with drone a moving >= 0.7: 37 -> 73 %); forest 72.9 -> 14.0 %, windows
  69.9 -> 1.1 %, scaffold 65.4 -> 32.3 % on the random sets, with the same slow field as without the TD / normal terms
  (travel time 7-20 per unit of path).
* **The x0 band on the scene clearance only** (no x0 placed at a length limit, `ntfsb`) changes nothing beyond seed
  noise: random -3.3 to +2.1 points, crossings -2.3 to +3.0 points.

## 4. Reproduce

```bash
python dataprocessing/make_link_scenes.py
python -m unittest tests.test_drone_link -v
python models/drone_link/scripts/preprocess_link.py --scene forest --out datasets/3dshape/data/link_forest
python models/drone_link/scripts/make_link_tests_prm.py --cell datasets/3dshape/data/link_forest --variant cross --n 1000
python /workspace/train/train_metric_link.py --dataPath /workspace/datasets/3dshape/data/link_forest \
    --name forest_ntf_s0 --seed 0 --td-weight 1e-3 --normal-weight 1e-3 --out_root /workspace/Experiments/drone_link   # cwd /export, NTF_NO_PLOT=1
python models/drone_link/scripts/plan_eval_link.py --checkpoint Experiments/drone_link/forest_ntf_s0/latest.pt \
    --dataPath testing_data/3dshape/link_forest_cross --out results/drone_link/plan/forest_ntf_s0/link_forest_cross.json
python models/drone_link/scripts/baseline_link.py --dataPath testing_data/3dshape/link_forest_cross \
    --out results/drone_link/baselines/link_forest_cross.json
python3 models/drone_link/scripts/run_link_queue.py            # everything (host, docker), resumable
python3 models/drone_link/scripts/aggregate_link.py; python models/drone_link/scripts/make_figures.py
```

## 5. Viewer

`scripts/link_viser.py` (viser, numpy only; container `link_viser`, image `bellman600-host:py3.11`, port 8110):
two rooms side by side, the planner's plan on the left and the same case flown by a link-unaware motion on the right
(independent drones, the straight 7-d motion or the certified roadmap path), one time slider and Play for both.
Drone a blue, drone b orange, their tracks red where the link is broken; the link box green while the drones see each
other within range, red where it breaks, amber where they are closer than L_min. The panel gives the case's outcome
and plots the link length over normalized time with the allowed band and both contact timelines. Filters: the cases
the planner solves, breaks, stalls on, those it solves and the independent drones lose (fewest contact first), those
where it shortens the link, the long trips; any planner run or seed in `results/drone_link/plan` (missing runs are
looked up again, so new evaluations appear while the queue runs). The baselines' link status is a numpy
separating-axis test of the link box; it matches the sign of `geometry.scene_clearance` on 80k random poses and
reproduces `baseline_link.py`'s recorded contact / valid fractions exactly.

```bash
docker run -d --name link_viser --network host --restart unless-stopped -v $PWD:/workspace -w /workspace \
    bellman600-host:py3.11 python -u models/drone_link/scripts/link_viser.py --port 8110
python models/drone_link/scripts/link_viser.py --scene windows --set crossing --case 73 --t 0.5   # start state
```
