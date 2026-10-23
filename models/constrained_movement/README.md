# `models/constrained_movement`: a camera pyramid that only turns about the shape it follows (2026-10-22)

A camera follows a shape (the bird of the camera + object test) while the shape flies along its own path through a
scene. The camera is modelled by its view volume, a rectangle-based pyramid: the base rectangle is centred at the
shape's centre and the apex is the camera. The base centre is glued to the shape: it moves with the shape along the
path and never leaves it; the camera can only rotate about it. A collision-free pyramid is a camera that never
touches the scene and whose view of the shape is never blocked. Baseline model: the original metric NTFields.
Log: `EXPERIMENTS.md`; results: `results/constrained_movement/`.

## 1. Formulation

**Poses: t + rotation.** The shape's centre follows a path p(t), t the arc length in [0, L]. The camera's pose is
(p(t), R) with R the pyramid's rotation about the base centre, so the poses the problem allows are

    M = { (p(t), R) : t in [0, L], R in SO(3) } = [0, L] x SO(3),       a 4-d submanifold of SE(3),

with the product metric of every SE(3) cell in this repository, dt^2 + |omega / 2 pi|^2. The models do not see
xyz: they read t on the x axis of the chart, (t - L / 2, 0, 0, rotvec / 2 pi) (the path unrolled onto a line), so
M is a slice of SE(3) and the SE(3) machinery (network, speeds, Lie gradient, MPPI planner) carries over.

**Eikonal on M.** S(x) |grad_M T(x, y)| = 1 at both endpoints, grad_M = the SE(3) gradient with its translation
block projected onto the path direction (e_x in the chart) and the rotation block unchanged
(`models/path_constrained/losses.py`, the same loss as the path-constrained prism study). The speed S is the
clearance speed of the pyramid and multiplies motion along the path and rotation alike.

**Planning.** MPPI on (t, r) with tangent perturbations (dt, dr). In the forward mode every sample has dt >= 0
(the shape only moves forward: the camera-follow query); the free mode lets t move both ways.

## 2. Geometry (`geometry.py`, tests `tests/test_constrained_movement.py`)

* **Shape.** The bird of `dataprocessing/make_camera3d.py` (5 boxes: fuselage, head, wings, tail plane, fin;
  48 x 18 x 48 units, flying along +x) at a fixed orientation, positioned by the centre of its bounding box. Its
  clearance is exact (box-box distances of its parts).
* **Pyramid.** Body frame: base corners (+-W/2, +-H/2, 0) around the anchor (the bird's centre), apex (0, 0, D).
  The camera of the camera + object test: W x H = 48 x 22, D = 50 units (51 x 25 degree field of view), now with
  its base centred on the bird. In the env3 room shell (scale 220.457) that is 0.218 x 0.100 x 0.227.
* **Exact clearance.** Room slack of the 5 vertices; per obstacle box the convex distance when one of the 26
  candidate separating axes separates (pyramid vertex - box, box corner - pyramid face, 8 x 12 edge pairs), minus
  the penetration depth over those axes otherwise. Checked against dense surface and interior sampling of the
  pyramid in forest, city and env4 (never above the sampled distance, within 4e-3 of it, same sign), witness
  points on the scene boundary, 1-Lipschitz under translations.
* **Scenes.** Room box minus axis-aligned boxes: forest (4 trees, trunk + crown boxes from `forest_boxes.json`),
  city (6 buildings), corridor (serpentine walls, `corridor_boxes.json`), env1, env4. The box rows are stored in
  every cell and test set (`constraint.boxes_raw`).

## 3. The bird's path (`scripts/make_shape_paths.py`)

RRT-Connect in R^3 for the bird's centre (exact bird clearance >= 0.01 on every motion, resolution 0.003), random
shortcuts plus a greedy pass to the farthest visible waypoint, averaging that never moves a point below the
clearance, resampled at 1e-3. Endpoints are drawn in the room (forest: centre height 30..75 units, below the
crowns; city: 30..90, below the tall roofs), >= 0.6 apart (city 0.5), with >= 10 % of the camera's rotations free.
Each candidate is scored for the camera: the fraction of random rotations free at 64 stations ("camera room") and
the success of the straight motion (t linear, R slerp) between random endpoint rotations. One path per scene,
chosen with the camera room never below ~0.15 and a straight motion that mostly fails:

| scene | candidate | L | camera room mean / min | straight motion ok | bird flight |
|---|---|---|---|---|---|
| forest | 5 of 6 | 0.964 | 0.62 / 0.31 | 27.3 % | low between the trunks, around one tree |
| city | 1 of 4 | 0.747 | 0.67 / 0.35 | 10.2 % | over a low roof, past a tall building, between rows |
| corridor | 13 of 24 | 2.140 | 0.62 / 0.29 | 0.0 % | the whole serpentine (two U-turns) |
| env1 | 8 of 24 | 0.836 | 0.41 / 0.19 | 0.0 % | diagonally down between the upper and lower slabs |
| env4 | 13 of 24 | 0.824 | 0.59 / 0.15 | 2.3 % | from above one box down past two others |

## 4. Data, test sets, protocol

* Training cells (`scripts/preprocess_cmove.py`, 800k pairs): x0 = (uniform t, uniform R) in the clearance band
  (0.001, 0.05), x1 = x0 moved by a 4-d tangent displacement of length U[0, D], D = sqrt(max(L, 1)^2 + 3) (2 for
  paths up to L = 1, as the prism study; 2.75 for the 2.14-long corridor, whose whole-path queries a bound of 2
  never covers), kept if on the path; speeds by the formulas of preprocess_obj.py with the pyramid's exact clearance
  and its angular clearance about the anchor.
* Test sets (`scripts/make_cmove_tests.py`, 1000 cases each): `full` (t 0 -> L) and `seg` (sub-paths with
  t1 - t0 >= 0.25); endpoint rotations uniform with clearance >= 0.02; every case solved by a forward-monotone
  RRT-Connect on M.
* NTFields: `train/train_metric_path.py` (the bundle's trainer, combined speed, 8000 epochs, projected residual).
* Evaluation (`scripts/plan_eval_cmove.py`): success = converged to within 0.01 in sqrt(dt^2 + (theta / 2 pi)^2)
  with exact pyramid clearance >= 0 at every waypoint and 4 sub-steps between waypoints.

## 5. Results (epoch 8000; `results/constrained_movement/summary/summary.md`)

Forward mode (the bird only moves forward), 1000 RRT-verified cases per set, success = converged with the pyramid
collision-free throughout (exact geometry, 4 sub-steps between waypoints). Outcomes partition into success /
collision / stall without collision. Mean +- std over seeds 0 and 1 (corridor: seed 0, 860 MPPI steps).

| scene | whole path: straight | NTFields | collision | stall | sub-paths: straight | NTFields |
|---|---|---|---|---|---|---|
| forest | 26.6 | **89.5 +- 2.8** | 8.4 | 2.1 | 58.4 | **96.0 +- 1.7** |
| city | 12.5 | **91.7 +- 1.8** | 4.3 | 4.0 | 47.1 | **96.4 +- 0.2** |
| env1 | 0.1 | **79.8 +- 5.1** | 13.4 | 6.9 | 18.0 | **91.8 +- 1.7** |
| env4 | 4.7 | **86.5 +- 4.7** | 5.7 | 7.8 | 48.8 | **99.0 +- 0.4** |
| corridor (corridor2) | 0.1 | **63.2** | 12.8 | 24.0 | 24.6 | **94.2** |

* NTFields learns the constrained problem: 79.8-91.7 % of the whole-path queries in the short scenes and 63.2 % in
  the corridor, where the straight camera motion manages 0.1-26.6 %; 91.8-99.0 % of sub-paths.
* It fails where the camera has the least room. In forest, env1 and env4, 75-90 % of the collisions (both seeds)
  happen in the first quarter of the path, where the fraction of free camera rotations is lowest; in env4 a second cluster sits
  just before the room collapses at t / L = 0.7. The start of every path is also the end of M with the fewest
  training pairs, so the two effects coincide.
* The forward constraint costs nothing: the free mode is 0.6-4.1 points worse on whole paths (forest 87.7, city
  91.1, env1 78.7, env4 82.4) and 11.9 points worse in the corridor.
* Single runs are noisy on whole paths: seeds 0 / 1 give forest 86.8 / 92.3, city 89.9 / 93.5, env1 74.6 / 84.9,
  env4 81.8 / 91.2, and seed 0 moves by up to 9 points between epochs 2000 / 4000 / 8000 (env1 83.3 / 78.7 / 74.6).
  Sub-paths agree within 3.5 points across seeds.
* Corridor (L = 2.14, 860 MPPI steps, straight motion 0.1 %): with training pairs at most 2.0 apart, 4.2 / 8.0 /
  33.0 % at epochs 2000 / 4000 / 8000; with the bound sqrt(L^2 + 3) = 2.75 (`cm_corridor_a2`, the run the report
  shows), 25.8 / 49.5 / **63.2 %** (sub-paths 94.2 %; free mode 51.3 %: here the forward constraint helps).
  Stretching the pairs along t (`--t-stretch`, `cm_corridor_a3`) gives 19.6 / 41.8 / 24.0 % (no gain). The
  corridor's remaining failures stall at the start, where the camera has plenty of room. EXPERIMENTS.md C6-C7.

Report (static images): https://claude.ai/artifact/KDzupgRi3uqnEcp2Wc47Pe, built by `report/build_report.py`
from summary.json, the figures (`scripts/make_figures_cmove.py`: observer view rendered with `render.py`, camera
views, camera room and clearance along the plans) and `report/notes.json`.

## 6. Reproduce

```bash
python -m unittest tests.test_constrained_movement -v
python models/constrained_movement/scripts/make_shape_paths.py --env forest --y-band 30,75 --band-bias 0.7 --min-len 0.7
python models/constrained_movement/scripts/make_shape_paths.py --env forest --export 5          # -> cm_forest_a.npz
python3 models/constrained_movement/scripts/run_cmove_queue.py --scenes forest,city,corridor,env1,env4 --seeds 0,1,2
python3 models/constrained_movement/scripts/aggregate_cmove.py
python models/constrained_movement/scripts/make_figures_cmove.py --seed 0
python3 models/constrained_movement/report/build_report.py
```

## 7. Viewer (`scripts/cmove_viser.py`, port 8103)

Interactive viser viewer of the forward-mode plans: pick a test set (5 scenes x whole path / sub-paths) and a case
(filters: NTFields succeeds / collides / stalls, straight motion fails but NTFields succeeds, the seeds disagree),
then play the plan. The bird moves along its path, the pyramid turns about it (red while it touches the scene), and
the camera track is drawn coloured by clearance. A side panel shows what the camera sees (render.py,
the base rectangle dashed); "Look through camera" moves the browser's own viewport to the apex. Plans: NTFields seeds 0 / 1,
the test set's RRT reference and the straight motion, one to show and one to compare. The data come from
`scripts/export_viser_cmove.py` (torch; recomputes the evaluator's exact clearances, the collision flags agree on
every case) into `results/constrained_movement/viser/<run>_<variant>.npz`.

```bash
python models/constrained_movement/scripts/export_viser_cmove.py                   # torch image, GPU optional
docker run -d --name cm_viser --network host --restart unless-stopped -u 1006:1007 \
    -v /media/4tbssd/jeffrey/G-NTFields:/workspace -w /workspace bellman600-host:py3.11 \
    python -u models/constrained_movement/scripts/cmove_viser.py --port 8103
```
