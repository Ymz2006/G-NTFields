# `models/path_constrained`: a prism glued to a path (camera follow), homotopy head vs metric NTFields (2026-10-21)

A camera follows a moving object and must keep it in view. The object is a point moving along a given path; the
camera's line of sight is a small square prism (side w = 0.02, length h = 0.2) with one end glued to the object
and the other end at the camera. The prism may turn freely about the glued end; its start and end orientations
are given. A collision-free prism is a camera whose view of the object is never blocked and which never touches the
scene. Log: `EXPERIMENTS.md`; results: `results/path_constrained/`.

## 1. Formulation

**The pose space is a slice of SE(3).** With the prism's body frame at the glued end, the anchor position is the
pose's translation and every rotation about the anchor keeps it on the path. The poses the problem allows are

    M = { (p(l), R) : l in [0, L], R in SO(3) }  =  [0, L] x SO(3),       a 4-d submanifold of SE(3),

l the arc length along the path. M inherits the SE(3) product metric of every cell in this repository,
dl^2 + |omega / 2 pi|^2 (translation in normalized units, rotation in turns).

**The path unrolled onto a line (arclength chart).** The models are fed t = (l - L / 2, 0, 0) instead of p(l):
the path straightened onto the x axis. Then M is literally the set of SE(3) poses whose translation lies on a
segment of the x axis, with the same metric. It is SE(3) planning restricted to that slice. Every piece of the
SE(3) machinery carries over unchanged: the networks, the speeds, the Lie gradient, the double cover of SO(3), and
the MPPI planner. The real anchor position p(l) is only needed to compute speeds and collisions. The world chart
t = p(l) (the real position, the path embedded in the room) is the ablation.

**Eikonal on M.** The travel time solves S(x) |grad_M T(x, y)| = 1 at both endpoints, where grad_M is the SE(3)
gradient with its translation block projected onto the path tangent (the only translation M allows; e_x in the
arclength chart) and the rotation block unchanged:

    |grad_M T|^2 = (tau . dT/dt)^2 + |rotation block|^2.

This is the "xyz speed along the direction of the path" of the request. The speed S is the clearance speed of the
prism, computed exactly as for every SE(3) cell. It multiplies the whole tangent space of M, both motion along the
path and rotation. It cannot multiply the rotation alone: with the metric dl^2 + |dtheta|^2 / S^2, sliding along
the path costs dl regardless of clearance, so a plan could pass through an obstacle by translating. Training the
fields only on M is what makes this the constrained problem rather than free SE(3) planning: pairs, speeds and
residuals all live on M.

**Planning.** The MPPI of the SE(3) evaluations (50 samples, horizon 5, step 0.015, temperature 50) runs on
(l, r) with tangent perturbations (dl, dr). In the forward mode every sample has dl >= 0 and l <= l_goal: the
object only moves forward, which is the camera-follow constraint. The fields are symmetric, so monotonicity is a
planner constraint; the free mode (l anywhere in [0, L]) shows how often each field's own descent would need the
object to move backwards. The homotopy head plans on the double cover as in SE(3): `dual` plans to both lifts of
the goal (the two homotopy classes of rotation paths) and keeps the converged plan.

## 2. Data, test sets, protocol

* Scenes env1, env3, env4 (no env2 mesh exists on this machine): room box + axis-aligned boxes. Clearance and
  collision are exact (oriented box against boxes; `geometry.py`). The 10k-point environment sampling of the SE(3)
  cells has ~0.02 spacing, as wide as the prism, so a sampled test would let it pass through walls.
* Paths (`scripts/make_paths.py`): a centripetal Catmull-Rom spline through random via points, with point
  clearance >= 0.04. Paths were chosen so the straight motion (l linear, R slerp) between random endpoint
  orientations usually collides. One main path per scene (a), plus a second one (b).
* Training cells (`scripts/preprocess_path.py`, 800k pairs): the sampler of preprocess_obj.py restricted to M.
  x0 = (uniform l, uniform R) in the clearance band (0.001, 0.05); x1 = x0 moved by a 4-d tangent displacement of
  length U[0, 2]. speed_dists, speed_angles and the combined speed use the formulas of preprocess_obj.py (the
  angular clearance is checked against the original implementation).
* Test sets (`scripts/make_path_tests.py`, 1000 cases each): `full` (l 0 -> L, the whole path) and `seg`
  (sub-paths, l1 - l0 >= 0.25). Endpoint orientations are uniform on SO(3) with clearance >= 0.02, and every case
  is solved by a forward-monotone RRT-Connect on M.
* Models: graph_homotopy (`train/train_homotopy_path.py`, the graph_lie pipeline, `--lift min`, isotropic speed,
  8000 epochs) and the original metric NTFields (`train/train_metric_path.py`, the bundle's trainer, combined
  speed, 8000 epochs). Each keeps its own recipe; the only change is the projected Eikonal residual
  (`losses.py`).
* Evaluation (`scripts/plan_eval_path.py`): success = converged to within 0.01 in sqrt(dl^2 + (theta / 2 pi)^2)
  with exact clearance >= 0 at every waypoint and at 4 sub-steps between waypoints.

## 3. Results

Full tables: `results/path_constrained/summary/summary.md` (`scripts/aggregate_path.py`); figures:
`results/path_constrained/figures/` (`scripts/make_figures.py`); log: `EXPERIMENTS.md` P5. Every test case is
RRT-verified solvable with the object only moving forward.

SE(3) success %, forward mode (the camera-follow query), main cells (path a of env1 / env3 / env4, prism 0.02 x
0.2, 1000 cases per set, 3 seeds), mean over the three scenes. "straight" = the direct motion (l linear, R slerp
the short way) is collision-free.

| set | straight | homotopy, dual planner | homotopy, base planner | NTFields |
|---|---|---|---|---|
| full (whole path) | 9.0 | **99.7** | 97.4 | 91.5 |
| tight (whole path, both ends 0.005-0.02 from the scene) | 14.3 | **99.8** | 98.5 | 85.9 |
| bigrot (whole path, >= 150 deg turn) | 11.0 | **99.7** | 97.4 | 91.7 |
| hard (tight and >= 135 deg) | 14.7 | **99.8** | 98.8 | 86.3 |
| seg (sub-paths, l1 - l0 >= 0.25) | 50.7 | **99.7** | 98.1 | 98.5 |

Per scene, whole path: env1 100.0 / 100.0 / 92.4 +- 3.2, env3 100.0 / 100.0 / 92.3 +- 2.2, env4 99.0 / 92.2 / 89.9
+- 4.0 (homotopy dual / base / NTFields). Camera farther away (prism length 0.3, seed 0), whole path / hard:
env1 100.0 / 99.4 vs 94.0 / 86.6, env3 100.0 / 98.7 vs 83.7 / 74.0, env4 99.3 / 99.5 vs 83.4 / 79.3. Second path
per scene (seed 0, whole path): env1 100.0 vs 95.5, env3 98.1 vs 98.1, env4 99.9 vs 92.3.

* **The homotopy head solves the constrained problem.** With the dual planner it reaches 99.0-100 % on every set
  of the main cells. Its plans never block the line of sight: 0 collisions in every forward-mode evaluation of the
  main cells, and its few failures (<= 1 %) are stalls. Paired per case, on the whole-path sets it solves 225-514
  cases (of 3000) that NTFields misses, against 0-21 the other way (p < 1e-58).
* **NTFields fails at the start of the path.** Its whole-path success is 82-92 %, with 2.8-7.2 % collisions and
  7.5-16.8 % stalls. The collisions happen in the first 1-19 % of the path (median) and the stalls stop by 4-12 %:
  the field is weakest near the end l = 0 of M, where training pairs are sparsest. On sub-paths, NTFields fails
  3-7x more often when the start lies in the first 10 % of the path. Tight endpoints cost it another 4.6-7.5 points.
  The homotopy head's analytic cone keeps its field right there.
* **The gap grows with difficulty.** With the prism length 0.3 the straight motion works in 1-5 % of the cases;
  the homotopy head stays at 98.7-100 % while NTFields drops to 74-94 %.
* **Planning both homotopy classes matters where obstacles separate them.** On env4 the base planner stalls on
  7.8 % of the whole-path cases, all at the same place (l ~ 0.58 L); planning both lifted goals and keeping the
  converged plan brings it to 99.0 % (the oracle over the two sheets is 99.0 %). On env1 / env3 one sheet suffices.
* **The forward constraint is free.** Forward and free modes give the same success (within 0.8 points). In the free
  mode 0.2-11 % of whole-path plans and 6-19 % of sub-path plans move the object backwards, so a camera-follow
  controller should plan in the forward mode.
* **Unrolling the path is the better chart.** The homotopy head is insensitive to the chart (arclength vs world:
  dual planner within 0.4 points, base planner within 2.3). NTFields loses 1.5-7.2 points on whole paths when it is
  fed the real anchor position instead of the unrolled arc length.
* Training length: 4000 and 8000 epochs agree within 1.2 points for both models.

## 4. Reproduce

```bash
python -m unittest tests.test_path_constrained -v
python models/path_constrained/scripts/make_paths.py --env env1                     # candidates + scores
python models/path_constrained/scripts/make_paths.py --env env1 --export 63 40      # -> datasets/3dshape/paths/env1_{a,b}.npz
python models/path_constrained/scripts/preprocess_path.py --path datasets/3dshape/paths/env1_a.npz --out datasets/3dshape/data/prism_env1_a
python models/path_constrained/scripts/make_path_tests.py --cell datasets/3dshape/data/prism_env1_a --variant full
python train/train_homotopy_path.py --dataPath datasets/3dshape/data/prism_env1_a --name e1a_hom_s0 --seed 0
python /workspace/train/train_metric_path.py --dataPath /workspace/datasets/3dshape/data/prism_env1_a --name e1a_ntf_s0 --seed 0   # cwd /export
python models/path_constrained/scripts/plan_eval_path.py --kind graph --planner dual --mode forward \
    --checkpoint Experiments/path_constrained/e1a_hom_s0/latest.pt --dataPath testing_data/3dshape/prism_env1_a_full --out <json>
python3 models/path_constrained/scripts/run_path_queue.py          # everything (host, docker), resumable
python3 models/path_constrained/scripts/aggregate_path.py; python models/path_constrained/scripts/make_figures.py
```

## 5. Viewer and report

* `scripts/export_viser_path.py` (torch, container) writes every test set's forward-mode plans (homotopy dual and
  NTFields, seed 0) with exact per-waypoint and per-motion clearance to `results/path_constrained/viser/<set>.npz`
  (27 files, ~2 min each on CPU).
* `scripts/path_viser.py` (numpy + viser only) is the viewer, in the style of `dataprocessing/shapes3d_viser.py`:
  test set and outcome filter, both plans on one clock, the line of sight drawn as the view pyramid (apex at the
  camera, base the w x w square around the object; it lies inside the prism) or the prism, red where the prism
  touches the scene, camera track and frustum, a **camera view** panel (the scene rendered from the camera end
  looking at the object, `ClientHandle.get_render`, adjustable field of view), "Look through camera" for the whole
  viewport, opaque obstacles, a clearance plot. Running in container `pc_viser_path` on port 8102 (`--restart
  unless-stopped`). `--look --opaque --dump DIR --strip N` writes N camera-view frames per model of the first case.
* Report page (static images): https://claude.ai/artifact/1PnNpLTDBQdiyWWekf4jXY, source `report/index.html`.
