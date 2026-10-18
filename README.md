## About
This is a minimal example. 

## Setup
1. git clone this repo
2. run `docker build -t ntrl:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run -u $(id -u):$(id -g) --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/home/n/Eikonal_Planning/ntrl-demo:/workspace" --volume="/usr/lib/x86_64-linux-gnu/:/glu" --volume="/home/n/.local:/.local" --env="QT_X11_NO_MITSHM=1"  --gpus all -ti --rm ntrl:demo` to start the docker container.
4. run `pip install scipy` inside the container to install the KD-tree dependency
5. run `python dataprocessing/preprocess.py --config configs/gibson.txt ` to sample training data
6. run `python train/train_gib.py` to start the training.

## SE(3) smooth graph travel-time fields (`models/graph_method`)

Implementation of `SE3_Smooth_Graph_Implementation.pdf`: a smooth latent
(a, b) predicted by the G-Shape encoder + an explicit analytic decoder
T = sqrt(a/s_t^2 + phi(b)/s_r^2), trained with intrinsic endpoint Eikonal and
Bellman losses on SE(3). See `models/graph_method/README.md` (what is
implemented, results, how to reproduce) and `models/graph_method/EXPERIMENTS.md`
(log of every experiment). Entry points: `train/train_graph.py`,
`python -m models.graph_method.evaluate`, `python -m unittest tests.test_graph_method`,
configs in `configs/graph_method/`.

## Triangle inequality by construction (`models/graph_method_triangle`)

A copy of `models/graph_method` with pair heads whose decoded time is a pseudometric
for every parameter value. A shared encoder gives each pose (u(x), V(x)) with V in SO(3),
a = |u_x - u_y|^2 (or an exact seminorm m(x, y)^2) and b = |V_x - V_y|_F^2 / 8 =
(3 - tr V_x^T V_y) / 4, decoded by the unchanged analytic decoder. See
`models/graph_method_triangle/README.md` (findings, results, how to reproduce) and
`EXPERIMENTS.md` (log T0-T9; results dashboard https://claude.ai/artifact/LHyUtVzc8h9u4KgKKvz4ts). Entry points: `train/train_graph_triangle.py`,
`python -m models.graph_method_triangle.evaluate`, `python -m unittest tests.test_graph_method_triangle`,
`models/graph_method_triangle/scripts/triangle_check.py`, configs in `configs/graph_method_triangle/`.

## Cusp / seam study vs. the original method (2026-10-12)

`results/graph_method_cusp/REPORT.md`: per-case MPPI planning of every head against the original
metric NTFields (chart input) on the standard and rotation-heavy test sets, with chart and SE(3)
convergence, seam-crossing analysis, path lengths, a head-to-head inside the original trainer
(`models/graph_native/`) and obstacle-free planning by relative angle. Summary tables in
`results/graph_method_cusp/summary/`, figures in `results/graph_method_cusp/figures/`.

## One cone, one ridge: the double-cover head (`models/graph_homotopy`, 2026-10-17)

`graph_homotopy` (`models/graph_method_triangle/homotopy_decoder.py`): the branch decoder reduced to its goal cone and
one ridge. The travel-time field is first defined on the double cover (R^3 x S^3) x (R^3 x S^3), deck-invariant by
construction (built from the invariants of flipping both quaternions), and the time between poses is the minimum over
the two lifts of the goal, i.e. over the two homotopy classes of paths; planning descends one sheet of the cover
towards +q_goal or -q_goal (`models/graph_homotopy/scripts/plan_eval_homotopy.py`, planners base / cover / dual).
With the base field's own training loss (8000 epochs) and the dual cover planner: SE(3) success T 98.9 +- 0.3,
L 98.8, A 99.1 (base planner on the same fields 97.0 / 97.8 / 98.4; the conformal branch decoder K = 4 retrained on
the same machine 94.7 +- 0.4 on T). See `models/graph_homotopy/README.md` and `EXPERIMENTS.md`; entry points:
`train/train_graph_homotopy.py`, `python -m unittest tests.test_graph_homotopy`, `models/graph_homotopy/scripts/`.

Second request (2026-10-17): targeted test sets (long paths, big rotations, tight endpoints; `testing_data/3dshape/README.md`),
rooms divided by a wall with a shrinking hole (`datasets/3dshape/data/hole{90,75,66}.obj`), and the original metric NTFields
retrained here (`train/train_metric_baseline.py`) as the baseline. Homotopy (8000 epochs, dual planner) vs NTFields (8000
epochs), SE(3) success on T / L / A: standard 98.9 / 98.9 / 98.9 vs 82.4 / 90.4 / 88.2; hard sets 96.7 / 97.8 / 96.5 vs
54.3 / 65.0 / 60.6; through a hole 1.1x the plate 85.2 / 93.5 / 81.8 vs 52.4 / 63.6 / 42.8. See `models/graph_homotopy/README.md`
section 4, `EXPERIMENTS.md` H11-H12 and `results/graph_homotopy/summary/expand_summary.md`.
