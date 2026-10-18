# `models/graph_lie`: Lie-gradient Eikonal training of the graph heads

This package trains the two best heads of this repository, `graph_metric` and the triangle head
(`graph_tri_metric` with equivariant V), with the Eikonal residual written in the **Lie gradient** of
the travel time on SE(3) instead of its **partial derivatives** in the rotation-vector chart. It does
this for the isotropic and the anisotropic formulation of Lie-Shape-NTFields, plus the combined-speed
formulation of the original method, and compares every case against the same model trained with the
partial derivatives.

Everything else is the original metric NTFields pipeline as used by `models/graph_native`: same data,
network, AdamW, 5 batches of 2000 per epoch, accept/reject rollback, `exp(-tau/2)` weighting,
`1e-2 (sqrt(N) - 1)^2` residual, 4000 epochs, and the same per-case MPPI evaluator. With the combined
formulation and the chart gradient this package reproduces `graph_native` runs bit for bit
(`EXPERIMENTS.md` L0), so the existing `graph_native` runs are reused as members of the comparison.

Results: `EXPERIMENTS.md` (log and findings), `results/graph_lie/summary/summary.md` (all tables,
generated), `results/graph_lie/figures/`.

## The change

A pose is stored as `x = (t, r)` with `r = rotvec / 2 pi` (turns), `R = exp(2 pi [r]_x)`. The graph
heads see only `G(x) = (t, vec R)`, so `tau(x, y)` is a function on SE(3) x SE(3).

| | endpoint gradient | what `|grad| = 1/S` means |
|---|---|---|
| chart (the original method) | `g = (d tau/d t, d tau/d r)` | unit speed in the flat metric `|dt|^2 + |dr|^2` of the rotation-vector ball |
| **Lie** | `c_i = d/de tau(x exp(e E_i))`: `c_v = R^T d tau/d t`, `c_w = 2 pi vee(A - A^T)`, `A = R^T d tau/d R` | unit speed in the left-invariant metric `|v|^2 + |w / 2 pi|^2` of body twists |

`c` is computed exactly from one autograd call with respect to `G` (no chart, no seam;
`lie.endpoint_gradients`, using `graph_method_triangle/geometry/tangents.py`). By the chain rule
through `R = exp(2 pi [r]_x)`,

    c_v = R^T g_v,        c_w = Jr^{-1}(2 pi r)^T g_w,

so the translation block keeps its norm and only the rotation block changes. In the plane orthogonal
to `r`, `|c_w| / |g_w| = (theta/2) / sin(theta/2)` with `theta = |rotvec|`: 1 at the identity, 1.11 at
90°, 1.27 at 134°, pi/2 = 1.57 at 180°. In the training scenes the median start rotation is 86° and a
quarter of the samples are beyond 134°, so this is not a corner case.

Why the Lie form is the right equation (and the tests that check it, `tests/test_graph_lie.py`):

* **The chart metric depends on the absolute orientation.** It charges a turn about an axis
  orthogonal to the current rotation vector up to pi/2 times more than the same turn at the identity.
  The identity of the chart is an arbitrary choice of world frame, so this is not physics. The Lie
  metric is the same at every pose and does not depend on the world frame.
* **The chart equation has no solution among functions on SE(3).** With all speeds 1, the exact
  travel time is the geodesic distance `sqrt(|dt|^2 + (theta/2 pi)^2)`. It satisfies `|c| = 1` at
  every pose and violates `|g| = 1` once `|rotvec|` is large
  (`test_free_space_geodesic_solves_lie_eikonal_but_not_chart`). The graph heads are always functions on
  SE(3), so under the chart loss they can only compromise.
* The Lie loss is finite and smooth at `|rotvec| = pi`, where the chart Jacobian is singular
  (`test_lie_loss_is_finite_at_the_chart_seam`).

## Speed formulations (`model_function_lie.py`)

Per endpoint, with `grad = (grad_v, grad_w)` either `g` (chart) or `c` (Lie):

| `--speed` | Eikonal `N = 1` | origin |
|---|---|---|
| `combined` | `S |grad|` | metric_baseline / `models/graph_native` (combined clearance `S = (S_d + angle/pi)/2`, warp `s^2(2-s)^2`) |
| `iso` | `S_d sqrt(|grad_v|^2 + |grad_w|^2)` | Lie-Shape-NTFields `lie_isotropic` (translational clearance) |
| `aniso` | `sqrt(S_d^2 |grad_v|^2 + S_a^2 |grad_w|^2)` | Lie-Shape-NTFields `lie_metric` (rotational clearance `S_a`) |

`S_d` and `S_a` are preprocessed exactly as in the Lie trainers (`model_train_lie.warp_speeds`; the
rotational clearance is cut at pi/6 and renormalised before the warp, without which `S_a` would have a
median of 0.009). The loss is `1e-2 (sqrt(N) - 1)^2` per endpoint, weighted by `exp(-tau/2)`.

## Layout

| file | content |
|---|---|
| `lie.py` | chart and Lie endpoint gradients, exp-perturbation reference, `Jr^{-1}`, chart -> Lie map |
| `model_function_lie.py` | `Function`: Eikonal loss for grad mode x speed model; TravelTimes / Speed / Gradient queries |
| `model_train_lie.py` | trainer (copy of `graph_native`'s, all three speeds passed) and `warp_speeds` |
| `../../train/train_graph_lie.py` | CLI: `--arch`, `--grad chart|lie`, `--speed combined|iso|aniso`, `--set` head options |
| `../../tests/test_graph_lie.py` | 11 tests: gradient identities, bitwise equality with `graph_native`'s loss, speed preprocessing parity with the Lie trainers, free-space exactness, checkpoint contract |
| `scripts/run_queue.py` | host-side multi-GPU queue (train in containers, then evaluate), resumable |
| `scripts/field_diag.py` | which Eikonal equation does a trained field satisfy, by `|rotvec|` |
| `scripts/make_free_dataset.py`, `scripts/free_eval.py` | obstacle-free control with exact ground truth |
| `scripts/aggregate.py`, `scripts/make_figures.py` | tables (paired Lie − chart statistics) and figures |

Checkpoints keep the `graph_native` contract (they add `speed_model`), so the per-case evaluator
loads them with `--kind graph_native`.

## Reproduce

Inside `pytorchserver:latest` with this repo at `/workspace` (and Lie-Shape-NTFields read-only at
`/lieshape` for evaluation):

```bash
python -m unittest tests.test_graph_lie
python train/train_graph_lie.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_gmetric_iso_lie_s0 \
    --seed 0 --arch graph_metric --grad lie --speed iso
python train/train_graph_lie.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_trieq_aniso_lie_s0 \
    --seed 0 --arch graph_tri_metric --grad lie --speed aniso --set rot_mode=equivariant
# the whole study (host side; 108 runs, 92 trained; reuses the bitwise-identical graph_native runs)
python3 models/graph_lie/scripts/run_queue.py --phase all --per-gpu 4 --eval-slots 3
python3 models/graph_lie/scripts/make_free_dataset.py --out datasets/3dshape/data/free_se3
python3 models/graph_lie/scripts/run_queue.py --phase free --per-gpu 4 --eval-slots 0
python models/graph_lie/scripts/free_eval.py --oracle Experiments/graph_lie/F_*/latest.pt
python models/graph_lie/scripts/field_diag.py Experiments/graph_lie/[TLA]_*/latest.pt
python models/graph_lie/scripts/aggregate.py && python models/graph_lie/scripts/make_figures.py
```
