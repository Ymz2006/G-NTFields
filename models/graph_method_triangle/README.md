# `models/graph_method_triangle`: triangle inequality by construction

This package is a copy of `models/graph_method`, the earlier implementation of
`SE3_Smooth_Graph_Implementation.pdf`. It adds pair heads whose decoded travel
time is a pseudometric for every parameter value. Everything else is unchanged:
geometry, analytic decoder, losses, data, planners, evaluator, the stage-E
obstacle pipeline and all training recipes. As a check, the copied package
reproduces `Experiments/graph_method_final/graph_gram_s0` exactly (same
validation curve, same best score `0.00828537752742355`; run `gram_repro_s0`).

See `EXPERIMENTS.md` for the chronological log (T0-T9). Results dashboard (five seeds of
the new model): https://claude.ai/artifact/LHyUtVzc8h9u4KgKKvz4ts (private until shared).

## The proposal

A shared per-pose encoder produces `f(x) = (u(x), V(x))` with `u(x) in R^m` and
`V(x) in SO(3)`. The pair latents are built only from `f(x)` and `f(y)`:

    a(x, y) = |u(x) - u(y)|^2
    b(x, y) = (3 - tr(V(x)^T V(y))) / 4
    T(x, y) = sqrt(a / s_t^2 + 4 asin(sqrt b)^2 / s_r^2)      (the unchanged decoder)

The decoder's two components are pseudometrics: `d_t = |u_x - u_y| / s_t`, and
`d_r = angle(V_x^T V_y) / s_r`, the bi-invariant geodesic distance on SO(3).
So `T = |(d_t, d_r)|_2` satisfies `T(x, z) <= |(d_t(x,y) + d_t(y,z), d_r(x,y) + d_r(y,z))|_2 <= T(x, y) + T(y, z)`.
`T(x, y) = T(y, x)` and `T(x, x) = 0` also hold exactly.

Implementation details that keep the guarantee exact:

* **b is computed as `|V_x - V_y|_F^2 / 8`.** For rotations this equals
  `(3 - tr)/4`. The trace form cancels catastrophically in float32 at small
  angles; the difference form does not.
* **The range adapter `b -> (1 - 1e-6) b`** (kept from graph_method so that
  `phi'(b)` stays finite) maps the angle to
  `f(theta) = 2 asin(sqrt(1 - 1e-6) sin(theta / 2))`. This f is increasing and
  concave with `f(0) = 0`, hence subadditive, so the result is still a
  pseudometric.
* **Rotation output.** V comes from the 6-d Gram-Schmidt map of Zhou et al.
  (`rot6d_to_matrix`). It is exactly orthonormal and has a continuous right
  inverse. A normalised 4-vector (quaternion) head cannot work: a continuous
  lift SO(3) -> S^3 of the identity does not exist.
* **Metric-type a-latent (for obstacles).** graph_metric's logsumexp distance is
  not subadditive. `graph_tri_metric` replaces it with an exact seminorm,
  `m = 0.2 sum_g |n(e_x)_g - n(e_y)_g|_p` over groups of 16 standardised
  embedding coordinates with p = 8 (`group_lp_norm`; max-scaled so that
  `|d|^p` cannot underflow, with exact derivatives).

## Heads (`model_network_graph.py`)

| arch | a | V | guarantee |
|---|---|---|---|
| `graph_tri` | `|A (e_x - e_y)|^2` (the proposal) | `rot_mode='free'`: `V = rot6d(W e(x))`; `'equivariant'`: `V = R_x rot6d(W e(x))` | every parameter value |
| `graph_tri_metric` | `m(x, y)^2`, exact group-l_p seminorm | same | every parameter value |
| `graph_tri_mix` | both, summed | same | every parameter value |
| `graph_gram_tri` | graph_gram's | relaxed Gram b until `tri_switch`, then `V = rot6d(R(P))` from the same Gram matrix P | after the switch / for loaded checkpoints |
| `graph_gram`, `graph_metric`, `metric`, ... | unchanged copies (graph_metric gains an optional `lse_beta`, default 10 = unchanged) | | none |

The `rot_mode` option and the `graph_gram_tri` head exist because of the main
finding below.

## Findings

**1. With a freely learned V(x), the rotation latent collapses.**
Every `graph_tri` run drives V(x) to a near-constant rotation, so b goes to 0
(mean b 0.001 vs b* 0.75). The Euclidean a-latent then absorbs the rotation as
a chordal distance. Median error is 13.3% vs graph_gram's 0.105%. This happened
in all 5 runs (2 seeds; random 6-d output, and near-identity starts with gain 1
or 0.3). My explanation is topological:

* Maps into SO(3) fall into homotopy classes. To move a learned V(x) into the
  class of `x -> R_x`, gradient descent would have to pass through poses where
  Gram-Schmidt is singular. There the field's Lipschitz constant, and hence
  the Eikonal loss, blows up, so training collapses V instead.
* graph_gram never meets this obstacle. Its unit-norm PSD Gram matrices form a
  contractible set, and b* is even linear in vec R.
* The hand-set solution is exactly representable by the free head
  (`test_analytic_solution_is_representable`); the problem is reaching it.

**2. Two ways around it, both keeping the guarantee:**

* **Equivariant V = R_x Q(x)** (`rot_mode='equivariant'`). The homotopy class is
  fixed by construction, and V is still an exact rotation for every parameter
  value. Obstacle-free results over 5 seeds: 0.016 ± 0.005% median and
  0.057 ± 0.019% p95 relative error, with 100% planning at cost ratio 1.000.
  That is about 6× better than graph_gram. Caveat: with Q = const, b = b* exactly, so in the
  obstacle-free benchmark this is a structural prior, not a learned b.
* **Learn in the relaxed space, read out exactly** (`graph_gram_tri`). For a
  unit quaternion q, R(q) is linear in qq^T. graph_gram's trained Gram matrices
  are rank 1 to four decimals (top eigenvalue share of the trace ≥ 0.9999 on
  20k poses). So the exact head V = rot6d(R(P)) evaluated on graph_gram's own
  parameters gives a guaranteed model with b learned from scratch, with
  identical accuracy. Switching to the exact head at step 11k, after the
  near-half-turn phase, also matches graph_gram (0.111%). Switching at 4k or
  9k ends worse (0.20% / 0.75%): the exact head has a hard learned cut locus,
  which disturbs training while near-half-turn pairs are being learned (T5).

**3. The baselines' metric heads violate the triangle inequality.** I searched
for violations with gradient ascent on all three poses, starting from the
geodesic midpoint (`triangle.py`, `scripts/triangle_check.py`, float64). Random
triples never come close to equality, so "0 violations on random triples" (the
old evaluator metric) is uninformative. Under the attack:

* Trained G-Shape metric heads and graph_metric violate it, obstacle-free and
  among obstacles. Keeping the sampled x, z fixed inside the training domain
  and moving only y, 55-88% of triples are violated, by up to 10-16% of
  T(x, z). Moving all three poses, 96-100% are violated, up to 17-20%.
* The direct scalar network violates it by up to 59% in-domain.
* Cause: the logsumexp group max with its log(16)/10 offset is convex with
  value 0 at zero difference, so it is locally superadditive, like a squared
  distance.
* Trained graph_gram shows no violations; its P is rank 1 in practice.
* The triangle heads cannot violate it (see the tests and table T3).

**4. Among obstacles (stage E), the merged head nearly matches the metric heads.**
In the previous agent's graph_metric and graph_gram, b supplies under 1% of T²
among obstacles; the field lives in the a-latent.

* Equivariant V makes the rotation latent active: 5-7% of T², with b tracking
  the free-space rotation distance at correlation 0.99.
* Merged design: graph_metric's metric-type a-latent, made an exact seminorm,
  plus equivariant V. With 5 seeds per scene it reaches 80.0% mean forward
  MPPI success over the T / L / A environments, vs 81.0-81.2% for the
  logsumexp heads and 73.7% for graph_gram.
* It is 0.4-2.0 points behind on L and A and 1.3-2.1 points behind on T; see
  the tables below and the logsumexp-temperature test in EXPERIMENTS.md T8.

## Results

The evaluator, planners and test sets are unchanged from graph_method. Full
tables are in `EXPERIMENTS.md` and `results/graph_method_triangle_*/summary*.md`.

### Obstacle-free SE(3), held-out test set (mean ± std over seeds)

| model | n | uni rel p50 | uni rel p95 | 179-180 deg rel p50 | exact-pi abs | grad plan succ / cost |
|---|---|---|---|---|---|---|
| graph_gram (graph_method) | 3 | 0.105 ± 0.003 % | 0.313 ± 0.023 % | 0.050 % | 0.0021 | 100 % / 1.000 |
| metric head, best recipe (graph_method) | 3 | 9.53 ± 0.15 % | 23.1 ± 0.1 % | 12.9 % | 0.455 | 98.6 % / 1.12 |
| graph_tri, free V (both inits) | 5 | 13.3 ± 0.1 % | 29.4 % | 28.6 % | 0.958 | 100 % / 1.04 |
| graph_tri_metric, free V | 2 | 9.32 ± 0.18 % | 31.7 % | 24.6 % | 0.839 | 98.0 % / 1.15 |
| **graph_tri, equivariant V** | 5 | **0.016 ± 0.005 %** | **0.057 ± 0.019 %** | 0.025 % | 0.0019 | 100 % / 1.000 |
| graph_tri_metric, equivariant V | 5 | 0.037 ± 0.007 % | 0.135 ± 0.022 % | 0.025 % | 0.0014 | 100 % / 1.000 |
| **graph_gram weights read by the exact head** | 3 | **0.105 ± 0.003 %** | **0.313 ± 0.023 %** | 0.048 % | 0.0015 | 100 % / 1.000 |
| graph_gram_tri, exact from step 11k | 1 | 0.111 % | 0.288 % | 0.051 % | 0.0013 | 100 % / 1.000 |
| graph_tri, equivariant V, no anchors | 1 | 2.32 % | 19.4 % | 1.16 % | 0.096 | 84.6 % / 2.24 |

### Obstacles (stage E), forward MPPI success over 1000 cases

| model | Tshape3d_env1 | Lshape3d_env3 | Ashape3d_env4 | mean |
|---|---|---|---|---|
| metric head (graph_method) | 80.0 ± 0.6 (3) | 81.4 ± 0.4 (2) | 81.5 ± 1.1 (2) | 81.0 |
| graph_metric (graph_method) | 79.2 ± 0.2 (3) | 82.9 ± 0.7 (2) | 81.5 ± 0.8 (2) | 81.2 |
| graph_gram (graph_method) | 75.1 ± 0.4 (3) | 70.3 (1) | 75.6 (1) | 73.7 |
| graph_tri_metric, free V | 75.6 ± 1.8 (3) | 79.4 ± 0.3 (2) | 79.7 ± 1.3 (2) | 78.2 |
| **graph_tri_metric, equivariant V** | 77.9 ± 1.8 (5) | 80.9 ± 1.0 (5) | 81.1 ± 0.7 (5) | **80.0** |

Among obstacles the guaranteed head with equivariant V is 0.4-2.0 points
behind the logsumexp heads on L and A and 1.3-2.1 points behind on T. The
triangle heads fail more by collision (34-94 per 1000 cases with equivariant V,
vs 13-26) and less by non-convergence (133-184 vs 153-198). Averaged over
seeds, their gradient toward obstacles is 7-18 % weaker in the two slowest
clearance bins. The l_p exponent does not matter (p = 1 ... 32 all give 72-74 %
on the T-shape, seed 0). Seeds 0-2 / 0-1 were reused from the first round: a
fresh re-run of T-shape seed 0 reproduced the final weights bit for bit.

## What is proven vs. empirical

**Proven, and tested for random parameters plus adversarial triples**
(`tests/test_graph_method_triangle.py`, 26 tests): every `graph_tri*` head, and
`graph_gram_tri` in exact mode, gives a symmetric T with T(x, x) = 0 and
T(x, z) <= T(x, y) + T(y, z), up to float rounding, for every parameter value.
The analytic solution is representable by both rotation modes.

**Empirical (this benchmark):** everything in Findings and Results. In
particular:

* The free-V collapse is an observation with a topological explanation, not a
  theorem about every possible training recipe.
* The guarantee does not remove the physics-only (no-anchor) degeneracy of the
  endpoint Eikonal. Fields like T = |P(x - y)| with an orthogonal projection P
  are valid pseudometrics (EXPERIMENTS.md E10 of graph_method; T4 here).
* Among obstacles, part of the logsumexp heads' planning edge seems to come
  from their non-metric behaviour. Raising graph_metric's logsumexp
  temperature to 100 shrinks its violations (in-domain max 14% -> 9%) and its
  T-shape success drops from 79.0% to 76.6%, the exact heads' level. This is
  one seed per setting and only suggestive (T8).

## Reproduce

Run inside `pytorchserver:latest` from the repo root (see graph_method's README
for the docker line):

```bash
python -m unittest tests.test_graph_method_triangle
# obstacle-free, equivariant V (graph_tri_eq.json = graph_tri.json + rot_mode=equivariant)
python train/train_graph_triangle.py --config configs/graph_method_triangle/graph_tri_eq.json --seed 0 \
    --name tri_eq_s0 --out_root ./Experiments/graph_method_triangle_free
python -m models.graph_method_triangle.evaluate --ckpt Experiments/graph_method_triangle_free/tri_eq_s0/best.pt \
    --out results/graph_method_triangle_free/tri_eq_s0
# graph_gram trained relaxed, exact head from step 11k (graph_gram_tri.json); or load any graph_gram
# checkpoint with arch='graph_gram_tri' (same parameters) to get the exact head without retraining
python train/train_graph_triangle.py --config configs/graph_method_triangle/graph_gram_tri.json --seed 0 \
    --name gramtri_sw11k_s0 --out_root ./Experiments/graph_method_triangle_free
# stage E, merged head (Shape-NTFields at /shape, Lie-Shape-NTFields at /lieshape, read-only);
# add --set data_path=datasets/3dshape/data/Lshape3d_env3 (or Ashape3d_env4) for the other environments
python train/train_graph_triangle.py --config configs/graph_method_triangle/stageE_graph_tri_metric_eq.json --seed 0 \
    --name T_trimetric_eq_s0 --out_root ./Experiments/graph_method_triangle_stageE
bash models/graph_method_triangle/scripts/eval_shape.sh Experiments/graph_method_triangle_stageE \
    T_trimetric_eq_s0 results/graph_method_triangle_stageE
# adversarial triangle check of any checkpoints (old graph_method ones load too)
python models/graph_method_triangle/scripts/triangle_check.py --out results/tri.json <ckpt> [<ckpt> ...]
```
