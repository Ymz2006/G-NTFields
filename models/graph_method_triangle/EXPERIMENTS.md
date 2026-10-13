# Experiment log: triangle-inequality heads (`models/graph_method_triangle`)

This log is chronological (2026-10-11). Setup:

* Hardware and image: 4x RTX 3090, `pytorchserver:latest` (torch 2.2.2).
* Data, recipes and evaluator are the ones of `models/graph_method`; see its
  EXPERIMENTS.md (E0-E15).
* Obstacle-free recipe (`configs/graph_method_triangle/graph_tri.json` =
  graph_gram.json with another arch):
  * objective L_E + L_B + 10 L_goal + L_T on 1024 declared anchors
  * batch 1024, 12k Adam updates, cosine LR, relative-angle curriculum
    60 -> 180 deg over 7k steps
  * Fourier sigma 0.03, linear input skip
* Obstacle-free metrics are held-out test-set numbers from `evaluate.py`
  (seed 2002, 10k uniform + 10k mixture pairs, 1000 planning pairs).
* Stage E recipe: `stageE_common.json` + Fourier sigma_t 0.5 / sigma_r 0.1 +
  skip input, scored on all 1000 RRT-verified cases by the unmodified
  Shape-NTFields MPPI evaluator.

Run directories: `Experiments/graph_method_triangle_free/<run>` and
`Experiments/graph_method_triangle_stageE/<run>`. Results are in
`results/graph_method_triangle_{free,stageE,triangle}/`.

## T0: copy and reproduction

The package is a copy of `models/graph_method` (imports renamed). The new parts are:

* in `model_network_graph.py`: `rot6d_to_matrix`, `group_lp_norm`,
  `GraphTriNN` / `GraphTriMetricNN` / `GraphTriMixNN`, `gram_to_matrix`,
  `GraphGramTriNN`
* `triangle.py`, `scripts/triangle_check.py`, `scripts/eval_shape.sh`,
  `scripts/eval_shape_daemon.sh`
* config keys `group`, `lp_p`, `rot_gain`, `rot_bias`, `rot_mode`, `tri_switch`,
  and `lse_beta` (graph_metric's logsumexp temperature, default 10 =
  unchanged; T8)
* the `set_phase` hook in the Functions
* best-checkpoint gating for `graph_gram_tri`

Reproduction check: `gram_repro_s0` (graph_gram trained through this package)
reproduces `graph_method_final/graph_gram_s0` exactly. Every validation number
matches and the best score is the same float, 0.00828537752742355. So
comparisons against the previous agent's runs are like-for-like.

## T1: unit tests

`python -m unittest tests.test_graph_method_triangle` runs 26 tests: the 19
copied spec tests (all pass unchanged) plus 7 in `TestTriangleHeads`:

* `rot6d_to_matrix` returns rotations and inverts "first two columns".
* `b = |V_x - V_y|^2 / 8` equals `(3 - tr)/4` and `sin^2(theta/2)` from the
  independent atan2 oracle.
* `group_lp_norm` matches `torch.linalg.vector_norm`. It is homogeneous and
  subadditive, does not underflow at 1e-20, and has a finite (zero) gradient at
  exactly 0.
* For every head (`graph_tri`, `graph_tri_metric`, `graph_tri_mix`,
  `graph_gram_tri`) × both rotation modes × two inits, all with perturbed
  random parameters:
  * T(x, x) = 0, exact symmetry, b in [0, 1);
  * no triangle violation above 1e-9 on 512 random triples, nor after a
    40-step adversarial ascent.
* Hand-set weights reproduce the oracle (relative error < 1e-4 away from the
  B_EPS margin at pi) in both rotation modes.
* `gram_to_matrix(qq^T) = R(q)`, and on rank-1 Gram matrices the exact b equals
  graph_gram's relaxed b.

Bug avoided: computing b as (3 - tr)/4 cancels catastrophically in float32 at
small angles. At 2e-3 rad, b ~ 1e-6 while the float32 rounding of tr is about
3e-7. All heads use |V_x - V_y|^2 / 8.

## T2: the proposal with a freely learned V(x) collapses

`graph_tri` (a = |A Δe|^2, V = rot6d(W e(x))) with the graph_gram recipe.

**Initial blow-up.** At random init the 6-d output has |a1| median 0.49 and
min 0.07. Gram-Schmidt divides by |a1|, so V(x) rotates fast wherever |a1| is
small. On 4096 pairs the initial |c_x|^2 has p99 72 and max 1907 (graph_gram:
max 1.3). The initial gradient norm is 1e7, and it varies strongly with the
seed (seed 1: max 10). Remedy tried: rot_bias = 1, rot_gain = 1 (Q ≈ I at
start). That gives max |c|^2 0.7-1.4.

**Every variant ends in the same state.** Seeds 0 and 1, random init, Q ≈ I
init with gain 1 or 0.3:

| run | test uni rel p50 | p95 | exact-pi abs | mean b (b* = 0.75) | V spread |
|---|---|---|---|---|---|
| tri_default_s0/s1 | 13.3 ± 0.03 % | 29.4 % | 0.957 | 0.002 | 0.05 |
| tri_b1g1_s0/s1 | 13.3 ± 0.08 % | 29.4 % | 0.957 | 0.002 | 0.08 |
| tri_b1g03_s0 | 13.4 % | 29.4 % | 0.959 | | |

V(x) becomes constant and b goes to 0. The Euclidean a-latent absorbs the
rotation (mean a 5.3 vs a* 2.0), i.e. a chordal approximation of the rotation
distance. graph_tri_metric and graph_tri_mix with free V behave the same:
9.3 % median, at the metric-head level. Training curves are flat at 11-12 %
from 6k steps on, while graph_gram from the same seed is at 1.2 % at 8k and
0.11 % at 12k.

**Explanation (topology).**

* The free head can represent the solution exactly (T1 hand-set test).
* Maps SE(3) -> SO(3) fall into homotopy classes. A V(x) in a different class
  from x -> R_x cannot be deformed continuously into it without passing through
  poses where Gram-Schmidt is singular (|a1| = 0 or a1 ∥ a2). A constant map is
  in a different class, and the random init evidently is too. At those poses
  the field's Lipschitz constant, and hence the Eikonal loss, blows up, and
  training collapses V instead.
* A quaternion head is not an alternative: no continuous lift SO(3) -> S^3 of
  the identity exists.
* graph_gram avoids all of this. Its unit-norm PSD Gram matrices form a
  contractible set, so any map into it can be deformed into any other, and b*
  is linear in vec R.

## T3: adversarial triangle check

Random triples are uninformative: the slack T(x, y) + T(y, z) - T(x, z) is
O(T), so the old `triangle_viol_frac` reads 0 % for every model, including ones
that violate the inequality badly. `triangle.attack` instead does Adam ascent
of v = T(x, z) - T(x, y) - T(y, z):

* over body twists of all three poses ('all') or of y only ('y');
* starting from y at the free-space geodesic midpoint;
* in float64, keeping the best exact value.

**Random init** (2048 triples, 300 steps, before and after N(0, 0.5) parameter
noise):

| head | max v / T(x, z) | triples violated |
|---|---|---|
| graph_tri, graph_tri_metric, graph_tri_mix | < 0 (no violation) | 0 % |
| graph_gram | < 0 | 0 % |
| graph_metric | 0.16-0.19 | 23-31 % |
| metric (G-Shape logsumexp head) | 0.16-0.20 | 24-30 % |

**Trained checkpoints** (1024 triples, 'all' attack;
`results/graph_method_triangle_triangle/prev_and_first.json`):

| checkpoint | max v / T(x, z) | p99 | triples violated |
|---|---|---|---|
| graph_method_final/graph_gram_s0 (obstacle-free) | < 0 | | 0 % |
| graph_method_final/metric_best_recipe_s0 | 0.178 | 0.167 | 100 % |
| graph_method_final/graph_metric_s0 | 0.204 | 0.195 | 100 % |
| graph_method_final/scalar_s0 | 0.849 | 0.813 | 100 % |
| graph_method_stageE/stageE_graph_gram_s0 (T-shape) | < 0 | | 0 % |
| graph_method_stageE/stageE_graph_metric_s0 | 0.167 | 0.150 | 98.5 % |
| graph_method_stageE/stageE_metric_s0 | 0.185 | 0.149 | 95.7 % |
| triangle_stageE/T_tri_b1g1_s0, T_trimetric_s0 | < 0 | | 0 % |
| triangle_free/tri_default_s0 | < 0 | | 0 % |

For the obstacle-free checkpoints most of these violating triples leave the
training box. The final check below gives in-domain numbers, which are still
large.

Why the metric heads fail: per group, the head computes
LSE_10(sqrt(d^2 + 1e-6)) - log(16)/10.

* This is convex, with value ~0 at d = 0, so along rays it is superadditive:
  LSE(2u) >= 2 LSE(u).
* It behaves like a mean of |d_i| at small scales and like a max at large
  scales.
* At the midpoint of a short segment in embedding space, T(x, y) + T(y, z) is
  therefore smaller than T(x, z).

graph_gram has no violations because its trained P is rank 1 (T5), which makes
it the exact head in practice.

**Final check, both attack modes**
(`results/graph_method_triangle_triangle/final.json`, 1024 triples, 300
steps). "in-dom" is the share of violating triples whose three translations
stay inside the training domain's bounding box. The all-pose attack on
obstacle-free models mostly escapes the [-1, 1]^3 box. The y-only attack keeps
the sampled x, z, and its violations are all in-domain.

| checkpoint | 'all': violated / max rel / in-dom | 'y only': violated / max rel / in-dom |
|---|---|---|
| graph_method_final/graph_gram_s0 | 0 % / < 0 / - | 0 % / < 0 / - |
| gram_proj_s0 (same weights, exact head) | 0 % / < 0 / - | 0 % / < 0 / - |
| graph_method_final/metric_best_recipe_s0 | 100 % / 0.178 / 2 % | 70.5 % / 0.110 / 100 % |
| graph_method_final/graph_metric_s0 | 100 % / 0.204 / 4 % | 88.0 % / 0.097 / 100 % |
| graph_method_final/scalar_s0 | 100 % / 0.849 / 1 % | 89.6 % / 0.590 / 98 % |
| tri_eq_s0, trimetric_eq_s0 | 0 % / < 0 / - | 0 % / < 0 / - |
| graph_method_stageE/stageE_metric_s0 | 95.7 % / 0.185 / 95 % | 55.5 % / 0.160 / 100 % |
| graph_method_stageE/stageE_graph_metric_s0 | 98.5 % / 0.167 / 95 % | 61.0 % / 0.143 / 100 % |
| graph_method_stageE/stageE_graph_gram_s0 | 0 % / < 0 / - | 0 % / < 0 / - |
| T_trimetric_b1g1_s0, T_trimetric_eq_s0 | 0 % / < 0 / - | 0 % / < 0 / - |

## T4: equivariant rotation latent V = R_x Q(x)

With `rot_mode='equivariant'`, V is still an exact rotation for every
parameter value (so the T1 guarantee holds), and its homotopy class is fixed by
construction. Q = const gives b = b* exactly. In the obstacle-free benchmark
this is therefore a **structural prior**, not a b learned from scratch.

| run | n | uni rel p50 | uni rel p95 | mix rel p95 | 179-180 rel p50 | exact-pi abs | eik res p50 | grad plan succ / cost | samp succ / cost |
|---|---|---|---|---|---|---|---|---|---|
| graph_gram (graph_method_final) | 3 | 0.105 ± 0.003 % | 0.313 ± 0.023 % | 0.341 % | 0.050 % | 0.0021 | 0.0041 | 100 % / 1.000 | 100 % / 1.01 |
| **graph_tri, equivariant** | 3 | **0.018 ± 0.006 %** | **0.064 ± 0.022 %** | 0.066 % | 0.028 % | 0.0019 | 0.00064 | 100 % / 1.000 | 100 % / 1.01 |
| graph_tri, equivariant, Q ≈ I init | 1 | 0.015 % | 0.051 % | 0.054 % | 0.021 % | 0.0018 | 0.00058 | 100 % / 1.000 | 100 % / 1.01 |
| graph_tri_metric, equivariant | 1 | 0.032 % | 0.121 % | 0.128 % | 0.022 % | 0.0014 | 0.00090 | 100 % / 1.000 | 100 % / 1.01 |
| graph_tri, equivariant, no curriculum | 1 | 0.112 % | 0.206 % | 0.209 % | 0.061 % | 0.0013 | 0.0013 | 100 % / 1.000 | 100 % / 1.01 |
| graph_tri, equivariant, **no anchors** | 1 | 2.32 % | 19.4 % | 18.8 % | 1.16 % | 0.096 | 0.0022 | 84.6 % / 2.24 | 88.9 % / 1.65 |

* All equivariant runs are below 0.2 % median by 1k updates (graph_gram reaches
  that at ~10k). The random-Q init runs show early loss spikes (75 at 1k) and
  still converge.
* With rotation handled by V, the exact-seminorm a-latent (graph_tri_metric) is
  as accurate as the Euclidean one. The metric heads' 9.5 % was caused by
  having to encode rotation inside an L-infinity-type metric.
* No anchors (physics only): 2.3 % / 19 %, the same as graph_gram without
  anchors (2.4 % / 19 %). The projection-metric degeneracy of the endpoint
  Eikonal (graph_method E10) lives in the a-latent. Fields T = |P (x - y)| are
  valid pseudometrics, so the triangle guarantee cannot exclude them.

## T5: learning in the relaxed Gram space, reading out exactly (graph_gram_tri)

For a unit quaternion q, R(q) is linear in qq^T (`gram_to_matrix`). So
V = rot6d(R(P)) turns graph_gram's Gram matrix P into an exact rotation, and on
rank-1 P the exact b equals graph_gram's b.

Trained graph_gram P is rank 1. Over 20k uniform poses, the top-eigenvalue
share of the trace is >= 0.99995 and the gap is 1.0000, for seeds 0 and 1.

| run | n | uni rel p50 | uni rel p95 | exact-pi abs | grad plan |
|---|---|---|---|---|---|
| graph_gram (graph_method_final) | 3 | 0.105 ± 0.003 % | 0.313 ± 0.023 % | 0.0021 | 100 % |
| **graph_gram checkpoints read through the exact head** (`gram_proj_s*`, no retraining) | 3 | **0.105 ± 0.003 %** | **0.313 ± 0.023 %** | **0.0015** | 100 % |
| **graph_gram_tri, switch at 11k** (1k exact updates) | 1 | **0.111 %** | **0.288 %** | 0.0013 | 100 % |
| graph_gram_tri, switch at 9k (3k exact updates) | 1 | 0.75 % | 2.21 % | 0.019 | 100 % |
| graph_gram_tri, switch at 4k (8k exact updates) | 1 | 0.196 % | 0.584 % | 0.0077 | 100 % |

* The same parameter vectors evaluated by the guaranteed architecture lose
  nothing, and b was learned from scratch.
* The late-switch run matches graph_gram seed 0 (0.104 % / 0.29 %); its best
  validation score is 0.00833 vs 0.00829.
* *Training* with the exact head through the near-half-turn phase of the
  curriculum is worse, even though the switch itself is seamless (identical
  validation at the switch step):
  * sw4k: 0.20 % at 6k, then 5.6 % when near-pi pairs enter at 7k, recovering
    to 1.7 % by 10k and 0.196 % on the test set at 12k;
  * sw9k: 0.73 % at 9k, then 1.6 % at 10k and 0.98 % at 12k, while the relaxed
    head reaches 0.11 %.
* Likely reason: the exact head has a hard learned cut locus (b -> 1 - 1e-6,
  second derivatives ~ 1/sqrt(B_EPS)). Before V settles onto the isometry, that
  locus lies on input-geometry-regular pairs. The relaxed head's b stays below
  1 there.
* Recommended use is therefore: train relaxed and read out exact, or switch
  after the near-half-turn phase has converged (11k here).

## T6: stage E (obstacles)

**b is (almost) unused among obstacles.** Share of T^2 from the rotation term on
4096 dataset pairs (Tshape3d_env1):

| checkpoint | share of T^2 from b |
|---|---|
| graph_method stageE graph_metric_s0 | 1.0 % |
| graph_method stageE graph_gram_s0 | 0.6 % |
| T_tri_b1g1_s0 (free V) | 0.0 % |

So in all stage-E graph heads the field lives in the a-latent. The free-V
collapse (T2) also happens here.

With **equivariant V** the rotation latent is active among obstacles:

| checkpoint | b vs free-space b* | share of T^2 from b |
|---|---|---|
| T_trimetric_eq_s0 / s1 | mean 0.29 (b* 0.30), corr 0.997 | 4.9 % / 4.7 % |
| A_trimetric_eq_s0 | corr 0.988 | 7.0 % |

Q(x) varies a little (spread 0.11-0.15) and modulates the free-space rotation
distance.

## T7: stage E results (1000 cases per environment, unmodified MPPI evaluator)

The comparison rows are the previous agent's stage-E results from the same
pipeline and evaluator (`results/graph_method_stageE*`). All new rows use
graph_tri_metric (exact seminorm a-latent, p = 8, metric_scale 0.2) with
Q ≈ I init (rot_bias 1). The evaluator is unseeded, so a re-score of the same
checkpoint differs by about ±1 point (T8).

**Tshape3d_env1** (`results/graph_method_triangle_stageE/summary_T.md`):

| model | seeds | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |
|---|---|---|---|---|---|---|---|
| metric head (graph_method) | 3 | 80.0 ± 0.6 | 80.1 ± 0.4 | 87.2 ± 0.2 | 82.1 ± 0.9 | 18.7 ± 6.0 | 188.3 ± 1.2 |
| graph_metric (graph_method) | 3 | 79.2 ± 0.2 | 80.8 ± 0.1 | 87.3 ± 0.2 | 82.5 ± 0.7 | 20.0 ± 6.0 | 194.0 ± 3.6 |
| graph_gram (graph_method) | 3 | 75.1 ± 0.4 | 76.2 ± 0.5 | 85.8 ± 0.3 | 80.0 ± 0.8 | 74.7 ± 10.0 | 200.3 ± 3.5 |
| graph_tri (Euclidean a, free V) | 1 | 76.1 | 75.5 | 85.4 | 80.1 | 56 | 201 |
| graph_tri_metric, free V | 3 | 75.6 ± 1.8 | 77.4 ± 1.0 | 86.9 ± 1.3 | 80.9 ± 2.3 | 60.0 ± 18.7 | 202.0 ± 5.6 |
| graph_tri_metric, free V, random 6-d init | 1 | 72.9 | 74.8 | 86.0 | 76.4 | 89 | 197 |
| **graph_tri_metric, equivariant V** | 3 | 77.2 ± 2.2 | 77.7 ± 2.4 | 85.3 ± 1.5 | 80.4 ± 2.4 | 73.7 ± 23.5 | 171.3 ± 12.6 |

**Lshape3d_env3** (`summary_L.md`):

| model | seeds | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |
|---|---|---|---|---|---|---|---|
| metric head (graph_method) | 2 | 81.4 ± 0.4 | 81.2 ± 1.1 | 86.8 ± 0.2 | 83.8 ± 0.8 | 18.0 ± 4.2 | 174.0 ± 2.8 |
| graph_metric (graph_method) | 2 | 82.9 ± 0.7 | 83.2 ± 0.1 | 88.8 ± 0.1 | 84.5 ± 0.6 | 17.0 ± 2.8 | 159.5 ± 9.2 |
| graph_gram (graph_method) | 1 | 70.3 | 72.2 | 82.2 | 80.6 | 39 | 277 |
| graph_tri_metric, free V | 2 | 79.4 ± 0.3 | 80.0 ± 0.2 | 87.5 ± 0.8 | 83.5 ± 1.4 | 60.5 ± 3.5 | 163.0 ± 2.8 |
| **graph_tri_metric, equivariant V** | 2 | 81.3 ± 1.1 | 81.6 ± 1.3 | 85.9 ± 1.2 | 84.2 ± 0.8 | 72.0 ± 8.5 | 136.5 ± 4.9 |

**Ashape3d_env4** (`summary_A.md`):

| model | seeds | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |
|---|---|---|---|---|---|---|---|
| metric head (graph_method) | 2 | 81.5 ± 1.1 | 81.2 ± 0.4 | 87.3 ± 0.1 | 83.5 ± 0.1 | 18.0 ± 0.0 | 173.0 ± 8.5 |
| graph_metric (graph_method) | 2 | 81.5 ± 0.8 | 80.6 ± 1.4 | 87.5 ± 0.1 | 83.4 ± 0.6 | 22.5 ± 4.9 | 170.5 ± 4.9 |
| graph_gram (graph_method) | 1 | 75.6 | 75.3 | 84.9 | 82.3 | 50 | 211 |
| graph_tri_metric, free V | 2 | 79.7 ± 1.3 | 79.1 ± 1.6 | 86.8 ± 0.4 | 83.6 ± 0.7 | 43.0 ± 19.8 | 175.5 ± 2.1 |
| **graph_tri_metric, equivariant V** | 2 | 81.1 ± 1.0 | 80.9 ± 0.4 | 84.9 ± 0.6 | 82.9 ± 0.4 | 46.5 ± 7.8 | 158.5 ± 4.9 |

Mean forward success over the three environments (mean of seed means):

| head | T / L / A mean |
|---|---|
| graph_metric | 81.2 % |
| metric head | 81.0 % |
| **graph_tri_metric, equivariant V** | **79.9 %** |
| graph_tri_metric, free V | 78.2 % |
| graph_gram | 73.7 % |

Observations:

* With equivariant V, the guaranteed head is on par with the logsumexp heads on
  L (81.3 vs 81.4 / 82.9) and A (81.1 vs 81.5 / 81.5). It is 2-3 points behind
  on T, where its seed spread is large (74.8-79.1).
* The failure modes differ. The triangle heads have 2-4x more collisions
  (29-94 per 1000 vs 13-26) but, with equivariant V, the fewest
  non-convergences (133-183 vs 153-198).
* Equivariant V is better than free V on every environment (+1.4 to +1.9
  points). It is the only variant whose rotation latent carries weight (T6).
* Near-obstacle gradients (`scripts/diag_shape_eik.py`, Tshape3d_env1 seed 0):
  in the slowest clearance bins the triangle heads' |c| is about 15-25 % below
  the logsumexp heads'. Median |c| is 0.10-0.12 vs 0.14-0.15 for S < 0.01, and
  0.51-0.54 vs 0.62-0.68 for S in [0.01, 0.1). That means a weaker barrier,
  consistent with more collisions.
* l_p exponent sweep (T-shape, free V, seed 0): p = 1 / 2 / 4 / 8 / 16 / 32 gives
  74.0 / 72.2 / 72.2 / 74.3 / 73.0 / 73.3 %. All are within evaluator noise, so
  the exponent does not matter.

## T8: is it the violations that help the logsumexp heads plan?

The logsumexp group max can violate the triangle inequality by up to
2 log(16)/beta per group. Raising the temperature beta moves graph_metric
toward an exact metric (a sum of group L-infinity norms) without changing its
form. New knob `lse_beta`; the default 10 is G-Shape's and numerically
unchanged.

Reproduction first. `T_gmetric_beta10_s0` retrains graph_metric through this
package. It matches the previous agent's stage-E run exactly (identical
held-out residuals and losses at 2k and 4k steps), and its re-score is 79.0 %
vs 79.1 % before. So the evaluator's re-score noise is small (it is unseeded);
seed-to-seed spread (±1-2 points) dominates.

Tshape3d_env1, seed 0, 1000 cases. Violations from `triangle_check.py`,
'y only' attack, all in-domain (`results/graph_method_triangle_triangle/beta.json`).

| head | in-domain violation: triples / max / p99 | forward | collision | no-converge |
|---|---|---|---|---|
| graph_metric, beta 10 | 61 % / 14.3 % / 12.0 % | 79.0 % | 22 | 193 |
| graph_metric, beta 30 | 58 % / 12.1 % / 9.9 % | 79.8 % | 22 | 188 |
| graph_metric, beta 100 | 47 % / 8.8 % / 6.3 % | 76.6 % | 32 | 211 |
| graph_tri_metric (exact), free V, 3 seeds | 0 | 75.6 ± 1.8 % | 60 ± 19 | 202 ± 6 |
| graph_tri_metric (exact), free V, metric_scale 0.14 | 0 | 75.5 % | 38 | 219 |
| graph_tri_metric (exact), equivariant V, 3 seeds | 0 | 77.2 ± 2.2 % | 74 ± 24 | 171 ± 13 |

What this suggests:

* Moving the logsumexp head toward an exact metric (beta 100) costs about 2.5
  points and adds collisions. That brings it to the level of the exact heads.
* So part of the logsumexp heads' planning advantage on this benchmark appears
  to come from their non-metric small-scale behaviour. Mechanism: they are
  mean-like for near pairs and max-like for far pairs, which decouples the
  near-diagonal local metric from the gradients toward far goals. A metric
  couples them through |T(x, g) - T(x', g)| <= T(x, x').
* MPPI only queries T(., goal), and the stage-E Eikonal loss mostly sees far
  pairs.
* This is one seed per beta, and beta 30 is level with beta 10. Treat it as
  suggestive, not established.

## Summary

* The architectural guarantee is exact for every head here and every parameter
  value (T1). The heads with no guarantee (logsumexp metric heads, scalar)
  violate the triangle inequality by 10-20 % in their training domains once a
  violation is searched for (T3).
* Building the proposal literally, with a freely learned V(x) in SO(3), fails
  for a topological reason (T2). Two fixes keep the guarantee:
  * V = R_x Q(x): obstacle-free 0.018 % median error (6x better than
    graph_gram, as a structural prior);
  * read graph_gram's rank-1 Gram matrix out exactly: identical to graph_gram,
    0.105 %, with b learned from scratch (T4, T5).
* Merged with the previous agent's stage-E work (metric-type a-latent, made an
  exact seminorm, plus equivariant V): 79.9 % mean MPPI success over T / L / A,
  vs 81.0-81.2 % for the non-metric logsumexp heads and 73.7 % for graph_gram.
  It is on par on L and A and 2-3 points behind on T (T7, T8).
* The guarantee does not cure the physics-only degeneracy of the endpoint
  Eikonal (T4).

## T9: five seeds of the new model, and the results dashboard

The merged obstacle head (graph_tri_metric, equivariant V,
`configs/graph_method_triangle/stageE_graph_tri_metric_eq.json`) and the two
obstacle-free equivariant heads now have 5 seeds each:

* T-shape seeds 3-4, L and A seeds 2-4, obstacle-free graph_tri seeds 3-4 and
  graph_tri_metric seeds 1-4 were trained on all four GPUs and scored with the
  unchanged evaluators.
* The existing seeds were reused. Training is deterministic: a fresh re-run of
  T-shape seed 0 (`Experiments/graph_method_triangle_rerun/`) reproduced every
  logged validation value, and after 20k steps all 25 weight tensors are
  bitwise identical to the original's.

**Obstacles, forward MPPI success (all test cases, mean ± SD over seeds):**

| scene | graph_tri_metric, equivariant V (5 seeds) | metric head (graph_method) | graph_metric (graph_method) |
|---|---|---|---|
| Tshape3d_env1 | 77.9 ± 1.8 % (77.8, 74.8, 79.1, 79.0, 78.9) | 80.0 ± 0.6 % (3) | 79.2 ± 0.2 % (3) |
| Lshape3d_env3 | 80.9 ± 1.0 % (82.1, 80.5, 80.6, 81.7, 79.6) | 81.4 ± 0.4 % (2) | 82.9 ± 0.7 % (2) |
| Ashape3d_env4 | 81.1 ± 0.7 % (80.4, 81.8, 81.4, 81.4, 80.3) | 81.5 ± 1.1 % (2) | 81.5 ± 0.8 % (2) |
| mean of scene means | **80.0 %** | 81.0 % | 81.2 % |

The baseline rows keep their original 2-3 seeds. Failure counts per 1000
cases:

| head | collisions | not converged |
|---|---|---|
| graph_tri_metric, equivariant V (15 runs) | 34-94 | 133-184 |
| metric head | 13-25 | 167-189 |
| graph_metric | 14-26 | 153-198 |

**Obstacle-free, 5 seeds:**

| head | median rel error | p95 | mix p95 | 179-180 deg rel p50 | exact-pi abs | planning |
|---|---|---|---|---|---|---|
| graph_tri, equivariant V | 0.016 ± 0.005 % | 0.057 ± 0.019 % | 0.060 % | 0.025 % | 0.0019 | 100 % / cost 1.000 |
| graph_tri_metric, equivariant V | 0.037 ± 0.007 % | 0.135 ± 0.022 % | 0.144 % | 0.025 % | 0.0014 | 100 % / cost 1.000 |

**Diagnostics over all seeds**
(`scripts/export_dashboard_data.py --diagnostics`, 8,192 held-out pairs per
checkpoint).

Rotation-latent share of T^2:

| head | T | L | A | correlation with free-space b* |
|---|---|---|---|---|
| graph_tri_metric, equivariant V | 4.7 % | 10.9 % | 7.0 % | 0.99 |
| graph_metric | 1.1 % | 0.6 % | 1.5 % | 0.11-0.17 |
| graph_gram | 0.3 % | 2.6 % | 2.5 % | |
| graph_tri_metric, free V | 0.0 % | 0.0 % | 0.0 % | |

Median endpoint speed × |grad T| on the T-shape, by clearance-speed bin:

| head | S < 0.01 | 0.01-0.1 | 0.1-0.5 | 0.5-1 |
|---|---|---|---|---|
| graph_tri_metric, equivariant V (5 seeds) | 0.117 | 0.600 | 1.039 | 1.339 |
| graph_tri_metric, free V (3) | 0.111 | 0.578 | 1.023 | 1.320 |
| metric head (3) | 0.136 | 0.642 | 1.019 | 1.271 |
| graph_metric (3) | 0.136 | 0.650 | 1.032 | 1.286 |
| graph_gram (3) | 0.119 | 0.556 | 0.965 | 1.362 |

The triangle heads are 14-18 % and 7-11 % weaker in the two slowest bins. That
corrects the single-seed "15-25 %" in T7.

**Dashboard.** https://claude.ai/artifact/LHyUtVzc8h9u4KgKKvz4ts (Dashboard
type, private until shared). Every figure on it is computed from eight CSVs in
`results/graph_method_triangle_dashboard/`, written by
`scripts/export_dashboard_data.py`; each row names its result file.

## C1-C6: convergence and path length near rotational cusps vs. the original method (2026-10-12)

Full write-up: `results/graph_method_cusp/REPORT.md` (tables generated by
`scripts/aggregate_cusp.py` / `scripts/aggregate_free.py`, figures in
`results/graph_method_cusp/figures/`). Tools added: `scripts/plan_eval_graph.py` (per-case port of the
official MPPI evaluator: chart AND SE(3) convergence, seam-crossing flag, path lengths, `--wrap`),
`scripts/plan_suite_graph.sh`, `scripts/plan_by_angle_free.py`, `scripts/field_slice.py`,
`scripts/make_cusp_figures.py`, and `models/graph_native/` (graph heads inside the original trainer,
`train/train_graph_native.py`).

* C1 (field slices): the original chart-input field jumps by 0.69 (median over 12 anchors) at the
  rotation-vector seam; every G(x)-based head is continuous there (0.001-0.003).
* C2 (standard sets, 1000 cases, forward MPPI): under the official chart test our heads trail the
  original by 0-8 points; under SE(3) convergence they lead: T 93.6-93.8 vs 81.9, L 95.4-96.4 vs 89.9,
  A 95.3-95.5 vs 87.1 (metric-type a-latents), trieq 90.2 / 92.4 / 94.6. Cause: ~145 of ~190 chart
  "non-converged" cases per seed are at the goal pose with |rotvec| > pi. The original's SE(3) and chart
  scores are identical (re-scored). Wrapping the rotation vector in the planner (--wrap) lifts the
  chart score of our metric head from 80.1 to 95.8.
* C3 (near-cusp suites): bigangle 35.0 (original) vs 74-80 (ours, SE(3)); rotheavy 78.9 vs 82-85;
  transonly 69.4 vs 85-87; tight 67.5 vs 83-87. Above 150 deg: T 84 vs 71, L 92 vs 81, A 90 vs 77.
* C4 (original trainer, `models/graph_native`): graph_metric trained by the original code (3 seeds):
  chart 80.3 +- 0.9 (original 82.1 +- 0.3), SE(3) 94.7 +- 0.2, collisions 1.4 vs 11.1; L 97.2 / A 96.4
  (2 seeds; original 89.9 / 87.1); trieq T 87.9 +- 3.2, L 93.5, A 94.0; bigangle 80.3 vs 34.2. The
  earlier "native-pipeline gap" was the chart artifact.
* C5 (free space, by angle): graph heads 100 % / cost ratio 1.0000 in every bin; metric head (best
  recipe) 93-95 % and cost ratio 1.27-1.28 at 170-180 deg; with the NTFields loss 78-79 % / 1.26-1.29.
* C6 (triangle head): lowest non-convergence of all fields (0.6-3.8 %), full seam advantage, 2-3x more
  collisions than the non-metric heads; still +2.5 to +8 points over the original.
