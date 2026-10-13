# Smooth-graph SE(3) fields vs. the original metric NTFields: convergence and path length near rotational cusps

Date: 2026-10-12. Code: `models/graph_method_triangle/scripts/{plan_eval_graph.py, plan_suite_graph.sh,
aggregate_cusp.py, plan_by_angle_free.py, aggregate_free.py, field_slice.py, make_cusp_figures.py}`,
`models/graph_native/`, `train/train_graph_native.py`. Raw per-case records:
`results/graph_method_cusp/plan/<tag>/<set>.json`; generated tables: `summary/plan_summary.md`,
`summary/free_summary.md`; figures: `figures/`. Everything in this report comes from those files.

## 1. Summary

The original method (G-Shape / metric NTFields: a metric head on random-Fourier features of the
rotation-vector chart `(t, rotvec / 2 pi)`) is not a function on SE(3): its travel time jumps at the
chart seam `|rotvec| = pi` (median jump 0.69 in tau along a 0.5-degree step, vs. 0.001-0.003 for the
smooth-graph fields, Section 4.1). The smooth-graph formulation of `se3_smooth_graph_report.pdf` is
built on a fixed embedding `G(x) = (t, vec R)` and an explicit decoder, so it is continuous across the
seam and reproduces the cut-locus corner. Measured with the unmodified Shape-NTFields MPPI planner on
the RRT-verified test sets, this changes convergence and path quality as follows.

* **Obstacle-free SE(3), 500 pairs per relative-rotation bin, gradient planner.** The graph heads
  (`graph_gram`, `graph_tri` with equivariant V, `graph_tri_metric`) succeed in 100 % of cases in
  every bin at path cost ratio 1.0000. The metric head (best recipe) degrades toward the cusp: at
  170-179 degrees 95.0 % success, median cost ratio 1.28, 95th percentile 1.53, 238 steps instead of
  167; with the original NTFields loss, 78.9 % success and cost ratio 1.29 (Section 4.6).
* **Among obstacles, under the physically meaningful convergence test (same pose on SE(3)):** the
  smooth-graph fields beat the original on all three environments, 1000 RRT-verified cases each:
  T-shape 93.6-93.8 % vs 81.9 %, L-shape 95.4-96.4 % vs 89.9 %, A-shape 95.3-95.5 % vs 87.1 %
  (metric-type a-latent heads, 2-3 seeds), with 2-6x fewer collisions (T 1.9 % vs 11.3 %) and 3-5x
  fewer non-convergences. On pairs with relative rotation above 150 degrees the gap is 11-14 points
  (T 84 vs 71 %, L 92 vs 81 %, A 90 vs 77 %). On the Lie repo's "bigangle" set, whose start and goal
  rotations both lie within 45 degrees of the half turn, the original succeeds on 35 % of cases
  (46 % never converge); the smooth-graph fields succeed on 71-80 % with 5-11 % non-convergence
  (Sections 4.2-4.4).
* **Why the original evaluator did not show this.** The official success test is `|x - goal| < 0.01`
  in the chart. A seam-free field reaches the goal pose through the other rotation vector in ~15 % of
  the test cases (exactly the seam-crossing pairs); the robot is at the goal on SE(3) (distance 0.006)
  but 1.0 away in the chart, and the case is scored "not converged". 144 of the 145 such cases per
  seed end with `|rotvec| > pi`. Canonicalizing the rotation vector after each executed MPPI step
  (one line, no change to the field) removes the artifact: the metric head on `G(x)` goes from 80.1 %
  to 95.8 % under the official chart test. The original method is unaffected by either change (its
  planner never crosses the seam; re-scored SE(3) and chart numbers are identical, Section 4.2).
* **Head-to-head in the original trainer.** With the graph heads dropped into the original training
  code (same data, loss, optimizer, accept/reject loop, 4000 epochs), `graph_metric` is within two
  points of the original under its own chart test (T 80.3 +- 0.9 % vs 82.1 +- 0.3 %, 3 seeds) and
  reaches 94.7 +- 0.2 % on SE(3) with 1.4 % collisions (L 97.2 %, A 96.4 %, 2 seeds each); the
  triangle-inequality head reaches 87.9 +- 3.2 % (T), 93.5 % (L), 94.0 % (A). So the gain is a
  property of the representation, not of our stage-E training recipe (Section 4.5).
* **Triangle inequality.** `graph_tri_metric` with equivariant V (a pseudometric for every parameter
  value) keeps the full seam/cusp advantage (T 90.2 %, L 92.4 %, A 94.6 % SE(3); bigangle 71 %;
  the lowest non-convergence of all heads, 0.6-3.8 %) and still beats the original by 3-8 points, but
  it collides 2-3x more often than the non-metric heads, which costs it 1-5 points (Section 4.7).

## 2. Background: the two singular sets and what the original method does with them

`se3_smooth_graph_report.pdf` (Section 8) identifies two singular sets of the travel time on SE(3): the
diagonal (a cone) and the rotational cut locus (relative rotation pi, a concave corner where two
shortest rotations tie). The smooth-graph construction keeps the fixed embedding `G(x) = (t, vec R)`
and learns a smooth latent `(a, b)`; the decoder `T = sqrt(a / s_t^2 + phi(b) / s_r^2)` re-creates the
corner from a smooth `b` (Theorem 6.5, Proposition 8.1).

The original method parametrizes rotations by the rotation vector divided by 2 pi and feeds it to
random Fourier features. The chart is discontinuous on the sphere `|rotvec| = pi`: `+pi n` and `-pi n`
are the same pose but lie 1 turn apart in the input space, and nothing in the loss (it constrains
only `|grad tau|`) glues the two sides. The Lie-Shape-NTFields analyses (`analysis/rotation_wrap_probe.py`,
`analysis/antipode_probe.py`, results of 2026-10-10) measured the consequence: `tau(+pi n, -pi n)` is
about 3 (larger than 99 % of all test start-goal travel times) and pairs whose shortest rotation
crosses the seam succeed 65 % of the time vs 85 % otherwise. They also found that smooth SO(3)
encodings (6-d, matrix, Euler) in the same Lie pipeline plan *worse* at large angles: a smooth scalar
field on compact SO(3) must flatten at the goal's cut locus, so the MPPI weights become uniform
there and the planner stalls. The smooth-graph decoder is the construction that resolves both: a
function on SO(3) (no seam) whose non-smoothness is explicit (the corner is in `D`, not rounded off).

## 3. Methodology

**Original method.** `metric_baseline` of Lie-Shape-NTFields (identical to this repository's
`models/metric` with the TD and normal losses off, which the Lie repo found to be *better* than with
them on), checkpoints trained by its own trainer: T-shape seeds 1-5, L-shape and A-shape seeds 1-3,
all at epoch 4000. Its per-case planning results (`results/compare/plan/`) are reused where they exist
and were re-scored with our evaluator (bitwise-identical success labels, Section 4.2).

**Our fields.** Stage-E checkpoints of `models/graph_method` and `models/graph_method_triangle`
(README/EXPERIMENTS of those packages): the metric head on `G(x)` ("ours: metric head"), `graph_metric`
(PSD-Gram `b`, logsumexp metric `a`, analytic decoder), `graph_gram` (Euclidean `a`), `graph_tri_metric`
with equivariant V = R_x Q(x) ("trieq", a pseudometric by construction) and with free V ("trifree").
All take `G(x)`; none sees the chart.

**Native-pipeline heads (new, `models/graph_native/`).** The same graph heads inside a copy of the
original trainer and loss: 800k pairs, 5 batches of 2000 per epoch, AdamW (wd 0.5, lr 5e-4), the
accept/reject rollback, the isotropic residual `1e-2 (sqrt(S |grad tau|) - 1)^2` weighted by
`exp(-tau / 2)`, 4000 epochs. The network maps the chart coordinates to `G(x)` differentiably, so the
loss can use either the original chart gradient (`--grad chart`) or the intrinsic body-frame gradient
of the report (`--grad lie`). Fourier scales 0.5 (translation) / 0.1 (rotation columns), the stage-E
values.

**Per-case evaluator (`plan_eval_graph.py`).** A port of the Lie repo's `analysis/plan_eval.py`: the
forward MPPI of `evaluate_training_3d_batched.py` verbatim (50 samples, horizon 5, step 0.015,
temperature 50, 200 iterations, chart convergence ball 0.01, the same `igl` point-in-solid collision
test at every waypoint, start/goal screening). It reproduces the stored results of the Lie repo
(first 200 T cases of `metric_s1`: 84.0 % success, 9.5 % collisions, median 61 iterations, identical).
In addition it records, per case: whether the shortest start-goal rotation crosses the seam; the first
iteration at which the SE(3) distance `sqrt(|dp|^2 + (theta / 2 pi)^2)` falls below 0.01 and the
success/collision/path length under that criterion (the executed trajectory is the official one in
both cases, only the bookkeeping differs); path translation length and rotation length (chart and
geodesic) relative to the straight start-goal values; and `--wrap`, which canonicalizes the rotation
vector after each executed step.

**Test sets.** The standard 1000-case RRT-verified sets (`Tshape3d_env1`, `Lshape3d_env3`,
`Ashape3d_env4`; 25-28 % of pairs above 150 degrees, 15-16 % seam-crossing) and the Lie repo's
T-shape suites: `bigangle` (200 cases, both endpoints with rotation angle in [135, 180) degrees,
44 % seam-crossing), `rotheavy` (|dp| <= 0.05, relative rotation >= 90 degrees), `transonly`
(identical rotations, |dp| >= 0.3, a control) and `tight_3d` (offset 0.005).

**Free space.** `plan_by_angle_free.py`: 500 start/goal pairs per relative-rotation bin (0-90, 90-150,
150-170, 170-179, 179-180 degrees, plus pure rotations at 170-180), uniform translations in the box,
Haar start rotations, random axes; the gradient planner (descent on the decoded time with
backtracking and sampling fallback) and the 128-candidate sampling planner of
`planning/rollout.py`, h = 0.02, 600 steps. Cost ratio = path time / exact optimal time.

## 4. Results

### 4.1 The field itself: seam vs. corner

`field_slice.py` rotates the start about a random body axis through a full turn from the goal pose
(position of a test-case goal) and evaluates tau(start, goal); 12 anchors, `figures/slices.png`.

| field | median max |delta tau| per 0.5-degree step | tau jump at the chart seam (median) |
|---|---|---|---|
| original metric NTFields | 0.690 | 0.690 |
| Lie repo's best chart variant (`lie_metric`, sqrt residual) | 0.347 | 0.347 |
| ours: metric head on G(x) | 0.139 | 0.002 |
| ours: graph_metric | 0.180 | 0.003 |
| ours: graph_tri_metric, equivariant V | 0.114 | 0.002 |
| ours: graph_gram | 0.088 | 0.001 |

The remaining per-step variation of our fields is the obstacle structure (clearance speeds down to
0.01 make tau rise by ~0.14 per half degree near walls); the original's largest step is always at the
seam.

### 4.2 Standard test sets: convergence criterion and mechanism

Forward MPPI success, 1000 cases per environment, mean +- sd over seeds. "chart" is the official
criterion, "SE(3)" the same trajectories scored by the pose distance. The original's two numbers
coincide by construction (its planner never crosses the seam); `LIE_*` rows are the Lie repo's stored
results and `*_native` the same checkpoints re-scored here.

| environment | model | seeds | success chart | success SE(3) | non-conv chart / SE(3) | collision | median iters | trans. ratio | rot. ratio |
|---|---|---|---|---|---|---|---|---|---|
| T | original (chart input) | 5 | 81.9 +- 0.5 | 81.9 | 13.1 | 11.3 +- 0.5 | 58 | 1.42 | 1.61 |
| T | ours: metric head on G(x) | 3 | 79.8 +- 0.4 | **93.8 +- 0.5** | 18.9 / 4.4 | 2.1 +- 0.6 | 59 | 1.37 | 1.64 |
| T | ours: graph_metric | 3 | 79.5 +- 0.5 | **93.6 +- 0.2** | 19.3 / 4.6 | 1.9 +- 0.3 | 59 | 1.37 | 1.64 |
| T | ours: graph_tri_metric, eq. V (triangle) | 5 | 77.9 +- 2.8 | 90.2 +- 2.9 | 17.4 / 3.8 | 6.3 +- 2.6 | 60 | 1.38 | 1.61 |
| T | ours: graph_gram | 3 | 75.1 +- 0.2 | 87.2 +- 0.3 | 20.1 / 6.1 | 7.5 +- 0.9 | 58 | 1.36 | 1.62 |
| T | graph_metric in the original trainer | 3 | 80.3 +- 0.9 | **94.7 +- 0.2** | 18.7 / 4.1 | 1.4 +- 0.5 | 59 | 1.37 | 1.64 |
| T | graph_tri_metric eq. V in the original trainer | 3 | 76.1 +- 3.0 | 87.9 +- 3.2 | 16.4 / 2.9 | 9.5 +- 3.8 | 58 | 1.37 | 1.58 |
| L | original | 3 | 89.9 +- 1.6 | 89.9 | 9.0 | 4.3 +- 0.4 | 56 | 1.30 | 1.35 |
| L | ours: metric head on G(x) | 2 | 81.6 +- 0.0 | **95.4 +- 0.1** | 17.0 / 2.7 | 1.9 +- 0.2 | 56 | 1.27 | 1.37 |
| L | ours: graph_metric | 2 | 82.8 +- 0.6 | **96.4 +- 0.5** | 16.0 / 2.2 | 1.6 +- 0.0 | 56 | 1.26 | 1.38 |
| L | ours: graph_tri_metric, eq. V | 5 | 81.3 +- 0.7 | 92.4 +- 0.8 | 13.8 / 1.2 | 6.8 +- 0.9 | 54 | 1.25 | 1.33 |
| A | original | 3 | 87.1 +- 1.3 | 87.1 | 11.1 | 5.8 +- 0.6 | 52 | 1.37 | 1.39 |
| A | ours: metric head on G(x) | 2 | 81.3 +- 0.5 | **95.5 +- 0.3** | 17.5 / 2.8 | 1.7 +- 0.2 | 52 | 1.33 | 1.41 |
| A | ours: graph_metric | 2 | 80.6 +- 0.2 | **95.3 +- 0.3** | 17.6 / 2.1 | 2.7 +- 0.3 | 52 | 1.33 | 1.40 |
| A | ours: graph_tri_metric, eq. V | 5 | 80.8 +- 0.8 | 94.6 +- 1.0 | 15.6 / 0.6 | 5.0 +- 1.0 | 51 | 1.31 | 1.36 |

Path lengths of successful cases (translation length and geodesic rotation length over the straight
start-goal values) are the same or slightly shorter for our fields (T: 1.37 vs 1.42 translation).

**Mechanism.** For the metric head on G(x), T-shape, per seed: 188-190 cases fail the chart test;
143-148 of them reached the goal on SE(3) (median final SE(3) distance 0.006, chart distance 1.0), and
in 144/145, 141/143, 146/148 of those the final rotation vector has `|rotvec| > pi`, i.e. the planner
arrived through the other representation of the goal orientation. 79-81 % of them are seam-crossing
pairs. The 42-47 remaining cases per seed are genuine non-convergences (median final angle to the goal
79-129 degrees).

**Wrap control.** Same checkpoint (metric head, seed 0), same planner with the rotation vector
canonicalized after each executed step (`--wrap`):

| criterion | unmodified planner | planner with wrap |
|---|---|---|
| chart success (official) | 80.1 % | **95.8 %** |
| SE(3) success | 94.5 % | 95.8 % |
| non-convergence (chart) | 18.8 % | 3.0 % |

**By seam crossing (SE(3) success, standard sets).** Original T (seed 1): crossing 66.7 % vs
non-crossing 85.2 %; L: 74.7 +- 5.1 vs 92.5 +- 1.0; A: 75.8 +- 5.1 vs 89.2 +- 0.6. Ours (graph_metric):
T 87.6 +- 3.0 vs 94.6 +- 0.6; L 92.8 +- 1.0 vs 97.0 +- 0.4; A 92.9 +- 2.2 vs 95.8 +- 0.1. The original
also collides 2.4-3.3x more often on crossing pairs (T 20.9 vs 8.7 %, L 10.5 vs 3.2 %, A 11.4 vs 4.7 %):
the chart path around the seam is longer and passes closer to obstacles (its rotation path is 1.73x
the geodesic on crossing pairs vs 1.60x otherwise, per the Lie repo's probe).

### 4.3 Success by relative rotation (`figures/success_by_angle.png`)

SE(3) success (chart in parentheses) by start-goal geodesic angle; the original's single number is
both.

| environment | model | 0-90 deg | 90-150 deg | 150-180 deg |
|---|---|---|---|---|
| T (n = 319 / 432 / 249) | original | 92.8 +- 1.4 | 80.2 +- 1.2 | 70.8 +- 1.0 |
| T | ours: metric head | 98.1 (96.3) | 96.1 (82.3) | **84.1** (54.2) |
| T | ours: graph_metric | 97.6 (95.9) | 96.0 (82.0) | **84.2** (53.9) |
| T | ours: graph_tri_metric eq. V | 91.0 (89.3) | 91.9 (79.7) | **86.1** (59.9) |
| T | graph_metric, original trainer (3) | 97.8 (96.1) | 96.5 (82.4) | **87.7** (56.5) |
| T | graph_tri_metric eq. V, original trainer (3) | 87.5 (85.9) | 89.7 (78.2) | **85.5** (60.0) |
| L (n = 321 / 415 / 264) | original | 96.8 +- 1.0 | 90.0 +- 1.7 | 81.4 +- 2.7 |
| L | ours: graph_metric | 98.3 (95.8) | 97.5 (85.9) | **92.4** (62.3) |
| L | ours: graph_tri_metric eq. V | 95.5 (93.1) | 92.9 (82.7) | **87.7** (64.6) |
| A (n = 299 / 423 / 278) | original | 94.8 +- 0.7 | 88.5 +- 1.6 | 76.6 +- 1.6 |
| A | ours: graph_metric | 97.8 (95.0) | 96.9 (83.8) | **90.1** (60.1) |
| A | ours: graph_tri_metric eq. V | 96.0 (93.2) | 95.1 (82.7) | **92.2** (64.4) |

The original loses 15-22 points from the first to the last bin; graph_metric and the metric head lose
6-14, the triangle head 4-8 and graph_gram 0-18 (the remaining loss is
collision-dominated and shared by all fields).

### 4.4 The near-cusp suites (T-shape; `figures/special_sets.png`)

Success (SE(3) for ours, chart = SE(3) for the original), non-convergence under the same criterion,
and collisions; mean +- sd over seeds.

| set (cases) | model | success | non-convergence | collision |
|---|---|---|---|---|
| bigangle (191; both endpoints 135-180 deg from identity, 44 % crossing) | original | 35.0 +- 3.3 | 46.1 +- 2.2 | 51.0 +- 2.4 |
| | ours: metric head | 74.2 +- 1.9 | 9.8 +- 1.4 | 17.1 +- 1.8 |
| | ours: graph_metric | 75.2 +- 1.7 | 10.8 +- 0.7 | 15.5 +- 1.7 |
| | ours: graph_tri_metric eq. V (5) | 71.0 +- 5.9 | **5.4 +- 1.5** | 26.0 +- 5.7 |
| | graph_metric, original trainer (3) | **80.3 +- 1.1** | 8.2 +- 2.0 | 13.6 +- 1.5 |
| | graph_tri_metric eq. V, original trainer (3) | 67.2 +- 7.0 | **5.2 +- 1.1** | 30.2 +- 7.5 |
| rotheavy (196; |dp| <= 0.05, rotation >= 90 deg) | original | 78.9 +- 2.7 | 4.2 +- 1.1 | 20.1 +- 2.4 |
| | ours: metric head | 83.7 +- 0.7 | 5.6 +- 1.5 | 12.2 +- 1.5 |
| | ours: graph_metric | 82.0 +- 1.5 | 7.0 +- 0.5 | 12.9 +- 1.0 |
| | ours: graph_tri_metric eq. V | 78.5 +- 1.9 | 3.5 +- 1.1 | 19.9 +- 2.6 |
| | graph_metric, original trainer (3) | **85.5 +- 2.1** | 4.9 +- 0.2 | 11.1 +- 2.1 |
| transonly (193; identical rotations, control) | original | 69.4 +- 2.9 | 16.7 +- 1.5 | 22.9 +- 2.4 |
| | ours: metric head | 85.1 +- 0.6 | 2.2 +- 1.9 | 13.8 +- 0.2 |
| | ours: graph_metric | 86.5 +- 0.7 | 1.9 +- 0.5 | 12.3 +- 1.4 |
| | ours: graph_tri_metric eq. V | 74.9 +- 4.0 | 7.2 +- 1.6 | 20.9 +- 3.2 |
| | graph_metric, original trainer (3) | **85.7 +- 1.3** | 2.8 +- 0.6 | 12.3 +- 0.5 |
| tight_3d (962; offset 0.005) | original | 67.5 +- 0.5 | 18.3 +- 1.1 | 24.7 +- 1.2 |
| | ours: metric head | 83.3 +- 0.4 | 5.3 +- 0.2 | 13.2 +- 0.5 |
| | ours: graph_metric | 84.1 +- 0.3 | 5.9 +- 0.4 | 11.9 +- 0.4 |
| | ours: graph_tri_metric eq. V | 77.1 +- 2.9 | 4.5 +- 0.9 | 20.6 +- 3.1 |
| | graph_metric, original trainer (3) | **85.6 +- 0.8** | 5.2 +- 0.7 | 11.1 +- 0.7 |

On the bigangle set under the official chart test our fields score 41-46 % (original 35 %): the seam
cases are 44 % of this set, so the chart artifact is largest here. On `rotheavy` the original's chart
score is actually the best of the chart scores (78.9 vs 62-66 %): with the start 0.05 from the goal,
the original's field around the seam acts as a wall that keeps its planner on the canonical side,
while our planners drift through the seam and then fail the chart test; on SE(3) ours lead by 3-6
points. The translation-only control shows that the gain is not only rotational: the original fails
31 % of pure translations (17 % non-convergence, 23 % collisions) against 13-15 % for the graph heads.

### 4.5 Head-to-head inside the original trainer (`models/graph_native`)

T-shape (seeds as noted; `summary/plan_summary.md` carries every seed):

| head in the original pipeline | gradient in the loss | chart success | SE(3) success | collision | bigangle SE(3) |
|---|---|---|---|---|---|
| original metric head (chart input), 5 seeds (3 re-scored here: 82.1 +- 0.3) | chart | 81.9 +- 0.5 | 81.9 | 11.3 | 35.0 |
| graph_metric (3 seeds) | chart | 80.3 +- 0.9 | **94.7 +- 0.2** | 1.4 +- 0.5 | **80.3 +- 1.1** |
| graph_metric (1 seed) | intrinsic (lie) | 80.8 | 94.4 | 1.3 | 75.9 |
| graph_tri_metric, equivariant V (3 seeds: 91.3 / 84.3 / 88.1 SE(3)) | chart | 76.1 +- 3.0 | 87.9 +- 3.2 | 9.5 +- 3.8 | 67.2 +- 7.0 |
| graph_tri_metric, equivariant V (1 seed) | intrinsic (lie) | 77.6 | 90.0 | 5.9 | 75.9 |
| graph_gram (1 seed) | chart | 75.5 | 90.1 | 6.6 | 68.6 |

L-shape and A-shape, two seeds per head (original: 3 seeds):

| environment | head in the original pipeline | chart success | SE(3) success | non-conv SE(3) | collision |
|---|---|---|---|---|---|
| L | original metric head | 89.9 +- 1.6 | 89.9 | 9.0 | 4.3 |
| L | graph_metric | 83.9 +- 0.6 | **97.2 +- 0.2** | 1.8 | 1.0 |
| L | graph_tri_metric, equivariant V | 81.9 +- 0.8 | 93.5 +- 0.3 | 1.3 | 5.5 |
| A | original metric head | 87.1 +- 1.3 | 87.1 | 11.1 | 5.8 |
| A | graph_metric | 81.7 +- 0.6 | **96.4 +- 0.3** | 1.8 | 1.8 |
| A | graph_tri_metric, equivariant V | 80.8 +- 0.1 | 94.0 +- 0.2 | 0.6 | 5.5 |

Training the graph heads with the original code reproduces our stage-E numbers within a point or two
(T graph_metric 94.7 vs 93.6, L 97.2 vs 96.4, A 96.4 vs 95.3; trieq 87.9 +- 3.2 vs 90.2 +- 2.9, with
the same seed spread), so the previous agents' unexplained "native-pipeline gap" (82 vs 80 % chart on
T, 90 vs 82 % on L) was the chart artifact, not a training deficit. Under the chart test the graph
heads in the original trainer sit 2 points below the original on T and 5-6 points below it on L and
A; under SE(3) they are 13, 7 and 9 points above it. The intrinsic (body-frame) gradient in the loss
gives the same numbers as the chart gradient for graph_metric (94.4 vs 94.7) and 2 points more for the
triangle head (90.0 vs 87.9 +- 3.2, one seed). 

### 4.6 Obstacle-free SE(3): convergence and path length by angle (`figures/free_by_angle.png`)

Gradient planner on the decoded time, 500 pairs per bin, mean +- sd over seeds (full tables incl. the
sampling planner in `summary/free_summary.md`).

| model | metric | 0-90 | 90-150 | 150-170 | 170-179 | 179-180 | pure rot. 170-180 |
|---|---|---|---|---|---|---|---|
| graph_gram (3) | success % | 100 | 100 | 100 | 100 | 100 | 100 |
| | cost ratio median / p95 | 1.0000 / 1.000 | 1.0000 / 1.000 | 1.0000 / 1.000 | 1.0000 / 1.000 | 1.0000 / 1.002 | 1.0000 / 1.000 |
| | steps | 80 | 124 | 156 | 167 | 171 | 152 |
| graph_tri, eq. V (5) and graph_tri_metric, eq. V (5) | success / cost / steps | identical to graph_gram | | | | | |
| graph_gram read by the exact triangle head (2) | success / cost / steps | identical to graph_gram | | | | | |
| metric head, best recipe (3) | success % | 100 | 99.9 | 98.0 +- 0.5 | 95.0 +- 1.4 | 93.2 +- 0.7 | 94.2 +- 1.3 |
| | cost ratio median / p95 | 1.034 / 1.08 | 1.096 / 1.23 | 1.210 / 1.50 | 1.283 / 1.53 | 1.273 / 1.51 | 1.141 / 1.40 |
| | steps | 83 | 139 | 202 | 238 | 249 | 204 |
| metric head, original NTFields loss (3) | success % | 100 | 97.3 | 82.5 +- 2.1 | 78.9 +- 1.2 | 78.3 +- 0.9 | 75.9 +- 1.6 |
| | cost ratio median / p95 | 1.056 / 1.12 | 1.116 / 1.39 | 1.247 / 1.76 | 1.291 / 1.64 | 1.262 / 1.62 | 1.294 / 1.71 |
| | steps | 85 | 157 | 273 | 301 | 305 | 299 |
| direct scalar network (3) | success % | 2.2 | 1.1 | 0.9 | 1.5 | 1.2 | 1.4 |

Near the cusp the metric heads need 40-80 % more steps, take 27-29 % longer paths (median) and fail
5-22 % of the time; the graph heads are exact. Among obstacles the same decoder is present, but the
rotation term is bounded (`phi(b) / s_r^2 <= 1/4` with rotations in turns) while obstacle-inflated
travel times are 2-5, so the corner it adds is small and the large-angle gains of Sections 4.2-4.4 come
from the seam-free embedding rather than from a sharper ridge; we measured no systematic difference
at 150-180 degrees between heads with an active rotation latent (trieq: 86.1 %) and without
(graph_metric: 84.2 %, metric head: 84.1 %).

### 4.7 Triangle inequality

`graph_tri_metric` with equivariant V is a pseudometric for every parameter value (README of
`models/graph_method_triangle`). Measured here it has the lowest non-convergence rates of all fields
(SE(3): T 3.8 %, L 1.2 %, A 0.6 %, bigangle 5.4 %; original 13.1 / 9.0 / 11.1 / 46.1 %) and the full
seam advantage (bigangle 71-76 %), i.e. the guarantee costs nothing in convergence; its deficit against
the non-metric heads is collisions (T 6.3 vs 2.1 %, L 6.8 vs 1.9 %), consistent with its weaker
near-obstacle gradient documented in EXPERIMENTS.md T9. Against the original it is still better on
every environment (T +8, L +2.5, A +7.5 points SE(3)), with 2-4x fewer collisions on T and A.

## 5. What is established and what is not

Established (per-case records, same planner, same collision test, 1000-case sets, 2-5 seeds):
the original field is discontinuous at the chart seam and our fields are not; under the SE(3)
convergence test (or the wrapped planner) the metric-type heads on `G(x)` (metric head, graph_metric)
beat the original by 5.5-12 points on every environment and by 9-14 points above 150 degrees, with
2-6x fewer collisions and equal or shorter paths; the triangle head beats it by 2.5-8 points (6-16
above 150 degrees); graph_gram does not (L-shape 81.2 vs 89.9 %);
the gain survives training inside the original pipeline; in free space the decoder gives exact paths
through the cusp where the metric heads degrade.

Not established: more than 2-3 seeds for the native-pipeline heads; an advantage of the
decoder's ridge *among obstacles* beyond the seam-free embedding (bounded rotation term, Section 4.6);
any claim under the unmodified evaluator's chart test, where our fields trail the original by 0-8
points because of the artifact described in Section 4.2 (the one-line wrap removes it, but that is a
planner change).

## 6. Reproduce

Inside `pytorchserver:latest` with this repo at `/workspace` and Lie-Shape-NTFields read-only at
`/lieshape` (it holds the test sets, meshes and the original checkpoints):

```bash
# per-case planning of one checkpoint on the T suites (host side; writes results/graph_method_cusp/plan/<tag>/)
bash models/graph_method_triangle/scripts/plan_suite_graph.sh 2 graph Experiments/graph_method_stageE/stageE_graph_metric_s0/latest.pt T_gmetric_s0
bash models/graph_method_triangle/scripts/plan_suite_graph.sh 2 metric /media/jeffrey/Lie-Shape-NTFields/Experiments/3dshape_metric_baseline/Tshape3d_env1_metric_baseline_s1/Model_Epoch_04000_ValLoss_1.748832e-03.pt T_native_s1
PLAN_EXTRA=--wrap bash models/graph_method_triangle/scripts/plan_suite_graph.sh 2 graph <ckpt> <tag>_wrap Tshape3d_env1
python3 models/graph_method_triangle/scripts/aggregate_cusp.py          # -> summary/plan_summary.{md,json}
# graph head inside the original trainer, then its evaluation
python train/train_graph_native.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_gmetric_chart_s0 --seed 0 --arch graph_metric --grad chart
bash models/graph_method_triangle/scripts/plan_suite_graph.sh 2 graph_native Experiments/graph_native/T_gmetric_chart_s0/latest.pt GN_T_gmetric_chart_s0
# obstacle-free planning by angle
python models/graph_method_triangle/scripts/plan_by_angle_free.py --out results/graph_method_cusp/free --ckpt graph_gram_s0=Experiments/graph_method_final/graph_gram_s0/best.pt ...
python3 models/graph_method_triangle/scripts/aggregate_free.py
# field slices and figures
python models/graph_method_triangle/scripts/field_slice.py --out results/graph_method_cusp/slices --model 'original_metric=metric:<ckpt>' --model 'ours=graph:<ckpt>'
python models/graph_method_triangle/scripts/make_cusp_figures.py
```
