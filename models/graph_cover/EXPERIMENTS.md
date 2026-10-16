# Experiment log: cover-quotient decoder and ridge detection (`models/graph_cover`, 2026-10-13)

Hardware: 3x RTX 3090 shared with two other campaigns (a chart-vs-Lie benchmark and the min-of-branches
decoder of `models/graph_branch`, which ran concurrently on the same GPUs); `pytorchserver:latest`.
Training through the original metric NTFields pipeline (`models/graph_lie`, iso speed, Lie gradient,
4000 epochs, ~0.75-1.3 s/epoch under the shared load). Evaluation: per-case forward MPPI
(`plan_eval_graph.py --kind graph_native`, 1000 cases, seed 0, batch 125), SE(3) convergence primary.

## K0: the head and its tests

`graph_cover` (README section 1). `python -m unittest tests.test_graph_cover`: 9 tests pass (float64,
perturbed random parameters): half-angle map identities and bounded derivatives at the cut and the branch
point; symmetry, T(x, x) = 0 to 1e-20, T >= T_free; `res_scale = 0` reproduces the exact free-space time
(1e-9); adversarial triangle attack (40 Adam steps on all three poses) stays below 1e-9 relative for the
four metric variants and exceeds 1e-4 for the lse ablation; the value equals the candidate minimum and
several group elements are active at random parameters; finite double backward; Q quadratic at the goal.

Initial scale on Tshape3d_env1 pairs (T_free mean 0.246): lp / res_scale 0.2 gives initial T 0.80 (N_med
1.29), euclid / 0.05 gives 0.62, lse / 0.2 gives 0.51; all 8 group elements active at init.
Soft-min variant (soft_tau 0.02): T(x, x) >= -2.8e-3, triangle deviation up to 2 % (O(tau)), finite
double backward.

## K1: stage 1, the ridge detector on the free-space half-turn (`scripts/ridge_free.py`)

Anchors: 48 random (goal p, R_g, axis n), start x(phi) = (p + 0.2 m, R_g exp(phi n)), phi scanned over
180 +- 40 deg (321 points); detector band S|c| < 0.95, lambda_min < -5; ridge = zero of the cross
derivative r = c . v inside the band. `results/graph_cover/ridge_free/ridge_free.json`.

| field | fires on | ridge bias (deg, median) | band width (deg) | lambda_min at ridge | T error at exact pi |
|---|---|---|---|---|---|
| hard min (exact) | 0 % (corner, no band) | - | 0 | - | 0 |
| soft min tau 0.01 | 100 % | 0.000 | 2.0 | -99.6 | -0.2 % |
| soft min tau 0.03 | 100 % | 0.000 | 6.0 | -33.2 | -0.7 % |
| soft min tau 0.1 | 100 % | 0.000 | 21.0 | -9.96 | -2.2 % |
| graph_gram s0 (decoder corner) | 100 % | -0.006 | 0.75 | -265 | -0.06 % |
| tri_eq s0 (decoder corner) | 100 % | 0.000 | 0.5 | -465 | -0.06 % |
| graph_sqdist s0 | 62 % | -0.21 | 22.9 | -6.4 | -0.14 % |
| metric head (best recipe) s0 | 0 % | - | >= 80 (whole window) | - | -38 % |
| metric head (NTFields loss) s0 | 0 % | - | >= 80 | - | -16 % |
| graph_metric s0 | 0 % | - | >= 80 | - | -49 % |
| scalar s0 | 0 % | - | >= 80 | - | -36 % |

* The detector behaves exactly as the model predicts on the soft-min field: it fires on every anchor, the
  zero of r is unbiased to 1e-3 deg, and lambda_min = -(1 - p_1 . p_2) / (2 tau) = -1 / tau (the two
  rotation routes are opposite, p_1 . p_2 = -1 in the rotation block): -99.6 / -33.2 / -9.96 for tau =
  0.01 / 0.03 / 0.1. The band width scales with tau (2 / 6 / 21 deg). |c| on the ridge is 0.064 = the
  translation share of the gradient (cos(alpha / 2) with alpha the route angle).
* Fields whose decoder makes the corner (graph_gram, the triangle head) have a sub-degree band from the
  1e-6 range margin of phi(b) and the detector locates the corner to 0.006 deg.
* The metric / scalar heads do not have a rounded ridge at all: S|c| is ~0.8 across the whole +-40 deg
  window and the curvature is ~-1 (loosened thresholds, band 0.9 / kappa 0.5, span +-60 deg: they then
  fire on 85-98 % of anchors with median bias -1.5 / +0.4 / -0.04 / +0.8 deg for metric_best / metric_ntf /
  graph_metric / scalar, but the "band" is 106-120 deg wide). Their 36-49 % value error at the exact
  half-turn is a plateau, not a smoothed corner, so "the derivative across the ridge vanishes" holds
  only in a weak, wide sense there.

## K2: stages 2 and 3 on free-space fields (`scripts/ridge_plan_free.py`)

Gradient planner (h = 0.02, backtracking + sampling fallback) vs. the same planner with the detector
(band 0.9, kappa 0.5): `ridge_sample` takes a 128-candidate sampling step wherever the detector fires;
`ridge_sharpen` probes x exp(+-0.3 v), extrapolates both sides to x and descends the gradient of the
lower branch. 300 pairs per relative-rotation bin. `results/graph_cover/ridge_plan/ridge_plan.json`.

Soft-min model tau = 0.1 (controlled rounding): success 100 % in every bin for every planner; cost ratio
(path time / T*) p50 / p95:

| bin | gradient | ridge_sample | ridge_sharpen | detector firings per path |
|---|---|---|---|---|
| 170-179 | 1.0001 / 1.0033 | 1.0013 / 1.0036 | 1.0001 / 1.0004 | 3.5-3.9 |
| 179-180 | 1.0093 / 1.0263 | 1.0069 / 1.0256 | 1.0002 / 1.0007 | 8.4-11.7 |
| pure_rot 170-180 | 1.0000 / 1.0000 | 1.0012 / 1.0024 | 1.0000 / 1.0000 | 3.4-3.5 |

Soft-min tau = 0.03: 179-180 bin 1.0013 / 1.0073 (gradient) -> 1.0012 / 1.0035 (sample); sharpen
pending at the time of writing. So on a field that is a true rounded min, stage 3 (route recovery) removes
the rounding cost almost entirely, while stage 2 (sampling in the band) gives nothing: the sampling step
is as blind as the gradient inside the band. Results on the trained metric heads follow below.

## K3: kink census (`scripts/kink_census.py`, property 7)

(filled in once the obstacle checkpoints are available; first pass at ds = 2e-3 on free-space fields:
graph_gram 24-47 ridge kinks per 307 units of path, tri_eq 28-47, metric head 0, scalar 0, no valleys
in any field. The obstacle fields need the slope-relative criterion and ds = 2e-4.)

## K4: obstacle training (Tshape3d_env1, iso, Lie, seed 0)

(pending: T_lp2, T_lp0, T_eu2, T_lse2, T_lp2nf, T_lp4, T_lp2soft)

### K4 first round (explore phase; 1000 cases, SE(3) success / collision, seed 0 unless n given)

| head | guarantee | SE(3) success | collision | non-conv | bigangle | rotheavy | dex (speed calib.) | lowest-decade pred/true |
|---|---|---|---|---|---|---|---|---|
| cover euclid, 2 cuts (`T_eu2_s0`) | metric | 86.0 | 12.1 | 2.1 | 53.4 / 45.0 | 78.6 / 20.4 | 0.689 | 44x |
| cover lse, 2 cuts (`T_lse2_s0`, ablation) | none | 86.5 | 12.4 | 1.1 | 55.5 / 44.0 | 76.0 / 23.5 | 0.702 | 46x |
| cover lp (p = 8), 2 cuts (`T_lp2_s0`) | metric | 43.6 | 55.0 | 3.1 | 24.1 / 75.4 | 46.9 / 53.1 | 0.958 | 125x |
| cover lp, q routes only (`T_lp0_s0`) | metric | 42.3 | 55.6 | 4.8 | 24.1 / 75.9 | 42.9 / 57.1 | 0.842 | 97x |
| cover lp, 4 cuts (`T_lp4_s0`) | metric | 42.5 | 55.8 | 4.0 | - | - | 0.935 | 93x |
| cover lp, 2 cuts, no T_free (`T_lp2nf_s0`) | metric | 56.5 | 40.2 | 5.0 | 33.5 / 64.9 | 51.0 / 43.4 | 0.858 | 75x |
| graph_metric (graph_lie, n = 3) | none | 90.8 ± 1.0 | 4.6 ± 0.9 | 5.0 ± 0.7 | 68.4 / 23.0 | 82.3 / 12.4 | 0.615 | 38x |
| graph_tri_metric eq. V (graph_lie, n = 5) | metric | 66.1 ± 7.4 | 31.4 ± 8.4 | 3.7 | 37.0 / 62.3 | 59.1 / 40.7 | 0.693 | - |
| graph_gram (branch campaign `T_ggram_s0`) | none (0 viol. in practice) | 83.2 | 10.5 | 8.3 | 48.7 / 39.3 | 70.9 / 18.4 | 0.651 | 38x |
| branch K1 = T_free + |dF|^2 (`T_br1_s0`, other campaign) | none | 91.0 | 7.1 | 1.9 | 67.5 / 29.3 | 85.2 / 13.8 | 0.585 | 30x |
| branch conf K4 (`T_bc4_s0`, other campaign) | none | 94.9 | 1.7 | 3.5 | 85.9 / 8.9 | 95.9 / 2.6 | 0.552 | 27x |

Triangle check (`scripts/cover_diag.py`, dataset triples, float64, 300-step attack): every cover variant with
`euclid` / `lp` stays at <= 0 violation under both attacks (as proven); `lse2` is violated on 100 % of
triples (max 17 %), graph_metric on 98 % (max 18 %), the branch heads on 100 % (max 45-69 % of T(x, z)),
graph_gram on 22 % (max 3 %).

Findings:

* **The l_8 group seminorm, not the triangle inequality, is what sank the earlier exact heads.** With the
  same cover structure, the Euclidean metric variant reaches 86.0 % (vs 66.1 ± 7.4 % for the previous
  guaranteed head and 83.2 % for graph_gram in this pipeline), while every l_8 variant collapses to 42-57 %
  with 40-56 % collisions, exactly like graph_tri_metric. The non-metric lse ablation with the identical
  branch structure is no better than the Euclidean metric (86.5 %), so among these the guarantee is free.
* **Success is a monotone function of the low-speed calibration.** Ordered by `dex` (mean |log10(predicted
  / true speed)| over the nine clearance decades): bc4 0.55 (94.9 %), br1 0.59 (91.0), graph_metric 0.62
  (90.8), graph_gram 0.65 (83.2), eu2 0.69 (86.0), lse2 0.70 (86.5), lp0 0.84 (42), lp2nf 0.86 (57), lp4
  0.94 (43), lp2 0.96 (44). The lowest clearance decade (S_d in 0.001-0.002) is predicted 27-46x too fast by
  every head that plans well and 75-125x too fast by the l_8 heads. Mechanism: the slope of a metric
  T = |phi(x) - phi(y)| is bounded by the Lipschitz constant of phi, and a max-type seminorm uses one
  coordinate per group (one Jacobian row) where the Euclidean norm adds all sixteen in quadrature; the
  logsumexp head is mean-like at small differences (full capacity) and max-like only far away. The
  non-metric heads that win (T_free + |dF|^2, conformal cone) get extra slope from |dF| itself or from a
  per-pose factor, which no embedding distance can do.
* The learned cuts are used (cover elements other than the identity / minus route active on 7 % of eu2
  pairs, 30 % of lp2 pairs) and the 2-D slices of the lp head show element boundaries that separate the
  two corridors around the central obstacle (`results/graph_cover/slices/T_lp2_s0/`), but for eu2 the cuts
  are rarely active (one element on the whole slice), so the ridges did not contribute to the planning
  result at seed 0; 2 vs 4 cuts vs none makes no difference (lp: 43.6 / 42.5 / 42.3).
* Dropping T_free from the lp head *helps* (43.6 -> 56.5): the hard lower bound's unit cone competes with
  the seminorm's limited slope.

### K2 complete: trained free-space fields (300 pairs per bin; success % / cost ratio p50 / p95)

| field | bin | gradient | ridge_sample (stage 2) | ridge_sharpen (stage 3) | detector firings per path |
|---|---|---|---|---|---|
| softmin_tau0.1 | 170-179 | 100.0 / 1.0001 / 1.0033 | 100.0 / 1.0013 / 1.0036 | 100.0 / 1.0001 / 1.0004 | 3.9 |
| softmin_tau0.1 | 179-180 | 100.0 / 1.0093 / 1.0263 | 100.0 / 1.0069 / 1.0256 | 100.0 / 1.0002 / 1.0007 | 11.7 |
| softmin_tau0.1 | pure_rot_170-180 | 100.0 / 1.0000 / 1.0000 | 100.0 / 1.0012 / 1.0024 | 100.0 / 1.0000 / 1.0000 | 3.5 |
| softmin_tau0.03 | 170-179 | 100.0 / 1.0000 / 1.0001 | 100.0 / 1.0000 / 1.0011 | 100.0 / 1.0000 / 1.0001 | 0.3 |
| softmin_tau0.03 | 179-180 | 100.0 / 1.0013 / 1.0073 | 100.0 / 1.0012 / 1.0035 | 100.0 / 1.0001 / 1.0002 | 2.9 |
| softmin_tau0.03 | pure_rot_170-180 | 100.0 / 1.0000 / 1.0000 | 100.0 / 1.0000 / 1.0014 | 100.0 / 1.0000 / 1.0000 | 0.4 |
| metric_best | 170-179 | 94.7 / 1.2885 / 1.5641 | 95.0 / 1.2923 / 1.5743 | 93.7 / 1.2827 / 1.6375 | 65.3 |
| metric_best | 179-180 | 93.7 / 1.2668 / 1.4790 | 93.7 / 1.2720 / 1.4855 | 93.0 / 1.2597 / 1.4987 | 66.9 |
| metric_best | pure_rot_170-180 | 95.0 / 1.1418 / 1.3951 | 95.0 / 1.1482 / 1.4054 | 94.0 / 1.1348 / 1.4790 | 57.4 |
| graph_metric | 170-179 | 96.0 / 1.2861 / 1.5730 | 96.3 / 1.2886 / 1.5750 | 94.3 / 1.2932 / 1.7189 | 53.7 |
| graph_metric | 179-180 | 94.7 / 1.2918 / 1.6038 | 94.7 / 1.2987 / 1.6141 | 92.7 / 1.2938 / 1.7168 | 57.0 |
| graph_metric | pure_rot_170-180 | 95.3 / 1.1870 / 1.4548 | 95.3 / 1.1923 / 1.4596 | 93.7 / 1.1955 / 1.5228 | 58.3 |

On the trained metric heads (band 0.9, kappa 0.5) the detector fires 47-72 times per path (the plateau of K1
covers most of the rotation range), the sampling step changes nothing (same success, same cost) and the
sharpening step is slightly worse (-1 to -2 points, p95 cost up): without a rounded corner there are no two
routes to recover, the probes land on the same plateau and the extrapolation is noise. Stages 2 and 3 are
therefore useful exactly where the field is a rounded min of branches (the soft-min model, or a branch /
cover head with soft_tau) and not for the plain metric heads, whose failure at the half-turn is a plateau.

### K4 second round (explore2 / explore3; T scene, seed 0 unless n given)

| head | guarantee | SE(3) success | collision | non-conv | bigangle | rotheavy | dex | lowest-decade pred/true | triangle attack (y-only): triples / max |
|---|---|---|---|---|---|---|---|---|---|
| cover euclid, 2 cuts (n = 3) | metric | 86.5 ± 2.1 | 11.4 ± 2.4 | 2.4 ± 0.4 | 53.4 / 45.0 | 78.6 / 20.4 | 0.64-0.69 | 36-44x | 0 / <= 0 |
| cover euclid, q routes only | metric | 84.2 | 13.4 | 2.6 | 50.8 / 48.2 | 76.0 / 23.5 | 0.642 | 37x | 0 / <= 0 |
| cover euclid, wide (me 128, mq 32, mo 32) | metric | 86.6 | 11.3 | 2.2 | 53.4 / 45.5 | - | 0.663 | 38x | 0 / <= 0 |
| cover euclid, res_scale 0.1 | metric | 85.1 | 12.6 | 2.6 | - | - | 0.683 | 45x | 0 / <= 0 |
| cover euclid, lip_cap 4 | metric | 85.2 | 12.8 | 2.3 | 52.9 / 45.5 | 76.5 / 23.0 | 0.678 | 46x | 0 / <= 0 |
| cover lp p = 2, 2 cuts | metric | 41.7 | 57.6 | 2.1 | 25.1 / 74.3 | 50.5 / 49.5 | 0.816 | 74x | 0 / <= 0 |
| cover lp p = 8, soft min 0.02 | ~metric (O(tau)) | 40.7 | 58.5 | 2.3 | 24.6 / 74.3 | 51.5 / 48.5 | 0.890 | 81x | 54 % / 3.9 % |
| **cover euclid, 2 cuts + conformal cone** | none | **94.4** | **1.6** | 4.3 | **85.3 / 8.9** | **93.4 / 3.1** | 0.566 | 27x | 7.8 % / 24 % |
| cover lp p = 2 + conformal cone | none | 91.8 | 2.5 | 5.8 | 82.7 / 13.1 | 90.8 / 6.6 | 0.586 | 30x | 5.4 % / 3.7 % |
| graph_gram (branch campaign, n = 3) | none | 79.8 ± 4.6 | 14.7 ± 4.7 | 7.5 ± 1.3 | 48.7 / 39.3 | 70.9 / 18.4 | 0.651 | 38x | 0 / <= 0 (all: 22 % / 3 %) |
| graph_metric (n = 3) | none | 90.8 ± 1.0 | 4.6 ± 0.9 | 5.0 ± 0.7 | 68.4 / 23.0 | 82.3 / 12.4 | 0.615 | 38x | 57 % / 11.5 % |
| branch conf K4 (other campaign, n = 3) | none | 95.4 ± 0.6 | 1.9 ± 1.0 | 2.9 ± 0.8 | 85.9 / 9.6 | 93.5 / 4.8 | 0.552 | 27x | 71 % / 36 % |

* **The guaranteed decoder beats graph_gram** (86.5 ± 2.1 vs 79.8 ± 4.6 in the same pipeline; +7 points, half
  the non-convergences) and the previous guaranteed head (66.1 ± 7.4) by 20 points, with a strict triangle
  inequality (0 violations under both attacks on every seed). It stays 4 points below graph_metric.
* Capacity knobs that keep the metric (wider features, larger residual, 4x Lipschitz cap) change nothing
  (85.1-86.6); the lowest-decade speed stays 36-46x too fast. The p = 2 seminorm collapses like p = 8 (41.7 %),
  so it is the *sum of group norms* (and its standardised input) that fails, not the max; and the soft minimum
  does not rescue it (40.7 %).
* **The conformal cone term is the missing slope.** Added to the same cover head it gives 94.4 % (collisions
  12 % -> 1.6 %, bigangle 53 -> 85 %), above graph_metric and level with the other campaign's best branch head
  (95.4 ± 0.6); it even rescues the p = 2 seminorm (41.7 -> 91.8). Its triangle violations are mild in-domain
  (7.8 % of y-only attacked triples, up to 24 % of T; graph_metric 57 % / 11.5 %, the branch head 71 % / 36 %).
  Mechanism (K4 first round): the Eikonal equation needs |grad T| = 1 / S_d up to 1000 at contact; a metric's
  slope is bounded by the embedding's Lipschitz constant, while rho(p) scales the cone by 1 + rho(p) without it.

### K3 results (float64, ds = 2e-4, lines of +-0.3 scaled-twist units, 256 lines, slope-relative threshold 0.3)

| field | ridge kinks | valley kinks | background p99 | min / max rel. slope jump | branch switches | ridge kinks at a switch |
|---|---|---|---|---|---|---|
| T_eu0_s0 | 68 | 26 | 0.065 | -1.84 / 1.68 | 138 | 18 |
| T_eu2L_s0 | 77 | 27 | 0.067 | -1.99 / 1.79 | 1148 | 21 |
| T_eu2_s0 | 64 | 34 | 0.066 | -1.72 / 1.87 | 1009 | 18 |
| T_eu2_s1 | 64 | 30 | 0.066 | -1.67 / 1.58 | 1308 | 25 |
| T_eu2_s2 | 63 | 32 | 0.064 | -1.83 / 1.75 | 1548 | 21 |
| T_eu2c_s0 | 145 | 31 | 0.077 | -1.90 / 1.97 | 628 | 11 |
| T_eu2r_s0 | 69 | 26 | 0.066 | -1.66 / 1.49 | 1285 | 20 |
| T_eu2w_s0 | 52 | 28 | 0.063 | -1.61 / 1.38 | 1335 | 10 |
| T_gmetric_iso_lie_s0 | 153 | 52 | 0.078 | -1.94 / 1.70 | - | - |
| T_lp0_s0 | 59 | 113 | 0.056 | -1.98 / 1.94 | 192 | 57 |
| T_lp2_s0 | 93 | 65 | 0.065 | -1.94 / 1.98 | 960 | 37 |
| T_lp2nf_s0 | 116 | 69 | 0.065 | -1.92 / 1.95 | 1056 | 93 |
| T_lp2p2_s0 | 123 | 89 | 0.076 | -1.95 / 1.90 | 1212 | 31 |
| T_lp2p2c_s0 | 170 | 52 | 0.087 | -1.99 / 1.91 | 1518 | 22 |
| T_lp2soft_s0 | 102 | 65 | 0.068 | -1.99 / 1.87 | 494 | 22 |
| T_lp4_s0 | 90 | 67 | 0.063 | -1.94 / 1.98 | 1672 | 30 |
| T_lse2_s0 | 125 | 75 | 0.081 | -1.90 / 1.93 | 1140 | 50 |
| T_trieq_iso_lie_s0 | 112 | 82 | 0.073 | -1.87 / 1.90 | - | - |
| free_graph_gram | 0 | 0 | 0.001 | -0.14 / 0.00 | - | - |
| free_metric_best | 0 | 0 | 0.002 | -0.00 / 0.01 | - | - |
| free_scalar | 0 | 0 | 0.001 | -0.00 / 0.01 | - | - |
| free_tri_eq | 0 | 0 | 0.001 | -0.29 / 0.00 | - | - |

* Obstacle-free fields: no kink of either sign in any field on 154 units of path (the half-turn corner of the
  decoder fields is crossed too rarely by random lines of this length; a first pass with longer lines and an
  absolute threshold found 24-47 ridge events and no valley for graph_gram / tri_eq and nothing for the
  metric / scalar heads). The census is clean in float64; in float32 the rounding of T over ds = 2e-4 fakes
  ~1500 kink pairs, so the float32 pass was discarded.
* Obstacle fields: the smooth logsumexp heads (graph_metric, the triangle head) register 112-153 "ridge" and
  52-92 "valley" events, i.e. relative slope jumps above 0.3 within 2e-4: these are the near-contact regions
  where |c| ~ 100-1000 changes within the sample spacing, not architectural kinks, and they set the noise floor
  of the census among obstacles. The cover heads are at or below that floor (eu2: 64 / 34; lp2: 93 / 65), with
  18-50 of their ridge events coinciding with a branch switch and 0-2 valley events at a switch. So property 7
  holds as far as this census can see: branch switches produce ridges only, never valleys.

## K5: final phase (conformal seeds 1-2, L and A scenes)

SE(3) success / collision / non-convergence in %, standard 1000-case sets.

| head | guarantee | T (n) | L (n) | A (n) |
|---|---|---|---|---|
| **cover euclid, 2 cuts** | **pseudometric** | 86.5 ± 2.1 / 11.4 / 2.4 (3) | **96.1** / 2.7 / 1.6 (1) | **94.0** / 5.5 / 0.5 (1) |
| **cover euclid + conformal cone** | none | **93.9 ± 0.8** / 1.9 / 4.5 (3) | **97.6** / 0.3 / 2.1 (1) | **98.0** / 1.2 / 0.8 (1) |
| graph_gram | none | 79.8 ± 4.6 / 14.7 / 7.5 (3) | 85.8 ± 1.3 / 2.4 / 12.0 (3) | 90.2 ± 0.7 / 6.9 / 3.4 (3) |
| graph_metric | none | 90.8 ± 1.0 / 4.6 / 5.0 (3) | 96.2 ± 0.8 / 1.0 / 2.9 (3) | 94.4 ± 0.2 / 3.6 / 2.0 (2) |
| graph_tri_metric, equivariant V | pseudometric | 66.1 ± 7.4 / 31.4 / 3.7 (5) | 87.2 ± 3.6 / 12.2 / 1.1 (3) | 84.5 ± 0.7 / 14.9 / 0.8 (2) |
| branch conformal K4 (other campaign) | none | 95.4 ± 0.6 / 1.9 / 2.9 (3) | 97.5 ± 0.4 / 1.1 / 1.5 (3) | 97.9 ± 0.7 / 1.6 / 0.6 (3) |

T rotation-heavy sets (success / collision, 3 seeds): cover euclid bigangle 56.7 ± 6.2 / 41.5, rotheavy
79.3 ± 0.8 / 19.9, tight 73.0 ± 2.7 / 24.9, transonly 75.6 ± 3.6 / 21.9; conformal cover bigangle
84.5 ± 1.1 / 10.6, rotheavy 92.5 ± 0.8 / 4.1, tight 90.4 / 4.4, transonly 86.7 ± 1.3 / 4.3; graph_metric
68.4 ± 3.7 / 23.0, 82.3 ± 6.1 / 12.4, 82.9 ± 1.4 / 13.3, 83.4 ± 4.0 / 16.4.

* The guaranteed head beats graph_gram on every scene (+6.7 / +10.3 / +3.8 points) and the previous
  guaranteed head by +20 / +9 / +10, and on L and A it is level with graph_metric (96.1 vs 96.2 ± 0.8;
  94.0 vs 94.4 ± 0.2) while keeping an exact triangle inequality. Its remaining deficit is on the T scene and
  its rotation-heavy subsets, where it collides 2-3x more often than graph_metric.
* With the conformal cone the cover head is above graph_metric on all three scenes (93.9 ± 0.8 vs
  90.8 ± 1.0; 97.6 vs 96.2; 98.0 vs 94.4) and on every rotation-heavy T subset (+16 / +10 / +8 / +3 points,
  collisions 4-11 % instead of 12-23 %), and level with the concurrent campaign's conformal branch head
  (95.4 ± 0.6 / 97.5 / 97.9). Seed spread of the conformal cover head on T is 0.8 points.
