# Experiment log: min-of-branches decoder (`models/graph_branch`, 2026-10-13)

Goal (user, 2026-10-13): the fixed (a, b) decoder of graph_gram, `T = sqrt(a + phi(b) / s_r^2)`, forces its
rotational kink where the learned `b` reaches 1 (the half turn of its Gram lift) and caps the rotation term at
`pi^2 / s_r^2`; that is not the truth among obstacles. Try decoders that satisfy the nine properties of
"Decoder properties that hold with or without obstacles", add constraints at the ends (the goal) of the
travel-time field, test every property, and show an improvement over graph_gram.

Hardware: 3 RTX 3090 (shared with the user's chart-vs-Lie bench and another session's `graph_cover` work),
36 cores, `pytorchserver:latest`. Pipeline for every obstacle run: models/graph_lie (original metric NTFields
trainer, 4000 epochs, AdamW wd 0.5, lr 5e-4, accept/reject, `exp(-tau/2)`, `1e-2 (sqrt(N) - 1)^2`), **Lie
gradient, isotropic speed** (`S_d |c| = 1`), the S-series baseline of graph_lie. Evaluation: per-case forward MPPI
(`plan_eval_graph.py --kind graph_native`, batch 125, seed 0), SE(3) success primary.

## B0: head and tests

`models/graph_method_triangle/branch_decoder.py` (heads `graph_branch`, `graph_branch_conf`), trainer
`models/graph_branch/` (+ end constraint), CLI `train/train_graph_branch.py`, queue
`scripts/run_branch_queue.py`. `tests/test_branch_decoder.py`: one test group per property, at random float64
parameters.

* First run of the tests: 10 / 13 pass. Failures and what they were:
  * cone (P5/P6) at eps 1e-3 vs 1e-4 differed by 52 %: at random weights the residual's curvature is huge
    (tau 5-30 at unit distance), so the quadratic regime is ~1e-4 wide. The limit is exact (checked at 1e-5 /
    1e-6, relative error < 1e-6 against `1 + rho(p)`); the radius of the metric-shaped cone is a property of
    the trained field and is measured there (`decoder_properties.py` P5 at eps 1e-2 / 1e-3 / 1e-4).
  * graph_gram "no lower bound" test: its random init happens to sit above T_free. The architecture admits
    `T < T_free` (a small A gives `T` = rotation term only); the test now uses such a parameter value (> 50 % of
    pairs below T_free).
  * peaks-only (P7) failed for the logsumexp residual on 1 of 12 lines: the coarse grid (h = 0.01) missed a
    narrow high-curvature region; at h = 1.25e-3 -> 1.6e-4 every positive second difference shrinks 8x (smooth),
    ridges keep their slope jumps (-20 to -600). Test now compares the two fine grids. By construction the
    logsumexp residual smooths |d| as sqrt(d^2 + 1e-6), i.e. it meets P7 only above a ~1e-3 scale; the group
    l_p residual (`res_kind='lp'`, p = 8, added) is smooth away from the goal and keeps the l_inf-like shape.
* The quaternion lift (Shepperd, pivot >= 1/2) round-trips to 1e-15; the value is invariant under flipping the
  sign of either lift (the routes and their residuals swap); the Lie gradient through the lift equals the
  exp-perturbation gradient and central finite differences (pairs within 1e-3 of a branch tie excluded).

## B1: exploratory round (T, seed 0)

| tag | head |
|---|---|
| ggram | graph_gram (previous decoder) |
| br1 | branch, K = 1 (only the +-q rotation routes), strict cone (property 6 as stated: free metric) |
| br4 | branch, K = 4 residual branches per route, strict cone |
| bc1e / bc4e | conformal cone (`1 + rho(p)` scale) + end constraint `S_d(p) (1 + rho(p)) = 1`, K = 1 / 4 |
| bc4 | conformal cone, no end constraint (ablation) |
| br4L / bc4Le | logsumexp residual (graph_metric's distance on the residual coordinates) |

Training (T, seed 0, iso / Lie, 4000 epochs, 8 runs at once on a saturated machine): 31 min (graph_gram) to
~45 min (branch), no rejected epochs, logged Eikonal loss 2.0-2.4e-3 for every head (the end-constraint runs log
~2e-3 more: the end term). With K = 4 the selected route is the long way round (> pi) for ~2 % of training pairs.

## B2: what the trained fields look like (decoder_properties.py, T scene, seed 0)

* graph_gram's decoder kink is never used among obstacles: along full-turn rotation slices at the goal's position
  its b peaks at a median of 0.29 and never exceeds 0.999 (0 / 200 slices): the Gram lift has collapsed towards a
  constant and the rotation term stays below 0.19, rotation is carried by the smooth a-latent. graph_metric
  (reference) the same (b_max median 0.80, 0 slices with a kink). Consequently neither has a single ridge along
  200 random lines (P7 "ridge lines" 0), while the branch heads cross ridges on 18-173 of them. The fixed decoder
  does not misplace the ridge among obstacles -- it loses it.
* Branch heads: T(x, x) = 0, symmetry exact, 0 pairs below T_free (graph_gram 0.46 %, min T / T_free 0.74),
  cone anisotropy exactly 1.00 (graph_gram 2.05, graph_metric 6.3). Conformal heads: S_d(p) x cone slope =
  0.91-0.98 at the goal (the cone matches the local speed; strict heads 0.355 by construction, graph_gram 0.81).
  Triangle violations 0.05-0.09 % of random triples (graph_gram 0).
* graph_metric's cone is not a cone below ~1e-3 (slope changes 66 % between eps = 1e-3 and 1e-4): its
  sqrt(d^2 + 1e-6) smoothing makes the metric quadratic at the goal; the branch heads' cones converge (< 1 %).

## B3: synthetic exact-truth benchmark, supervised fit (synth_fit.py, seed 0)

Representation only (labels everywhere): yaw slow sector (A = 4 / 20: the rotational ridge lies 61 / 96 deg
from the half turn and the max pure-rotation time is 0.90 / 2.49, above graph_gram's cap 0.5).

| sector A = 20 / A = 4 | graph_gram | graph_metric | branch K=1 | branch K=4 | branch K=4 lse |
|---|---|---|---|---|---|
| near-ridge error % | 6.73 / 5.89 | 0.52 / 0.95 | 3.00 / 2.13 | 0.70 / 0.62 | 0.32 / 0.30 |
| ridge height error % | 15.4 / 13.8 | 3.0 / 4.3 | 7.5 / 6.0 | 1.4 / 2.5 | 0.58 / 1.14 |
| whole-field median % | 1.42 / 1.67 | 0.31 / 0.38 | 2.01 / 1.68 | 1.02 / 0.71 | 0.71 / 0.43 |
| gradient-planner cost | 1.01 / 1.01 | 1.00 / 1.00 | 1.05 / 1.01 | 1.03 / 1.01 | 1.09 / 1.00 |

* With labels, graph_gram places its ridge correctly (0.75-2 deg median error; its Gram lift is learnable), but its
  ridge height is 14-15 % too low and the field near the ridge 6-7 % off: the capped rotation term. The branch
  heads are 10-25x better at the ridge; graph_metric's l_inf-type latent remains the best whole-field fit.
* Steepest descent on the branch fields costs 1-9 % more than on graph_metric: a gradient follower is sensitive to
  every branch switch; the obstacle evaluation uses the sampling MPPI.

## B4: exploratory planning (T, seed 0; per-case forward MPPI, batch 125; SE(3) success primary)

| head | standard SE(3) | collision | non-conv. | bigangle SE(3) | rotheavy SE(3) |
|---|---|---|---|---|---|
| graph_gram | 83.2 | 10.5 | 8.3 | 48.7 | 70.9 |
| graph_metric (graph_lie runs, 3 seeds) | 90.8 +- 1.0 | 4.6 | 5.0 | 68.4 +- 3.7 | 82.3 +- 6.1 |
| branch K=1, strict cone | 91.0 | 7.1 | 1.9 | 67.5 | (pending) |
| branch K=4, strict cone | 87.6 | 10.4 | 2.1 | 62.3 | 77.6 |
| branch K=4, logsumexp residual, strict | 89.5 | 4.2 | 6.5 | (pending) | (pending) |
| branch K=1, conformal + end | 94.5 | 3.0 | 3.0 | (pending) | (pending) |
| branch K=4, conformal + end | 93.8 | 3.6 | 2.9 | (pending) | (pending) |
| branch K=4, conformal, no end | **94.9** | **1.7** | 3.5 | **85.9** | **95.9** |

* Every branch head converges more often than graph_gram (non-convergence 1.9-6.5 % vs 8.3 %) and plans more direct
  paths (successful cases: translation ratio 1.22-1.23 vs 1.28, geodesic rotation ratio 1.32-1.33 vs 1.43).
* Attribution on the standard set: graph_gram 83.2 -> branch decoder with the strict cone 91.0 (K = 1; the exact
  +-q routes, T >= T_free, metric cone) -> + conformal cone and end constraint 94.5. The strict cone keeps
  graph_gram's collision rate (7-10 %); the conformal cone removes most of it.

## B5: why the conformal cone matters among obstacles (conf_diag.py, T seed 0)

* The training data lie mostly inside the clearance band (warped S_d median 0.15), where the Eikonal slope at the
  goal is 1 / S_d, not 1: the strict cone (slope 1) violates the equation at every slow goal by construction
  (S_d x slope 0.355 median, B2). The conformal heads learn the cone factor from the local speed: Spearman
  correlation of rho(p) with 1 / S_d(p) - 1 = 0.91 (bc4) / 0.92 (bc4e); (1 + rho) S_d per speed band
  (S_d < 0.01 / 0.01-0.03 / 0.03-0.1 / 0.1-0.3 / 0.3-0.6 / > 0.6): bc4 0.05 / 0.17 / 0.42 / 0.82 / 1.00 / 1.12,
  with the end constraint 0.07 / 0.24 / 0.55 / 0.93 / ... (closer to 1; the lowest decades stay compressed, as in
  the S-series of graph_lie).
* Far from the goal the conformal term saturates at rho_bar lam_bar, an additive cost that grows when either endpoint
  is close to an obstacle: a learned escape-time potential (the S-series' best planning change, there with an
  external clearance predictor). On random pairs tau ~ 1.45 T_free, of which T_free ~ 69 %, the conformal term
  6-17 % (largest for short pairs), the route residual 11-24 %.

## B6: properties 3, 4 and 6 cannot all hold by construction (theory note)

If a decoder guarantees, at every pose of SE(3), the triangle inequality (3), the speed-limit bound T >= T_free (4) and
the strict metric cone T(p exp(xi), p) = |xi|_g + O(|xi|^2) (6, uniform constant), then T = T_free. Proof: split the
free geodesic from x to y (length L = T_free(x, y)) into n equal steps; by (6) each step costs L / n + O(1 / n^2); by (3)
T(x, y) <= sum of the steps -> L as n -> infinity; with (4), T(x, y) = T_free(x, y). The network is defined on all of
SE(3), including poses inside obstacles, so obstacles do not interrupt the chain. Hence a strict-cone head with a
lower bound must give up the triangle inequality to represent any obstacle, which is why the branch head measures (3)
instead of guaranteeing it, and why the note can build (3) in only in special cases. With the conformal cone (scale
1 / S(p)) the same chain gives T <= int 1 / S along paths, the Riemannian distance of the speed-weighted metric: no
contradiction, the bound is the correct physics.

## B7: physics-trained synthetic problems (synth_dataset.py + synth_eval.py; Eikonal only, no labels)

Slow-sector datasets written in the NTFields format (400 k local + 20 % uniform pairs, S_d = 1, S_a = 1 / h(psi)),
trained by the same trainer with `--speed aniso --planar` (in-plane Eikonal norm), 2000 epochs, seed 0; branch heads
with res_scale 0.3 (init tau within ~3x of T*; a larger start sits in the exp(-tau/2) inflation basin of graph_lie L5).
The accept / reject rule rejects many epochs here (loss ~1e-5, noise > 20 %): 14-300 rejections per run.

| A = 20 | graph_gram | graph_metric (scale 0.05) | branch K=4 | branch K=4 Huber |
|---|---|---|---|---|
| field error median / p95 % | 1.96 / 54.0 | 9.6 / 62.9 | 1.29 / 14.1 | 1.42 / 9.4 |
| near-ridge error % | 27.2 | 51.4 | 7.9 | 5.8 |
| ridge position median / max (deg) | 8.3 / 136 | 11.8 / 179 | 3.3 / 60 | 1.5 / 24 |
| ridge height error % | 30.9 | 59.3 | 18.0 | 12.5 |
| gradient planner success % (cost) | 33 (1.00) | 73 (1.72) | 100 (1.23) | 100 (1.08) |

graph_gram's physics-trained field has a false goal: along pure-rotation slices its time drops to ~0.08 at a goal yaw
110-135 deg away from the start (true ~0.6), the almost-everywhere Eikonal solution with a fake second goal; steepest
descent stops there (33 % success). The architecture allows it (no lower bound, zero possible off the diagonal). The
branch heads cannot have one (T >= T_free > 0 off the goal) and place the ridge where the routes tie.

## B8: properties measured on all trained fields (decoder_properties.py; T: 3 seeds, L: seed 0; per-variant means)

| measure | graph_gram T / L | graph_metric T / L | branch conformal T / L | branch conformal + end T / L | branch strict K=4 T |
|---|---|---|---|---|---|
| P1 max T(p, p) / P2 max asymmetry | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| P3 triangle violations (% of triples) | 0.00 / 0.00 | 0.02 / 0.00 | 0.04 / 0.07 | 0.06 / 0.12 | 0.08 |
| P4 pairs below T_free % (min T / T_free) | 0.39 / 4.71 (0.74 / 0.53) | 0.61 / 4.91 (0.75 / 0.60) | 0 / 0 (1.08 / 1.04) | 0 / 0 (1.06 / 1.03) | 0 (1.07) |
| P5 cone: slope change 1e-3 -> 1e-4 | 0.4 % / 0.3 % | 66 % / 113 % | 0.4 % / 0.4 % | 0.6 % / 0.4 % | 0.8 % |
| P6 cone anisotropy | 2.05 / 1.84 | 6.35 / 9.48 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 |
| P6 S x cone slope at the goal (all / S < 0.05) | 0.83 / 0.08 (T) | 0.20 / 0.03 (T) | 0.97 / 0.12 (T) | 0.94 / 0.15 (T) | 0.36 / 0.01 |
| P7 lines with a V-kink / with a ridge (of 200) | 0 / 0 | 0 / 0 | 0 / 31 (T) | 0 / 82 (T) | 0 / 160 |
| P8 slices where b reaches the kink | 0 % / 0 % | 0 % / 0 % | - | - | - |
| P8 route switches > 10 deg off the half turn | - | - | 4 % / 1 % | 11 % / 3 % | 15 % |
| P9 max rotation term (cap 0.25) / rotation-slice T max | 0.21 / 1.83 (T) | 0.21 / 2.50 (T) | - / 3.34 (T) | - / 2.70 (T) | - / 1.85 |

The same picture as B2 on every seed and on L: graph_gram's decoder kink is never realised among obstacles and its
field has no ridges; it falls below the speed-limit time on up to 4.7 % of pairs (L). The branch heads keep the
architectural guarantees on trained weights and have real ridges; the conformal cone matches the local speed at the
goal (0.94-0.97 in the median; the slowest decade stays compressed).

B7 continued, A = 4 (milder sector): graph_gram field error 18.7 % (median) / rotation-slice error 81 % / ridge height
error 86 %, gradient planner 3.3 % success; graph_metric (scale 0.05) 19.7 % / 81 % / 87 %, 4.3 %; branch K=4 0.60 % /
1.25 % / 3.4 %, 100 % (cost 1.01); branch K=4 Huber 0.73 % / 1.5 % / 3.7 %, 100 %. Both earlier heads collapse to the
rotation-blind field (Eikonal loss ~1e-6, graph_gram 6000+ rejected epochs): the projection-metric degeneracy of the
physics-only endpoint Eikonal (graph_method report, Proposition 3). With T >= T_free the branch heads cannot.
Figures: results/graph_branch/synth_phys/sector_A{4,20}_phys_slices.png.

## B9: ablation -- the decoder structure vs the conformal goal constraint (T, 3 seeds each)

`graph_gram_conf` = graph_gram's own decoder plus only the branch head's conformal term C = rho_bar lam_bar tanh(D / lam_bar)
(no routes, no T_free, no metric cone; no end constraint, its cone slope is not 1).

| head | standard: SE(3) / collision / non-conv / iters | big-angle SE(3) / non-conv | rotation-heavy SE(3) / non-conv |
|---|---|---|---|
| graph_gram | 79.8 +- 4.6 / 14.7 / 7.5 / 61 | 45.9 +- 2.5 / 16.1 | 68.9 +- 5.9 / 12.9 |
| graph_gram + conformal term | 91.3 +- 1.5 / 1.9 / 7.0 / 64.7 | 80.5 +- 0.6 / 9.1 | 88.8 +- 4.0 / 6.6 |
| branch K=4 conformal | 95.4 +- 0.6 / 1.9 / 2.9 / 59.3 | 85.9 +- 5.2 / 4.9 | 93.5 +- 3.3 / 1.7 |
| branch K=4 conformal + end | 93.9 +- 0.9 / 3.8 / 2.7 / 59.7 | 83.2 +- 4.8 / 4.5 | 93.4 +- 2.0 / 1.4 |

Attribution. The conformal goal term is what removes the collisions (14.7 % -> 1.9 % on its own): its far-field part is
a learned escape-time cost. It does not fix convergence (non-convergence 7.5 % -> 7.0 %, more iterations). The branch
decoder structure on top is what fixes convergence (7.0 % -> 2.7-2.9 %; big-angle 9.1 % -> 4.5-4.9 %; rotation-heavy
6.6 % -> 1.4-1.7 %), shortens the rotation paths (ratio 1.42 -> 1.35) and adds +2.6 to +4.1 points on the standard set,
+2.7 to +5.4 on big-angle and +4.6 on rotation-heavy. On the exact-truth problems trained by the physics loss alone
(B7) the structure decides whether the field is right at all.

B9 continued: the structure alone (branch K=1, strict cone, no conformal term), 3 seeds: standard 90.4 +- 1.3 (collision
8.2, non-conv 1.7), big-angle 63.9 +- 3.3 (non-conv 4.9), rotation-heavy 82.3 +- 2.8 (non-conv 2.0). So each change alone
is worth about +11 points on the T standard set, through different failure modes: the structure removes the
non-convergence (7.5 % -> 1.7 %), the conformal goal term removes the collisions (14.7 % -> 1.9 %); together +15.6.

## B10: final 3-seed planning results (results/graph_branch/summary/plan_summary.md)

| SE(3) success % | graph_gram | graph_metric | branch strict K=1 | branch strict K=4 | gram + conformal | branch conformal | branch conformal + end |
|---|---|---|---|---|---|---|---|
| T standard | 79.8 +- 4.6 | 90.8 +- 1.0 | 90.4 +- 1.3 | 86.7 +- 2.8 | 91.3 +- 1.5 | **95.4 +- 0.6** | 93.9 +- 0.9 |
| L standard | 85.8 +- 1.3 | 96.2 +- 0.8 | - | - | - | **97.5 +- 0.4** | 96.8 +- 0.2 |
| A standard | 90.2 +- 0.7 | 94.4 +- 0.2 (2) | - | - | - | **97.9 +- 0.7** | 96.9 +- 0.5 |
| T big-angle | 45.9 +- 2.5 | 68.4 +- 3.7 | 63.9 +- 3.3 | 58.1 +- 4.2 | 80.5 +- 0.6 | **85.9 +- 5.2** | 83.2 +- 4.8 |
| T rotation-heavy | 68.9 +- 5.9 | 82.3 +- 6.1 | 82.3 +- 2.8 | 75.9 +- 2.5 | 88.8 +- 4.0 | **93.5 +- 3.3** | 93.4 +- 2.0 |

Every branch-head row is a paired improvement over graph_gram on every set and seed. Paths of successful cases are
more direct (geodesic rotation ratio 1.19-1.33 vs 1.30-1.47) and converge in fewer MPPI iterations (51-60 vs 56-62).
Strict K=4 < strict K=1: extra residual branches without the conformal cone add collisions. The end constraint
calibrates the cone (B8) and costs 0.7-2.7 points of success relative to the same head without it.
Report page: https://claude.ai/artifact/LNcBSPjSW5bt4bxrTbbs4B (results/graph_branch/report/branch-decoder.html).

## B11: both metric baselines in every comparison; methodology report (2026-10-13, later)

Baselines now in every chart and table: graph_metric (graph_lie runs, iso / Lie) and the **original metric NTFields**
("metric baseline": chart input, combined speed, chart gradient, its own trainer; Lie-Shape-NTFields checkpoints seeds 1-3).
Its per-case results were already scored by the same evaluator (results/graph_method_cusp/plan/{T,L,A}_native_s{1,2,3},
batch 125, seed 0); aggregate_branch.py now reads them (unpaired: its seeds do not pair with ours).

| SE(3) success % | metric baseline | graph_gram | graph_metric | branch conformal | branch conformal + end |
|---|---|---|---|---|---|
| T standard | 82.1 +- 0.4 | 79.8 +- 4.6 | 90.8 +- 1.0 | **95.4 +- 0.6** | 93.9 +- 0.9 |
| L standard | 89.9 +- 2.0 | 85.8 +- 1.3 | 96.2 +- 0.8 | **97.5 +- 0.4** | 96.8 +- 0.2 |
| A standard | 87.1 +- 1.6 | 90.2 +- 0.7 | 94.4 +- 0.2 (2) | **97.9 +- 0.7** | 96.9 +- 0.5 |
| T big-angle | 34.2 +- 3.9 | 45.9 +- 2.5 | 68.4 +- 3.7 | **85.9 +- 5.2** | 83.2 +- 4.8 |
| T rotation-heavy | 80.1 +- 3.6 | 68.9 +- 5.9 | 82.3 +- 6.1 | **93.5 +- 3.3** | 93.4 +- 2.0 |

Paired gain of the conformal branch head over graph_metric: +4.6 (T), +1.3 (L, within noise), +3.2 (A, 2 seeds), +17.5
(big-angle), +11.2 (rotation-heavy).

New tools: `chart_metric.py` (the original network as arch `metric_chart`; `load_original` maps its checkpoints; matches
the original forward to 3e-6), `decoder_properties.py --orig` (+ jump count: a step that does not shrink under 8x
refinement; T(p,p) offset subtracted in the slopes), `cone_calibration.py` (speed x cone slope at the goal by speed bin,
own speed model and metric per head), `pdf_figures.py`, `pdf_synth_slices.py`, `make_latex_tables.py`, `style.py`.

Findings / corrections:
* Metric baseline fields: T(p,p) = 3.2e-3, never below T_free (T/T_free >= 1.6 on T), cone broken below 1e-3
  (anisotropy 4.6-5.9), jumps at the chart seam on 80-105 of 200 lines; every graph head has 0 jumps.
* graph_gram's lift: the median peak of b on rotation slices is 0.29-0.85 across scenes / seeds (0.29 was seed 0 only);
  no slice reaches 0.999 for graph_gram or graph_metric.
* Cone calibration (T, results/graph_branch/summary/cone_calibration_T.json): median S x slope at 1e-4 = 1.01 (branch
  conformal) and 0.96 (+ end) -- the conformal head learns the scale from the Eikonal loss alone; the end constraint only
  raises the slow-goal bins by 25-35 % (0.16 -> 0.23 at S ~ 0.02), which stay far below 1. graph_gram 0.90 (exact cone),
  graph_metric 0.72 -> 0.20 and metric baseline 0.63 -> 0.06 from 1e-2 to 1e-4 (cone lost below the logsumexp smoothing).
  This corrects B10's "the end constraint calibrates the cone".
* Speed statistics corrected: S_d (iso speed of the graph heads) median 0.39 at the T training endpoints, 23 % below 0.1;
  0.15-0.16 was the combined speed S_c of the original method.
* Synthetic, original network (label fit `morig`, physics `S4/S20_morig05_aniso_s0`): label fit 0.4-0.65 % field error,
  ridge height 3.5-9.2 % (rounded, like graph_metric). Physics only: it does NOT collapse (raw rotation-vector input, sigma 1),
  median error 1.0 / 0.8 %, but p95 22 % (A = 4) and ridge height 33 / 12 % from the jump at its seam (in the middle of the
  slow sector here). On A = 20 it beats the branch heads in median and p95 error.
* The one V-kink flagged on L_bc4e_s2 (line 55) is a smooth valley ~1e-3 wide (second derivative 1.4e5, constant under
  float64 refinement), below the detector's step; not a kink.

Methodology report (LaTeX, 21 pages): results/graph_branch/report/branch_decoder_methodology.pdf (source:
results/graph_branch/report/latex/main.tex, tables generated by make_latex_tables.py, figures by pdf_figures.py,
pdf_synth_slices.py, obstacle_slices.py). Report page updated (version 2, same URL).
