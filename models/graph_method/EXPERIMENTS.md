# Experiment log — SE(3) smooth graph travel-time fields

Chronological record of everything tried (successes and failures). Times are
wall-clock on 4x RTX 3090 inside the `pytorchserver:latest` image
(torch 2.2.2, CUDA 12.1). Obstacle-free benchmark, s_t = s_r = 1, box [-1,1]^3,
L_ref = 1 m, tau_ref = 1 s unless stated.

Relative error metrics exclude pairs with T* < 0.05 tau_ref. "uni" = 10k
uniform held-out pairs, "mix" = 10k pairs from the training mixture
(60 % uniform / 20 % small relative perturbation / 20 % near half-turn).

## TL;DR (final state)

* **Obstacle-free SE(3) (the spec's scope, stages A-D):** `graph_gram` (siamese
  G-Shape encoder + linear input skip, PSD-Gram rotation latent, analytic
  decoder), trained with the spec objective + relative-angle curriculum + cosine
  LR, reaches **0.105 ± 0.003 % median / 0.31 ± 0.02 % p95 relative error, exact
  half-turn error 0.002, 100 % planning success at cost ratio 1.000** (3 seeds).
  The G-Shape metric head under identical data/budget/planner: 9.5 % / 23 %,
  98.6 % / 1.12. The decoder reproduces the cut-locus corner (E7, final table in
  README).
* Decisive fixes, in order of discovery: Fourier frequency for 12-d pose features
  (E1-E2), the PSD-Gram head, cosine LR + angle curriculum + input skip (E6-E7).
  Physics-only (no anchors) training is limited by a projection-metric
  degeneracy of the endpoint Eikonal (E10).
* **Obstacles (stage E, 1000 cases per environment):** a graph head with a
  metric-type a-latent (`graph_metric`) is on par with the metric head in the
  same pipeline — T-shape 79.2 ± 0.2 vs 80.0 ± 0.6 % (3 seeds), L-shape 82.9 ± 0.7
  vs 81.4 ± 0.4 %, A-shape 81.5 ± 0.8 vs 81.5 ± 1.1 % (2 seeds; mean 81.2 vs 81.0 %); `graph_gram`'s Euclidean
  a-latent is 5-11 points behind among obstacles. The native baseline pipeline
  is still ahead (T 81.2 % on this machine; L/A 86-92 % per its notes) (E14, E15).

## E0 — Stage A: analytic geometry (no learning)

`python -m unittest tests.test_graph_method` — 19/19 pass (float64):
SE(3)/SO(3) exp/log round trips including near-pi, orthogonality/determinant,
quaternion sign invariance, endpoint symmetry and left-invariance of the
oracle, pure translation / pure rotation times, exact-pi and diagonal values,
decoder vs independent atan2 oracle (< 1e-8 relative), phi series and
derivative (phi'(0) = 4), tangent projection == exp-perturbation autodiff ==
central finite differences (step sweep converges), both endpoint Eikonal
equations |c|^2 = 1 to < 1e-6 on regular pairs, oracle Bellman admissibility,
no oracle-label leakage in the physics loss, sampler reproducibility, full
gradient and sampling rollouts to the goal on the analytic field.

Bugs found by the tests and fixed:
* phi series switch at b < 1e-2 had 6e-12 truncation error -> switch at 1e-3.
* graph_sqdist saturated b to exactly 1.0 in float -> phi'(1) = inf -> NaN
  gradients. All learned range adapters now keep b <= 1 - 1e-6 (half-turn
  representable to 2e-3 rad; documented).

## E1 — first smoke runs (1000 steps, batch 256)

| run | val uni p50 | uni p95 | note |
|---|---|---|---|
| graph_gram physics | 0.84 | 0.90 | field collapses to ~16 % of T*; 80 ms/step |
| metric physics | 0.80 | 0.86 | same collapse |
| graph_gram supervised | 0.21 | 0.67 | supervised fit stalls -> representation/optimisation problem |

Diagnosis: all heads were limited by the random-Fourier input frequency, not
by the head. With 12-d (pairwise 24-d) pose features the original
`B ~ N(0, sigma^2)` gives |B_j| ~ sigma sqrt(d) cycles per unit, i.e. 1.7-2.4
cycles/unit at sigma = 0.5 -- far above the quadratic latents' content. The
G-Shape default was tuned for 3-d inputs. The joint (24-d) networks could not
learn at all at sigma = 0.5 (predicted T ~ 0).

## E2 — Stage B: supervised representation diagnostic, Fourier sigma sweep

2000 steps, batch 256, Adam 1e-3, L = L_Z + L_T + 10 L_goal (graph) or
L_T + 10 L_goal (value heads). Validation relative error (4000 pairs):

| head | sigma | uni p50 | uni p95 | mix p50 | mix p95 | exact-pi abs |
|---|---|---|---|---|---|---|
| graph_gram | 0.03 | **0.0062** | **0.0180** | **0.0071** | **0.0195** | **0.047** |
| graph_gram | 0.08 | 0.0095 | 0.0283 | 0.0113 | 0.0293 | 0.068 |
| graph_gram | 0.15 | 0.0168 | 0.0524 | 0.0198 | 0.0602 | 0.108 |
| graph_sqdist | 0.03 | 0.0160 | 0.0627 | 0.0176 | 0.0664 | 0.154 |
| graph_joint | 0.03 | 0.0146 | 0.0508 | 0.0184 | 0.9680 | 0.095 |
| scalar | 0.03 | 0.0167 | 0.0483 | 0.0204 | 0.9733 | 0.122 |
| metric | 0.03 | 0.0556 | 0.1568 | 0.0595 | 0.3767 | 0.261 |
| metric | 0.08 | 0.0453 | 0.1362 | 0.0518 | 0.2905 | 0.226 |
| metric | 0.15 | 0.0495 | 0.1442 | 0.0541 | 0.2419 | 0.227 |
| metric | 0.25 | 0.0857 | 0.3040 | 0.0916 | 0.3697 | 0.316 |

* graph_gram (siamese encoder, PSD-Gram rotation latent) reproduces the
  analytic latents/time best; inside the provisional 1 % / 5 % gates already.
* Joint/scalar heads fail on the mixture's small-relative-pose pairs (p95 ~
  97 %): without an exact diagonal zero they cannot form the T = |.| cone.
* The G-Shape metric head is limited to ~5 % median even with labels: a sum
  of group-wise (smooth) L-infinity norms cannot represent the Euclidean
  local norm nor the cut-locus corner.

## E3 — throughput

Physics step after restructuring (pre-generated sample pool, one batched value
forward, greedy controls reused from the Eikonal pass, no per-step syncs):
44 ms/step for batch 256 *and* 1024 (launch-overhead bound) -> batch 1024 used
for all physics runs; several runs share a GPU.

## E4 — Stage C first physics runs (batch 1024, Bellman h = 0.2 unless noted)

Spec objective L = L_E + L_B + 10 L_goal + L_T(1024 declared anchors), Adam 1e-3, clip 1.
Validation relative error (uni p50 / p95):

| run | 1k | 3k | 5k | 8k | 14-20k |
|---|---|---|---|---|---|
| graph_gram | 9.9 / 32 % | 4.4 / 20 % | — | 4.0 / 20 % | 4.6 / 20 % (plateau) |
| graph_sqdist | 5.8 / 21 % | 3.7 / 17 % | — | 3.2 / 16 % | 3.2 / 17.5 % |
| graph_sqdist + input skip | — | 3.2 / 14 % | 3.2 / 16 % | — | — |
| metric (spec loss) | 9.3 / 23 % | 8.9 / 23 % (best ~1.5k) | — | 10.8 / 35 % | 12.3 / 37 % (degrades) |
| metric (NTFields loss, intrinsic) | 10.7 / 35 % | 14 / 50 % | — | 13 / 42 % | 11.4 / 31 % |
| scalar | 25 / 50 % | 10.5 / 34 % | 5.6 / 26 % | 7.3 / 20 % | 5.9 / 21 % (mix p95 39 %) |
| graph_gram, no anchors | 48 / 78 % (collapse to ~half of T*) | | | | |
| metric NTFields, no anchors | 37 / 65 % | 39 / 67 % | | | |

Recipe probes on graph_gram / graph_sqdist (5k steps): Bellman h 0.02 / 0.2 / 0.5,
EMA 0.9, lr 5e-4 / 3e-3, anchor weight 3 / 10, batch 4096, Eikonal form
'abs' / 'log', T-weighted Eikonal (p = 1, 2), no Bellman, narrower cut-locus band
1e-6, 40 % near-pi sampling, spectrally sharpened Gram (k = 2, 3).
None moved the plateau by more than ~0.5 pt; Bellman x10, lr 3e-3, log-form,
T-weighting and Gram sharpening were clearly worse. Removing Bellman changed
little (3.3 / 17.8 %): the Eikonal + anchors dominate.

## E5 — where does physics training fail? (scripts/diag_eikonal.py)

Near-pure rotations (|dt| = 0.3), per relative-angle bin, plateaued checkpoints:

| angle | sqdist |c|^2 | |c_t|^2 (true) | |c_r|^2 (true) | b (b*) | metric |c_r|^2 | metric T/T* |
|---|---|---|---|---|---|---|
| 0-15 | 1.09 | 0.76 (0.75) | 0.33 (0.25) | 0.011 (0.008) | 0.20 | 1.05 |
| 90-105 | 1.04 | 0.10 (0.03) | 0.94 (0.97) | 0.53 (0.57) | 0.23 | 0.73 |
| 150-165 | 0.76 | 0.34 (0.01) | 0.42 (0.99) | 0.89 (0.96) | 0.09 | 0.58 |
| 165-180 | 0.47 | 0.40 (0.01) | 0.08 (0.99) | 0.92 (0.99) | 0.06 | 0.54 |

Failure mode: near the cut locus the learned b stays below 1 with a smooth
maximum, so the rotational slope vanishes and the network "explains" |c| = 1 with
spurious translation sensitivity. The sqdist adapter makes this worse: its smooth
min has db/dz -> 0 as b -> 1, so dT/dz ~ 2 pi sqrt(1-b)/delta -> 0. The metric
baseline shows the same failure much more strongly (rotation almost ignored,
T/T* = 0.54 at the half-turn).

## E6 — is the correct solution a stable minimum of the physics loss? (warm start)

Stage-B checkpoint (graph_gram supervised, batch 1024, 4k steps: **0.11 % / 0.33 %**,
exact-pi 0.010) used to initialise physics training — labelled as warm-start runs,
not training from scratch:

| run | 0.5k | 1k | 2k | 2.5k |
|---|---|---|---|---|
| warm, anchors, lr 1e-3 | 2.0 / 5.8 % | 1.8 / 5.4 % | 1.8 / 5.1 % | 1.75 / 5.1 % |
| warm, **no anchors**, lr 1e-3 | 2.5 / 6.7 % | 2.5 / 6.9 % | 1.8 / 5.5 % | 2.6 / 7.6 % |
| warm, **no anchors**, lr 1e-4 | 0.8 / 2.1 % | 1.1 / 2.7 % | 0.66 / 1.7 % | 0.73 / 1.9 % |

The analytic solution sits in a stable basin of the physics loss (no anchors
needed to stay there); lr 1e-3 gradient noise alone costs ~2 pt. Training from
scratch therefore fails by reaching the wrong basin, not because the loss is
biased. -> E7: cosine LR decay + relative-angle curriculum.

## E7 — cosine LR + relative-angle curriculum (+ input skip) — breakthrough

Curriculum (input geometry only): training pairs (and anchors) restricted to
relative angle <= theta_max, theta_max growing linearly 60 deg -> 180 deg over
the first 6k of 10k steps; cosine LR 1e-3 -> 5e-5. `skip_input` concatenates the
fixed embedding G(x) to the encoder output before the pair head (a linear path;
the head must still learn the projections — no analytic latent is hard-coded).

| run | 1k | 2k | 3k | 4k (<=140 deg) | 5k (<=160 deg) | 6k (<=180 deg) | 8-10k |
|---|---|---|---|---|---|---|---|
| gram, cosine only (no curriculum) | 9.6 / 31 % | 5.1 / 22 % | 4.0 / 19 % | 3.7 / 18 % | 3.6 / 18 % | 3.9 / 18 % | |
| gram + curriculum (no skip) | 23 / 50 % | 20 / 47 % | 18 / 43 % | 17 / 41 % | | | |
| sqdist+skip + curriculum, b_soft 0.02 | 0.53 / 5.3 % | 0.29 / 4.3 % | 0.34 / 4.1 % | 0.22 / 4.1 % | 0.50 / 3.9 % | 3.4 / 9.8 % | 3.2 / 15 % |
| sqdist+skip + curriculum, b_soft 1e-4 | 0.54 / 3.8 % | 0.56 / 3.4 % | 0.44 / 2.8 % | 0.25 / 1.6 % | | | |
| same, **no anchors** | 0.56 / 4.1 % | 0.41 / 2.7 % | 0.62 / 3.8 % | 0.19 / 1.2 % | 13 / 50 % (blow-up) | 14 / 49 % | 13 / 50 % |
| sqdist+skip, b_soft 1e-4, no curriculum | 9.1 / 27 % | 7.3 / 23 % | 7.0 / 22 % | 6.6 / 22 % | 6.1 / 20 % | | |
| **gram+skip + curriculum** | 14 / 29 % | 0.51 / 2.5 % | 0.46 / 1.4 % | 0.23 / 0.74 % | **0.14 / 0.41 %** (pi 0.021) | 1.7 / 4.8 % | |
| metric+skip + curriculum | 31 / 55 % | 32 / 56 % | 28 / 51 % | 24 / 47 % | 21 / 44 % | 16 / 39 % | |

(values: validation uni rel p50 / p95)

* The siamese graph field trained on pairs up to 140-160 deg extrapolates to the
  whole of SO(3) almost exactly (the per-pose embedding is shared by all pairs).
* Admitting the last near-pi pairs (> 160 deg) into L_E disrupts training:
  phi'(b) -> inf makes (|c|^2 - 1)^2 explode on a few samples (loss spikes to
  1e3-1e4). The sqdist head does not recover; the PSD-Gram head degrades to
  1.7 / 4.8 % but stays in the right basin. -> E8: band / residual-form sweep.
* The metric head does not benefit from the curriculum (16 / 39 %).

## E8 — cut-locus band / residual-form sweep on gram+skip+curriculum (10k steps)

| run | 5k | 6k (near-pi enters) | 7k | 9-10k (final) |
|---|---|---|---|---|
| pi_band 1e-4 (spec default) | 0.14 / 0.41 % | 1.7 / 4.8 % | 1.1 / 3.0 % | **0.10 / 0.32 %**, exact-pi 0.0025 |
| pi_band 1e-3 | 0.09 / 0.27 % | 0.76 / 2.2 % | 0.18 / 0.38 % | (see final runs) |
| pi_band 1e-2 | 0.09 / 0.26 % | 0.30 / 0.94 % | 0.37 / 0.80 % | 0.38 / 0.69 % (wider band = less near-pi supervision) |
| eik_form 'abs' / 'log' | similar trajectories to 'sq' at 2-4k | | | |
| **no anchors** | 2.6 / 21 % | 4.2 / 20 % | 2.3 / 19 % | p95 stuck ~20 % |

No-anchor failure analysis (evaluate.py): rotation is learned well (pure
rotation 0.9 % / 2.7 %) but the translation latent folds (pure translation
9.7 % median, 66 % p95): without labels |grad sqrt(a)| = 1 admits folded
embeddings (non-viscosity Eikonal solutions) and the Bellman term only fills
spurious minima by ~h per target update.

## E9 — Stage E (obstacles, Tshape3d_env1), first attempts — failed so far

Speed model: ellipsoidal with s_t = 1, s_r = 2 pi (rotations in turns, as the
baseline), per-pose multipliers from the dataset after the baseline warp
s^2 (2-s)^2. Graph_gram + skip, Fourier sigma_t 1.0 / sigma_r 0.3, batch 2000,
20k steps, cosine LR, no anchors (no ground truth exists), evaluated with the
UNMODIFIED Shape-NTFields evaluate_training_3d_batched.py through
models/graph_method/shape_adapter.py (scripts/run_shape_eval.py).

Reference (same evaluator, same 1000 RRT-verified test cases, from
Lie-Shape-NTFields/results): metric baseline 82.1 % / 80.3 % (seeds 1, 2);
Lie ellipsoidal metric 67.5 %.

| run (12k steps, 200 cases) | success | note |
|---|---|---|
| aniso speeds, spec residual | 0 % | all failures non-convergence |
| iso speeds, spec residual | 0 % | |
| aniso, sqrt residual | 0.5 % | |
| iso, sqrt residual | 1.0 % | |

Diagnosis (scripts/diag_shape_field.py): our fields have T / T_free ~ 0.5-1.3
(T_free = obstacle-free time, a physical lower bound since speeds <= 1) while
the working baseline field has T / T_free ~ 3.0 (p5 2.4, p95 4.1). With
c = S grad T the spec residual's push on |grad T| scales like S^2 (~3e-6 at
the band's S ~ 0.0016), so the near-obstacle Eikonal is effectively ignored.
Log-form residuals (round 2) did not help (held-out residual ~0.98). A control
run of the *metric head inside our pipeline* failed the same way, so the gap
is in the obstacle training recipe, not the graph head. One recipe difference
found: NTFields weights the residual by exp(-0.5 T) WITHOUT detaching T, which
pushes T up wherever the residual is large (an implicit anti-collapse force)
-> round 3 with that weighting (graph and metric heads).

## E10 — why physics-only training under-estimates: projection-metric degeneracy

`scripts/diag_translation.py` on the no-anchor checkpoint: for pure
translations T/|dt| has median 0.9 but p5 ~0.35 at *every* distance, including
|dt| < 0.25 — the learned local metric is degenerate in some directions, not
folded far away. Explanation: near the diagonal T(x, x + d) ~ sqrt(d^T M d) and
the endpoint Eikonal requires |M d|^2 = d^T M d for all d, i.e. M^2 = M. Every
orthogonal projection satisfies it, not just M = I: fields such as
T = |P (x - y)| have zero Eikonal residual everywhere and extra zeros along
null(P). The Bellman min picks the steepest-descent action and never "sees" the
zero-cost direction, so it does not penalise them either (only the spurious
zero set itself, by ~h per target update). Declared anchors remove the
degeneracy (E7/E8); without labels it is the dominant failure. The same
under-estimation appears among obstacles (T / T_free < 1, E9).

Stage-E remedy (valid only there): all speeds are <= 1, so T(x, y) >= T_free(x, y)
(free-space time at maximal speed) holds for every environment. A hinge
penalty relu(T_free - T)^2 uses pose geometry only. It is NOT used for the
obstacle-free benchmark, where T_free is the exact answer.

Also tried for the obstacle-free no-anchor case: NTFields exp(-0.5 T) residual
weighting with non-detached T (pushes T up where residual is large):
2.8 / 21 % at 4k — no improvement over 2.6 / 21 %.

## E11 — Stage E rounds 3-6 (Tshape3d_env1, 20k steps, batch 2000, iso speed, sqrt residual)

All with the NTFields exp(-0.5 T) residual weighting with non-detached T ("ewg").
Planning success = Shape-NTFields evaluator, forward MPPI, first 200 test cases.

| run | head | key settings | success |
|---|---|---|---|
| g_iso_sqrt_ewg | graph_gram + skip | sigma_t 1.0, sigma_r 0.3, Adam cosine | 22.5 % (collision 143 / no-conv 77) |
| m_iso_sqrt_ewg | metric | same | 5.5 % (no-conv 188) |
| g_noskip_iso_sqrt_ewg | graph_gram | no skip | 6.0 % |
| sq_noskip_iso_sqrt_ewg | graph_sqdist | no skip | 5.0 % |
| g_skip_lb | graph_gram + skip | + speed lower bound w 10 | 22.0 % |
| m_lb | metric | + speed lower bound w 10 | 10.1 % |
| g_adamw | graph_gram + skip | AdamW wd 0.5, lr 5e-4 | 26.5 % |
| **g_sr01** | graph_gram + skip | **sigma_r 0.1**, Adam cosine | **67.0 %** |
| **g_sr01_adamw** | graph_gram + skip | sigma_r 0.1, AdamW | **69.0 %** |
| m_sr01 | metric | sigma_r 0.1, Adam cosine | 60.3 % |
| m_sr01_adamw | metric | sigma_r 0.1, AdamW | 71.0 % |
| c_m | metric, **coords mode** (baseline inputs/gradients) | sigma 1.0, AdamW | 68.5 % |

* Speed-bin diagnostic (scripts/diag_shape_eik.py vs the baseline field): the
  sigma_r 0.3 fields had |grad T| nearly independent of the clearance speed;
  the baseline's varies ~10x. The decisive fix was the rotation Fourier
  frequency: with G(x) = (t, vec R) inputs, sigma_r = 0.3 gives ~2.7 rad of
  phase per radian of rotation versus ~1 for the baseline's (t, rotvec/2pi)
  inputs at sigma 1; sigma_r = 0.1 matches it.
* With matched frequency the graph field is on par with the metric head in the
  same intrinsic pipeline and with a replica of the baseline pipeline (coords
  mode) on these 200 cases.
* Bug found on the way: checkpoints written before `feature_dim` existed could
  not be reloaded (KeyError) -> `load_function` now backfills defaults; several
  evaluations were re-run.

## E12 — Stage E rounds 7-9: frequencies, rollback, baseline-replica calibration

Baseline calibration: the actual metric-baseline checkpoint (Lie-Shape-NTFields
Tshape3d_env1_metric_baseline_s1, epoch 4000) scored with its own evaluator on
the same first 200 cases: **82.0 %** (its 1000-case score is 82.1 %; research notes
give 81.9 +- 0.5 over 5 seeds). The Lie-Shape and Shape-NTFields evaluators are
identical for the default chart MPPI sampler, the test pairs and the training
data are byte-identical, and the baseline was trained with TD and normal losses
OFF (research_tree), matching our replica.

| run | head / pipeline | Fourier sigma (t / R) | other | success (200) |
|---|---|---|---|---|
| g_t1_r005 | graph, intrinsic | 1.0 / 0.05 | AdamW | 67.5 % |
| g_t2_r01 | graph, intrinsic | 2.0 / 0.1 | AdamW | 52.5 % |
| **g_t05_r01** | graph, intrinsic | **0.5 / 0.1** | AdamW | **77.5 %** |
| g_t1_r01_lb | graph, intrinsic | 1.0 / 0.1 | + speed lower bound | 70.5 % |
| g_t1_r01_aniso | graph, intrinsic | 1.0 / 0.1 | anisotropic speeds | 11.0 % |
| m_t1_r005 | metric, intrinsic | 1.0 / 0.05 | AdamW | 75.5 % |
| c_m | metric, coords (replica) | 1.0 (clamped B) | AdamW | 68.5 % |
| c_m_rb | metric, coords (replica) | 1.0 (resampled trunc-normal B) | + NTFields rollback (0 rollbacks fired) | 74.5 % |

* Lower translation frequency helps the graph head monotonically
  (sigma_t 2 -> 1 -> 0.5: 52 -> 68 -> 77.5 %).
* The NTFields accept/reject rollback never triggered (0 rollbacks) — not the gap.
* Residual replica gap to the real baseline (74.5 vs 82 %) is unexplained so far;
  remaining differences are gradient clipping (norm 1; mean grad norm ~0.3) and
  the regular-sample mask (~4 % of pairs within 3.6 deg of a half-turn).
  -> round 9 removes both (`clip=1e9, diag_band=0, pi_band=0`).
* 200 cases => standard error ~3 points; single-seed differences below ~6
  points are not meaningful.

## E13 — Stage E rounds 9-10 (200 cases, 1 seed each)

| run | head / pipeline | sigma t / R | notes | success | collision | no-conv |
|---|---|---|---|---|---|---|
| c_m_exact | metric, coords replica | 1.0 | no grad clip, no regular mask | 76.0 % | | |
| c_g_exact | graph, coords | 1.0 | no clip, no mask | 67.0 % | | |
| g_exact | graph, intrinsic | 1.0 / 0.1 | no clip, no mask | 70.0 % | | |
| g_t03_r005 | graph, intrinsic | 0.3 / 0.05 | | 78.0 % | | |
| g_t03_r01 | graph, intrinsic | 0.3 / 0.1 | | 77.5 % | | |
| g_t05_r005 | graph, intrinsic | 0.5 / 0.05 | | 78.5 % | 15 | 31 |
| m_t05_r01 | metric, intrinsic | 0.5 / 0.1 | | **81.0 %** | 1 | 37 |
| baseline checkpoint | metric, native pipeline | 1.0 | reference | 82.0 % | 19 | 29 |

* Replica details (clipping, mask) are within noise: the replica sits at
  74.5-76 % vs the checkpoint's 82 %, gap unexplained (one seed each).
* At matched frequencies the metric head is ~3 points ahead of graph_gram on
  these 200 cases (within the ~3-point standard error) -> decisive comparison
  with 3 seeds x 1000 cases (E14).
* Hypothesis for a structural gap among obstacles: graph_gram's a-latent is a
  *Euclidean* squared embedding distance, but geodesic distances among
  obstacles are generally not Euclidean-embeddable, whereas the metric head's
  sum of (smooth) L-infinity norms can approximate arbitrary metrics. Test:
  `graph_metric` head, a = m(x, y)^2 with m the G-Shape metric distance, the
  same PSD-Gram b and the same analytic decoder (spec Stage E: "consider
  richer latents").

## E14 — Stage E final comparison (Tshape3d_env1, all 1000 RRT-verified cases)

Identical pipeline for every head: `configs/graph_method/stageE_common.json`
(intrinsic tangent gradients, G(x) inputs, iso speed, sqrt residual, NTFields
exp(-0.5 T) weighting with grad, AdamW wd 0.5 lr 5e-4, batch 2000, 20k updates,
Fourier sigma_t 0.5 / sigma_r 0.1, resampled truncated-normal B), unmodified
Shape-NTFields evaluator, forward MPPI success:

| head | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| metric head (G-Shape) | 80.2 % | 80.5 % | 79.3 % |
| graph_gram (Euclidean a-latent) | (pending) | 74.7 % | 75.1 % |
| graph_metric (a = metric-head distance^2, Gram b, analytic decoder) | 79.1 % | (running) | (running) |

* The metric head inside our intrinsic pipeline matches the native baseline
  (82.1 / 80.3 %, 5-seed 81.9 +- 0.5): the Stage E pipeline is validated.
* graph_gram is ~5 points behind among obstacles. Supports the E13 hypothesis
  that its Euclidean a-latent (squared distance of a linear projection of the
  embedding) is too restrictive for obstacle geodesics; the hybrid
  `graph_metric` (same smooth-latent + analytic-decoder structure, a-latent from
  the metric head) recovers parity on the first seed.

Update (E14): graph_gram seed 0 = 75.4 % -> graph_gram 75.1 +- 0.4 % (3 seeds).
Width test: graph_gram with k_a = 128 (vs 16) = 74.2 % -> latent *width* is not
the limitation; replacing the Euclidean a-latent by a metric-type one is what
recovers parity (graph_metric 79.1 % on seed 0).

The hybrid is NOT a universal head: on the obstacle-free benchmark (same final
recipe as graph_gram, seed 0) graph_metric reaches only 8.7 % / 29 % median / p95
relative error, exact-pi error 0.76, planning 99.1 % / cost 1.15 — i.e. metric-
baseline level. The exact Euclidean a-latent is the right inductive bias for the
free product metric (a* = |dt|^2), a general metric latent is better among obstacles.

Evaluation note: the Shape-NTFields checkout only contains the T-shape/env1
meshes, so Lshape3d_env3 / Ashape3d_env4 are scored with the Lie-Shape-NTFields
copy of the same evaluator (identical code path for the default chart sampler,
identical test sets — verified with cmp).

## E15 — Stage E on two more environments (1 seed, 1000 cases, same pipeline)

| env | metric head | graph_metric | graph_gram |
|---|---|---|---|
| Tshape3d_env1 (3 seeds) | 80.0 ± 0.6 % | 79.2 ± 0.2 % | 75.1 ± 0.4 % |
| Lshape3d_env3 (seeds 0, 1) | 81.1, 81.7 -> 81.4 ± 0.4 % | 83.4, 82.4 -> **82.9 ± 0.7 %** | 70.3 % (seed 0) |
| Ashape3d_env4 (seeds 0, 1) | 82.3, 80.7 -> 81.5 ± 1.1 % | 80.9, 82.1 -> 81.5 ± 0.8 % | 75.6 % (seed 0) |
| mean of env means | 81.0 % | 81.2 % | 73.7 % |

Native-pipeline metric baseline (research notes, not re-run here): L 87.7-91.6 %,
A 85.9-88.9 % — the native pipeline's lead over our pipeline is larger on these
environments and affects both heads equally.
