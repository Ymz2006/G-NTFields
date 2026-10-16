# Experiment log: Lie gradient vs chart gradient for the graph heads (`models/graph_lie`)

Chronological (2026-10-12). Hardware and image as before: 4x RTX 3090, `pytorchserver:latest`
(torch 2.2.2). Every training run uses the original metric NTFields pipeline of `models/graph_native`
(4000 epochs, 5 x 2000 pairs per epoch, AdamW wd 0.5 lr 5e-4, accept/reject rollback, `exp(-tau/2)`
weighting, `1e-2 (sqrt(N) - 1)^2` residual). Only the gradient in the Eikonal residual (chart / Lie)
and the speed formulation (combined / iso / aniso) vary. Heads: `graph_metric` and the triangle head
(`graph_tri_metric`, `rot_mode=equivariant`, `rot_bias=1`), default `graph_native` options.

## L0: reproduction of `models/graph_native`

`graph_lie` with `--speed combined --grad chart` re-trains `graph_native`'s T_gmetric_chart_s0 bit for
bit: the logged loss / mean tau at epochs 1, 50, 100 are 1.1585e-02 / 4.387, 3.7691e-03 / 1.415,
3.2902e-03 / 1.391 in both. Likewise `--grad lie` against T_gmetric_lie_s0 (1.2730e-02 / 4.395,
3.8148e-03 / 1.364, 3.3551e-03 / 1.340). Logs: `Experiments/graph_lie/repro_check/`. So the 16
existing `graph_native` runs (combined; chart T s0-2, L/A s0-1; Lie T s0) are members of this study,
and their existing per-case results are reused (same checkpoints, same evaluator, seed 0).

## L1: unit tests

`python -m unittest tests.test_graph_lie`: 11 tests, all pass.

* Lie gradient by tangent projection = autograd through `x exp(xi)` (relative 1e-9, both heads, float64).
* Lie gradient = `Jr^{-1}(2 pi r)^T` applied to the chart gradient up to 175°, translation norms equal.
* chart gradient = central finite differences; the two coincide at the identity rotation.
* rotation-block factor `(theta/2)/sin(theta/2)` orthogonal to `r`, 1 along `r`.
* the exact free-space geodesic has `|c| = 1` to 1e-6 everywhere but `|g| != 1` at large `|rotvec|`
  (median deviation > 1 %, max > 10 % beyond 126°).
* `combined` loss and its parameter gradients are bitwise equal to `graph_native`'s, both modes.
* `warp_speeds` is bitwise the speed preprocessing of `graph_native` (S) and of Lie-Shape-NTFields'
  `lie_metric` / `lie_isotropic` trainers (S_d, S_a); the iso / aniso formulas.
* the Lie loss is finite with finite parameter gradients at `|rotvec| = pi` exactly.
* a 3-epoch run writes checkpoints that `graph_native`'s loader (the evaluator's) restores to the
  identical field.

One test tolerance was wrong at first (iso vs aniso with `S_a = S_d` differ by where the `1e-8`
regulariser sits, exactly as in the Lie trainers): fixed in the test, not the code.

## L2: throughput

100-epoch probes with 1, 2, 3 and 4 runs per GPU: ~0.37 s/epoch in every case (launch-bound, the GPU
is not the limit). The study runs 16 trainings at once (4 per GPU) plus 3-5 evaluation containers;
with the CPU saturated (load ~40 on 36 cores) that is 0.47 s/epoch for Lie runs and 0.53 s/epoch for
chart runs: the Lie path differentiates with respect to the 12-d embedding once, the chart path goes
through `so3_exp` in the double backward.

## L3: protocol

* 108 runs: {graph_metric, triangle} x {combined, iso, aniso} x {chart, Lie} x {T, L, A} x 3 seeds
  (92 trained here, 16 reused from `graph_native`). Same seed = same initialisation and data order in
  both gradient modes, so Lie vs chart is a paired comparison.
* Evaluation 1, the official planner: `plan_eval_graph.py --kind graph_native` (forward MPPI of
  `evaluate_training_3d_batched.py`, seed 0, chart convergence plus SE(3) convergence recorded). T: the
  standard 1000-case set, bigangle, rotheavy, transonly, tight_3d; L, A: standard and tight_3d.
* Evaluation 2, the Lie planner: `scripts/plan_eval_lie.py --sampler lie` (Lie-Shape-NTFields
  `controllers/lie_mppi.py`, unchanged). Checked first: with `--sampler chart` the copy reproduces
  `plan_eval_graph.py` exactly (GN_T_gmetric_chart_s0, rotheavy: chart 66.3 %, SE(3) 85.2 %). T:
  standard, bigangle, rotheavy; L, A: standard.
* Field diagnostics (`scripts/field_diag.py`): both Eikonal equations evaluated on every trained
  field, binned by `|rotvec|`.
* Obstacle-free control with exact ground truth (`scripts/make_free_dataset.py`, `scripts/free_eval.py`):
  {graph_metric, triangle} x {chart, Lie} x 3 seeds on `datasets/3dshape/free_se3` (all speeds 1, so
  the three formulations coincide). The exact field planned by the same MPPI reaches 98.9 % SE(3)
  success at path cost 1.12 (the planner's own floor) and only 52.6 % under the chart criterion.
* Primary metric: SE(3) success. Chart success undercounts seam-free fields
  (`results/graph_method_cusp/REPORT.md` 4.2); it is reported alongside.

## L4: restart after the shared disk filled up

The first launch (12:47) ran 16 trainings for ~25 min; at ~13:09, when the runs were writing their
epoch-3000 checkpoints, the shared root filesystem reached 100 % (36 MB free) and every run died in
`torch.save` (`unexpected pos ...`). Two Lie-MPPI result files were truncated mid-write. The disk was
already close to full before the launch; free space was not checked first. After space was freed
(25.8 GB available) the study was relaunched at 13:39 with:

* the crashed runs, their logs and the truncated results moved to
  `Experiments/graph_lie/crashed_diskfull_20261012/` (not reused);
* weights-only checkpoints and only `latest.pt` per run (written atomically through a `.tmp` file):
  ~2.5 MB per run instead of ~36 MB (the evaluator's loader needs no optimizer state);
* `run_queue.py --min-free-gb 3`: nothing is launched while less than 3 GB are free, and a result file
  counts as done only if it ends with `}` (complete JSON).

## L5: the obstacle-free ground-truth control and the NTFields loss

Goal: with all speeds 1 the exact travel time is known (T* = SE(3) geodesic), so the Lie and chart fields
can be scored against ground truth. In this pipeline physics-only training does not find T* for either
gradient, for two loss-level reasons that have nothing to do with the gradient:

* **NTFields weight `exp(-tau/2)` (not detached): degenerate inflation.** Runs `F_*_s0` (4 runs, 4000
  epochs): every field ends at tau ~ 20 T* (median relative error 19.5-20.5, bias +19.5-20.5, MPPI 33-37 %
  SE(3) success), mean tau 17 vs T* ~ 0.7, while the logged loss is small (3.6e-4) because the weight
  suppresses it. For a rescaled field tau = lambda T* the loss is ~ (sqrt(lambda) - 1)^2 exp(-lambda T*/2):
  lambda = 1 is a minimum, but past a barrier at lambda ~ 5 (unit speed, T* ~ 0.7) the loss decreases
  monotonically to 0 as lambda grows, and the random initialisation starts at lambda ~ 15. In the obstacle
  scenes the small speeds move the initial field much closer to the solution and the runs converge (mean
  tau 0.6-1.2), which is why this never showed up there. Chart and Lie collapse identically.
* **Detached weight: the accept / reject rule stalls.** As the inflated field deflates the detached weights
  grow, so the loss rises; the trainer rolls back every epoch whose loss rises by >= 20 % (84 rejections in
  the first 8 epochs of a probe, `Experiments/graph_lie/probes/F2probe_*`).
* **Unweighted residual (`--weight none`): converges, to a rotation-blind field.** graph_metric probes
  (800 epochs, chart and Lie): |c| = |g| = 1.00 in every |rotvec| bin, i.e. the Eikonal equation holds in
  both metrics, which is only possible because the field barely depends on rotation (tau ~ |dt|): T* is
  underestimated by 16 % (6 % below 90° relative rotation, 25 % at 170-180°) and MPPI reaches the goal pose
  in 5-6 % of the pairs. This is the projection-metric degeneracy of the physics-only endpoint Eikonal
  (graph_method EXPERIMENTS.md E10: fields |P(x - y)| solve it); with no rotational gradient the chart and
  Lie equations coincide, so it cannot discriminate them either.

Logging note: like the original trainer, the logged `T_mean` / `N_med` (and the loss) are sums over the 5
batches of an epoch divided by 4, i.e. 1.25 x the batch mean / median.

# Speed range of the graph_metric field: speed-conditioned Fourier embedding (S-series, 2026-10-12)

Problem (user observation, confirmed by `results/graph_lie/iso_report/speed_calibration.json` and the new
`scripts/speed_range_eval.py`): the trained graph_metric fields compress the speed range. Predicted speed
`1 / |c|` (Lie gradient) vs the warped clearance speed `S_d` on 8192 training pairs (16384 endpoints) per
seed, T scene, iso formulation, 3 seeds pooled (`results/graph_lie/speed_range/table.md`):

| field | dex all | dex bins | dex low (<0.05) | dex high (>=0.5) | N med | median pred/true per log bin (0.001 ... 0.5-1) |
|---|---|---|---|---|---|---|
| graph_metric, Lie gradient | 0.248 | 0.654 | 0.770 | 0.090 | 0.95 | 29.9 20.2 9.0 5.4 2.8 1.8 1.3 1.07 0.89 |
| graph_metric, chart gradient | 0.248 | 0.644 | 0.752 | 0.102 | 1.00 | 30.0 19.2 8.8 5.1 2.7 1.7 1.2 1.01 0.85 |

(`dex` = mean |log10(pred / true)|; `dex bins` averages it over the nine log bins so every speed decade
counts equally.) The lowest decade is predicted 20-30x too fast, the top bin ~12 % too slow.

Diagnosis: `|grad tau|` is bounded by the architecture. The embedding `sin(2 pi B x)` has
`|B| <= 2 sigma_t = 1` cycle per unit, the encoder's rows are Lipschitz-normalised to 1, and the
metric head scales by `metric_scale = 0.2`; an untrained field has `|c| ~ 5-10`, while `1 / S_d` reaches
1000 (`S_d` is clamped at 0.001; 5 % of endpoints have `S_d < 0.01`). The field cannot vary on the scale of
the clearance (margin 0.05).

## S0: the speed predictor

`speed_fourier.SpeedPredictor`: G(x) -> log S_d (Fourier features + SiLU MLP, output bounded to
[log 1e-3, 0]), trained by `scripts/train_speed_predictor.py` with an MSE in log space.

* From the training endpoints alone (1.52 M poses, 5 % of pairs held out) it does not generalise: a
  128-frequency / 256-wide / 3-layer model reaches validation RMSE 1.25 (log units; 50 % within x2) and
  predicts the lowest bin 66x too fast; larger / higher-frequency models (256-512 frequencies, sigma_t 2-8,
  512-1024 wide) memorise (train MSE 0.16-0.4, validation RMSE 1.5-2.0, rising). Cause, measured with a
  k-d tree over 400 k endpoints: the nearest other training pose is 0.05 away (median, chart units) and
  the clearance speed changes by a factor 1.6 (median) over that distance, by 1.08 below 0.01. The samples
  are too sparse for the function they are supposed to teach.
* `scripts/clearance_oracle.py` therefore evaluates the clearance with Lie-Shape-NTFields' own placement
  evaluator (exact point-to-triangle distance, tetrahedral inside test; 20 k env surface points) at
  random poses: dense supervision from the geometry, the same quantity the dataset stores. Checked on 50 k
  stored endpoints: median |dist error| 0.001 (= the offset, 2 % of the margin), 84 % within x1.5 in speed;
  the tail is 5 % of stored endpoints that the oracle (with its own surface sampling) marks as colliding.

## S1: more frequency capacity alone makes the low-speed calibration worse (T, iso, Lie, seed 0)

Controls without a speed predictor, same trainer, 4000 epochs, scored by `speed_range_eval.py`
(16384 endpoints; the baseline rows are 3 seeds pooled):

| field | dex all | dex bins | dex low | dex high | N med | median pred/true per log bin (0.001 ... 0.5-1) |
|---|---|---|---|---|---|---|
| baseline graph_metric (sigma_t 0.5) | 0.248 | 0.654 | 0.770 | 0.090 | 0.95 | 29.9 20.2 9.0 5.4 2.8 1.8 1.3 1.07 0.89 |
| Lipschitz cap 4 (`lip_cap=4`) | 0.242 | 0.636 | 0.740 | 0.090 | 0.97 | 28.8 18.2 8.4 5.0 2.6 1.7 1.2 1.03 0.89 |
| four fixed octave bands (sigma_t 0.5 ... 4, `sf_mode=fixed_bands`) | 0.294 | 0.764 | 0.913 | 0.110 | 0.93 | 53.1 35.9 12.8 7.7 3.8 2.4 1.6 1.18 0.84 |
| sigma_t 2, sigma_r 0.4 | 0.372 | 0.943 | 1.172 | 0.153 | 0.85 | 138 77 30 17 7.3 3.7 2.1 1.19 0.80 |
| `sf_mode=scale` with the sample-trained predictor (alpha 0.5, f_max 16) | 0.372 | 0.944 | 1.147 | 0.130 | 0.82 | 168 74 34 16 7.2 3.4 2.3 1.36 0.86 |
| `sf_mode=bands` with the sample-trained predictor (K 4, alpha 0.34) | 0.349 | 0.880 | 1.076 | 0.133 | 0.92 | 100 56 25 13 6.4 3.4 2.1 1.24 0.79 |

Raising the input frequencies (globally or in fixed bands) does not let the field follow the clearance:
the fields become *more* uniform in gradient magnitude (free-space speed drops to 0.80-0.84 of the
truth, band speed rises to 50-170x). Relaxing the Lipschitz cap changes nothing. The sample-trained
predictor spans only a 2.5x range of frequency factors (its own predictions cover 0.08-0.53), so the two
speed-conditioned runs behaved like global frequency increases. Conclusion: the bound on |grad tau| is
not the binding constraint; a Fourier layer that is sharp everywhere has to be *cancelled* by the
Lipschitz-normalised layers in free space, and training does not find that solution. The conditioning
has to be spatially selective (S2, with the geometry-trained predictor), or the 1/S gradient has to be
supplied structurally (S3, escape-time potential).

## S2: the structured (exact) clearance predictor

The dense oracle only helps a little: a free pose of this scene is within the margin of something 90 % of
the time (the room is tight: 4 boxes + a wall around a 0.29-unit shape), so the near-contact layer that
matters (S_d < 0.05, 5 % of poses) stays sparse at any affordable sample count in 6-d. The clearance is,
however, a composition of two simple objects, so `clearance.py` computes it instead of regressing it:

    d(x) = min_i sdf_env(R p_i + t),   p_i = 2048 samples of the shape surface (vertices, edges, faces),

with `sdf_env` the exact signed distance of the env (env1 = five axis-aligned boxes, one of them the room
itself whose sign is flipped automatically; a 192^3 igl grid is the general fallback for mesh scenes).
The minimising point is selected without autograd and the sdf re-evaluated at it, so memory is O(N) and
the gradient is the exact gradient of the active point. `raw_speed = clip(d / margin, offset/margin, 1)` and
the trainer warp give S_d; the module has no trainable parameters, its buffers travel in the checkpoint.

Validation (`build_clearance_predictor.py`, 100 k stored endpoints and 100 k dense-oracle poses, dex over
warped speeds, median |clearance error| in the band):

| predictor | vs stored labels: dex / within x1.5 / median |d err| | vs 20 k-point oracle: dex / within x1.5 / median |d err| |
|---|---|---|
| 6-d MLP from samples (S0) | 0.39 / 0.50 (val) | - |
| exact, 1024 points, smooth-min tau 0.003 | 0.58 / 0.35 / 0.0068 (logsumexp bias ~ tau log K) | 0.51 / 0.40 / 0.0061 |
| exact, 1024 points, hard min | 0.26 / 0.68 / 0.0017 | 0.17 / 0.77 / 0.0010 |
| exact, 2048 points, hard min, d_bias 0.002 | **0.21 / 0.74 / 0.0017** | **0.14 / 0.81 / 0.0015** |
| exact, 2048 points, hard min, d_bias 0.004 | 0.20 / 0.72 / 0.0028 | 0.17 / 0.73 / 0.0031 |

The stored labels are the distance to 10 k *sampled* env surface points, which overestimates the exact
clearance by a fraction of the point spacing; `d_bias = 0.002` mimics that (the oracle, 20 k points, is
biased by about half as much). Below S_d ~ 0.01 the labels and the exact clearance still disagree by a
factor 2-4 in warped speed (the label noise floor at near contact). `Experiments/graph_lie/speed_pred/T_clearance.pt`.

## S3: speed conditioning with the exact predictor (T, iso, Lie, seed 0, 4000 epochs)

| variant (`--arch graph_metric_sf`, `speed_pred_ckpt=T_clearance.pt`) | dex all | dex bins | dex low | dex high | N med | median pred/true per log bin (0.001 ... 0.5-1) |
|---|---|---|---|---|---|---|
| baseline graph_metric (3 seeds) | 0.248 | 0.654 | 0.770 | 0.090 | 0.95 | 29.9 20.2 9.0 5.4 2.8 1.8 1.3 1.07 0.89 |
| `scale` alpha 0.5 f_max 16, all 12 columns | 1.228 | 1.135 | 1.179 | 1.036 | 14.1 | does not train (tau_mean 8.9, N_med 17.6) |
| `scale` alpha 1.0 f_max 64, all columns | 1.951 | 1.707 | 1.668 | 1.589 | 63 | does not train |
| `scale` alpha 0.5 f_max 16, translation columns only | 0.353 | 1.038 | 1.297 | 0.099 | 0.88 | 349 178 79 32 6.2 2.5 1.6 1.18 0.90 |
| `bands` K 4, alpha 0.34 | 0.225 | 0.622 | 0.706 | 0.089 | 0.89 | 36.4 20.8 9.0 4.8 2.2 1.37 1.15 1.09 1.01 |
| `bands` K 4, alpha 0.34, translation columns only | 0.260 | 0.698 | 0.815 | 0.089 | 0.89 | 48.9 27.7 11.5 5.6 2.7 1.63 1.23 1.14 0.96 |
| **`bands` K 8, alpha 0.7** (sigma_t 0.5 ... 64) | 0.217 | **0.573** | **0.628** | 0.089 | 0.89 | **27.7 14.6 6.5 3.7 1.9 1.41 1.19 1.16 0.99** |
| potential `sf_psi=fixed` (rho 0.2) | 0.233 | 0.619 | 0.712 | 0.089 | 0.97 | 28.0 17.2 8.0 4.6 2.5 1.62 1.24 1.05 0.88 |
| potential `sf_psi=fixed`, rho 0.05 | 0.231 | 0.614 | 0.700 | 0.088 | 0.97 | 31.5 16.8 7.7 4.4 2.4 1.67 1.21 1.06 0.89 |
| potential `sf_psi=learned` | 0.228 | 0.603 | 0.686 | 0.088 | 0.98 | 27.0 17.2 7.4 4.2 2.3 1.54 1.19 1.04 0.90 |
| potential fixed + `scale` 0.5/16 all columns | 1.225 | 1.149 | 1.197 | 1.025 | 13.7 | does not train |

Observations.

* Frequency *scaling* by the predicted speed fails outright on the 12-d input (the rotation columns
  times f up to 16-64 make the embedding noise-like at band poses; the trainer rejects most epochs and
  tau stays ~9). On the translation columns only it trains but produces the worst low-speed calibration
  seen (349x): fast *oscillatory* features at band poses are averaged out by the encoder into a plateau.
  High Fourier frequency is the wrong primitive for a sharp *monotone* ramp.
* Speed-gated octave *bands* are the first positive result: K 8 lowers `dex bins` 0.654 -> 0.573, and
  fixes the top of the range (0.89 -> 0.99) and the 0.01-0.2 decades (5.4/2.8/1.8 -> 3.7/1.9/1.4). The
  lowest decade stays ~28x. The fixed-band control without the predictor was *worse* than the baseline
  (0.764), so the selectivity is what helps, not the bandwidth.
* The additive escape-time potential is switched off by training: `psi_gain` 1 -> 0.07 (both variants).
  Decomposing the Lie gradient of the trained field (`scratchpad/psi_diag.py`): at band endpoints the
  potential contributes |c| ~ 3-4 on top of the encoder's ~15, where 1/S is 70-600. Two causes:
  (i) the training pairs are *local* (x1 = x0 + a random displacement, both in the band), and for a local
  pair the potential form `(psi_x + psi_y)(1 - kappa)` has the wrong gradient (it should be the conformal
  stretch |x - y| / S, not an escape time); (ii) at S_d < 0.02 the exact predictor's 1/S exceeds the
  labels' by 2-4x (label noise floor, S2), and the residual `(sqrt N - 1)^2` punishes a 3x overshoot as
  much as a 14x undershoot, so the gain is driven down.
* `N med` of the baseline (0.95) is *not* improved by any variant; `dex high` is identical everywhere
  (0.088-0.090): the free-space calibration is set by the loss / weighting, not by the architecture.

## S4: the planner is the arbiter (per-case MPPI, T standard 1000-case set, `plan_eval_graph.py --kind graph_native`)

| field (seed 0) | SE(3) success | SE(3) converged | collision | chart success | median iters | median tau(start) |
|---|---|---|---|---|---|---|
| baseline graph_metric s0 / s1 / s2 | 91.9 / 90.3 / 90.1 | 95.5 / 94.2 / 95.4 | 4.0 / 4.3 / 5.6 | 78.8 / 77.9 / 78.7 | - | - |
| potential `sf_psi=learned` | **92.4** | 92.6 | **0.2** | **80.0** | 60 | 0.74 |
| `bands` K 8 | 0.6 | 0.6 | 0.2 | 0.4 | - (never converges) | 0.18 |

The 8-band field, best by the speed-range statistics, is useless for planning: tau is 3-4x smaller
than the baseline's, increases along 25 % of the path steps (median `tau_frac_increase` 0.255 vs 0.0),
paths wander (median translation length 1.02 vs 0.62) and nothing converges. The sigma 32-64 bands give
the field local minima that the pointwise Eikonal statistics do not see. The learned-potential field,
whose speed calibration moved little, plans *better* than every baseline seed: collisions drop from 4-6 %
to 0.2 % (the escape-time term raises tau wherever the predicted clearance is small, so the planner keeps
away from contact), SE(3) success +0.5-2 points, chart success +1-2 points. A speed-range gain is only
worth having if the planner confirms it; from here on every candidate is planned.

## S5: seeds and the coordinate / input forms of the escape time (T, iso, Lie, 4000 epochs; CPU evaluation)

GPU 2 fell off the bus (Xid 79) during this batch; the NVIDIA driver then hung every CUDA teardown
machine-wide (processes stuck in `uvm_va_space_destroy`, `nvidia-smi` in D state). The seven runs below
finished and saved before that, the evaluations ran on CPU (`speed_range_eval.py --device cpu`); the runs
`psil_s2` and `bands8psil_s0` (GPU 2) are lost. Same protocol as S3: 8192 pairs, seed 0; dex = mean
|log10 pred/true|; per-bin = median predicted/true speed, bins 0.001-0.002 ... 0.5-1.

| group | dex bins | dex low (<0.05) | dex high | N_med | per-bin median ratio |
|---|---|---|---|---|---|
| baseline s0 / s1 / s2 | 0.665 / 0.660 / 0.636 | 0.786 / 0.780 / 0.743 | 0.084-0.094 | 0.92-0.99 | 29.9 20.2 9.0 5.4 2.8 1.8 1.3 1.07 0.89 (pooled) |
| `bands` K 8 s0 / s1 / s2 | 0.573 / 0.604 / 0.570 | 0.628 / 0.673 / 0.623 | 0.078-0.089 | 0.88-0.90 | 26.7 15.2 6.8 3.7 2.0 1.44 1.22 1.18 0.98 (pooled) |
| `sf_psi=learned` s0 / s1 | 0.603 / 0.621 | 0.686 / 0.715 | 0.088 | 0.97-0.98 | 27.0 17.8 7.4 4.4 2.4 1.59 1.22 1.06 0.89 (pooled) |
| `sf_psi=coord` (a += (gain (Psi_x - Psi_y))^2) | 0.635 | 0.737 | 0.090 | 0.99 | 28.1 18.4 7.8 4.9 2.6 1.66 1.19 1.01 0.88 |
| `sf_psi_in=0.3` (Psi as 13th Fourier input) | 0.746 | 0.884 | 0.081 | 0.95 | 87.3 43.3 16.1 6.2 2.7 1.48 1.15 1.08 0.91 |
| coord + psi_in 0.3 | 0.729 | 0.855 | 0.080 | 0.96 | 83.7 38.9 14.0 5.8 2.5 1.44 1.14 1.09 0.91 |
| `bands` K 8 + coord | 0.572 | 0.610 | 0.103 | 0.85 | 28.9 14.6 6.3 3.5 1.85 1.35 1.17 1.19 1.03 |
| predictor s_hat itself (vs labels) | 0.375 | 0.580 | 0.024 | - | 0.94 0.61 0.28 0.23 0.44 0.78 0.92 1.01 1.01 |

- Seed spread of the baseline is 0.03 in dex bins; the learned potential (-0.05) and the 8 bands (-0.08)
  are outside it, the coordinate form (-0.02) is not. Nothing moves the lowest decade: every field predicts
  the 0.001-0.002 bin 27-30x too fast, and the only variants that change it (psi as an input) make it 3x
  worse (87x), i.e. feeding the escape time to the Fourier layer gives the network a sharp feature it uses to
  flatten the field near contact instead of steepening it.
- `bands` K 8 + coord equals `bands` K 8 alone on these statistics and inherits its sigma 32-64 bands, so
  it is expected to fail planning the same way (S4); not planned.
- The predictor row explains the ceiling: against the stored labels the exact clearance predictor is itself
  3-4x too *slow* at S_d 0.005-0.02 (ratio 0.23-0.28; a 0.0015 clearance error is 60 % of the clearance there),
  so the `psi` and `bands` terms are driven by a signal that is wrong by the same factor as the error they are
  meant to remove, in the opposite direction. The residual (sqrt N - 1)^2 then punishes the overshoot and the
  trained field settles between the two.
- Conclusion of the S-series: the low-speed compression is not a frequency-capacity problem (S1) and is not
  fixed by conditioning on an external speed signal (S3, S5); planning, not the pointwise speed statistics,
  decides (S4): `sf_psi=learned` is the one variant that helps the planner (collision 0.2 % vs 4-6 %).

## S6: planning after the reboot (T standard 1000 cases, `plan_eval_graph.py --kind graph_native`, batch 125, seed 0)

The machine was rebooted (GPU 2 recovered); the CPU planning runs were lost and everything below was
re-planned on GPU with the baseline's settings (batch 125: the planner seeds torch once, so the batch
size changes the MPPI noise stream).

| field | SE(3) success | SE(3) converged | collision | chart success | median iters | median tau(start) |
|---|---|---|---|---|---|---|
| baseline s0 / s1 / s2 | 91.9 / 90.3 / 90.1 | 95.5 / 94.2 / 95.4 | 4.0 / 4.3 / 5.6 | 78.8 / 77.9 / 78.7 | 59 | 0.75-0.79 |
| `sf_psi=fixed` s0 | **94.4** | 94.7 | 0.4 | **80.3** | 60 | 0.75 |
| `sf_psi=learned` s0 / s1 | 92.4 / 92.6 | 92.6 / 93.6 | **0.2 / 1.0** | 80.0 / 79.3 | 60 / 59 | 0.74 / 0.76 |
| `sf_psi=coord` s0 | 91.8 | 94.4 | 3.0 | 78.9 | 60 | 0.78 |
| `bands` K 4 s0 | 12.5 | 12.5 | 0.7 | 11.9 | 84 | 0.28 |
| `bands` K 8 s0 | 0.6 | 0.6 | 0.2 | 0.4 | 154 | 0.18 |

- The additive escape-time potential is a robust planning gain: both forms and both seeds cut collisions
  from 4-6 % to 0.2-1.0 % and raise chart success by 1-2 points. The fixed form (gain and rho trained,
  no correction MLP) is the best field planned so far: SE(3) success 94.4 % (+2.5 over the best baseline
  seed) with the baseline's convergence rate; the learned form trades 2 points of convergence for the
  lowest collision rate.
- The coordinate form (|Psi_x - Psi_y| as an extra metric coordinate) changes nothing: the potential has
  to enter additively to shape tau near contact.
- 4 bands (sigma up to 8) already lose the planner (12.5 %); the band gating itself, not only the sigma
  32-64 bands of K 8, creates the local minima. Speed conditioning of the Fourier frequencies is closed.
- Queued: `sf_psi=fixed` seeds 1-2, learned seed 2, bands K 8 + learned (speed-range only), and both
  potential forms on the L and A scenes (predictors `speed_pred/{L,A}_clearance.pt`, built in grid mode:
  env3 / env4 have a hollow wall group, so no sign flip; vs labels dex 0.25 / 0.27, median clearance
  error 0.0019 / 0.0021).

### S6b: `sf_psi=learned` on the L and A scenes (seed 0, standard 1000-case sets)

| scene | field | dex bins | dex low | SE(3) success | SE(3) converged | collision | chart success |
|---|---|---|---|---|---|---|---|
| L | baseline s0 / s1 / s2 | 0.633 (pooled) | 0.753 | 96.3 / 95.4 / 96.9 | 96.9 / 96.7 / 97.7 | 0.9 / 1.3 / 0.9 | 82.7 / 81.5 / 83.3 |
| L | `sf_psi=learned` s0 | 0.604 | 0.710 | 96.8 | 97.5 | 0.8 | 82.7 |
| A | baseline s0 / s1 | 0.675 (pooled, 3 seeds) | 0.806 | 94.3 / 94.6 | 97.7 / 98.2 | 3.4 / 3.8 | 81.0 / 81.0 |
| A | `sf_psi=learned` s0 | 0.657 | 0.778 | 95.0 | 97.0 | 2.0 | 81.2 |

The potential's effect scales with the baseline's collision rate: on L (baseline collisions ~1 %) it is
within the seed spread on every metric; on A (3.4-3.8 %) collisions drop to 2.0 % and SE(3) success rises
0.4-0.7 points, with the same 0.7-1.2 point convergence cost seen on T. The speed-range gain is the same
-0.02 to -0.03 dex bins as on T; the lowest decade stays at 22-39x. Predictors: `speed_pred/L_clearance.pt`,
`A_clearance.pt` (grid mode, 2048 shape points, d_bias 0.002).

## S7: seeds of the fixed potential, and both potentials on L / A (speed range; CPU evaluation)

GPU 2 (PCI 67:00.0) fell off the bus a second time (Xid 79, 01:55) while running `psil_s2` and
`bands8psil_s0`; `psil_s2` had already saved, `bands8psil_s0` stopped at epoch 3350 and is dropped (its
line is closed by S6). After the second reboot the card is gone and the machine has three GPUs (the old
GPU 3 is now index 2).

| group | dex bins | dex low (<0.05) | dex high | N_med | per-bin median ratio |
|---|---|---|---|---|---|
| T baseline (3 seeds) | 0.654 | 0.770 | 0.090 | 0.95 | 29.9 20.2 9.0 5.4 2.8 1.8 1.3 1.07 0.89 |
| T `sf_psi=fixed` (3 seeds) | 0.617 | 0.708 | 0.087 | 0.96 | 26.8 16.9 7.8 4.6 2.43 1.65 1.24 1.07 0.90 |
| T `sf_psi=learned` (3 seeds) | 0.609 | 0.692 | 0.091 | 0.99 | 27.1 17.0 7.3 4.4 2.32 1.58 1.20 1.03 0.88 |
| L baseline (3 seeds) | 0.633 | 0.753 | 0.070 | 0.97 | 28.1 19.0 8.2 5.2 2.9 1.8 1.34 1.09 0.93 |
| L `sf_psi=fixed` / `learned` | 0.604 / 0.604 | 0.704 / 0.710 | 0.072 | 1.00 | 24.7 17.0 7.1 4.5 2.5 1.6 1.2 1.04 0.92 |
| A baseline (3 seeds) | 0.675 | 0.806 | 0.087 | 0.97 | 38.8 18.9 10.2 5.9 3.0 1.9 1.37 1.09 0.90 |
| A `sf_psi=fixed` / `learned` | 0.669 / 0.657 | 0.788 / 0.778 | 0.086 / 0.091 | 0.97 / 1.00 | 39.8 17.8 9.3 5.7 2.9 1.8 1.33 1.07 0.91 |

Both potential forms give the same, seed-stable speed-range change on all three scenes: -0.01 to -0.05
dex bins, concentrated in the 0.005-0.2 band (per-bin ratios 10-20 % lower), nothing in the lowest decade
and nothing above 0.5. The predicted-speed range is not materially widened by any method in this series.

### S7b: planning of the fixed potential, all seeds and scenes (standard 1000-case sets, batch 125, seed 0)

| scene | field | SE(3) success | SE(3) converged | collision | chart success | median iters |
|---|---|---|---|---|---|---|
| T | baseline s0 / s1 / s2 | 91.9 / 90.3 / 90.1 | 95.5 / 94.2 / 95.4 | 4.0 / 4.3 / 5.6 | 78.8 / 77.9 / 78.7 | 59 |
| T | `sf_psi=fixed` s0 / s1 / s2 | **94.4 / 93.7 / 94.1** | 94.7 / 94.3 / 94.6 | **0.4 / 0.6 / 0.5** | **80.3 / 80.5 / 80.5** | 59-60 |
| T | `sf_psi=learned` s0 / s1 | 92.4 / 92.6 | 92.6 / 93.6 | 0.2 / 1.0 | 80.0 / 79.3 | 59-60 |
| L | baseline s0 / s1 / s2 | 96.3 / 95.4 / 96.9 | 96.9 / 96.7 / 97.7 | 0.9 / 1.3 / 0.9 | 82.7 / 81.5 / 83.3 | 55-56 |
| L | `sf_psi=fixed` / `learned` s0 | 96.1 / 96.8 | 96.8 / 97.5 | 0.7 / 0.8 | 82.7 / 82.7 | 55 |
| A | baseline s0 / s1 | 94.3 / 94.6 | 97.7 / 98.2 | 3.4 / 3.8 | 81.0 / 81.0 | 52-53 |
| A | `sf_psi=fixed` / `learned` s0 | **96.0** / 95.0 | 97.6 / 97.0 | **1.7** / 2.0 | **82.1** / 81.2 | 52 |

Final conclusion of the S-series. The fixed escape-time potential (T = D + (Psi_x + Psi_y)(1 - kappa),
Psi from the exact clearance predictor, two trained scalars) is a seed-stable improvement on every scene
where the baseline collides: T +2.2 to +4.0 points SE(3) success (collisions 4-6 % -> 0.4-0.6 %, chart
success +1.5 to +2.6), A +1.4 to +1.7 points (collisions 3.4-3.8 % -> 1.7 %), L unchanged (baseline
collisions already ~1 %). The learned form is slightly worse on success (it loses 1-2 points of
convergence). The pointwise predicted-speed range, the original target, moves by 0.01-0.05 dex only;
the planner gain comes from tau being raised near contact, not from a better 1/|grad T| at low speed.

# Targeted chart-vs-Lie benchmark (B-series, 2026-10-12/13; `models/graph_lie/scripts/bench/`, report `results/graph_lie/bench/REPORT.md`)

Question: the standard sets show the Lie-trained field 1-3 points behind and the Lie sampler equal; does the
gradient definition change what the field *predicts*? 18 checkpoints (comb/iso/aniso × chart/Lie × 3 seeds)
per scene, two MPPI samplers, 300-case targeted subsets (nearid, bothfar, startfar, goalfar, perp/par paired,
longtrans_near/far, tightstart, standard300; T also bigangle/rotheavy/transonly), 1044 evaluations with a
per-path audit, plus exact free-space ground truth (1000 saturated-clearance pairs per scene) and 200k
training pairs per scene. Everything paired by seed and by case.

## B1: predicted travel time differs by exactly the chart factor

τ_Lie-field / τ_chart-field on the same query (median; iso, other formulations within 0.02): nearid 0.99-1.00,
par 0.98-1.00, startfar/goalfar 0.97-1.01, standard300 0.97-0.98, bothfar 0.80-0.82, perp 0.72-0.75 in T/L/A.
perp vs par (same start, same geodesic angle, chart segment 1.30×): chart-trained τ_perp/τ_par 1.14/1.23/1.22,
Lie-trained 0.87/0.88/0.90 (T/L/A) — the two fields differ by 1.31-1.40, the chart factor.

## B2: exact truth (saturated-clearance pairs)

τ / T_exact by start |rotvec| bin (0-45 … 160-180°): Lie-trained iso flat (T 1.11 1.11 1.10 1.11 1.11;
L 1.02 0.99 0.99 0.98 0.98; A 1.01 1.02 1.03 1.02 1.02); chart-trained iso tracks the chart segment
(T 1.11→1.35, L 1.01→1.22, A 1.03→1.28; τ/T_chart constant at 0.93-1.13). Paired log(τ_Lie/τ_chart):
0, −4…−6, −10…−13, −16…−20, −17…−23 % in every scene/formulation. The Lie field is calibrated at every
angle; the chart field measures rotation-vector length and is 20-35 % too slow above 135°.

## B3: along executed paths the Lie field is over-optimistic near obstacles

Rotation steps at 160-180° on official-MPPI paths (iso): N_Lie chart-trained 1.04/1.04/1.08, Lie-trained
0.78/0.80/0.80; slope −Δτ/Δcost_Lie 0.77/0.93/0.95 vs 0.53/0.64/0.66 (T/L/A). Below 90° identical. On
bothfar τ_0/cost_Lie 0.61-0.85 (chart-trained) vs 0.48-0.64 (Lie-trained). Training endpoints agree:
paired N_Lie (Lie − chart) ≈ 0 below 45°, −0.07…−0.10 at 160-180°, −0.09…−0.15 for fast endpoints at
θ ≥ 150°; chart-trained N_Lie 1.02-1.08 / N_chart 0.94-0.99 there, Lie-trained N_Lie 0.92-1.00 / N_chart
0.86-0.94.

## B4: planning success

Only both-ends-far subsets separate the fields: bothfar/bigangle Δ(Lie − chart) −6…−11 points in T (sign test
p < 1e-3), −1…−4 in L/A; the loss is non-convergence (T bothfar 11-13 % vs 6-7 %) plus collisions. Single far
endpoint, perp vs par, long translations, tightstart, standard300: within ±3 (chart slightly ahead on
standard300 iso/aniso, Lie slightly ahead on near-identity translation sets). perp is not harder than par for
either field although the chart field believes it 12-29 % longer. The sampler dominates startfar: the official
chart MPPI leaves 11-22 % unconverged for both fields, the Lie body-twist MPPI lifts success 70-74→84-89 (T),
83→94-97 (L), 79-83→89-92 (A), identically for both fields; it halves but does not close the bothfar gap.

Reading: the gradient does define the predicted speed, and only the Lie-trained field is frame-consistent
(use it, or correct the chart field by up to the chart factor, wherever τ itself is consumed). For steering
MPPI the chart field is as good or better, because the Lie loss trades near-contact accuracy at large angles
for free-space calibration; the next lever for the Lie field is near-contact calibration at large |rotvec|,
not the gradient definition. Infrastructure notes: planning containers peak at 8-14 GB (Lie-repo clearance
evaluator) — at most 2 per 24 GB GPU; a full disk makes `docker run` fail silently; host GPU 3 dropped off
the bus during the run (2026-10-13 ~03:20) and the runner was restarted on GPUs 0-2.
