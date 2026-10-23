# Experiment log: double-cover (homotopy) head (`models/graph_homotopy`, 2026-10-17)

Goal (user, 2026-10-17): a new method `graph_homotopy`, the branch decoder reduced to its cone and its ridge:
no residual branches, one minimum, and the ridge reformulated on the double cover. The field is first a
function T~ on (R^3 x S^3) x (R^3 x S^3), deck-invariant by construction (flipping both quaternions is the same
pair of poses), and the time between poses is the minimum over the two lifts of the goal, i.e. over the two
homotopy classes of paths; planning is done on the cover, towards +q_goal or -q_goal. Budget: 8 hours on one
RTX 5070 (12 GB), docker image `ntfields:rtx5070`, every experiment concurrent.

Pipeline for every run: models/graph_lie trainer (original metric NTFields: 4000 epochs, AdamW wd 0.5, lr 5e-4,
batch 2000 x 5 per epoch, accept / reject, `exp(-tau/2)`, `1e-2 (sqrt(N) - 1)^2`), Lie gradient, isotropic speed
`S_d |c| = 1`; 5 trainings share the GPU (~23 min each). Evaluation: per-case forward MPPI of
`scripts/plan_eval_homotopy.py` (50 samples, horizon 5, step 0.015, 200 iterations, 1000 RRT-verified cases of the
standard T / L / A sets, batch 125, seed 0), SE(3) success primary; planners `base` (official planner on the base
field), `cover` (one plan on the sheet the field prefers at the start), `dual` (both sheets planned, the converged
one kept). Only the standard sets exist on this machine (the big-angle / rotation-heavy sets of the earlier
campaigns are not available here).

## H0: head, loss, tests, tooling

* `models/graph_method_triangle/homotopy_decoder.py` (arch `graph_homotopy`): T~ = T_free (1 + P) + C + delta on the
  cover; `value` = min over the two lifts (selection without autograd, selected sheet recomputed with gradients),
  `sheet_value(gx, gy, sigma)`, `sheet_values`, `value_cover(t, q, t, q)` (lifted inputs for the cover planner),
  `quat_exp_turns` (continuous lift of the unwrapped rotation-vector chart).
* `models/graph_homotopy/model_function_homotopy.py`: the Eikonal loss on the cover with the training pairs lifted
  (`lift` = min / random / both). `lift min` is bit-identical to the conformal branch head with K = 1 (same loss
  trace, 3.6940e-03 at epoch 100 on T seed 0 for both).
* `tests/test_graph_homotopy.py` (5 tests, float64, random parameters): deck invariance and sheet swap, base value =
  min of the sheets for any lifts, symmetry, exact zero, lower bound, conformal cone limit, lift exp / relative
  quaternion identities, finite double backward for every lift mode.
* Timing on the RTX 5070: 100 epochs 15.8 s alone, ~25 s with 3 runs, ~34 s with 5 runs (GPU at 100 %); a 1000-case
  dual evaluation ~3 min.

## H1: exploration on T, seed 0 (1000 cases; SE(3) success %, planners base / cover / dual / either)

| run | training lift | head | base | coll | nonconv | cover (sigma*) | dual | either | + sheet | - sheet |
|---|---|---|---|---|---|---|---|---|---|---|
| T_hmin_s0 | min (= conformal branch K=1) | 16/16 | 94.6 | 3.1 | 2.4 | 95.0 | **97.1** | 98.8 | 96.6 | 61.6 |
| T_hminP_s0 | min | + pair term, 32/32 | 94.8 | 1.9 | 3.6 | (pending) | | | | |
| T_hrandP_s0 | random 0.5 | + pair term, 32/32 | 93.5 | 4.0 | 2.6 | 93.7 | 95.1 | 97.7 | 94.9 | 59.2 |
| T_hrand_s0 | random 0.5 | 16/16 | 91.3 | 6.5 | 2.3 | 90.9 | 91.5 | 97.5 | 90.7 | 56.8 |
| T_hboth_s0 | both | 16/16 | 91.1 | 6.0 | 3.1 | (pending) | | | | |

* **Planning on the cover is the gain.** With the same field (T_hmin_s0), the official planner on the base field
  reaches 94.6; planning towards both lifted goals and keeping the converged plan (dual) reaches 97.1, and at least
  one of the two sheet plans succeeds on 98.8 % of the cases. By relative rotation angle (base -> dual): 0-60 deg
  95.1 -> 96.8, 60-120 98.5 -> 97.3, 120-150 93.5 -> 97.0, 150-180 90.0 -> 97.2: the base planner loses its cases
  near the ridge (large relative rotations, where the two homotopy classes tie), the cover planner never meets it.
  Of the 54 base failures (31 collisions, 24 non-convergences), the dual planner recovers 35.
* **The field's own choice of class (sigma*) is unreliable when it says "the long way".** sigma* = - on 15 % of the
  cases (34 % of those with 150-180 deg): there the - plan succeeds on 84 % but the + plan on 94.7 %; where sigma* =
  +, + succeeds on 96.9 % and - on 57.6 %. Hence cover (one plan on sigma*) 95.0 < + sheet alone 96.6 < dual 97.1.
* **Training the far sheet (lift random / both) costs base-field quality and does not make sigma* reliable.** The
  cover-lifted fields have more collisions (6.0-6.5 % vs 3.1 %), a worse speed calibration (dex 0.576-0.596 vs
  0.545; `results/graph_homotopy/diag/`), and their near-sheet Eikonal residual is higher (median 0.022-0.023 vs
  0.017) while the far sheet of the min-trained field is already reasonable (0.023): the architecture T_free + smooth
  terms keeps the untrained sheet sane. With the far sheet trained, sigma* = - cases still plan equally well on
  either sheet (hrand: 82.2 / 83.6). The remaining random-lift variants were dropped from the queue; the min lift
  (the base field, i.e. the branch-decoder loss) is kept for the main phase, with cover planning on top.
* The pair term with the min lift (hminP) has the lowest collision rate (1.9 %) at seed 0; decided with the dual
  planner and seeds below.

## H2: second exploration round (T, seed 0; base / dual / either SE(3) success %)

| run | lift | head | base | coll | nonconv | dual | either |
|---|---|---|---|---|---|---|---|
| T_hmin_s0 | min | 16/16 | 94.6 | 3.1 | 2.4 | **97.1** | 98.8 |
| T_hminP_s0 | min | pair 64, 32/32 | 94.8 | 1.9 | 3.6 | 96.9 | 98.1 |
| T_hrand02_s0 | random, far 0.2 | 16/16 | 93.7 | 3.0 | 3.4 | 96.4 | 98.4 |
| T_hrand02P_s0 | random, far 0.2 | pair 64, 32/32 | 92.7 | 3.3 | 4.1 | 95.3 | 97.7 |
| T_hrandP_s0 | random 0.5 | pair 64, 32/32 | 93.5 | 4.0 | 2.6 | 95.1 | 97.7 |
| T_hbothP_s0 | both | pair 64, 32/32 | 92.4 | 4.3 | 3.5 | 94.9 | 97.4 |
| T_hrand32_s0 | random 0.5 | 32/32 | 92.4 | 5.4 | 2.7 | 93.8 | 97.6 |
| T_hboth_s0 | both | 16/16 | 91.1 | 6.0 | 3.1 | 93.4 | 97.3 |
| T_hrandP128_s0 | random 0.5 | pair 128 (256 wide), 32/32 | 90.5 | 7.3 | 2.7 | 90.9 | 96.2 |
| T_hrand_s0 | random 0.5 | 16/16 | 91.3 | 6.5 | 2.3 | 91.5 | 97.5 |

* Monotone in the share of far-sheet training: min (94.6-94.8 base) > far 0.2 (92.7-93.7) > far 0.5 / both
  (90.5-93.5). The pair term P changes little at seed 0 (base +0.2, dual -0.2 with the min lift; it helps the
  random-lift variants by 1-2 points); the wide pair MLP (128 / 256) is worst.
* Cover planning (dual) adds +2.0 to +3.4 points over the base planner for every min / far-0.2 field; the dual
  planner recovers the ridge cases, not the collisions: `either` is 97.3-98.8 for every field.
* Selection rules between the two sheet plans (`scripts/select_study.py`, T_hmin_s0, planner-side signals only):
  converged-then-shorter-path 97.6, converged-then-fewer-iterations (dual) 97.2, converged-then-sigma* 96.6,
  always + 96.6, converged-then-larger-minimum-predicted-speed 95.8, sigma* 95.0, oracle 98.8. The field-side
  clearance estimate along a plan does not identify the colliding plan; the shorter-path rule is recorded alongside
  (`dual-short`) in every later evaluation.

## H3: third round, min lift: capacity, pair-term init, light far-sheet training (T, seed 0)

| run | head | base | coll | nonconv | dual | dual-short | either |
|---|---|---|---|---|---|---|---|
| T_hminPs_s0 | pair 64 (init 0.01), 32/32 | **96.2** | 1.4 | 2.5 | **97.9** | **98.1** | 98.9 |
| T_hmin32_s0 | 32/32 | 95.6 | 2.2 | 2.2 | 97.4 | 97.7 | 98.6 |
| T_hrand01_s0 | random, far 0.1, 16/16 | 95.0 | 2.2 | 2.9 | **98.0** | **98.6** | 98.9 |
| T_hminP_s0 | pair 64 (init 0.05), 32/32 | 94.8 | 1.9 | 3.6 | 96.9 | - | 98.1 |
| T_hmin_s0 | 16/16 | 94.6 | 3.1 | 2.4 | 97.1 | - | 98.8 |
| T_hminP16_s0 | pair 64 (init 0.05), 16/16 | 94.2 | 4.0 | 2.0 | 95.6 | 96.3 | 98.1 |
| T_hmin64_s0 | 64/64 | 92.9 | 4.4 | 2.8 | 95.3 | 95.5 | 97.8 |

* 32 even + 32 odd residual coordinates beat 16 + 16 (+1.0 base, +0.3 dual) and 64 + 64 (-1.7 / -0.8 vs 16): one
  sheet with K = 1 wants a little more than the K = 1 branch head's 16 + 16, not four times more.
* The invariant pair term P helps when it starts small (init 0.01: base 96.2, collisions 1.4 %, the best single run
  of the campaign) and not at init 0.05 (94.8 / 94.2): P multiplies T_free, so a large start inflates every time and
  the Eikonal loss spends the first epochs undoing it.
* Training the far sheet lightly (10 % of the pairs, `hrand01`) keeps the base field (95.0, collisions 2.2 %) and
  makes the other class usable: - sheet 62.4 % (hmin 61.6), dual 98.0, converged-then-shorter 98.6, within 0.3 of the
  oracle. At 20 % (hrand02: 93.7 / 96.4) and 50 % the base field already degrades (H1-H2).
* Speed calibration (dex, `results/graph_homotopy/diag/`): hminP 0.505, hminPs 0.513, hmin32 0.518 < hmin 0.545 <
  hminP16 0.548, hrand02 0.561, hmin64 0.562, hrand 0.596; the same ordering as the base-planner success, as in the
  cover campaign.
* Main phase (3 seeds x T, L, A): `hmin` (the reduced branch head, 16/16), `hminPs` (pair term, 32/32) and `hrand01`
  (10 % far-sheet lift); extra seeds 3-4 on T; T seed 0 of the combinations `hrand01Ps`, `hrand01_32`; the conformal
  branch head K = 4 (`bc4`, base planner only) retrained on this machine as the paired baseline.

## H4: training length (T, seed 0) and the first main-phase seeds

| run | head | epochs | final loss | base | coll | nonconv | dual | dual-short | either |
|---|---|---|---|---|---|---|---|---|---|
| T_hminL_s0 | 16/16 | 8000 | 1.49e-3 | **96.7** | 1.0 | 2.5 | **98.5** | **98.9** | 99.3 |
| T_hmin_s0 | 16/16 | 4000 | 1.68e-3 | 94.6 | 3.1 | 2.4 | 97.1 | - | 98.8 |
| T_hminPL_s0 | pair 64 (init 0.05), 32/32 | 8000 | 1.49e-3 | 95.8 | 1.0 | 3.5 | (pending) | | |
| T_hmin_s1 / s2 | 16/16 | 4000 | | 95.9 / 95.9 | 1.6 / 1.8 | 2.6 / 2.5 | 97.7 / 97.6 | 97.7 / 98.0 | 98.7 / 98.6 |

* The Eikonal loss is still falling at epoch 4000 (the original schedule); 8000 epochs take it from 1.68e-3 to
  1.49e-3 and the base-planner success from 94.6 to 96.7 (collisions 3.1 % -> 1.0 %), cover planning to 98.5 (the
  converged-then-shorter rule 98.9, oracle 99.3). The main phase of the plain head is therefore repeated at 8000
  epochs (`hminL`, 3 seeds x 3 scenes, ~50 min per run with 5 concurrent); `hrand01L` and `hminPsL` at seed 0.
* hmin at 4000 epochs, 3 seeds on T: base 95.5 +- 0.6 (the previous campaign's conformal branch K = 4: 95.4 +- 0.6
  on 3 RTX 3090 with the same code), dual 97.5 +- 0.3, converged-then-shorter 97.9 (2 seeds), oracle 98.7 +- 0.1.
  L seed 0: base 97.5, dual 98.7 (150-180 deg: 94.7 -> 98.9).

## H5: combinations (T, seed 0) and the 4000-epoch main phase of the plain head (3 seeds x T, L, A)

Combinations at seed 0 (base / dual / shorter / either): hrand01 + 32/32 (`hrand01_32`) 95.5 / 97.1 / 97.1 / 98.6,
hrand01 + pair term (`hrand01Ps`) 93.8 / 97.0 / 97.5 / 98.5; neither improves on hrand01 (95.0 / 98.0 / 98.6 / 98.9)
or hminPs (96.2 / 97.9 / 98.1 / 98.9). Every min / far-0.1 variant with 16-32 residual coordinates lands at 97-98
with the dual planner at seed 0; the differences are within the seed spread.

`hmin` (reduced branch head, 16/16, 4000 epochs), 3 seeds, SE(3) success %:

| scene | base | collision | non-conv | cover (sigma*) | dual | dual, shorter rule | either |
|---|---|---|---|---|---|---|---|
| T | 95.5 +- 0.6 | 2.2 | 2.5 | 95.5 | 97.5 +- 0.3 | 97.8 +- 0.2 | 98.7 +- 0.1 |
| L | 97.1 +- 0.3 | 1.3 | 1.6 | 97.2 | 98.4 +- 0.4 | 98.5 | 98.9 +- 0.2 |
| A | 97.4 +- 0.7 | 1.9 | 0.8 | 97.3 | 98.1 +- 0.7 | 98.5 | 99.5 +- 0.2 |

(previous campaign, conformal branch K = 4, 3 RTX 3090: T 95.4 +- 0.6, L 97.5 +- 0.4, A 97.9 +- 0.7 with the
base planner.) Paired per-case test on T (3000 cases): dual recovers 85 base failures and loses 25 (net +60,
exact McNemar p = 8e-9); the sigma* cover planner alone is a wash (+32 / -30). Trained-weight properties
(`results/graph_homotopy/diag/`): T(p, p) = 0 and asymmetry <= 1e-6 on every run, min T / T_free 1.04-1.14 (the
lower bound holds), the field prefers the long way on 2-7 % of random pairs, exactly one ridge on every rotation
slice of `ridge_slice.py`, 3-11 deg from the free-space half-turn position for the 8000-epoch field.

## H6: main phase at 8000 epochs (`hminL`: the plain head, 16/16, min lift)

T, 3 seeds (1000 cases each), SE(3) success %:

| planner | success | collision | non-conv. |
|---|---|---|---|
| base field, official planner | 96.8 +- 0.3 | 0.5 | 2.8 |
| cover, one plan (sigma*) | 97.0 +- 0.1 | | |
| cover, both classes (dual) | 98.7 +- 0.1 | | |
| cover, both classes, shorter-path rule | 98.8 +- 0.1 | | |
| either sheet (oracle) | 99.0 +- 0.2 | | |

Paired over the 3000 cases: dual recovers 66 base failures and loses 10 (net +56, exact McNemar p = 3e-11); the
shorter-path rule +67 / -6 (p = 4e-14); sigma* alone +16 / -10 (p = 0.33). By relative rotation (base -> dual):
0-60 deg 96.9 -> 97.7, 60-120 98.4 -> 99.2, 120-150 97.6 -> 98.7, 150-180 93.8 -> 98.7. The base planner's
remaining failures sit at the half turn; the cover planner removes them, and what is left (1.0-1.3 %) is shared
by both planners.

L and A (filled in as the seeds complete): L seeds 0-1 base 97.8 +- 0.1, dual 98.6 +- 0.1, either 99.3; A seed 0
base 98.4, dual 99.3, either 99.8.

Also launched: `T_hminXL_s0` (16000 epochs) to see whether the training-length gain continues, `T_hrand01L_s0`
(10 % far-sheet lift at 8000 epochs), `hminL` seeds 3-4 on T, `bc4` seeds 0-2 on T (the conformal branch head
K = 4 retrained here, base planner) for a same-machine paired comparison.

## H7: baseline on this machine, pair term at 3 seeds

* `bc4` (conformal branch decoder, K = 4, `train/train_graph_branch.py`, 4000 epochs), T seeds 0-2, base planner:
  94.8 / 94.2 / 95.1 = 94.7 +- 0.4 (collisions 2.6 %, non-convergence 3.0 %); the earlier campaign had 95.4 +- 0.6
  for the same head on other hardware. The reduced head `hmin` at the same schedule: 95.5 +- 0.6; at 8000 epochs
  96.8 +- 0.3; with cover planning 98.7 +- 0.1.
* `hminPs` (invariant pair term, init 0.01, 32/32), T seeds 0-2: base 95.4 +- 0.8, dual 97.5 +- 0.5, either 98.7
  (hmin: 95.5 +- 0.6 / 97.5 +- 0.3 / 98.7); L seed 0: 97.9 / 98.6 / 99.1 (hmin 97.5 / 98.7 / 99.1). A wash; its
  remaining L / A seeds were cancelled in favour of the 8000-epoch seeds 3-4 and the 16000-epoch run.

## H8: training length beyond 8000, far-sheet lift at 8000 (T, seed 0)

| run | epochs | final loss | base | coll | nonconv | dual | shorter | either |
|---|---|---|---|---|---|---|---|---|
| T_hmin_s0 | 4000 | 1.68e-3 | 94.6 | 3.1 | 2.4 | 97.1 | 97.6 | 98.8 |
| T_hminL_s0 | 8000 | 1.49e-3 | 96.7 | 1.0 | 2.5 | 98.5 | 98.9 | 99.3 |
| T_hminXL_s0 | 16000 | 1.50e-3 | 97.1 | 0.1 | 2.9 | 98.6 | 98.9 | 98.9 |
| T_hrand01L_s0 (10 % far sheet) | 8000 | 1.51e-3 | 96.4 | 0.6 | 3.3 | 98.6 | 98.6 | 98.9 |

The loss and the planning result are flat from 8000 epochs on; the light far-sheet lift at 8000 epochs matches the
plain head (its seed-0 advantage at 4000 epochs, 98.0 vs 97.1, does not survive the longer schedule). 8000 epochs of
the plain head is the recommended configuration.

## H9: final results (queue complete 09:06; seeds 3-4 of the 8000-epoch head on L and A added afterwards)

`results/graph_homotopy/summary/readme_table.md` (also in README.md), SE(3) success %, seed means +- population std:

| head | scene | seeds | base planner | collision | non-conv. | cover, one plan (sigma*) | cover, both classes | + shorter-path rule | either (oracle) |
|---|---|---|---|---|---|---|---|---|---|
| graph_homotopy, 8000 epochs | T | 5 | 97.0 +- 0.3 | 0.4 | 2.7 | 97.1 +- 0.2 | **98.9 +- 0.3** | 99.0 +- 0.2 | 99.1 +- 0.2 |
| graph_homotopy, 8000 epochs | L | 5 | 97.9 +- 0.2 | 0.7 | 1.4 | 97.8 +- 0.3 | **98.9 +- 0.2** | 98.9 +- 0.3 | 99.3 +- 0.1 |
| graph_homotopy, 8000 epochs | A | 5 | 98.4 +- 0.3 | 0.9 | 0.8 | 98.3 +- 0.5 | **98.9 +- 0.2** | 99.1 +- 0.2 | 99.6 +- 0.1 |
| graph_homotopy, 4000 epochs | T | 3 | 95.5 +- 0.6 | 2.2 | 2.5 | 95.5 +- 0.4 | 97.5 +- 0.3 | 97.8 +- 0.2 | 98.7 +- 0.1 |
| graph_homotopy, 4000 epochs | L | 3 | 97.1 +- 0.3 | 1.3 | 1.6 | 97.0 +- 0.4 | 98.4 +- 0.4 | 98.5 +- 0.3 | 98.9 +- 0.2 |
| graph_homotopy, 4000 epochs | A | 3 | 97.4 +- 0.7 | 1.9 | 0.8 | 97.3 +- 0.8 | 98.1 +- 0.7 | 98.5 +- 0.5 | 99.5 +- 0.2 |
| branch decoder K = 4 (retrained here) | T | 3 | 94.7 +- 0.4 | 2.6 | 3.0 | - | - | - | - |

T, 8000 epochs, 5 seeds, paired per case (5000): dual +109 / -12 (p = 1e-20), shorter-path rule +109 / -9 (p = 6e-23),
sigma* alone +22 / -15 (p = 0.32), oracle +112 / -5. By relative rotation (base -> dual): 0-60 deg 97.3 -> 98.3,
60-120 98.7 -> 99.4, 120-150 97.3 -> 98.6, 150-180 94.1 -> 99.0. L, 5 seeds: dual +62 / -13 (p = 8e-9), 150-180 deg
95.8 -> 98.9; A, 5 seeds: dual +57 / -29 (p = 0.003), shorter-path rule +59 / -22 (p = 5e-5), 150-180 deg 96.4 -> 99.0.

Campaign totals: 53 training runs (50 homotopy + 3 branch-decoder baselines; 21 of them 8000-16000 epochs), 103
per-case evaluations (1000 cases each), 5 trainings and 2 evaluations concurrent on the one GPU throughout
(03:16-10:31).
Report page: results/graph_homotopy/report/one-cone-one-ridge.html (`scripts/make_report_page.py`).

## H10: last slot, 32 + 32 residual coordinates at 8000 epochs (T, 3 seeds)

`hmin32L`: base 97.2 / 97.4 / 96.5 = 97.0 +- 0.4 (collisions 0.4 %), dual 99.2 / 98.9 / 98.8 = 99.0 +- 0.2, shorter-path
rule 99.0 +- 0.1, oracle 99.2 +- 0.2; `hminL` (16 + 16, 5 seeds): 97.0 +- 0.3 / 98.9 +- 0.3 / 99.0 +- 0.2 / 99.1 +- 0.2. The
seed-0 advantage of 32 + 32 at 4000 epochs (+1.0 base) does not survive the longer schedule either: the plain 16 + 16
head at 8000 epochs with the dual cover planner is the campaign's recommendation, and every head variant tried
(pair term, 32 / 64 coordinates, far-sheet lifts, 16000 epochs) lands within its seed spread.

# Second request (2026-10-17, 12:10): expanded test sets, the original metric NTFields baseline, walls with holes

## H11: targeted test sets and the metric baseline (design)

The user asked for more test data on the same shapes and environments (long paths, big rotations, less endpoint
offset), the 8000-epoch homotopy head against the original metric NTFields, and (second message) environments
with a dividing wall whose hole shrinks from environment to environment, both models trained and tested on them.

**Baseline.** The original metric NTFields = Lie-Shape-NTFields' `metric_baseline`, taken from the self-contained
bundle `/home/jeffrey/export_metric` (its `models/metric` is byte-identical to this repository's `models/metric` up to
the `--alpha 1.0` speed stretch that the bundle needs on these datasets; the 2026-10-12 cusp study and the stress
bench compared against the same code). No checkpoint of it exists on this machine, so it is retrained here with
its own trainer (`train/train_metric_baseline.py` wraps it with a seed and a run name: AdamW wd 0.5, lr 5e-4 in the
loop, 5 x 2000 pairs per epoch, accept / reject rollback, 128 random Fourier features of the chart) on the same
800k-pair training arrays as the homotopy runs, for 8000 epochs with the epoch-4000 checkpoint kept, 3 seeds per
scene. It needs 3.0 GB of GPU memory per run (the homotopy trainer 1.9 GB), so at most three train concurrently.
Scored with the same per-case evaluator (`plan_eval_homotopy.py --kind metric`, same MPPI, same collision test,
SE(3) criterion); the loader was checked on an intermediate checkpoint.

**Targeted sets** (`scripts/make_test_sets.py`, 1000 cases each, written to `testing_data/3dshape/<scene>_<variant>`):
same placement distribution as the standard sets (independent uniform placements, rotation = uniform axis x angle
uniform in [0, pi]), the variant's constraint, exact endpoint checks against three independently drawn 50k-point
clouds, and an RRT-Connect in SE(3) (this file's own: OMPL is not installed here; step 0.05, resolution 0.005 in
sqrt(|dp|^2 + (theta / 2 pi)^2), motion check with a 0.003 margin on a trilinear signed-distance grid of the shape,
the returned path re-walked with the exact test). Variants: long (|dp| >= 0.6; standard sets: 15-19 % of the cases),
bigrot (relative rotation >= 150 deg; 25 %), tight (both endpoint clearances in [0.005, 0.02], i.e. inside the slow
band and below the standard sets' 0.02 offset), hard (clearances in [0.005, 0.02], |dp| >= 0.5, rotation >= 135 deg).
The first set (T long): 1000 pairs in 139 s, every formed pair RRT-solvable, |dp| median 0.64, the straight
interpolation collides for 49 % of its poses on average.

**Comparison.** The five 8000-epoch homotopy runs per scene (`hminL` seeds 0-4; base planner and the dual cover
planner) and the three baseline seeds (8000 epochs; the 4000-epoch checkpoint on the standard sets and, with lower
priority, on the targeted sets) on the 12 targeted sets; the baseline also on the three standard sets so the
standard-set comparison uses one evaluator. Seed-matched paired McNemar tests (homotopy seed s vs baseline seed s),
success by rotation / distance / clearance bins. Queue: `scripts/run_expand_queue.py` (GPU-memory-aware; results in
`results/graph_homotopy/plan_expand`), tables: `scripts/aggregate_expand.py` ->
`results/graph_homotopy/summary/expand_summary.md`.

## H12: wall with a hole (design)

`scripts/make_wall_envs.py`: the room of env1 (its wall group, so the normalized frame is that of the standard
cells: env_scale 210.25, the shapes keep their size) divided by a slab of thickness 8 units perpendicular to z at the
room's z centre, with a centred square hole of side 90, 75 or 66 units (`datasets/3dshape/data/hole{90,75,66}.obj`; the
slab is four non-overlapping boxes). The three shapes are flat plates 60 x 10 x 60 (T), 35 x 10 x 60 (L),
54 x 10 x 60 (A), so a square hole below 60 is impassable for all of them; face-on (plate parallel to the slab) the
clearance in the hole is 15 / 7.5 / 3 units = 0.071 / 0.036 / 0.014 normalized (the training margin is 0.05, the
standard endpoint offset 0.02). Cells `<shape>_hole<side>` (9), each preprocessed like the standard cells (Lie repo
`preprocess_obj.py`, 800k correlated pairs, offset 0.001, batch 500 to fit beside the trainings). Test sets
`<cell>_through` (1000 cases): start and goal on opposite sides of the slab, >= 0.05 from it, endpoint offset
0.02, verified by RRT to an aligned approach pose in front of the hole (the plate parallel to the slab; the best of
eight axis-aligned orientations by exact clearance), a straight translation through the hole at resolution 0.0025
with the exact test, and RRT to the goal. Both models are trained on every cell (8000 epochs; the baseline's
epoch-4000 checkpoint kept), seed 0 for all nine cells first, seed 1 afterwards, and scored on the through sets
(homotopy: base and dual planners; baseline: base). A success is a plan that reaches the goal on the other side of
the slab without any waypoint in collision, i.e. a passage through the hole.

## H11 results (targeted track complete at 17:35; `results/graph_homotopy/summary/expand_summary.md`)

Generation: 12 targeted sets in 52-142 s each (every formed pair RRT-solvable; e.g. T long: |dp| median 0.64, the
straight interpolation collides for 49 % of its poses on average; T bigrot: relative rotation median 163 deg). The
baseline trainings took 32 min each (three concurrent) and 3.0 GB; the plot files its trainer writes every 100
epochs were removed afterwards (100 MB per run).

SE(3) success %, seed means +- std (homotopy 5 seeds, NTFields 3 seeds; dual = cover planner, both classes):

| scene | set | homotopy base | homotopy dual | NTFields 8000 | NTFields 4000 | collisions dual / NTFields | non-conv. dual / NTFields |
|---|---|---|---|---|---|---|---|
| T | standard | 97.0 +- 0.3 | **98.9 +- 0.3** | 82.4 +- 2.1 | 81.2 +- 0.5 | 0.3 / 14.2 | 0.8 / 12.5 |
| T | long | 95.5 +- 0.2 | **97.9 +- 0.3** | 66.4 +- 2.9 | | 0.5 / 25.4 | 1.6 / 26.1 |
| T | bigrot | 95.8 +- 0.5 | **99.0 +- 0.2** | 80.4 +- 2.8 | | 0.1 / 15.4 | 0.9 / 10.8 |
| T | tight | 94.1 +- 0.8 | **95.8 +- 0.8** | 67.4 +- 1.2 | | 2.5 / 28.4 | 2.0 / 16.7 |
| T | hard | 92.9 +- 0.5 | **96.7 +- 0.5** | 54.3 +- 2.0 | | 1.8 / 37.5 | 1.7 / 27.3 |
| L | standard | 97.9 +- 0.2 | **98.9 +- 0.2** | 90.4 +- 1.1 | 89.6 +- 1.2 | 0.7 / 5.4 | 0.4 / 7.6 |
| L | long | 98.0 +- 0.2 | **99.2 +- 0.3** | 80.6 +- 0.9 | | 0.5 / 7.8 | 0.3 / 17.8 |
| L | bigrot | 98.3 +- 0.3 | **99.6 +- 0.2** | 91.7 +- 1.1 | | 0.3 / 4.8 | 0.2 / 6.4 |
| L | tight | 94.7 +- 0.6 | **97.0 +- 0.3** | 74.2 +- 1.3 | | 1.8 / 16.9 | 1.3 / 18.4 |
| L | hard | 96.6 +- 0.2 | **97.8 +- 0.5** | 65.0 +- 1.3 | | 1.7 / 20.0 | 0.5 / 26.1 |
| A | standard | 98.4 +- 0.3 | **98.9 +- 0.2** | 88.2 +- 0.9 | 87.1 +- 0.8 | 1.0 / 6.8 | 0.0 / 8.6 |
| A | long | 97.6 +- 0.3 | **98.4 +- 0.4** | 72.7 +- 0.9 | | 1.5 / 12.9 | 0.1 / 23.0 |
| A | bigrot | 98.1 +- 0.2 | **98.7 +- 0.1** | 89.0 +- 1.4 | | 1.3 / 6.4 | 0.0 / 7.3 |
| A | tight | 95.3 +- 0.5 | **96.2 +- 0.4** | 70.0 +- 0.7 | | 3.6 / 21.8 | 0.2 / 16.2 |
| A | hard | 94.3 +- 0.6 | **96.5 +- 0.5** | 60.6 +- 2.2 | | 3.5 / 25.5 | 0.0 / 25.0 |

The NTFields number on the standard T set (82.4 SE(3)) is in line with the Lie repo's own metric_baseline seeds
(84 % chart success on the first 200 cases, cusp study 2026-10-12) once the SE(3) criterion is used; this
machine's run is a faithful reproduction, not a weakened baseline. Seed-matched paired tests (3000 pairs per set,
homotopy dual vs NTFields 8000): T +509/-22 (standard), +967/-25 (long), +571/-16 (bigrot), +882/-38 (tight),
+1285/-21 (hard); L +264/-12, +566/-7, +240/-5, +722/-30, +997/-8; A +333/-8, +791/-15, +307/-15, +822/-30,
+1095/-23; all p < 1e-60. Pooled over the targeted sets: distance > 0.7 succeeds 94.8-97.6 % (homotopy dual) vs
42-51 % (NTFields); endpoint clearance < 0.01: 95.8-97.2 vs 57-67; rotation 150-180 deg: 97.5-98.8 vs 66-77;
rotation 0-60 deg: 94.3-98.6 vs 78-92. The base planner alone wins 226-1226 cases per set against 15-75.

Findings: (1) the baseline's deficit grows with difficulty (8-16 points on the standard sets, 20-30 on long /
tight, 30-42 on hard), both by collisions (15-38 % on the targeted sets) and by stalls (11-27 %); the homotopy head
loses 2-3 points from standard to hard; (2) training the baseline twice as long adds about one point; (3) the cover
planner's gain over the base planner persists on the targeted sets (T hard 92.9 -> 96.7, T bigrot 95.8 -> 99.0) and
stays within 0.2-2 points of the two-sheet oracle.

## H12 results (queue complete 21:03; `results/graph_homotopy/summary/expand_summary.md`, report `results/graph_homotopy/report/expanded-tests.html`)

Hole cells: 200k pairs each with the pre-filtered preprocessing (`scripts/preprocess_hole.py`, 51 min per cell at
50-110 pairs/s on the shared GPU; the Lie preprocessor alone ran at 15-30 pairs/s and 7-9 GB because 97 % of the
drawn placements collide with the room or the slab). Pre-filter audit per cell: 8-14 of 300 rejected placements
are called free by the original point-sampled test (grazing contacts). Through sets: 1000 pairs each in 27-42 s,
aligned passage clearance 0.030 (capped) / 0.030 / 0.0143 for holes 90 / 75 / 66.

SE(3) success % on the through sets (homotopy: dual planner, base planner in parentheses; T cells 2 seeds, L and A
1 seed; NTFields 8000 epochs, epoch-4000 checkpoint in parentheses):

| hole (clearance) | T homotopy | T NTFields | L homotopy | L NTFields | A homotopy | A NTFields |
|---|---|---|---|---|---|---|
| 90 (0.071) | **98.0 +- 0.2** (95.5) | 70.0 +- 2.3 (68.8) | **98.5** (96.2) | 75.6 (70.5) | **99.5** (97.1) | 74.6 (75.9) |
| 75 (0.036) | **96.4 +- 1.0** (93.2) | 59.3 +- 3.0 (56.7) | **95.7** (91.8) | 62.2 (61.7) | **96.8** (94.0) | 65.7 (61.5) |
| 66 (0.014) | **85.2 +- 1.9** (79.8) | 52.4 +- 0.4 (47.8) | **93.5** (86.6) | 63.6 (55.7) | **81.8** (75.5) | 42.8 (43.7) |

Collisions (homotopy dual / NTFields 8k): hole 90 0.0-0.1 / 7.7-10.8; hole 75 0.5-1.0 / 19.3-30.7; hole 66
2.0-16.4 / 24.6-52.0. Non-convergence: 0.5-1.9 / 24.0-29.4; 2.4-3.3 / 32.0-36.0; 2.0-4.6 / 32.4-37.6. Oracle over
the two sheets at hole 66: 89.5 / 94.8 / 88.1. Seed-matched paired tests (homotopy dual vs NTFields 8k): hole 90
+578/-17 (T, 2000 pairs), +236/-7 (L), +252/-3 (A); hole 75 +766/-24, +352/-17, +327/-16; hole 66 +772/-116,
+330/-31, +455/-65; all p < 1e-60.

Failure modes (per-case records): NTFields' collisions penetrate the slab by 0.018-0.023 median (the slab is
0.038 thick: the plan goes through the wall) and its stalls end at 0.8-0.99 of the start-goal distance (it never
leaves the start side); the homotopy head's collisions are grazes (median penetration 0.0001-0.004) and its stalls
end at 0.3-0.45 (in front of the hole). The tightest hole, whose clearance is below the standard 0.02 endpoint
offset, costs the homotopy head 5-15 points, mostly grazes; its successful paths keep a median minimum clearance of
0.010-0.013 there (0.018-0.023 at the wider holes).

Campaign totals (second request): 21 baseline trainings (9 standard, 12 hole cells) and 12 homotopy wall
trainings, 258 per-case evaluations (1000 cases each), 12 targeted and 9 through-the-hole sets (24270 formed pairs,
15 unsolved within the RRT budget, none failed the exact re-walk), 9 hole cells; queue 12:13-21:03 with 4-7
containers on the GPU throughout, no failed job.

## H13: new shapes x env4-9, T / L / A x env1 / env3 (third request, design)

Machine: three RTX 3090 (24 GB), 36 cores, image `pytorchserver:latest`; the Lie repo at
`/media/jeffrey/Lie-Shape-NTFields` (/lieshape), the metric bundle rebuilt at `/media/4tbssd/jeffrey/export_metric`
(Lie `models/metric_baseline` as `models/metric`, alpha 1.0; only import paths renamed). No checkpoint of H1-H12 is on
this machine, so every model is trained here.

Data: the five shapes of `dataprocessing/make_shapes3d.py` on env4..env9 (30 training cells, 800k pairs each, and the
1000-pair standard test sets) brought over from G-Shape-NTFields-old into the new layout (`datasets/3dshape/data/<cell>`,
meshes in `datasets/3dshape/obj`, `meta.json` paths rewritten), with the newer dataprocessing scripts of that repository
(env9, the planar sofa pipeline) ported to the new paths. The T / L / A grid on env1 and env3: Tshape3d_env1 and
Lshape3d_env3 existed; Lshape3d_env1, Ashape3d_env1, Tshape3d_env3, Ashape3d_env3 are preprocessed here
(`preprocess_sharded.py`, 4 x 200k pairs, `preprocess_obj_fast.py`, offset 0.001); test sets: the Lie repo's standard
sets and the targeted variants long / bigrot / tight / hard (H11; generated here for the four new cells).

Runs: per cell and seed (0, 1, 2) the homotopy head (`--lift min --lift-far 0.5`, 8000 epochs) and the original metric
NTFields (8000 epochs, epoch-4000 checkpoint kept); scored with `plan_eval_homotopy.py` on every set of the cell (homotopy
base and dual planners, NTFields 8000 and 4000 epochs). 216 trainings, 720 evaluations. Queue
`scripts/run_shapes_queue.py` (one GPU per job, memory-aware, seed tiers in order: the complete seed-0 comparison
first); tables `scripts/aggregate_shapes.py` -> `results/graph_homotopy/summary/shapes_summary.md`. The old metric runs of
G-Shape-NTFields-old (`results/shapes3d`: 4000 epochs, seed 1, Lie `analysis/plan_eval.py`, chart criterion) are quoted
as a reference for the reproduction of the baseline.

Throughput on this machine: 0.37 s / epoch for one homotopy run alone, 0.57 with six on a GPU, about 0.95 with eight
trainings per GPU plus preprocessing (8000 epochs in about 2.1 h); NTFields 3.1 GB, homotopy 0.7 GB. Preprocessing at
batch 1000 swung between 3 and 14 GB per shard (caching allocator) and pushed two trainings into an OOM at the start;
shards were restarted at batch 500 with expandable segments.

## H13 results (queue complete 2026-10-18 16:57; `results/graph_homotopy/summary/shapes_summary.md`)

216 trainings and 720 evaluations, none failed. One trainer change during the run: `loop3d_env6_hminL_s0` hit a NaN loss
at epoch 72 in every attempt and the trainer aborted; a non-finite epoch is now rolled back and repeated like any
rejected one (abort after 20 in a row), as the original NTFields trainer does. The run recovered (NaN repeats at 72 and
152) and trained to 8000. The evaluator's chart and SE(3) success coincide on these runs.

SE(3) success %, mean over 3 seeds, homotopy dual / NTFields 8000:

| env | tripod | crank | loop | twistI | chair | mean |
|---|---|---|---|---|---|---|
| env4 open | 98.1 / 88.7 | 98.6 / 89.2 | 96.3 / 85.0 | 98.9 / 88.5 | 98.2 / 88.8 | 98.0 / 88.0 |
| env8 wall, 70 | 98.4 / 87.6 | 98.9 / 88.9 | 89.3 / 81.0 | 99.4 / 90.6 | 97.5 / 88.5 | 96.7 / 87.3 |
| env7 wall, 60 | 82.2 / 72.3 | 88.1 / 75.4 | 64.0 / 53.3 | 92.8 / 85.8 | 86.9 / 71.1 | 82.8 / 71.6 |
| env5 wall, 50 | 53.9 / 51.5 | 63.5 / 54.1 | 48.3 / 47.6 | 57.1 / 53.2 | 55.3 / 50.2 | 55.6 / 51.3 |
| env9 2 walls, 70 | 94.5 / 67.5 | 96.0 / 68.4 | 88.7 / 62.7 | 95.0 / 76.1 | 96.0 / 67.4 | 94.1 / 68.4 |
| env6 2 walls, 50 | 28.1 / 25.4 | 41.4 / 28.0 | 25.5 / 24.8 | 37.2 / 28.0 | 36.0 / 27.7 | 33.6 / 26.8 |

- The dual planner is ahead in all 30 shape cells; every seed-matched McNemar test p < 1e-4 (smallest margins loop3d
  env5 / env6, +24 / -3 and +20 / -0).
- The gap is in the window crossings: one partition crossed, dual / NTFields 93.5 / 76.7 (env8), 65.9 / 45.5 (env7),
  11.0 / 4.2 (env5); env9 two partitions 87.2 / 32.3. Same-side cases are 97-100 for both. At 50-unit windows (env5,
  env6) both mostly fail to cross (env6 two partitions 1.9 / 0.0); the homotopy head nearly always converges there and fails
  on collisions (non-convergence 1-42 %), NTFields also fails to converge in 55-68 % of env6 cases.
- The NTFields 4000-epoch numbers reproduce the old G-Shape-NTFields-old runs within a few points on most cells
  (larger differences: env9 loop / chair 58.8 / 64.9 vs 68.7 / 76.1).
- T / L / A, mean of the 6 scenes, dual / NTFields 8000: std 99.0 / 85.0, long 98.8 / 71.7, bigrot 99.3 / 84.9,
  tight 96.8 / 68.5, hard 97.1 / 57.3. The homotopy head stays at 95-99.7 on every targeted set; NTFields loses 13-28
  points on long / tight / hard.
- Path length (median translation ratio to straight line, successes): 1.16-1.36 dual vs 1.23-1.44 NTFields.

## H14: Gibson scene eastville, cube and small rectangle (2026-10-18; `shapes_summary.md` section 4)

Cells cube_eastville and rect_small_eastville as preprocessed in G-Shape-NTFields-old (800k pairs, standard 1000-case
test sets), seeds 0-5; queue `run_shapes_queue.py --grid gibson`. Homotopy head as H13 (8000 epochs, `--lift min`).
Two NTFields configurations, because the old Gibson runs (results/gibson_metric) did not use the bundle: they trained
G-Shape-NTFields-old `models/metric` with the TD and normal-alignment terms on (1e-3 each, alpha 1.0, 5000 epochs),
while the bundle (Lie metric_baseline, used for every other cell here and for the old shapes runs) has both off.
`train_metric_baseline.py` now takes `--td-weight` / `--normal-weight` (default 0). The evaluators agree: the old
checkpoints score the same here as in the old evaluator (66.0 / 87.1, 6 seeds), and the bundle checkpoints the same
in the old evaluator as here (seed 1: cube 54.0 in both, rect_small 74.1 in both).

SE(3) success %, 6 seeds:

| cell | homotopy dual | NTF+TD 8000 | NTF+TD 5000 | old NTF+TD 5000 | NTF (bundle) 8000 |
|---|---|---|---|---|---|
| cube_eastville | **84.6 +- 1.9** | 69.8 +- 3.3 | 67.2 | 66.0 | 54.1 |
| rect_small_eastville | 74.0 +- 1.3 | **88.1 +- 1.7** | 87.3 | 87.1 | 75.9 |

- Cube: the homotopy head is ahead of both (paired vs NTF+TD +1164 / -275, p 3e-130). Its failures are collisions
  (10.7 %) and non-convergence (6.1 %); NTF+TD collides in 29.9 %.
- Small rectangle: NTF+TD is ahead (paired +320 / -1168, p 2e-113), the bundle NTFields about level with the homotopy
  head (+642 / -757, p 0.002). The homotopy head collides in 24.0 % against 9.8 % for NTF+TD: the first cell of these
  campaigns where it is behind.
- The TD and normal terms are worth +12 to +16 points for NTFields on this scene. The shapes and T / L / A comparisons
  (H13) use the bundle, as the old shapes runs did; NTF+TD was not run there.

## H15: the full metric NTFields (TD + normal terms) on the shapes and T / L / A cells (queue complete 2026-10-19 07:21; `shapes_summary.md` section 5)

The H14 configuration (TD and normal-alignment terms 1e-3, 8000 epochs, epoch-4000 checkpoint) on all 36 cells of H13,
seeds 0-2, GPUs 0 and 1 only: 108 trainings, 360 evaluations, none failed (about 11 h, 14 trainings at a time).

| group | cells | homotopy dual | NTF+TD 8k | NTF 8k | dual ahead / behind (p < 0.01) |
|---|---|---|---|---|---|
| shapes, all envs | 30 | 76.8 | 68.1 | 65.6 | 30 / 0 |
| shapes env4 / env8 / env7 | 5 each | 98.0 / 96.7 / 82.8 | 90.2 / 89.7 / 73.0 | 88.0 / 87.3 / 71.6 | 15 / 0 |
| shapes env5 / env9 / env6 | 5 each | 55.6 / 94.1 / 33.6 | 51.5 / 76.5 / 27.7 | 51.3 / 68.4 / 26.8 | 15 / 0 |
| T / L / A std / long / bigrot | 6 each | 99.0 / 98.8 / 99.3 | 86.7 / 75.0 / 86.9 | 85.0 / 71.7 / 84.9 | 18 / 0 |
| T / L / A tight / hard | 6 each | 96.8 / 97.1 | 72.1 / 61.6 | 68.5 / 57.3 | 12 / 0 |

The terms add 2-4 points on most groups and 8 on env9 (smaller than the 12-16 on eastville); the homotopy head stays
ahead on all 30 shape cells and all 30 T / L / A sets (smallest margins loop3d env5 / env6: +18 / -3, +10 / -0). With
H14 the only cell where it is behind remains rect_small_eastville.

## H16. Lifted encoder: the homotopy head with its encoder on the double cover (2026-10-19/20)

Arch `graph_homotopy_lift` (homotopy_decoder.HomotopyLiftNN): the encoder reads the lifted pose (t, q) (7-d Fourier
features: the translation columns of the 12-d matrix, quaternion columns truncated normal std 0.2) and is run on both lifts
(t, q), (t, -q); e_even = (f(q) + f(-q)) / 2 feeds E, rho, lam, e_odd = (f(q) - f(-q)) / 2 feeds Q = W e_odd (16 x 263).
Deck invariance, symmetry, T(x, x) = 0, the conformal cone and the Lie gradients are tested (tests/test_graph_homotopy.py).
Everything else as hminL (--lift min, 8000 epochs); same 36 cells x 3 seeds and Gibson x 6 seeds (run code hlift,
run_shapes_queue.py --grid all). Summary section 6 of results/graph_homotopy/summary/shapes_summary.md.

| group | sets | lifted dual | homotopy dual | NTF+TD 8k | lifted vs homotopy ahead / behind (p < 0.01) |
|---|---|---|---|---|---|
| shapes + T / L / A | 60 | 87.9 | 87.5 | 72.3 | 9 / 4 |
| new shapes | 30 | 77.9 | 76.8 | 68.1 | 9 / 1 |
| shapes env9 | 5 | 95.7 | 94.1 | 76.5 | 3 / 0 |
| T / L / A | 30 | 98.0 | 98.2 | 76.4 | 0 / 3 |
| cube_eastville | 1 | 85.4 | 84.6 | 69.8 | 0 / 0 |
| rect_small_eastville | 1 | 78.6 | 74.0 | 88.1 | 1 / 0 |

The lifted encoder helps where obstacles separate the classes (walled rooms, Gibson: fewer collisions, rect_small 24.0 ->
17.4 %) and is level to slightly behind on the T / L / A sets, where both heads are near 98 %. It stays ahead of the full
NTFields on all 60 shape / T / L / A sets; rect_small_eastville is still behind the full NTFields (78.6 vs 88.1).

## H17. Camera + object: a bird followed by a camera that must keep it in view (2026-10-21)

Application test of the SE(3) planners (user: "sell this idea"). Bodies (`dataprocessing/make_camera3d.py`): `bird3d` = a
polycube glider (fuselage, head, wings, tail plane, fin; 48 long, 48 span, 18 tall) plus a 1-unit cube at the camera point
50 units behind it on its axis — the *rigid follower*, the camera is attached but has no view volume; `birdcam3d` = the same
bird, a 2-unit plate covering its silhouette, and a rectangle-based pyramid whose tip is the camera (field of view
51° x 25°, base 48 x 22) — one rigid body in SE(3), so a collision-free path means the bird flies, the camera flies and
nothing enters the line of sight. Both bodies share the bounding-box centre, so one test set scores both. Scenes in the
env3 room shell (env_scale 220.457): `forest` (four trees 2 x 2: 12-unit trunks to y = 96, 30-unit crowns at y 90..120) and
`city` (six buildings 45..100 high in two rows). Free random placements of the camera-aware body: 3.5 % forest, 4.5 % city
(env9: 4.3 %). Data: 400k narrow-band pairs per cell (4 shards x 100k; `preprocess_obj_fast.py` gained an exact
signed-distance-grid gate for the band test and a tighter exact env cull in evaluate_placements — selftest: identical pairs,
values within 1e-5; ~40 pairs/s per GPU instead of ~4). Models hlift / hminL / mbtd x seeds 0-2, 8000 epochs, same recipes
as H15/H16 (`run_camera_queue.py`, 98 jobs, 7.4 h on 4 GPUs). Test sets `testing_data/3dshape/birdcam3d_{forest,city}`:
1000 RRT-verified pairs each (`make_test_sets.py --variant std`, offset 0.02). Scoring `plan_eval_homotopy.py --camera
birdcam3d_camera.json`: besides the SE(3) collision test, at every waypoint the camera ray-casts (Moller-Trumbore, torch) to
its 256 visible bird surface samples against the scene mesh; *follow success* = converged, bird collision-free, >= 99 % of
the samples seen at every waypoint. Summary `results/graph_homotopy/summary/camera_summary.md`; demo dashboard
https://claude.ai/artifact/7ufo8euaZkFA85pdV5E4uZ (page source `models/graph_homotopy/report/camera_dash/`, data export
`aggregate_camera.py`, `export_camera_dash.py`, `export_camera_showcase.py`).

Percent of 1000 pairs, mean of 3 seeds. "body" = trained on birdcam3d (the planner knows the view pyramid), "follower" =
trained on bird3d (plans the bird with the camera as a point), both scored on the camera-aware body's pairs.

| scene | model (planner) | body SE(3) | body follow | follower SE(3) | follower follow | follower view blocked |
|---|---|---|---|---|---|---|
| forest | lifted encoder (dual) | 95.8 | 94.4 | 77.6 | 79.7 | 19.3 |
| forest | homotopy head (dual) | 95.9 | 95.1 | 78.5 | 79.8 | 19.3 |
| forest | full NTFields (base) | 86.3 | 85.7 | 67.3 | 68.3 | 29.1 |
| city | lifted encoder (dual) | 98.4 | 97.6 | 80.7 | 86.4 | 12.7 |
| city | homotopy head (dual) | 98.2 | 97.9 | 80.4 | 86.5 | 12.8 |
| city | full NTFields (base) | 86.8 | 86.3 | 74.3 | 77.5 | 17.3 |

Paired (seed-matched, exact McNemar): body vs follower on follow success +476 / -36 (forest, lifted), +357 / -21 (city,
lifted), +496 / -38 and +366 / -25 for the homotopy head, all p < 1e-78 — the follower's losses are view blocks
(12-19 %), not bird collisions (1-2 %), so planning the view pyramid as part of the shape is what keeps the bird in frame.
Lifted vs homotopy: no difference on either scene (p 0.13-0.93); both beat the full NTFields on the body by 9-11 points
(+307..+379 / -32..-56, p < 1e-47). The base planner of the homotopy heads is within 1.5 points of dual here. Caveats:
the camera is attached rigidly (one body), so the test does not optimise a separate camera trajectory; 400k pairs per cell
against 800k in the shape grids.

H17 viewers (2026-10-21). Dashboard: the showcase file is now rows ({part, data}; the Dashboard type reads JSON only as
rows, which is why the first publish showed no figures), and each scene card has a Scene view / Camera view toggle;
the camera view renders six waypoints per plan from the camera (z-buffer, the image is the pyramid's 51 x 25 degree
cross-section; obstacles inside the pyramid in front of the bird tinted red, everything beyond the bird faded), with
the first view break, the worst view and the first bird collision swapped in. Interactive viewer:
`dataprocessing/camera3d_viser.py` (port 8099, container cam_viser): scene / planner / body / seed, outcome filters
(incl. "camera-aware succeeds, follower fails"), bird and camera paths (camera path red where the view breaks), the body
flying along the path, a Camera POV toggle (browser view at the camera, everything outside the pyramid masked) and a
server-rendered camera image in the panel (`--render-test DIR` writes test images).

## H18. Drone + chase camera: five scenes of increasing difficulty, six camera set-ups (2026-10-22, in progress)

Follow-up of H17 (user: more realistic -- a smaller aircraft, the camera further away, taller and more detailed city,
trees with overhanging branches the drone weaves through; five scenes mixing buildings and forest, increasing
difficulty; six camera points of view per scene (closer / further, wider / narrower); lifted model vs NTFields at
4000 epochs; the scenes must be rigorous enough that "attach a camera behind the drone and plan the drone alone"
performs very badly). Bodies (`dataprocessing/make_dronecam3d.py`): `drone_only` = a delta-wing polycube, fuselage
14 x 4 x 4, wing span 20, fin 4 tall (the H17 bird was 48 x 18 x 48); `dronecam<L><f>` = drone + 2-unit plate + view
pyramid with the camera L = 30 / 55 / 80 units behind the wing (1.5 / 2.75 / 4 spans; H17: 50 behind a 48-span bird)
and base `t` (tight, 22 x 10: the silhouette + 1) or `w` (wide, 34 x 16), fields of view 59 x 30 (30w) .. 16 x 7 deg
(80t); `drone<L>` = drone + 1-unit cube at the camera (the rigid follower; same frame centre as the two camera bodies of
that L, so one test set scores both). Scenes in the env3 shell (scale 220.457), box soups (12 triangles per box;
`<env>_boxes.json` carries the boxes and part labels): meadow (5 trees, 2 sheds), suburb (6 houses, 8 trees with
22-30-unit branches over the streets), campus (6 blocks 64..95 high, 3 sky bridges, 6 trees, porticos), downtown
(3 x 3 tower grid 84..130 high, streets 40 / 44, 4 sky bridges, masts to y 152, porticos, 4 street trees), overgrown
(the tower grid with 8 kerb trees whose 20-26-unit branches cross the streets, 3 cables, 3 sky bridges). Trees: 6 x 6
trunk, stepped crown 40 tall, 2-4 branches 4 x 4 with a leaf clump. Free fraction of uniform random placements
(`--check`, room shell counted), drone55 / dronecam55w / dronecam80w: meadow 31 / 16 / 9.5 %, suburb 22 / 7.7 / 3.8,
campus 17 / 6.0 / 2.7, downtown 16 / 6.0 / 2.9, overgrown 10 / 1.8 / 0.8 (H17 bodies: 3.5-4.5 %).

Queue `models/graph_homotopy/scripts/run_dronecam_queue.py` (status `results/graph_homotopy/queue_status_dronecam.json`):
45 cells (5 scenes x 3 distances x {follower, tight, wide}), 4 x 100k pairs each, hlift (graph_homotopy_lift) and
mbtd x seeds 0-2 at 4000 epochs, 30 RRT test sets (one per camera-aware cell, `--round 12000 --max-rounds 150`),
hlift dual / mbtd base; the follower of a distance is scored on both sets of that distance against the full camera
body. Tier A (the 55w camera and drone55 on every scene) runs first. Results `results/graph_homotopy/plan_dronecam/`,
summary `aggregate_dronecam.py` -> `results/graph_homotopy/summary/dronecam_summary.md` and
`results/graph_homotopy/dronecam/dash/dronecam_results.csv`. Viewer `dataprocessing/dronecam_viser.py` (port 8097,
container dronecam_viser): scene / camera set-up / planner / body / seed, Camera POV and panel image as H17, a demo
pose where results are missing, `--snapshot DIR` overview renders.

Preprocessing (`preprocess_obj_fast.py`, 2026-10-22): the collision and clearance queries now run on signed-distance
grids of the shape (fine h 0.005 for inside / outside, coarse h 0.02 padded 0.6 for the nearest-point ranking) with the
exact test only on the env points the grids cannot decide, plus an exact grid collision gate on x1 before its full
evaluation; the clearance cull bound from the coarse grid instead of a (B, 250, E) distance table; env points 75 % on the obstacles / 25 % on the room shell (`env_weighted_sampling`, 12k points). The rotational-clearance stage is kept (a third of the time; speed.npy = (distance speed + angle / pi) / 2 needs it -- a first run without it halved the speeds and was redone). Selftest (original
vs fast, same seeds): identical pairs on forest/birdcam3d and campus/dronecam55w, values within float round-off
(directions up to 3e-5 where a clearance of 1e-3 divides). With the machine shared with another session's trainings
(load ~100) the old code ran at ~1 pair/s per shard on these bodies; the first grid version profiled at ~23/s alone.

### H18 interim results (tier A, meadow complete at 06:00; queue running, 44 / 850 jobs)

Meadow, camera 55 behind, wide (34 x 17 deg), 1000 pairs, percent (mean +- sd over seeds; lifted body 1 seed so far):

| model | planned as | clean frame | follow success | drone collision | view blocked | not converged |
|---|---|---|---|---|---|---|
| lifted encoder | camera-aware body | 95.9 | 96.2 | 0.9 | 3.0 | 0.0 |
| lifted encoder | rigid follower | 52.8 +- 0.5 | 73.8 +- 1.4 | 0.1 | 26.0 +- 1.5 | 0.0 |
| full NTFields | camera-aware body | 70.3 +- 1.6 | 71.8 +- 2.5 | 9.7 | 21.0 +- 1.9 | 20.3 +- 2.0 |
| full NTFields | rigid follower | 44.6 +- 1.0 | 57.5 +- 1.4 | 8.5 | 39.3 +- 1.0 | 10.7 +- 0.8 |

Paired (seed-matched, exact McNemar): lifted body vs follower +216 / -6 on follow success (p 5e-56), +440 / -4 on
clean frame; lifted vs NTFields on the body +252 / -10 (p 1e-61). So even in the easiest scene the rigid follower
loses the drone on a quarter of the pairs, and NTFields on the same body fails to converge on a fifth. Dashboard
https://claude.ai/artifact/7Qrf1ARx4mWe17HGGMxQyw (store: datasets results / paired / showcase from
`dronecam_results.csv`, `dronecam_paired.csv`, `export_dronecam_showcase.py`; page
`models/graph_homotopy/report/dronecam_dash/`). The machine is shared (load ~195 on 36 cores, GPUs full): one
evaluation takes ~30 min of CPU collision checking, so the full queue needs 1-2 days.

Update 08:20 (95 / 810 jobs; suburb and campus at 55w complete, 3 seeds except the campus lifted / NTFields bodies at
1): drone kept in frame, lifted body / lifted follower / NTFields body / NTFields follower: meadow 96.9 / 72.1 / 71.8 /
57.5, suburb 98.4 / 78.2 / 73.0 / 62.8, campus 97.5 / 73.0 / 72.0 / 57.1. The follower's typical failure is its camera
point flying into a tree crown or leaf clump (view 0 % for a few steps; the point-sampled body check can miss a 1-unit
cube fully inside a solid, the ray cast does not). `binom_two_sided` moved to log space (2^n overflowed at n > 1024).

Update 12:10 (201 / 765 jobs): meadow complete for all six camera set-ups (bodies 1 seed, followers 2-3). Drone kept
in frame, lifted body / lifted follower / NTFields body: 30t 99.6 / 91.5 / 76.0, 30w 98.0 / 92.6 / 77.8, 55t 95.7 /
67.6 / 71.2, 55w 96.9 / 72.1 / 71.8, 80t 94.9 / 50.8 / 64.7, 80w 94.4 / 57.6 / 65.5. The follower degrades with the
camera distance (its camera point sweeps a longer lever arm through the crowns); the camera-aware body stays above 94 %.

#### 2026-10-22 14:20 — exact clearance check moved to the GPU

The evaluation's long pole was the ground-truth clearance check: one libigl signed-distance call per waypoint
(≈600k per evaluation across the full plan, the drone-only plan, both sheets and the goal) in a 4-process pool,
20–50 min per evaluation on this shared machine against 1–4 min of GPU planning.  `scripts/gpu_clearance.py`
computes the same quantity batched on the GPU (Ericson point–triangle distance + generalized winding number over
(scene sample, body triangle) pairs, bounding-box cut for samples that cannot beat the 0.03 cap); the evaluator takes
`--gpu-clearance` and the dronecam queue passes it.  Equivalence: on three finished runs (475k waypoints) zero sign
mismatches vs the stored libigl values; A/B of 100 cases of meadow 55w lifted body: identical paths, all 17 boolean
record fields identical, clearance magnitudes within 3.4e-5 (libigl's fast-winding path itself is off by up to 6e-5
from its exact unsigned distance, the GPU version matches the exact one to 1e-17).  Clearance time 153 s → 15 s on
100 cases; queue restarted 14:20 with `--max-eval 8` (266/735 jobs done).  Finished cells keep their CPU-checked
results (same signs, hence same success rates).

#### 2026-10-22 14:35 — suburb complete for all six cameras (13 of 30 cells; dashboard datasets results/paired v6)

Drone kept in frame (follow success %, lifted body / lifted follower / NTFields body / NTFields follower; seeds 1–3):
30t 97.9 / 92.6 / 79.6 / 69.6 · 30w 98.1 / 94.8 / 75.4 / 72.8 · 55t 97.9 / 71.6 / 71.3 / 53.6 · 55w 98.4 / 78.2 / 73.0 / 62.8 ·
80t 96.0 / 64.5 / 67.7 / 50.0 · 80w 95.5 / 71.0 / 66.2 / 59.2.  Same shape as meadow: the farther and tighter the camera,
the more the rigid follower loses the drone, while the camera-aware lifted body stays above 95 %.

#### 2026-10-22 15:40 — campus 5 of 6 cameras (17 of 30 cells)

Campus, drone kept in frame (%, lifted body / lifted follower / NTFields body / NTFields follower; 55w 3 seeds, others 1):
30t 99.2 / 88.0 / 84.5 / 73.6 · 30w 97.9 / 93.5 / 81.3 / 76.8 · 55t 97.1 / 67.3 / 72.3 / 50.5 · 55w 96.9 / 72.2 / 72.8 / 57.3 ·
80t 95.2 / 63.6 / 67.9 / 45.8.  Queue 379/725, 28 jobs running, no failures; downtown and overgrown still training.

#### 2026-10-22 16:55 — downtown 55w first cell, campus complete (19 of 30 cells; dashboard results/paired v8, showcase v5)

Campus now 3 seeds on five cameras (30t 99.4 / 89.7 / 84.3 / 73.0 · 30w 98.0 / 93.4 / 82.8 / 77.5 · 55t 97.0 / 67.5 / 73.8 / 50.5 ·
55w 96.9 / 72.2 / 72.2 / 57.3 · 80t 94.3 / 62.7 / 67.0 / 46.1 · 80w 93.9 / 64.7 / 65.6 / 51.5, 1 seed).
Downtown 55w (seed 0): lifted body 95.2, lifted follower 52.0, NTFields body 69.8, NTFields follower 47.2 — the tall
blocks halve the rigid follower's in-frame rate, the camera-aware body barely moves.  Showcase gains the downtown scene
(pair 657, tier 0 of 173 candidates: every rigid follower and the NTFields body lose the drone by step 6).
Queue 445/700, 34 running, no failures.

#### 2026-10-22 17:10 — overgrown 55w first cell: all five showcase scenes on the dashboard (results/paired v9, showcase v6)

Overgrown 55w (seed 0): lifted body 86.0, lifted follower 53.4, NTFields body 55.3, NTFields follower 47.7 — the hardest
scene: the overhanging crowns cost even the camera-aware lifted body 10 points, and the full NTFields body drops to the
follower's level.  Downtown 55w now 3 seeds: 95.2 / 52.0 / 69.3 / 46.6.  Showcase overgrown pair 521 (tier 0 of 205
candidates).  Queue 468/700, 35 running, no failures.

#### 2026-10-22 19:00 — downtown complete for all six cameras (25 of 30 cells; dashboard results/paired v10)

Downtown, drone kept in frame (%, lifted body / lifted follower / NTFields body / NTFields follower; 3 seeds, 80w 1):
30t 99.2 / 79.8 / 86.0 / 69.4 · 30w 98.8 / 83.3 / 83.9 / 74.9 · 55t 97.1 / 49.0 / 78.0 / 43.3 · 55w 95.2 / 52.0 / 69.3 / 46.6 ·
80t 87.8 / 34.4 / 58.2 / 28.8 · 80w 87.0 / 43.1 / 54.6 / 36.3.  The rigid follower with the far camera keeps the drone
in frame on about a third of the pairs; the camera-aware lifted body stays at 87–99.  Overgrown 55w 3 seeds:
85.6 / 55.8 / 57.3 / 47.0.  Queue 546/670, 34 running, no failures; only overgrown's other five cameras remain.

#### 2026-10-22 20:20 — overgrown 4 of 6 cameras (28 of 30 cells; dashboard results/paired v11)

Overgrown, drone kept in frame (%, lifted body / lifted follower / NTFields body / NTFields follower):
30t 91.7 / 66.9 / 69.2 / 48.4 (3 seeds) · 30w 94.7 / 75.5 / 69.1 / 62.3 (1) · 55t 86.8 / 44.0 / 49.9 / 31.9 (2) ·
55w 85.6 / 55.8 / 57.3 / 47.0 (3).  Queue 606/665, 19 running, no failures; 80t / 80w overgrown remain.

#### 2026-10-22 22:35 — H18 complete: 5 scenes x 6 cameras x 4 plans x 3 seeds (queue 660/660, no failures; dashboard results/paired v12)

Drone kept in frame, mean over the six camera set-ups (%, lifted body / lifted follower / NTFields body / NTFields follower):
meadow 96.5 / 71.9 / 70.8 / 57.6 · suburb 97.1 / 78.4 / 73.1 / 61.0 · campus 96.6 / 75.0 / 74.3 / 59.3 ·
downtown 94.1 / 56.9 / 71.7 / 49.9 · overgrown 86.4 / 55.2 / 58.3 / 42.5 · all 30 cells 94.2 / 67.5 / 69.6 / 54.0.
Per cell the lifted body ranges 77.8 (overgrown 80t) – 99.5 (meadow 30t); the rigid follower 34.4 (downtown 80t) – 94.1
(suburb 30w).  Paired (80w, all scenes, seed-matched): lifted body vs follower +904 / −22 (follow success), +1055 / −6
(clean frame); lifted vs NTFields body +761 / −21.  Two difficulty axes behave as intended: camera distance / narrowness
(the follower loses 5–15 points per step back), and scene tier (downtown halves the follower, overgrown is the only scene
that costs the camera-aware body itself more than 10 points).  Timing: 97 evaluations on the CPU check (median clearance
1536 s) vs 263 on the GPU check (median 245 s, on GPUs shared with the trainings).  Dashboard: 5 showcase scenes,
results/paired tables; viewer dronecam_viser on 8097.  Nothing committed.
