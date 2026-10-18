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
