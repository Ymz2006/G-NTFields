# `models/graph_branch`: min-of-branches decoder with goal ("end") constraints

Implements the reference design of "Decoder properties that hold with or without obstacles" (2026-10-13) as a
pair head for the SE(3) travel-time field, trains it in the graph_lie pipeline (original metric NTFields
trainer, Lie-gradient Eikonal residual, isotropic clearance speed) and tests every property of the note.
Head: `models/graph_method_triangle/branch_decoder.py` (`graph_branch`, `graph_branch_conf`), registered with
`build_network`, so graph_native's loader and the per-case evaluator (`--kind graph_native`) use it unchanged.
Log: `EXPERIMENTS.md`. Results: `results/graph_branch/`.

## 1. The graph architecture it replaces (analysis)

Every graph head of this repository has the same three layers.

1. **Fixed pose embedding** `G(x) = (t, vec R)` in R^12 (smooth, injective; keeps `H = (G(x), G(y), Z)` an
   embedding of the pair space whatever the latent `Z` is).
2. **Siamese G-Shape encoder** `e(x)`: random Fourier features of `G(x)` (sigma 0.5 on translation columns,
   0.1 on rotation columns), Lipschitz row-normalised sine-gated residual blocks, 256 outputs plus the input
   skip `G(x)`; both endpoints go through the same encoder, so every latent is a comparison of per-pose features.
3. **Smooth latent + fixed decoder**. graph_gram: `a = |A (e_x - e_y)|^2` (Euclidean embedding distance),
   `b = (1 - eps) (1 - <P_x, P_y>_F)` with `P = L L^T / |L L^T|_F` a 4x4 PSD Gram lift (`P = q q^T` gives
   `b = sin^2(theta / 2)` exactly); graph_metric replaces `a` by the square of the G-Shape logsumexp group
   distance. Decoder `T = sqrt(a / s_t^2 + phi(b) / s_r^2)`, `phi(b) = 4 asin^2 sqrt(b)` (exact in free space).

All non-smoothness sits in the decoder: the goal cone (`sqrt` at `Q = 0`) and one ridge, where `b -> 1`
(`phi'` blows up). Consequences once obstacles are present (the note's critique, measured in `results/`):

* the only rotational ridge is the preimage of `b = 1`, i.e. where the learned Gram lifts become orthogonal --
  the half turn of the lift. The lift is learned, so the ridge can be moved by reparametrising it, but each start
  has exactly one such surface, and its height is fixed by the decoder:
* the rotation term is capped, `phi(b) / s_r^2 <= pi^2 / s_r^2 = 1/4` (rotations in turns): a pure rotation can
  cost at most 0.5. Any extra rotational time must go into `a`, which is an embedding distance and has no
  ridges, so the field is rounded where two rotation routes tie;
* nothing bounds `T` from below by the speed-limit time `T_free`, and the cone at the goal is a learned quadratic
  form (`J^T J` of the latents), not the metric's shape.

## 2. The branch decoder

    T(x, y) = min_{sigma in {+,-}} min_{k=1..K} [ T_free,sigma(x, y) + C_sigma(x, y) + delta_{sigma,k}(x, y) ]

* `T_free,sigma = sqrt(|t_x - t_y|^2 / s_t^2 + psi_sigma^2 / s_r^2)`, with unit quaternions of either sign,
  `c = q_x . q_y`, route + turning by `psi_+ = 2 atan2(|v|, c)` in `[0, 2 pi]` and route - by `2 pi - psi_+`.
  The two routes are the copies of the R^3 x S^3 product metric under `q -> -q` (property 3b); a route keeps
  its identity through the half turn, so it can stay selected as the long way round.
* `delta_{sigma,k} = |F_k(x~) - sigma F_k(y~)|^2` (or a squared group-l_p / logsumexp distance), with
  `F_k(x~) = [U_k e(x), sum_i q_i W_{k,i} e(x)]`: an even part (a function of `G(x)`) and a part odd in the lift.
  Flipping the sign of `q_y` swaps the two routes and their residuals, so `T` is a function on SE(3) x SE(3)
  whatever signs the quaternion extraction picks (`quat_from_matrix`, Shepperd; the test suite checks it).
* `C_sigma = rho_bar lam_bar tanh(T_free,sigma / lam_bar)` (`graph_branch_conf` only): near the goal
  `C = rho(p) T_free + O(T_free^3)`, i.e. the cone keeps the metric's shape scaled by `1 + rho(p)`; far away it
  saturates (an escape-time offset).
* Hard minimum; the selected branch is found without autograd and recomputed with gradients, so the unused
  routes' singular points cannot produce NaN in the double backward of the Eikonal loss.

### Constraints at the ends (the goal)

* Architectural, both endpoints (properties 1, 5, 6): `T(x, x) = 0` exactly; `T = sqrt(Q)` with `Q` smooth to
  second order and `grad Q = 0` at the goal; the cone is the free metric's (`graph_branch`, the note's
  statement) or the speed-weighted metric's `g / S(p)^2` with `1 + rho(p)` as scale (`graph_branch_conf`).
* End constraint (loss, `--end-weight`): in the limit `x -> y` the Eikonal equation `S |c| = 1` reduces to the
  algebraic condition `S(p) (1 + rho(p)) = 1` on the cone scale. It is imposed at both endpoints of every
  training pair with the residual form of the Eikonal loss: `1e-2 [(sqrt(S_x (1 + rho_x)) - 1)^2 + (... y)^2]`.

### Which properties hold for every parameter value

| # | property | graph_gram | branch (this head) | test |
|---|---|---|---|---|
| 1 | `T >= 0`, zero exactly at the goal | pseudometric (zero off-diagonal possible) | yes (`T >= T_free > 0` off the goal) | `test_p1_*` |
| 2 | symmetry | yes | yes (bitwise) | `test_p2_*` |
| 3 | triangle inequality | rank-1 lift only | not guaranteed (measured) | `test_p3_*`, `decoder_properties.py` |
| 4 | `T >= T_free` | no | yes | `test_p4_*` |
| 5 | cone at the goal, both endpoints | yes | yes | `test_p5_p6_*` |
| 6 | the cone has the metric's shape | no (learned form) | yes (free metric; conformal scale for `_conf`) | `test_p5_p6_*`, `test_p6_fails_for_graph_gram` |
| 7 | peaks only | yes | yes (`sq`, `lp`; `lse` above a 1e-3 smoothing scale) | `test_p7_peaks_only` |
| 8 | movable ridges | one surface per start (`b = 1`) | yes (route / branch ties) | `test_p8_*`, synthetic benchmark |
| 9 | no cap on time | rotation term `<= 1/4` | yes | `test_p9_no_cap`, synthetic benchmark |

## 3. Results (details: EXPERIMENTS.md B1-B8, tables: results/graph_branch/summary/)

* **Obstacle planning** (graph_lie pipeline, Lie gradient, iso speed, per-case forward MPPI, SE(3) success, 3 seeds,
  `results/graph_branch/summary/plan_summary.md`):

  | set | metric baseline (original) | graph_gram | graph_metric | branch, strict cone (K=1) | branch, conformal | branch, conformal + end |
  |---|---|---|---|---|---|---|
  | T-shape | 82.1 +- 0.4 | 79.8 +- 4.6 | 90.8 +- 1.0 | 90.4 +- 1.3 | **95.4 +- 0.6** | 93.9 +- 0.9 |
  | L-shape | 89.9 +- 2.0 | 85.8 +- 1.3 | 96.2 +- 0.8 | - | **97.5 +- 0.4** | 96.8 +- 0.2 |
  | A-shape | 87.1 +- 1.6 | 90.2 +- 0.7 | 94.4 +- 0.2 (2 seeds) | - | **97.9 +- 0.7** | 96.9 +- 0.5 |
  | T big-angle | 34.2 +- 3.9 | 45.9 +- 2.5 | 68.4 +- 3.7 | 63.9 +- 3.3 | **85.9 +- 5.2** | 83.2 +- 4.8 |
  | T rotation-heavy | 80.1 +- 3.6 | 68.9 +- 5.9 | 82.3 +- 6.1 | 82.3 +- 2.8 | **93.5 +- 3.3** | 93.4 +- 2.0 |

  Metric baseline: the original metric NTFields checkpoints (chart input, combined speed, its own trainer; seeds 1-3),
  scored by the same evaluator. Methodology report: `results/graph_branch/report/branch_decoder_methodology.pdf`.

  Ablation (T): graph_gram + the conformal term alone 91.3 +- 1.5 (big-angle 80.5, rotation-heavy 88.8). The decoder
  structure removes non-convergence (7.5 % -> 1.7 %), the conformal goal term removes collisions (14.7 % -> 1.9 %); each is
  worth ~11 points alone, together +15.6. The conformal head learns the cone scale from the Eikonal loss alone (median
  speed x slope 1.01); the end constraint only raises slow goals (which stay too shallow) and costs ~1-1.5 points (B11).
* **Trained fields** (`decoder_properties.py`): graph_gram's decoder kink is never reached among obstacles (b peaks
  far below 1 on every rotation slice; its field has no ridge along 200 random lines) and it falls below T_free on up to
  4.7 % of pairs; the branch heads keep every guaranteed property on trained weights and have real ridges.
* **Exact truth** (synthetic slow-sector / disc problems): with labels, graph_gram finds the ridge position but its
  capped rotation term leaves the ridge 14-16 % too low; branch K = 4 is within 1-2 %. Trained by the Eikonal loss
  alone, graph_gram learns a false goal (a near-zero minimum away from the start) and steepest descent fails on 67 %
  of the pairs; the branch heads cannot have one (T >= T_free) and every plan succeeds.
* **Triangle inequality**: properties 3, 4 and the strict 6 cannot all hold by construction for a field that sees
  obstacles (EXPERIMENTS.md B6), so it is measured: 0.04-0.12 % of random triples violate it.

## 4. Reproduce

Inside `pytorchserver:latest` with this repo at `/workspace`:

```bash
python -m unittest tests.test_branch_decoder -v                      # architectural property tests
python train/train_graph_branch.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_br4_s0 --seed 0 \
    --arch graph_branch --grad lie --speed iso --set n_branch=4
python train/train_graph_branch.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_bc4e_s0 --seed 0 \
    --arch graph_branch_conf --grad lie --speed iso --end-weight 1 --set n_branch=4
python models/graph_branch/scripts/decoder_properties.py --scene Tshape3d_env1 --out <json> <ckpt>
python models/graph_branch/scripts/synth_fit.py --kind sector --A 4 --heads ggram,br1,br4 --out results/graph_branch/synth/sector_A4
python models/graph_branch/scripts/synth_dataset.py --kind sector --A 4 --out datasets/3dshape/synth_sector_A4
```

Host side (queue of trainings + per-case MPPI evaluations, shares GPUs with memory guards):

```bash
python3 models/graph_branch/scripts/run_branch_queue.py --phase explore --gpus 0,1,2 --per-gpu 3
python3 models/graph_branch/scripts/aggregate_branch.py
```
