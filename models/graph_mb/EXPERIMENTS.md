# Experiment log: multi-branch decoder that learns the cut (`models/graph_mb`, 2026-10-13/14)

Goal (user, 2026-10-13): implement "Spec: multi-branch decoder that learns the cut" (K learned smooth branches,
min decoder with an annealed soft minimum, band-limited per-branch losses, distinctness at ties, branch management:
usage, revival, pruning, merging), taking what made the previous campaign's heads the best (the folded min-of-branches
decoder with the +-q rotation routes and the conformal cone, "branch K = 4 conformal" 95.4 / 97.5 / 97.9 % on T / L / A,
with the end constraint 93.9 / 96.8 / 96.9 %), and report whether the idea works.

Pipeline for every SE(3) run: models/graph_lie's trainer (original metric NTFields procedure: 4000 epochs x 5 batches of
2000, AdamW wd 0.5, lr 5e-4, accept / reject), Lie gradient, isotropic clearance speed, `exp(-tau/2)` causality weight,
NTFields residual `1e-2 (sqrt(S |c|) - 1)^2`; per-case forward MPPI evaluation (batch 125, seed 0), SE(3) success primary.
Hardware: 3 shared RTX 3090 (GPU 3 dead), 36 cores; 9-13 trainings at once (one run ~2.5 h under that load).

## M0: what the previous K = 4 heads actually learned (diagnostic, before any new training)

`mb_diag.py` on the previous campaign's checkpoints (20 000 training pairs; eta = 0.006, Delta = 0.03, gamma = 0.95):

| checkpoint | units with > 1 % usage | usage per unit | pairs within eta of a tie | duplicates among them (cos > 0.95) | pairs at a real ridge |
|---|---|---|---|---|---|
| T_bc4_s0 | 3 / 4 | 0.82 0.07 0.00 0.11 | 51 % | 99 % | 0.7 % |
| T_bc4_s1 | 4 / 4 | 0.01 0.88 0.07 0.03 | 42 % | 99 % | 0.6 % |
| T_bc4_s2 | 3 / 4 | 0.31 0.02 0.66 0.00 | 55 % | 99 % | 0.8 % |
| T_bc4e_s0 | 4 / 4 | 0.04 0.31 0.54 0.11 | 61 % | 99 % | 0.7 % |
| L_bc4_s0 | 2 / 4 | 0.01 0.98 0.00 0.01 | 26 % | 97 % | 0.9 % |
| A_bc4_s0 | 2 / 4 | 0.00 0.96 0.00 0.04 | 32 % | 98 % | 0.7 % |

The extra residual units of the old heads are near-duplicates of the winner (same value within 2 % of T, parallel
gradients) that win on random patches: the spec's "collapse" and "fake cut" failure modes. The real ridges (ties with
distinct gradients) are < 1 % of the pairs, and almost all of them are route switches (+q vs -q). This matches the
previous campaign's planning numbers (K = 1 >= K = 4 for both the strict and the conformal cone).

## M1: implementation

* Heads `models/graph_method_triangle/multibranch.py`: `graph_mb` (folded: branch = route sigma x unit k,
  `T = T_free,sigma + C_sigma + |F_k(x~) - sigma F_k(y~)|^2 + o_k`, o_0 = 0, o_k = softplus(beta_k) the detour
  offsets; conformal cone on by default) and `graph_mb_free` (the spec's free heads: direct `sqrt(Q_0)` with
  `Q_0 = (|A_0 de|^2 + rho |dG|^2) exp(2 tanh g_0)`, detours `softplus f_k` on the symmetric features (s, d)).
  Per-branch API (`all_values`, `value_branch` on replicated rows, `alive` buffer, `copy_unit`, optimiser-state reset).
* Loss `models/graph_mb/model_function_mb.py`: band rows (every branch within Delta of the minimum) evaluated in one
  autograd call; decoded Eikonal on the winner with the `g > 3 tau` gap mask; sheet Eikonal on every band row; distinctness
  at ties (top-2, hinge on the gradient cosine); end constraint; management statistics. In this Eikonal-only pipeline the
  soft minimum has no separate role (the spec needs it for the Bellman target, which this pipeline does not have): the
  decoded value is the hard minimum and tau only sets the decoded term's exclusion zone. Bellman / sheet consistency need
  a collision oracle at arbitrary poses and are not implemented.
* Trainer `model_train_mb.py`: graph_lie loop + temperature schedule, near-cut stream (20 % of each batch = smallest
  gaps of a second batch), usage checks every 100 epochs (= 500 steps) with revival / pruning / merging, `branch_log.json`.
* Tests `tests/test_multibranch.py` (12): permutation invariance, exact zero at the goal with positive detours,
  symmetry, soft-min -> hard-min, per-branch gradients vs finite differences and vs the Lie module, detached masks /
  finite gradients, one-row-per-pair ablation, schedule, near-cut selection, revival + optimiser reset, dead units.
* Defaults: tau_ref = 0.3 (median T on the scenes); Delta = 0.1 tau_ref, eta = 0.02 tau_ref, tau 0.1 -> 1e-3 tau_ref
  over 60 %, weights 1 / 1 / 0.1, near-cut 20 %, dead < 1 % for 3 checks, 2 revivals then prune, merge at > 50 %
  duplicates. Changed from the spec: revival raise 0.05 tau_ref instead of 0.2 tau_ref (see M2).

## M2: the spec's defaults on T (seed 0, 7 variants): winner takes all

Variants mb4 (folded conformal K = 4), mb4e (+ end), mb4F (free heads), mb4A (losses on the active branch only),
mb6 (K = 6), mb4nd (no distinctness), mb4D (Delta = 0.1 absolute). In every one of them unit 0 holds 93-100 % of the
pairs by epoch 100-200 and 99.7-100 % by epoch 300; the band empties (rows per pair 1.00-1.10), near-ties fall to
0-0.6 %, and the usage checks then run the full revive / revive / prune cycle on every detour unit (6 revivals, 3 prunes
per run; K = 6: 10 / 5). Final learned count: 1 unit (= 2 route branches) in all seven runs.

Why (traced in the logs and on the circle, M3):
1. The detour offsets `o_k = k * 0.015` keep the detours from tying with the direct unit, so they only win where their
   residual is smaller by o_k; with random initial residuals the direct unit's residual shrinks fastest and the others
   fall out of the band within 200 epochs.
2. The spec's revival (copy of the max-residual unit, raised by 0.2 tau_ref = 0.06) lands *outside* the band
   (Delta = 0.03): the revived unit receives no gradient at all, stays dead and is pruned. Lowering the raise below Delta
   (mb4r) or to zero (mb4z, no offsets) only produces duplicates: mb4z's unit 1 held 17 % of the pairs with 74 % of all
   pairs within eta of a tie and 100 % of those duplicates; the merge rule then masked it (14 merges in total).
3. More fundamentally, nothing in the Eikonal-only losses makes a losing branch descend below the winner. A copy
   trained on its band (= everywhere, since it coincides with the winner) rounds a ridge exactly like the winner. The
   committed two-sheet configuration (each branch continues its own family straight through the ridge and leaves the
   band on the other side) is a zero-loss minimum but is not reached from a duplicate by gradient descent: the band loss
   keeps training the loser on the side where it should be free to leave.

## M3: stage 0 (circle, free heads): negative in an Eikonal-only pipeline

`scripts/stage0_circle.py` (goal at 0, unit speed, the spec's free heads, the spec's losses and management, 20 %
near-cut stream, CPU). Three loss forms were tried for the single-head baseline (K = 1) before any cut can be judged:

| loss on the circle | result of the single direct head after 6 000-12 000 steps |
|---|---|
| spec residual `(c^2 - 1)^2`, causality weight | flat plateau between 60 and 135 deg (T = 0.6 where the truth is 1.6-2.4): zero slope is a stationary point of this residual (its gradient `4 c (c^2 - 1)` vanishes at c = 0), median error 36 % |
| NTFields residual `(sqrt(S\|c\|) - 1)^2`, no weight | zigzag with slope +-1 almost everywhere and smooth turning points (fake goals), T at 42 % of the truth, median error 72 % |
| NTFields residual with `exp(-T/2)`, lr 3e-3, 12 000 steps | still a zigzag (turning points at 60, 80, 180, 220, 260, 320 deg; Eikonal residual 0.02 in the ridge band), T at 74 % of the truth, median error 25 % |

The 1-D Eikonal equation has infinitely many almost-everywhere solutions with slope +-1; a smooth free head can
place smooth local minima (false goals) at almost no cost in the residual, and the causality weight only penalises
them weakly. The folded SE(3) head cannot do this (`T >= T_free` by construction, the previous campaign's property 4),
and the spec relies on the Bellman term and the stage-3 lower bound for the same purpose. Without either, stage 0 with
free heads does not pass even its single-branch prerequisite, so its gates (branch count 2, cut within 1 deg of 180)
could not be evaluated. With the gradient-sign split revival the K = 4 runs on the circle behaved like the SE(3) runs:
the direct head keeps 100 % of the points, every revived copy stays a duplicate and is pruned.

## M4: split revival on SE(3) (T, seed 0)

The addition of this campaign (README section 2): a revived unit is an exact copy of its source (2 % weight noise, no
raise) and for 150 epochs the pair is trained on complementary halves of its pairs, split by the sign of
`<c_x^(src), v>` for a random body direction `v`; afterwards the hard minimum decides. No initial offsets, merge rule off.

| run | mechanism | branch population | SE(3) success T standard / big-angle / rotation-heavy |
|---|---|---|---|
| mb4s, mb2s (v1) | reactive: split on revival of a dead unit | units die by epoch 300-700 as before; each split copy holds its forced half during the warm-up and loses it within 100-200 epochs afterwards; 1 unit alive from epoch 1000 (mb2s) / 1400 (mb4s); stopped at epoch 2800 / 2600 | - |
| mb4p | proactive: every 400 epochs the least-used unit is re-split from the max-residual alive unit (6 splits), plus reactive revivals | after every split the copy holds 22-42 % during the warm-up with 60-70 % of pairs near-tied and 99 % of those duplicates, then decays to 0 within 200-300 epochs; final: 1 unit, 4 prunes | 94.7 / 85.3 / 94.9 |
| mb2p | same, K = 2 | same cycle 6 times; the last reactive split fires at epoch 3800, so the checkpoint is mid-warm-up (53 / 47 %, 99 % duplicates) | (evaluated below) |

A proactive split fires inside the accept / reject loop of the trainer: the forced halves raise the logged loss by
> 20 %, the epoch was rejected and rolled back to a pre-split state while the forced assignment stayed on, forever
(both first attempts stalled at epoch 401 with ~3 000 rejections). Fixed by resetting the accept / reject reference and
the rollback queue after any management event; the numbers above are from the fixed runs.

Why the split does not create sheets: the halves are the two route families only where a ridge exists, which is
< 1 % of the pairs (M0). Everywhere else the two halves belong to the same family, so the two copies remain the same
function and the hard minimum assigns their pairs by noise; the usage then drifts to one of them (the one with the
smaller residual gets the decoded term more often, a rich-get-richer dynamic) and the other dies. On the pairs at the
real ridges nothing distinguishes the final one-unit field from the previous campaign's (same route-switch ridges;
`mb_diag`: real-ridge fraction 0.25-0.32 % for every new run vs 0.6-0.9 % for the old heads, which had the duplicates'
near-ties counted in).

Planning with the splits is the previous campaign's number (T seed 0: bc4 94.9): the mechanism neither helps nor
hurts because the final field is the same one-unit field.

## M5: final 3-seed planning results (T / L / A, big-angle, rotation-heavy; `results/graph_mb/summary/plan_summary.md`)

Every learned-cut run ends with one unit (its two rotation routes): 24 of 24 SE(3) runs (3 seeds x 3 scenes x {mb4, mb4e},
plus the T seed-0 variants), so the rows below compare the *training procedure* (near-cut stream, sheet losses on the
band of the route pair, decoded term masked at ties, revival / pruning of the detours) on the same final architecture.

| SE(3) success % | metric baseline | graph_metric | bc4 (winner-only) | **mb4 (spec training)** | bc4e (+ end) | **mb4e (spec training + end)** |
|---|---|---|---|---|---|---|
| T standard | 82.1 +- 0.4 | 90.8 +- 1.0 | 95.4 +- 0.6 | 95.2 +- 2.2 | 93.9 +- 0.9 | 95.6 +- 0.6 |
| L standard | 89.9 +- 2.0 | 96.2 +- 0.8 | 97.5 +- 0.4 | 98.0 +- 0.3 | 96.8 +- 0.2 | 97.2 +- 0.4 |
| A standard | 87.1 +- 1.6 | 94.4 +- 0.2 (2) | 97.9 +- 0.7 | 97.5 +- 0.9 | 96.9 +- 0.5 | 97.6 +- 0.4 |
| T big-angle | 34.2 +- 3.9 | 68.4 +- 3.7 | 85.9 +- 5.2 | 84.8 +- 4.2 | 83.2 +- 4.8 | 86.2 +- 2.6 |
| T rotation-heavy | 80.1 +- 3.6 | 82.3 +- 6.1 | 93.5 +- 3.3 | 92.2 +- 4.2 | 93.4 +- 2.0 | 94.2 +- 1.6 |

Paired differences on common seeds (SE(3) success points): mb4 - bc4 = -0.2 (T), +0.5 (L), -0.4 (A), -1.0 (big-angle),
-1.4 (rotation-heavy); mb4e - bc4e = +1.7 (T), +0.4 (L), +0.7 (A), +3.0 (big-angle), +0.8 (rotation-heavy). Against
graph_metric: mb4 +4.5 / +1.8 / +3.1 / +16.4 / +9.9. T seed-0 single runs: free heads 94.8 (big-angle 80.6), active-only
ablation 96.1, proactive splits K = 4 94.7, K = 2 95.1 -- all within the seed spread of the folded head.

Verdict: with the end constraint the spec's procedure recovers the 1-1.5 points the constraint used to cost (mb4e is
now level with bc4), without it the procedure changes nothing beyond seed noise (T seed 0 alone looked like +2.8; seeds
1-2 took it back, sd 2.2). The cut is not learned: no run keeps a second unit, the real ridges in the final fields are
the +-q route switches that the fold already builds in (0.25-0.33 % of pairs), and the planning numbers are those of the
previous campaign's head. Honest comparison for the spec's stage 3b: the learned-cut decoder beats graph_metric and the
metric baseline by the margins of the previous campaign and does not beat the branch K = 4 conformal head it is built on.

What would be needed to test the idea properly (not done here): a Bellman / sheet-consistency term with a collision
oracle at arbitrary poses (the spec's own mechanism for the level of a losing sheet and against false goals), or
route anchors from a planner as a warm start; and scenes whose geometric ridges (over / under a bar) occupy more than
the < 1 % of training pairs they occupy on T / L / A.
