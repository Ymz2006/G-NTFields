# `models/graph_mb`: multi-branch decoder that learns the cut

Implementation of "Spec: multi-branch decoder that learns the cut" (2026-10-13) on top of the best head of the
previous campaign (the folded min-of-branches decoder with the +-q rotation routes and the conformal cone,
`models/graph_method_triangle/branch_decoder.py`, "branch K = 4 conformal"), trained in the graph_lie pipeline.
Log with every finding and number: `EXPERIMENTS.md`. Results: `results/graph_mb/` (planning tables
`summary/plan_summary.md`, branch populations `summary/branch_summary.md`, diagnostics `diag/`, stage 0 `stage0*/`).

## 1. What the spec asks for and how it maps onto this code

| spec | here |
|---|---|
| `T = min_k T_k`, K smooth branches, cut = ties | `multibranch.py`: `graph_mb` (folded) and `graph_mb_free` (free heads); `alive` buffer = learned branch count |
| direct head zero only at the goal, detours positive there | folded: route + of unit 0 is `T_free + C + |dF_0|^2` (exactly 0 at x = y), detours carry `o_k = softplus(beta_k) > 0`; free: `Q_0 = (|A_0 de|^2 + rho |dG|^2) e^{2 tanh g_0}`, `Q_k = softplus(f_k)^2` |
| exchange symmetry | folded: every term is a symmetric comparison of per-pose quantities; free: features `s = e_x + e_y`, `d = (e_x - e_y)^2` |
| soft-min with annealed tau (training), hard min (evaluation) | hard min everywhere; tau anneals `0.1 -> 1e-3 tau_ref` over 60 % and sets the decoded term's exclusion zone `g > 3 tau` (the soft minimum only matters for the Bellman target, which this pipeline does not have) |
| decoded Eikonal, sheet Eikonal on bands, distinctness at ties | `model_function_mb.Function.Loss`: one autograd call over the replicated band rows gives every in-band branch's endpoint Lie gradients; NTFields residual, causality weight |
| Bellman, sheet consistency, lower bound | not implemented: they need the speed / collision state at arbitrary poses (the NTFields datasets carry speeds only at the sampled pairs); the folded head has `T >= T_free` by construction |
| near-cut sampling, usage checks, revival, pruning, merging | `model_train_mb.Model`: 20 % of each batch from the smallest gaps of a second batch; checks every 100 epochs (500 steps); `branch_log.json` |
| route anchors (ablation) | not used |
| unit tests of section 10 | `tests/test_multibranch.py` (12 tests) |
| stages 0-2 ground truth | stage 0 (circle): `scripts/stage0_circle.py`; stages 1-2 (fast marching) not run |
| stage 3b | T / L / A scenes, per-case MPPI, against graph_metric, the original metric head and the previous campaign's heads |

Defaults (`model_function_mb.DEFAULT_SPEC`): `tau_ref = 0.3` (median T on the scenes), `Delta = 0.1 tau_ref`,
`eta = 0.02 tau_ref`, `gamma = 0.95`, weights decoded / sheet / distinct = 1 / 1 / 0.1, near-cut 20 %, dead < 1 % for
3 checks, 2 revivals then prune, merge above 50 % duplicates. Deviations from the spec are listed in EXPERIMENTS.md M1.

## 2. Additions beyond the spec (why they were needed)

* **Revival inside the band.** The spec raises a revived copy by `0.2 tau_ref`, twice the band margin, so the copy
  receives no gradient and dies again. The default here is `0.05 tau_ref`.
* **Split revival (gradient-sign halves).** A revived copy that is trained on its whole band rounds every ridge
  exactly like its source; nothing in the Eikonal losses makes it take one side. The two sheets at a ridge differ by
  their gradient direction, so a revived unit is made an exact copy of its source and, for `warm_epochs`, the pair is
  trained on complementary halves of its pairs, split by the sign of `<c_x^(src), v>` for a random body direction `v`
  (`split_revive`); afterwards the hard minimum decides. `split_every` applies the same split proactively to the
  least-used unit during the annealing phase. The merge rule is switched off with splits (coinciding sheets away from
  ridges are expected).
* **Stage 0 loss form.** The spec's residual `(|c|^2 - 1)^2` has zero gradient at zero slope: a flat region is a
  stationary point and the single head stays flat on the circle. The NTFields form `(sqrt(S |c|) - 1)^2` is used
  everywhere, with the causality weight.

## 3. Train / evaluate

```bash
python -m unittest tests.test_multibranch -v
python train/train_graph_mb.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_mb4_s0 --seed 0 --arch graph_mb \
    --set n_branch=4 --spec delta=0.1                                       # spec defaults
python train/train_graph_mb.py ... --set n_branch=4 offset_init=0.0 --spec split_revive=True split_every=400 merge_frac=1.01
python3 models/graph_mb/scripts/run_mb_queue.py --phase explore --gpus 0,1,2    # host-side queue (train + per-case MPPI)
python3 models/graph_mb/scripts/aggregate_mb.py; python3 models/graph_mb/scripts/branch_summary.py
python models/graph_mb/scripts/mb_diag.py --out results/graph_mb/diag/T_mb4_s0.json Experiments/graph_mb/T_mb4_s0/latest.pt
python models/graph_mb/scripts/stage0_circle.py --out results/graph_mb/stage0 --seeds 0 1 2
```

Checkpoints keep the graph_native contract (`--kind graph_native` in `plan_eval_graph.py`); the `alive` buffer in the
state dict carries the learned branch count into evaluation.

## 4. Results (3 seeds; details and all variants in EXPERIMENTS.md M0-M5, `results/graph_mb/summary/`)

| SE(3) success % | bc4 (previous, winner-only) | mb4 (spec training) | bc4e (+ end) | mb4e (spec training + end) |
|---|---|---|---|---|
| T / L / A standard | 95.4 / 97.5 / 97.9 | 95.2 / 98.0 / 97.5 | 93.9 / 96.8 / 96.9 | 95.6 / 97.2 / 97.6 |
| T big-angle / rotation-heavy | 85.9 / 93.5 | 84.8 / 92.2 | 83.2 / 93.4 | 86.2 / 94.2 |

* The cut is not learned: every SE(3) run (24 of 24) ends with one unit (its two rotation routes). The spec's defaults
  reach winner-take-all by epoch 300; revival as specified lands outside the band; without offsets the extra units are
  duplicates; the gradient-sign split revival produces duplicates that decay. The Eikonal-only losses contain no force
  that makes a losing branch take one side of a ridge (EXPERIMENTS.md M2, M4).
* The training procedure itself is neutral to slightly positive: it recovers the 1-1.5 points the end constraint used to
  cost; without the end constraint the differences to the previous head are seed noise.
* Stage 0 (circle, free heads) fails its single-branch prerequisite in an Eikonal-only pipeline (false goals / zigzag
  solutions of the 1-D equation); it needs the spec's Bellman term or a lower-bound structure (M3).
* Report page: results/graph_mb/report/learned-cut.html (`scripts/make_report_page.py`).
