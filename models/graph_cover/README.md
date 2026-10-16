# `models/graph_cover`: cover-quotient metric decoder, ridge detection, kink census (2026-10-13)

Campaign goals (user request of 2026-10-13): a decoder that beats the graph_gram decoder, with the
latent free to change; a working triangle inequality (earlier guaranteed heads were 1-25 points behind
the non-metric logsumexp heads among obstacles); the "peaks only" and "movable ridges" properties of
*Decoder properties that hold with or without obstacles*; and the staged ridge-detection idea (detect
the rounded ridge of a smooth field where the cross derivative vanishes, use it for checking, planning
and route recovery). Experiment log: `EXPERIMENTS.md`. Results: `results/graph_cover/`.

## 1. The decoder (`models/graph_method_triangle/cover_decoder.py`, arch `graph_cover`)

Property 3 of the note allows the triangle inequality in two cases: one distance between embeddings
(no ridges), or branches that are copies of one metric under a symmetry group, T = min_g d(x, g.y).
The second case is more general than the +-q rotation routes: any finite group acting freely by
isometries on a covering space gives a quotient pseudometric, and the group does not have to come from
the geometry. The head builds covers with *learned branch points*:

    even      E(x)    = U e(x)
    q-odd     Q(x~)   = q_x (x) W e(x)                       odd under q -> -q (rotation double cover)
    cut-odd   O_i(x~) = w(z_i(x)) (x) A_i e(x),  i = 1..m    odd under the deck map of the i-th learned cover

with e(x) the G-Shape encoder embedding of G(x) = (t, vec R) (standardised), z_i(x) = Z_i e(x) in R^2 and
w(z) = |z|^{1/2} sqrt(z) the complex square root (scaled so that |w| = |z|, Lipschitz at the branch point).
A loop around a zero of z_i flips the sign of w, so O_i lives on a double cover branched at {z_i = 0}: the
branch point is learned and is meant to sit inside an obstacle. G = Z_2^{1+m} acts by sign flips of the
odd blocks (isometries of every l_p norm). With the exact free-space routes

    T_free,sigma^2 = |t_x - t_y|^2 / s_t^2 + psi_sigma^2 / s_r^2,   psi_+ = 2 atan2(|v|, q_x . q_y),  psi_- = 2 pi - psi_+

and a squared seminorm rho of the feature difference (`euclid`: |d|^2; `lp`: (0.2 sum_g |d_g|_8)^2, the
exact seminorm of graph_tri_metric), the field is

    T(x, y)^2 = min_{g in G} [ T_free,sigma(g)(x, y)^2 + rho(F(x~) - g . F(y~)) ],   F = [E; Q; O_1; ..; O_m].

sqrt(T_free^2 + rho) is a G-invariant pseudometric on the cover, so its minimum over G is a pseudometric
on the base: for every parameter value T >= 0 with T(x, x) = 0, T symmetric, **T(x, z) <= T(x, y) + T(y, z)**,
T >= T_free, a sqrt(Q) cone at the goal, only min / sqrt on the output path (**peaks only**), ridges wherever
two group elements tie (**movable**: the sigma tie is the rotation half-turn modulated by Q, the s_i tie is
the surface where z_i(x) and z_i(y) are antiparallel, a learned sheet emanating from the branch point, i.e.
the shape of an obstacle's shadow), and no cap (rho unbounded). The branch is selected without autograd and
only the selected element is recomputed with gradients. `res_kind='lse'` (graph_metric's logsumexp group
max) is the non-metric ablation with the same branch structure; `tfree=False` drops the lower bound;
`soft_tau > 0` replaces the min over the learned cuts by a soft minimum of width tau (differentiable
everywhere, T(x, x) within tau log 2^m of 0, triangle deviation O(tau)).

Tests: `tests/test_graph_cover.py` (9): half-angle identities and finite derivatives at the cut and the
branch point; symmetry, exact zero, lower bound; residual off = exact free-space time; adversarial
triangle attack at rounding level for every metric variant and violable for the lse ablation; the min
equals the candidate minimum and several elements are active; finite double backward; quadratic cone.

Training: `train/train_graph_cover.py` = the original metric NTFields pipeline of `models/graph_lie`
(4000 epochs, AdamW, accept / reject, exp(-tau/2) weight, 1e-2 (sqrt N - 1)^2 residual), iso speed
(translational clearance S_d), Lie gradient. Evaluation: the per-case forward MPPI of
`plan_eval_graph.py --kind graph_native` (1000 RRT-verified cases, seed 0, batch 125), SE(3) convergence.

## 2. Ridge detection (`ridge.py`)

Body-frame gradient c and Hessian H of xi -> T(x exp(xi), y) at xi = 0 by autograd; detector: S|c| < band,
lambda_min(H) < -kappa, cross derivative r = c . v (v the eigenvector of lambda_min) = 0. Tools:
`scripts/ridge_free.py` (stage 1: validation on the half-turn ridge of the soft-min model and of trained
free-space fields), `scripts/ridge_plan_free.py` (stage 2: detector-triggered sampling; stage 3: recover
the two routes on either side of the ridge and descend the min of their extrapolations),
`scripts/slice_ridge.py` (obstacle slices: detector vs the exact ridges of the branch heads),
`scripts/kink_census.py` (property 7: ridge vs valley kinks along random lines), `scripts/cover_diag.py`
(triangle attack, element usage, speed calibration of native checkpoints).

## 3. Results

See `EXPERIMENTS.md` (filled in as the campaign runs) and `results/graph_cover/summary/plan_summary.md`.
