"""Losses of the multi-branch decoder that learns the cut (spec section 5), in the graph_lie / graph_branch
pipeline (original metric NTFields trainer, Lie-gradient Eikonal residual, isotropic clearance speed).

Per training pair i the head gives every branch's value V_ib (forward only). With T_(1) <= T_(2) the two smallest
values, g = T_(2) - T_(1) the gap and Delta the band margin, the band rows are the pairs (i, b) with
V_ib <= T_(1) + Delta (the winner is always one of them). Every band row is evaluated once more with gradients
(value_branch on replicated inputs: one autograd call gives the per-branch endpoint Lie gradients c_x^(b), c_y^(b)
of all rows at once), and the usual NTFields residual r_ib = 1e-2 [(sqrt(S_x |c_x^(b)|) - 1)^2 + (sqrt(S_y |c_y^(b)|)
- 1)^2] with the causality weight exp(-T_b / 2) is formed per row. Terms:

  decoded Eikonal   L_dec   = sum_i 1{g_i > 3 tau} r_{i, win} w / B           (winner rows, away from the cut)
  sheet Eikonal     L_sheet = sum_{band rows} r_ib w / #rows                  (every branch on its band: this is
                                                                               what places the cut; the loser
                                                                               is trained where it loses)
  distinct at ties  L_dist  = sum_i 1{g_i < eta} 1e-2 mean_{x,y} relu(cos(c^(1), c^(2)) - gamma)^2 / B
  end constraint    L_end   = end_weight 1e-2 [(sqrt(S_x (1 + rho_x)) - 1)^2 + (... y)^2]   (graph_branch's,
                                                                               cone scale of the direct head)

tau is the spec's annealed temperature (0.1 tau_ref -> 1e-3 tau_ref, log-linear over the first 60 % of training).
In this Eikonal-only pipeline the soft minimum itself has no separate role (the spec uses it for the Bellman
target, which this pipeline does not have; the decoded Eikonal is masked where soft and hard min differ), so the
decoded value is the hard minimum and tau only sets the exclusion zone 3 tau of the decoded term around the cut.
The Bellman and sheet-consistency terms of the spec need the speed at arbitrary poses (a collision oracle), which
the dataset pipeline does not provide; they are not implemented.

Masks (band, gap, winner, ties) are computed from detached values. Statistics for the trainer's branch management
(usage, per-branch sheet residual in band, duplicate counts per unit pair, gap fractions) are accumulated in
``self.stats`` and reset by the trainer at every check.
"""
import math

import torch

from models.graph_branch import model_function_branch as mfb
from models.graph_lie import model_function_lie as mfl
from models.graph_native import _sib

TWO_PI = 2.0 * math.pi
DEFAULT_SPEC = dict(
    tau_ref=0.3,        # time scale of the spec's defaults (median T on the obstacle scenes ~ 0.33)
    delta=0.1,          # band margin Delta / tau_ref
    tau0=0.1,           # temperature start / tau_ref
    tau_end=1e-3,       # temperature end / tau_ref
    anneal_frac=0.6,    # fraction of training over which tau decays log-linearly
    gap_mask=3.0,       # decoded Eikonal only where g > gap_mask * tau
    w_dec=1.0, w_sheet=1.0, w_dist=0.1,
    eta=0.02,           # tie threshold / tau_ref
    gamma=0.95,         # duplicate cosine threshold
    near_cut_frac=0.2,  # fraction of each batch drawn from the smallest-gap rows of a pool
    band_only_active=False,   # ablation: sheet losses on the winner only
    check_every=100,    # epochs (5 steps each) between usage checks = 500 steps
    dead_frac=0.01, patience=3, revive_noise=0.1, revive_raise=0.05, max_revive=2,   # raise 0.05 tau_ref: inside the band (spec: 0.2, outside it)
    merge_frac=0.5, merge_min_shared=20, pool_size=8192,
    # split revival (this campaign's addition, see README): the revived unit is an exact copy of its source
    # (relative noise split_noise, no raise) and for warm_epochs the two are trained on complementary halves of
    # their pairs, split by the sign of <c_x^(src), v> for a random body direction v (the gradient direction flips
    # across a ridge, so the halves are the two route families wherever a ridge exists)
    split_revive=False, split_noise=0.02, warm_epochs=150,
    # proactive splits: every split_every epochs (within the annealing phase) the least-used detour unit is re-split
    # from the alive unit with the largest in-band sheet residual (0 = off)
    split_every=0,
)


class Function(mfb.Function):
    def __init__(self, path, device, network, dim, grad_mode='lie', speed_model='iso', weight_mode='exp',
                 end_weight=0.0, planar=False, spec=None, epochs=4000):
        super().__init__(path, device, network, dim, grad_mode, speed_model, weight_mode, end_weight, planar)
        self.spec = dict(DEFAULT_SPEC)
        self.spec.update(spec or {})
        self.epochs = int(epochs)
        self.splits = {}   # unit k -> (src unit, v (6,), until epoch)
        self.reset_stats()

    # ------------------------------------------------------------------ schedule / helpers
    @property
    def head(self):
        return self.network.net

    def tau(self, epoch):
        s = self.spec
        f = min(1.0, max(0.0, (epoch - 1) / max(1.0, s['anneal_frac'] * self.epochs)))
        return s['tau_ref'] * s['tau0'] * (s['tau_end'] / s['tau0']) ** f

    def reset_stats(self):
        nb = self.head.n_branches
        nu = self.head.n_units
        dev = self.device
        self.stats = {'win': torch.zeros(nb, device=dev), 'res_sum': torch.zeros(nb, device=dev),
                      'res_n': torch.zeros(nb, device=dev), 'shared': torch.zeros(nu, nu, device=dev),
                      'dup': torch.zeros(nu, nu, device=dev), 'n_pairs': 0, 'n_rows': 0, 'n_gap3tau': 0,
                      'n_near': 0, 'n_near_dup': 0}

    @torch.no_grad()
    def branch_table(self, points):
        """V (B, nb), sorted values, order for chart rows (forward only)."""
        g0 = self.network.features(points[:, :self.dim])
        g1 = self.network.features(points[:, self.dim:])
        V = self.head.all_values(g0, g1)
        srt, order = V.sort(1)
        return g0, g1, V, srt, order

    @torch.no_grad()
    def near_cut_indices(self, pool_points, n):
        """Indices of the n rows of the pool with the smallest gap T_(2) - T_(1) (spec section 6, step 4)."""
        _, _, _, srt, _ = self.branch_table(pool_points)
        gap = srt[:, 1] - srt[:, 0]
        gap = torch.where(torch.isfinite(gap), gap, torch.full_like(gap, 1e9))
        return gap.topk(min(n, gap.shape[0]), largest=False).indices

    # ------------------------------------------------------------------ loss
    def _row_gradients(self, points, g0, g1, rows_i, rows_b):
        """T_b and the endpoint gradients (c_x, c_y) of every band row, with graph."""
        net, head = self.network, self.head
        if self.grad_mode == 'lie':
            gx = g0[rows_i].detach().requires_grad_(True)
            gy = g1[rows_i].detach().requires_grad_(True)
            T = head.value_branch(gx, gy, rows_b)
            dgx, dgy = torch.autograd.grad(T.sum(), (gx, gy), create_graph=True)
            Rx = gx.detach()[:, 3:].reshape(-1, 3, 3)
            Ry = gy.detach()[:, 3:].reshape(-1, 3, 3)
            cx = _sib.tangents.project_feature_gradient(dgx, Rx, 1.0, TWO_PI)
            cy = _sib.tangents.project_feature_gradient(dgy, Ry, 1.0, TWO_PI)
            return T, cx, cy
        X = points[rows_i].clone().detach().requires_grad_(True)
        T = head.value_branch(net.features(X[:, :self.dim]), net.features(X[:, self.dim:]), rows_b)
        d = torch.autograd.grad(T.sum(), X, create_graph=True)[0]
        return T, d[:, :self.dim], d[:, self.dim:]

    def Loss(self, points, speeds, beta, gamma, epoch):
        s = self.spec
        head = self.head
        B = points.shape[0]
        tau = self.tau(epoch)
        Delta, eta = s['delta'] * s['tau_ref'], s['eta'] * s['tau_ref']
        nb = head.n_branches
        with torch.no_grad():
            g0, g1, V, srt, order = self.branch_table(points)
            Tmin, win, second = srt[:, 0], order[:, 0], order[:, 1]
            gap = srt[:, 1] - srt[:, 0]
            gap = torch.where(torch.isfinite(gap), gap, torch.full_like(gap, 1e9))
            if s['band_only_active']:
                band = torch.nn.functional.one_hot(win, nb).bool()
            else:
                band = V <= (Tmin + Delta)[:, None]
            rows_i, rows_b = band.nonzero(as_tuple=True)
            n_rows = rows_i.shape[0]
            row_of = torch.full((B, nb), -1, dtype=torch.long, device=points.device)
            row_of[rows_i, rows_b] = torch.arange(n_rows, device=points.device)
            is_win = rows_b == win[rows_i]
            sp_rows = {k: v[rows_i] for k, v in speeds.items()}
        T, cx, cy = self._row_gradients(points, g0, g1, rows_i, rows_b)
        if self.planar:
            N0 = mfb.planar_N(cx, sp_rows, 0, self.speed_model)
            N1 = mfb.planar_N(cy, sp_rows, 1, self.speed_model)
        else:
            N0 = mfl.eikonal_N(cx, sp_rows, 0, self.speed_model)
            N1 = mfl.eikonal_N(cy, sp_rows, 1, self.speed_model)
        lw = 1e-2
        r = lw * (torch.sqrt(N0) - 1) ** 2 + lw * (torch.sqrt(N1) - 1) ** 2
        if self.weight_mode == 'exp':
            w = torch.exp(-0.5 * T)
        elif self.weight_mode == 'exp_detached':
            w = torch.exp(-0.5 * T.detach())
        else:
            w = torch.ones_like(T)
        rw = r * w
        m_dec = (is_win & (gap[rows_i] > s['gap_mask'] * tau)).to(rw.dtype)
        w_sheet = torch.ones_like(rw)
        K = head.n_units
        for k, (src, v, until) in list(self.splits.items()):
            if epoch > until or head.alive[k] < 0.5:
                del self.splits[k]
                continue
            with torch.no_grad():
                wu = head.unit_of_branch(win)
                route = win // K if head.FORM == 'folded' else torch.zeros_like(win)
                sel = ((wu == src) | (wu == k)).nonzero(as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                rs = row_of[sel, route[sel] * K + src]
                rk = row_of[sel, route[sel] * K + k]
                ok = (rs >= 0) & (rk >= 0)
                rs, rk = rs[ok], rk[ok]
                if rs.numel() == 0:
                    continue
                to_k = (cx.detach()[rs] @ v) < 0
                m_dec[rs] = (~to_k).to(rw.dtype)
                m_dec[rk] = to_k.to(rw.dtype)
                w_sheet[rs] = (~to_k).to(rw.dtype)
                w_sheet[rk] = to_k.to(rw.dtype)
        L_dec = (rw * m_dec).sum() / B
        L_sheet = (rw * w_sheet).sum() / max(float(w_sheet.sum().item()), 1.0)
        # distinct at ties: the two smallest branches of every near-tied pair
        near = gap < eta
        i_near = near.nonzero(as_tuple=True)[0]
        L_dist = torch.zeros((), device=points.device, dtype=rw.dtype)
        cos_near = None
        if i_near.numel() > 0:
            r1, r2 = row_of[i_near, win[i_near]], row_of[i_near, second[i_near]]
            ok = (r1 >= 0) & (r2 >= 0)
            i_near, r1, r2 = i_near[ok], r1[ok], r2[ok]
            if i_near.numel() > 0:
                def cosine(u, v):
                    return (u * v).sum(1) / (u.norm(dim=1) * v.norm(dim=1) + 1e-12)
                cos_x, cos_y = cosine(cx[r1], cx[r2]), cosine(cy[r1], cy[r2])
                pen = 0.5 * (torch.relu(cos_x - s['gamma']) ** 2 + torch.relu(cos_y - s['gamma']) ** 2)
                L_dist = lw * pen.sum() / B
                cos_near = 0.5 * (cos_x + cos_y).detach()
        loss_n = s['w_dec'] * L_dec + s['w_sheet'] * L_sheet + s['w_dist'] * L_dist
        terms = {}
        last = getattr(head, '_last', {})
        if self.end_weight > 0 and last.get('rho_x') is not None:
            wrows = is_win.nonzero(as_tuple=True)[0]
            E0 = self.end_speed(sp_rows, 0)[wrows] * (1.0 + last['rho_x'][wrows])
            E1 = self.end_speed(sp_rows, 1)[wrows] * (1.0 + last['rho_y'][wrows])
            l_end = lw * ((torch.sqrt(E0) - 1) ** 2 + (torch.sqrt(E1) - 1) ** 2)
            loss_end = self.end_weight * l_end.mean()
            loss_n = loss_n + loss_end
            terms = {'End': loss_end.item(), 'Nend_med': torch.cat([E0, E1]).median().item()}
        with torch.no_grad():
            st = self.stats
            st['win'] += torch.bincount(win, minlength=nb).to(st['win'].dtype)
            st['res_sum'] += torch.bincount(rows_b, weights=r.detach(), minlength=nb).to(st['res_sum'].dtype)
            st['res_n'] += torch.bincount(rows_b, minlength=nb).to(st['res_n'].dtype)
            st['n_pairs'] += B
            st['n_rows'] += n_rows
            st['n_gap3tau'] += int((gap <= s['gap_mask'] * tau).sum())
            # duplicate statistics per unit pair, on (winner, second) where both are in band
            both = (row_of[torch.arange(B, device=points.device), second] >= 0)
            ib = both.nonzero(as_tuple=True)[0]
            if ib.numel() > 0:
                ku, ks = head.unit_of_branch(win[ib]), head.unit_of_branch(second[ib])
                a_, b_ = torch.minimum(ku, ks), torch.maximum(ku, ks)
                flat = a_ * head.n_units + b_
                st['shared'] += torch.bincount(flat, minlength=head.n_units ** 2).reshape_as(st['shared']).to(
                    st['shared'].dtype)
            if cos_near is not None:
                dup = cos_near > s['gamma']
                st['n_near'] += int(i_near.numel())
                st['n_near_dup'] += int(dup.sum())
                ku, ks = head.unit_of_branch(win[i_near]), head.unit_of_branch(second[i_near])
                a_, b_ = torch.minimum(ku, ks), torch.maximum(ku, ks)
                flat = (a_ * head.n_units + b_)[dup]
                if flat.numel() > 0:
                    st['dup'] += torch.bincount(flat, minlength=head.n_units ** 2).reshape_as(st['dup']).to(
                        st['dup'].dtype)
            self.last_terms = {'Eik': loss_n.item(), 'Dec': L_dec.item(), 'Sheet': L_sheet.item(),
                               'Dist': L_dist.item(), 'T_mean': Tmin.mean().item(),
                               'N_med': torch.cat([N0, N1]).median().item(), 'rows': n_rows / B,
                               'gap3tau': float((gap <= s['gap_mask'] * tau).float().mean()),
                               'near': float(near.float().mean()), 'tau': tau, **terms}
            if 'long_way' in last:
                self.last_terms['long_way'] = last['long_way'][is_win].float().mean().item()
        return beta * loss_n, loss_n, r
