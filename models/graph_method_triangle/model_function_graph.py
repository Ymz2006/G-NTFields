"""Loss / query object for the SE(3) smooth-graph field (repo `Function` style).

Training modes
--------------
physics     L = w_eik L_E + w_bell L_B + w_goal L_goal + w_anchor L_T   (spec page 4)
            L_E on input-geometry-regular pairs only, L_B on every pair with an
            EMA target network, L_goal on a separate diagonal batch through
            Q = T^2, L_T only on the declared fixed anchor set.
supervised  L = L_Z + L_T + w_goal L_goal                               (stage B)
ntfields    the original NTFields metric loss of models/metric (sqrt-form
            Eikonal + one-step TD along -grad, weighted by exp(-0.5 T)), with
            coordinate gradients replaced by the intrinsic scaled tangent
            gradients (baseline training recipe; no oracle labels).
"""

import copy
import math

import torch

from .data.poses import PoseSampler
from .fields import decoder, oracle
from .geometry import se3, tangents
from .losses import bellman, eikonal
from . import model_network_graph as nets


def default_config(**overrides):
    cfg = dict(
        name='run', arch='graph_gram', mode='physics', seed=0, device='cuda:0', dtype='float32',
        # motion model / domain
        s_t=1.0, s_r=1.0, box=1.0, L_ref=1.0,
        # network (Fourier sigma_t / sigma_r override sigma for the t / rotation columns)
        h_size=256, nl1=2, fourier_sigma=0.5, fourier_seed=None, fourier_sigma_t=None,
        fourier_sigma_r=None, fourier_trunc='clamp', feature_dim=12, k_a=16, gram_k=4, k_b=16,
        normalize=False, skip_input=False, metric_scale=0.2, a_scale=1.0, b_soft=0.02, gram_power=1,
        group=16, lp_p=8.0,  # graph_tri_metric / graph_tri_mix: l_p norm over groups of `group` coords
        rot_gain=1.0, rot_bias=0.0,  # graph_tri*: init of the 6-d rotation output (W *= gain, bias e1/e2)
        rot_mode='free',  # graph_tri*: V = Q(x) ('free') or V = R_x Q(x) ('equivariant')
        tri_switch=0,  # graph_gram_tri: relaxed Gram b before this step, exact triangle head after
        lse_beta=10.0,  # graph_metric: logsumexp temperature of the group soft max (G-Shape: 10)
        init_from=None,
        # optimisation (ntf_*: optional NTFields accept/reject epoch rollback)
        optimizer='adam', lr=1e-3, weight_decay=0.0, lr_schedule='none', lr_min_frac=0.05,
        batch_size=256, diag_batch=64, steps=20000, clip=1.0, ema=0.995, val_every=500,
        ntf_rollback=False, ntf_epoch=5, ntf_ratio=1.2,
        # loss weights
        w_eik=1.0, w_bell=1.0, w_goal=10.0, w_anchor=1.0, n_anchor=1024, anchor_seed=777,
        w_latent=1.0, eik_form='sq', eik_weight_pow=0.0, eik_exp_weight=0.0, eik_exp_weight_grad=False,
        # bellman
        bell_h=0.05, bell_k_rand=16, bell_k_pert=15, bell_sigma=0.3, bell_terminal=True,
        bell_greedy='online',
        # relative-angle curriculum (input geometry only): max angle grows from
        # curriculum_start_deg to 180 over curriculum_steps updates (0 = off)
        curriculum_steps=0, curriculum_start_deg=60.0,
        # ntfields baseline recipe
        nt_h=0.02, nt_eik_w=1e-2, nt_td_w=1e-3,
        # regular-sample policy
        diag_band=1e-3, pi_band=1e-4,
        # sampler
        mix=(0.6, 0.2, 0.2), small_min=1e-3, small_max=0.3, pi_delta_min=1e-3, pi_delta_max=0.5,
        # held-out
        val_seed=1001, test_seed=2002, n_val=10000,
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError('unknown config keys: {}'.format(sorted(unknown)))
    cfg.update(overrides)
    if cfg['fourier_seed'] is None:
        cfg['fourier_seed'] = cfg['seed']
    return cfg


def network_kwargs(cfg):
    keys = ('h_size', 'nl1', 'fourier_sigma', 'fourier_seed', 'fourier_sigma_t', 'fourier_sigma_r',
            'feature_dim', 'fourier_trunc', 'k_a', 'gram_k', 'k_b', 'normalize', 'skip_input',
            'metric_scale', 'a_scale', 'b_soft', 'gram_power', 'group', 'lp_p', 'rot_gain', 'rot_bias',
            'rot_mode', 'tri_switch', 'lse_beta')
    return {k: cfg[k] for k in keys}


class Function:
    def __init__(self, cfg, network=None):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.dtype = torch.float64 if cfg['dtype'] == 'float64' else torch.float32
        self.s_t, self.s_r = float(cfg['s_t']), float(cfg['s_r'])
        self.tau_ref = cfg['L_ref'] / cfg['s_t']
        if network is None:
            torch.manual_seed(cfg['seed'])
            network = nets.build_network(cfg['arch'], s_t=self.s_t, s_r=self.s_r, **network_kwargs(cfg))
        self.network = network.to(self.device, self.dtype)
        if cfg.get('init_from'):
            # warm start (e.g. from a stage-B supervised checkpoint); such runs
            # are labelled separately from training from scratch
            ck = torch.load(cfg['init_from'], map_location=self.device)
            self.network.load_state_dict(ck['model_state_dict'])
        self.target = copy.deepcopy(self.network)
        for p in self.target.parameters():
            p.requires_grad_(False)
        skw = dict(box=cfg['box'], mix=tuple(cfg['mix']), s_t=self.s_t, s_r=self.s_r,
                   small_min=cfg['small_min'], small_max=cfg['small_max'],
                   pi_delta_min=cfg['pi_delta_min'], pi_delta_max=cfg['pi_delta_max'])
        self.sampler_kw = skw
        self.sampler = PoseSampler(seed=cfg['seed'], device=self.device, dtype=self.dtype, **skw)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(cfg['seed'] + 100003)
        self._pool, self._pool_pos = None, 0
        self.anchor = None
        if cfg['mode'] in ('physics', 'ntfields') and cfg['n_anchor'] > 0 and cfg['w_anchor'] > 0:
            # Declared anchor set: fixed regular pairs with oracle times.
            s = PoseSampler(seed=cfg['anchor_seed'], device=self.device, dtype=torch.float64, **skw)
            tx, Rx, ty, Ry, _ = s.training_pairs(4 * cfg['n_anchor'])
            m = eikonal.regular_mask(tx, Rx, ty, Ry, self.s_t, self.s_r, self.tau_ref,
                                     cfg['diag_band'], cfg['pi_band'])
            idx = m.nonzero().squeeze(-1)[:cfg['n_anchor']]
            T = oracle.time_star(tx[idx], Rx[idx], ty[idx], Ry[idx], self.s_t, self.s_r)
            self.anchor = (se3.features(tx[idx], Rx[idx]).to(self.dtype),
                           se3.features(ty[idx], Ry[idx]).to(self.dtype), T.to(self.dtype))
            self.anchor_angle = oracle.relative_angle(Rx[idx], Ry[idx]).to(self.dtype)

    # ------------------------------------------------------------------ utils
    def max_angle(self, step):
        c = self.cfg
        if not c['curriculum_steps'] or step is None:
            return math.pi
        frac = min(1.0, step / c['curriculum_steps'])
        return math.radians(c['curriculum_start_deg'] + frac * (180.0 - c['curriculum_start_deg'])) + 1e-6

    def next_batch(self, n, pool_batches=200, step=None):
        """Training pairs drawn from a pre-generated pool (same distribution,
        far fewer kernel launches). Deterministic given the sampler seed.
        Under the angle curriculum, pairs above the current max relative angle
        are skipped."""
        th_max = self.max_angle(step)
        out, have = [], 0
        while have < n:
            if self._pool is None or self._pool_pos >= self._pool[0].shape[0]:
                pool = self.sampler.training_pairs(n * pool_batches)
                perm = torch.randperm(n * pool_batches, device=self.device, generator=self.gen)
                pool = tuple(v[perm] for v in pool)
                ang = oracle.relative_angle(pool[1], pool[3])
                self._pool = pool + (ang,)
                self._pool_pos = 0
            i = self._pool_pos
            chunk = tuple(v[i:i + 2 * n] for v in self._pool)
            self._pool_pos += 2 * n
            if th_max < math.pi:
                keep = chunk[-1] <= th_max
                chunk = tuple(v[keep] for v in chunk)
            out.append(chunk[:-1])
            have += chunk[0].shape[0]
        return tuple(torch.cat([o[k] for o in out])[:n] for k in range(5))

    @torch.no_grad()
    def update_target(self):
        d = self.cfg['ema']
        for pt, p in zip(self.target.parameters(), self.network.parameters()):
            pt.mul_(d).add_(p.detach(), alpha=1 - d)
        for bt, b in zip(self.target.buffers(), self.network.buffers()):
            bt.copy_(b)

    def _greedy(self, model, gx, gy):
        """-c_x/|c_x| from `model` (no graph); non-finite rows -> random."""
        with torch.enable_grad():
            _, cx, _ = tangents.tangent_gradients(model.value, gx, gy, self.s_t, self.s_r,
                                                  create_graph=False)
        cx = cx.detach()
        n = torch.linalg.norm(cx, dim=-1, keepdim=True)
        u = -cx / n.clamp_min(1e-12)
        bad = ~torch.isfinite(u).all(-1) | (n[:, 0] < 1e-8)
        if bad.any():
            r = torch.randn(int(bad.sum()), 6, device=gx.device, dtype=gx.dtype, generator=self.gen)
            u[bad] = r / torch.linalg.norm(r, dim=-1, keepdim=True)
        return u, int(bad.sum())

    def _goal_loss(self):
        td, Rd = self.sampler.uniform_poses(self.cfg['diag_batch'])
        gd = se3.features(td, Rd)
        return self.network.value_sq(gd, gd).mean() / self.tau_ref ** 2

    def _anchor_loss(self):
        if self.anchor is None:
            return None
        gx, gy, T = self.anchor
        return ((self.network.value(gx, gy) - T) ** 2).mean() / self.tau_ref ** 2

    def set_phase(self, step):
        """graph_gram_tri: relaxed head before tri_switch, exact head from then on."""
        for m in (self.network, self.target):
            if hasattr(m, 'tri_switch'):
                m.exact = step is None or step >= m.tri_switch

    # ------------------------------------------------------------------ losses
    def loss(self, step):
        self.set_phase(step)
        mode = self.cfg['mode']
        if mode == 'physics':
            return self._physics_loss(step)
        if mode == 'supervised':
            return self._supervised_loss(step)
        if mode == 'ntfields':
            return self._ntfields_loss(step)
        raise ValueError(mode)

    def _physics_loss(self, step):
        c = self.cfg
        net = self.network
        tx, Rx, ty, Ry, fam = self.next_batch(c['batch_size'], step=step)
        gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
        logs = {}
        total = 0.0
        B = gx.shape[0]
        th_max = self.max_angle(step)
        m = eikonal.regular_mask(tx, Rx, ty, Ry, self.s_t, self.s_r, self.tau_ref,
                                 c['diag_band'], c['pi_band'])
        u = None
        if c['w_eik'] > 0:
            # derivative-dependent evaluations on regular pairs only
            Te, cx, cy = eikonal.eikonal_terms(net, gx[m], gy[m], create_graph=True)
            res = eikonal.endpoint_residual(cx, cy, c['eik_form'])
            if c['eik_weight_pow'] != 0:
                # emphasize far pairs (slow PINN propagation); weights use the
                # detached prediction, never oracle labels
                w = (Te.detach() / self.tau_ref).clamp_min(1e-3) ** c['eik_weight_pow']
                res = res * (w / w.mean())
            if c['eik_exp_weight'] > 0:
                # NTFields exp(-k T) residual weighting; with _grad the weight is not
                # detached, which also pushes T up where the residual is large
                Tw = Te if c['eik_exp_weight_grad'] else Te.detach()
                res = res * torch.exp(-c['eik_exp_weight'] * Tw / self.tau_ref)
            LE = res.mean()
            total = total + c['w_eik'] * LE
            logs['L_E'] = LE.detach()
            logs['frac_regular'] = m.float().mean()
            if c['bell_greedy'] == 'online':
                cxd = cx.detach()
                n = torch.linalg.norm(cxd, dim=-1, keepdim=True)
                u = torch.randn(B, 6, device=gx.device, dtype=gx.dtype, generator=self.gen)
                u = u / torch.linalg.norm(u, dim=-1, keepdim=True)
                ug = -cxd / n.clamp_min(1e-12)
                ok = torch.isfinite(ug).all(-1) & (n[:, 0] > 1e-8)
                u[m.nonzero().squeeze(-1)[ok]] = ug[ok]
        # one plain forward for: Bellman pairs, diagonal (goal) pairs, anchors
        td, Rd = self.sampler.uniform_poses(c['diag_batch']) if c['w_goal'] > 0 else (tx[:0], Rx[:0])
        gd = se3.features(td, Rd)
        parts_x, parts_y = [gx, gd], [gy, gd]
        if self.anchor is not None:
            parts_x.append(self.anchor[0]); parts_y.append(self.anchor[1])
        GX, GY = torch.cat(parts_x), torch.cat(parts_y)
        if net.has_latent:
            a, b = net.predict_latent(GX, GY)
            Q = decoder.decode_Q(a, b, self.s_t, self.s_r)
        else:
            Tall = net.value(GX, GY)
            Q = None
        nd = gd.shape[0]
        sl_b, sl_d, sl_a = slice(0, B), slice(B, B + nd), slice(B + nd, None)
        val = (lambda sl: torch.sqrt(Q[sl])) if Q is not None else (lambda sl: Tall[sl])
        if c['w_bell'] > 0:
            T = val(sl_b)
            if u is None:
                greedy_model = self.target if c['bell_greedy'] == 'target' else net
                u, _ = self._greedy(greedy_model, gx, gy)
            U = bellman.candidate_controls(u, c['bell_k_rand'], c['bell_k_pert'], c['bell_sigma'], self.gen)
            tgt, idx, reached = bellman.bellman_target(self.target, tx, Rx, ty, Ry, U, c['bell_h'],
                                                       self.s_t, self.s_r, terminal=c['bell_terminal'])
            LB = ((T - tgt) ** 2).mean() / self.tau_ref ** 2
            total = total + c['w_bell'] * LB
            logs['L_B'] = LB.detach()
            logs['bell_reached'] = reached.float().mean()
            logs['bell_greedy_pick'] = (idx == 0).float().mean()
        if c['w_goal'] > 0:
            Qd = Q[sl_d] if Q is not None else Tall[sl_d] ** 2
            Lg = Qd.mean() / self.tau_ref ** 2
            total = total + c['w_goal'] * Lg
            logs['L_goal'] = Lg.detach()
        if self.anchor is not None:
            err2 = (val(sl_a) - self.anchor[2]) ** 2
            if th_max < math.pi:
                am = (self.anchor_angle <= th_max).to(err2.dtype)
                La = (err2 * am).sum() / am.sum().clamp_min(1.0) / self.tau_ref ** 2
            else:
                La = err2.mean() / self.tau_ref ** 2
            total = total + c['w_anchor'] * La
            logs['L_T_anchor'] = La.detach()
        return total, logs

    def _supervised_loss(self, step):
        c = self.cfg
        tx, Rx, ty, Ry, fam = self.next_batch(c['batch_size'])
        gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
        T_star = oracle.time_star(tx, Rx, ty, Ry, self.s_t, self.s_r)
        logs = {}
        net = self.network
        if net.has_latent:
            a, b = net.predict_latent(gx, gy)
            a_s, b_s = oracle.latent_star(tx, Rx, ty, Ry)
            b_s = b_s.clamp(0, 1)
            LZ = ((a - a_s) ** 2 / self.cfg['L_ref'] ** 4 + (b - b_s) ** 2).mean()
            T = net.decode(a, b)
            logs['L_Z'] = LZ.detach()
        else:
            LZ = 0.0
            T = net.value(gx, gy)
        LT = ((T - T_star) ** 2).mean() / self.tau_ref ** 2
        Lg = self._goal_loss()
        logs.update(L_T=LT.detach(), L_goal=Lg.detach())
        return c['w_latent'] * LZ + LT + c['w_goal'] * Lg, logs

    def _ntfields_loss(self, step):
        """models/metric Function.Loss with intrinsic gradients (speed = 1)."""
        c = self.cfg
        tx, Rx, ty, Ry, fam = self.next_batch(c['batch_size'])
        m = eikonal.regular_mask(tx, Rx, ty, Ry, self.s_t, self.s_r, self.tau_ref,
                                 c['diag_band'], c['pi_band'])
        tx, Rx, ty, Ry = tx[m], Rx[m], ty[m], Ry[m]
        gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
        T, cx, cy = eikonal.eikonal_terms(self.network, gx, gy, create_graph=True)
        nx = torch.sqrt((cx * cx).sum(-1) + 1e-8)
        ny = torch.sqrt((cy * cy).sum(-1) + 1e-8)
        eik = c['nt_eik_w'] * ((torch.sqrt(nx) - 1) ** 2 + (torch.sqrt(ny) - 1) ** 2)
        h = c['nt_h']
        sc = bellman.scale_vector(self.s_t, self.s_r, cx)
        with torch.no_grad():
            tx0, Rx0 = se3.right_perturb(tx, Rx, -h * cx.detach() * sc)
            T0 = self.network.value(se3.features(tx0, Rx0), gy) + h
            ty1, Ry1 = se3.right_perturb(ty, Ry, -h * cy.detach() * sc)
            T1 = self.network.value(gx, se3.features(ty1, Ry1)) + h
        td0 = c['nt_td_w'] * (T - T0) ** 2
        td1 = c['nt_td_w'] * (T - T1) ** 2
        td0 = torch.where(T < h, torch.zeros_like(td0), td0)
        td1 = torch.where(T < h, torch.zeros_like(td1), td1)
        L = ((eik + td0 + td1) * torch.exp(-0.5 * T)).mean()
        logs = {'L_nt_eik': eik.mean().detach(), 'L_nt_td': (td0 + td1).mean().detach()}
        La = self._anchor_loss()
        if La is not None:
            L = L + c['w_anchor'] * La
            logs['L_T_anchor'] = La.detach()
        return L, logs

    # ------------------------------------------------------------------ queries
    def TravelTimes(self, gx, gy, chunk=65536):
        out = []
        for i in range(0, gx.shape[0], chunk):
            out.append(self.network.value(gx[i:i + chunk], gy[i:i + chunk]))
        return torch.cat(out)

    def Gradient(self, gx, gy):
        """Scaled tangent gradients (c_x, c_y) of the decoded time."""
        T, cx, cy = tangents.tangent_gradients(self.network.value, gx, gy, self.s_t, self.s_r,
                                               create_graph=False)
        return T.detach(), cx.detach(), cy.detach()
