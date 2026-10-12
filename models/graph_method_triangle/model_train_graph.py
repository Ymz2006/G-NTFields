"""Training driver for the SE(3) smooth-graph field (repo `Model` style).

Checkpoints store architecture metadata, network / EMA-target / optimizer
states, scales, sampler and RNG states, code revision, dtype, exclusions,
loss weights and training mode (spec page 8).  The best checkpoint is chosen
on held-out validation value error, never on the training Eikonal residual.
"""

import copy
import json
import os
import shutil
import subprocess
import time

import torch

from .data.poses import build_heldout_suites, suites_to
from .fields import oracle
from .geometry import se3
from . import model_function_graph as mf
from . import model_network_graph as nets


def git_revision():
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(['git', 'diff', '--quiet'], stderr=subprocess.DEVNULL) != 0
        return rev + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


def relative_errors(T, T_star, tau_ref, min_rel_time=0.05):
    m = T_star >= min_rel_time * tau_ref
    rel = ((T - T_star).abs() / T_star.clamp_min(1e-12))[m]
    return rel


@torch.no_grad()
def quick_validation(function, suites):
    out = {}
    net = function.network
    net.eval()
    for key in ('uniform', 'mixture', 'exact_pi', 'near_diag'):
        s = suites[key]
        gx, gy = se3.features(s['tx'], s['Rx']), se3.features(s['ty'], s['Ry'])
        T = function.TravelTimes(gx, gy).double()
        T_star = oracle.time_star(s['tx'].double(), s['Rx'].double(), s['ty'].double(),
                                  s['Ry'].double(), function.s_t, function.s_r)
        err = (T - T_star).abs()
        out[key + '_abs_mean'] = err.mean().item()
        if key in ('uniform', 'mixture'):
            rel = relative_errors(T, T_star, function.tau_ref)
            out[key + '_rel_p50'] = rel.median().item()
            out[key + '_rel_p95'] = rel.quantile(0.95).item()
        out[key + '_nonfinite'] = int((~torch.isfinite(T)).sum())
    net.train()
    out['score'] = (out['uniform_rel_p50'] + out['uniform_rel_p95'] +
                    out['mixture_rel_p50'] + out['mixture_rel_p95'])
    return out


class Model:
    def __init__(self, cfg, out_root='./Experiments/graph_method_triangle'):
        self.cfg = cfg
        self.folder = os.path.join(out_root, cfg['name'])
        self.history = []

    # ---------------------------------------------------------------- setup
    def _snapshot_code(self):
        src = os.path.dirname(os.path.abspath(__file__))
        dst = os.path.join(self.folder, 'models_snapshot')
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__'))

    def _make_optimizer(self):
        c = self.cfg
        params = self.function.network.parameters()
        if c['optimizer'] == 'adamw':
            opt = torch.optim.AdamW(params, lr=c['lr'], weight_decay=c['weight_decay'])
        else:
            opt = torch.optim.Adam(params, lr=c['lr'], weight_decay=c['weight_decay'])
        if c['lr_schedule'] == 'cosine':
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['steps'], eta_min=c['lr'] * c['lr_min_frac'])
        else:
            sched = None
        return opt, sched

    # ---------------------------------------------------------------- train
    def train(self):
        c = self.cfg
        os.makedirs(self.folder, exist_ok=True)
        self._snapshot_code()
        with open(os.path.join(self.folder, 'config.json'), 'w') as f:
            json.dump(c, f, indent=2)
        torch.manual_seed(c['seed'])
        shape = c['mode'] == 'shape'
        if shape:
            from .model_function_shape import ShapeFunction
            self.function = ShapeFunction(c)
        else:
            self.function = mf.Function(c)
        net = self.function.network
        self.optimizer, self.scheduler = self._make_optimizer()
        dev = self.function.device
        suites = None if shape else suites_to(build_heldout_suites(
            c['val_seed'], n_main=c['n_val'], n_suite=1000, **self.function.sampler_kw), dev, self.function.dtype)
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(dev)
        n_params = nets.count_parameters(net)
        print('[{}] arch={} mode={} params={} anchors={}'.format(
            c['name'], c['arch'], c['mode'], n_params,
            0 if self.function.anchor is None else self.function.anchor[0].shape[0]), flush=True)
        best = float('inf')
        skipped = 0
        t_start = time.time()
        running = {}
        # optional NTFields accept/reject loop (models/metric Model.train): every
        # `ntf_epoch` updates form an epoch; an epoch whose mean loss exceeds
        # ntf_ratio x the previous accepted epoch is rolled back to a random one of
        # the last 5 epoch-start snapshots and repeated.
        rollback = bool(c.get('ntf_rollback', False))
        E = int(c.get('ntf_epoch', 5))
        assert not (rollback and self.scheduler is not None), 'rollback assumes a constant LR'
        snaps, ep_losses, prev_ep, n_rollbacks, retries = [], [], None, 0, 0
        step = 0
        while step < c['steps']:
            step += 1
            if rollback and (step - 1) % E == 0:
                snaps.append((copy.deepcopy(net.state_dict()), copy.deepcopy(self.optimizer.state_dict()),
                              copy.deepcopy(self.function.target.state_dict())))
                snaps = snaps[-5:]
            loss, logs = self.function.loss(step)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), c['clip'])
            if not torch.isfinite(gnorm):  # one sync per step; skip non-finite updates
                skipped += 1
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.step()
                self.function.update_target()
            if self.scheduler is not None:
                self.scheduler.step()
            logs['grad_norm'] = gnorm.detach()
            logs['loss'] = loss.detach()
            for k, v in logs.items():
                running.setdefault(k, []).append(v)
            if rollback:
                ep_losses.append(loss.detach())
                if step % E == 0:
                    cur = torch.stack(ep_losses).mean().item()
                    ep_losses = []
                    if prev_ep is not None and not (0 < cur / prev_ep < c.get('ntf_ratio', 1.2)) and retries < 50:
                        sd, od, td = snaps[torch.randint(len(snaps), (1,)).item()]
                        net.load_state_dict(sd)
                        self.optimizer.load_state_dict(od)
                        self.function.target.load_state_dict(td)
                        step -= E
                        n_rollbacks += 1
                        retries += 1
                        continue
                    prev_ep, retries = cur, 0
            if step % c['val_every'] == 0 or step == c['steps']:
                if shape:  # no oracle among obstacles: held-out Eikonal residual
                    val = self.function.holdout_residual()
                    val['score'] = val['hold_eik_mean']
                    val.update({k: float('nan') for k in ('uniform_rel_p50', 'uniform_rel_p95', 'mixture_rel_p50',
                                                          'mixture_rel_p95', 'exact_pi_abs_mean')})
                else:
                    val = quick_validation(self.function, suites)
                rec = {'step': step, 'time': time.time() - t_start, 'skipped': skipped,
                       'lr': self.optimizer.param_groups[0]['lr']}
                rec.update({'train_' + k: float(torch.stack([torch.as_tensor(x, dtype=torch.float64).cpu() for x in v]).mean())
                            for k, v in running.items()})
                rec.update({'val_' + k: v for k, v in val.items()})
                running = {}
                self.history.append(rec)
                # graph_gram_tri: only exact-phase checkpoints may become best.pt
                improved = val['score'] < best and step >= getattr(net, 'tri_switch', 0)
                if improved:
                    best = val['score']
                    self.save('best.pt', step, val)
                self.save('latest.pt', step, val)
                if shape:
                    print('[{}] step {:6d} loss {:.3e} | hold eik p50 {:.4f} mean {:.4f} | T {:.3f} | {:.0f}s'.format(
                        c['name'], step, rec['train_loss'], val['hold_eik_p50'], val['hold_eik_mean'],
                        val['hold_T_mean'], rec['time']), flush=True)
                    with open(os.path.join(self.folder, 'history.json'), 'w') as f:
                        json.dump(self.history, f)
                    continue
                print('[{}] step {:6d} loss {:.3e} | val uni p50 {:.4f} p95 {:.4f} | mix p50 {:.4f} p95 {:.4f}'
                      ' | pi {:.4f} | {:.0f}s{}'.format(
                          c['name'], step, rec['train_loss'], val['uniform_rel_p50'], val['uniform_rel_p95'],
                          val['mixture_rel_p50'], val['mixture_rel_p95'], val['exact_pi_abs_mean'],
                          rec['time'], ' *' if improved else ''), flush=True)
                with open(os.path.join(self.folder, 'history.json'), 'w') as f:
                    json.dump(self.history, f)
        summary = {'wall_time_s': time.time() - t_start, 'skipped_steps': skipped, 'params': n_params,
                   'rollbacks': n_rollbacks,
                   'best_score': best,
                   'peak_mem_MB': (torch.cuda.max_memory_allocated(dev) / 2 ** 20) if dev.type == 'cuda' else None}
        with open(os.path.join(self.folder, 'train_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print('[{}] done {}'.format(c['name'], summary), flush=True)
        return summary

    # ---------------------------------------------------------------- io
    def save(self, fname, step, val=None):
        f = self.function
        ckpt = {
            'config': self.cfg, 'arch': self.cfg['arch'], 'step': step, 'val': val,
            'model_state_dict': f.network.state_dict(),
            'target_state_dict': f.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'sampler_state': f.sampler.state_dict(),
            'rng_state': {'torch': torch.get_rng_state(), 'gen': f.gen.get_state()},
            'scales': {'s_t': f.s_t, 's_r': f.s_r, 'L_ref': self.cfg['L_ref'], 'tau_ref': f.tau_ref},
            'exclusions': {'diag_band': self.cfg['diag_band'], 'pi_band': self.cfg['pi_band']},
            'code_revision': git_revision(), 'dtype': self.cfg['dtype'],
            'n_anchor': 0 if f.anchor is None else int(f.anchor[0].shape[0]),
        }
        torch.save(ckpt, os.path.join(self.folder, fname))


def load_function(path, device=None, use_target=False):
    """Rebuild a Function (network + scales) from a checkpoint for evaluation."""
    ckpt = torch.load(path, map_location='cpu')
    cfg = mf.default_config()          # backfill keys added after the checkpoint was written
    cfg.update(ckpt['config'])
    if device is not None:
        cfg['device'] = device
    cfg['n_anchor'] = 0  # anchors are a training-time construct
    f = mf.Function(cfg)
    f.network.load_state_dict(ckpt['target_state_dict' if use_target else 'model_state_dict'])
    f.network.eval()
    return f, ckpt
