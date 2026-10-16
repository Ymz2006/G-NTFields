"""Trainer of the multi-branch decoder that learns the cut: graph_lie's loop (the original metric NTFields
procedure: batch 2000, 5 batches per epoch, AdamW wd 0.5, lr 5e-4, accept / reject rollback) with the spec's
additions (section 6):

  * temperature schedule tau(epoch) (model_function_mb.Function.tau);
  * near-cut sampling: a fraction of every batch (spec 20 %) is replaced by the smallest-gap rows of a second
    batch (the pool), so the cut regions are seen often;
  * branch management every `check_every` epochs (100 epochs = 500 steps): usage u_k of every unit on a fixed
    held-out pool (fraction of pairs where one of its branches is the hard minimum); a detour unit with
    u_k < dead_frac for `patience` consecutive checks is revived (copy of the alive unit with the largest mean
    sheet residual in its band, relative weight noise, offset raised by revive_raise tau_ref, optimiser moments
    reset) or, after max_revive revivals, pruned (alive[k] = 0, permanent); two units that are duplicates
    (value within eta and gradient cosine above gamma) on more than merge_frac of the (winner, second) pairs where
    both are in band are merged by masking the one with lower usage. Unit 0 (direct head, goal cone) is never
    touched. Every check is appended to branch_log.json; the surviving count is reported at the end.

Management runs after an accepted epoch; the accept / reject rule may roll the weights back to one of the five
previous states (which include `alive`, a buffer), the counters are not rolled back (rejections are rare on the
obstacle scenes: 0 in the previous campaign). Checkpoints keep the graph_native contract, so the per-case evaluator
loads them with ``--kind graph_native``; config.json stores the spec dictionary.
"""
import json
import os
import pickle
import random
import time

import numpy as np
import torch

from models.graph_branch import model_train_branch as mtb
from models.graph_lie.model_train_lie import warp_speeds
from models.graph_native import model_network_metric as model_network
from models.graph_native.model_train_metric import FastTensorDataLoader, load_dataset

from . import model_function_mb as model_function


class Model(mtb.Model):
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        super().__init__(ModelPath, DataPath, dim, source, device)
        self.Params['Training']['Spec'] = {}

    def _build(self, B):
        dev = self.Params['Device']
        P = self.Params['Training']
        self.B = B
        self.network = model_network.NN(dev, self.dim, B, graph=self.Params['Network']['Graph'])
        self.network.to(dev)
        self.network.float()
        self.function = model_function.Function(self.folder, dev, self.network, self.dim,
                                                grad_mode=P['Grad mode'], speed_model=P['Speed model'],
                                                weight_mode=P.get('Weight', 'exp'),
                                                end_weight=P.get('End weight', 0.0), planar=P.get('Planar', False),
                                                spec=P.get('Spec', {}), epochs=P['Number of Epochs'])

    def _log(self, msg):
        if msg.startswith('Epoch'):
            lt = getattr(self.function, 'last_terms', {})
            extra = ['{} = {:.3g}'.format(k, lt[k]) for k in ('Dec', 'Sheet', 'Dist', 'End', 'Nend_med', 'rows',
                                                              'gap3tau', 'near', 'tau', 'long_way') if k in lt]
            if extra:
                msg = msg + ' -- ' + ' -- '.join(extra)
        print(msg, flush=True)
        with open(os.path.join(self.folder, 'train_log.txt'), 'a') as fh:
            fh.write(msg + '\n')

    # ------------------------------------------------------------------ branch management
    @torch.no_grad()
    def _usage(self):
        head = self.network.net
        _, _, V, _, order = self.function.branch_table(self.pool)
        win = order[:, 0]
        ub = torch.bincount(win, minlength=head.n_branches).float() / win.shape[0]
        uu = torch.zeros(head.n_units, device=ub.device).index_add_(0, head.unit_index, ub)
        return ub.cpu().numpy(), uu.cpu().numpy()

    def _manage(self, epoch):
        head, f, s = self.network.net, self.function, self.function.spec
        st = f.stats
        ub, uu = self._usage()
        res = (st['res_sum'] / st['res_n'].clamp_min(1)).cpu().numpy()
        res_n = st['res_n'].cpu().numpy()
        unit_res = np.array([(res[head.unit_index.cpu().numpy() == k] * res_n[head.unit_index.cpu().numpy() == k]).sum()
                             / max(res_n[head.unit_index.cpu().numpy() == k].sum(), 1) for k in range(head.n_units)])
        events = []
        alive = head.alive_units()
        for k in alive:
            if k in head.direct_units:
                continue
            if uu[k] < s['dead_frac']:
                self.streak[k] += 1
            else:
                self.streak[k] = 0
            if self.streak[k] >= s['patience']:
                if self.n_revived[k] >= s['max_revive']:
                    head.alive[k] = 0.0
                    events.append({'event': 'prune', 'unit': k, 'usage': float(uu[k])})
                else:
                    cands = [j for j in head.alive_units() if j != k]
                    src = max(cands, key=lambda j: unit_res[j]) if cands else None
                    if src is not None:
                        ev = {'event': 'revive', 'unit': k, 'from': src, 'usage': float(uu[k]),
                              'src_residual': float(unit_res[src])}
                        if s['split_revive']:
                            head.copy_unit(k, src, s['split_noise'], 0.0)
                            v = torch.randn(6, device=head.alive.device)
                            v = v / v.norm()
                            f.splits[k] = (src, v, epoch + int(s['warm_epochs']))
                            ev.update({'split': True, 'v': [round(float(x), 3) for x in v],
                                       'until': epoch + int(s['warm_epochs'])})
                        else:
                            head.copy_unit(k, src, s['revive_noise'], s['revive_raise'] * s['tau_ref'])
                        head.reset_optimizer_unit(self.optimizer, k)
                        self.n_revived[k] += 1
                        events.append(ev)
                self.streak[k] = 0
        # proactive split of the least-used detour unit (split_every > 0, annealing phase only)
        if s['split_every'] and epoch % int(s['split_every']) == 0 and epoch <= s['anneal_frac'] * f.epochs:
            detours = [k for k in range(head.n_units) if k not in head.direct_units and k not in f.splits]
            if detours:
                k = min(detours, key=lambda j: uu[j])
                cands = [j for j in head.alive_units() if j != k]
                if cands:
                    src = max(cands, key=lambda j: unit_res[j])
                    head.copy_unit(k, src, s['split_noise'], 0.0)
                    v = torch.randn(6, device=head.alive.device)
                    v = v / v.norm()
                    f.splits[k] = (src, v, epoch + int(s['warm_epochs']))
                    head.reset_optimizer_unit(self.optimizer, k)
                    self.streak[k] = 0
                    events.append({'event': 'split', 'unit': k, 'from': src, 'usage': float(uu[k]),
                                   'src_residual': float(unit_res[src]), 'v': [round(float(x), 3) for x in v],
                                   'until': epoch + int(s['warm_epochs'])})
        # merge duplicates
        shared, dup = st['shared'].cpu().numpy(), st['dup'].cpu().numpy()
        for a in head.alive_units():
            for b in head.alive_units():
                if b <= a:
                    continue
                if shared[a, b] >= s['merge_min_shared'] and dup[a, b] / shared[a, b] > s['merge_frac']:
                    victim = b if (a in head.direct_units or uu[a] >= uu[b]) else a
                    if victim in head.direct_units or head.alive[victim] < 0.5:
                        continue
                    head.alive[victim] = 0.0
                    events.append({'event': 'merge', 'units': [a, b], 'masked': victim,
                                   'dup_frac': float(dup[a, b] / shared[a, b]), 'shared': int(shared[a, b])})
        self.last_events = events
        rec = {'epoch': epoch, 'tau': f.tau(epoch), 'usage_branch': [round(float(x), 4) for x in ub],
               'usage_unit': [round(float(x), 4) for x in uu], 'sheet_res_branch': [float(x) for x in res],
               'rows_per_pair': st['n_rows'] / max(st['n_pairs'], 1),
               'frac_gap_below_3tau': st['n_gap3tau'] / max(st['n_pairs'], 1),
               'near_tie_frac': st['n_near'] / max(st['n_pairs'], 1),
               'near_tie_dup_frac': st['n_near_dup'] / max(st['n_near'], 1),
               'alive': [int(x) for x in head.alive.cpu().numpy()], 'events': events}
        self.branch_log.append(rec)
        with open(os.path.join(self.folder, 'branch_log.json'), 'w') as fh:
            json.dump(self.branch_log, fh, indent=1)
        self._log('Check epoch {} -- usage {} -- alive {} -- rows/pair {:.2f} -- near {:.3f} (dup {:.2f}) -- {}'.format(
            epoch, ' '.join('{:.3f}'.format(x) for x in uu), rec['alive'], rec['rows_per_pair'],
            rec['near_tie_frac'], rec['near_tie_dup_frac'], events if events else 'no events'))
        f.reset_stats()

    # ------------------------------------------------------------------ train
    def train(self):
        os.makedirs(self.folder, exist_ok=True)
        P = self.Params['Training']
        seed = int(P['Seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.Params['Network']['Graph'].setdefault('fourier_seed', seed)
        B = torch.normal(0, 1, size=(128, self.dim))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
        self._build(B)
        P['Spec'] = dict(self.function.spec)
        with open(os.path.join(self.folder, 'config.json'), 'w') as fh:
            json.dump(self.Params, fh, indent=1)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=P['Learning Rate'],
                                           weight_decay=P['Weight decay'])
        data = load_dataset(self.Params['DataPath'])
        dataloader = FastTensorDataLoader(data, batch_size=int(P['Batch Size']), shuffle=True)
        s = self.function.spec
        dev = self.Params['Device']
        gpool = torch.Generator().manual_seed(12345)
        self.pool = data[torch.randperm(data.shape[0], generator=gpool)[:int(s['pool_size'])], :2 * self.dim].to(dev)
        head = self.network.net
        self.streak = [0] * head.n_units
        self.n_revived = [0] * head.n_units
        self.branch_log = []
        self.last_events = []
        n_near = int(round(s['near_cut_frac'] * int(P['Batch Size'])))
        self._log('dataset {} rows; params {}; grad {}; speed model {}; form {}; units {}; branches {}; spec {}'.format(
            data.shape[0], sum(p.numel() for p in self.network.parameters() if p.requires_grad), P['Grad mode'],
            P['Speed model'], head.FORM, head.n_units, head.n_branches, json.dumps(s)))
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1000.0
        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        prev_state_queue, prev_optimizer_queue = [], []
        n_reject = 0
        t_start = time.time()
        for epoch in range(1, P['Number of Epochs'] + 1):
            alpha = P['Alpha']
            gamma = 0.001
            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            if len(prev_state_queue) > 5:
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            self.optimizer.param_groups[0]['lr'] = 5e-4
            prev_diff = current_diff
            terms = {}
            while True:
                total_train_loss = 0
                total_diff = 0
                terms = {}
                ii = 0
                it = iter(dataloader)
                for wholedata in it:
                    if ii > 4:
                        break
                    ii = ii + 1
                    d = wholedata[0].to(dev)
                    if n_near > 0:
                        try:
                            pool_d = next(it)[0].to(dev)
                        except StopIteration:
                            pool_d = None
                        if pool_d is not None and pool_d.shape[0] >= n_near:
                            sel = self.function.near_cut_indices(pool_d[:, :2 * self.dim], n_near)
                            d = torch.cat([d[:d.shape[0] - n_near], pool_d[sel]], 0)
                    points = d[:, :2 * self.dim]
                    speeds = warp_speeds(d, self.dim, alpha)
                    loss_value, loss_n, wv = self.function.Loss(points, speeds, beta, gamma, epoch)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_train_loss += loss_value.item()
                    total_diff += loss_n.item()
                    for k, v in self.function.last_terms.items():
                        terms[k] = terms.get(k, 0.0) + v / 4
                    del points, speeds, loss_value, loss_n
                total_train_loss /= 4
                total_diff /= 4
                current_diff = total_diff
                diff_ratio = current_diff / prev_diff
                if diff_ratio < 1.2 and diff_ratio > 0:
                    break
                n_reject += 1
                with torch.no_grad():
                    random_number = random.randint(0, len(prev_state_queue) - 1)
                    self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                    self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
                self._log('RepeatEpoch = {} -- Loss = {:.4e}'.format(epoch, total_diff))
                if not np.isfinite(total_diff):
                    self._log('non-finite loss; aborting')
                    return
            self.total_train_loss.append(total_train_loss)
            beta = 1.0 / total_diff
            if epoch % P['Print Every * Epoch'] == 0 or epoch == 1:
                self._log('Epoch = {} -- Loss = {:.4e} -- T_mean = {:.3f} -- N_med = {:.3f} -- rejects = {} -- {:.0f}s'.format(
                    epoch, total_diff, terms.get('T_mean', float('nan')), terms.get('N_med', float('nan')), n_reject,
                    time.time() - t_start))
            if epoch % int(s['check_every']) == 0:
                self._manage(epoch)
                if self.last_events:
                    # a revival / split / prune / merge changes the loss (and its decomposition) by construction:
                    # restart the accept / reject reference and the rollback queue from the new state, otherwise
                    # the next epoch is rejected forever and rolled back to a state the management no longer matches
                    current_diff = 1000.0
                    current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
                    current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
                    prev_state_queue.clear()
                    prev_optimizer_queue.clear()
            if epoch % 100 == 0 or epoch == P['Number of Epochs']:
                self.save(epoch=epoch, val_loss=total_diff, name='latest.pt')
        alive = head.alive_units()
        self._log('surviving units: {} of {} ({}); revivals {}; branch usage {}'.format(
            len(alive), head.n_units, alive, self.n_revived,
            self.branch_log[-1]['usage_unit'] if self.branch_log else '-'))
        with open(os.path.join(self.folder, 'train_summary.json'), 'w') as fh:
            json.dump({'wall_time_s': time.time() - t_start, 'rejects': n_reject, 'final_loss': total_diff,
                       'epochs': P['Number of Epochs'], 'grad_mode': P['Grad mode'],
                       'speed_model': P['Speed model'], 'alive_units': alive, 'n_units': head.n_units,
                       'revivals': self.n_revived,
                       'usage_unit': self.branch_log[-1]['usage_unit'] if self.branch_log else None,
                       'form': head.FORM}, fh, indent=1)
