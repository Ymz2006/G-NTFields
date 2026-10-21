"""Original metric NTFields training driver with the graph heads and the Lie-gradient loss.

A copy of models/graph_native/model_train_metric.py (itself identical to
Lie-Shape-NTFields/models/metric_baseline in every training detail: batch 2000 reshuffled each
epoch, 5 batches per epoch, losses divided by 4, AdamW with weight decay 0.5 and the learning
rate hard-set to 5e-4 inside the loop, beta = 1 / total_diff, accept / reject rollback to one of
the last five states when the loss ratio is >= 1.2). Changes:

* the loss is model_function_lie.Function (grad mode x speed model);
* all three clearance speeds are passed to it, each preprocessed as in its original trainer
  (warp_speeds): S as in metric_baseline / graph_native, S_d and S_a as in Lie-Shape-NTFields
  lie_metric / lie_isotropic;
* checkpoints additionally store 'speed_model' (they load with graph_native's Model.load, which
  is what the per-case evaluator uses for --kind graph_native).

The random-number sequence (seeding, network init, data order) is the one of graph_native, so a
run with speed 'combined' and grad 'chart' reproduces the graph_native run with the same seed.
"""
import importlib
import json
import os
import pickle
import random
import time

import numpy as np
import torch

from models.graph_native import model_network_metric as model_network
from models.graph_native.model_train_metric import FastTensorDataLoader, load_dataset

from . import model_function_lie as model_function

torch.backends.cudnn.benchmark = True

ANGLE_MAX = 1.0 / 6.0     # Lie-Shape-NTFields: rotational clearance (stored as angle / pi) is cut at
ANGLE_MIN = 1e-4          # pi / 6 and renormalised to [ANGLE_MIN / ANGLE_MAX, 1] before the warp


def warp_speeds(d, dim=6, alpha=1.0):
    """Rows of load_dataset -> dict of warped (B, 2) speeds.

    S  : combined speed (speed.npy = (S_d + angle / pi) / 2), warp s^2 (2 - s)^2, alpha stretch
         (graph_native / metric_baseline, alpha = 1)
    Sd : translational clearance speed (speed_dists.npy), warp, clamp >= 1e-3
    Sa : rotational clearance speed (speed_angles.npy = angle / pi), clamp(sa / (1/6), 6e-4, 1),
         warp, clamp >= 1e-3 (Lie-Shape-NTFields lie_metric / lie_isotropic trainers)
    """
    S = d[:, 2 * dim:2 * dim + 2]
    S = S * S * (2 - S) * (2 - S)
    S = alpha * S + 1 - alpha
    Sd = d[:, 4 * dim + 2:4 * dim + 4]
    Sa = d[:, 4 * dim + 4:4 * dim + 6]
    Sa = torch.clamp(Sa / ANGLE_MAX, min=ANGLE_MIN / ANGLE_MAX, max=1)
    Sd = Sd ** 2 * (2 - Sd) ** 2
    Sa = Sa ** 2 * (2 - Sa) ** 2
    Sd = torch.clamp(Sd, min=0.001)
    Sa = torch.clamp(Sa, min=0.001)
    return {'S': S, 'Sd': Sd, 'Sa': Sa}


class Model:
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        self.Params = {'ModelPath': ModelPath, 'DataPath': DataPath, 'Device': device}
        self.dim = dim
        self.source = source
        self.folder = os.path.join(ModelPath, os.path.basename(DataPath.rstrip('/')) + '_graph_lie')
        self.Params['Network'] = {'Graph': {}}
        # 'Save Every * Epoch' 0 = only latest.pt (rewritten every 100 epochs and at the end); 'Save optimizer'
        # False = weights-only checkpoints (~2.5 MB instead of ~7 MB; the evaluator's loader needs no optimizer)
        self.Params['Training'] = {
            'Batch Size': 2000, 'Number of Epochs': 4000, 'Print Every * Epoch': 50, 'Save Every * Epoch': 0,
            'Save optimizer': False, 'Learning Rate': 1e-3, 'Alpha': 1.0, 'Seed': 0, 'Grad mode': 'lie',
            'Speed model': 'iso', 'Weight decay': 0.5, 'Weight': 'exp',
        }
        self.total_train_loss = []
        self.total_val_loss = []

    # ------------------------------------------------------------------ setup
    def _build(self, B):
        dev = self.Params['Device']
        self.B = B
        self.network = model_network.NN(dev, self.dim, B, graph=self.Params['Network']['Graph'])
        self.network.to(torch.device(dev))
        self.network.float()
        self.function = model_function.Function(self.folder, dev, self.network, self.dim,
                                                grad_mode=self.Params['Training']['Grad mode'],
                                                speed_model=self.Params['Training']['Speed model'],
                                                weight_mode=self.Params['Training'].get('Weight', 'exp'))

    def _log(self, msg):
        print(msg, flush=True)
        with open(os.path.join(self.folder, 'train_log.txt'), 'a') as fh:
            fh.write(msg + '\n')

    # ------------------------------------------------------------------ train
    def train(self):
        os.makedirs(self.folder, exist_ok=True)
        P = self.Params['Training']
        seed = int(P['Seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.Params['Network']['Graph'].setdefault('fourier_seed', seed)
        G = self.Params['Network']['Graph']
        if G.get('speed_pred_ckpt') and 'sf_pred_kw' not in G:
            # the predictor checkpoint knows how to rebuild its module (kind + constructor kwargs)
            ck = torch.load(G['speed_pred_ckpt'], map_location='cpu')
            G['sf_pred_kw'] = ck.get('kw', {})
            G['sf_pred_kind'] = ck.get('kind', 'mlp')
        B = torch.normal(0, 1, size=(128, self.dim))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
        self._build(B)
        if self.Params['Network']['Graph'].get('speed_pred_ckpt'):
            from models.graph_native import _sib
            sf = importlib.import_module(_sib.BASE + '.speed_fourier')
            ok = sf.load_speed_predictor(self.network.net, self.Params['Network']['Graph']['speed_pred_ckpt'],
                                         self.Params['Device'])
            self._log('speed predictor {} from {}'.format('loaded' if ok else 'NOT loaded (head has none)',
                                                           self.Params['Network']['Graph']['speed_pred_ckpt']))
        with open(os.path.join(self.folder, 'config.json'), 'w') as fh:
            json.dump(self.Params, fh, indent=1)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=P['Learning Rate'],
                                           weight_decay=P['Weight decay'])
        data = load_dataset(self.Params['DataPath'])
        dataloader = FastTensorDataLoader(data, batch_size=int(P['Batch Size']), shuffle=True)
        self._log('dataset {} rows; speed min {:.4f}; params {}; grad {}; speed model {}'.format(
            data.shape[0], data[:, 2 * self.dim:2 * self.dim + 2].min().item(),
            sum(p.numel() for p in self.network.parameters() if p.requires_grad), P['Grad mode'], P['Speed model']))
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1000.0
        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        prev_state_queue, prev_optimizer_queue = [], []
        n_reject = 0
        t_start = time.time()
        dev = self.Params['Device']
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
            n_nonfinite = 0
            while True:
                total_train_loss = 0
                total_diff = 0
                terms = {}
                ii = 0
                for i, wholedata in enumerate(dataloader, 0):
                    if ii > 4:
                        break
                    ii = ii + 1
                    d = wholedata[0].to(dev)
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
                # as the original trainer: a non-finite epoch is rolled back and repeated like any rejected one;
                # give up only if it keeps happening
                n_nonfinite = n_nonfinite + 1 if not np.isfinite(total_diff) else 0
                if n_nonfinite > 20:
                    self._log('non-finite loss in 20 consecutive repeats; aborting')
                    return
            self.total_train_loss.append(total_train_loss)
            beta = 1.0 / total_diff
            if epoch % P['Print Every * Epoch'] == 0 or epoch == 1:
                self._log('Epoch = {} -- Loss = {:.4e} -- T_mean = {:.3f} -- N_med = {:.3f} -- rejects = {} -- {:.0f}s'.format(
                    epoch, total_diff, terms.get('T_mean', float('nan')), terms.get('N_med', float('nan')), n_reject,
                    time.time() - t_start))
            if P['Save Every * Epoch'] and (epoch % P['Save Every * Epoch'] == 0 or epoch == P['Number of Epochs']):
                self.save(epoch=epoch, val_loss=total_diff)
            if epoch % 100 == 0 or epoch == P['Number of Epochs']:
                self.save(epoch=epoch, val_loss=total_diff, name='latest.pt')
        with open(os.path.join(self.folder, 'train_summary.json'), 'w') as fh:
            json.dump({'wall_time_s': time.time() - t_start, 'rejects': n_reject, 'final_loss': total_diff,
                       'epochs': P['Number of Epochs'], 'grad_mode': P['Grad mode'],
                       'speed_model': P['Speed model']}, fh, indent=1)

    def save(self, epoch='', val_loss='', name=None):
        name = name or 'Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(str(epoch).zfill(5), val_loss)
        ck = {'epoch': epoch, 'model_state_dict': self.network.state_dict(), 'B_state_dict': self.B,
              'graph': self.network.graph, 'grad_mode': self.Params['Training']['Grad mode'],
              'speed_model': self.Params['Training']['Speed model'],
              'weight_mode': self.Params['Training'].get('Weight', 'exp'),
              'train_loss': self.total_train_loss, 'val_loss': self.total_val_loss}
        if self.Params['Training'].get('Save optimizer', False):
            ck['optimizer_state_dict'] = self.optimizer.state_dict()
        tmp = os.path.join(self.folder, name + '.tmp')
        torch.save(ck, tmp)
        os.replace(tmp, os.path.join(self.folder, name))  # never leave a truncated checkpoint behind

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.Params['Device']))
        self.Params['Network']['Graph'] = checkpoint['graph']
        self.Params['Training']['Grad mode'] = checkpoint.get('grad_mode', 'chart')
        self.Params['Training']['Speed model'] = checkpoint.get('speed_model', 'combined')
        self.Params['Training']['Weight'] = checkpoint.get('weight_mode', 'exp')
        self._build(checkpoint['B_state_dict'])
        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.eval()
