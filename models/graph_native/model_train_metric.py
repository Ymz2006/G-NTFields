"""Original metric NTFields training driver (metric_baseline) with the graph-head network.

Identical to Lie-Shape-NTFields/models/metric_baseline/model_train_metric.py in every training
detail: FastTensorDataLoader over the whole dataset (batch 2000, reshuffled each epoch), 5
batches per epoch, losses divided by 4, speed warp s^2 (2 - s)^2 with alpha = 1, AdamW
(weight decay 0.5) with the learning rate hard-set to 5e-4 inside the loop, beta = 1 / total_diff,
and the accept / reject scheme (an epoch whose loss ratio to the previous epoch is >= 1.2 is
rolled back to a random one of the last five states). Differences: the network / loss come
from this package, the travel-time plot is skipped, checkpoints are written every 1000 epochs
plus ``latest.pt``, the run folder gets ``config.json`` / ``train_log.txt`` / ``train_summary.json``.
"""
import json
import os
import pickle
import random
import time

import numpy as np
import torch

from . import model_function_metric as model_function
from . import model_network_metric as model_network

torch.backends.cudnn.benchmark = True


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def load_dataset(path):
    """points (N, 12) | speed (N, 2) | normal (N, 12) | speed_dists (N, 2) | speed_angles (N, 2)"""
    cols = [np.load(os.path.join(path, f + '.npy')) for f in ('sampled_points', 'speed', 'normal')]
    for f in ('speed_dists', 'speed_angles'):
        p = os.path.join(path, f + '.npy')
        cols.append(np.load(p) if os.path.exists(p) else np.ones((cols[0].shape[0], 2), np.float32))
    return torch.from_numpy(np.concatenate(cols, 1).astype(np.float32))


class Model:
    def __init__(self, ModelPath, DataPath, dim, source, device='cpu'):
        self.Params = {'ModelPath': ModelPath, 'DataPath': DataPath, 'Device': device}
        self.dim = dim
        self.source = source
        self.folder = os.path.join(ModelPath, os.path.basename(DataPath.rstrip('/')) + '_graph_native')
        self.Params['Network'] = {'Graph': {}}
        self.Params['Training'] = {
            'Batch Size': 2000, 'Number of Epochs': 4000, 'Print Every * Epoch': 50, 'Save Every * Epoch': 1000,
            'Learning Rate': 1e-3, 'Alpha': 1.0, 'Seed': 0, 'Grad mode': 'chart', 'Weight decay': 0.5,
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
                                                grad_mode=self.Params['Training']['Grad mode'])

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
        B = torch.normal(0, 1, size=(128, self.dim))
        torch.nn.init.trunc_normal_(B, mean=0.0, std=1, a=-2.0, b=2.0)
        self._build(B)
        with open(os.path.join(self.folder, 'config.json'), 'w') as fh:
            json.dump(self.Params, fh, indent=1)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=P['Learning Rate'],
                                           weight_decay=P['Weight decay'])
        data = load_dataset(self.Params['DataPath'])
        dataloader = FastTensorDataLoader(data, batch_size=int(P['Batch Size']), shuffle=True)
        self._log('dataset {} rows; speed min {:.4f}; params {}'.format(
            data.shape[0], data[:, 2 * self.dim:2 * self.dim + 2].min().item(),
            sum(p.numel() for p in self.network.parameters() if p.requires_grad)))
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
                    speed = d[:, 2 * self.dim:2 * self.dim + 2]
                    normal = d[:, 2 * self.dim + 2:4 * self.dim + 2]
                    speed = speed * speed * (2 - speed) * (2 - speed)
                    speed = alpha * speed + 1 - alpha
                    loss_value, loss_n, wv = self.function.Loss(points, speed, normal, beta, gamma, epoch)
                    loss_value.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_train_loss += loss_value.item()
                    total_diff += loss_n.item()
                    for k, v in self.function.last_terms.items():
                        terms[k] = terms.get(k, 0.0) + v / 4
                    del points, speed, loss_value, loss_n
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
                self._log('Epoch = {} -- Loss = {:.4e} -- T_mean = {:.3f} -- rejects = {} -- {:.0f}s'.format(
                    epoch, total_diff, terms.get('T_mean', float('nan')), n_reject, time.time() - t_start))
            if epoch % P['Save Every * Epoch'] == 0 or epoch == P['Number of Epochs']:
                self.save(epoch=epoch, val_loss=total_diff)
            if epoch % 100 == 0 or epoch == P['Number of Epochs']:
                self.save(epoch=epoch, val_loss=total_diff, name='latest.pt')
        with open(os.path.join(self.folder, 'train_summary.json'), 'w') as fh:
            json.dump({'wall_time_s': time.time() - t_start, 'rejects': n_reject, 'final_loss': total_diff,
                       'epochs': P['Number of Epochs']}, fh, indent=1)

    def save(self, epoch='', val_loss='', name=None):
        name = name or 'Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(str(epoch).zfill(5), val_loss)
        torch.save({'epoch': epoch, 'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(), 'B_state_dict': self.B,
                    'graph': self.network.graph, 'grad_mode': self.Params['Training']['Grad mode'],
                    'train_loss': self.total_train_loss, 'val_loss': self.total_val_loss},
                   os.path.join(self.folder, name))

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device(self.Params['Device']))
        self.Params['Network']['Graph'] = checkpoint['graph']
        self.Params['Training']['Grad mode'] = checkpoint.get('grad_mode', 'chart')
        self._build(checkpoint['B_state_dict'])
        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.eval()
