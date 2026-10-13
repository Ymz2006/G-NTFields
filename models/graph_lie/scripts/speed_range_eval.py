"""Predicted vs true speed over the whole speed range, for any graph_lie / graph_native checkpoint.

For a fixed random subset of the scene's training pairs (seed 0, --rows pairs, both endpoints) the
predicted speed is 1 / |c| with c the Lie (body-frame) endpoint gradient of tau, the truth is the
warped S_d (isotropic Eikonal target) or the combined S (--speed combined). Per log bin of the true
speed: median predicted / true ratio, mean |log10 ratio| (dex error) and the quantiles of the
predicted speed. Headline numbers:

    dex_all   mean |log10(pred / true)| over all endpoints
    dex_bins  the same averaged over the log bins (every speed decade counts equally)
    dex_low   mean |log10| over true speed < 0.05,   dex_high   over true speed >= 0.5
    N_med     median of S |c| (1 = Eikonal equation)

    python models/graph_lie/scripts/speed_range_eval.py --out results/graph_lie/speed_range/baseline.json \
        Experiments/graph_lie/T_gmetric_iso_lie_s0 Experiments/graph_lie/T_gmetric_iso_lie_s1 ...
Several run directories of one configuration are pooled with --group; the markdown table is appended
to --md (one row per group).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.append('.')

from models.graph_lie import lie, model_train_lie as mtl  # noqa: E402
from models.graph_native.model_train_metric import load_dataset  # noqa: E402

EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]
Q = [10, 50, 90]
_DATA = {}


def rows_for(data_path, n_rows):
    if data_path not in _DATA:
        data = load_dataset(data_path)
        g = torch.Generator().manual_seed(0)
        _DATA[data_path] = data[torch.randperm(data.shape[0], generator=g)[:n_rows]]
    return _DATA[data_path]


def endpoint_speeds(run_dir, rows, device, speed_key):
    m = mtl.Model('/tmp/cal', 'datasets/3dshape/x', 6, [0.0] * 6, device=device)
    m.load(os.path.join(run_dir, 'latest.pt'))
    gt, pred, shat = [], [], []
    enc = getattr(m.network.net, 'enc', None)
    has_pred = enc is not None and getattr(enc, 'speed_pred', None) is not None
    for i in range(0, rows.shape[0], 4096):
        d = rows[i:i + 4096].to(device)
        sp = mtl.warp_speeds(d)
        _, cx, cy = lie.endpoint_gradients(m.network, d[:, :12], 'lie', create_graph=False)
        gt.append(torch.cat([sp[speed_key][:, 0], sp[speed_key][:, 1]]).cpu())
        pred.append(1.0 / torch.cat([cx.norm(dim=1), cy.norm(dim=1)]).clamp_min(1e-9).detach().cpu())
        if has_pred:
            with torch.no_grad():
                g0, g1 = m.network.features(d[:, :6]), m.network.features(d[:, 6:12])
                shat.append(torch.cat([enc.speed_pred(g0), enc.speed_pred(g1)]).cpu())
    out = {'gt': torch.cat(gt).numpy(), 'pred': torch.cat(pred).numpy()}
    if has_pred:
        out['shat'] = torch.cat(shat).numpy()
    out['config'] = m.Params['Network']['Graph']
    out['grad_mode'], out['speed_model'] = m.Params['Training']['Grad mode'], m.Params['Training']['Speed model']
    return out


def summarise(gt, pred):
    lr = np.log10(pred / gt)
    bins = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (gt >= lo) & (gt < hi)
        if s.sum() < 20:
            continue
        q = np.percentile(pred[s], Q)
        bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(s.sum()), 'gt_median': float(np.median(gt[s])),
                     'median_ratio': float(10 ** np.median(lr[s])), 'dex': float(np.abs(lr[s]).mean()),
                     'bias_dex': float(lr[s].mean()), **{'p{}'.format(k): float(v) for k, v in zip(Q, q)}})
    low, high = gt < 0.05, gt >= 0.5
    return {'dex_all': float(np.abs(lr).mean()), 'dex_bins': float(np.mean([b['dex'] for b in bins])),
            'dex_low': float(np.abs(lr[low]).mean()), 'dex_high': float(np.abs(lr[high]).mean()),
            'ratio_low': float(10 ** np.median(lr[low])), 'ratio_high': float(10 ** np.median(lr[high])),
            'N_med': float(np.median(gt / pred)), 'n': int(gt.size), 'bins': bins}


def md_row(name, s, extra=''):
    b = {round(x['lo'], 3): x for x in s['bins']}
    cells = ['{:.2f}'.format(b[e]['median_ratio']) if e in b else '-' for e in [round(x, 3) for x in EDGES[:-1]]]
    return '| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {} |{}'.format(
        name, s['n'], s['dex_all'], s['dex_bins'], s['dex_low'], s['dex_high'], s['N_med'], ' '.join(cells), extra)


MD_HEADER = ('| run / group | n endpoints | dex all | dex bins | dex low (<0.05) | dex high (>=0.5) | N med | '
             'median pred/true per log bin (0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5-1) |\n'
             '|---|---|---|---|---|---|---|---|')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--md', default=None, help='append a markdown row per group to this file')
    ap.add_argument('--group', default=None, help='name under which all runs are pooled (default: per run)')
    ap.add_argument('--dataPath', default=None, help='default: the run\'s config DataPath')
    ap.add_argument('--rows', type=int, default=8192)
    ap.add_argument('--speed', default='Sd', choices=['Sd', 'S'])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    res = {'edges': EDGES, 'runs': {}, 'groups': {}}
    pooled = {}
    for r in a.runs:
        r = r.rstrip('/')
        cfg = json.load(open(os.path.join(r, 'config.json')))
        dp = a.dataPath or cfg['DataPath']
        out = endpoint_speeds(r, rows_for(dp, a.rows), a.device, a.speed)
        s = summarise(out['gt'], out['pred'])
        s.update({'config': out['config'], 'grad_mode': out['grad_mode'], 'speed_model': out['speed_model'],
                  'dataPath': dp})
        if 'shat' in out:
            s['speed_pred'] = summarise(out['gt'], out['shat'])
        res['runs'][os.path.basename(r)] = s
        print('{}: dex all {:.3f} bins {:.3f} low {:.3f} high {:.3f} N_med {:.2f} | ratio low {:.2f} high {:.2f}'.format(
            os.path.basename(r), s['dex_all'], s['dex_bins'], s['dex_low'], s['dex_high'], s['N_med'],
            s['ratio_low'], s['ratio_high']), flush=True)
        key = a.group or os.path.basename(r)
        p = pooled.setdefault(key, {'gt': [], 'pred': [], 'shat': [], 'runs': []})
        p['gt'].append(out['gt']), p['pred'].append(out['pred']), p['runs'].append(os.path.basename(r))
        if 'shat' in out:
            p['shat'].append(out['shat'])
    lines = []
    for key, p in pooled.items():
        s = summarise(np.concatenate(p['gt']), np.concatenate(p['pred']))
        s['runs'] = p['runs']
        if p['shat']:
            s['speed_pred'] = summarise(np.concatenate(p['gt']), np.concatenate(p['shat']))
        res['groups'][key] = s
        lines.append(md_row('{} ({} runs)'.format(key, len(p['runs'])), s))
        if p['shat']:
            lines.append(md_row('  s_hat of ' + key, s['speed_pred']))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(MD_HEADER)
    print('\n'.join(lines))
    if a.md:
        new = not os.path.exists(a.md)
        with open(a.md, 'a') as fh:
            if new:
                fh.write(MD_HEADER + '\n')
            fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
