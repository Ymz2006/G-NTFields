"""Build the structured clearance predictor (clearance.py) for a scene and validate it.

    docker ... python models/graph_lie/scripts/build_clearance_predictor.py --dataPath datasets/3dshape/data/Tshape3d_env1 \
        --out Experiments/graph_lie/speed_pred/Tshape3d_env1_clearance.pt [--dense datasets/3dshape/data/Tshape3d_env1/dense_speed_s1.npz]

Validation: predicted vs stored speed_dists on 100 k dataset endpoints (and on the dense oracle poses when
given): per-bin median ratio, mean |log10|, fraction within x1.5. Writes the predictor state dict + kw.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.append('.')
sys.path.insert(0, '/lieshape/dataprocessing')

from models.graph_native import _sib  # noqa: E402

cl = __import__(_sib.BASE + '.clearance', fromlist=['clearance'])
EDGES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]


def load_obj(path):
    import preprocess_obj as pp
    V, F, names = pp.load_obj(path)
    names = [str(n) for n in names]
    ids = {n: i for i, n in enumerate(sorted(set(names)))}
    return V, F, np.array([ids[n] for n in names])


def features(cfg):
    x = torch.from_numpy(cfg).float()
    return _sib.se3.features(x[:, :3], _sib.se3.so3_exp(2.0 * math.pi * x[:, 3:6]))


def validate(pred, cfg, sd_raw, device, label):
    g = features(cfg)
    out = []
    with torch.no_grad():
        for i in range(0, g.shape[0], 4096):
            out.append(pred.raw_speed(g[i:i + 4096].to(device)).cpu())
    mine = torch.cat(out).numpy()
    warp = lambda s: np.clip(s ** 2 * (2 - s) ** 2, 1e-3, None)
    lr = np.log10(warp(mine) / warp(sd_raw))
    rep = {'n': int(sd_raw.size), 'dex_all': float(np.abs(lr).mean()), 'within_x1.5': float(np.mean(np.abs(lr) < math.log10(1.5))),
           'median_abs_dist_err_band': float(np.median(np.abs(mine - sd_raw)[(mine < 1) & (sd_raw < 1)]) * pred.margin), 'bins': []}
    truth = warp(sd_raw)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (truth >= lo) & (truth < hi)
        if m.sum() > 20:
            rep['bins'].append({'lo': lo, 'hi': min(hi, 1.0), 'n': int(m.sum()), 'median_ratio': float(10 ** np.median(lr[m])),
                                'dex': float(np.abs(lr[m]).mean())})
    print('{}: n {} dex {:.3f} within x1.5 {:.3f} median |dist err| in band {:.4f}'.format(
        label, rep['n'], rep['dex_all'], rep['within_x1.5'], rep['median_abs_dist_err_band']))
    print('   per bin median ratio: ' + ' '.join('{:.2f}'.format(b['median_ratio']) for b in rep['bins']))
    print('   per bin dex:          ' + ' '.join('{:.2f}'.format(b['dex']) for b in rep['bins']), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--res', type=int, default=192)
    ap.add_argument('--n_points', type=int, default=1024)
    ap.add_argument('--tau', type=float, default=0.0, help='smooth-min temperature; 0 = exact min')
    ap.add_argument('--pad', type=float, default=0.05, help='grid box = sampling box + pad')
    ap.add_argument('--dense', nargs='*', default=[])
    ap.add_argument('--n_val', type=int, default=100000)
    ap.add_argument('--d_bias', type=float, default=0.0)
    ap.add_argument('--mode', default='auto', choices=['auto', 'grid'], help='auto: exact boxes when the env is boxes')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.dataPath, 'meta.json')))
    env = os.path.join('datasets/3dshape/obj', os.path.basename(meta['env_obj']))
    shape = os.path.join('datasets/3dshape/obj', os.path.basename(meta['shape_obj']))
    t0 = time.time()
    V_env, F_env, G_env = load_obj(env)
    bb_min, bb_max = V_env.min(0), V_env.max(0)
    scale = float((bb_max - bb_min).max())
    V_env_n = (V_env - 0.5 * (bb_min + bb_max)) / scale
    V_sh, F_sh, _ = load_obj(shape)
    V_sh_local = (V_sh - 0.5 * (V_sh.min(0) + V_sh.max(0))) / scale * meta['shape_scale']
    half = (bb_max - bb_min) / scale * 0.5
    bmin, bmax = -half - a.pad, half + a.pad
    boxes = None if a.mode == 'grid' else cl.fit_boxes(V_env_n, F_env, G_env, np.zeros(3))
    if boxes is not None:
        kw = dict(mode='boxes', n_boxes=len(boxes[0]))
        pred = cl.ClearancePredictor(n_points=a.n_points, tau=a.tau, margin=meta['margin'], offset=meta['offset'], d_bias=a.d_bias, **kw)
        for name, arr in zip(('box_center', 'box_half', 'box_rot', 'box_sign'), boxes):
            getattr(pred, name).copy_(torch.from_numpy(arr))
        print('exact box env: {} boxes, signs {}'.format(len(boxes[0]), boxes[3].tolist()))
    else:
        kw = dict(mode='grid', res=a.res)
        pred = cl.ClearancePredictor(res=a.res, n_points=a.n_points, tau=a.tau, margin=meta['margin'], offset=meta['offset'], d_bias=a.d_bias)
        grid, flipped = cl.build_env_sdf_grid(V_env_n, F_env, bmin, bmax, a.res, groups=G_env)
        pred.grid.copy_(grid)
        print('sdf grid; groups with flipped sign (containers):', flipped)
        pred.bmin.copy_(torch.tensor(bmin, dtype=torch.float32))
        pred.bmax.copy_(torch.tensor(bmax, dtype=torch.float32))
    pred.points.copy_(torch.from_numpy(cl.sample_shape_points(V_sh_local, F_sh, a.n_points)))
    pred = pred.to(a.device)
    print('predictor ready ({:.0f}s), {} shape points'.format(time.time() - t0, a.n_points), flush=True)
    reports = {}
    pts = np.load(os.path.join(a.dataPath, 'sampled_points.npy'))
    sd = np.load(os.path.join(a.dataPath, 'speed_dists.npy'))
    rng = np.random.default_rng(0)
    idx = rng.choice(pts.shape[0], a.n_val // 2, replace=False)
    cfg = np.concatenate([pts[idx, :6], pts[idx, 6:12]], 0)
    reports['dataset'] = validate(pred, cfg, np.concatenate([sd[idx, 0], sd[idx, 1]]), a.device, 'dataset endpoints')
    for f in a.dense:
        z = np.load(f)
        n = min(a.n_val, z['configs'].shape[0])
        reports[os.path.basename(f)] = validate(pred, z['configs'][:n], z['speed_dists'][:n], a.device, 'dense oracle ' + f)
    kw.update(n_points=a.n_points, tau=a.tau, margin=meta['margin'], offset=meta['offset'], d_bias=a.d_bias)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({'state_dict': {k: v.cpu() for k, v in pred.state_dict().items()}, 'kw': kw, 'kind': 'clearance',
                'validation': reports, 'args': vars(a)}, a.out)
    with open(a.out.replace('.pt', '.json'), 'w') as fh:
        json.dump({'kw': kw, 'validation': reports, 'args': vars(a)}, fh, indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
