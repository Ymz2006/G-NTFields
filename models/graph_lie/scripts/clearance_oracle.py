"""Dense clearance supervision for the speed predictor, from the scene geometry.

The training pairs sample pose space too sparsely to learn the clearance speed by regression: the
nearest training poses are ~0.05 apart (chart units), the scale on which the clearance changes
(margin 0.05). The clearance is a known function of the geometry, so this script evaluates it with
Lie-Shape-NTFields' own placement evaluator (dataprocessing/preprocess_obj.py: exact point-to-triangle
distance between the placed shape and the sampled env surface, tetrahedral inside test) at as many
random poses as needed, and writes (chart pose, dist, angle, speed_dists) for predictor training.

    docker ... -v Lie-Shape-NTFields:/lieshape:ro ... python models/graph_lie/scripts/clearance_oracle.py \
        --dataPath datasets/3dshape/data/Tshape3d_env1 --n 6000000 --out datasets/3dshape/data/Tshape3d_env1/dense_speed.npz

It first scores the oracle against the dataset's own speed_dists on --validate stored endpoints (the
stored values were computed with an independent surface sampling, so the agreement is the label noise).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/lieshape/dataprocessing')
import preprocess_obj as pp  # noqa: E402


def build(args, meta):
    V_env, F_env, names_env = pp.load_obj(args.env)
    bb_min, bb_max = V_env.min(axis=0), V_env.max(axis=0)
    center_env = 0.5 * (bb_min + bb_max)
    scale = float((bb_max - bb_min).max())
    assert abs(scale - meta['env_scale']) < 1e-3 * scale, (scale, meta['env_scale'])
    V_env_n = (V_env - center_env) / scale
    sample_mask = np.array(['null' not in str(n).lower() for n in names_env])
    env_points = pp.sample_surface_points(V_env_n, F_env[sample_mask], args.num_env_points).astype(np.float32)
    ranges = (bb_max - bb_min) / scale
    half_extent = ranges * 0.5 - 0.01
    V_sh, F_sh, _ = pp.load_obj(args.shape)
    shape_center = 0.5 * (V_sh.min(axis=0) + V_sh.max(axis=0))
    V_sh_local = (V_sh - shape_center) / scale * meta['shape_scale']
    TV, TT, TF = pp.tetrahedralize_shape(V_sh_local, F_sh, switches=pp.DEFAULT_TET_SWITCHES)
    tet_verts = torch.tensor(TV[TT], dtype=torch.float32)
    face_verts = torch.tensor(TV[TF], dtype=torch.float32)
    rad_points, rad_bins = pp.generate_radius_surface_points(V_sh_local, F_sh, 1000, 10)
    return dict(env_points=torch.from_numpy(env_points), half_extent=half_extent, tet_verts=tet_verts,
                face_verts=face_verts, rad_points=rad_points, rad_bins=rad_bins)


def evaluate(geo, configs_rad, device, batch):
    """configs (N, 6) with rotvec in radians -> is_free, dist, min_angle (CPU tensors)."""
    free, dist, ang = [], [], []
    for i in range(0, configs_rad.shape[0], batch):
        f, d, a, *_ = pp.evaluate_placements(configs_rad[i:i + batch], geo['tet_verts'], geo['face_verts'],
                                             geo['env_points'], geo['rad_points'], geo['rad_bins'], device)
        free.append(f), dist.append(d), ang.append(a)
    return torch.cat(free), torch.cat(dist), torch.cat(ang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--env', default=None, help='default: from meta.json basename under datasets/3dshape')
    ap.add_argument('--shape', default=None)
    ap.add_argument('--num_env_points', type=int, default=50000)
    ap.add_argument('--n', type=int, default=6000000, help='random poses to evaluate (free ones are kept)')
    ap.add_argument('--validate', type=int, default=50000)
    ap.add_argument('--batch', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
    args.env = args.env or os.path.join('datasets/3dshape/obj', os.path.basename(meta['env_obj']))
    args.shape = args.shape or os.path.join('datasets/3dshape/obj', os.path.basename(meta['shape_obj']))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    geo = build(args, meta)
    margin, offset = meta['margin'], meta['offset']
    print('geometry ready ({:.0f}s): {} env points, {} tets, {} faces, half extent {}'.format(
        time.time() - t0, geo['env_points'].shape[0], geo['tet_verts'].shape[0], geo['face_verts'].shape[0],
        np.round(geo['half_extent'], 3)), flush=True)
    report = {}
    if args.validate:
        pts = np.load(os.path.join(args.dataPath, 'sampled_points.npy'))
        sd = np.load(os.path.join(args.dataPath, 'speed_dists.npy'))
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], args.validate // 2, replace=False)
        cfg = np.concatenate([pts[idx, :6], pts[idx, 6:12]], 0).copy()
        cfg[:, 3:] *= meta['rot_norm']
        truth = np.concatenate([sd[idx, 0], sd[idx, 1]])
        free, dist, ang = evaluate(geo, torch.from_numpy(cfg).float(), args.device, args.batch)
        mine = np.clip(dist.numpy() / margin, offset / margin, 1.0)
        lr = np.log(mine / truth)
        both = (mine < 1) & (truth < 1)
        report = {'n': int(truth.size), 'frac_free': float(free.float().mean()),
                  'rmse_log_all': float(np.sqrt(np.mean(lr ** 2))), 'within_x1.5_all': float(np.mean(np.abs(lr) < math.log(1.5))),
                  'rmse_log_band': float(np.sqrt(np.mean(lr[both] ** 2))),
                  'within_x1.5_band': float(np.mean(np.abs(lr[both]) < math.log(1.5))),
                  'median_abs_dist_err_band': float(np.median(np.abs(mine[both] - truth[both]) * margin))}
        for lo, hi in [(0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.0)]:
            m = (truth >= lo) & (truth < hi)
            report['ratio_bin_{}_{}'.format(lo, hi)] = [float(np.exp(np.median(lr[m]))), float(np.mean(np.abs(lr[m])))]
        print('validation vs stored speed_dists:', json.dumps(report), flush=True)
    he = geo['half_extent']
    keep_cfg, keep_d, keep_a = [], [], []
    done = 0
    while done < args.n:
        n = min(args.batch * 10, args.n - done)
        c = pp._sample_configs(n, he[0], he[1], he[2])
        free, dist, ang = evaluate(geo, c, args.device, args.batch)
        ok = free & (dist > offset)
        keep_cfg.append(c[ok]), keep_d.append(dist[ok]), keep_a.append(ang[ok])
        done += n
        if done % (args.batch * 100) == 0 or done == args.n:
            print('{} / {} evaluated, {} kept, {:.0f}s'.format(done, args.n, sum(len(k) for k in keep_d), time.time() - t0),
                  flush=True)
    cfg = torch.cat(keep_cfg).numpy()
    cfg[:, 3:] /= meta['rot_norm']  # chart of the datasets: rotvec / 2 pi
    dist, ang = torch.cat(keep_d).numpy(), torch.cat(keep_a).numpy()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, configs=cfg.astype(np.float32), dists=dist.astype(np.float32), angles=ang.astype(np.float32),
             speed_dists=np.clip(dist / margin, offset / margin, 1.0).astype(np.float32),
             speed_angles=(ang / np.pi).astype(np.float32), validation=json.dumps(report),
             args=json.dumps(vars(args)))
    print('wrote {}: {} poses ({:.1f} % with speed_dists < 1), {:.0f}s'.format(
        args.out, cfg.shape[0], 100.0 * np.mean(dist < margin), time.time() - t0))


if __name__ == '__main__':
    main()
