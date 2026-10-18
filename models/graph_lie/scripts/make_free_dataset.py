"""Obstacle-free SE(3) dataset in the 3dshape layout, for the ground-truth control of models/graph_lie.

All clearance speeds are 1 (speed, speed_dists and speed_angles; every warp maps 1 to 1), so the three
speed formulations coincide and the exact travel time is the left-invariant geodesic distance

    T*(x, y) = sqrt(|t_x - t_y|^2 + (theta(R_x^T R_y) / 2 pi)^2)       (rotations in turns)

which solves the Lie Eikonal |c| = 1 at both endpoints. Poses: translations uniform in the T-shape
scene's box, rotations Haar-uniform, stored as (t, rotvec / 2 pi) with |rotvec| <= pi like the scenes.

    python3 models/graph_lie/scripts/make_free_dataset.py --out datasets/3dshape/data/free_se3 --n 800000
"""
import argparse
import json
import math
import os

import numpy as np

LO = np.array([-0.465, -0.323, -0.489])
HI = np.array([0.465, 0.323, 0.490])


def haar_rotvec_turns(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0                       # w >= 0: angle in [0, pi]
    s = np.linalg.norm(q[:, 1:], axis=1)
    ang = 2.0 * np.arctan2(s, q[:, 0])
    axis = q[:, 1:] / np.maximum(s, 1e-12)[:, None]
    return axis * (ang / (2.0 * math.pi))[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='datasets/3dshape/data/free_se3')
    ap.add_argument('--n', type=int, default=800000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = a.n
    t0 = LO + (HI - LO) * rng.uniform(size=(n, 3))
    t1 = LO + (HI - LO) * rng.uniform(size=(n, 3))
    P = np.concatenate([t0, haar_rotvec_turns(rng, n), t1, haar_rotvec_turns(rng, n)], 1).astype(np.float32)
    os.makedirs(a.out, exist_ok=True)
    ones = np.ones((n, 2), np.float32)
    np.save(os.path.join(a.out, 'sampled_points.npy'), P)
    np.save(os.path.join(a.out, 'speed.npy'), ones)
    np.save(os.path.join(a.out, 'speed_dists.npy'), ones)
    np.save(os.path.join(a.out, 'speed_angles.npy'), ones)
    np.save(os.path.join(a.out, 'normal.npy'), np.zeros((n, 12), np.float32))
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump({'obstacle_free': True, 'box_lo': LO.tolist(), 'box_hi': HI.tolist(), 'rot_norm': 2 * math.pi,
                   'seed': a.seed, 'n': n, 'note': 'models/graph_lie ground-truth control; T* = SE(3) geodesic'}, fh,
                  indent=1)
    print('wrote', a.out, P.shape, 'max |r| (turns)', float(np.linalg.norm(P[:, 3:6], axis=1).max()))


if __name__ == '__main__':
    main()
