"""Figures of the path-constrained comparison (results/path_constrained/figures).

    scene_<cell>.png      the scene, the object path and, for three test cases of the full set (both models succeed /
                          homotopy succeeds and NTFields does not / the converse or a common failure), the camera end
                          (anchor + h R e_z) along each model's plan (forward mode, seed 0), red where the prism
                          touches the scene
    success.png           SE(3) success per set and model, forward mode (main cells with seed spread; second
                          paths and the 0.3 prism)

    python models/path_constrained/scripts/make_figures.py          (container, repo root)
"""
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.path_constrained import geometry as G  # noqa: E402
from models.path_constrained.path import Path  # noqa: E402

RES = os.path.join(REPO, 'results/path_constrained')
OUT = os.path.join(RES, 'figures')
SETS = os.path.join(REPO, 'testing_data/3dshape')
TWO_PI = 2 * math.pi
CELLS = [('e1a', 'prism_env1_a'), ('e3a', 'prism_env3_a'), ('e4a', 'prism_env4_a'),
         ('e1b', 'prism_env1_b'), ('e3b', 'prism_env3_b'), ('e4b', 'prism_env4_b')]
COL = {'hom': '#1f6fb4', 'ntf': '#d9822b'}


def box_faces(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [c[list(f)] for f in idx]


def load_eval(run, s, planner, mode):
    f = os.path.join(RES, 'plan', run, '{}__{}_{}.json'.format(s, planner, mode))
    p = f[:-5] + '_paths.npz'
    if not (os.path.exists(f) and os.path.exists(p)):
        return None
    d = json.load(open(f))
    z = np.load(p)
    return {'rec': {r['case']: r for r in d['records']}, 'path': {int(c): z['path'][i] for i, c in enumerate(z['case'])},
            'iters3': {int(c): int(z['iters3'][i]) for i, c in enumerate(z['case'])}}


def camera_track(path, prism, scene, Y, k3):
    Y = Y[~np.isnan(Y[:, 0])]
    if k3 >= 0:
        Y = Y[:k3 + 2]
    l = torch.as_tensor(Y[:, 0], dtype=torch.float64)
    R = G.so3_exp(TWO_PI * torch.as_tensor(Y[:, 1:], dtype=torch.float64))
    t = path.pos(l)
    cam = t + prism.h * R[:, :, 2]
    c = G.prism_clearance(scene, prism, t, R)
    return t.numpy(), cam.numpy(), c.numpy()


def scene_figure(code, cell):
    meta = json.load(open(os.path.join(SETS, cell + '_full', 'meta.json')))
    con = meta['constraint']
    path = Path.load(os.path.join(SETS, cell + '_full', con['path_file']))
    scene = G.BoxScene(os.path.join(REPO, meta['env_obj_rel']))
    prism = G.Prism(con['prism_width'], con['prism_length'])
    H = load_eval('{}_hom_s0'.format(code), cell + '_full', 'dual', 'forward')
    N = load_eval('{}_ntf_s0'.format(code), cell + '_full', 'base', 'forward')
    if H is None or N is None:
        return False
    cases = sorted(set(H['rec']) & set(N['rec']))
    both = [c for c in cases if H['rec'][c]['success_se3'] and N['rec'][c]['success_se3']]
    honly = [c for c in cases if H['rec'][c]['success_se3'] and not N['rec'][c]['success_se3']]
    other = [c for c in cases if not H['rec'][c]['success_se3']]
    picks = [(both[:1], 'both succeed'), (honly[:1], 'homotopy only'), (other[:1], 'homotopy fails')]
    picks = [(p[0], t) for p, t in picks if p]
    fig = plt.figure(figsize=(6 * len(picks), 6))
    for k, (case, title) in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), k + 1, projection='3d')
        for lo, hi in zip(scene.box_lo, scene.box_hi):
            ax.add_collection3d(Poly3DCollection(box_faces(lo, hi), facecolor='0.7', edgecolor='0.4', alpha=0.25, lw=0.5))
        ax.plot(*path.pts.T, color='k', lw=2, label='object path')
        for name, E in (('hom', H), ('ntf', N)):
            t, cam, c = camera_track(path, prism, scene, E['path'][case], E['iters3'][case])
            ax.plot(*cam.T, color=COL[name], lw=1.5,
                    label='{} camera ({})'.format('homotopy dual' if name == 'hom' else 'NTFields',
                                                  'ok' if E['rec'][case]['success_se3'] else
                                                  ('collision' if E['rec'][case]['collision_se3'] else 'no conv.')))
            bad = c < 0
            if bad.any():
                ax.scatter(*cam[bad].T, color='red', s=6)
            for i in range(0, len(t), max(1, len(t) // 12)):
                ax.plot(*np.stack([t[i], cam[i]]).T, color=COL[name], lw=0.6, alpha=0.6)
        lo, hi = scene.room_lo, scene.room_hi
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=22, azim=-60)
        ax.set_title('{} case {}: {}'.format(cell, case, title), fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'scene_{}.png'.format(cell)), dpi=90)
    plt.close(fig)
    return True


def success_figure():
    """SE(3) success (forward mode) per set: main cells (3 seeds, mean +- std), second paths and the 0.3 prism (seed 0)."""
    rows = []
    for f in glob.glob(os.path.join(RES, 'plan', '*', '*.json')):
        run = os.path.basename(os.path.dirname(f))
        m = re.match(r'(e\d[abs])_(hom|ntf)_s(\d+)$', run)
        b = re.match(r'(.+)_(full|seg|tight|bigrot|hard)__(base|dual)_(free|forward)\.json$', os.path.basename(f))
        if m and b and b.group(4) == 'forward':
            rows.append((m.group(1), m.group(2), b.group(2), b.group(3), json.load(open(f))['summary']['success_rate_se3']))
    if not rows:
        return
    combos = [('hom', 'dual'), ('hom', 'base'), ('ntf', 'base')]
    lab = {('hom', 'dual'): 'homotopy, dual planner', ('hom', 'base'): 'homotopy, base planner', ('ntf', 'base'): 'NTFields'}
    clr = {('hom', 'dual'): '#1f6fb4', ('hom', 'base'): '#8fbbe3', ('ntf', 'base'): '#d9822b'}
    panels = [('main cells (path a, prism 0.2, 3 seeds)', [('e1a', 'env1'), ('e3a', 'env3'), ('e4a', 'env4')],
               ('full', 'tight', 'bigrot', 'hard', 'seg')),
              ('second path (b) and camera farther away (prism 0.3), seed 0',
               [('e1b', 'env1 b'), ('e3b', 'env3 b'), ('e4b', 'env4 b'), ('e1s', 'env1 h0.3'), ('e3s', 'env3 h0.3'),
                ('e4s', 'env4 h0.3')], ('full', 'hard'))]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    for ax, (title, codes, variants) in zip(axes, panels):
        groups = [(c, n, v) for c, n in codes for v in variants
                  if any(r[0] == c and r[2] == v for r in rows)]
        x = np.arange(len(groups))
        for k, cb in enumerate(combos):
            vals, errs = [], []
            for c, _, v in groups:
                xs = [r[4] for r in rows if r[0] == c and r[1] == cb[0] and r[3] == cb[1] and r[2] == v]
                vals.append(100 * np.mean(xs) if xs else np.nan)
                errs.append(100 * np.std(xs) if len(xs) > 1 else 0)
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, yerr=errs, color=clr[cb], label=lab[cb], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(['{}\n{}'.format(n, v) for _, n, v in groups], fontsize=8)
        ax.set_ylim(70, 100.5)
        ax.set_ylabel('SE(3) success %, forward mode')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'success.png'), dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for code, cell in CELLS:
        print(cell, scene_figure(code, cell))
    success_figure()
