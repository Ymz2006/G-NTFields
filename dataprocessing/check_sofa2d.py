"""Planar configuration-space check of the moving-sofa meshes (make_sofa2d.py).

The free space of the sofa in a hall is computed on a grid over (x, y, theta):
0.5-unit cells, 1-degree steps.  A cell is free when the sofa footprint placed
there (rasterized at the cell's offsets) covers no wall cell -- one FFT
correlation of the wall raster with the rotated footprint per angle; for a
clearance c the walls are first grown by c.  Free cells are grouped into
connected components (6-neighbour, theta periodic).

    check   for every hall, the largest clearance at which the far end of one leg
            connects to the far end of the other, notch toward the inner corner
            (theta 0 -> 270 degrees) and notch outward (180 -> 90), for the U and for
            its bounding box (bisection on c)
    filter  keep the pairs of a --testing_data set whose start and goal lie in
            the same component (clearance 0), first --keep of them; every per-pair
            array is cut the same way, the originals are kept as <name>_all.npy

    python dataprocessing/check_sofa2d.py check
    python dataprocessing/check_sofa2d.py filter --data testing_data/3dshape/Usofa2d_2dhall30 --keep 1000
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_sofa2d as mk  # noqa: E402

RES = 0.5
NTH = 360
XS = np.arange(-mk.WALL + RES / 2, mk.LEG + mk.WALL, RES)      # cell centres, x and y


def wall_raster(W):
    X, Y = np.meshgrid(XS, XS, indexing='ij')
    occ = np.zeros(X.shape, bool)
    for b in mk.hall_boxes(W):
        occ |= (X >= b[0]) & (X <= b[1]) & (Y >= b[2]) & (Y <= b[3])
    return occ


def footprint(boxes, th):
    """Cells covered by the sofa at angle th, as offsets from its centre."""
    r = max(np.hypot(max(abs(b[0]), abs(b[1])), max(abs(b[2]), abs(b[3]))) for b in boxes) + RES
    k = np.arange(-np.ceil(r / RES), np.ceil(r / RES) + 1) * RES
    X, Y = np.meshgrid(k, k, indexing='ij')
    c, s = np.cos(th), np.sin(th)
    u, v = c * X + s * Y, -s * X + c * Y          # world offset -> sofa frame
    m = np.zeros(X.shape, bool)
    for b in boxes:
        m |= (u >= b[0]) & (u <= b[1]) & (v >= b[2]) & (v <= b[3])
    return m.astype(np.float32)


def free_grid(W, boxes, clear=0.0):
    """(NTH, nx, ny) cells where the sofa clears the walls grown by ``clear``
    (shrunk by -clear when negative: only cells deeper than that inside a wall count)."""
    occ = wall_raster(W)
    if clear > 0:
        occ = distance_transform_edt(~occ) * RES < clear
    elif clear < 0:
        occ = distance_transform_edt(occ) * RES > -clear
    occ = occ.astype(np.float32)
    free = np.zeros((NTH,) + occ.shape, bool)
    for i in range(NTH):
        K = footprint(boxes, 2 * np.pi * i / NTH)
        free[i] = fftconvolve(occ, K[::-1, ::-1], mode='same') < 0.5
    return free


def components(W, boxes, clear=0.0):
    """(NTH, nx, ny) component labels of the free cells (0 = blocked)."""
    lab, n = label(free_grid(W, boxes, clear))
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    a, b = lab[0], lab[-1]                       # theta wraps
    both = (a > 0) & (b > 0)
    for p, q in set(zip(a[both].tolist(), b[both].tolist())):
        parent[find(p)] = find(q)
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab]


def cell(x, y, th):
    i = int(np.round(np.mod(th, 2 * np.pi) / (2 * np.pi) * NTH)) % NTH
    return i, int(np.argmin(abs(XS - x))), int(np.argmin(abs(XS - y)))


def turns(W, boxes, clear):
    """(notch-in connected, notch-out connected) between the far ends of the legs."""
    comp = components(W, boxes, clear)
    hx, vy = (mk.LEG - 40, 0.5 * W), (0.5 * W, mk.LEG - 40)
    out = []
    for th_h, th_v in ((0, 1.5 * np.pi), (np.pi, 0.5 * np.pi)):
        a, b = comp[cell(*hx, th_h)], comp[cell(*vy, th_v)]
        out.append(bool(a > 0 and a == b))
    return out


def critical(args):
    W, which, boxes = args
    lo, hi = 0.0, 0.5 * W
    if not turns(W, boxes, 0.0)[which]:
        return None
    for _ in range(7):
        m = 0.5 * (lo + hi)
        if turns(W, boxes, m)[which]:
            lo = m
        else:
            hi = m
    return lo


def check():
    p = mk.SOFA['Usofa2d']
    sofas = {'U': mk.sofa_boxes(**p), 'box': mk.sofa_boxes(p['length'], p['depth'], 0, 0)}
    jobs = [(W, k, b) for W in mk.HALLS.values() for b in sofas.values() for k in (0, 1)]
    with Pool(len(jobs)) as pool:
        res = pool.map(critical, jobs)
    fmt = lambda c: 'blocked' if c is None else '{:.2f}'.format(c)
    print('largest clearance of a turn (env units; grid {} units, {} deg):'.format(RES, 360 // NTH))
    print('{:9s} {:>4s}  {:>10s} {:>11s}  {:>10s} {:>11s}'.format('hall', 'W', 'U in', 'U out', 'box in', 'box out'))
    for i, (name, W) in enumerate(mk.HALLS.items()):
        r = res[4 * i:4 * i + 4]
        print('{:9s} {:4.0f}  {:>10s} {:>11s}  {:>10s} {:>11s}'.format(name, W, *map(fmt, r)))


def filt(data, keep):
    meta_p = os.path.join(data, 'meta.json')
    meta = json.load(open(meta_p))
    env = os.path.splitext(os.path.basename(meta['env_obj']))[0]
    shape = os.path.splitext(os.path.basename(meta['shape_obj']))[0]
    assert meta.get('two_d') and env in mk.HALLS and shape in mk.SOFA, (env, shape)
    allp = os.path.join(data, 'sampled_points_all.npy')
    x = np.load(allp if os.path.exists(allp) else os.path.join(data, 'sampled_points.npy'))
    comp = components(mk.HALLS[env], mk.sofa_boxes(**mk.SOFA[shape]))
    c, s = np.asarray(meta['env_center']), float(meta['env_scale'])

    def lab(q):
        return np.array([comp[cell(*(c[:2] + s * p[:2]), 2 * np.pi * p[5])] for p in q])
    l0, l1 = lab(x[:, :6]), lab(x[:, 6:])
    ok = (l0 > 0) & (l0 == l1)
    idx = np.nonzero(ok)[0][:keep]
    n = len(x)
    for f in sorted(glob.glob(os.path.join(data, '*.npy'))):
        if f.endswith('_all.npy'):
            continue
        a = np.load(f)
        if a.shape[0] != n:
            continue
        src = f[:-4] + '_all.npy'
        if not os.path.exists(src):
            os.replace(f, src)
        np.save(f, np.load(src)[idx])
    meta['feasible_filter'] = {
        'generated': int(n), 'start_blocked_on_grid': int((l0 == 0).sum()),
        'goal_blocked_on_grid': int((l1 == 0).sum()), 'connected': int(ok.sum()), 'kept': int(len(idx)),
        'components_seen': int(len(set(l0[l0 > 0].tolist()) | set(l1[l1 > 0].tolist()))),
        'by': 'dataprocessing/check_sofa2d.py filter (grid {} units, {} deg)'.format(RES, 360 // NTH)}
    json.dump(meta, open(meta_p, 'w'), indent=2)
    print(json.dumps(meta['feasible_filter']))
    if len(idx) < keep:
        print('WARNING: only {} connected pairs (wanted {})'.format(len(idx), keep))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check')
    f = sub.add_parser('filter')
    f.add_argument('--data', required=True)
    f.add_argument('--keep', type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == 'check':
        check()
    else:
        filt(a.data, a.keep)


if __name__ == '__main__':
    main()
