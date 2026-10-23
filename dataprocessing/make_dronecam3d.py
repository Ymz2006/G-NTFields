"""Drone + chase camera, six camera set-ups, five scenes of increasing difficulty (2026-10-22, the second camera test).

The drone is a small delta-wing polycube (fuselage 14 x 4 x 4 flying along +x, wing span 20, fin 4 tall; y up); the
camera follows it on its axis at a distance L behind the wing.  What the camera sees up to the drone is a rectangle-based
pyramid: tip = camera, base = a 2-unit plate pressed on the drone's back.  Two bases (the field of view at a given
distance): ``t`` (tight) 22 x 10, just the drone's silhouette with a 1-unit margin, and ``w`` (wide) 34 x 16.  Three
distances L = 30 / 55 / 80 (1.5 / 2.75 / 4 wing spans), so six camera set-ups with fields of view from 59 x 30
degrees (30w) down to 16 x 7 degrees (80t).  Bodies, all in the same frame per distance (same bounding-box centre, so
one test set scores both):

    dronecam<L><t|w>   drone + plate + view pyramid: planned together, a collision-free path keeps the drone, the camera
                       and the whole field of view clear of the scene
    drone<L>           drone + a 1-unit cube at the camera: the rigid follower (plans the drone, the camera is just
                       attached behind it; no view model)
    drone_only         the drone alone (visibility target of plan_eval_homotopy.py --camera)

Scenes (the env3 room shell, 198 x 138 x 218 units, so normalization 220.457 as every shape cell; written as a soup
of closed boxes, 12 triangles each, so a scene is 1-2k triangles), easiest to hardest:

    meadow      five trees and two sheds; free air above everything
    suburb      six houses along two streets, eight trees whose branches overhang the streets
    campus      six mid-rise buildings with two sky bridges, eight tall trees in the courtyards, porticos
    downtown    a 3 x 3 block grid of 85..130-unit towers, streets 40 / 44 wide, sky bridges, masts, porticos; only
                a few units of air above the tallest roofs
    overgrown   the tower grid taken over by forest: eight trees at the kerbs with 20-26-unit branches across the
                streets, cables between the towers, sky bridges

A tree is a 6 x 6 trunk, a stepped crown (34 / 44 / 34 / 18 wide) and 2-4 branches (4 x 4, 18-28 long) with a leaf
clump at the end; a building is a box with optional setback, water tank, mast (3 x 3 x 22) and portico (a 2-unit slab
10 out from the face at height 24).  Thin high obstacles -- branches, bridges, porticos, cables -- are what the camera
test is about: the drone clears them easily, a camera 30-90 units behind it does not.

    python dataprocessing/make_dronecam3d.py                     # writes datasets/3dshape/obj/{drone*,dronecam*,<scene>}.obj
    python dataprocessing/make_dronecam3d.py --preview DIR       # + PNG previews (matplotlib)
    python dataprocessing/make_dronecam3d.py --check 20000       # + free fraction of random placements per body x scene (igl)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_shapes3d as ms  # noqa: E402
from make_camera3d import attach_pyramid  # noqa: E402

OUT_DIR = ms.OUT_DIR
X0, X1, Y0, Y1, Z0, Z1 = ms.X0, ms.X1, ms.Y0, ms.Y1, ms.Z0, ms.Z1

# Boxes are (x0, x1, y0, y1, z0, z1); the drone's bounding box is x 0..14, y 0..8, z -10..10 (centre (7, 4, 0)).
DRONE = [
    (0, 14, 0, 4, -2, 2),         # fuselage
    (0, 7, 3, 4, -10, 10),        # delta wing at the back (span 20, chord 7, 1 thick, flush with the fuselage top)
    (0, 3, 4, 8, -0.5, 0.5),      # fin
]
# base of the view pyramid (x -2..0), symmetric about the drone's silhouette centre (y 4, z 0)
PLATES = {'t': (-2, 0, -1, 9, -11, 11), 'w': (-2, 0, -4, 12, -17, 17)}
DISTANCES = (30, 55, 80)
CAMS = ['{}{}'.format(L, f) for L in DISTANCES for f in ('t', 'w')]
SCENES = ('meadow', 'suburb', 'campus', 'downtown', 'overgrown')


def apex_of(L):
    return (PLATES['t'][0] - L, 4.0, 0.0)


def body_boxes(fov):
    """Drone + plate + a 1-unit lip at the wing's height across the plate's front face (so the face has no hole:
    polycube_mesh(merge=True) needs simple faces)."""
    p = PLATES[fov]
    return DRONE + [p, (0, 1, 3, 4, p[4], p[5])]


def follower_boxes(L):
    a = apex_of(L)
    return DRONE + [(a[0], a[0] + 1, a[1] - 0.5, a[1] + 0.5, a[2] - 0.5, a[2] + 0.5)]


# ----------------------------------------------------------------------------- scene primitives (box soups)
def box_mesh(b):
    x0, x1, y0, y1, z0, z1 = b
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    F = np.array([[0, 2, 1], [0, 3, 2],      # z0 face (normal -z)
                  [4, 5, 6], [4, 6, 7],      # z1 face (+z)
                  [0, 1, 5], [0, 5, 4],      # y0 (-y)
                  [3, 7, 6], [3, 6, 2],      # y1 (+y)
                  [0, 4, 7], [0, 7, 3],      # x0 (-x)
                  [1, 2, 6], [1, 6, 5]],     # x1 (+x)
                 dtype=np.int64)
    return V, F


def soup(boxes):
    Vs, Fs, off = [], [], 0
    for b in boxes:
        V, F = box_mesh(b)
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def tree(x, z, top, rng, n_branch=3, reach=(18, 28)):
    """A tree at (x, z) whose crown ends at y = top: trunk 6 x 6, stepped crown 40 tall, ``n_branch`` branches (4 x 4,
    ``reach`` long) with a 12 x 8 x 12 leaf clump at the end, between 35 and 80 % of the way up the trunk."""
    cb = top - 40.0
    boxes = [((x - 3, x + 3, Y0, cb + 6, z - 3, z + 3), 'trunk')]
    for w, y0, y1 in ((34, cb, cb + 10), (44, cb + 8, cb + 24), (34, cb + 22, cb + 34), (18, cb + 32, top)):
        h = w / 2.0
        boxes.append(((x - h, x + h, y0, y1, z - h, z + h), 'crown'))
    for d in rng.permutation(4)[:n_branch]:
        y = Y0 + rng.uniform(0.35, 0.8) * (cb - Y0)
        r = rng.uniform(*reach)
        dx, dz = ((1, 0), (-1, 0), (0, 1), (0, -1))[d]
        ex, ez = x + dx * r, z + dz * r
        if dx:
            boxes.append(((min(x, ex), max(x, ex), y - 2, y + 2, z - 2, z + 2), 'branch'))
        else:
            boxes.append(((x - 2, x + 2, y - 2, y + 2, min(z, ez), max(z, ez)), 'branch'))
        boxes.append(((ex - 6, ex + 6, y - 3, y + 5, ez - 6, ez + 6), 'leaves'))
    return boxes


def building(x0, x1, z0, z1, h, setback=0.0, tank=None, mast=None, porticos=()):
    """A flat-roofed block; ``setback`` > 0 insets the top part by that much from 60 % of the height; ``tank`` /
    ``mast`` = corner ('++', '+-', ...) of a 8-cube water tank / a 3 x 3 x 22 mast on the roof; ``porticos`` = faces
    ('x-', 'x+', 'z-', 'z+') that get a 2-unit slab 10 units out at height 24."""
    top = Y0 + h
    if setback > 0:
        ys = Y0 + 0.6 * h
        boxes = [((x0, x1, Y0, ys, z0, z1), 'wall'), ((x0 + setback, x1 - setback, ys - 1, top, z0 + setback, z1 - setback), 'wall')]
        rx0, rx1, rz0, rz1 = x0 + setback, x1 - setback, z0 + setback, z1 - setback
    else:
        boxes = [((x0, x1, Y0, top, z0, z1), 'wall')]
        rx0, rx1, rz0, rz1 = x0, x1, z0, z1

    def corner(c, s):
        cx = rx1 - 4 - s if c[0] == '+' else rx0 + 4 + s
        cz = rz1 - 4 - s if c[1] == '+' else rz0 + 4 + s
        return cx, cz
    if tank:
        cx, cz = corner(tank, 0)
        boxes.append(((cx - 4, cx + 4, top - 1, top + 8, cz - 4, cz + 4), 'tank'))
    if mast:
        cx, cz = corner(mast, 6)
        boxes.append(((cx - 1.5, cx + 1.5, top - 1, top + 22, cz - 1.5, cz + 1.5), 'mast'))
    for f in porticos:
        y = Y0 + 24
        if f == 'x-':
            boxes.append(((x0 - 10, x0 + 1, y, y + 2, z0, z1), 'portico'))
        elif f == 'x+':
            boxes.append(((x1 - 1, x1 + 10, y, y + 2, z0, z1), 'portico'))
        elif f == 'z-':
            boxes.append(((x0, x1, y, y + 2, z0 - 10, z0 + 1), 'portico'))
        else:
            boxes.append(((x0, x1, y, y + 2, z1 - 1, z1 + 10), 'portico'))
    return boxes


def bridge(a, b, y, w=10, h=7):
    """A sky bridge (w wide, h tall, bottom at y) between the footprints a and b (x0, x1, z0, z1), which are separated
    along x or along z; it overlaps 2 units into each block."""
    if a[1] <= b[0] or b[1] <= a[0]:         # separated along x
        lo, hi = (a, b) if a[1] <= b[0] else (b, a)
        zc = 0.5 * (max(lo[2], hi[2]) + min(lo[3], hi[3]))
        return [((lo[1] - 2, hi[0] + 2, y, y + h, zc - w / 2.0, zc + w / 2.0), 'bridge')]
    lo, hi = (a, b) if a[3] <= b[2] else (b, a)
    xc = 0.5 * (max(lo[0], hi[0]) + min(lo[1], hi[1]))
    return [((xc - w / 2.0, xc + w / 2.0, y, y + h, lo[3] - 2, hi[2] + 2), 'bridge')]


def cable(a, b, y):
    return [(bridge(a, b, y, w=2, h=2)[0][0], 'cable')]


# ----------------------------------------------------------------------------- the five scenes
def scene_meadow(rng):
    out = []
    for k, (x, z, top, nb) in enumerate([(40, -160, 112, 2), (120, -170, 118, 3), (165, -95, 105, 2),
                                         (70, -80, 115, 3), (130, -25, 108, 2)]):
        out.append(('tree_{}'.format(k + 1), tree(x + rng.uniform(-4, 4), z + rng.uniform(-4, 4), top, rng, nb)))
    out.append(('shed_1', building(18, 50, -60, -30, 22, porticos=('x+',))))
    out.append(('shed_2', building(150, 185, -150, -118, 28, tank='++')))
    return out


def scene_suburb(rng):
    out = []
    houses = [(12, 44, -190, -160, 36, '+-'), (70, 104, -192, -164, 42, None), (140, 176, -188, -158, 32, '--'),
              (14, 46, -70, -40, 40, None), (76, 108, -72, -44, 34, '++'), (142, 178, -68, -36, 45, None)]
    for k, (x0, x1, z0, z1, h, tank) in enumerate(houses):
        out.append(('house_{}'.format(k + 1), building(x0, x1, z0, z1, h, tank=tank, mast='-+' if k == 1 else None,
                                                       porticos=('z+',) if k < 3 else ('z-',))))
    # two tree rows along the street z -140..-100 and the yard z -30..10, branches 22-30 long over the streets
    spots = [(30, -128, 118), (92, -124, 125), (156, -130, 112), (62, -96, 108), (124, -100, 120),
             (28, -14, 110), (96, -8, 122), (160, -16, 115)]
    for k, (x, z, top) in enumerate(spots):
        out.append(('tree_{}'.format(k + 1), tree(x + rng.uniform(-5, 5), z + rng.uniform(-4, 4), top, rng,
                                                  n_branch=3 if k % 2 else 4, reach=(22, 30))))
    return out


def scene_campus(rng):
    out = []
    blocks = [(8, 56, -192, -144, 78, dict(setback=6, mast='++')), (120, 170, -194, -148, 92, dict(tank='-+')),
              (10, 58, -92, -46, 64, dict(porticos=('x+', 'z+'))), (122, 172, -96, -48, 86, dict(setback=6, porticos=('x-',))),
              (40, 90, -14, 14, 70, dict(tank='+-')), (140, 190, -12, 14, 95, dict(mast='-+'))]
    foot = []
    for k, (x0, x1, z0, z1, h, kw) in enumerate(blocks):
        out.append(('block_{}'.format(k + 1), building(x0, x1, z0, z1, h, **kw)))
        foot.append((x0, x1, z0, z1))
    out.append(('bridge_1', bridge(foot[0], foot[1], Y0 + 50)))
    out.append(('bridge_2', bridge(foot[2], foot[3], Y0 + 44)))
    out.append(('bridge_3', bridge(foot[1], foot[3], Y0 + 62, w=8, h=6)))
    spots = [(88, -175, 128), (30, -118, 124), (150, -122, 130), (88, -70, 112), (16, -22, 118), (116, 2, 122)]
    for k, (x, z, top) in enumerate(spots):
        out.append(('tree_{}'.format(k + 1), tree(x + rng.uniform(-4, 4), z + rng.uniform(-4, 4), top, rng,
                                                  n_branch=3, reach=(18, 24))))
    return out


def tower_grid():
    """3 x 3 footprints of the tower scenes: 36 x 40 blocks, streets 40 (x) and 44 (z) wide."""
    xs = [(5, 41), (81, 117), (157, 193)]
    zs = [(-194, -154), (-110, -70), (-26, 14)]
    return [(x0, x1, z0, z1) for z0, z1 in zs for x0, x1 in xs]


def scene_downtown(rng):
    out = []
    G = tower_grid()
    spec = [(118, dict(mast='++')), (96, dict(setback=6, porticos=('z+',))), (130, dict(tank='+-')),
            (104, dict(porticos=('x+', 'x-'))), (88, dict(setback=6, tank='--', mast='++')), (124, dict(porticos=('z-',))),
            (110, dict(tank='++')), (128, dict(mast='-+', porticos=('x-',))), (92, dict(setback=6, porticos=('z-', 'x+')))]
    for k, ((x0, x1, z0, z1), (h, kw)) in enumerate(zip(G, spec)):
        if k == 4:      # the centre block is two narrower towers
            out.append(('tower_{}a'.format(k + 1), building(x0, x0 + 15, z0, z1, 100, tank='-+')))
            out.append(('tower_{}b'.format(k + 1), building(x1 - 15, x1, z0, z1, 84, porticos=('x+',))))
            continue
        out.append(('tower_{}'.format(k + 1), building(x0, x1, z0, z1, h, **kw)))
    out.append(('bridge_1', bridge(G[0], G[1], Y0 + 58)))
    out.append(('bridge_2', bridge(G[5], G[8], Y0 + 70)))
    out.append(('bridge_3', bridge(G[3], G[6], Y0 + 48)))
    out.append(('bridge_4', bridge(G[7], G[8], Y0 + 82, w=8, h=6)))
    # four street trees at the inner corners of the blocks
    for k, (x, z) in enumerate([(50, -142), (148, -122), (50, -58), (148, -38)]):
        out.append(('tree_{}'.format(k + 1), tree(x, z, 72, rng, n_branch=2, reach=(14, 20))))
    return out


def scene_overgrown(rng):
    out = []
    G = tower_grid()
    hs = [112, 126, 98, 120, 90, 130, 104, 116, 124]
    kws = [dict(mast='++'), dict(porticos=('z+',)), dict(tank='+-'), dict(porticos=('x+', 'x-')), dict(setback=6),
           dict(porticos=('z-',), tank='-+'), dict(tank='++'), dict(mast='-+'), dict(porticos=('z-',))]
    for k, ((x0, x1, z0, z1), h, kw) in enumerate(zip(G, hs, kws)):
        out.append(('tower_{}'.format(k + 1), building(x0, x1, z0, z1, h, **kw)))
    out.append(('bridge_1', bridge(G[0], G[1], Y0 + 56)))
    out.append(('bridge_2', bridge(G[4], G[7], Y0 + 66)))
    out.append(('bridge_3', bridge(G[2], G[5], Y0 + 40)))
    for k, (a, b, y) in enumerate([(G[1], G[2], Y0 + 36), (G[3], G[4], Y0 + 52), (G[6], G[7], Y0 + 34)]):
        out.append(('cable_{}'.format(k + 1), cable(a, b, y)))
    # trees at the kerbs: trunks just off the block faces, 28-unit branches reaching across the 40 / 44-unit streets
    spots = [(48, -174, 118, 3), (150, -176, 126, 3), (61, -132, 130, 3), (137, -132, 112, 3),
             (150, -90, 128, 3), (61, -48, 116, 3), (48, -6, 120, 3), (99, -30, 118, 3)]
    for k, (x, z, top, nb) in enumerate(spots):
        out.append(('tree_{}'.format(k + 1), tree(x + rng.uniform(-2, 2), z + rng.uniform(-3, 3), top, rng,
                                                  n_branch=nb, reach=(20, 26))))
    return out


SCENE_FN = {'meadow': scene_meadow, 'suburb': scene_suburb, 'campus': scene_campus, 'downtown': scene_downtown,
            'overgrown': scene_overgrown}
SCENE_NOTES = {
    'meadow': 'five trees (crowns to y 105..118) and two sheds; 40 units of free air above the trees',
    'suburb': 'six houses 32..45 high along two streets, eight trees (to y 108..125) with 22-30-unit branches over the streets',
    'campus': 'six blocks 64..95 high, three sky bridges, six trees (to y 112..130) with 18-24-unit branches, porticos',
    'downtown': '3 x 3 tower grid 84..130 high (streets 40 / 44), four sky bridges, masts to y 152, porticos, four street trees',
    'overgrown': 'the tower grid 90..130 high with eight kerb trees (to y 112..130, 20-26-unit branches across the streets), '
                 'three cables and three sky bridges',
}


# ----------------------------------------------------------------------------- free-placement check
def surface_samples(V, F, n, rng):
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    k = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.uniform(size=(2, n))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return T[k, 0] + u[:, None] * (T[k, 1] - T[k, 0]) + v[:, None] * (T[k, 2] - T[k, 0])


def random_rotations(n, rng):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def free_fraction(body_V, body_F, scene_V, scene_F, n, rng, n_pts=300):
    """Fraction of uniformly random placements (centre in the room, random rotation) of the body whose surface samples
    (plus its vertices) all lie outside the scene solids -- a cheap lower bound of the collision-free volume."""
    import igl
    c = 0.5 * (body_V.min(0) + body_V.max(0))
    P = np.concatenate([surface_samples(body_V, body_F, n_pts, rng), body_V]) - c
    R = random_rotations(n, rng)
    t = np.stack([rng.uniform(X0, X1, n), rng.uniform(Y0, Y1, n), rng.uniform(Z0, Z1, n)], 1)
    Q = (np.einsum('nij,pj->npi', R, P) + t[:, None, :]).reshape(-1, 3)
    S = igl.signed_distance(np.ascontiguousarray(Q), scene_V, scene_F,
                            igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
    lo, hi = np.array([X0, Y0, Z0]), np.array([X1, Y1, Z1])
    bad = (S < 0) | (Q < lo).any(1) | (Q > hi).any(1)        # inside a solid or through the room shell
    return 1.0 - bad.reshape(n, -1).any(1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--shell', default=os.path.join(OUT_DIR, 'env3.obj'))
    ap.add_argument('--preview', default=None)
    ap.add_argument('--check', type=int, default=0, help='random placements per body x scene for the free fraction')
    ap.add_argument('--scenes', nargs='*', default=list(SCENES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
    hdr = 'written by dataprocessing/make_dronecam3d.py'

    # bodies
    bodies = {}
    Vd, Fd = ms.polycube_mesh(DRONE, merge=True)
    cd = 0.5 * (Vd.min(0) + Vd.max(0))
    ms.write_obj(os.path.join(a.out, 'drone_only.obj'), [('drone', Vd, Fd)],
                 ['the drone alone (visibility target of the camera score; frame centre of drone<L> / dronecam<L>x is '
                  'shifted back by L / 2 + 1 along x)', hdr])
    print('drone_only  {:3d} verts {:3d} tris  bbox {} .. {}'.format(len(Vd), len(Fd), Vd.min(0), Vd.max(0)))
    for L in DISTANCES:
        Vf, Ff = ms.polycube_mesh(follower_boxes(L), merge=True)
        vol, closed = ms.mesh_stats(Vf, Ff)
        assert closed
        name = 'drone{}'.format(L)
        ms.write_obj(os.path.join(a.out, name + '.obj'), [(name, Vf, Ff)],
                     ['{}: drone + 1-unit cube at the camera {} units behind the wing (rigid follower)'.format(name, L),
                      'bbox {} .. {}, outward-facing triangles'.format(Vf.min(0).tolist(), Vf.max(0).tolist()), hdr])
        cf = 0.5 * (Vf.min(0) + Vf.max(0))
        bodies[name] = (Vf, Ff)
        print('{:12s} {:3d} verts {:3d} tris  centre {}  volume {:.0f}'.format(name, len(Vf), len(Ff), cf, vol))
        for fov in ('t', 'w'):
            p = PLATES[fov]
            Vc, Fc = ms.polycube_mesh(body_boxes(fov), merge=True)
            Vc, Fc = attach_pyramid(Vc, Fc, p[0], apex_of(L))
            vol, closed = ms.mesh_stats(Vc, Fc)
            assert closed
            cc = 0.5 * (Vc.min(0) + Vc.max(0))
            assert np.allclose(cc, cf), (cc, cf)
            pyr = (p[3] - p[2]) * (p[5] - p[4]) * L / 3.0
            v_flat = ms.mesh_stats(*ms.polycube_mesh(body_boxes(fov), merge=True))[0]
            assert abs(vol - v_flat - pyr) < 1e-6 * pyr, (vol - v_flat, pyr)
            name = 'dronecam{}{}'.format(L, fov)
            fov_deg = [2 * np.degrees(np.arctan(0.5 * (p[5] - p[4]) / L)), 2 * np.degrees(np.arctan(0.5 * (p[3] - p[2]) / L))]
            ms.write_obj(os.path.join(a.out, name + '.obj'), [(name, Vc, Fc)],
                         ['{}: drone + plate + view pyramid, camera {} units behind the wing, base {} x {} '
                          '(field of view {:.1f} x {:.1f} degrees)'.format(name, L, p[5] - p[4], p[3] - p[2], *fov_deg),
                          'bbox {} .. {}, outward-facing triangles'.format(Vc.min(0).tolist(), Vc.max(0).tolist()), hdr])
            cam = {'apex': list(apex_of(L)), 'base': [[p[0], p[2], p[4]], [p[0], p[3], p[4]], [p[0], p[3], p[5]], [p[0], p[2], p[5]]],
                   'frame_centre': cc.tolist(), 'drone_centre': cd.tolist(), 'bird_obj': 'drone_only.obj',
                   'body_obj': name + '.obj', 'follower_obj': 'drone{}.obj'.format(L), 'distance': L, 'fov': fov,
                   'units': 'raw OBJ units', 'fov_deg': fov_deg}
            with open(os.path.join(a.out, name + '_camera.json'), 'w') as fh:
                json.dump(cam, fh, indent=1)
            bodies[name] = (Vc, Fc)
            print('{:12s} {:3d} verts {:3d} tris  centre {}  volume {:.0f}  fov {:.1f} x {:.1f}'.format(
                name, len(Vc), len(Fc), cc, vol, *fov_deg))
            if a.preview and L == 55:
                ms.render(os.path.join(a.preview, name + '.png'), [(name, Vc, Fc, 0.6)], name)
    if a.preview:
        ms.render(os.path.join(a.preview, 'drone55.png'), [('drone55', bodies['drone55'][0], bodies['drone55'][1], 0.9)], 'drone55')

    # scenes
    shell_V, shell_F = ms.read_obj_group(a.shell, 'wall (1)')
    scenes = {}
    for name in a.scenes:
        rng = np.random.default_rng(SCENES.index(name) + 11)
        obstacles = SCENE_FN[name](rng)
        groups = [('wall (1)', shell_V, shell_F)]
        preview = [('wall (1)', shell_V, shell_F, 0.03)]
        boxes_json = []
        for gname, bp in obstacles:
            boxes = [b for b, _ in bp]
            V, F = soup(boxes)
            groups.append((gname, V, F))
            preview.append((gname, V, F, 0.75))
            boxes_json.append({'name': gname, 'kind': gname.split('_')[0], 'boxes': [list(map(float, b)) for b in boxes],
                               'parts': [part for _, part in bp]})
        tmp = os.path.join(a.out, name + '.obj.tmp')       # atomic: the queue may be reading the scene
        ms.write_obj(tmp, groups, ['{}: env3 room shell + {}'.format(name, SCENE_NOTES[name]), hdr])
        os.replace(tmp, os.path.join(a.out, name + '.obj'))
        with open(os.path.join(a.out, name + '_boxes.json'), 'w') as fh:
            json.dump({'scene': name, 'note': SCENE_NOTES[name], 'room': [X0, X1, Y0, Y1, Z0, Z1], 'groups': boxes_json}, fh)
        V_all = np.concatenate([g[1] for g in groups])
        F_all = np.concatenate([g[2] + off for g, off in zip(groups, np.cumsum([0] + [len(g[1]) for g in groups[:-1]]))])
        scenes[name] = (V_all, F_all)
        print('{:10s} {:3d} obstacles, {:4d} boxes, {:5d} triangles'.format(
            name, len(obstacles), sum(len(bp) for _, bp in obstacles), len(F_all)))
        if a.preview:
            ms.render(os.path.join(a.preview, name + '.png'), preview, name, elev=30, azim=-55)

    if a.check:
        rng = np.random.default_rng(0)
        names = [n for L in DISTANCES for n in ('drone{}'.format(L), 'dronecam{}t'.format(L), 'dronecam{}w'.format(L))]
        print('\nfree fraction of {} random placements (%)'.format(a.check))
        print('{:10s}'.format('scene') + ''.join('{:>12s}'.format(n) for n in names))
        for sname, (V, F) in scenes.items():
            row = []
            for n in names:
                row.append(100 * free_fraction(bodies[n][0], bodies[n][1], V, F, a.check, rng))
            print('{:10s}'.format(sname) + ''.join('{:12.1f}'.format(v) for v in row), flush=True)


if __name__ == '__main__':
    main()
