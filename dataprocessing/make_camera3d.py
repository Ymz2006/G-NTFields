"""The camera-follow bodies and the two scenes of the camera + object test (2026-10-21).

A bird-like glider (polycube: fuselage, head, wings, tail plane, fin; 48 x 18 x 48 units, flying along +x) is
followed by a camera, a point 50 units behind it.  What the camera sees is a rectangle-based pyramid with its
tip at the camera and its base, 48 x 22 (a 2-unit plate covering the bird's silhouette with a 2-unit margin),
pressed against the bird's back, so the pyramid is the camera's view volume up to the bird (51 x 25 degree
field of view).  Two bodies in the same frame (same bounding box, so the frame of one is the frame of the
other and models of either can be scored on the same test pairs):

    bird3d      the bird plus a 1-unit cube at the camera (the rigid follower with no view model)
    birdcam3d   the bird, the plate and the pyramid: planned together, a collision-free path keeps the bird,
                the camera and the whole line of sight clear of the scene

Both are watertight and tetrahedralize as the shapes of make_shapes3d.py (bird3d is two components).
``birdcam3d_camera.json`` records the camera tip, the base rectangle and the bird mesh for the visibility
score of plan_eval_homotopy.py --camera.

Scenes (the room shell of env3, so the normalization is the one of all the shape cells; y is up):

    forest   four trees on a jittered 2 x 2 grid: a 12 x 12 trunk from the floor to y = 96 and a 30-unit crown
             (y 90..120); crown gaps 46 (x) / 80 (z), 35 units above the crowns.  The body passes between the
             crowns along x only rolled (its 22-unit side across the gap) and flies above the crowns only flat.
    city     six buildings 48..60 wide, 45..100 high in two rows along x; streets 40 (rolled only), 56 and 60
             wide, the low roofs reachable flat, 38 units above the tallest.

    python dataprocessing/make_camera3d.py                 # writes datasets/3dshape/obj/{bird3d,birdcam3d,forest,city}.obj
    python dataprocessing/make_camera3d.py --preview out/  # also renders PNG previews
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_shapes3d as ms  # noqa: E402

OUT_DIR = ms.OUT_DIR
# Boxes are (x0, x1, y0, y1, z0, z1); the bird's bounding box is x 0..48, y 10..28, z -4..44.
BIRD = [
    (0, 40, 10, 20, 15, 25),      # fuselage
    (40, 48, 10, 18, 15, 25),     # head
    (14, 26, 16, 20, -4, 44),     # wings (chord 12, 4 thick, span 48, flush with the fuselage top)
    (0, 8, 16, 20, -4, 44),       # tail plane (full span, so the plate's front face stays hole-free)
    (0, 8, 20, 28, 18, 22),       # fin
]
# (every box is flush with a face edge of the box it meets, so no polycube face has a hole)
PLATE = (-2, 0, 8, 30, -4, 44)    # base of the view pyramid, pressed on the bird's back
CAM_L = 50.0                      # camera distance behind the plate
APEX = (PLATE[0] - CAM_L, 0.5 * (PLATE[2] + PLATE[3]), 0.5 * (PLATE[4] + PLATE[5]))
CAM_CUBE = (APEX[0], APEX[0] + 1, APEX[1] - 0.5, APEX[1] + 0.5, APEX[2] - 0.5, APEX[2] + 0.5)

X0, X1, Y0, Y1, Z0, Z1 = ms.X0, ms.X1, ms.Y0, ms.Y1, ms.Z0, ms.Z1


def trees(rng):
    """(name, boxes) of the forest: trunk 12 x 12 to y 96, crown 30 x 30 x 30 (y 90..120).

    Four trees on a jittered 2 x 2 grid: with six (2 x 3, 36-unit crowns) only 0.4 % of uniformly random
    placements of birdcam3d were free, against 4.3 % in env9; this layout gives 3.5 % (city 4.5 %)."""
    out = []
    k = 0
    for xc in (62.0, 138.0):
        for zc in (-145.0, -35.0):
            x, z = xc + rng.uniform(-5, 5), zc + rng.uniform(-5, 5)
            k += 1
            out.append(('tree_{}'.format(k), [(x - 6, x + 6, Y0, 96.0, z - 6, z + 6),
                                               (x - 15, x + 15, 90.0, 120.0, z - 15, z + 15)]))
    return out


def buildings():
    """(name, boxes) of the city: two rows of three along x (streets 56 / 40 / 60 / 40 units)."""
    rows = [((-195.0, -155.0), [(10, 58, 100), (114, 162, 60)]),
            ((-95.0, -55.0), [(1.12, 40, 45), (96, 150, 85)]),
            ((-15.0, 15.0), [(10, 66, 70), (122, 182, 95)])]
    out = []
    k = 0
    for (z0, z1), bs in rows:
        for x0, x1, h in bs:
            k += 1
            out.append(('building_{}'.format(k), [(x0, x1, Y0, Y0 + h, z0, z1)]))
    return out


def attach_pyramid(V, F, x_plane, apex):
    """Replace the face of the mesh in the plane x = x_plane (a rectangle: 2 triangles on 4 vertices) by the
    four sides of the pyramid from its boundary to ``apex`` (outward orientation)."""
    on = np.isclose(V[:, 0], x_plane)
    face = np.where(on[F].all(1))[0]
    assert len(face) == 2, 'rear face has {} triangles'.format(len(face))
    corners = np.unique(F[face])
    assert len(corners) == 4
    # order the corners around the rectangle (y, z plane)
    c = V[corners]
    ang = np.arctan2(c[:, 2] - c[:, 2].mean(), c[:, 1] - c[:, 1].mean())
    corners = corners[np.argsort(ang)]
    V2 = np.concatenate([V, np.asarray(apex, np.float64)[None]], 0)
    a = len(V)
    centroid = 0.5 * (np.asarray(apex) + c.mean(0))
    sides = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        tri = [p, q, a]
        n = np.cross(V2[q] - V2[p], V2[a] - V2[p])
        if np.dot(n, V2[p] - centroid) < 0:
            tri = [q, p, a]
        sides.append(tri)
    F2 = np.concatenate([np.delete(F, face, 0), np.asarray(sides, np.int64)], 0)
    return V2, F2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--shell', default=os.path.join(OUT_DIR, 'env3.obj'))
    ap.add_argument('--preview', default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)

    # bodies
    Vb, Fb = ms.polycube_mesh(BIRD + [CAM_CUBE], merge=True)
    Vc, Fc = ms.polycube_mesh(BIRD + [PLATE], merge=True)
    Vc, Fc = attach_pyramid(Vc, Fc, PLATE[0], APEX)
    bird_only_V, bird_only_F = ms.polycube_mesh(BIRD, merge=True)
    base_area = (PLATE[3] - PLATE[2]) * (PLATE[5] - PLATE[4])
    for name, V, F, expect in (('bird3d', Vb, Fb, None), ('birdcam3d', Vc, Fc, base_area * CAM_L / 3.0)):
        vol, closed = ms.mesh_stats(V, F)
        lo, hi = V.min(0), V.max(0)
        print('{:10s} {:3d} verts {:3d} tris  bbox {} .. {}  centre {}  volume {:.0f}  closed {}'.format(
            name, len(V), len(F), lo, hi, 0.5 * (lo + hi), vol, closed))
        assert closed
        if expect is not None:
            vp = vol - ms.mesh_stats(*ms.polycube_mesh(BIRD + [PLATE], merge=True))[0]
            assert abs(vp - expect) < 1e-6 * expect, (vp, expect)
        ms.write_obj(os.path.join(a.out, name + '.obj'), [(name, V, F)],
                     ['{}: {}'.format(name, 'bird + 1-unit cube at the camera' if name == 'bird3d'
                                      else 'bird + plate + view pyramid (tip = camera)'),
                      'bbox {} .. {}, outward-facing triangles'.format(lo.tolist(), hi.tolist()),
                      'written by dataprocessing/make_camera3d.py'])
    cb, cc = 0.5 * (Vb.min(0) + Vb.max(0)), 0.5 * (Vc.min(0) + Vc.max(0))
    assert np.allclose(cb, cc), (cb, cc)
    ms.write_obj(os.path.join(a.out, 'bird3d_only.obj'), [('bird', bird_only_V, bird_only_F)],
                 ['the bird alone (visibility target of the camera score; frame = bird3d / birdcam3d bbox centre '
                  + str(cc.tolist()) + ')', 'written by dataprocessing/make_camera3d.py'])
    cam = {'apex': list(APEX), 'base': [[PLATE[0], PLATE[2], PLATE[4]], [PLATE[0], PLATE[3], PLATE[4]],
                                        [PLATE[0], PLATE[3], PLATE[5]], [PLATE[0], PLATE[2], PLATE[5]]],
           'frame_centre': cc.tolist(), 'bird_obj': 'bird3d_only.obj', 'body_obj': 'birdcam3d.obj',
           'follower_obj': 'bird3d.obj', 'units': 'raw OBJ units of bird3d.obj / birdcam3d.obj',
           'fov_deg': [2 * np.degrees(np.arctan(0.5 * (PLATE[5] - PLATE[4]) / CAM_L)),
                       2 * np.degrees(np.arctan(0.5 * (PLATE[3] - PLATE[2]) / CAM_L))]}
    with open(os.path.join(a.out, 'birdcam3d_camera.json'), 'w') as fh:
        json.dump(cam, fh, indent=1)
    print('camera', cam)

    # scenes
    shell_V, shell_F = ms.read_obj_group(a.shell, 'wall (1)')
    rng = np.random.default_rng(7)
    for name, bodies, note in (('forest', trees(rng), 'four trees (trunk 12 x 12 to y 96, crown 30 x 30 x 30 at y 90..120)'),
                               ('city', buildings(), 'six buildings in two rows, 45..100 high, streets 40 / 56 / 60')):
        groups = [('wall (1)', shell_V, shell_F)]
        preview = [('wall (1)', shell_V, shell_F, 0.04)]
        for gname, boxes in bodies:
            V, F = ms.polycube_mesh(boxes)       # grid quads: a crown's underside has the trunk inside it
            groups.append((gname, V, F))
            preview.append((gname, V, F, 0.8))
        ms.write_obj(os.path.join(a.out, name + '.obj'), groups,
                     ['{}: env3 room shell + {}'.format(name, note), 'written by dataprocessing/make_camera3d.py'])
        print('{:8s} {} obstacles'.format(name, len(bodies)))
        if a.preview:
            ms.render(os.path.join(a.preview, name + '.png'), preview, name, elev=28, azim=-50)
    if a.preview:
        ms.render(os.path.join(a.preview, 'birdcam3d.png'), [('birdcam3d', Vc, Fc, 0.6)], 'birdcam3d')
        ms.render(os.path.join(a.preview, 'bird3d.png'), [('bird3d', Vb, Fb, 0.9)], 'bird3d')


if __name__ == '__main__':
    main()
