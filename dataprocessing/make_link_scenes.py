"""The four scenes of the drone-pair link test (2026-10-22): the forest of the camera test and three new ones.

All in env3's room shell (the normalization of every shape cell, scale 220.457; y is up; interior x 1.12..199.24,
y 17.09..155.09, z -199.14..19.10), obstacles axis-aligned boxes. Each scene breaks the line of sight between two
drones in a different way:

    forest     dataprocessing/make_camera3d.py's forest, unchanged (four trees on a jittered 2 x 2 grid: 12 x 12
               trunks to y = 96, 30-unit crowns at y 90..120); only its box list is written here
    windows    a partition wall (z in [-95, -85]) with two 40 x 40 windows, one low (x 30..70, y 40..80), one high
               (x 130..170, y 95..135): two rooms that see each other only through the windows
    corridor   two full-height walls along x, z in [-131, -123] open at x > 145 and z in [-56, -48] open at x < 55:
               one serpentine corridor of three 67-unit legs joined by 54-unit turns
    scaffold   a lattice: 3 x 3 columns 8 x 8 (x 40 / 100 / 160, z -160 / -90 / -20) and 8 x 8 beams across the
               room, at y 60..68 along x (z -160, -20) and along z (x 100), at y 110..118 along x (z -90) and along
               z (x 40, 160)

Writes datasets/3dshape/obj/<scene>.obj for the three new scenes and <scene>_boxes.json for all four (raw boxes
(x0, x1, y0, y1, z0, z1) with their group names; models/drone_link/geometry.Scene reads them).

    python dataprocessing/make_link_scenes.py                 # writes the OBJ / JSON files
    python dataprocessing/make_link_scenes.py --preview out/  # also renders PNG previews
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_camera3d as mc  # noqa: E402
import make_shapes3d as ms  # noqa: E402

OUT_DIR = ms.OUT_DIR
X0, X1, Y0, Y1, Z0, Z1 = ms.X0, ms.X1, ms.Y0, ms.Y1, ms.Z0, ms.Z1


def forest():
    """The forest's trees exactly as make_camera3d.py draws them (np.random.default_rng(7), first call)."""
    return mc.trees(np.random.default_rng(7))


def windows():
    return [('partition', ms.wall_with_windows('z', -95, -85, [(30, 70, 40, 80), (130, 170, 95, 135)]))]


def corridor():
    return [('partition_back', [(X0, 145, Y0, Y1, -131, -123)]),
            ('partition_front', [(55, X1, Y0, Y1, -56, -48)])]


def scaffold():
    cols = [(x - 4, x + 4, Y0, Y1, z - 4, z + 4) for x in (40.0, 100.0, 160.0) for z in (-160.0, -90.0, -20.0)]
    beams = [(X0, X1, 60, 68, z - 4, z + 4) for z in (-160.0, -20.0)] + [(96, 104, 60, 68, Z0, Z1)]
    beams += [(X0, X1, 110, 118, -94, -86)] + [(x - 4, x + 4, 110, 118, Z0, Z1) for x in (40.0, 160.0)]
    return [('scaffold', cols + beams)]


SCENES = {
    'forest': (forest, 'four trees (trunk 12 x 12 to y 96, crown 30 x 30 x 30 at y 90..120)'),
    'windows': (windows, 'partition wall z -95..-85 with two 40 x 40 windows (low x 30..70 y 40..80, high x 130..170 '
                         'y 95..135)'),
    'corridor': (corridor, 'serpentine corridor: walls z -131..-123 (x < 145) and z -56..-48 (x > 55), full height'),
    'scaffold': (scaffold, 'lattice of 3 x 3 columns 8 x 8 and 8 x 8 beams at y 60..68 / 110..118'),
}


def check_forest(boxes, path):
    """The regenerated trees must be the ones in forest.obj (every polycube vertex is a box corner)."""
    V = []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
    V = np.asarray(V)
    b = np.asarray(boxes)
    inside = np.zeros(len(V), bool)
    for x0, x1, y0, y1, z0, z1 in b:
        inside |= ((V[:, 0] >= x0 - 1e-4) & (V[:, 0] <= x1 + 1e-4) & (V[:, 1] >= y0 - 1e-4) & (V[:, 1] <= y1 + 1e-4)
                   & (V[:, 2] >= z0 - 1e-4) & (V[:, 2] <= z1 + 1e-4))
    shell = (V[:, 0] < X0 + 1e-3) | (V[:, 0] > X1 - 1e-3) | (V[:, 1] < Y0 + 1e-3) | (V[:, 1] > Y1 - 1e-3) \
        | (V[:, 2] < Z0 + 1e-3) | (V[:, 2] > Z1 - 1e-3)
    assert (inside | shell).all(), 'forest.obj has vertices outside the regenerated trees'
    for x0, x1, y0, y1, z0, z1 in b:     # every box corner is a vertex
        for c in [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]:
            if not (np.abs(V - np.asarray(c)).max(1) < 1e-4).any():
                # corners buried in the union (a trunk top inside its crown) are not vertices
                assert any(bb[0] < c[0] < bb[1] and bb[2] < c[1] < bb[3] and bb[4] < c[2] < bb[5]
                           for bb in b), c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--shell', default=os.path.join(OUT_DIR, 'env3.obj'))
    ap.add_argument('--preview', default=None)
    a = ap.parse_args()
    shell_V, shell_F = ms.read_obj_group(a.shell, 'wall (1)')
    if a.preview:
        os.makedirs(a.preview, exist_ok=True)
    for name, (fn, note) in SCENES.items():
        bodies = fn()
        groups = [('wall (1)', shell_V, shell_F)]
        preview = [('wall (1)', shell_V, shell_F, 0.04)]
        for gname, boxes in bodies:
            V, F = ms.polycube_mesh(boxes)
            groups.append((gname, V, F))
            preview.append((gname, V, F, 0.8))
        obj = os.path.join(a.out, name + '.obj')
        if name == 'forest':
            check_forest([b for _, bs in bodies for b in bs], obj)
        else:
            ms.write_obj(obj, groups, ['{}: env3 room shell + {}'.format(name, note),
                                       'written by dataprocessing/make_link_scenes.py'])
        js = {'scene': name, 'note': note, 'units': 'raw OBJ units, rows (x0, x1, y0, y1, z0, z1)',
              'boxes': [list(map(float, b)) for _, bs in bodies for b in bs],
              'groups': [g for g, bs in bodies for _ in bs], 'written_by': 'dataprocessing/make_link_scenes.py'}
        with open(os.path.join(a.out, name + '_boxes.json'), 'w') as fh:
            json.dump(js, fh, indent=1)
        print('{:9s} {:2d} boxes in {} groups'.format(name, len(js['boxes']), len(bodies)))
        if a.preview:
            ms.render(os.path.join(a.preview, name + '.png'), preview, name, elev=28, azim=-50)


if __name__ == '__main__':
    main()
