#!/usr/bin/env python3
"""Wall-with-hole environments: the room of env1 (its 'wall' group, so the normalized frame is the one of the
standard Tshape3d_env1 / Lshape3d / Ashape3d cells, env_scale 210.25) divided by a slab of thickness --thick,
perpendicular to z at the room's z centre, with a centred square hole of side --sides (raw OBJ units; the three
shapes are 60 units long and 10 thick, so a hole below 60 is impassable for all of them). The slab is written as
four non-overlapping boxes around the hole (obstacle groups, names without 'wall' / 'null').

    python3 models/graph_homotopy/scripts/make_wall_envs.py --sides 90,75,66 --out datasets/3dshape/obj

writes datasets/3dshape/obj/hole90.obj, hole75.obj, hole66.obj (standard library only).
"""
import argparse
import os

BOX_TRIS = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
            [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]]


def box_corners(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    return [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0],
            [x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]]


def wall_box(path):
    """bbox of the vertices used by faces of the group whose name contains 'wall'."""
    V, groups, cur = [], {}, None
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == 'v':
            V.append([float(x) for x in t[1:4]])
        elif t[0] in ('g', 'o'):
            cur = ' '.join(t[1:])
        elif t[0] == 'f':
            groups.setdefault(cur, set()).update(int(x.split('/')[0]) - 1 for x in t[1:])
    walls = [g for g in groups if g and 'wall' in g.lower()]
    assert walls, 'no wall group in ' + path
    idx = sorted(set().union(*[groups[g] for g in walls]))
    P = [V[i] for i in idx]
    return [min(p[k] for p in P) for k in range(3)], [max(p[k] for p in P) for k in range(3)]


def write_env(out, lo, hi, side, thick):
    xc, yc, zc = [(lo[k] + hi[k]) / 2.0 for k in range(3)]
    z0, z1 = zc - thick / 2.0, zc + thick / 2.0
    h = side / 2.0
    pieces = [('Body_frame_left', [lo[0], lo[1], z0], [xc - h, hi[1], z1]),
              ('Body_frame_right', [xc + h, lo[1], z0], [hi[0], hi[1], z1]),
              ('Body_frame_bottom', [xc - h, lo[1], z0], [xc + h, yc - h, z1]),
              ('Body_frame_top', [xc - h, yc + h, z0], [xc + h, hi[1], z1])]
    lines = ['# room of env1 with a dividing wall (z-normal, thickness {}) and a centred square hole of side {}\n'.format(thick, side)]
    base = 0
    for name, a, b in pieces + [('wall (1)', lo, hi)]:
        lines.append('\ng {}\n\n'.format(name))
        for x, y, z in box_corners(a, b):
            lines.append('v {:.6f} {:.6f} {:.6f}\n'.format(x, y, z))
        lines.append('\n')
        for p, q, r in BOX_TRIS:
            lines.append('f {} {} {}\n'.format(base + p + 1, base + q + 1, base + r + 1))
        base += 8
    with open(out, 'w') as fh:
        fh.writelines(lines)
    return dict(center=[xc, yc, zc], hole=[xc - h, xc + h, yc - h, yc + h], slab_z=[z0, z1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--room', default='datasets/3dshape/obj/env1.obj')
    ap.add_argument('--sides', default='90,75,66')
    ap.add_argument('--thick', type=float, default=8.0)
    ap.add_argument('--out', default='datasets/3dshape/obj')
    a = ap.parse_args()
    lo, hi = wall_box(a.room)
    print('room {} .. {}'.format([round(x, 3) for x in lo], [round(x, 3) for x in hi]))
    for s in a.sides.split(','):
        side = float(s)
        out = os.path.join(a.out, 'hole{}.obj'.format(s))
        info = write_env(out, lo, hi, side, a.thick)
        print('{}: hole x {:.1f}..{:.1f} y {:.1f}..{:.1f}, slab z {:.1f}..{:.1f}'.format(
            out, *info['hole'], *info['slab_z']))


if __name__ == '__main__':
    main()
