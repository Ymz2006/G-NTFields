"""Write the moving-sofa problem as OBJ meshes for the planar (--2d) pipeline.

The sofa is a U (a box-only stand-in for Hammersley's sofa: a long body with a
notch on its inner side that the hallway's inner corner slides into while it
turns), and the environments are a 90-degree turn of a corridor of width W:

    Usofa2d    60 x 20 U, 24 x 10 notch centred on one long side (opening +y),
               so arms 18 x 10 and a 60 x 10 base
    2dhall44   W = 44  easy    turns either way round (notch toward the inner
                               corner: clearance 9.0; notch outward: 5.4)
    2dhall34   W = 34  medium  turns only with the notch toward the inner corner
                               (clearance 3.5); its 60 x 20 bounding box cannot
    2dhall30   W = 30  hard    as 2dhall34 with clearance 1.4 (blocked below ~29)

"Clearance" is the largest c for which a path exists with every placement at
least c from the walls (env units; dataprocessing/check_sofa2d.py computes it).

Both legs run 150 units from the outer corner (the sofa fits straight in each
leg), closed at the far ends; the walls are the 170 x 170 box minus the
corridor, so there is no free space outside it.  Everything is a z-prism (the
planar slice is z = 0, as preprocess_obj.py --2d expects z-up meshes): the walls
are 30 high, the sofa 10 (--2d squashes it to a thin slab anyway).

    python dataprocessing/make_sofa2d.py                # writes datasets/3dshape/obj/*.obj
    python dataprocessing/make_sofa2d.py --preview out/ # also renders PNG previews
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_shapes3d as ms  # noqa: E402

OUT_DIR = ms.OUT_DIR
LEG = 150.0          # leg length from the outer corner (0, 0)
WALL = 10.0          # outer wall thickness
ENV_Z = (-15.0, 15.0)
SOFA_Z = (-5.0, 5.0)

# Planar boxes (x0, x1, y0, y1).
SOFA = {
    'Usofa2d': {'length': 60.0, 'depth': 20.0, 'notch_w': 24.0, 'notch_d': 10.0},
}
HALLS = {'2dhall44': 44.0, '2dhall34': 34.0, '2dhall30': 30.0}
HALL_NOTE = {'2dhall44': 'easy: turns either way round',
             '2dhall34': 'medium: turns only notch-in (clearance 3.5)',
             '2dhall30': 'hard: turns only notch-in (clearance 1.4)'}


def sofa_boxes(length, depth, notch_w, notch_d):
    """U centred on its bounding box, base along -y, notch opening +y."""
    a = 0.5 * (length - notch_w)
    L, h = 0.5 * length, 0.5 * depth
    if notch_w <= 0 or notch_d <= 0:
        return [(-L, L, -h, h)]
    return [(-L, L, -h, h - notch_d), (-L, -L + a, h - notch_d, h), (L - a, L, h - notch_d, h)]


def hall_boxes(W, leg=LEG, t=WALL):
    """The (-t, leg + t)^2 box minus the corridor {0<=y<=W, 0<=x<=leg} u {0<=x<=W, 0<=y<=leg}."""
    return [(-t, leg + t, -t, 0),          # outer wall of the x leg
            (-t, 0, 0, leg + t),           # outer wall of the y leg
            (leg, leg + t, 0, W),          # end of the x leg
            (0, W, leg, leg + t),          # end of the y leg
            (W, leg + t, W, leg + t)]      # inner block


def prism(boxes, z):
    return [(b[0], b[1], b[2], b[3], z[0], z[1]) for b in boxes]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--preview', default=None, help='Directory for PNG previews.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    for name, p in SOFA.items():
        V, F = ms.polycube_mesh(prism(sofa_boxes(**p), SOFA_Z), merge=True)
        vol, closed = ms.mesh_stats(V, F)
        ms.write_obj(os.path.join(args.out, name + '.obj'), [(name, V, F)],
                     ['{}: {:g} x {:g} U, {:g} x {:g} notch opening +y, z-prism {:g} high'.format(
                         name, p['length'], p['depth'], p['notch_w'], p['notch_d'], SOFA_Z[1] - SOFA_Z[0]),
                      'written by dataprocessing/make_sofa2d.py'])
        print('{:9s} {:3d} verts {:3d} tris  volume {:.0f}  closed {}'.format(name, len(V), len(F), vol, closed))
        if args.preview:
            ms.render(os.path.join(args.preview, name + '.png'), [(name, V, F, 0.9)], name)

    for name, W in HALLS.items():
        V, F = ms.polycube_mesh(prism(hall_boxes(W), ENV_Z))
        vol, closed = ms.mesh_stats(V, F)
        ms.write_obj(os.path.join(args.out, name + '.obj'), [('walls', V, F)],
                     ['{}: 90-degree corridor turn, width {:g}, legs {:g} -- {}'.format(
                         name, W, LEG, HALL_NOTE[name]),
                      'written by dataprocessing/make_sofa2d.py'])
        print('{:9s} {:3d} verts {:4d} tris  volume {:.0f}  closed {}'.format(name, len(V), len(F), vol, closed))
        if args.preview:
            ms.render(os.path.join(args.preview, name + '_top.png'), [('walls', V, F, 0.55)],
                      name + ' (top, z up)', elev=90, azim=-90)


if __name__ == '__main__':
    main()
