"""Write the non-extruded 3-D shapes and the tight environments as OBJ meshes.

Every mesh here is a union of axis-aligned boxes (a polycube on a non-uniform
grid), so it is watertight, has outward-facing triangles and tetrahedralizes into
a few dozen tets -- the same kind of mesh as Tshape3d / Lshape3d / Ashape3d.
Unlike those letters, which are 2-D drawings extruded 10 units, these shapes
have arms along all three axes:

    tripod3d   three orthogonal arms meeting at a corner (a branch point)
    crank3d    a chain of three bars bent along x -> y -> z (chiral)
    loop3d     the closed skew hexagon on six edges of a cube (genus 1)
    twistI3d   two crossbars at 90 degrees joined by a stem
    chair3d    seat, back and four legs (pockets between the legs)

Bars are 10 x 10 units and every shape fits in 40 x 40 x 40 (the chair is
30 x 30 x 50), against 60 x 10 x 60 for Tshape3d.

The environments keep env3/env4's room shell (group ``wall (1)``, copied from
env3.obj), so they share its bounding box and normalization (scale 220.457), and
add new obstacles:

    env5   a partition wall across the room with one 50 x 50 window, plus a
           40-unit block in each half
    env6   two crossing partition walls cut the room into four rooms
           (about 94 x 138 x 104) joined in a ring by four 50 x 50 windows,
           alternating low / high
    env7   env5 with a 60 x 60 window   (window-size ladder between env4 and env5)
    env8   env5 with a 70 x 70 window
    env9   env6 with four 70 x 70 windows (same centres)

Every shape passes each window in an axis-aligned pose with 5 units of clearance
per side or more, and turns freely in each room (bounding radius <= 34.7), so
start and goal are connected whenever both lie in the rooms
(dataprocessing/check_shapes3d.py verifies both).

    python dataprocessing/make_shapes3d.py                # writes datasets/3dshape/obj/*.obj
    python dataprocessing/make_shapes3d.py --preview out/ # also renders PNG previews
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'datasets', '3dshape', 'obj')

# Boxes are (x0, x1, y0, y1, z0, z1).
SHAPES = {
    'tripod3d': [
        (0, 10, 0, 10, 0, 10),
        (10, 40, 0, 10, 0, 10),
        (0, 10, 10, 40, 0, 10),
        (0, 10, 0, 10, 10, 40),
    ],
    'crank3d': [
        (0, 40, 0, 10, 0, 10),
        (30, 40, 10, 40, 0, 10),
        (30, 40, 30, 40, 10, 40),
    ],
    'loop3d': [
        (0, 40, 0, 10, 0, 10),        # +x along y=0, z=0
        (30, 40, 0, 40, 0, 10),       # +y along x=40, z=0
        (30, 40, 30, 40, 0, 40),      # +z along x=40, y=40
        (0, 40, 30, 40, 30, 40),      # -x along y=40, z=40
        (0, 10, 0, 40, 30, 40),       # -y along x=0, z=40
        (0, 10, 0, 10, 0, 40),        # -z along x=0, y=0
    ],
    'twistI3d': [
        (15, 25, 15, 25, 10, 30),     # stem
        (0, 40, 15, 25, 30, 40),      # top crossbar along x
        (15, 25, 0, 40, 0, 10),       # bottom crossbar along y
    ],
    'chair3d': [
        (0, 10, 0, 10, 0, 15),
        (20, 30, 0, 10, 0, 15),
        (0, 10, 20, 30, 0, 15),
        (20, 30, 20, 30, 0, 15),
        (0, 30, 0, 30, 15, 25),       # seat
        (0, 30, 20, 30, 25, 50),      # back
    ],
}

SHAPE_NOTES = {
    'tripod3d': 'three orthogonal 10x10 arms, 40 long, meeting at a corner',
    'crank3d': 'chain of three 10x10 bars bent x -> y -> z (chiral)',
    'loop3d': 'closed skew hexagon on six edges of a 40-cube (genus 1)',
    'twistI3d': 'two 40-long crossbars at 90 degrees joined by a stem',
    'chair3d': 'chair: 30x30 seat, 25-high back, four 15-high legs',
}

# Interior of the env3/env4 room shell (inner faces of 'wall (1)'); y is up.
X0, X1 = 1.12, 199.24
Y0, Y1 = 17.09, 155.09
Z0, Z1 = -199.14, 19.10


def wall_with_windows(axis, lo, hi, windows):
    """Boxes of a full-room partition slab ``lo <= axis <= hi`` with rectangular
    windows cut out.  ``axis`` is 'z' (slab spans x, y) or 'x' (slab spans z, y);
    each window is (u0, u1, y0, y1) in the slab's in-plane coordinates (u = x for
    a z-slab, u = z for an x-slab)."""
    u_lo, u_hi = (X0, X1) if axis == 'z' else (Z0, Z1)
    us = sorted({u_lo, u_hi, *[w[0] for w in windows], *[w[1] for w in windows]})
    ys = sorted({Y0, Y1, *[w[2] for w in windows], *[w[3] for w in windows]})
    boxes = []
    for u0, u1 in zip(us[:-1], us[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            um, ym = 0.5 * (u0 + u1), 0.5 * (y0 + y1)
            if any(w[0] < um < w[1] and w[2] < ym < w[3] for w in windows):
                continue
            boxes.append((u0, u1, y0, y1, lo, hi) if axis == 'z' else (lo, hi, y0, y1, u0, u1))
    return boxes


ENVS = {
    'env5': {
        'note': 'env3 room shell + partition wall (z in [-95,-85]) with one 50x50 window + two 40-unit blocks',
        'bodies': {
            'partition': wall_with_windows('z', -95, -85, [(75, 125, 55, 105)]),
            'block_front': [(10, 50, Y0, Y0 + 40, -30, 10)],
            'block_back': [(150, 190, 100, 140, -180, -140)],
        },
    },
    # env5 with a wider window (same centre x 100, y 80): the window-size ladder.
    'env7': {
        'note': 'env5 with a 60x60 window (env3 room shell + partition z in [-95,-85] + two 40-unit blocks)',
        'bodies': {
            'partition': wall_with_windows('z', -95, -85, [(70, 130, 50, 110)]),
            'block_front': [(10, 50, Y0, Y0 + 40, -30, 10)],
            'block_back': [(150, 190, 100, 140, -180, -140)],
        },
    },
    'env8': {
        'note': 'env5 with a 70x70 window (env3 room shell + partition z in [-95,-85] + two 40-unit blocks)',
        'bodies': {
            'partition': wall_with_windows('z', -95, -85, [(65, 135, 45, 115)]),
            'block_front': [(10, 50, Y0, Y0 + 40, -30, 10)],
            'block_back': [(150, 190, 100, 140, -180, -140)],
        },
    },
    'env6': {
        'note': 'env3 room shell + two crossing partitions (z in [-95,-85], x in [95,105]) -> four rooms '
                'joined in a ring by four 50x50 windows (low / high)',
        # One body, so the crossing slabs have no faces inside each other.
        'bodies': {
            'partitions': wall_with_windows('z', -95, -85, [(127, 177, 90, 140),     # front-right <-> back-right
                                                            (23, 73, 90, 140)])      # back-left <-> front-left
                          + wall_with_windows('x', 95, 105, [(-51, -1, 35, 85),      # front-left <-> front-right
                                                             (-167, -117, 35, 85)]),  # back-right <-> back-left
        },
    },
    # env6 with every window grown 10 units per side (same centres): 70x70.
    'env9': {
        'note': 'env6 with 70x70 windows (env3 room shell + two crossing partitions (z in [-95,-85], x in [95,105]) '
                '-> four rooms joined in a ring, low / high)',
        'bodies': {
            'partitions': wall_with_windows('z', -95, -85, [(117, 187, 80, 150),     # front-right <-> back-right
                                                            (13, 83, 80, 150)])      # back-left <-> front-left
                          + wall_with_windows('x', 95, 105, [(-61, 9, 25, 95),       # front-left <-> front-right
                                                             (-177, -107, 25, 95)]),  # back-right <-> back-left
        },
    },
}


def polycube_mesh(boxes, merge=False):
    """Watertight boundary of a union of axis-aligned boxes.

    The union is voxelized on the grid of all box coordinates; every face between
    a filled and an empty cell becomes one quad (two triangles) with its normal
    pointing out of the solid.  Each boundary quad is exactly one grid face, so
    there are no T-junctions and every edge is shared by two triangles (the
    shapes never touch along an edge only).

    ``merge=True`` merges the coplanar quads of each face into one polygon, drops
    the grid vertices that are not corners of any face, and ear-clips each
    polygon -- the same solid with far fewer triangles (and tets).  Faces must be
    simple polygons (no holes), which holds for the shapes but not for the walls
    with windows.
    """
    b = np.asarray(boxes, dtype=np.float64)
    grids = [np.unique(np.concatenate([b[:, 2 * a], b[:, 2 * a + 1]])) for a in range(3)]
    centers = [0.5 * (g[:-1] + g[1:]) for g in grids]
    C = np.stack(np.meshgrid(*centers, indexing='ij'), axis=-1)            # (nx, ny, nz, 3)
    filled = np.zeros(C.shape[:3], dtype=bool)
    for x0, x1, y0, y1, z0, z1 in b:
        filled |= ((C[..., 0] > x0) & (C[..., 0] < x1) & (C[..., 1] > y0) & (C[..., 1] < y1)
                   & (C[..., 2] > z0) & (C[..., 2] < z1))
    pad = np.pad(filled, 1)

    vid = {}
    verts = []

    def v(i, j, k):
        key = (i, j, k)
        if key not in vid:
            vid[key] = len(verts)
            verts.append((grids[0][i], grids[1][j], grids[2][k]))
        return vid[key]

    quads = {}          # (axis, sign, plane index) -> quads, counter-clockwise seen from outside
    for a in range(3):
        b1, b2 = (a + 1) % 3, (a + 2) % 3
        e = np.zeros(3, dtype=int)
        e[a] = 1
        for idx in np.argwhere(filled):
            for sgn in (+1, -1):
                n = idx + sgn * e
                if pad[tuple(n + 1)]:
                    continue
                # Quad on the cell face at grid index idx[a] (+1 when sgn > 0).
                base = idx.copy()
                base[a] += 1 if sgn > 0 else 0
                q = []
                for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    p = base.copy()
                    p[b1] += d1
                    p[b2] += d2
                    q.append(v(*p))
                # (b1, b2) is right-handed about +a, so this winding faces +a.
                if sgn < 0:
                    q = q[::-1]
                quads.setdefault((a, sgn, int(base[a])), []).append(q)
    V = np.asarray(verts, dtype=np.float64)
    if not merge:
        faces = [t for qs in quads.values() for q in qs for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
        return V, np.asarray(faces, dtype=np.int64)

    # Split each plane's quads into faces (regions connected through edges, so two
    # faces that only touch at a corner stay apart), then trace each face's
    # boundary loop: the directed quad edges whose reverse is absent.
    loops = []
    for key, all_qs in quads.items():
        parent = list(range(len(all_qs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        by_edge = {}
        for qi, q in enumerate(all_qs):
            for i in range(4):
                by_edge.setdefault(frozenset((q[i], q[(i + 1) % 4])), []).append(qi)
        for qis in by_edge.values():
            for qi in qis[1:]:
                parent[find(qi)] = find(qis[0])
        regions = {}
        for qi, q in enumerate(all_qs):
            regions.setdefault(find(qi), []).append(q)
        for qs in regions.values():
            directed = {(q[i], q[(i + 1) % 4]) for q in qs for i in range(4)}
            nxt = {}
            for u, w in directed:
                if (w, u) not in directed:
                    assert u not in nxt, 'face pinched at a vertex'
                    nxt[u] = w
            start = next(iter(nxt))
            loop, u = [start], nxt.pop(start)
            while u != start:
                loop.append(u)
                u = nxt.pop(u)
            assert not nxt, 'face with a hole'
            loops.append((key, loop))
    # A vertex can go only if it is a straight-through point of every loop it is on.
    keep = np.zeros(len(V), dtype=bool)
    for _, loop in loops:
        m = len(loop)
        for k in range(m):
            d0 = V[loop[k]] - V[loop[k - 1]]
            d1 = V[loop[(k + 1) % m]] - V[loop[k]]
            if np.linalg.norm(np.cross(d0, d1)) > 1e-9:
                keep[loop[k]] = True
    faces = []
    for (a, sgn, _), loop in loops:
        loop = [i for i in loop if keep[i]]
        b1, b2 = (a + 1) % 3, (a + 2) % 3
        # Mirror the -a faces so every loop is counter-clockwise in 2-D.
        P = np.stack([V[loop, b1], sgn * V[loop, b2]], axis=1)
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        assert area > 0, 'face with a hole'
        faces += [(loop[i], loop[j], loop[k]) for i, j, k in ear_clip(P)]
    F = np.asarray(faces, dtype=np.int64)
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def ear_clip(P, eps=1e-9):
    """Triangulate a simple counter-clockwise polygon (n, 2) by ear clipping.

    An ear is accepted only if no other vertex lies inside or on it, so a
    vertex on a diagonal is never skipped over (no T-junctions)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    idx = list(range(len(P)))
    tris = []
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= eps:
                continue
            blocked = any(cross(a, b, P[j]) >= -eps and cross(b, c, P[j]) >= -eps
                          and cross(c, a, P[j]) >= -eps
                          for j in idx if j not in (i0, i1, i2))
            if not blocked:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            raise RuntimeError('ear clipping failed')
    assert cross(P[idx[0]], P[idx[1]], P[idx[2]]) > eps, 'degenerate last triangle'
    tris.append(tuple(idx))
    return tris


def read_obj_group(path, group):
    """Vertices and triangles of one ``g``/``o`` group of an OBJ (indices re-based)."""
    verts, faces, cur = [], [], ''
    for line in open(path):
        if line.startswith('v '):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('g ') or line.startswith('o '):
            cur = line[2:].strip()
        elif line.startswith('f ') and cur == group:
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    F = np.asarray(faces, dtype=np.int64)
    used = np.unique(F)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(verts, dtype=np.float64)[used], remap[F]


def write_obj(path, groups, header):
    """``groups``: list of (name, V, F); written as ``g <name>`` blocks."""
    with open(path, 'w') as fh:
        for line in header:
            fh.write('# {}\n'.format(line))
        off = 0
        for name, V, F in groups:
            fh.write('g {}\n'.format(name))
            for p in V:
                fh.write('v {:.6f} {:.6f} {:.6f}\n'.format(*p))
            for f in F + off + 1:
                fh.write('f {} {} {}\n'.format(*f))
            off += len(V)


def mesh_stats(V, F):
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    vol = float(np.einsum('ij,ij->i', T[:, 0], cr).sum() / 6.0)
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return vol, bool((counts == 2).all())


def render(path, groups, title, elev=22, azim=-58):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    allV = []
    for i, (name, V, F, alpha) in enumerate(groups):
        pc = Poly3DCollection(V[F], alpha=alpha, facecolor=colors[i % len(colors)],
                              edgecolor='k', linewidths=0.2)
        ax.add_collection3d(pc)
        allV.append(V)
    allV = np.concatenate(allV)
    lo, hi = allV.min(0), allV.max(0)
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--shell', default=os.path.join(OUT_DIR, 'env3.obj'),
                    help='OBJ whose "wall (1)" group is the room shell.')
    ap.add_argument('--preview', default=None, help='Directory for PNG previews.')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    for name, boxes in SHAPES.items():
        V, F = polycube_mesh(boxes, merge=True)
        vol, closed = mesh_stats(V, F)
        ext = V.max(0) - V.min(0)
        write_obj(os.path.join(args.out, name + '.obj'), [(name, V, F)],
                  ['{}: {}'.format(name, SHAPE_NOTES[name]),
                   'union of {} boxes, extent {:g} x {:g} x {:g}, outward-facing triangles'.format(
                       len(boxes), *ext),
                   'written by dataprocessing/make_shapes3d.py'])
        print('{:9s} {:3d} verts {:3d} tris  extent {}  volume {:.0f}  closed {}'.format(
            name, len(V), len(F), ext, vol, closed))
        if args.preview:
            render(os.path.join(args.preview, name + '.png'), [(name, V, F, 0.9)], name)

    shell_V, shell_F = read_obj_group(args.shell, 'wall (1)')
    for name, spec in ENVS.items():
        groups = [('wall (1)', shell_V, shell_F)]
        for body, boxes in spec['bodies'].items():
            V, F = polycube_mesh(boxes)
            vol, closed = mesh_stats(V, F)
            groups.append((body, V, F))
            print('{} {:12s} {:3d} verts {:4d} tris  volume {:.0f}  closed {}'.format(
                name, body, len(V), len(F), vol, closed))
        write_obj(os.path.join(args.out, name + '.obj'), groups,
                  ['{}: {}'.format(name, spec['note']),
                   'written by dataprocessing/make_shapes3d.py'])
        if args.preview:
            show = [(g, V, F, 0.05 if g == 'wall (1)' else 0.55) for g, V, F in groups]
            render(os.path.join(args.preview, name + '.png'), show, name, elev=30, azim=-60)
            render(os.path.join(args.preview, name + '_top.png'), show, name + ' (top, y up)',
                   elev=90, azim=-90)


if __name__ == '__main__':
    main()
