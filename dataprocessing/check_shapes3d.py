"""Check the meshes written by make_shapes3d.py.

1. Tetrahedralization: each shape goes through the same tetgen call as
   preprocess_obj.py (env-normalized units) and the tet volumes must add up to
   the mesh volume.
2. Windows: for every window of env5 / env6 / env7 / env8 and every shape, the best of the 24
   axis-aligned orientations is translated straight through the window, from one
   room to the next; the minimum clearance along that path must be positive.
3. Rooms: at the centre of every room, random orientations must all be
   collision-free (the shape can turn around in each room).

Collision and clearance use the evaluator's test (analysis/plan_eval.py): 50 000
env surface points, signed distance to the shape mesh by fast winding number.
All distances are printed in env units (the room is about 198 x 138 x 218).

    python dataprocessing/check_shapes3d.py
"""
import importlib.util
import itertools
import os
import sys

import igl
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'dataprocessing'))
import make_shapes3d as ms  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'pre', os.path.join(ROOT, 'dataprocessing', 'preprocess_obj_gibson.py'))
pre = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pre)

D = os.path.join(ROOT, 'datasets', '3dshape')
SIGN = igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER

# Windows as (travel axis, wall centre along it, window centre (u, y), size) with
# u the other horizontal axis; rooms as centres (x, y, z).
YC = 0.5 * (ms.Y0 + ms.Y1)
WINDOWS = {
    'env5': [('z', -90.0, (100.0, 80.0))],
    'env7': [('z', -90.0, (100.0, 80.0))],
    'env8': [('z', -90.0, (100.0, 80.0))],
    'env6': [('x', 100.0, (-26.0, 60.0)), ('z', -90.0, (152.0, 115.0)),
             ('x', 100.0, (-142.0, 60.0)), ('z', -90.0, (48.0, 115.0))],   # window centres
}
ROOMS = {
    'env5': [(100.0, YC, 0.5 * (-85 + ms.Z1)), (100.0, YC, 0.5 * (ms.Z0 - 95))],
    'env7': [(100.0, YC, 0.5 * (-85 + ms.Z1)), (100.0, YC, 0.5 * (ms.Z0 - 95))],
    'env8': [(100.0, YC, 0.5 * (-85 + ms.Z1)), (100.0, YC, 0.5 * (ms.Z0 - 95))],
    'env6': [(0.5 * (ms.X0 + 95), YC, 0.5 * (-85 + ms.Z1)), (0.5 * (105 + ms.X1), YC, 0.5 * (-85 + ms.Z1)),
             (0.5 * (105 + ms.X1), YC, 0.5 * (ms.Z0 - 95)), (0.5 * (ms.X0 + 95), YC, 0.5 * (ms.Z0 - 95))],
}


def axis_rotations():
    """The 24 proper rotations that map coordinate axes onto coordinate axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


class Scene:
    def __init__(self, env_name, n_pts=50000, seed=0):
        V, F, _ = pre.load_obj(os.path.join(D, env_name + '.obj'))
        lo, hi = V.min(0), V.max(0)
        self.c, self.s = 0.5 * (lo + hi), float((hi - lo).max())
        np.random.seed(seed)
        self.pts = pre.sample_surface_points((V - self.c) / self.s, F, n_pts)   # normalized
        self.tree = cKDTree(self.pts)

    def clearance(self, SV, SF, R, t, cap=0.05):
        """Min signed distance (env units) of the env points to the shape placed
        at rotation R, world position t (env units).  Negative = collision."""
        tn = (np.asarray(t) - self.c) / self.s
        r = float(np.linalg.norm(SV, axis=1).max())
        sel = self.tree.query_ball_point(tn, r + cap)
        if not sel:
            return cap * self.s
        P = (self.pts[sel] - tn) @ R          # world -> shape frame (rows: p^T R = (R^T p)^T)
        S = igl.signed_distance(np.ascontiguousarray(P), SV, SF, SIGN)[0]
        return min(float(S.min()), cap) * self.s


def shape_local(name, scale):
    V, F, _ = pre.load_obj(os.path.join(D, name + '.obj'))
    return np.ascontiguousarray((V - 0.5 * (V.min(0) + V.max(0))) / scale), np.ascontiguousarray(F)


def main():
    np.set_printoptions(precision=3, suppress=True)
    scale = Scene('env5', n_pts=10).s
    print('env scale {:.3f} (env3: 220.457)'.format(scale))
    print('\n== tetrahedralization (preprocess_obj.py settings)')
    for name in ms.SHAPES:
        SV, SF = shape_local(name, scale)
        TV, TT, TF = pre.tetrahedralize_shape(SV, SF)
        T = TV[TT]
        tet_vol = np.abs(np.einsum('ij,ij->i', T[:, 1] - T[:, 0],
                                   np.cross(T[:, 2] - T[:, 0], T[:, 3] - T[:, 0]))).sum() / 6
        vol, closed = ms.mesh_stats(SV, SF)
        print('{:9s} {:3d} tets  {:3d} boundary tris  volume tets/mesh {:.6f}  radius {:.1f} units'.format(
            name, len(TT), len(TF), tet_vol / vol, np.linalg.norm(SV, axis=1).max() * scale))

    rots = axis_rotations()
    rand_R = Rotation.random(300, random_state=0).as_matrix()
    ok = True
    for env in WINDOWS:
        sc = Scene(env)
        print('\n== {}: min clearance through each window (best axis-aligned pose) | in each room (300 random rotations)'.format(env))
        for name in ms.SHAPES:
            SV, SF = shape_local(name, scale)
            win = []
            for axis, wc, (u, y) in WINDOWS[env]:
                best = -np.inf
                for R in rots:
                    # The shape is fully on either side at +-60 from the wall centre.
                    path = np.linspace(wc - 60, wc + 60, 49)
                    worst = np.inf
                    for a in path:
                        t = (u, y, a) if axis == 'z' else (a, y, u)
                        worst = min(worst, sc.clearance(SV, SF, R, t))
                        if worst <= best:
                            break
                    best = max(best, worst)
                win.append(best)
            room = []
            for c in ROOMS[env]:
                room.append(min(sc.clearance(SV, SF, R, c) for R in rand_R))
            ok &= min(win) > 0 and min(room) > 0
            print('  {:9s} windows {}  rooms {}'.format(name, np.round(win, 2), np.round(room, 2)))
    print('\nALL PASSAGES OPEN' if ok else '\nSOME PASSAGE IS BLOCKED')


if __name__ == '__main__':
    main()
