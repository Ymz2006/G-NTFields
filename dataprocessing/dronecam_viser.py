"""Interactive viser viewer for the drone + chase-camera test (dataprocessing/make_dronecam3d.py, EXPERIMENTS.md H18).

A small delta-wing drone is followed by a camera L = 30 / 55 / 80 units behind it with a tight or a wide field of
view (six camera set-ups); the body is the drone plus the camera's view pyramid.  Five scenes (meadow, suburb,
campus, downtown, overgrown), the test pairs of results/graph_homotopy/plan_dronecam (plan_eval_homotopy.py
--camera --save-paths), and for one scene, camera set-up, model, body and seed:

  - the drone's path and the camera's path; the camera path is red where the camera sees < 99 % of the drone
  - the body flying along the path (slider or Play): drone, camera point and view pyramid (red while the view breaks)
  - Camera POV: puts your browser view at the camera, looking down the pyramid; everything outside the pyramid
    is masked, so what you see is the camera's view and nothing else
  - Camera image: the same view rendered on the server (obstacles inside the pyramid, in front of the drone, are
    tinted red; everything beyond the drone is faded), shown in the panel next to the 3D scene

Without results for a choice the body is shown at a demo pose, so the scenes can be inspected before the queue
is through.  Coordinates are the planner's normalized frame (env units / 220.457 around the room centre, y up).

    python dataprocessing/dronecam_viser.py --port 8097
    python dataprocessing/dronecam_viser.py --snapshot DIR     # overview PNG of every scene (+ the bodies), exit
"""
import argparse
import csv
import json
import os
import threading
import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ = os.path.join(ROOT, 'datasets', '3dshape', 'obj')
RES = os.path.join(ROOT, 'results', 'graph_homotopy', 'plan_dronecam')
DASH_CSV = os.path.join(ROOT, 'results', 'graph_homotopy', 'dronecam', 'dash', 'dronecam_results.csv')
SCALE = 220.457
CENTER = np.array([100.180961, 86.09025399999999, -89.991084])
ENVS = ['meadow', 'suburb', 'campus', 'downtown', 'overgrown']
CAMS = ['30t', '30w', '55t', '55w', '80t', '80w']
MODELS = {'lifted encoder': ('hlift', 'dual'), 'full NTFields': ('mbtd', 'base')}
BODIES = {'camera-aware body': 'dronecam', 'rigid follower': 'drone'}
FILTERS = ['all', 'follow success', 'view blocked', 'drone collides', 'not converged',
           'camera-aware succeeds, follower fails']
VIEW_OK = 0.99

COL_DRONE = (235, 104, 52)
COL_CAM = (42, 120, 214)
COL_HIT = (227, 73, 72)
COL_GOAL = (27, 175, 122)
COL_WALL = (200, 200, 210)
COL_MASK = (24, 26, 30)
PART_COL = {'trunk': (118, 86, 58), 'branch': (118, 86, 58), 'crown': (96, 150, 88), 'leaves': (122, 172, 98),
            'wall': (178, 184, 192), 'tank': (122, 126, 132), 'mast': (92, 94, 98), 'portico': (198, 192, 180),
            'bridge': (150, 160, 176), 'cable': (72, 72, 78)}


def rot(p):
    return Rotation.from_rotvec(np.asarray(p[3:6], dtype=np.float64) * 2 * np.pi)


def wxyz(r):
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def load_obj_simple(path):
    V, F = [], []
    for line in open(path):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            F.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def box_mesh(b):
    x0, x1, y0, y1, z0, z1 = b
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]], dtype=np.int64)
    return V, F


class Camera:
    """One camera set-up (dronecam<cfg>_camera.json) in the bodies' normalized frame."""

    def __init__(self, cfg):
        self.cfg = cfg
        cj = json.load(open(os.path.join(OBJ, 'dronecam{}_camera.json'.format(cfg))))
        c = np.array(cj['frame_centre'], dtype=np.float64)
        self.centre = c
        self.L = int(cj['distance'])
        self.apex = (np.array(cj['apex']) - c) / SCALE
        self.base = (np.array(cj['base'], dtype=np.float64) - c) / SCALE
        self.dist = float(self.base[0, 0] - self.apex[0])                       # apex -> base plane, along +x
        self.tan_h = float(np.abs(self.base[:, 2]).max() / self.dist)           # half-width / distance (body z)
        self.tan_v = float(np.abs(self.base[:, 1]).max() / self.dist)           # half-height / distance (body y)
        self.fov = cj['fov_deg']
        self.label = '{} units behind, {} field of view ({:.0f} x {:.0f} deg)'.format(
            self.L, 'tight' if cfg.endswith('t') else 'wide', *self.fov)
        V, F = load_obj_simple(os.path.join(OBJ, cj['bird_obj']))
        self.bird_V = ((V - c) / SCALE).astype(np.float32)
        self.bird_F = F.astype(np.int32)
        V, F = load_obj_simple(os.path.join(OBJ, cj['body_obj']))
        self.body_V = ((V - c) / SCALE).astype(np.float32)
        self.body_F = F.astype(np.int32)
        self.pyr_V = np.concatenate([self.base, self.apex[None]]).astype(np.float32)
        self.pyr_F = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2], [0, 2, 3]], np.int32)
        # the POV mask: a frame just in front of the camera with a hole that is exactly the pyramid's cross-section
        d, big = 0.004, 40.0
        inner = [self.apex + d * np.array([1, sy * self.tan_v, sz * self.tan_h]) for sy, sz in
                 ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        outer = [self.apex + d * np.array([1, sy * big * self.tan_v, sz * big * self.tan_h]) for sy, sz in
                 ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        self.mask_V = np.array(inner + outer, np.float32)
        self.mask_F = np.array([[i, (i + 1) % 4, 4 + i] for i in range(4)] +
                               [[(i + 1) % 4, 4 + (i + 1) % 4, 4 + i] for i in range(4)], np.int32)

    def eye_axes(self, P):
        """World eye position and camera axes (forward, up, right) at chart pose P."""
        R = rot(P).as_matrix()
        return P[:3] + R @ self.apex, R[:, 0], R[:, 1], R[:, 2]


class Scene:
    """A scene's boxes (<env>_boxes.json) in the normalized frame: meshes per obstacle part for viser and
    triangles with colours / object ids for the renderer."""

    def __init__(self, env):
        J = json.load(open(os.path.join(OBJ, env + '_boxes.json')))
        self.note = J['note']
        r = np.asarray(J['room'], dtype=np.float64)
        self.lo = ((np.array([r[0], r[2], r[4]]) - CENTER) / SCALE).astype(np.float32)
        self.hi = ((np.array([r[1], r[3], r[5]]) - CENTER) / SCALE).astype(np.float32)
        self.groups = {}                                    # name -> (V, F, colour)
        tris, cols, objs = [], [], []
        for gi, g in enumerate(J['groups']):
            tint = 1.0 + 0.08 * (((gi * 7919) % 5) - 2) / 2.0   # a little variation between buildings
            by_part = {}
            for b, part in zip(g['boxes'], g['parts']):
                V, F = box_mesh(b)
                Vn = (V - CENTER) / SCALE
                col = np.array(PART_COL[part], dtype=np.float64)
                if part == 'wall':
                    col = np.clip(col * tint, 0, 255)
                by_part.setdefault(part, []).append((Vn, F, col))
                tris.append(Vn[F])
                cols.append(np.tile(col, (12, 1)))
                objs.append(np.full(12, 10 + len(objs)))
            for part, items in by_part.items():
                off, Vs, Fs = 0, [], []
                for Vn, F, col in items:
                    Vs.append(Vn)
                    Fs.append(F + off)
                    off += len(Vn)
                self.groups['{}/{}'.format(g['name'], part)] = (np.concatenate(Vs).astype(np.float32),
                                                              np.concatenate(Fs).astype(np.int32), tuple(int(v) for v in items[0][2]))
        # the room's inner faces for the renderer (floor, ceiling, walls)
        lo, hi = self.lo.astype(np.float64), self.hi.astype(np.float64)
        X, Y, Z = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
        quads = [([X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[0], Z[1]], [X[0], Y[0], Z[1]], (214, 210, 200)),
                 ([X[0], Y[1], Z[0]], [X[1], Y[1], Z[0]], [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]], (238, 238, 236)),
                 ([X[0], Y[0], Z[0]], [X[0], Y[1], Z[0]], [X[0], Y[1], Z[1]], [X[0], Y[0], Z[1]], (226, 226, 222)),
                 ([X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[1], Y[1], Z[1]], [X[1], Y[0], Z[1]], (226, 226, 222)),
                 ([X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[0], Y[1], Z[0]], (226, 226, 222)),
                 ([X[0], Y[0], Z[1]], [X[1], Y[0], Z[1]], [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]], (226, 226, 222))]
        n_obst = sum(len(t) for t in tris)
        for q, (a, b, c, d, col) in enumerate(quads):
            tris.append(np.array([[a, b, c], [a, c, d]]))
            cols.append(np.array([col, col]))
            objs.append(np.array([1 + q, 1 + q]))
        self.tris = np.concatenate(tris).astype(np.float64)
        self.cols = np.concatenate(cols).astype(np.float64)
        self.kinds = np.concatenate([np.full(n_obst, 2), np.full(12, 1)])
        self.objs = np.concatenate(objs)


def _raster(T, col, kind, obj, shade, eye, f, u, s, w, h, fl, near=0.002):
    """Z-buffer rasterizer: triangles T (N, 3, 3) with per-triangle colour / kind / object id / shade, seen from
    ``eye`` along ``f`` (up ``u``, right ``s``) with focal length ``fl`` pixels; returns 1/z, object id, kind, rgb."""
    C = (T - eye) @ np.stack([s, u, f], 1)                 # camera coordinates: x right, y up, z forward
    iz = np.zeros((h, w))
    oid = np.full((h, w), -1)
    okind = np.zeros((h, w), np.int8)
    rgb = np.zeros((h, w, 3))
    for i in range(len(C)):
        poly = C[i]
        if poly[:, 2].max() < near:
            continue
        if poly[:, 2].min() < near:                        # Sutherland-Hodgman against z >= near
            out = []
            for j in range(3):
                a, b = poly[j], poly[(j + 1) % 3]
                if a[2] >= near:
                    out.append(a)
                if (a[2] >= near) != (b[2] >= near):
                    t = (near - a[2]) / (b[2] - a[2])
                    out.append(a + t * (b - a))
            poly = np.array(out)
        pr = np.stack([0.5 * w + fl * poly[:, 0] / poly[:, 2], 0.5 * h - fl * poly[:, 1] / poly[:, 2],
                       1.0 / poly[:, 2]], 1)
        for j in range(1, len(pr) - 1):
            A, B, Cq = pr[0], pr[j], pr[j + 1]
            area = (B[0] - A[0]) * (Cq[1] - A[1]) - (B[1] - A[1]) * (Cq[0] - A[0])
            if abs(area) < 1e-12:
                continue
            x0, x1 = max(0, int(np.floor(min(A[0], B[0], Cq[0])))), min(w - 1, int(np.ceil(max(A[0], B[0], Cq[0]))))
            y0, y1 = max(0, int(np.floor(min(A[1], B[1], Cq[1])))), min(h - 1, int(np.ceil(max(A[1], B[1], Cq[1]))))
            if x0 > x1 or y0 > y1:
                continue
            X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((B[0] - X) * (Cq[1] - Y) - (B[1] - Y) * (Cq[0] - X)) / area
            w1 = ((Cq[0] - X) * (A[1] - Y) - (Cq[1] - Y) * (A[0] - X)) / area
            w2 = 1 - w0 - w1
            z = w0 * A[2] + w1 * B[2] + w2 * Cq[2]
            sub = iz[y0:y1 + 1, x0:x1 + 1]
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (z > sub)
            if not m.any():
                continue
            sub[m] = z[m]
            oid[y0:y1 + 1, x0:x1 + 1][m] = obj[i]
            okind[y0:y1 + 1, x0:x1 + 1][m] = kind[i]
            rgb[y0:y1 + 1, x0:x1 + 1][m] = col[i] * shade[i]
    return iz, oid, okind, rgb


def _shade(T, eye, f, kind):
    """Per-triangle lighting (and the outward normal given the eye side) for the two renderers."""
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    facing = np.einsum('ij,ij->i', n, eye - T[:, 0]) > 0
    n = np.where(facing[:, None], n, -n)
    light = np.array([0.35, 1.0, 0.55]) / np.linalg.norm([0.35, 1.0, 0.55])
    nl = np.clip(n @ light, 0, None)
    k = np.where(kind == 1, 0.9 + 0.1 * nl,
                 np.where(kind == 3, 0.78 + 0.22 * nl + 0.12 * np.clip(-(n @ f), 0, None),
                          0.55 + 0.45 * nl + 0.1 * np.clip(n[:, 1], 0, None)))
    return k, facing


def _outline(rgb, oid, iz, hit, ss):
    h, w = oid.shape
    edge = np.zeros((h, w), bool)
    for dy, dx in ((0, ss), (ss, 0)):
        a, b = oid[:h - dy, :w - dx], oid[dy:, dx:]
        za, zb = iz[:h - dy, :w - dx], iz[dy:, dx:]
        diff = a != b
        edge[:h - dy, :w - dx] |= diff & (zb < za)
        edge[dy:, dx:] |= diff & (za < zb)
    rgb[edge & hit] *= 0.6


def render_pov(scene, cam, P, W=420, ss=2):
    """The camera's view at pose P as an (H, W, 3) uint8 image: obstacles inside the pyramid in front of the
    drone tinted red, everything beyond the drone faded."""
    eye, f, u, s = cam.eye_axes(P)
    R = rot(P).as_matrix()
    bird = (P[:3] + cam.bird_V[cam.bird_F].astype(np.float64) @ R.T)
    T = np.concatenate([scene.tris, bird])
    col = np.concatenate([scene.cols, np.tile(COL_DRONE, (len(bird), 1))])
    kind = np.concatenate([scene.kinds, np.full(len(bird), 3)])
    obj = np.concatenate([scene.objs, np.full(len(bird), 9999)])
    k, facing = _shade(T, eye, f, kind)
    keep = facing | (kind != 3)                            # the drone is closed: cull its back faces
    w, h = W * ss, int(round(W * cam.tan_v / cam.tan_h)) * ss
    fl = 0.5 * w / cam.tan_h
    iz, oid, okind, rgb = _raster(T[keep], col[keep], kind[keep], obj[keep], k[keep], eye, f, u, s, w, h, fl)
    sky = np.array([235.0, 238.0, 241.0])
    depth = np.where(iz > 0, 1.0 / np.maximum(iz, 1e-12), np.inf)
    hit = oid >= 0
    other = hit & (okind != 3)
    block = other & (depth < cam.dist)                     # inside the pyramid, in front of the drone
    beyond = other & ~block
    rgb[block] = 0.45 * rgb[block] + 0.55 * np.array([227.0, 73.0, 72.0])
    rgb[beyond] = 0.55 * rgb[beyond] + 0.45 * sky
    rgb[~hit] = sky
    _outline(rgb, oid, iz, hit, ss)
    img = rgb.reshape(h // ss, ss, w // ss, ss, 3).mean((1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def render_view(scene, eye, look_at, vfov_deg=40.0, W=720, H=480, ss=2, extra=(), room=True):
    """An overview of the scene from ``eye`` towards ``look_at`` (y up); ``extra`` = [(V, F, colour, kind)] world
    meshes drawn with it (kind 3 = a body: back faces culled)."""
    eye, look_at = np.asarray(eye, np.float64), np.asarray(look_at, np.float64)
    f = look_at - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, [0, 1, 0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = (scene.kinds != 1) if not room else (scene.objs != 2)      # room object 2 is the ceiling
    T, col, kind, obj = [scene.tris[m]], [scene.cols[m]], [scene.kinds[m]], [scene.objs[m]]
    for i, (V, F, c, kd) in enumerate(extra):
        T.append(np.asarray(V, np.float64)[F])
        col.append(np.tile(np.asarray(c, np.float64), (len(F), 1)))
        kind.append(np.full(len(F), kd))
        obj.append(np.full(len(F), 5000 + i))
    T, col, kind, obj = (np.concatenate(x) for x in (T, col, kind, obj))
    k, facing = _shade(T, eye, f, kind)
    # the room's walls facing the eye would hide the scene: keep only the far faces
    keep = (kind == 2) | ((kind == 3) & facing) | ((kind == 1) & ~facing)
    w, h = W * ss, H * ss
    fl = 0.5 * h / np.tan(np.radians(vfov_deg) / 2)
    iz, oid, okind, rgb = _raster(T[keep], col[keep], kind[keep], obj[keep], k[keep], eye, f, u, s, w, h, fl, near=0.01)
    sky = np.array([240.0, 242.0, 245.0])
    hit = oid >= 0
    rgb[~hit] = sky
    _outline(rgb, oid, iz, hit, ss)
    img = rgb.reshape(h // ss, ss, w // ss, ss, 3).mean((1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def body_name(kind, cfg):
    return 'dronecam' + cfg if kind == 'dronecam' else 'drone' + cfg[:-1]


def load_run(kind, cfg, env, model, seed):
    """{case: dict(P, view, bird_clear, clear, rec)} of the model trained on <body>_<env>, scored on dronecam<cfg>_<env>."""
    code, planner = MODELS[model]
    base = os.path.join(RES, '{}_{}_{}_s{}'.format(body_name(kind, cfg), env, code, seed),
                        'dronecam{}_{}__{}'.format(cfg, env, planner))
    if not os.path.exists(base + '_chosen.npz') or not os.path.exists(base + '.json'):
        return None
    try:
        z = np.load(base + '_chosen.npz')
        recs = {r['case']: r for r in json.load(open(base + '.json'))['records']}
    except (ValueError, KeyError):
        return None
    runs = {}
    for i, case in enumerate(z['case']):
        P, V, CB, CL, k3 = z['path'][i], z['view'][i], z['bird_clear'][i], z['clear'][i], int(z['iters3'][i])
        ok = ~np.isnan(P[:, 0])
        P, V, CB, CL = P[ok], V[ok], CB[ok], CL[ok]
        if k3 >= 0:                                        # cut at SE(3) convergence, goal appended (as scored)
            P = np.concatenate([P[:k3 + 2], z['goal'][i][None]])
            V = np.concatenate([V[:k3 + 2], V[k3 + 1:k3 + 2]])
            CB = np.concatenate([CB[:k3 + 2], CB[k3 + 1:k3 + 2]])
            CL = np.concatenate([CL[:k3 + 2], CL[k3 + 1:k3 + 2]])
        runs[int(case)] = dict(P=P.astype(np.float64), view=V, bird_clear=CB, clear=CL, rec=recs[int(case)])
    return runs


DEMO_POSE = np.concatenate([(np.array([61.0, 62.0, -132.0]) - CENTER) / SCALE, [0.0, 0.0, 0.0]])


def results_markdown():
    if not os.path.exists(DASH_CSV):
        return 'results not exported yet (aggregate_dronecam.py)'
    rows = list(csv.DictReader(open(DASH_CSV)))
    t = '| scene | camera | model | planned as | follow % | clean frame % |\n|---|---|---|---|---|---|\n'
    for r in rows:
        t += '| {} | {} | {} | {} | {} | {} |\n'.format(r['scene'], r['camera'], r['model'], r['trained_on'],
                                                       r['follow_success'], r['success_se3'])
    return ('**Follow success** (converged, drone collision-free, camera sees >= 99 % of the drone at every '
            'waypoint) and **clean frame** (nothing of the scene inside the field of view), 1000 pairs, '
            'mean over seeds\n\n' + t)


class Viewer:
    def __init__(self, server):
        self.server = server
        self.sc = server.scene
        self.sc.set_up_direction('+y')
        self.cams = {c: Camera(c) for c in CAMS}
        self.scenes = {e: Scene(e) for e in ENVS}
        self.cache = {}
        self.handles = {}
        self.case_ids = []
        self.playing = False
        self.lock = threading.RLock()
        self.saved_cams = {}
        self.pov_req = None
        self.pov_evt = threading.Event()
        self.demo = False

        gui = server.gui
        cam_labels = {'{}: {}'.format(c, self.cams[c].label): c for c in CAMS}
        self.cam_of_label = cam_labels
        with gui.add_folder('Scene'):
            self.env_dd = gui.add_dropdown('Scene', ENVS, initial_value='meadow')
            self.cam_dd = gui.add_dropdown('Camera set-up', list(cam_labels), initial_value=list(cam_labels)[3])
            self.model_dd = gui.add_dropdown('Planner', list(MODELS), initial_value='lifted encoder')
            self.body_dd = gui.add_dropdown('Planned as', list(BODIES), initial_value='camera-aware body')
            self.seed_dd = gui.add_dropdown('Seed', ['0', '1', '2'], initial_value='0')
            self.wall_cb = gui.add_checkbox('Show room shell', True)
            self.scene_md = gui.add_markdown('')
        with gui.add_folder('Test pairs'):
            self.filter_dd = gui.add_dropdown('Filter', FILTERS, initial_value='all')
            self.case_sl = gui.add_slider('Case #', 0, 1, 1, 0)
            self.prev_bt = gui.add_button('Previous')
            self.next_bt = gui.add_button('Next')
            self.case_md = gui.add_markdown('')
        with gui.add_folder('Motion'):
            self.t_sl = gui.add_slider('Along path', 0.0, 1.0, 0.002, 0.0)
            self.play_bt = gui.add_button('Play / pause')
            self.speed_sl = gui.add_slider('Play speed', 0.1, 3.0, 0.1, 1.0)
            self.trail_cb = gui.add_checkbox('Ghost trail', True)
            self.trail_n = gui.add_slider('Ghosts', 3, 20, 1, 6)
        with gui.add_folder('Camera'):
            self.pov_cb = gui.add_checkbox('Camera POV', False, hint='Look through the camera: the view is the pyramid')
            self.pyr_cb = gui.add_checkbox('Show view pyramid', True)
            self.img_cb = gui.add_checkbox('Camera image', True)
            self.img = gui.add_image(np.full((96, 210, 3), 235, np.uint8), label='camera view', format='jpeg')
            self.view_md = gui.add_markdown('')
        with gui.add_folder('Results', expand_by_default=False):
            gui.add_markdown(results_markdown())

        for w in (self.env_dd, self.cam_dd, self.model_dd, self.body_dd, self.seed_dd):
            w.on_update(lambda _: self.redraw_all())
        self.wall_cb.on_update(lambda _: self.draw_env())
        self.filter_dd.on_update(lambda _: self.refilter())
        self.case_sl.on_update(lambda _: self.draw_case())
        self.prev_bt.on_click(lambda _: self.step_case(-1))
        self.next_bt.on_click(lambda _: self.step_case(+1))
        self.t_sl.on_update(lambda _: self.draw_mover())
        self.play_bt.on_click(lambda _: self.toggle_play())
        self.trail_cb.on_update(lambda _: self.draw_case())
        self.trail_n.on_update(lambda _: self.draw_case())
        self.pov_cb.on_update(lambda _: self.pov_changed())
        self.pyr_cb.on_update(lambda _: self.draw_case())
        self.img_cb.on_update(lambda _: self.draw_mover())
        server.on_client_connect(self.client_connected)
        threading.Thread(target=self.player, daemon=True).start()
        threading.Thread(target=self.pov_worker, daemon=True).start()
        self.redraw_all()

    # ---------- data
    @property
    def cam(self):
        return self.cams[self.cam_of_label[self.cam_dd.value]]

    def run(self, kind=None):
        key = (kind or BODIES[self.body_dd.value], self.cam.cfg, self.env_dd.value, self.model_dd.value, int(self.seed_dd.value))
        if self.cache.get(key) is None:                  # missing runs are re-checked: the queue adds them over time
            self.cache[key] = load_run(*key)
        return self.cache[key]

    # ---------- scene graph helpers
    @staticmethod
    def _remove(h):
        try:
            h.remove()
        except KeyError:
            pass

    def add(self, name, handle):
        old = self.handles.get(name)
        if old is not None and old is not handle:
            self._remove(old)
        self.handles[name] = handle
        return handle

    def clear(self, prefix):
        for name in [n for n in self.handles if n.startswith(prefix)]:
            self._remove(self.handles.pop(name))

    # ---------- drawing
    def redraw_all(self):
        with self.lock:
            self.playing = False
            self.clear('/')
            self.draw_env()
        self.refilter()

    def draw_env(self):
        self.clear('/env')
        S = self.scenes[self.env_dd.value]
        self.scene_md.content = '*{}*'.format(S.note)
        if self.wall_cb.value:
            lo, hi = S.lo, S.hi
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)
            e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
            seg = np.stack([c[[a for a, _ in e]], c[[b for _, b in e]]], 1)
            self.add('/env/room_edges', self.sc.add_line_segments('/env/room_edges', seg, (120, 120, 130), line_width=1.5))
            floor = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]], np.float32)
            self.add('/env/floor', self.sc.add_mesh_simple('/env/floor', floor, np.array([[0, 2, 1], [0, 3, 2]], np.int32),
                                                           color=(222, 218, 208), side='double', cast_shadow=False))
        for g, (V, F, col) in S.groups.items():
            name = '/env/' + g.replace(' ', '_')
            self.add(name, self.sc.add_mesh_simple(name, V, F, color=col, flat_shading=True))

    def refilter(self):
        runs = self.run()
        self.demo = runs is None
        if runs is None:
            self.case_ids = [0]
            self.case_sl.max = 1
            self.case_sl.value = 0
            self.draw_case()
            return
        cases = sorted(runs)
        f = self.filter_dd.value
        r = lambda c: runs[c]['rec']  # noqa: E731
        if f == 'all':
            ids = cases
        elif f == 'follow success':
            ids = [c for c in cases if r(c)['follow_success_se3']]
        elif f == 'view blocked':
            ids = [c for c in cases if not r(c)['cam_view_ok']]
        elif f == 'drone collides':
            ids = [c for c in cases if r(c)['bird_collision_se3']]
        elif f == 'not converged':
            ids = [c for c in cases if not r(c)['converged_se3']]
        else:
            body, fol = self.run('dronecam'), self.run('drone')
            ids = [c for c in cases if body and fol and c in body and c in fol and
                   body[c]['rec']['follow_success_se3'] and not fol[c]['rec']['follow_success_se3']]
        self.case_ids = ids
        self.case_sl.max = max(len(ids) - 1, 1)
        self.case_sl.value = 0
        self.draw_case()

    def step_case(self, k):
        if self.case_ids:
            self.case_sl.value = int(np.clip(self.case_sl.value + k, 0, len(self.case_ids) - 1))

    def draw_case(self):
        with self.lock:
            self.clear('/case')
            cam = self.cam
            if self.demo:
                self.case_d = dict(P=np.array([DEMO_POSE, DEMO_POSE + [0.25, 0, 0, 0, 0, 0]]), view=np.array([1.0, 1.0]),
                                   bird_clear=np.array([1.0, 1.0]), clear=np.array([1.0, 1.0]), rec=None)
                self.case_md.content = ('No results yet for **{} / {} / {} / {} / seed {}** -- showing the body at a demo '
                                        'pose (the queue fills results/graph_homotopy/plan_dronecam as it runs).'.format(
                                            self.env_dd.value, cam.cfg, self.model_dd.value, self.body_dd.value, self.seed_dd.value))
                self.clear('/mover')
                self._mover_bad = None
                pov_only = True
            else:
                if not self.case_ids:
                    self.case_md.content = 'No test pair matches this filter.'
                    return
                runs = self.run()
                case = self.case_ids[int(min(self.case_sl.value, len(self.case_ids) - 1))]
                d = runs[case]
                self.case_d = d
                if self.pov_cb.value:                      # through the camera: only the scene and the drone
                    self.describe(case, d)
                    self.clear('/mover')
                    self._mover_bad = None
                    pov_only = True
                else:
                    pov_only = False
        if pov_only:
            self.draw_mover()
            return
        with self.lock:
            P, view = d['P'], d['view']
            # paths: drone (orange), camera (blue, red where the view breaks)
            seg = np.stack([P[:-1, :3], P[1:, :3]], 1).astype(np.float32)
            bcol = np.where((np.minimum(d['bird_clear'][:-1], d['bird_clear'][1:]) < 0)[:, None], COL_HIT, COL_DRONE)
            self.add('/case/drone_path', self.sc.add_line_segments(
                '/case/drone_path', seg, np.repeat(bcol[:, None], 2, 1).astype(np.uint8), line_width=3))
            E = np.array([cam.eye_axes(p)[0] for p in P])
            cseg = np.stack([E[:-1], E[1:]], 1).astype(np.float32)
            ccol = np.where((np.minimum(view[:-1], view[1:]) < VIEW_OK)[:, None], COL_HIT, COL_CAM)
            self.add('/case/cam_path', self.sc.add_line_segments(
                '/case/cam_path', cseg, np.repeat(ccol[:, None], 2, 1).astype(np.uint8), line_width=2.5))
            for nm, p, c, op in (('start', P[0], COL_DRONE, 0.35), ('goal', P[-1], COL_GOAL, 0.45)):
                self.add('/case/' + nm, self.sc.add_mesh_simple('/case/' + nm, cam.bird_V, cam.bird_F, color=c,
                                                                opacity=op, position=p[:3], wxyz=wxyz(rot(p))))
            if self.trail_cb.value:
                for k in np.linspace(0, len(P) - 1, int(self.trail_n.value) + 2)[1:-1].astype(int):
                    bad = view[k] < VIEW_OK
                    self.add('/case/ghost%d' % k, self.sc.add_mesh_simple(
                        '/case/ghost%d' % k, cam.bird_V, cam.bird_F, color=COL_HIT if bad else (170, 170, 170),
                        opacity=0.2, position=P[k, :3], wxyz=wxyz(rot(P[k]))))
            self.describe(case, d)
            self.clear('/mover')
            self._mover_bad = None
        self.draw_mover()

    def describe(self, case, d):
        view, rec, P = d['view'], d['rec'], d['P']
        lost = np.nonzero(view < VIEW_OK)[0]
        self.case_md.content = (
                '**{} / camera {} / {}**, {} seed {}: test pair {} ({} of {} shown)\n\n'
                '- follow success: **{}** (converged {}, drone collides {})\n'
                '- camera sees the drone: lowest {:.0f} %, mean {:.1f} %; first view break at step {}\n'
                '- clean frame (nothing inside the field of view all along): {}\n- waypoints: {}'.format(
                    self.env_dd.value, self.cam.cfg, self.body_dd.value, self.model_dd.value, self.seed_dd.value, case,
                    int(self.case_sl.value) + 1, len(self.case_ids),
                    'yes' if rec['follow_success_se3'] else 'no', rec['converged_se3'], rec['bird_collision_se3'],
                    100 * rec['view_min'], 100 * rec['view_mean'], int(lost[0]) if len(lost) else '-',
                    'yes' if rec['success_se3'] else 'no', len(P)))

    def pose_at(self, t):
        P = self.case_d['P']
        x = t * (len(P) - 1)
        k = int(np.floor(x))
        k1 = min(k + 1, len(P) - 1)
        a = x - k
        r0, r1 = rot(P[k]), rot(P[k1])
        r = r0 * Rotation.from_rotvec(a * (r0.inv() * r1).as_rotvec())
        pos = (1 - a) * P[k, :3] + a * P[k1, :3]
        return np.concatenate([pos, r.as_rotvec() / (2 * np.pi)]), (k if a < 0.5 else k1)

    def draw_mover(self):
        if not self.case_ids or not hasattr(self, 'case_d'):
            return
        with self.lock:
            p, k = self.pose_at(self.t_sl.value)
            d, cam = self.case_d, self.cam
            bad = bool(d['view'][k] < VIEW_OK)
            pov = self.pov_cb.value
            state = (bad, pov, self.pyr_cb.value, cam.cfg)
            frame = self.handles.get('/mover')
            if frame is None or self._mover_bad != state:
                self.clear('/mover')
                frame = self.add('/mover', self.sc.add_frame('/mover', show_axes=False))
                self.add('/mover/drone', self.sc.add_mesh_simple('/mover/drone', cam.bird_V, cam.bird_F,
                                                                 color=COL_DRONE, flat_shading=True))
                self.add('/mover/cam', self.sc.add_icosphere('/mover/cam', radius=0.006,
                                                             color=COL_HIT if bad else COL_CAM,
                                                             position=cam.apex.astype(np.float32)))
                if self.pyr_cb.value and not pov:
                    self.add('/mover/pyr', self.sc.add_mesh_simple(
                        '/mover/pyr', cam.pyr_V, cam.pyr_F, color=COL_HIT if bad else COL_CAM,
                        opacity=0.35 if bad else 0.18, side='double', cast_shadow=False))
                    self.add('/mover/pyr_edges', self.sc.add_mesh_simple(
                        '/mover/pyr_edges', cam.pyr_V, cam.pyr_F, color=COL_HIT if bad else COL_CAM, wireframe=True))
                if pov:
                    self.add('/mover/mask', self.sc.add_mesh_simple('/mover/mask', cam.mask_V, cam.mask_F,
                                                                    color=COL_MASK, side='double', cast_shadow=False))
                self._mover_bad = state
            frame.position = p[:3]
            frame.wxyz = wxyz(rot(p))
            self.view_md.content = 'step {} of {}: the camera sees **{:.0f} %** of the drone{}'.format(
                k, len(d['P']) - 1, 100 * d['view'][k], ' (view broken)' if bad else '')
        if pov:
            self.set_pov(p)
        if self.img_cb.value:
            self.pov_req = p
            self.pov_evt.set()

    # ---------- camera POV
    def set_pov(self, p):
        eye, f, u, _ = self.cam.eye_axes(p)
        for client in self.server.get_clients().values():
            aspect = getattr(client.camera, 'aspect', 16 / 9) or 16 / 9
            vfov = 2 * np.arctan(max(self.cam.tan_v, self.cam.tan_h / aspect) * 1.02)
            with client.atomic():
                client.camera.up_direction = tuple(u)
                client.camera.position = tuple(eye)
                client.camera.look_at = tuple(eye + 0.3 * f)
                client.camera.fov = float(vfov)
                try:
                    client.camera.near = 0.001
                except AttributeError:
                    pass

    def client_connected(self, client):
        with client.atomic():                              # a three-quarter view of the whole room
            client.camera.up_direction = (0.0, 1.0, 0.0)
            client.camera.position = (0.62, 0.42, 0.78)
            client.camera.look_at = (0.0, -0.08, 0.0)
        self.pov_changed()

    def pov_changed(self):
        clients = self.server.get_clients()
        if self.pov_cb.value:
            for cid, c in clients.items():
                if cid not in self.saved_cams:
                    self.saved_cams[cid] = (tuple(c.camera.position), tuple(c.camera.look_at), float(c.camera.fov))
        else:
            for cid, (pos, look, fov) in list(self.saved_cams.items()):
                c = clients.get(cid)
                if c is not None:
                    with c.atomic():
                        c.camera.up_direction = (0.0, 1.0, 0.0)
                        c.camera.position = pos
                        c.camera.look_at = look
                        c.camera.fov = fov
            self.saved_cams.clear()
        self._mover_bad = None
        self.draw_case()

    def pov_worker(self):        # renders the panel image off the GUI thread, newest request wins
        while True:
            self.pov_evt.wait()
            self.pov_evt.clear()
            p = self.pov_req
            if p is None:
                continue
            try:
                self.img.image = render_pov(self.scenes[self.env_dd.value], self.cam, p)
            except Exception as e:                         # keep the viewer alive
                print('render_pov failed:', e, flush=True)

    # ---------- play
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing and self.t_sl.value >= 1.0:
            self.t_sl.value = 0.0

    def player(self):
        while True:
            time.sleep(0.05)
            if self.playing and self.case_ids and hasattr(self, 'case_d'):
                n = max(len(self.case_d['P']) - 1, 1)
                t = self.t_sl.value + self.speed_sl.value * 0.4 / n
                if t >= 1.0:
                    t, self.playing = 1.0, False
                self.t_sl.value = t


def snapshot(out_dir):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    cam = Camera('55w')
    R = rot(DEMO_POSE).as_matrix()
    body = (DEMO_POSE[:3] + cam.body_V.astype(np.float64) @ R.T, cam.body_F, COL_CAM, 3)
    drone = (DEMO_POSE[:3] + cam.bird_V.astype(np.float64) @ R.T, cam.bird_F, COL_DRONE, 3)
    for env in ENVS:
        S = Scene(env)
        t0 = time.time()
        im = render_view(S, eye=(0.95, 0.75, 1.15), look_at=(0.0, -0.12, 0.0), vfov_deg=46, extra=[body, drone])
        Image.fromarray(im).save(os.path.join(out_dir, env + '.png'))
        print(env, '{:.1f}s'.format(time.time() - t0), flush=True)
    # the bodies: the six camera set-ups next to each other, drone in orange, view pyramid in blue
    S = Scene('meadow')
    extra = []
    for i, cfg in enumerate(CAMS):
        c = Camera(cfg)
        p = np.array([-0.55, 0.05 - 0.085 * (i // 2) * 2, -0.3 + 0.3 * (i % 2)])  # rows: distance, columns: fov
        p = np.array([-0.45 + 0.0, 0.12 - 0.11 * (i // 2), -0.16 + 0.32 * (i % 2)])
        extra.append((p + c.body_V.astype(np.float64), c.body_F, COL_CAM, 3))
        extra.append((p + c.bird_V.astype(np.float64), c.bird_F, COL_DRONE, 3))
    im = render_view(S, eye=(0.55, 0.42, 0.75), look_at=(-0.2, -0.0, 0.0), vfov_deg=38, extra=extra, room=False)
    Image.fromarray(im).save(os.path.join(out_dir, 'bodies.png'))
    print('bodies', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8097)
    ap.add_argument('--snapshot', default=None, help='write overview PNGs of the scenes and bodies to this dir, exit')
    args = ap.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
        return
    server = viser.ViserServer(host=args.host, port=args.port)
    Viewer(server)
    print('viser at http://{}:{}'.format(args.host, args.port), flush=True)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
