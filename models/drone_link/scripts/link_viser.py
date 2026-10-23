"""Interactive viser viewer for the stretchable-link study (models/drone_link/README.md).

Two drones keep a radio link: the box from drone a to drone b must stay clear of the scene (line of sight) and
between L_min and L_max long (in range, apart).  For one scene, test set, planner run and test case:

  - left room: the planner's 7-DOF plan: drone a (blue) and drone b (orange) with their tracks (red where the link
    is broken), the link box (green while the drones see each other within range, red where it breaks), the start
    and goal links and a few link snapshots along the way
  - right room, side by side: the same case flown without planning the link: the independent drones (each on its
    own shortest collision-free path, leaving and arriving together), the straight 7-d motion, or the test set's
    certified roadmap path
  - one time slider (normalized time) and Play drive both rooms; the panel plots the link length over time with
    the allowed band and the contact timeline of both

Read from results/drone_link (plan/<run>/<set>.json + _paths.npz, baselines/<set>.json + _independent.npz) and
testing_data/3dshape/<set>.  The baselines' link status is the separating-axis test of the link box against the
scene's boxes and the room (numpy; the overlap condition of geometry.scene_clearance < 0), the plans' status the
evaluation's own exact clearance.  Missing runs are re-checked on every change: new evaluations appear as the queue
writes them.  Coordinates are the planner's normalized frame (y up).

    python models/drone_link/scripts/link_viser.py --port 8110
    python models/drone_link/scripts/link_viser.py --scene corridor --set crossing --case 111 --t 0.3   (start state)
"""
import argparse
import json
import math
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import viser
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, 'results', 'drone_link')
SETS = os.path.join(ROOT, 'testing_data', '3dshape')
OBJ = os.path.join(ROOT, 'datasets', '3dshape', 'obj')
TWO_PI = 2 * math.pi
SCENES = ['forest', 'windows', 'corridor', 'scaffold']
SET_OF = {'random': 'std', 'crossing': 'cross'}
KINDS = OrderedDict([('NTFields 7 DOF, full recipe', 'ntf'), ('NTFields, 4000 epochs', 'ntf4k'),
                     ('without the TD / normal terms', 'ntfb'), ('long-range pairs, half', 'ntflr'),
                     ('long-range pairs, a fifth', 'ntflr20'), ('scene-clearance band', 'ntfsb')])
COMPARE = ['independent drones', 'straight 7-d motion', 'certified roadmap path', 'none']
FILTERS = ['all', 'planner solves', 'planner breaks the link', 'planner stalls',
           'planner keeps the link, independent drones lose it', 'independent drones keep it, planner fails',
           'planner shortens the link', 'long trips (drone a moves >= 0.7)']
VIEWS = ['three-quarter', 'top', 'front', 'side']
K_BASE = 400                  # time samples of the baselines (baseline_link.py scores them at 400)
GAP = 0.14                    # between the two rooms
PLAY_SECONDS = 10.0
OPACITY = {'forest': 1.0, 'windows': 0.45, 'corridor': 0.35, 'scaffold': 1.0}
VIEW_OF = {'forest': 'three-quarter', 'windows': 'three-quarter', 'corridor': 'top', 'scaffold': 'three-quarter'}

COL_A = (47, 111, 181)
COL_B = (224, 138, 30)
COL_OK = (46, 158, 87)
COL_BAD = (214, 39, 40)
COL_CLOSE = (232, 170, 20)
COL_END = (120, 120, 128)
COL_WALL = (200, 204, 212)
PART_COL = {'trunk': (118, 86, 58), 'crown': (96, 150, 88), 'partition': COL_WALL, 'scaffold': (128, 138, 152)}
# link status codes along a motion
OK, BLOCKED, FAR, CLOSE, HIT = 0, 1, 2, 3, 4
STATUS_TXT = {OK: 'link kept', BLOCKED: 'line of sight blocked', FAR: 'out of range', CLOSE: 'too close',
              HIT: 'a drone hits the scene'}


# ----------------------------------------------------------------------------- scene and geometry (numpy)
def read_obj(path):
    V, groups, cur = [], [], None
    for line in open(path):
        if line.startswith('v '):
            V.append([float(t) for t in line.split()[1:4]])
        elif line.startswith(('g ', 'o ')):
            cur = [line[2:].strip(), []]
            groups.append(cur)
        elif line.startswith('f '):
            if cur is None:
                cur = ['', []]
                groups.append(cur)
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            cur[1].extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return np.asarray(V, np.float64), [(n, np.asarray(F, np.int64)) for n, F in groups]


def ray_hit(o, d, tris):
    """Smallest positive Moller-Trumbore hit distance of one ray with triangles (m, 3, 3)."""
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    p = np.cross(d, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - tris[:, 0]
    u = (s * p).sum(1) * inv
    q = np.cross(s, e1)
    v = (q * d).sum(1) * inv
    t = (e2 * q).sum(1) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    return float(t[hit].min())


class Scene:
    """A link scene in the planner's frame (geometry.Scene's normalization): obstacle boxes, room, link limits."""

    def __init__(self, name):
        self.name = name
        V, groups = read_obj(os.path.join(OBJ, name + '.obj'))
        lo, hi = V.min(0), V.max(0)
        self.center, self.scale = 0.5 * (lo + hi), float((hi - lo).max())
        Vn = (V - self.center) / self.scale
        tris = np.concatenate([Vn[F] for n, F in groups if n.lower().startswith('wall')])
        o = np.zeros(3)
        self.room_lo = np.array([-ray_hit(o, -np.eye(3)[k], tris) for k in range(3)])
        self.room_hi = np.array([ray_hit(o, np.eye(3)[k], tris) for k in range(3)])
        J = json.load(open(os.path.join(OBJ, name + '_boxes.json')))
        self.note = J['note']
        b = np.asarray(J['boxes'], np.float64).reshape(-1, 6)
        self.lo = (b[:, [0, 2, 4]] - self.center) / self.scale
        self.hi = (b[:, [1, 3, 5]] - self.center) / self.scale
        self.parts = []
        for g, bb in zip(J['groups'], b):
            if g.startswith('tree'):
                self.parts.append('trunk' if bb[1] - bb[0] < 20 else 'crown')
            else:
                self.parts.append('scaffold' if g.startswith('scaffold') else 'partition')
        meta = json.load(open(os.path.join(SETS, 'link_{}_std'.format(name), 'meta.json')))['link']
        self.w, self.L_min, self.L_max = float(meta['width']), float(meta['L_min']), float(meta['L_max'])

    @property
    def size(self):
        return self.room_hi - self.room_lo


def box_hits(S, a, R, L):
    """(N,) True where the link box [-w/2, w/2]^2 x [0, L] at drone a (N, 3), R (N, 3, 3) overlaps an obstacle box
    or leaves the room (separating-axis theorem, 15 axes)."""
    n = len(L)
    e = np.stack([np.full(n, S.w / 2), np.full(n, S.w / 2), L / 2], 1)
    c = a + R[:, :, 2] * (L / 2)[:, None]
    r = np.einsum('nj,nkj->nk', e, np.abs(R))                  # half extent along the world axes
    hit = ((c - r) < S.room_lo).any(1) | ((c + r) > S.room_hi).any(1)
    cb, h = 0.5 * (S.lo + S.hi), 0.5 * (S.hi - S.lo)
    D = c[:, None, :] - cb[None]
    sep = (np.abs(D) > r[:, None, :] + h[None]).any(2)
    sep |= (np.abs(np.einsum('nmk,nkj->nmj', D, R)) > e[:, None, :] + np.einsum('mk,nkj->nmj', h, np.abs(R))).any(2)
    for i in range(3):
        for j in range(3):
            ax = np.cross(np.eye(3)[i], R[:, :, j])
            ra = sum(e[:, l] * np.abs((ax * R[:, :, l]).sum(1)) for l in range(3))
            sep |= np.abs(np.einsum('nmk,nk->nm', D, ax)) > ra[:, None] + np.abs(ax) @ h.T + 1e-9
    return hit | (~sep).any(1)


def point_hits(S, p):
    """(N,) True where the points (N, 3) are inside an obstacle box or outside the room."""
    out = ((p < S.room_lo) | (p > S.room_hi)).any(1)
    return out | ((p[:, None] > S.lo[None]) & (p[:, None] < S.hi[None])).all(2).any(1)


def rot_to(d):
    """Shortest rotations (N, 3, 3) taking e_z to the unit vectors d (N, 3)."""
    v = np.cross(np.array([0.0, 0.0, 1.0]), d)
    s = np.linalg.norm(v, axis=1)
    rv = v / np.maximum(s, 1e-12)[:, None] * np.arctan2(s, d[:, 2])[:, None]
    rv[(s < 1e-9) & (d[:, 2] < 0)] = [math.pi, 0.0, 0.0]
    return Rotation.from_rotvec(rv).as_matrix()


def resample(P, k):
    """k points uniform in arclength along the polyline P (constant speed)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] < 1e-12:
        return np.repeat(P[:1], k, 0)
    u = np.linspace(0, s[-1], k)
    return np.stack([np.interp(u, s, P[:, j]) for j in range(3)], 1)


class Track:
    """A motion of the link sampled at k times (normalized time u_i = i / (k - 1)): drone a (k, 3), R (k, 3, 3),
    L (k,), the link status per sample (OK, BLOCKED, ...)."""

    def __init__(self, a, R, L, status, name):
        self.a, self.R, self.L, self.status, self.name = a, R, L, np.asarray(status), name
        self.b = a + R[:, :, 2] * L[:, None]
        self.rot = Rotation.from_matrix(R)

    def at(self, u):
        k = len(self.L)
        if k == 1:
            return self.a[0], self.R[0], self.L[0], int(self.status[0])
        x = float(np.clip(u, 0, 1)) * (k - 1)
        i = min(int(x), k - 2)
        f = x - i
        R = Slerp([0, 1], self.rot[i:i + 2])(f).as_matrix()
        st = self.status[i + 1] if f > 1e-9 else self.status[i]
        return (1 - f) * self.a[i] + f * self.a[i + 1], R, (1 - f) * self.L[i] + f * self.L[i + 1], int(st)

    def kept(self):
        return float((self.status == OK).mean())


def plan_track(S, run, case):
    """The planner's path of one case as scored (cut at SE(3) convergence) with the evaluation's clearance."""
    j = run['index'].get(case)
    if j is None:
        return None
    P = run['path'][j].astype(np.float64)
    k3 = int(run['iters3'][j])
    if k3 >= 0:
        P, cl = P[:k3 + 2], run['clear'][j][:k3 + 1]
    else:
        P = P[~np.isnan(P[:, 0])]
        cl = run['clear'][j][:len(P) - 1]
    R = Rotation.from_rotvec(P[:, 3:6] * TWO_PI).as_matrix()
    a, L = P[:, :3], P[:, 6]
    b = a + R[:, :, 2] * L[:, None]
    st = np.zeros(len(P), np.int8)                           # waypoint i: the motion that arrives there
    st[1:] = np.where(np.nan_to_num(cl, nan=0.0).min(1) < 0, BLOCKED, OK)
    st[(st != OK) & (point_hits(S, a) | point_hits(S, b))] = HIT
    return Track(a, R, L, st, 'plan')


def pair_track(S, A, B, name):
    """Two drones at synchronized positions A, B (k, 3); the link between them is the box from a to b."""
    D = B - A
    L = np.linalg.norm(D, axis=1)
    R = rot_to(D / np.maximum(L, 1e-12)[:, None])
    blocked = box_hits(S, A, R, np.maximum(L, 1e-6))
    st = np.where(blocked, BLOCKED, np.where(L > S.L_max, FAR, np.where(L < S.L_min, CLOSE, OK)))
    st[point_hits(S, A) | point_hits(S, B)] = HIT
    return Track(A, R, L, st, name)


def chart_track(S, W, name, k=K_BASE):
    """Constant-speed motion through chart waypoints W (n, 8) = (t, quaternion w x y z, L): t and L linear, R slerp."""
    q = W[:, [4, 5, 6, 3]]                                     # scipy wants x y z w
    rot = Rotation.from_quat(q)
    rel = [(rot[i].inv() * rot[i + 1]).magnitude() for i in range(len(W) - 1)]
    seg = np.sqrt(((W[1:, :3] - W[:-1, :3]) ** 2).sum(1) + (np.array(rel) / TWO_PI) ** 2 + (W[1:, 7] - W[:-1, 7]) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], k)
    i = np.clip(np.searchsorted(s, u, side='right') - 1, 0, len(W) - 2)
    f = np.clip(np.where(seg[i] > 1e-12, (u - s[i]) / np.maximum(seg[i], 1e-12), 0.0), 0.0, 1.0)
    a = W[i, :3] + f[:, None] * (W[i + 1, :3] - W[i, :3])
    L = W[i, 7] + f * (W[i + 1, 7] - W[i, 7])
    R = np.empty((k, 3, 3))
    for ii in np.unique(i):
        m = i == ii
        R[m] = Slerp([0, 1], rot[ii:ii + 2])(f[m]).as_matrix()
    b = a + R[:, :, 2] * L[:, None]
    st = np.where(box_hits(S, a, R, L), BLOCKED, OK)
    st[(st != OK) & (point_hits(S, a) | point_hits(S, b))] = HIT
    return Track(a, R, L, st, name)


# ----------------------------------------------------------------------------- meshes
def box_mesh(lo, hi):
    """Triangles of axis-aligned boxes lo, hi (m, 3) -> V (8m, 3), F (12m, 3)."""
    F0 = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    Vs, Fs = [], []
    for k, (l, h) in enumerate(zip(lo, hi)):
        Vs.append(np.array([[l[0], l[1], l[2]], [h[0], l[1], l[2]], [h[0], h[1], l[2]], [l[0], h[1], l[2]],
                            [l[0], l[1], h[2]], [h[0], l[1], h[2]], [h[0], h[1], h[2]], [l[0], h[1], h[2]]]))
        Fs.append(F0 + 8 * k)
    return np.concatenate(Vs).astype(np.float32), np.concatenate(Fs).astype(np.int32)


def _merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.concatenate(V).astype(np.float32), np.concatenate(F).astype(np.int32)


def _obox(c, half, R):
    V, F = box_mesh(-half[None], half[None])
    return V.astype(np.float64) @ R.T + c, F


def _disc(c, r, h, n=14):
    a = np.linspace(0, TWO_PI, n, endpoint=False)
    ring = np.stack([r * np.cos(a), np.zeros(n), r * np.sin(a)], 1)
    V = np.concatenate([ring - [0, h / 2, 0], ring + [0, h / 2, 0], [[0, -h / 2, 0], [0, h / 2, 0]]]) + c
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return V, np.array(F)


def drone_meshes(span=0.05):
    """A small quadrotor around the origin (y up): (frame V, F), (rotors V, F)."""
    s = span / 2 / math.sqrt(2)
    frame = [_obox(np.zeros(3), np.array([0.006, 0.0028, 0.006]), np.eye(3))]
    for ang in (45, 135):
        frame.append(_obox(np.zeros(3), np.array([span / 2, 0.0011, 0.0014]), Rotation.from_euler('y', ang, degrees=True).as_matrix()))
    rotors = [_disc(np.array([sx * s, 0.0018, sz * s]), 0.0085, 0.001) for sx in (-1, 1) for sz in (-1, 1)]
    return _merge(frame), _merge(rotors)


def yaw_wxyz(d):
    """Upright drone turned towards the horizontal direction d (scene y up)."""
    yaw = math.atan2(d[0], d[2]) if abs(d[0]) + abs(d[2]) > 1e-9 else 0.0
    q = Rotation.from_euler('y', yaw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def wxyz(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


# ----------------------------------------------------------------------------- the panel chart
try:
    FONT, FONT_S = ImageFont.load_default(size=22), ImageFont.load_default(size=19)
except TypeError:                                              # Pillow < 10.1
    FONT = FONT_S = ImageFont.load_default()
CW, CH = 640, 330
PX0, PX1, PY0, PY1 = 64, 624, 44, 226                          # plot box (length vs time)
TL = (250, 270)                                                # timeline rows' top y
STATUS_COL = {OK: COL_OK, BLOCKED: COL_BAD, FAR: COL_BAD, CLOSE: COL_CLOSE, HIT: COL_BAD}
COL_LEFT, COL_RIGHT = (40, 40, 44), (150, 84, 200)


def chart_base(S, left, right, names):
    """Link length over normalized time (left: dark, right: purple) in the allowed band, contact timelines below."""
    im = Image.new('RGB', (CW, CH), (255, 255, 255))
    d = ImageDraw.Draw(im)
    tracks = [t for t in (left, right) if t is not None]
    top = max([0.32] + [float(t.L.max()) + 0.02 for t in tracks])
    X = lambda u: PX0 + u * (PX1 - PX0)  # noqa: E731
    Y = lambda L: PY1 - L / top * (PY1 - PY0)  # noqa: E731
    d.rectangle([PX0, Y(S.L_max), PX1, Y(S.L_min)], fill=(234, 245, 237))
    for v in (S.L_min, S.L_max):
        for x in range(PX0, PX1, 14):
            d.line([x, Y(v), min(x + 7, PX1), Y(v)], fill=COL_OK, width=2)
    d.line([PX0, PY0, PX0, PY1, PX1, PY1], fill=(150, 150, 150), width=2)
    for v in (0.1, 0.2, 0.3):
        if v < top:
            d.text((PX0 - 8, Y(v)), '{:.1f}'.format(v), fill=(90, 90, 90), font=FONT_S, anchor='rm')
    for u in (0, 0.5, 1):
        d.text((X(u), PY1 + 6), '{:g}'.format(u), fill=(90, 90, 90), font=FONT_S, anchor='ma')
    d.text((PX0, 6), 'link length', fill=(60, 60, 60), font=FONT, anchor='la')
    lx = PX0 + 150
    for t, col, nm in ((left, COL_LEFT, names[0]), (right, COL_RIGHT, names[1])):
        if t is None:
            continue
        u = np.linspace(0, 1, len(t.L))
        d.line(list(zip(X(u), Y(t.L))), fill=col, width=3, joint='curve')
        d.line([lx, 18, lx + 26, 18], fill=col, width=4)
        d.text((lx + 32, 18), nm, fill=(60, 60, 60), font=FONT_S, anchor='lm')
        lx += 46 + int(d.textlength(nm, font=FONT_S))
    for row, (t, nm) in enumerate(((left, names[0]), (right, names[1]))):
        y0 = TL[row]
        d.text((PX0 - 8, y0 + 9), nm[:6], fill=(90, 90, 90), font=FONT_S, anchor='rm')
        if t is None:
            continue
        k = len(t.status)
        for i in range(k):
            x0, x1 = X(i / k), X((i + 1) / k)
            d.rectangle([x0, y0, max(x1 - 0.5, x0), y0 + 18], fill=STATUS_COL[int(t.status[i])])
    d.text((PX0, CH - 4), 'normalized time', fill=(90, 90, 90), font=FONT_S, anchor='ld')
    return im


def chart_at(base, u):
    im = base.copy()
    d = ImageDraw.Draw(im)
    x = PX0 + float(np.clip(u, 0, 1)) * (PX1 - PX0)
    d.line([x, PY0 - 4, x, TL[1] + 22], fill=(20, 20, 20), width=2)
    return np.asarray(im)


# ----------------------------------------------------------------------------- data
class Data:
    """Lazy, cached loading of planner runs, baselines and test sets (missing runs are looked up again)."""

    def __init__(self):
        self.runs = OrderedDict()
        self.sets = {}

    def run(self, scene, set_, kind, seed):
        key = (scene, set_, kind, seed)
        if key in self.runs:
            self.runs.move_to_end(key)
            return self.runs[key]
        base = os.path.join(RES, 'plan', '{}_{}_s{}'.format(scene, kind, seed), 'link_{}_{}'.format(scene, set_))
        if not (os.path.exists(base + '.json') and os.path.exists(base + '_paths.npz')):
            return None
        try:
            recs = {r['case']: r for r in json.load(open(base + '.json'))['records']}
            z = np.load(base + '_paths.npz')
            r = dict(recs=recs, path=z['path'], clear=z['clear'], iters3=z['iters3'],
                     index={int(c): i for i, c in enumerate(z['case'])})
        except (OSError, ValueError, KeyError):
            return None
        self.runs[key] = r
        while len(self.runs) > 6:
            self.runs.popitem(last=False)
        return r

    def test_set(self, scene, set_):
        key = (scene, set_)
        if key not in self.sets:
            d = os.path.join(SETS, 'link_{}_{}'.format(scene, set_))
            X = np.load(os.path.join(d, 'sampled_points.npy')).astype(np.float64)
            cz = np.load(os.path.join(d, 'cases.npz'))
            off = np.concatenate([[0], np.cumsum(cz['rrt_len'])])
            out = dict(X=X, cert=[cz['rrt_flat'][off[i]:off[i + 1]] for i in range(len(X))], ind=None, base=None)
            bp = os.path.join(RES, 'baselines', 'link_{}_{}'.format(scene, set_))
            if os.path.exists(bp + '.json') and os.path.exists(bp + '_independent.npz'):
                out['base'] = {r['case']: r for r in json.load(open(bp + '.json'))['records']}
                z = np.load(bp + '_independent.npz', allow_pickle=True)
                out['ind'] = {int(c): (np.asarray(a, np.float64), np.asarray(b, np.float64))
                              for c, a, b in zip(z['case'], z['a'], z['b'])}
            self.sets[key] = out
        return self.sets[key]


def results_markdown():
    p = os.path.join(RES, 'summary', 'summary.json')
    if not os.path.exists(p):
        return 'no summary yet (aggregate_link.py)'
    S = json.load(open(p))['sets']
    t = '| scene | set | direct | indep. | NTFields | a fifth long-range |\n|---|---|---|---|---|---|\n'
    for sc in SCENES:
        for nm, s in SET_OF.items():
            r = S.get('link_{}_{}'.format(sc, s))
            if not r:
                continue
            b, ntf = r['baselines'], r.get('ntf') or {}
            seeds = ntf.get('success_per_seed') or []
            l20 = (r.get('ntflr20_per_seed') or {}).get('success_rate_se3') or []
            t += '| {} | {} | {:.1f} | {:.1f} | {} | {} |\n'.format(
                sc, nm, 100 * b['straight']['success_rate'], 100 * b['independent']['valid_rate'],
                '{:.1f} +- {:.1f}'.format(100 * np.mean(seeds), 100 * np.std(seeds)) if seeds else '-',
                ' / '.join('{:.0f}'.format(100 * x) for x in l20) or '-')
    return ('Percent of 1000 certified cases solved with the link valid the whole way (NTFields: mean +- std over '
            'the seeds; a fifth of the training pairs long-range: seeds 0 / 1 / 2)\n\n' + t)


# ----------------------------------------------------------------------------- the viewer
class Viewer:
    def __init__(self, server, args):
        self.server, self.sc = server, server.scene
        self.sc.set_up_direction('+y')
        server.gui.configure_theme(control_width='large')
        self.scenes = {s: Scene(s) for s in SCENES}
        self.data = Data()
        self.lock = threading.RLock()
        self.handles = {}
        self.case_ids = []
        self.tracks = (None, None)
        self.chart = None
        self.playing = False
        self.drone_frame, self.drone_rotors = drone_meshes()

        gui = server.gui
        with gui.add_folder('Scene'):
            self.scene_dd = gui.add_dropdown('Scene', SCENES, initial_value=args.scene)
            self.set_dd = gui.add_dropdown('Test set', list(SET_OF), initial_value=args.set)
            self.kind_dd = gui.add_dropdown('Planner', list(KINDS), initial_value=args.planner)
            self.seed_dd = gui.add_dropdown('Seed', ['0', '1', '2'], initial_value=str(args.seed))
            self.cmp_dd = gui.add_dropdown('Compare with', COMPARE, initial_value=args.compare)
            self.view_dd = gui.add_dropdown('View', VIEWS, initial_value=args.view or VIEW_OF[args.scene])
            self.opa_sl = gui.add_slider('Obstacle opacity', 0.15, 1.0, 0.05, OPACITY[args.scene])
            self.scene_md = gui.add_markdown('')
        with gui.add_folder('Test cases'):
            self.filter_dd = gui.add_dropdown('Show', FILTERS, initial_value=args.filter)
            self.case_sl = gui.add_slider('Case', 0, 1, 1, 0)
            self.prev_bt = gui.add_button('Previous')
            self.next_bt = gui.add_button('Next')
            self.case_md = gui.add_markdown('')
        with gui.add_folder('Motion'):
            self.t_sl = gui.add_slider('Time', 0.0, 1.0, 0.002, float(args.t))
            self.play_bt = gui.add_button('Play / pause')
            self.speed_sl = gui.add_slider('Play speed', 0.1, 3.0, 0.1, 1.0)
            self.snap_cb = gui.add_checkbox('Link snapshots', True)
            self.snap_n = gui.add_slider('Snapshots', 2, 24, 1, 7)
            self.img = gui.add_image(np.full((CH // 2, CW // 2, 3), 255, np.uint8), label='link length', format='png')
            gui.add_markdown('<sub>timelines: green = the drones see each other within range; red = line of sight '
                             'blocked, out of range or a drone in the scene; amber = closer than L_min</sub>')
            self.now_md = gui.add_markdown('')
        with gui.add_folder('Results', expand_by_default=False):
            gui.add_markdown(results_markdown())

        self.scene_dd.on_update(lambda _: self.scene_changed())
        for w in (self.set_dd, self.kind_dd, self.seed_dd):
            w.on_update(lambda _: self.refilter())
        self.cmp_dd.on_update(lambda _: self.compare_changed())
        self.view_dd.on_update(lambda _: self.set_view())
        self.opa_sl.on_update(lambda _: self.draw_env())
        self.filter_dd.on_update(lambda _: self.refilter())
        self.case_sl.on_update(lambda _: self.draw_case())
        self.prev_bt.on_click(lambda _: self.step_case(-1))
        self.next_bt.on_click(lambda _: self.step_case(+1))
        self.t_sl.on_update(lambda _: self.draw_mover())
        self.play_bt.on_click(lambda _: self.toggle_play())
        self.snap_cb.on_update(lambda _: self.draw_case(keep_t=True))
        self.snap_n.on_update(lambda _: self.draw_case(keep_t=True))
        server.on_client_connect(self.client_connected)
        threading.Thread(target=self.player, daemon=True).start()
        self.draw_env()
        self.refilter(first_case=args.case, keep_t=True)

    # ---------- state
    @property
    def S(self):
        return self.scenes[self.scene_dd.value]

    def key(self):
        return self.scene_dd.value, SET_OF[self.set_dd.value], KINDS[self.kind_dd.value], int(self.seed_dd.value)

    def run(self):
        return self.data.run(*self.key())

    def test_set(self):
        return self.data.test_set(self.scene_dd.value, SET_OF[self.set_dd.value])

    def offset(self, side):
        """x offset of a room: the comparison room sits on the -x side (on screen right of the planner's room when
        seen from the scenes' front, -z)."""
        return 0.0 if side == 'left' else -(self.S.size[0] + GAP)

    def sides(self):
        return ['left'] if self.cmp_dd.value == 'none' else ['left', 'right']

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

    # ---------- callbacks
    def scene_changed(self):
        self.opa_sl.value = OPACITY[self.scene_dd.value]       # redraws the rooms
        self.refilter()
        if self.view_dd.value != VIEW_OF[self.scene_dd.value]:
            self.view_dd.value = VIEW_OF[self.scene_dd.value]  # moves the camera
        else:
            self.set_view()

    def compare_changed(self):
        self.draw_env()
        self.draw_case(keep_t=True)
        self.set_view()

    def step_case(self, k):
        if self.case_ids:
            self.case_sl.value = int(np.clip(self.case_sl.value + k, 0, len(self.case_ids) - 1))

    # ---------- rooms
    def draw_env(self):
        with self.lock:
            self.clear('/env')
            S = self.S
            self.scene_md.content = '*{}*. Link width {:.2f}, length {:.2f}-{:.2f} (room {:.2f} x {:.2f} x {:.2f}).'.format(
                S.note, S.w, S.L_min, S.L_max, *S.size)
            for side in self.sides():
                o = np.array([self.offset(side), 0.0, 0.0])
                lo, hi = S.room_lo + o, S.room_hi + o
                c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)
                e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
                seg = np.stack([c[[i for i, _ in e]], c[[j for _, j in e]]], 1)
                p = '/env/' + side
                self.add(p + '/edges', self.sc.add_line_segments(p + '/edges', seg, (130, 130, 138), line_width=1.5))
                floor = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]], np.float32)
                self.add(p + '/floor', self.sc.add_mesh_simple(p + '/floor', floor, np.array([[0, 2, 1], [0, 3, 2]], np.int32),
                                                               color=(226, 222, 212), side='double', cast_shadow=False))
                for part in sorted(set(S.parts)):
                    idx = [i for i, q in enumerate(S.parts) if q == part]
                    V, F = box_mesh(S.lo[idx] + o, S.hi[idx] + o)
                    op = float(self.opa_sl.value)
                    self.add(p + '/' + part, self.sc.add_mesh_simple(
                        p + '/' + part, V, F, color=PART_COL[part], flat_shading=True,
                        opacity=None if op >= 0.999 else op, side='front' if op >= 0.999 else 'double'))
                title = self.kind_dd.value if side == 'left' else self.cmp_dd.value
                self.add(p + '/title', self.sc.add_label(p + '/title', title, position=(
                    0.5 * (lo[0] + hi[0]), lo[1], lo[2] - 0.06)))      # on the floor plane, in front of the room

    # ---------- cases
    def refilter(self, first_case=None, keep_t=False):
        with self.lock:
            self.playing = False
            run, ts = self.run(), self.test_set()
            n = len(ts['X'])
            f = self.filter_dd.value
            rec = (lambda c: run['recs'].get(c)) if run else (lambda c: None)
            ind = (lambda c: (ts['base'] or {}).get(c, {}).get('independent', {}))
            if f == 'all' or (run is None and f != 'all'):
                ids = list(range(n))
            elif f == 'planner solves':
                ids = [c for c in range(n) if rec(c) and rec(c)['success_se3']]
            elif f == 'planner breaks the link':
                ids = [c for c in range(n) if rec(c) and rec(c)['collision_se3']]
            elif f == 'planner stalls':
                ids = [c for c in range(n) if rec(c) and not rec(c)['converged_se3']]
            elif f == 'planner keeps the link, independent drones lose it':
                ids = [c for c in range(n) if rec(c) and rec(c)['success_se3'] and ind(c).get('solved') and not ind(c)['valid']]
                ids.sort(key=lambda c: (ind(c)['contact_frac'], ind(c)['valid_frac']))
            elif f == 'independent drones keep it, planner fails':
                ids = [c for c in range(n) if rec(c) and not rec(c)['success_se3'] and ind(c).get('valid')]
            elif f == 'planner shortens the link':
                ids = [c for c in range(n) if rec(c) and rec(c)['success_se3'] and
                       rec(c)['L_min_used'] < min(rec(c)['L0'], rec(c)['L1']) - 0.02]
                ids.sort(key=lambda c: rec(c)['L_min_used'] - min(rec(c)['L0'], rec(c)['L1']))
            else:
                ids = [c for c in range(n) if np.linalg.norm(ts['X'][c, 7:10] - ts['X'][c, :3]) >= 0.7]
            self.case_ids = ids
            self.case_sl.max = max(len(ids) - 1, 1)
            pos = ids.index(first_case) if first_case is not None and first_case in ids else 0
            if self.case_sl.value != pos:
                self._keep_t = keep_t
                self.case_sl.value = pos                        # draws through the callback
            else:
                self.draw_case(keep_t=keep_t)

    def case_tracks(self, case):
        S, ts, run = self.S, self.test_set(), self.run()
        left = plan_track(S, run, case) if run else None
        right, X = None, ts['X'][case]
        cmp = self.cmp_dd.value
        if cmp == 'independent drones' and ts['ind'] is not None and case in ts['ind']:
            A, B = ts['ind'][case]
            right = pair_track(S, resample(A, K_BASE), resample(B, K_BASE), 'independent')
        elif cmp == 'straight 7-d motion':
            W = np.stack([np.concatenate([x[:3], Rotation.from_rotvec(x[3:6] * TWO_PI).as_quat()[[3, 0, 1, 2]], x[6:7]])
                          for x in (X[:7], X[7:])])
            right = chart_track(S, W, 'straight')
        elif cmp == 'certified roadmap path':
            right = chart_track(S, ts['cert'][case], 'certified')
        return left, right

    def draw_case(self, keep_t=False):
        keep_t = keep_t or getattr(self, '_keep_t', False)
        self._keep_t = False
        with self.lock:
            self.playing = False
            self.clear('/case')
            self.clear('/mover')
            if not self.case_ids:
                self.case_md.content = 'No test case matches this filter.'
                self.tracks, self.chart = (None, None), None
                return
            case = self.case_ids[int(min(self.case_sl.value, len(self.case_ids) - 1))]
            self.case = case
            left, right = self.case_tracks(case)
            self.tracks = (left, right)
            names = ('plan', {'independent drones': 'indep.', 'straight 7-d motion': 'straight',
                              'certified roadmap path': 'certif.', 'none': ''}[self.cmp_dd.value])
            self.chart = chart_base(self.S, left, right, names)
            for side, t in zip(('left', 'right'), (left, right)):
                if t is not None and side in self.sides():
                    self.draw_track(side, t)
            if left is None:                                   # no plan: show the case's start and goal links
                X = self.test_set()['X'][case]
                for nm, x in (('start', X[:7]), ('goal', X[7:])):
                    R = Rotation.from_rotvec(x[3:6] * TWO_PI).as_matrix()
                    self.end_link('/case/left/' + nm, x[:3], R, x[6], 0.0, nm)
            self.describe(case, left, right)
            if not keep_t:
                self.t_sl.value = 0.0
        self.draw_mover()

    def end_link(self, name, a, R, L, off, label):
        c = a + R[:, 2] * L / 2 + [off, 0, 0]
        self.add(name, self.sc.add_box(name, color=COL_END, dimensions=(self.S.w, self.S.w, float(L)), opacity=0.35,
                                       position=c, wxyz=wxyz(R), cast_shadow=False))
        b = a + R[:, 2] * L
        top = max(a[1], b[1]) + 0.035
        self.add(name + '_label', self.sc.add_label(name + '_label', label, position=(
            0.5 * (a[0] + b[0]) + off, top, 0.5 * (a[2] + b[2]))))

    def draw_track(self, side, t):
        off = self.offset(side)
        o = np.array([off, 0.0, 0.0], np.float32)
        p = '/case/' + side
        bad = t.status != OK
        segbad = bad[1:] | bad[:-1]
        for nm, P, col in (('a', t.a, COL_A), ('b', t.b, COL_B)):
            seg = np.stack([P[:-1], P[1:]], 1).astype(np.float32) + o
            cols = np.where(segbad[:, None], COL_BAD, col).astype(np.uint8)
            self.add(p + '/track_' + nm, self.sc.add_line_segments(p + '/track_' + nm, seg, np.repeat(cols[:, None], 2, 1),
                                                                   line_width=2.5))
        self.end_link(p + '/start', t.a[0], t.R[0], t.L[0], off, 'start')
        self.end_link(p + '/goal', t.a[-1], t.R[-1], t.L[-1], off, 'goal')
        if self.snap_cb.value:
            for i, u in enumerate(np.linspace(0, 1, int(self.snap_n.value) + 2)[1:-1]):
                a, R, L, st = t.at(u)
                nm = p + '/snap%d' % i
                self.add(nm, self.sc.add_box(nm, color=COL_OK if st == OK else (COL_CLOSE if st == CLOSE else COL_BAD),
                                             dimensions=(self.S.w, self.S.w, float(max(L, 1e-4))), opacity=0.28,
                                             position=a + R[:, 2] * L / 2 + o, wxyz=wxyz(R), cast_shadow=False))

    def describe(self, case, left, right):
        S, ts, run = self.S, self.test_set(), self.run()
        X = ts['X'][case]
        da = np.linalg.norm(X[7:10] - X[:3])
        Rs, Rg = (Rotation.from_rotvec(x[3:6] * TWO_PI).as_matrix() for x in (X[:7], X[7:]))
        db = np.linalg.norm((X[7:10] + Rg[:, 2] * X[13]) - (X[:3] + Rs[:, 2] * X[6]))
        lines = ['**{} / {} / case {}** ({} of {} shown): drone a moves {:.2f}, drone b {:.2f} (straight line); '
                 'link {:.2f} at the start, {:.2f} at the goal'.format(
                     S.name, self.set_dd.value, case, int(self.case_sl.value) + 1, len(self.case_ids), da, db, X[6], X[13])]
        if run is None:
            k = self.key()
            lines.append('- **{}**: no evaluation of `{}_{}_s{}` on this set yet'.format(self.kind_dd.value, k[0], k[2], k[3]))
        elif left is not None:
            r = run['recs'][case]
            out = ('**solved**' if r['success_se3'] else
                   '**link broken** on the way' if r['collision_se3'] and r['converged_se3'] else
                   '**stalls** (never reaches the goal){}'.format(', link broken on the way' if r['collision_se3'] else ''))
            lines.append('- **plan** ({}, seed {}): {}; link kept {:.0f} % of the time; length {:.2f}-{:.2f}{}; '
                         'drone a flies {:.2f}, drone b {:.2f}'.format(
                             self.kind_dd.value, self.seed_dd.value, out, 100 * left.kept(), r['L_min_used'], r['L_max_used'],
                             ' (shortened to pass)' if r['success_se3'] and r['L_min_used'] < min(r['L0'], r['L1']) - 0.02 else '',
                             r['trans_len_a'], r['trans_len_b']))
        if right is not None:
            st = right.status
            parts = ', '.join('{} {:.0f} %'.format(STATUS_TXT[k], 100 * (st == k).mean()) for k in (BLOCKED, FAR, CLOSE, HIT)
                              if (st == k).any())
            la = np.linalg.norm(np.diff(right.a, axis=0), axis=1).sum()
            lb = np.linalg.norm(np.diff(right.b, axis=0), axis=1).sum()
            lines.append('- **{}**: link kept {:.0f} % of the time{}; drone a flies {:.2f}, drone b {:.2f}'.format(
                self.cmp_dd.value, 100 * right.kept(), ' ({})'.format(parts) if parts else ' (valid all the way)', la, lb))
        elif self.cmp_dd.value == 'independent drones':
            lines.append('- independent drones: no baseline paths for this case')
        self.case_md.content = '\n'.join(lines)

    # ---------- the moving link
    def draw_mover(self):
        if self.tracks == (None, None) or not self.case_ids:
            return
        with self.lock:
            u = float(self.t_sl.value)
            now = []
            for side, t in zip(('left', 'right'), self.tracks):
                p = '/mover/' + side
                if t is None or side not in self.sides():
                    self.clear(p)
                    continue
                a, R, L, st = t.at(u)
                o = np.array([self.offset(side), 0.0, 0.0])
                b = a + R[:, 2] * L
                col = COL_OK if st == OK else (COL_CLOSE if st == CLOSE else COL_BAD)
                box = self.handles.get(p + '/link')
                if box is None:
                    box = self.add(p + '/link', self.sc.add_box(p + '/link', color=col, dimensions=(self.S.w, self.S.w, float(L)),
                                                                opacity=0.6, cast_shadow=False))
                    for nm, c in (('a', COL_A), ('b', COL_B)):
                        self.add(p + '/drone_' + nm, self.sc.add_frame(p + '/drone_' + nm, show_axes=False))
                        self.add(p + '/drone_' + nm + '/frame', self.sc.add_mesh_simple(
                            p + '/drone_' + nm + '/frame', *self.drone_frame, color=c, flat_shading=True))
                        self.add(p + '/drone_' + nm + '/rotors', self.sc.add_mesh_simple(
                            p + '/drone_' + nm + '/rotors', *self.drone_rotors, color=tuple(int(0.5 * v + 120) for v in c),
                            flat_shading=True))
                box.dimensions = (self.S.w, self.S.w, float(max(L, 1e-4)))
                box.position = a + R[:, 2] * L / 2 + o
                box.wxyz = wxyz(R)
                box.color = col
                h = (b - a) / max(L, 1e-9)
                for nm, q, d in (('a', a, h), ('b', b, -h)):
                    fr = self.handles[p + '/drone_' + nm]
                    fr.position = q + o
                    fr.wxyz = yaw_wxyz(d)
                who = 'plan' if side == 'left' else self.cmp_dd.value
                now.append('{}: length **{:.3f}**, {}'.format(who, L, STATUS_TXT[st] if st != OK else 'link kept'))
            self.now_md.content = 'time {:.3f}: '.format(u) + '; '.join(now)
            if self.chart is not None:
                self.img.image = chart_at(self.chart, u)

    # ---------- views and playback
    def set_view(self, clients=None):
        S = self.S
        lo, hi = S.room_lo.copy(), S.room_hi.copy()
        if 'right' in self.sides():
            lo[0] += self.offset('right')
        c = 0.5 * (lo + hi)
        span = float(max(hi[0] - lo[0], hi[2] - lo[2]))
        look = (c[0], lo[1] + 0.25 * (hi[1] - lo[1]), c[2])
        dirs = {'three-quarter': (-0.35, 0.75, -1.0), 'top': (0.0, 1.0, -0.08), 'front': (0.0, 0.3, -1.0),
                'side': (1.0, 0.45, -0.2)}
        d = np.array(dirs[self.view_dd.value])
        pos = np.array(look) + d / np.linalg.norm(d) * span * (1.25 if self.view_dd.value == 'top' else 1.55)
        for client in (clients or self.server.get_clients().values()):
            with client.atomic():
                client.camera.up_direction = (0.0, 1.0, 0.0)
                client.camera.position = tuple(float(v) for v in pos)
                client.camera.look_at = tuple(float(v) for v in look)
                client.camera.fov = math.radians(50)

    def client_connected(self, client):
        self.set_view([client])
        # a client that is still setting up its camera controls can drop the first camera message: send it again
        threading.Timer(1.5, lambda: self.set_view([client]) if client.client_id in self.server.get_clients() else None).start()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing and self.t_sl.value >= 1.0:
            self.t_sl.value = 0.0

    def player(self):                                          # a motion takes PLAY_SECONDS at speed 1
        last = time.time()
        while True:
            time.sleep(0.04)
            now = time.time()
            dt, last = now - last, now
            if self.playing and self.case_ids and self.tracks != (None, None):
                t = self.t_sl.value + self.speed_sl.value * dt / PLAY_SECONDS
                if t >= 1.0:
                    t, self.playing = 1.0, False
                self.t_sl.value = t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8110)
    ap.add_argument('--scene', default='corridor', choices=SCENES)
    ap.add_argument('--set', default='crossing', choices=list(SET_OF))
    ap.add_argument('--planner', default=list(KINDS)[0], choices=list(KINDS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--compare', default='independent drones', choices=COMPARE)
    ap.add_argument('--filter', default='planner keeps the link, independent drones lose it', choices=FILTERS)
    ap.add_argument('--view', default=None, choices=VIEWS, help='camera (default: per scene)')
    ap.add_argument('--case', type=int, default=None, help='test case index to open (if the filter keeps it)')
    ap.add_argument('--t', type=float, default=0.0)
    args = ap.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    Viewer(server, args)
    print('viser at http://{}:{}'.format(args.host, server.get_port()), flush=True)   # viser moves to a free port
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
