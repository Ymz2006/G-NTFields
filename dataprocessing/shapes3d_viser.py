"""Interactive viser viewer for the shapes3d campaign.

Two modes (dropdown "View"):
  Shapes        the five new shapes next to Tshape3d / Lshape3d / Ashape3d, same scale
  Planning      an environment (env4 / env8 / env7 / env5 / env9 / env6), one shape, and the
                metric-baseline MPPI paths of its test pairs: path coloured by clearance
                (red waypoints collide), start / goal ghosts, the shape moving along the
                path (slider or Play), optional ghost trail, filters by outcome and by
                partitions crossed.  A results panel shows the success grid.
                The moving-sofa halls (2dhall44 / 34 / 30, U sofa, planar --2d runs) are in
                the same list; there "crossing" means start and goal in different legs.

Reads datasets/3dshape/obj/*.obj, testing_data/3dshape/<cell>/meta.json,
results/shapes3d/paths/<cell>.npz (dataprocessing/shapes3d_dump_all.sh; other path sets with --paths
LABEL=DIR ..., chosen in the "Model" dropdown, e.g. models/graph_homotopy/scripts/export_viser_paths.py) and
results/shapes3d/summary.json (dataprocessing/shapes3d_report.py).  Everything is
shown in the planner's normalized coordinates of the cell (env units / 220.457, y up;
sofa halls / 170, z up).

    python dataprocessing/shapes3d_viser.py --port 8097
    python dataprocessing/shapes3d_viser.py --port 8098 --initial-env env9 \
        --paths 'homotopy head=results/graph_homotopy/viser_paths/homotopy' \
                'full NTFields=results/graph_homotopy/viser_paths/ntf_full'
"""
import argparse
import functools
import json
import os
import threading
import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'datasets', '3dshape', 'obj')
SHAPES = ['tripod3d', 'crank3d', 'loop3d', 'twistI3d', 'chair3d']
OLD_SHAPES = ['Tshape3d', 'Lshape3d', 'Ashape3d']
ENVS = ['env4', 'env8', 'env7', 'env5', 'env9', 'env6']
HALLS = ['2dhall44', '2dhall34', '2dhall30']
SHAPES_2D = ['Usofa2d']
ENV_NOTE = {'env4': "A's room (no partition)", 'env8': 'partition, 70x70 window', 'env7': 'partition, 60x60 window',
            'env5': 'partition, 50x50 window', 'env6': 'two crossing partitions, four 50x50 windows',
            'env9': 'two crossing partitions, four 70x70 windows',
            '2dhall44': 'sofa, corridor W 44 (easy: turns either way round, clearance 8.9 / 5.5)',
            '2dhall34': 'sofa, corridor W 34 (medium: turns only notch toward the inner corner, clearance 3.45)',
            '2dhall30': 'sofa, corridor W 30 (hard: turns only notch toward the inner corner, clearance 1.41)'}
PLANES = {'env5': [(2, -90.0)], 'env7': [(2, -90.0)], 'env8': [(2, -90.0)], 'env6': [(2, -90.0), (0, 100.0)],
          'env9': [(2, -90.0), (0, 100.0)]}
SCALE = 220.457
CENTER = np.array([100.180961, 86.09025399999999, -89.991084])

COL_NEW = (66, 135, 245)
COL_OLD = (160, 160, 160)
COL_OBST = (205, 170, 125)
COL_WALL = (200, 200, 210)
COL_START = (40, 170, 90)
COL_GOAL = (60, 110, 230)
COL_MOVE = (245, 150, 40)
COL_HIT = (230, 40, 40)


def load_obj_groups(path):
    """{group: (V, F)} with faces re-indexed per group (polygons fanned)."""
    V, groups, cur = [], {}, 'default'
    for line in open(path):
        if line.startswith('v '):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith(('g ', 'o ')):
            cur = line[2:].strip()
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            groups.setdefault(cur, []).extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    V = np.array(V)
    out = {}
    for g, F in groups.items():
        F = np.array(F)
        used, inv = np.unique(F, return_inverse=True)
        out[g] = (V[used], inv.reshape(-1, 3))
    return out


@functools.lru_cache(maxsize=None)
def frame(env):
    """(centre, scale) of the planner's normalization for env: the shapes3d room, or a
    sofa hall's (flattened, --2d) bbox from its test set's meta.json."""
    if env.startswith('2d'):
        m = json.load(open(os.path.join(ROOT, 'testing_data', '3dshape', 'Usofa2d_' + env, 'meta.json')))
        return np.array(m['env_center']), float(m['env_scale'])
    return CENTER, SCALE


def shape_local(name, scale=SCALE):
    V, F = next(iter(load_obj_groups(os.path.join(D, name + '.obj')).values()))
    return ((V - 0.5 * (V.min(0) + V.max(0))) / scale).astype(np.float32), F.astype(np.int32)


def pose(p):
    """(position, wxyz) of a chart pose (t, rotvec / 2pi)."""
    q = Rotation.from_rotvec(np.asarray(p[3:6], dtype=np.float64) * 2 * np.pi).as_quat()   # xyzw
    return np.asarray(p[:3], dtype=np.float64), np.array([q[3], q[0], q[1], q[2]])


def crossings(env, start, goal):
    if env.startswith('2d'):                 # different legs: sides of the diagonal x = y
        c, s = frame(env)
        a, b = c[:2] + s * np.asarray(start[:2]), c[:2] + s * np.asarray(goal[:2])
        return int((a[0] > a[1]) != (b[0] > b[1]))
    n = 0
    for axis, v in PLANES.get(env, []):
        p = (v - CENTER[axis]) / SCALE
        n += int((start[axis] > p) != (goal[axis] > p))
    return n


def results_markdown():
    p = os.path.join(ROOT, 'results', 'shapes3d', 'summary.json')
    if not os.path.exists(p):
        return 'results/shapes3d/summary.json not found (run dataprocessing/shapes3d_report.py)'
    cells = json.load(open(p))['cells']
    head = '| shape | ' + ' | '.join(ENVS) + ' |\n|---|' + '---|' * len(ENVS) + '\n'
    tab1, tab2 = head, head
    for s in SHAPES:
        r1, r2 = [], []
        for e in ENVS:
            c = cells.get('{}_{}'.format(s, e))
            r1.append('{:.1f}'.format(100 * c['plan']['success_rate']) if c and c['plan'] else '-')
            b = c and c.get('by_partitions_crossed')
            r2.append('{:.1f}'.format(100 * b['1']['success_rate']) if b and '1' in b else '-')
        tab1 += '| {} | {} |\n'.format(s, ' | '.join(r1))
        tab2 += '| {} | {} |\n'.format(s, ' | '.join(r2))
    tab3 = '| hall | success | same leg | around the corner |\n|---|---|---|---|\n'
    for h in HALLS:
        c = cells.get('Usofa2d_' + h)
        b = (c and c.get('by_partitions_crossed')) or {}
        f = lambda k: '{:.1f}'.format(100 * b[k]['success_rate']) if k in b else '-'
        tab3 += '| {} | {} | {} | {} |\n'.format(
            h, '{:.1f}'.format(100 * c['plan']['success_rate']) if c and c['plan'] else '-', f('0'), f('1'))
    return ('**Success % (all test pairs)**\n\n' + tab1 +
            '\n**Success % through one window** (start and goal on opposite sides of a partition)\n\n' + tab2 +
            '\nReference (same recipe): T/env1 82.1, L/env3 89.9, A/env4 87.1 (mean of 3 seeds).' +
            '\n\n**Moving sofa (U, planar), success %**\n\n' + tab3)


class Viewer:
    def __init__(self, server, paths=None, initial_env='env7'):
        self.server = server
        self.paths = paths or {'metric baseline': os.path.join(ROOT, 'results', 'shapes3d', 'paths')}
        self.sc = server.scene
        self.sc.set_up_direction('+y')
        self.shapes = {n: shape_local(n) for n in SHAPES + OLD_SHAPES + SHAPES_2D}
        self.envs = {e: load_obj_groups(os.path.join(D, e + '.obj')) for e in ENVS + HALLS}
        self.cache = {}
        self.case_ids = []
        self.cur = None            # (cell, data) of the shown cell
        self.playing = False
        self.lock = threading.RLock()

        gui = server.gui
        self.view = gui.add_dropdown('View', ['Planning', 'Shapes'], initial_value='Planning')
        with gui.add_folder('Scene'):
            self.model_dd = gui.add_dropdown('Model', list(self.paths), initial_value=list(self.paths)[0])
            self.env_dd = gui.add_dropdown('Environment', ENVS + HALLS, initial_value=initial_env)
            self.env_md = gui.add_markdown('')
            self.shape_dd = gui.add_dropdown('Shape', SHAPES, initial_value='tripod3d')
            self.wall_cb = gui.add_checkbox('Show room shell', True)
        with gui.add_folder('Test pairs'):
            self.filter_dd = gui.add_dropdown('Filter', ['all', 'success', 'failure', 'crossing: success',
                                                         'crossing: failure', 'no crossing'],
                                              initial_value='crossing: success')
            gui.add_markdown('crossing = through a window (rooms) / around the corner (sofa halls)')
            self.case_sl = gui.add_slider('Case #', 0, 1, 1, 0)
            self.prev_bt = gui.add_button('Previous')
            self.next_bt = gui.add_button('Next')
            self.case_md = gui.add_markdown('')
        with gui.add_folder('Motion'):
            self.t_sl = gui.add_slider('Along path', 0.0, 1.0, 0.002, 0.0)
            self.play_bt = gui.add_button('Play / pause', icon=None)
            self.speed_sl = gui.add_slider('Play speed', 0.1, 3.0, 0.1, 1.0)
            self.trail_cb = gui.add_checkbox('Ghost trail', True)
            self.trail_n = gui.add_slider('Ghosts', 3, 30, 1, 10)
        with gui.add_folder('Results', expand_by_default=False):
            gui.add_markdown(results_markdown())

        self.view.on_update(lambda _: self.redraw_all())
        self.model_dd.on_update(lambda _: self.redraw_all())
        self.env_dd.on_update(lambda _: self.env_changed())
        self.shape_dd.on_update(lambda _: self.redraw_all())
        self.wall_cb.on_update(lambda _: self.draw_env())
        self.filter_dd.on_update(lambda _: self.refilter())
        self.case_sl.on_update(lambda _: self.draw_case())
        self.prev_bt.on_click(lambda _: self.step_case(-1))
        self.next_bt.on_click(lambda _: self.step_case(+1))
        self.t_sl.on_update(lambda _: self.draw_mover())
        self.play_bt.on_click(lambda _: self.toggle_play())
        self.trail_cb.on_update(lambda _: self.draw_case())
        self.trail_n.on_update(lambda _: self.draw_case())
        threading.Thread(target=self.player, daemon=True).start()
        self.redraw_all()

    def env_changed(self):
        planar = self.env_dd.value.startswith('2d')
        opts = SHAPES_2D if planar else SHAPES
        if tuple(self.shape_dd.options) != tuple(opts):
            self.shape_dd.options = opts           # resets the value, which redraws via shape_dd
            self.shape_dd.value = opts[0]
        self.redraw_all()

    def mesh(self, name):
        """Shape mesh in the normalized frame of the shown environment."""
        s = frame(self.env_dd.value)[1]
        key = (name, s)
        if key not in self.shapes:
            self.shapes[key] = shape_local(name, s)
        return self.shapes[key]

    def scale(self):
        return frame(self.env_dd.value)[1]

    # ---------- data
    def cell_data(self, cell):
        key = (self.model_dd.value, cell)
        if key not in self.cache:
            p = os.path.join(self.paths[self.model_dd.value], cell + '.npz')
            if not os.path.exists(p):
                return None            # not dumped yet; look again next time
            else:
                z = np.load(p, allow_pickle=True)
                env = cell.split('_')[1]
                d = {k: z[k] for k in z.files}
                d['cross'] = np.array([crossings(env, P[0], P[-1]) for P in d['path']])
                self.cache[key] = d
        return self.cache[key]

    # ---------- drawing
    def clear(self, prefix):
        for name in list(self.handles.keys()) if hasattr(self, 'handles') else []:
            if name.startswith(prefix):
                self._remove(self.handles.pop(name))

    @staticmethod
    def _remove(h):
        try:                       # re-adding a node name already replaced it inside viser
            h.remove()
        except KeyError:
            pass

    def add(self, name, handle):
        if not hasattr(self, 'handles'):
            self.handles = {}
        old = self.handles.get(name)
        if old is not None and old is not handle:
            self._remove(old)
        self.handles[name] = handle
        return handle

    def redraw_all(self):
        with self.lock:
            self.playing = False
            self.clear('/')
            if self.view.value == 'Shapes':
                self.draw_gallery()
                return
            self.sc.set_up_direction('+z' if self.env_dd.value.startswith('2d') else '+y')
            self.draw_env()
            cell = '{}_{}'.format(self.shape_dd.value, self.env_dd.value)
            self.cur = (cell, self.cell_data(cell))
        self.refilter()

    def draw_gallery(self):
        names = SHAPES + SHAPES_2D + OLD_SHAPES
        for k, n in enumerate(names):
            V, F = self.shapes[n]
            x = (k - (len(names) - 1) / 2) * 0.32
            col = COL_NEW if n in SHAPES + SHAPES_2D else COL_OLD
            self.add('/gallery/' + n, self.sc.add_mesh_simple('/gallery/' + n, V, F, color=col, flat_shading=True,
                                                              position=(x, 0.0, 0.0)))
            self.add('/gallery/' + n + '_edges', self.sc.add_mesh_simple(
                '/gallery/' + n + '_edges', V, F, color=(30, 30, 30), wireframe=True, position=(x, 0.0, 0.0)))
            R = np.linalg.norm(V, axis=1).max() * SCALE
            self.add('/gallery/' + n + '_label', self.sc.add_label(
                '/gallery/' + n + '_label', '{} ({} tris, R {:.0f})'.format(n, len(F), R), position=(x, 0.2, 0.0)))
        self.add('/gallery/note', self.sc.add_label(
            '/gallery/note', 'blue: new (bars 10x10, extent 40; Usofa2d 60x20 U, planar) | grey: extruded letters (60x10x60)',
            position=(0.0, -0.22, 0.0)))

    def draw_env(self):
        self.clear('/env')
        if self.view.value != 'Planning':
            return
        env = self.env_dd.value
        self.env_md.content = '**{}**: {}'.format(env, ENV_NOTE[env])
        c, s = frame(env)
        for g, (V, F) in self.envs[env].items():
            Vn = ((V - c) / s).astype(np.float32)
            if g == 'wall (1)':
                if self.wall_cb.value:
                    self.add('/env/wall', self.sc.add_mesh_simple('/env/wall', Vn, F, color=COL_WALL, opacity=0.12,
                                                                  side='double', cast_shadow=False))
                    self.add('/env/wall_edges', self.sc.add_mesh_simple('/env/wall_edges', Vn, F, color=(120, 120, 130),
                                                                        wireframe=True))
            else:
                self.add('/env/' + g, self.sc.add_mesh_simple('/env/' + g, Vn, F, color=COL_OBST, opacity=0.55,
                                                              side='double', flat_shading=True))
                self.add('/env/' + g + '_edges', self.sc.add_mesh_simple('/env/' + g + '_edges', Vn, F,
                                                                         color=(90, 70, 40), wireframe=True))

    def refilter(self):
        if self.view.value != 'Planning':
            return
        cell, d = self.cur
        if d is None:
            self.case_ids = []
            self.case_md.content = 'No paths for {} yet (results/shapes3d/paths/{}.npz).'.format(cell, cell)
            self.clear('/case')
            return
        f = self.filter_dd.value
        env = cell.split('_')[1]
        if env not in PLANES and not env.startswith('2d') and f.startswith('crossing'):
            f = f.split(': ')[1]               # env4 has no partition: fall back to success / failure
        ok = {'all': np.ones(len(d['case']), bool), 'success': d['success'], 'failure': ~d['success'],
              'crossing: success': (d['cross'] > 0) & d['success'],
              'crossing: failure': (d['cross'] > 0) & ~d['success'], 'no crossing': d['cross'] == 0}[f]
        self.case_ids = list(np.nonzero(ok)[0])
        n = max(len(self.case_ids), 1)
        self.case_sl.max = n - 1
        self.case_sl.value = 0
        self.draw_case()

    def step_case(self, k):
        if self.case_ids:
            self.case_sl.value = int(np.clip(self.case_sl.value + k, 0, len(self.case_ids) - 1))

    def draw_case(self):
        with self.lock:
            self.clear('/case')
            cell, d = self.cur if self.cur else (None, None)
            if self.view.value != 'Planning' or d is None:
                return
            if not self.case_ids:
                self.case_md.content = 'No test pair matches this filter ({}).'.format(cell)
                return
            i = self.case_ids[int(self.case_sl.value)]
            P, C = d['path'][i], d['clear'][i]
            V, F = self.mesh(self.shape_dd.value)
            succ = bool(d['success'][i])
            seg = np.stack([P[:-1, :3], P[1:, :3]], 1)
            col = np.where((np.minimum(C[:-1], C[1:]) < 0)[:, None], COL_HIT, (30, 160, 70) if succ else (150, 60, 160))
            self.add('/case/path', self.sc.add_line_segments('/case/path', seg.astype(np.float32),
                                                             np.repeat(col[:, None], 2, 1).astype(np.uint8),
                                                             line_width=3))
            for nm, p, c in (('start', P[0], COL_START), ('goal', P[-1], COL_GOAL)):
                pos, q = pose(p)
                self.add('/case/' + nm, self.sc.add_mesh_simple('/case/' + nm, V, F, color=c, opacity=0.45,
                                                                position=pos, wxyz=q, flat_shading=True))
            if self.trail_cb.value:
                for k in np.linspace(0, len(P) - 1, int(self.trail_n.value) + 2)[1:-1].astype(int):
                    pos, q = pose(P[k])
                    c = COL_HIT if C[k] < 0 else (170, 170, 170)
                    self.add('/case/ghost%d' % k, self.sc.add_mesh_simple('/case/ghost%d' % k, V, F, color=c,
                                                                          opacity=0.18, position=pos, wxyz=q))
            hits = np.nonzero(C < 0)[0]
            if len(hits):
                self.add('/case/hit', self.sc.add_icosphere('/case/hit', radius=0.008, color=COL_HIT,
                                                            position=P[hits[0], :3]))
            self.case_md.content = (
                '**{}** test pair {} ({} of {} shown by this filter)\n\n'
                '- outcome: **{}** (converged {}, collision {})\n'
                '- crossings (partitions / corner): {}\n- waypoints: {}, first collision at waypoint {}\n'
                '- min clearance {:.1f} env units\n- straight start-goal distance {:.1f} env units'.format(
                    cell, int(d['case'][i]), int(self.case_sl.value) + 1, len(self.case_ids),
                    'success' if succ else 'failure', bool(d['converged'][i]), bool(d['collision'][i]),
                    int(d['cross'][i]), len(P), int(hits[0]) if len(hits) else '-', float(C.min()) * self.scale(),
                    float(np.linalg.norm(P[-1, :3] - P[0, :3])) * self.scale()))
            self.case_P, self.case_C = P, C
        self.draw_mover()

    def draw_mover(self):
        if self.view.value != 'Planning' or not getattr(self, 'case_ids', None) or self.cur[1] is None:
            return
        P, C = self.case_P, self.case_C
        x = self.t_sl.value * (len(P) - 1)
        k = int(np.floor(x))
        k1 = min(k + 1, len(P) - 1)
        a = x - k
        r0 = Rotation.from_rotvec(P[k, 3:6] * 2 * np.pi)
        r1 = Rotation.from_rotvec(P[k1, 3:6] * 2 * np.pi)
        rel = (r0.inv() * r1).as_rotvec()
        q = (r0 * Rotation.from_rotvec(a * rel)).as_quat()
        pos = (1 - a) * P[k, :3] + a * P[k1, :3]
        V, F = self.mesh(self.shape_dd.value)
        hit = C[k] < 0 or C[k1] < 0
        name = '/case/mover'
        if name in getattr(self, 'handles', {}) and getattr(self, '_mover_hit', None) == hit:
            h = self.handles[name]
            h.position = pos
            h.wxyz = np.array([q[3], q[0], q[1], q[2]])
        else:
            self._mover_hit = hit
            self.add(name, self.sc.add_mesh_simple(name, V, F, color=COL_HIT if hit else COL_MOVE, flat_shading=True,
                                                   position=pos, wxyz=np.array([q[3], q[0], q[1], q[2]])))

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing and self.t_sl.value >= 1.0:
            self.t_sl.value = 0.0

    def player(self):
        while True:
            time.sleep(0.04)
            if self.playing and getattr(self, 'case_ids', None):
                n = max(len(self.case_P) - 1, 1)
                t = self.t_sl.value + self.speed_sl.value * 0.5 / n
                if t >= 1.0:
                    t, self.playing = 1.0, False
                self.t_sl.value = t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8097)
    ap.add_argument('--paths', nargs='*', default=None, help='LABEL=DIR path sets (default: results/shapes3d/paths)')
    ap.add_argument('--initial-env', default='env7')
    args = ap.parse_args()
    paths = None
    if args.paths:
        paths = {}
        for kv in args.paths:
            k, v = kv.split('=', 1)
            paths[k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    server = viser.ViserServer(host=args.host, port=args.port)
    Viewer(server, paths, args.initial_env)
    print('viser at http://{}:{}'.format(args.host, args.port), flush=True)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
