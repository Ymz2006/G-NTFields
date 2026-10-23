"""Interactive viser viewer for constrained movement: a camera pyramid that only turns about the bird it follows.

The bird (orange) flies along its RRT path (black line). The camera is the apex of the view pyramid, whose base
rectangle stays centred on the bird, so the camera can only rotate about the bird. For one test case of a test set
the viewer shows
  * the scene boxes and the room shell, the bird's path (the case's span thicker), the start / goal camera poses
    (green / pink pyramids)
  * a plan: the bird at t and the pyramid at R, moved together; the camera track (the apex) coloured by clearance
    (red = the pyramid touches the scene: the camera hits something or its view of the bird is blocked), and a
    second plan to compare (dimmer)
  * the camera view: what the camera at the apex sees (render.py's numpy z-buffer, no browser needed), the
    pyramid's base rectangle dashed, or your own viewport moved to the camera ("Look through camera")
  * the clearance and the bird's progress t / L along the plans
Plans: metric NTFields seeds 0 / 1 (forward MPPI as evaluated), the test set's RRT reference and the straight
motion. All plans run on the same clock (one waypoint per step; RRT and straight motion one per 0.015 of M, the
MPPI step); a plan that has converged or stopped stays at its last pose.

Reads results/constrained_movement/viser/<run>_<variant>.npz (export_viser_cmove.py, torch side). Coordinates are
the planner's normalized ones (env units / env_scale, y up). Needs numpy and viser only.

    python models/constrained_movement/scripts/cmove_viser.py --port 8103
"""
import argparse
import contextlib
import glob
import importlib.util
import math
import os
import threading
import time

import numpy as np
import viser
import viser.uplot as uplot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, 'results', 'constrained_movement')


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(HERE), fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RD = _load('_cm_render', 'render.py')          # numpy only: the camera-view rasterizer of the report's figures

RUNS = ['forest', 'city', 'env1', 'env4', 'corridor2']
SCENE_NAME = {'corridor2': 'corridor'}
VARIANT_NOTE = {'full': 'whole path (t 0 -> L)', 'seg': 'sub-paths (t1 - t0 >= 0.25)'}
MODELS = ('ntf0', 'ntf1', 'rrt', 'line')
COL = {'ntf0': (31, 111, 180), 'ntf1': (124, 77, 186), 'rrt': (0, 150, 170), 'line': (110, 110, 110)}
COL_DIM = {'ntf0': (140, 180, 215), 'ntf1': (185, 160, 220), 'rrt': (120, 200, 210), 'line': (175, 175, 175)}
COL_BIRD = tuple(int(round(255 * v)) for v in RD.SHAPE_COLOR)
COL_WALL = (200, 200, 210)
COL_PATH = (25, 25, 25)
COL_START = (60, 180, 75)
COL_GOAL = (225, 55, 140)
COL_HIT = (230, 40, 40)
FILTERS = ['all', 'NTFields succeeds', 'NTFields collides', 'NTFields stalls (no collision)',
           'straight motion fails, NTFields succeeds', 'NTFields seeds disagree']
# overview (elevation, azimuth in degrees, distance) per scene, as make_figures_cmove.py
OBS = {'forest': (32, -35, 1.6), 'city': (38, -30, 1.6), 'corridor': (78, -10, 2.0), 'env1': (35, -150, 1.5),
       'env4': (35, -150, 1.5)}


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)
    E = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]
    return np.stack([c[[i for i, _ in E]], c[[j for _, j in E]]], 1)


def pyramid_mesh(W, H, D):
    """Body frame of geometry.Pyramid: base W x H at z = 0 around the bird, apex (the camera) at (0, 0, D)."""
    a, b = W / 2, H / 2
    V = np.array([[-a, -b, 0], [a, -b, 0], [a, b, 0], [-a, b, 0], [0, 0, D]], np.float32)
    F = np.array([(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], np.int32)
    E = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)])
    return V, F, np.stack([V[E[:, 0]], V[E[:, 1]]], 1)


def set_label(s):
    """forest_full -> 'forest, whole path (t 0 -> L)'."""
    run, v = s.rsplit('_', 1)
    return '{}, {}'.format(SCENE_NAME.get(run, run), VARIANT_NOTE.get(v, v))


def results_markdown(files):
    rows = []
    for run in RUNS:
        cells = []
        for v in VARIANT_NOTE:
            f = files.get('{}_{}'.format(run, v))
            if f is None:
                cells.append('-')
                continue
            z = np.load(f)
            r = lambda m: '{:.1f}'.format(100 * z[m + '_success'].mean()) if m + '_success' in z.files else '-'  # noqa: E731
            cells.append('{} / {} / {}'.format(r('line'), r('ntf0'), r('ntf1')))
        rows.append('| {} | {} |'.format(SCENE_NAME.get(run, run), ' | '.join(cells)))
    return ('**Success %, forward mode**, 1000 cases per set: straight motion / NTFields seed 0 / seed 1 (corridor: '
            'one seed)\n\n| scene | whole path | sub-paths |\n|---|---|---|\n' + '\n'.join(rows))


class Viewer:
    def __init__(self, server, data_dir, initial, lens):
        self.server = server
        self.sc = server.scene
        self.sc.set_up_direction('+y')
        self.files = {os.path.basename(f)[:-4]: f for f in sorted(glob.glob(os.path.join(data_dir, '*.npz')))}
        if not self.files:
            raise SystemExit('no exports in {} (run export_viser_cmove.py)'.format(data_dir))
        order = lambda s: (RUNS.index(s.rsplit('_', 1)[0]) if s.rsplit('_', 1)[0] in RUNS else 99,   # noqa: E731
                           list(VARIANT_NOTE).index(s.rsplit('_', 1)[1]))
        self.sets = sorted(self.files, key=order)
        self.labels = {set_label(s): s for s in self.sets}
        self.cache = {}
        self.handles = {}
        self.style = {}
        self.case_ids = []
        self.d = None
        self.models = []
        self.name_of = {}
        self.plans = {}
        self.case_key = None
        self.playing = False
        self.lock = threading.RLock()
        self.mute = 0
        self.view_dirty = True
        self.cam_pose = None          # (bird centre, R, touching, clearance, t) of the shown camera
        self.n_rendered = 0

        gui = server.gui
        cb = lambda f: (lambda _: None if self.mute else f())   # noqa: E731
        with gui.add_folder('Test case'):
            lab = set_label(initial) if initial in self.files else set_label(self.sets[0])
            self.set_dd = gui.add_dropdown('Test set', list(self.labels), initial_value=lab)
            self.filter_dd = gui.add_dropdown('Filter', FILTERS, initial_value='all')
            self.case_sl = gui.add_slider('Case #', 0, 1, 1, 0)
            self.prev_bt = gui.add_button('Previous')
            self.next_bt = gui.add_button('Next')
            self.short_md = gui.add_markdown('')
        with gui.add_folder('Motion'):
            self.t_sl = gui.add_slider('Step', 0, 1, 1, 0)
            self.play_bt = gui.add_button('Play / pause')
            self.speed_sl = gui.add_slider('Play speed (steps / s)', 5, 120, 5, 30)
            self.time_md = gui.add_markdown('')
        with gui.add_folder('Camera view'):
            self.cam_img = gui.add_image(np.full((110, 240, 3), 235, np.uint8), label='what the camera sees',
                                         format='jpeg', jpeg_quality=90)
            self.cam_md = gui.add_markdown('')
            self.lens_sl = gui.add_slider('Lens (x the pyramid\'s view)', 1.0, 3.0, 0.1, lens)
            self.res_dd = gui.add_dropdown('Panel width (px)', ['320', '480', '640'], initial_value='480')
            self.look_cb = gui.add_checkbox('Look through camera', False)
            self.cam_cb = gui.add_checkbox('Camera view panel', True)
            gui.add_markdown('The camera at the apex looks at the bird. The dashed rectangle is the pyramid\'s base '
                             'around the bird (lens 1 = exactly the pyramid); a plan keeps the whole pyramid clear, '
                             'so nothing may cut into it (red: something does). "Look through camera" moves your own '
                             'viewport there.')
        with gui.add_folder('Plans'):
            self.main_dd = gui.add_dropdown('Camera of', ['NTFields, seed 0'], initial_value='NTFields, seed 0')
            self.cmp_dd = gui.add_dropdown('Compare with', ['none'], initial_value='none')
            self.ends_cb = gui.add_checkbox('Start / goal poses', True)
            self.track_cb = gui.add_checkbox('Camera track', True)
            self.ghost_cb = gui.add_checkbox('Sight-line trail', False)
            self.ghost_n = gui.add_slider('Trail lines', 3, 40, 1, 12)
            gui.add_markdown('Pyramid colours: NTFields seed 0 blue, seed 1 purple, RRT reference teal, straight '
                             'motion grey; red while it touches the scene. Start / goal poses green / pink. The '
                             'bird keeps a fixed heading.')
        with gui.add_folder('Details'):
            self.set_md = gui.add_markdown('')
            self.case_md = gui.add_markdown('')
        with gui.add_folder('Scene', expand_by_default=False):
            self.wall_cb = gui.add_checkbox('Show room shell', True)
            self.opaque_cb = gui.add_checkbox('Opaque obstacles', False)
        self.plot_folder = gui.add_folder('Clearance along the plans', expand_by_default=False)
        self.plots = []
        with gui.add_folder('Results', expand_by_default=False):
            gui.add_markdown(results_markdown(self.files))

        self.set_dd.on_update(cb(self.load_set))
        self.wall_cb.on_update(cb(self.draw_env))
        self.opaque_cb.on_update(cb(self.draw_env))
        self.filter_dd.on_update(cb(self.refilter))
        self.case_sl.on_update(cb(self.draw_case))
        self.prev_bt.on_click(lambda _: self.step_case(-1))
        self.next_bt.on_click(lambda _: self.step_case(+1))
        self.main_dd.on_update(cb(self.main_changed))
        for h in (self.cmp_dd, self.ends_cb, self.track_cb, self.ghost_cb, self.ghost_n):
            h.on_update(cb(self.draw_case))
        self.t_sl.on_update(cb(self.draw_movers))
        self.play_bt.on_click(lambda _: self.toggle_play())
        for h in (self.cam_cb, self.lens_sl, self.res_dd):
            h.on_update(cb(self.mark_view))
        self.look_cb.on_update(cb(self.look_changed))
        threading.Thread(target=self.player, daemon=True).start()
        threading.Thread(target=self.renderer, daemon=True).start()
        self.load_set()

    # ---------- node bookkeeping (as path_viser.py)
    @contextlib.contextmanager
    def quiet(self):
        """Set GUI values without their callbacks redrawing."""
        self.mute += 1
        try:
            yield
        finally:
            self.mute -= 1

    def clear(self, prefix):
        for name in list(self.handles):
            if name.startswith(prefix):
                self._remove(self.handles.pop(name))
                self.style.pop(name, None)

    def drop(self, name):
        h = self.handles.pop(name, None)
        if h is not None:
            self._remove(h)
        self.style.pop(name, None)

    @staticmethod
    def _remove(h):
        try:
            h.remove()
        except KeyError:
            pass

    def add(self, name, handle):
        # adding a node under an existing name already replaced it inside viser: removing the old handle would remove
        # the new node
        self.handles[name] = handle
        return handle

    # ---------- data
    def set_name(self):
        return self.labels[self.set_dd.value]

    def load_set(self):
        s = self.set_name()
        if s not in self.cache:
            z = np.load(self.files[s], allow_pickle=False)
            d = {k: z[k] for k in z.files}
            for m in MODELS:
                if m + '_flat' not in d:
                    continue
                ends = np.cumsum(d[m + '_len'])
                d[m + '_start'] = ends - d[m + '_len']
                d[m + '_row'] = {int(c): i for i, c in enumerate(d[m + '_case'])}
            self.cache[s] = d
        with self.lock:
            self.playing = False
            self.d = d = self.cache[s]
            self.models = [m for m in MODELS if m + '_flat' in d]
            self.name_of = {m: str(d[m + '_label']) for m in self.models}
            names = [self.name_of[m] for m in self.models]
            with self.quiet():
                main = self.main_dd.value if self.main_dd.value in names else names[0]
                cmp_ = self.cmp_dd.value if self.cmp_dd.value in names else 'none'
                self.main_dd.options = names
                self.main_dd.value = main
                self.cmp_dd.options = ['none'] + names
                self.cmp_dd.value = cmp_
            W, H, D, L = (float(d[k]) for k in ('pyr_W', 'pyr_H', 'pyr_D', 'L'))
            rate = lambda m: '{:.1f} %'.format(100 * d[m + '_success'].mean()) if m in self.models else '-'  # noqa: E731
            self.set_md.content = (
                '**{}**: {}, {} cases. Path length L = {:.2f}; pyramid base {:.3f} x {:.3f}, camera {:.3f} from the '
                'bird (48 x 22 x 50 units, 51 x 25 deg view). Success: NTFields {} (seed 0) / {} (seed 1), straight '
                'motion {}.'.format(SCENE_NAME.get(str(d['run']), str(d['run'])), VARIANT_NOTE[s.rsplit('_', 1)[1]],
                                    len(d['case']), L, W, H, D, rate('ntf0'), rate('ntf1'), rate('line')))
            self.pyr = pyramid_mesh(W, H, D)
            bird = np.concatenate([RD.box_tris(lo, hi) for lo, hi in zip(d['shape_lo'], d['shape_hi'])])
            self.bird_tris = bird
            self.bird_mesh = (bird.reshape(-1, 3).astype(np.float32),
                              np.arange(3 * len(bird), dtype=np.int32).reshape(-1, 3))
            # the camera view's static triangles: room (inner faces), obstacles; colours as render.scene_mesh
            tris, cols = [], []
            room = RD.box_tris(d['room_lo'], d['room_hi'], inward=True)
            for k in range(12):
                tris.append(room[k])
                cols.append(RD.ROOM['floor' if k // 2 == 2 else ('ceiling' if k // 2 == 3 else 'wall')])
            for lo, hi, c in zip(d['box_lo'], d['box_hi'], d['box_rgb']):
                T = RD.box_tris(lo, hi)
                tris += list(T)
                cols += [tuple(c)] * len(T)
            self.static = (np.asarray(tris), np.asarray(cols, float))
            self.case_key = None
            self.clear('/')
            self.draw_env()
        self.refilter()
        if not self.look_cb.value:
            for client in self.server.get_clients().values():
                self.observe(client)

    def outcome(self, m, c):
        d = self.d
        i = d[m + '_row'].get(c) if m + '_row' in d else None
        if i is None:
            return None
        return bool(d[m + '_success'][i]), bool(d[m + '_collision'][i]), bool(d[m + '_converged'][i])

    def main_model(self):
        return [m for m in self.models if self.name_of[m] == self.main_dd.value][0]

    def ntf_key(self):
        m = self.main_model()
        return m if m in ('ntf0', 'ntf1') else 'ntf0'

    def main_changed(self):
        if self.filter_dd.value == 'all':
            self.draw_case()
        else:
            self.refilter()

    def refilter(self):
        d, f = self.d, self.filter_dd.value
        k = self.ntf_key()
        other = 'ntf1' if k == 'ntf0' else 'ntf0'
        ids = []
        for c in d['case']:
            c = int(c)
            n, o = self.outcome(k, c), self.outcome(other, c)
            if f == 'all':
                ok = True
            elif n is None:
                ok = False
            elif f == 'NTFields succeeds':
                ok = n[0]
            elif f == 'NTFields collides':
                ok = n[1]
            elif f == 'NTFields stalls (no collision)':
                ok = not n[2] and not n[1]
            elif f == 'straight motion fails, NTFields succeeds':
                ok = n[0] and not bool(d['straight_ok'][c])
            else:
                ok = o is not None and o[0] != n[0]
            if ok:
                ids.append(c)
        self.case_ids = ids
        with self.quiet():
            self.case_sl.max = max(len(ids) - 1, 1)
            self.case_sl.value = 0
        self.draw_case()

    def step_case(self, k):
        if self.case_ids:
            self.case_sl.value = int(np.clip(self.case_sl.value + k, 0, len(self.case_ids) - 1))

    def plan(self, m, c):
        d = self.d
        i = d[m + '_row'][c]
        a, n = d[m + '_start'][i], d[m + '_len'][i]
        F = d[m + '_flat'][a:a + n].astype(np.float64)
        return {'t': F[:, 0], 'p': F[:, 1:4], 'q': F[:, 4:8], 'R': np.stack([quat_matrix(q) for q in F[:, 4:8]]),
                'c': F[:, 8], 'seg': d[m + '_seg'][a:a + n].astype(np.float64)}

    def shown_models(self):
        main = self.main_model()
        out = [main]
        other = [m for m in self.models if self.name_of[m] == self.cmp_dd.value]
        if other and other[0] != main:
            out.append(other[0])
        return out

    # ---------- drawing
    def draw_env(self):
        self.clear('/env')
        self.mark_view()
        d = self.d
        for k, (lo, hi, c) in enumerate(zip(d['box_lo'], d['box_hi'], d['box_rgb'])):
            col = tuple(int(round(255 * v)) for v in c)
            self.add('/env/box%d' % k, self.sc.add_box('/env/box%d' % k, color=col, dimensions=hi - lo,
                                                      position=(lo + hi) / 2, side='double',
                                                      opacity=None if self.opaque_cb.value else 0.5))
            self.add('/env/box%d_e' % k, self.sc.add_line_segments('/env/box%d_e' % k, box_edges(lo, hi),
                                                                  tuple(int(0.6 * v) for v in col), line_width=1.5))
        if self.wall_cb.value:
            lo, hi = d['room_lo'], d['room_hi']
            self.add('/env/room', self.sc.add_line_segments('/env/room', box_edges(lo, hi), (120, 120, 130),
                                                            line_width=1.5))
            floor = np.array([hi[0] - lo[0], 0.002, hi[2] - lo[2]])
            self.add('/env/floor', self.sc.add_box('/env/floor', color=COL_WALL, dimensions=floor, opacity=0.35,
                                                   position=((lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2)))
        P = d['path_pts'].astype(np.float32)
        self.add('/env/path', self.sc.add_line_segments('/env/path', np.stack([P[:-1], P[1:]], 1), (95, 95, 95),
                                                        line_width=1.5))

    def mover(self, name, V, F, p, q, color, opacity):
        """A mesh moved in place while its style stays the same."""
        h = self.handles.get(name)
        if h is not None and self.style.get(name) == (color, opacity):
            h.position, h.wxyz = p, q
        else:
            self.style[name] = (color, opacity)
            self.add(name, self.sc.add_mesh_simple(name, V, F, color=color, opacity=opacity, position=p, wxyz=q,
                                                   flat_shading=True, side='double'))

    def mover_lines(self, name, S, p, q, color, width):
        h = self.handles.get(name)
        if h is not None and self.style.get(name) == (color, width):
            h.position, h.wxyz = p, q
        else:
            self.style[name] = (color, width)
            self.add(name, self.sc.add_line_segments(name, S, color, line_width=width, position=p, wxyz=q))

    def draw_case(self):
        with self.lock:
            if not self.case_ids:
                self.clear('/case')
                self.case_md.content = self.short_md.content = 'No test case matches this filter.'
                self.plans = {}
                self.cam_pose = None
                self.draw_plot()
                return
            d = self.d
            c = self.case_ids[min(int(self.case_sl.value), len(self.case_ids) - 1)]
            new = (self.set_name(), c) != self.case_key
            self.case_key = (self.set_name(), c)
            self.clear('/case')
            self.plans = {m: self.plan(m, c) for m in self.shown_models() if c in d[m + '_row']}
            main = self.main_model()
            V, F, _ = self.pyr
            Dp = float(d['pyr_D'])
            # the case's span of the path
            pts = d['path_pts']
            ell = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            span = (ell >= d['l0'][c] - 1e-9) & (ell <= d['l1'][c] + 1e-9)
            S = pts[span].astype(np.float32)
            if len(S) > 1:
                self.add('/case/span', self.sc.add_line_segments('/case/span', np.stack([S[:-1], S[1:]], 1), COL_PATH,
                                                                 line_width=3.5))
            for nm, l, rv, col in (('start', d['l0'][c], d['rv0'][c], COL_START),
                                   ('goal', d['l1'][c], d['rv1'][c], COL_GOAL)) if self.ends_cb.value else ():
                p = np.array([np.interp(l, ell, pts[:, k]) for k in range(3)])
                th = np.linalg.norm(rv)
                q = np.concatenate([[np.cos(th / 2)], rv / max(th, 1e-12) * np.sin(th / 2)])
                self.add('/case/' + nm, self.sc.add_mesh_simple('/case/' + nm, V, F, color=col, opacity=0.35,
                                                                position=p, wxyz=q, side='double'))
                self.add('/case/' + nm + '_e', self.sc.add_line_segments('/case/' + nm + '_e', self.pyr[2], col,
                                                                         line_width=1.5, position=p, wxyz=q))
            for m, P in self.plans.items():
                cam = P['p'] + Dp * P['R'][:, :, 2]
                if self.track_cb.value and len(cam) > 1:
                    seg = np.stack([cam[:-1], cam[1:]], 1).astype(np.float32)
                    base = COL[m] if m == main else COL_DIM[m]
                    col = np.where((P['seg'][:-1] < 0)[:, None], COL_HIT, base).astype(np.uint8)
                    self.add('/case/%s_track' % m, self.sc.add_line_segments(
                        '/case/%s_track' % m, seg, np.repeat(col[:, None], 2, 1), line_width=3 if m == main else 2))
                if self.ghost_cb.value:
                    ks = np.linspace(0, len(cam) - 1, int(self.ghost_n.value)).astype(int)
                    seg = np.stack([P['p'][ks], cam[ks]], 1).astype(np.float32)
                    col = np.where((P['c'][ks] < 0)[:, None], COL_HIT, COL_DIM[m]).astype(np.uint8)
                    self.add('/case/%s_trail' % m, self.sc.add_line_segments(
                        '/case/%s_trail' % m, seg, np.repeat(col[:, None], 2, 1), line_width=1.5))
            n = max(len(P['t']) for P in self.plans.values())
            with self.quiet():
                self.t_sl.max = max(n - 1, 1)
                if new or self.t_sl.value > n - 1:
                    self.t_sl.value = 0
            self.case_md.content = self.case_text(c)
            self.short_md.content = self.short_text(c)
            self.draw_plot()
        self.draw_movers()

    def case_text(self, c):
        d = self.d
        L = float(d['L'])
        k = min(int(self.case_sl.value), len(self.case_ids) - 1)
        out = ('**case {}** ({} of {} in this filter)\n\n- bird t {:.2f} -> {:.2f} (t / L {:.2f} -> {:.2f}); the '
               'camera turns {:.0f} deg between the end poses; start / goal clearance {:.3f} / {:.3f}\n'.format(
                   c, k + 1, len(self.case_ids), d['l0'][c], d['l1'][c], d['l0'][c] / L, d['l1'][c] / L,
                   d['angle_deg'][c], d['start_clear'][c], d['goal_clear'][c]))
        for m in self.models:
            o = self.outcome(m, c)
            if o is None:
                continue
            i = d[m + '_row'][c]
            a, n = d[m + '_start'][i], d[m + '_len'][i]
            fb = d[m + '_first_block'][i]
            verdict = 'success' if o[0] else ('blocked' if o[1] else 'stalled (never reaches the goal)')
            out += '- {}: **{}**, {} steps, smallest clearance {:+.3f}{}\n'.format(
                self.name_of[m], verdict, n - 1, float(d[m + '_seg'][a:a + n].min()),
                ', first blocked at t / L {:.2f}'.format(fb / L) if np.isfinite(fb) else '')
        return out

    def short_text(self, c):
        k = min(int(self.case_sl.value), len(self.case_ids) - 1)
        short = {'ntf0': 'seed 0', 'ntf1': 'seed 1', 'rrt': 'RRT', 'line': 'straight'}
        out = []
        for m in self.models:
            o = self.outcome(m, c)
            if o is not None:
                out.append('{} {}'.format(short[m], 'ok' if o[0] else ('blocked' if o[1] else 'stalled')))
        return '**case {}** ({} of {}): {}'.format(c, k + 1, len(self.case_ids), ', '.join(out))

    def draw_plot(self):
        for h in self.plots:
            h.remove()
        self.plots = []
        if not self.plans:
            return
        n = max(len(P['t']) for P in self.plans.values())
        x = np.arange(n, dtype=np.float64)
        pad = lambda y: np.concatenate([y, np.full(n - len(y), y[-1])]).astype(np.float64)   # noqa: E731
        rgb = lambda c: 'rgb({},{},{})'.format(*c)   # noqa: E731
        main = self.main_model()
        ys, s1, ts, s2 = [], [uplot.Series(label='step')], [], [uplot.Series(label='step')]
        for m, P in self.plans.items():
            w = 2 if m == main else 1.5
            ys.append(pad(P['seg']))
            s1.append(uplot.Series(label=self.name_of[m], stroke=rgb(COL[m]), width=w))
            ts.append(pad(P['t']) / float(self.d['L']))
            s2.append(uplot.Series(label=self.name_of[m], stroke=rgb(COL[m]), width=w))
        ys.append(np.zeros(n))
        s1.append(uplot.Series(label='contact', stroke=rgb(COL_HIT), width=1))
        with self.plot_folder:
            self.plots.append(self.server.gui.add_uplot(data=(x, *ys), series=tuple(s1), aspect=1.6,
                                                        title='clearance of the pyramid (below 0: touching)'))
            self.plots.append(self.server.gui.add_uplot(data=(x, *ts), series=tuple(s2), aspect=1.6,
                                                        title='bird progress t / L'))

    def pose_at(self, P, t):
        """Pose of a plan at (fractional) step t, held at its last pose."""
        n = len(P['t'])
        t = min(t, n - 1)
        k = int(np.floor(t))
        k1 = min(k + 1, n - 1)
        a = t - k
        p = (1 - a) * P['p'][k] + a * P['p'][k1]
        q = slerp(P['q'][k], P['q'][k1], a)
        hit = min(P['seg'][k], P['c'][k1]) < 0 if a > 0 else P['c'][k] < 0
        return p, q, hit, k

    def draw_movers(self):
        with self.lock:
            if not self.plans:
                return
            t = float(self.t_sl.value)
            main = self.main_model()
            V, F, E = self.pyr
            L = float(self.d['L'])
            lines = []
            for m, P in self.plans.items():
                p, q, hit, k = self.pose_at(P, t)
                is_main = m == main
                col = COL_HIT if hit else (COL[m] if is_main else COL_DIM[m])
                if is_main and self.look_cb.value:          # seen from its own apex, the faces only hide the view
                    self.drop('/case/%s_pyr' % m)
                else:
                    self.mover('/case/%s_pyr' % m, V, F, p, q, col, 0.45 if is_main else 0.25)
                self.mover_lines('/case/%s_pyr_e' % m, E, p, q, col, 2.5 if is_main else 1.5)
                self.mover('/case/%s_bird' % m, *self.bird_mesh, p, (1.0, 0.0, 0.0, 0.0), COL_BIRD,
                           1.0 if is_main else 0.4)
                if is_main:
                    self.cam_pose = (p, quat_matrix(q), hit, P['c'][k], P['t'][k])
                note = ' **touching**' if hit else (' (touches on the way to the next step)' if P['seg'][k] < 0 else '')
                lines.append('{}: step {} of {}, t / L {:.2f}, clearance {:+.3f}{}{}'.format(
                    self.name_of[m], k, len(P['t']) - 1, P['t'][k] / L, P['c'][k], note,
                    ' (done)' if t >= len(P['t']) - 1 else ''))
            self.time_md.content = '\n\n'.join(lines)
        self.mark_view()

    # ---------- camera view
    def render_view(self, width):
        p, R, hit, clear, t = self.cam_pose
        d = self.d
        tris, cols = self.static
        W, H, D = float(d['pyr_W']), float(d['pyr_H']), float(d['pyr_D'])
        z = float(self.lens_sl.value)
        height = max(8, int(round(width * H / W)))
        T = np.concatenate([tris, self.bird_tris + p])
        C = np.concatenate([cols, np.tile(RD.SHAPE_COLOR, (len(self.bird_tris), 1))])
        apex = p + R @ np.array([0.0, 0.0, D])
        img, _ = RD.rasterize(T, C, apex, R, z * W / 2 / D, z * H / 2 / D, width, height)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # the pyramid's base rectangle: the centre 1 / z of the image, dashed (red while the pyramid touches)
        col = np.array(COL_HIT if hit else (255, 255, 255), np.uint8)
        x0, x1 = int(round(width * (1 - 1 / z) / 2)), int(round(width * (1 + 1 / z) / 2)) - 1
        y0, y1 = int(round(height * (1 - 1 / z) / 2)), int(round(height * (1 + 1 / z) / 2)) - 1
        xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
        xs, ys = xs[(xs - x0) // 5 % 2 == 0], ys[(ys - y0) // 5 % 2 == 0]
        for o in (0, 1):
            img[min(y0 + o, height - 1), xs] = col
            img[max(y1 - o, 0), xs] = col
            img[ys, min(x0 + o, width - 1)] = col
            img[ys, max(x1 - o, 0)] = col
        if hit:
            img[:3], img[-3:], img[:, :3], img[:, -3:] = col, col, col, col
        return img, 't / L {:.2f}, clearance {:+.3f}{}'.format(t / float(d['L']), clear,
                                                               ': **the pyramid touches the scene**' if hit else '')

    def mark_view(self):
        self.view_dirty = True
        if self.look_cb.value and self.cam_pose is not None:
            p, R, _, _, _ = self.cam_pose
            D = float(self.d['pyr_D'])
            fov = 2 * math.atan(float(self.lens_sl.value) * float(self.d['pyr_H']) / 2 / D)
            for client in self.server.get_clients().values():
                with client.atomic():
                    client.camera.fov = fov
                    client.camera.up_direction = R[:, 1]
                    client.camera.position = p + D * R[:, 2]
                    client.camera.look_at = p

    def look_changed(self):
        if not self.look_cb.value:
            for client in self.server.get_clients().values():
                self.observe(client)
        self.draw_movers()
        self.mark_view()

    def observe(self, client):
        """Move a client's viewport to the scene's overview."""
        pts = self.d['path_pts']
        target = 0.5 * (pts.min(0) + pts.max(0))
        el, az, dist = OBS.get(str(self.d['env']), (30, -40, 1.7))
        el, az = math.radians(el), math.radians(az)
        eye = target + dist * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
        with client.atomic():
            client.camera.up_direction = (0.0, 1.0, 0.0)
            client.camera.fov = math.radians(50)
            client.camera.position = eye
            client.camera.look_at = target

    def renderer(self):
        """Render the camera view into the GUI panel whenever the shown pose changed."""
        while True:
            time.sleep(0.03)
            if not (self.view_dirty and self.cam_cb.value and self.cam_pose is not None):
                continue
            self.view_dirty = False
            try:
                img, note = self.render_view(int(self.res_dd.value))
                self.cam_img.image = img
                self.cam_md.content = note
                self.n_rendered += 1
            except Exception as e:          # a set switched mid-render: the next pass draws it
                print('render failed:', repr(e), flush=True)
                self.view_dirty = True

    # ---------- playback
    def toggle_play(self):
        self.playing = not self.playing
        if self.playing and self.t_sl.value >= self.t_sl.max:
            self.t_sl.value = 0

    def player(self):
        last = time.time()
        acc = 0.0
        while True:
            time.sleep(0.05)
            now = time.time()
            if self.playing and self.plans:
                acc += (now - last) * self.speed_sl.value
                k = int(acc)
                if k:
                    acc -= k
                    t = self.t_sl.value + k
                    if t >= self.t_sl.max:
                        t, self.playing = self.t_sl.max, False
                    self.t_sl.value = t
            else:
                acc = 0.0
            last = now


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8103)
    ap.add_argument('--data', default=os.path.join(RES, 'viser'))
    ap.add_argument('--initial', default='forest_full', help='test set shown first (<run>_<variant>)')
    ap.add_argument('--filter', default='all', choices=FILTERS)
    ap.add_argument('--case', type=int, default=0, help='index within the filter')
    ap.add_argument('--time', type=int, default=0, help='step shown first')
    ap.add_argument('--model', default='ntf0', choices=MODELS, help='whose camera')
    ap.add_argument('--compare', default='none', choices=('none',) + MODELS)
    ap.add_argument('--lens', type=float, default=2.0, help='camera view: x the pyramid\'s field of view')
    ap.add_argument('--look', action='store_true', help='start with "Look through camera" on')
    ap.add_argument('--play', action='store_true', help='start playing')
    args = ap.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width='large')
    v = Viewer(server, args.data, args.initial, args.lens)
    if args.model in v.models:
        v.main_dd.value = v.name_of[args.model]
    if args.compare in v.models:
        v.cmp_dd.value = v.name_of[args.compare]
    v.filter_dd.value = args.filter
    v.case_sl.value = min(args.case, max(len(v.case_ids) - 1, 0))
    v.t_sl.value = min(args.time, v.t_sl.max)
    v.look_cb.value = args.look
    v.playing = args.play

    @server.on_client_connect
    def _(client):
        if v.look_cb.value:
            v.mark_view()
        else:
            v.observe(client)
    print('viser at http://{}:{}'.format(args.host, server.get_port()), flush=True)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
