"""Interactive viser viewer for the path-constrained (camera follow) plans: homotopy head vs metric NTFields.

The object moves along its path (black line); the camera is the far end of the planned prism, which is glued to the
object at its other end. For one test case of a test set the viewer shows
  * the scene boxes and the room shell, the object path, the start / goal poses (green / blue ghosts)
  * each model's plan: the object (sphere) and the camera's line of sight, drawn as the view pyramid (apex at the
    camera, base the w x w square around the object, so it lies inside the planned prism: the prism clear => the
    pyramid clear) or as the planned prism; the camera track coloured by clearance (red = the prism touches the scene)
  * a camera view: the scene rendered from the camera, looking at the object (GUI panel, or the whole viewport with
    "Look through camera"); the square in the middle is the pyramid's base, the region the plan keeps clear
  * the clearance of both plans along the motion
Both plans run on the same clock (one MPPI step per waypoint): a plan that has converged or stopped stays at its
last pose.

Reads results/path_constrained/viser/<set>.npz (export_viser_path.py, torch side) and
results/path_constrained/summary/summary.json (aggregate_path.py). Coordinates are the planner's normalized ones
(env units / env_scale, y up). Needs numpy and viser only.

    python models/path_constrained/scripts/path_viser.py --port 8102
"""
import argparse
import glob
import json
import os
import threading
import time

import numpy as np
import viser
import viser.uplot as uplot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(ROOT, 'results', 'path_constrained')
VARIANT_NOTE = {'full': 'whole path (l 0 -> L)', 'seg': 'sub-paths (l1 - l0 >= 0.25)',
                'tight': 'whole path, both ends 0.005-0.02 from the scene',
                'bigrot': 'whole path, turn >= 150 deg', 'hard': 'whole path, tight ends and turn >= 135 deg'}
MODELS = ('hom', 'ntf')
NAME = {'hom': 'homotopy (dual)', 'ntf': 'NTFields'}
COL = {'hom': (31, 111, 180), 'ntf': (217, 130, 43)}
COL_DIM = {'hom': (140, 180, 215), 'ntf': (235, 190, 140)}
COL_OBST = (205, 170, 125)
COL_WALL = (200, 200, 210)
COL_PATH = (25, 25, 25)
COL_START = (40, 170, 90)
COL_GOAL = (60, 110, 230)
COL_OK = (40, 160, 70)
COL_HIT = (230, 40, 40)
FILTERS = ['all', 'both succeed', 'homotopy only', 'NTFields only', 'both fail', 'NTFields collides',
           'NTFields stalls']


def quat_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quat(R):
    """(3, 3) -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


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


def pyramid_mesh(w, h):
    """View pyramid in the prism's body frame: base the w x w square at the object (z = 0), apex at the camera."""
    s = w / 2
    V = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [0, 0, h]], np.float32)
    F = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]], np.int32)
    return V, F


def prism_mesh(w, h):
    s = w / 2
    V = np.array([[x, y, z] for z in (0, h) for x, y in ((-s, -s), (s, -s), (s, s), (-s, s))], np.float32)
    F = []
    for a, b, c, d in [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]:
        F += [[a, b, c], [a, c, d]]
    return V, np.array(F, np.int32)


def box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], np.float32)
    E = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count('1') == 1]
    return np.stack([c[[i for i, _ in E]], c[[j for _, j in E]]], 1)


def set_label(s):
    """prism30_env1_a_hard -> 'env1 path a, hard (prism 0.3)'."""
    head, var = s.rsplit('_', 1)
    p, env, path = head.split('_')
    return '{} path {}, {}{}'.format(env, path, var, ' (prism 0.3)' if p == 'prism30' else '')


def results_markdown():
    p = os.path.join(RES, 'summary', 'summary.json')
    if not os.path.exists(p):
        return 'results/path_constrained/summary/summary.json not found (run aggregate_path.py)'
    d = json.load(open(p))
    f = lambda k: '{:.1f}'.format(100 * d[k]['success_rate_se3']) if k in d else '-'   # noqa: E731
    out = '**SE(3) success %, forward mode**, homotopy dual / base / NTFields (mean over seeds)\n\n'
    out += '| cell | ' + ' | '.join(VARIANT_NOTE) + ' |\n|---|' + '---|' * len(VARIANT_NOTE) + '\n'
    for cell in ['prism_env1_a', 'prism_env3_a', 'prism_env4_a', 'prism_env1_b', 'prism_env3_b', 'prism_env4_b',
                 'prism30_env1_a', 'prism30_env3_a', 'prism30_env4_a']:
        row = []
        for v in VARIANT_NOTE:
            ks = ['{}|{}|{}_forward'.format(cell, v, m) for m in ('hom_dual', 'hom_base', 'ntf_base')]
            row.append(' / '.join(f(k) for k in ks) if any(k in d for k in ks) else '-')
        out += '| {} | {} |\n'.format(cell.replace('prism', ''), ' | '.join(row))
    return out


class Viewer:
    def __init__(self, server, data_dir, initial, dump=None):
        self.server = server
        self.sc = server.scene
        self.sc.set_up_direction('+y')
        self.files = {os.path.basename(f)[:-4]: f for f in sorted(glob.glob(os.path.join(data_dir, '*.npz')))}
        if not self.files:
            raise SystemExit('no exports in {} (run export_viser_path.py)'.format(data_dir))
        order = lambda s: (s.startswith('prism30'), s.split('_')[2], s.split('_')[1],   # noqa: E731
                           list(VARIANT_NOTE).index(s.rsplit('_', 1)[1]))
        self.sets = sorted(self.files, key=order)
        self.labels = {set_label(s): s for s in self.sets}
        self.cache = {}
        self.handles = {}
        self.style = {}
        self.case_ids = []
        self.d = None
        self.plans = {}
        self.playing = False
        self.lock = threading.RLock()
        self.view_dirty = True
        self.dump = dump
        self.n_rendered = 0
        self.dead = set()          # clients that did not answer a render request
        self.cam_pose = None

        gui = server.gui
        with gui.add_folder('Scene'):
            lab = set_label(initial) if initial in self.files else set_label(self.sets[0])
            self.set_dd = gui.add_dropdown('Test set', list(self.labels), initial_value=lab)
            self.set_md = gui.add_markdown('')
            self.wall_cb = gui.add_checkbox('Show room shell', True)
            self.opaque_cb = gui.add_checkbox('Opaque obstacles', False)
        with gui.add_folder('Test case'):
            self.filter_dd = gui.add_dropdown('Filter', FILTERS, initial_value='homotopy only')
            self.case_sl = gui.add_slider('Case #', 0, 1, 1, 0)
            self.prev_bt = gui.add_button('Previous')
            self.next_bt = gui.add_button('Next')
            self.case_md = gui.add_markdown('')
        with gui.add_folder('Models'):
            self.main_dd = gui.add_dropdown('Camera of', [NAME[m] for m in MODELS], initial_value=NAME['hom'])
            self.other_cb = gui.add_checkbox('Show the other model', True)
            self.body_dd = gui.add_dropdown('Line of sight', ['view pyramid', 'planned prism', 'both'],
                                            initial_value='view pyramid')
            self.ends_cb = gui.add_checkbox('Start / goal poses', True)
            self.track_cb = gui.add_checkbox('Camera track', True)
            self.ghost_cb = gui.add_checkbox('Sight-line trail', True)
            self.ghost_n = gui.add_slider('Trail lines', 3, 40, 1, 14)
        with gui.add_folder('Motion'):
            self.t_sl = gui.add_slider('Time (MPPI steps)', 0, 1, 1, 0)
            self.play_bt = gui.add_button('Play / pause')
            self.speed_sl = gui.add_slider('Play speed (steps / s)', 5, 120, 5, 40)
            self.time_md = gui.add_markdown('')
        with gui.add_folder('Camera view'):
            self.cam_cb = gui.add_checkbox('Camera view panel', True)
            self.look_cb = gui.add_checkbox('Look through camera', False)
            self.fov_sl = gui.add_slider('Field of view (deg)', 10, 90, 1, 40)
            self.roll_cb = gui.add_checkbox('Roll with the prism', False)
            self.res_dd = gui.add_dropdown('Panel size', ['320 x 240', '480 x 360', '640 x 480'],
                                           initial_value='320 x 240')
            self.cam_img = gui.add_image(np.full((240, 320, 3), 235, np.uint8), label='camera of the shown model')
            gui.add_markdown('The square in the middle is the base of the view pyramid (side w around the object); '
                             'the plan keeps it unobstructed (red: the prism touches the scene). "Look through '
                             'camera" moves your own viewport there; "Opaque obstacles" (Scene) shows occlusion.')
        self.plot_folder = gui.add_folder('Clearance', expand_by_default=False)
        self.plot = None
        with gui.add_folder('Results', expand_by_default=False):
            gui.add_markdown(results_markdown())

        self.set_dd.on_update(lambda _: self.load_set())
        self.wall_cb.on_update(lambda _: self.draw_env())
        self.opaque_cb.on_update(lambda _: self.draw_env())
        self.filter_dd.on_update(lambda _: self.refilter())
        self.case_sl.on_update(lambda _: self.draw_case())
        self.prev_bt.on_click(lambda _: self.step_case(-1))
        self.next_bt.on_click(lambda _: self.step_case(+1))
        for h in (self.main_dd, self.other_cb, self.body_dd, self.ends_cb, self.track_cb, self.ghost_cb, self.ghost_n):
            h.on_update(lambda _: self.draw_case())
        self.t_sl.on_update(lambda _: self.draw_movers())
        self.play_bt.on_click(lambda _: self.toggle_play())
        for h in (self.cam_cb, self.look_cb, self.fov_sl, self.roll_cb, self.res_dd):
            h.on_update(lambda _: self.draw_movers())
        threading.Thread(target=self.player, daemon=True).start()
        threading.Thread(target=self.renderer, daemon=True).start()
        self.load_set()

    # ---------- node bookkeeping (same scheme as dataprocessing/shapes3d_viser.py)
    def clear(self, prefix):
        for name in list(self.handles):
            if name.startswith(prefix):
                self._remove(self.handles.pop(name))

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
            self.d = self.cache[s]
            d = self.d
            self.models = [m for m in MODELS if m + '_flat' in d]
            v = s.rsplit('_', 1)[1]
            self.set_md.content = ('**{}**: {}. Prism {:.2f} x {:.2f}, path length L = {:.2f} (normalized units), '
                                   '{} cases.'.format(s, VARIANT_NOTE[v], float(d['prism_width']),
                                                      float(d['prism_length']), float(d['L']), len(d['case'])))
            self.mesh = {'view pyramid': pyramid_mesh(float(d['prism_width']), float(d['prism_length'])),
                         'planned prism': prism_mesh(float(d['prism_width']), float(d['prism_length']))}
            self.clear('/')
            self.draw_env()
        self.refilter()

    def outcome(self, m, c):
        d = self.d
        i = d[m + '_row'].get(c)
        if i is None:
            return None
        return bool(d[m + '_success'][i]), bool(d[m + '_collision'][i]), bool(d[m + '_converged'][i])

    def refilter(self):
        d, f = self.d, self.filter_dd.value
        ids = []
        for c in d['case']:
            c = int(c)
            h, n = self.outcome('hom', c), self.outcome('ntf', c)
            if h is None or n is None:
                ok = f == 'all'
            else:
                ok = {'all': True, 'both succeed': h[0] and n[0], 'homotopy only': h[0] and not n[0],
                      'NTFields only': n[0] and not h[0], 'both fail': not (h[0] or n[0]),
                      'NTFields collides': n[1], 'NTFields stalls': not n[2]}[f]
            if ok:
                ids.append(c)
        self.case_ids = ids
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
        return {'l': F[:, 0], 'p': F[:, 1:4], 'q': F[:, 4:8], 'c': F[:, 8], 'seg': d[m + '_seg'][a:a + n]}

    def main_model(self):
        return [m for m in MODELS if NAME[m] == self.main_dd.value][0]

    # ---------- drawing
    def draw_env(self):
        self.clear('/env')
        self.mark_view()
        d = self.d
        for k, (lo, hi) in enumerate(zip(d['box_lo'], d['box_hi'])):
            self.add('/env/box%d' % k, self.sc.add_box('/env/box%d' % k, color=COL_OBST, dimensions=hi - lo,
                                                      position=(lo + hi) / 2, side='double',
                                                      opacity=None if self.opaque_cb.value else 0.55))
            self.add('/env/box%d_e' % k, self.sc.add_line_segments('/env/box%d_e' % k, box_edges(lo, hi),
                                                                  (90, 70, 40), line_width=1.5))
        if self.wall_cb.value:
            lo, hi = d['room_lo'], d['room_hi']
            self.add('/env/room', self.sc.add_line_segments('/env/room', box_edges(lo, hi), (120, 120, 130),
                                                            line_width=1.5))
            floor = np.array([hi[0] - lo[0], 0.002, hi[2] - lo[2]])
            self.add('/env/floor', self.sc.add_box('/env/floor', color=COL_WALL, dimensions=floor, opacity=0.35,
                                                   position=((lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2)))
        P = d['path_pts'].astype(np.float32)
        self.add('/env/path', self.sc.add_line_segments('/env/path', np.stack([P[:-1], P[1:]], 1), COL_PATH,
                                                        line_width=3))

    def body(self, name, p, q, color, opacity):
        """The line of sight at pose (p, q); moved in place while its style and colour stay the same."""
        kinds = list(self.mesh) if self.body_dd.value == 'both' else [self.body_dd.value]
        for k, (V, F) in self.mesh.items():
            nm = '{}_{}'.format(name, k.split()[1])
            if k not in kinds:
                self.clear(nm)
                continue
            op = opacity if k == 'view pyramid' or len(kinds) == 1 else 0.25 * opacity
            h = self.handles.get(nm)
            if h is not None and self.style.get(nm) == (color, op):
                h.position, h.wxyz = p, q
            else:
                self.style[nm] = (color, op)
                self.add(nm, self.sc.add_mesh_simple(nm, V, F, color=color, opacity=op, position=p, wxyz=q,
                                                     flat_shading=True))

    def draw_case(self):
        with self.lock:
            self.clear('/case')
            if not self.case_ids:
                self.case_md.content = 'No test case matches this filter.'
                self.plans = {}
                return
            d = self.d
            c = self.case_ids[min(int(self.case_sl.value), len(self.case_ids) - 1)]
            self.case = c
            self.plans = {m: self.plan(m, c) for m in self.models if c in d[m + '_row']}
            main = self.main_model()
            shown = [m for m in self.plans if m == main or self.other_cb.value]
            # start / goal ghosts of the line of sight (the main model's body style)
            w, h = float(d['prism_width']), float(d['prism_length'])
            pts = d['path_pts']
            ell = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            ends = (('start', d['l0'][c], d['rv0'][c], COL_START), ('goal', d['l1'][c], d['rv1'][c], COL_GOAL))
            for nm, l, rv, col in ends if self.ends_cb.value else ():
                p = np.array([np.interp(l, ell, pts[:, k]) for k in range(3)])
                th = np.linalg.norm(rv)
                q = np.concatenate([[np.cos(th / 2)], rv / max(th, 1e-12) * np.sin(th / 2)])
                V, F = self.mesh['view pyramid']
                self.add('/case/' + nm, self.sc.add_mesh_simple('/case/' + nm, V, F, color=col, opacity=0.5,
                                                                position=p, wxyz=q))
                self.add('/case/' + nm + '_obj', self.sc.add_icosphere('/case/' + nm + '_obj', radius=0.004,
                                                                       color=col, position=p))
            for m in shown:
                P = self.plans[m]
                R = np.stack([quat_matrix(q) for q in P['q']])
                cam = P['p'] + h * R[:, :, 2]
                if self.track_cb.value:
                    seg = np.stack([cam[:-1], cam[1:]], 1).astype(np.float32)
                    bad = P['seg'][:-1] < 0
                    base = COL[m] if m == main else COL_DIM[m]
                    col = np.where(bad[:, None], COL_HIT, base).astype(np.uint8)
                    self.add('/case/%s_track' % m, self.sc.add_line_segments(
                        '/case/%s_track' % m, seg, np.repeat(col[:, None], 2, 1), line_width=3 if m == main else 2))
                if self.ghost_cb.value:
                    ks = np.linspace(0, len(cam) - 1, int(self.ghost_n.value)).astype(int)
                    seg = np.stack([P['p'][ks], cam[ks]], 1).astype(np.float32)
                    col = np.where((P['c'][ks] < 0)[:, None], COL_HIT, COL_DIM[m]).astype(np.uint8)
                    self.add('/case/%s_trail' % m, self.sc.add_line_segments(
                        '/case/%s_trail' % m, seg, np.repeat(col[:, None], 2, 1), line_width=1.5))
            n = max(len(P['l']) for P in self.plans.values())
            self.t_sl.max = max(n - 1, 1)
            self.t_sl.value = 0
            self.case_md.content = self.case_text(c)
            self.draw_plot()
        self.draw_movers()

    def case_text(self, c):
        d = self.d
        th = np.degrees(np.linalg.norm(d['rv0'][c])), np.degrees(np.linalg.norm(d['rv1'][c]))
        out = ('**case {}** ({} of {} shown by this filter)\n\n- object l {:.2f} -> {:.2f}, start / goal clearance '
               '{:.3f} / {:.3f}\n'.format(c, min(int(self.case_sl.value), len(self.case_ids) - 1) + 1, len(self.case_ids), d['l0'][c], d['l1'][c],
                                          d['start_clear'][c], d['goal_clear'][c]))
        for m, P in self.plans.items():
            s, col, conv = self.outcome(m, c)
            verdict = 'success' if s else ('line of sight blocked' if col else 'stalled (no convergence)')
            bad = np.nonzero(P['seg'] < 0)[0]
            out += '- {}: **{}**, {} steps, min clearance {:.3f}{}\n'.format(
                NAME[m], verdict, len(P['l']) - 1, float(P['seg'].min()),
                ', first blocked at step {} (l = {:.2f})'.format(bad[0], P['l'][bad[0]]) if len(bad) else '')
        return out

    def draw_plot(self):
        if self.plot is not None:
            self.plot.remove()
            self.plot = None
        if not self.plans:
            return
        n = max(len(P['l']) for P in self.plans.values())
        x = np.arange(n, dtype=np.float64)
        ys, series = [], [uplot.Series(label='step')]
        for m, P in self.plans.items():
            y = np.concatenate([P['seg'], np.full(n - len(P['seg']), P['seg'][-1])]).astype(np.float64)
            ys.append(y)
            series.append(uplot.Series(label=NAME[m], stroke='rgb({},{},{})'.format(*COL[m]), width=2))
            yl = np.concatenate([P['l'], np.full(n - len(P['l']), P['l'][-1])]) / float(self.d['L'])
            ys.append(0.1 * yl)
            series.append(uplot.Series(label='{} l / L x 0.1'.format(NAME[m]), stroke='rgb({},{},{})'.format(
                *COL_DIM[m]), width=1, dash=[4, 4]))
        ys.append(np.zeros(n))
        series.append(uplot.Series(label='contact', stroke='rgb(230,40,40)', width=1))
        with self.plot_folder:
            self.plot = self.server.gui.add_uplot(data=(x, *ys), series=tuple(series), aspect=1.4,
                                                  title='clearance along the plan (red line: contact)')

    def pose_at(self, P, t):
        """Pose of a plan at (fractional) step t, held at its last pose."""
        n = len(P['l'])
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
            lines = []
            for m, P in self.plans.items():
                if m != main and not self.other_cb.value:
                    self.clear('/case/%s_mover' % m)
                    self.clear('/case/%s_obj' % m)
                    continue
                p, q, hit, k = self.pose_at(P, t)
                col = COL_HIT if hit else (COL[m] if m == main else COL_DIM[m])
                self.body('/case/%s_mover' % m, p, q, col, 0.85 if m == main else 0.6)
                o = self.handles.get('/case/%s_obj' % m)
                if o is not None:
                    o.position = p
                else:
                    self.add('/case/%s_obj' % m, self.sc.add_icosphere('/case/%s_obj' % m, radius=0.007,
                                                                       color=(255, 196, 0), position=p))
                if m == main:
                    self.cam_pose = (p, q)
                    cam, wxyz, _, _ = self.camera_frame()
                    W, H = (int(v) for v in self.res_dd.value.split(' x '))
                    fr = self.handles.get('/case/frustum')
                    if fr is not None and self.style.get('/case/frustum') == (self.fov_sl.value, W / H, hit):
                        fr.position, fr.wxyz = cam, wxyz
                    else:
                        self.style['/case/frustum'] = (self.fov_sl.value, W / H, hit)
                        self.add('/case/frustum', self.sc.add_camera_frustum(
                            '/case/frustum', fov=np.radians(self.fov_sl.value), aspect=W / H, scale=0.025,
                            color=COL_HIT if hit else (20, 20, 20), position=cam, wxyz=wxyz, line_width=1.5))
                    self.add('/case/view_base', self.sc.add_line_segments(
                        '/case/view_base', self.base_square(p, q), (20, 20, 20) if not hit else COL_HIT,
                        line_width=2))
                lines.append('{}: step {} of {}, l = {:.2f}, clearance {:.3f}{}'.format(
                    NAME[m], k, len(P['l']) - 1, P['l'][k], P['c'][k], ' (done)' if t >= len(P['l']) - 1 else ''))
            self.time_md.content = '\n\n'.join(lines)
        self.mark_view()

    def base_square(self, p, q):
        w = float(self.d['prism_width'])
        R = quat_matrix(q)
        s = w / 2
        c = np.array([p + R @ np.array([x, y, 0.0]) for x, y in ((-s, -s), (s, -s), (s, s), (-s, s))])
        return np.stack([c, np.roll(c, -1, 0)], 1).astype(np.float32)

    # ---------- camera view
    def camera_frame(self):
        """(position, wxyz, up) of the camera at the apex looking at the object, viser's OpenCV camera frame."""
        p, q = self.cam_pose
        R = quat_matrix(q)
        h = float(self.d['prism_length'])
        cam = p + h * R[:, 2]
        f = -R[:, 2]
        up = R[:, 1] if self.roll_cb.value else np.array([0.0, 1.0, 0.0])
        if abs(np.dot(up, f)) > 0.98:            # looking straight up or down: fall back to the prism's own axis
            up = R[:, 1]
        up = up - np.dot(up, f) * f
        up /= np.linalg.norm(up)
        right = np.cross(f, up)
        Rc = np.stack([right, -up, f], 1)        # columns: x right, y down, z forward
        return cam, matrix_quat(Rc), up, p

    def mark_view(self):
        self.view_dirty = True
        if self.look_cb.value and self.cam_pose is not None:
            cam, _, up, target = self.camera_frame()
            for client in self.server.get_clients().values():
                with client.atomic():
                    client.camera.fov = np.radians(self.fov_sl.value)
                    client.camera.up_direction = up
                    client.camera.position = cam
                    client.camera.look_at = target

    def renderer(self):
        """Render the camera view into the GUI panel (through any connected client) when the pose changed."""
        while True:
            time.sleep(0.08)
            if not (self.view_dirty and self.cam_cb.value and self.cam_pose is not None):
                continue
            clients = [c for k, c in sorted(self.server.get_clients().items()) if k not in self.dead]
            if not clients:
                continue
            self.view_dirty = False
            W, H = (int(v) for v in self.res_dd.value.split(' x '))
            cam, wxyz, _, _ = self.camera_frame()
            tag = '{}_case{}_{}_t{:03d}.png'.format(self.set_name(), self.case, self.main_model(), int(self.t_sl.value))
            client, out = clients[-1], []
            # ClientHandle.get_render waits forever if the browser leaves before answering: run it aside
            th = threading.Thread(target=lambda: out.append(client.get_render(
                H, W, wxyz=wxyz, position=cam, fov=np.radians(self.fov_sl.value))), daemon=True)
            th.start()
            th.join(30.0)
            if not out:
                self.dead.add(client.client_id)
                self.view_dirty = True
                continue
            try:
                img = out[0]
                self.cam_img.image = img
                if self.dump:
                    import imageio.v3 as iio
                    os.makedirs(self.dump, exist_ok=True)
                    iio.imwrite(os.path.join(self.dump, tag), img)
                self.n_rendered += 1
            except Exception as e:          # client left mid-render
                print('render failed:', e, flush=True)

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
    ap.add_argument('--port', type=int, default=8102)
    ap.add_argument('--data', default=os.path.join(RES, 'viser'))
    ap.add_argument('--initial', default='prism_env4_a_full', help='test set shown first')
    ap.add_argument('--filter', default='homotopy only', choices=FILTERS)
    ap.add_argument('--case', type=int, default=0, help='index within the filter')
    ap.add_argument('--time', type=int, default=0, help='MPPI step shown first')
    ap.add_argument('--model', default='hom', choices=MODELS, help='whose camera')
    ap.add_argument('--look', action='store_true', help='start with "Look through camera" on')
    ap.add_argument('--opaque', action='store_true', help='start with opaque obstacles')
    ap.add_argument('--dump', default=None, help='also write every rendered camera view as a png into this directory')
    ap.add_argument('--strip', type=int, default=0,
                    help='with --dump: when a browser connects, render N evenly spaced frames of each model\'s camera')
    args = ap.parse_args()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width='large')
    v = Viewer(server, args.data, args.initial, args.dump)
    v.filter_dd.value = args.filter
    v.main_dd.value = NAME[args.model]
    v.case_sl.value = min(args.case, len(v.case_ids) - 1)
    v.t_sl.value = min(args.time, v.t_sl.max)
    v.look_cb.value = args.look
    v.opaque_cb.value = args.opaque

    def strip():
        v.other_cb.value = v.track_cb.value = v.ghost_cb.value = v.ends_cb.value = False   # an uncluttered view
        for m in list(v.plans):
            v.main_dd.value = NAME[m]
            P = v.plans[m]
            ts = np.linspace(0, len(P['l']) - 1, args.strip).round().astype(int)
            if P['c'].min() < 0:                 # replace the nearest frame by the most blocked step
                k = int(np.argmin(P['c']))
                ts[np.argmin(np.abs(ts - k))] = k
            for t in sorted(set(ts.tolist())):
                k = v.n_rendered
                v.t_sl.value = int(t)
                v.draw_movers()
                while v.view_dirty or v.n_rendered == k:
                    time.sleep(0.1)
        print('strip done', flush=True)

    @server.on_client_connect
    def _(client):
        if args.strip and args.dump:
            threading.Thread(target=strip, daemon=True).start()
        if not v.look_cb.value:
            lo, hi = v.d['room_lo'], v.d['room_hi']
            c = (lo + hi) / 2
            client.camera.position = c + np.array([0.55, 0.6, 0.85]) * (hi - lo)
            client.camera.look_at = c
        v.mark_view()
    print('viser at http://{}:{}'.format(args.host, server.get_port()), flush=True)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    main()
