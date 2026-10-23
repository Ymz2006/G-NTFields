"""Static figures of the constrained-movement test (one NTFields run per scene, forward mode, whole path).

Per scene, from the evaluation's saved plans (plan_eval_cmove.py --save-paths):

    <scene>_overview.png   the scene, the bird's RRT path, and one successful plan: the camera pyramid at six points
                           of the plan with the bird, the camera track; beside it the camera's room along the path
                           (fraction of random rotations free) and the clearance along that plan and a failed one
    <scene>_views.png      what the camera sees along the successful plan (top) and along a failed plan (bottom),
                           rendered with a 2.5x wider lens; the dashed rectangle is the pyramid's base, the view the
                           plan must keep clear

The showcase success is the case with the largest rotation among the successes whose straight motion collides; the
failure is the first collision (else the first stall).

    python models/constrained_movement/scripts/make_figures_cmove.py --seed 0
"""
import argparse
import json
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.constrained_movement import geometry as G  # noqa: E402
from models.constrained_movement import render as RD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--scenes', default='forest,city,corridor2,env1,env4', help='runs <scene>_ntf_s<seed>')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--variant', default='full')
ap.add_argument('--mode', default='forward')
ap.add_argument('--plan', default='plan', help='results/constrained_movement/<plan>/<run>/...')
ap.add_argument('--out', default='results/constrained_movement/figures')
ap.add_argument('--report', default='models/constrained_movement/report/img')
ap.add_argument('--zoom', type=float, default=2.5)
ap.add_argument('--device', default='cuda:0')
args = ap.parse_args()

dev = args.device if torch.cuda.is_available() else 'cpu'
TWO_PI = 2 * math.pi
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
GOOD, CRIT, WARN, CAM, BIRD = '#0ca30c', '#d03b3b', '#fab219', '#2a78d6', '#eb6834'
SET_OF = {'corridor2': 'corridor'}   # run -> scene of its test sets (corridor2: the corridor with longer pairs)
OBS = {'forest': (32, -35, 1.6), 'city': (38, -30, 1.6), 'corridor': (78, -10, 2.0), 'env1': (35, -150, 1.5),
       'env4': (35, -150, 1.5)}   # observer elevation, azimuth (deg), distance
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': '#52514e', 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})


def poses(path, Y):
    """Planner states (K, 4) (t, rotvec / 2 pi) -> centres (K, 3) and rotations (K, 3, 3) (numpy)."""
    Yt = torch.as_tensor(Y, dtype=torch.float64)
    return path.pos(Yt[:, 0]).numpy(), G.so3_exp(TWO_PI * Yt[:, 1:]).numpy()


def plan_of(z, case, iters3):
    """Saved waypoints of a case up to its SE(3) convergence (all of them if it never converged)."""
    i = int(np.nonzero(z['case'] == case)[0][0])
    P = z['path'][i]
    P = P[np.isfinite(P[:, 0])]
    k3 = int(z['iters3'][i])
    return P[:k3 + 2] if k3 >= 0 else P


def clearance(scene, pyr, C, R):
    return G.pyramid_clearance(scene, pyr, torch.as_tensor(C, dtype=torch.float32, device=dev),
                               torch.as_tensor(R, dtype=torch.float32, device=dev)).cpu().numpy()


def free_profile(scene, pyr, path, stations=120, rots=2048, seed=0):
    rng = np.random.default_rng(seed)
    ell = np.linspace(0, path.L, stations)
    C = path.pos(torch.as_tensor(ell, dtype=torch.float64)).numpy()
    R = G.so3_exp(torch.as_tensor(G.random_rotvecs(rots, rng)))
    out = []
    for c in C:
        cl = clearance(scene, pyr, np.repeat(c[None], rots, 0), R.numpy())
        out.append((cl > 0).mean())
    return ell / path.L, np.asarray(out)


def frame_indices(P, k=6, extra=None):
    """k waypoint indices at equal fractions of the plan's progress along the path (plus ``extra``)."""
    t = P[:, 0]
    lo, hi = t[0], t[-1]
    idx = [int(np.argmax(t >= lo + f * (hi - lo) - 1e-9)) for f in np.linspace(0, 1, k)]
    idx[-1] = len(P) - 1
    if extra is not None and extra not in idx:
        j = int(np.argmin([abs(i - extra) for i in idx[1:-1]])) + 1
        idx[j] = extra
    return sorted(set(idx))


def observer(name, scene, shape, pyr, path, P, width=960, height=700):
    """Outside view of the scene (near room walls culled) with the bird at the start, middle and end of the plan
    and the camera pyramid (translucent) at five points; returns the image, its z-buffer and the camera."""
    C, R = poses(path, P)
    idx = frame_indices(P, 5)
    birds = [C[i] for i in (idx[0], idx[len(idx) // 2], idx[-1])]
    tris, cols, room = RD.scene_mesh(scene, shape, birds, room_flags=True)
    target = 0.5 * (path.pts.min(0) + path.pts.max(0))
    el, az, dist = (math.radians(v) if k < 2 else v for k, v in enumerate(OBS.get(name, (30, -40, 1.7))))
    eye = target + dist * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    cam_R = RD.look_at(eye, target)
    tan_h = 0.40
    tan_w = tan_h * width / height
    cam = (eye, cam_R, tan_w, tan_h, width, height)
    img, zb = RD.rasterize(tris, cols, eye, cam_R, tan_w, tan_h, width, height, cull=room)
    rgb = matplotlib.colors.to_rgb(CAM)
    for i in idx:
        T = RD.pyramid_world(pyr, C[i], R[i])[pyr.tris]
        RD.rasterize(T, [rgb] * len(T), eye, cam_R, tan_w, tan_h, width, height, img=img, zbuf=zb, alpha=0.2)
    return img, zb, cam, C, R, idx


def draw_line(ax, P3, cam, zb, **kw):
    """Plot a world polyline over an observer image, hiding the parts behind the opaque scene."""
    xy, d = RD.project(P3, *cam)
    vis = RD.visible(xy, d, zb)
    runs = np.split(np.arange(len(vis)), np.nonzero(np.diff(vis.astype(int)))[0] + 1)
    first = True
    for r in runs:
        if len(r) > 1 and vis[r[0]]:
            ax.plot(xy[r, 0], xy[r, 1], label=kw.pop('label', None) if first else None, **kw)
            first = False


def densify(P, k=4):
    u = np.linspace(0, 1, k, endpoint=False)
    out = (P[:-1, None, :] + (P[1:] - P[:-1])[:, None, :] * u[None, :, None]).reshape(-1, P.shape[1])
    return np.concatenate([out, P[-1:]])


def overview(name, scene, shape, pyr, path, extra, P, Pf, fail_kind, prof, out):
    fig = plt.figure(figsize=(13.5, 5.6))
    ax = fig.add_axes([0.0, 0.02, 0.6, 0.9])
    img, zb, cam, C, R, idx = observer(name, scene, shape, pyr, path, P)
    ax.imshow(img, interpolation='bilinear')
    rrt = extra.get('rrt')
    if rrt is not None:
        draw_line(ax, densify(rrt, 40), cam, zb, color=MUTED, lw=1.0, ls=(0, (3, 2)), label='bird: RRT waypoints')
    draw_line(ax, path.pts, cam, zb, color=INK, lw=2.2, label='bird: path (t)')
    apex = C + R[:, :, 2] * pyr.D
    draw_line(ax, densify(apex, 4), cam, zb, color=CAM, lw=1.6, label='camera track')
    for i in idx:
        V = RD.pyramid_world(pyr, C[i], R[i])
        for a, b in pyr.edges:
            draw_line(ax, V[a] + np.linspace(0, 1, 40)[:, None] * (V[b] - V[a]), cam, zb, color=CAM, lw=0.9)
    xy, d = RD.project(path.pts[[0, -1]], *cam)
    ax.scatter(xy[:, 0], xy[:, 1], s=[36, 36], c=[GOOD, INK], marker='o', zorder=5)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    ax.legend(loc='upper left', frameon=True, framealpha=0.85, fontsize=8, edgecolor='none')
    fig.text(0.3, 0.975, '{}: the bird\'s path (L = {:.2f}) and the camera pyramid along a successful NTFields plan'.format(
        name, path.L), fontsize=10, color=INK, ha='center', va='top')
    # camera room along the path
    a1 = fig.add_axes([0.66, 0.6, 0.31, 0.3])
    a1.fill_between(prof[0], prof[1], color='#86b6ef', alpha=0.5, lw=0)
    a1.plot(prof[0], prof[1], color=CAM, lw=1.5)
    a1.set_ylim(0, 1)
    a1.set_xlim(0, 1)
    a1.set_ylabel('free rotations')
    a1.set_title('camera room along the path (fraction of random rotations free)', fontsize=9, color=INK, loc='left')
    a1.grid(axis='y', color=GRID, lw=0.6)
    # clearance along the plans
    a2 = fig.add_axes([0.66, 0.12, 0.31, 0.3])
    cs = clearance(scene, pyr, C, R)
    a2.plot(P[:, 0] / path.L, cs, color=GOOD, lw=1.5, label='successful plan')
    if Pf is not None:
        Cf, Rf = poses(path, Pf)
        cf = clearance(scene, pyr, Cf, Rf)
        a2.plot(Pf[:, 0] / path.L, cf, color=CRIT if fail_kind == 'collision' else WARN, lw=1.5,
                label='failed plan ({})'.format(fail_kind))
    a2.axhline(0, color=INK, lw=0.8)
    a2.set_xlim(0, 1)
    a2.set_xlabel('progress of the bird along its path, t / L')
    a2.set_ylabel('pyramid clearance')
    a2.set_title('clearance of the camera pyramid along the plans', fontsize=9, color=INK, loc='left')
    a2.grid(axis='y', color=GRID, lw=0.6)
    a2.legend(frameon=False, fontsize=8, loc='lower right')
    fig.savefig(out, dpi=110)
    plt.close(fig)


def views(name, scene, shape, pyr, path, rows, out):
    """rows: [(label, waypoints, colour)]; 4 camera views per row with the pyramid's base outlined."""
    ncol = 4
    fig, axs = plt.subplots(len(rows), ncol, figsize=(3.1 * ncol, 1.72 * len(rows) + 0.3), squeeze=False)
    for r, (label, P, col) in enumerate(rows):
        C, R = poses(path, P)
        cs = clearance(scene, pyr, C, R)
        extra = int(np.argmin(cs)) if cs.min() < 0 else None
        idx = frame_indices(P, ncol, extra)
        while len(idx) < ncol:
            idx.append(idx[-1])
        for c, i in enumerate(idx[:ncol]):
            ax = axs[r, c]
            img = RD.camera_view(scene, shape, pyr, C[i], R[i], width=420, zoom=args.zoom)
            h, w = img.shape[:2]
            ax.imshow(img, interpolation='bilinear')
            fw, fh = w / args.zoom, h / args.zoom
            ax.add_patch(plt.Rectangle(((w - fw) / 2, (h - fh) / 2), fw, fh, fill=False,
                                       ec=CRIT if cs[i] < 0 else 'white', lw=1.4, ls=(0, (4, 2))))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title('t/L {:.2f}   clearance {:+.3f}'.format(P[i, 0] / path.L, cs[i]), fontsize=8.5,
                         color=CRIT if cs[i] < 0 else '#52514e')
        axs[r, 0].text(-0.06, 0.5, label, transform=axs[r, 0].transAxes, rotation=90, ha='right', va='center',
                       fontsize=8, color=col)
    fig.suptitle('{}: what the camera sees ({:g}x wider lens; dashed = the pyramid\'s base around the bird)'.format(
        name, args.zoom),
                 fontsize=9.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def concept(out):
    """What the camera sees at one pose of the forest path (the report's model section)."""
    scene = G.Scene(os.path.join(REPO, 'datasets/3dshape/obj/forest.obj'))
    shape = G.Shape(G.BIRD, scene.env_scale)
    pyr = G.default_pyramid(scene.env_scale)
    path = G.Path.load(os.path.join(REPO, 'datasets/3dshape/paths/cm_forest_a.npz'))
    c = path.pos(torch.tensor([0.55 * path.L], dtype=torch.float64))[0].numpy()
    z = np.array([-1.0, 0.45, -0.25])
    z /= np.linalg.norm(z)                                        # from the bird to the camera: behind, above
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z], 1)
    fig = plt.figure(figsize=(6.4, 2.94))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    img = RD.camera_view(scene, shape, pyr, c, R, width=640, zoom=args.zoom)
    h, w = img.shape[:2]
    ax.imshow(img, interpolation='bilinear')
    ax.add_patch(plt.Rectangle(((w - w / args.zoom) / 2, (h - h / args.zoom) / 2), w / args.zoom, h / args.zoom,
                               fill=False, ec='white', lw=1.6, ls=(0, (4, 2))))
    ax.set_axis_off()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)
    concept(os.path.join(args.out, 'concept_view.png'))
    shutil.copy2(os.path.join(args.out, 'concept_view.png'), os.path.join(args.report, 'concept_view.png'))
    picks = {}
    for name in [s for s in args.scenes.split(',') if s]:
        sname = SET_OF.get(name, name)
        sdir = os.path.join(REPO, 'testing_data/3dshape/cm_{}_a_{}'.format(sname, args.variant))
        res = os.path.join(REPO, 'results/constrained_movement', args.plan, '{}_ntf_s{}'.format(name, args.seed),
                           'cm_{}_a_{}__base_{}'.format(sname, args.variant, args.mode))
        if not (os.path.exists(res + '.json') and os.path.exists(res + '_paths.npz')):
            print('skip', name, '(no evaluation yet)')
            continue
        meta = json.load(open(os.path.join(sdir, 'meta.json')))
        con = meta['constraint']
        scene = G.scene_from_meta(meta, REPO)
        shape = G.Shape(con['shape_parts_raw'], scene.env_scale)
        pyr = G.Pyramid(con['pyramid']['W'], con['pyramid']['H'], con['pyramid']['D'])
        path = G.Path.load(os.path.join(sdir, 'path.npz'))
        ver = json.load(open(os.path.join(sdir, 'verification.json')))['cases']
        J = json.load(open(res + '.json'))
        z = np.load(res + '_paths.npz')
        recs = J['records']
        succ = [r for r in recs if r['success_se3'] and not ver[r['case']]['direct_short']] or \
            [r for r in recs if r['success_se3']]
        if not succ:
            print('skip', name, '(no success)')
            continue
        s = max(succ, key=lambda r: r['rot_len_geo'])
        fails = [r for r in recs if r['collision_se3']] or [r for r in recs if not r['success_se3']]
        f = fails[0] if fails else None
        fk = None if f is None else ('collision' if f['collision_se3'] else 'stall')
        P = plan_of(z, s['case'], s['iters_se3'] if s['iters_se3'] is not None else -1)
        Pf = None if f is None else plan_of(z, f['case'], f['iters_se3'] if f['iters_se3'] is not None else -1)
        prof = free_profile(scene, pyr, path)
        overview(sname, scene, shape, pyr, path, path.extra, P, Pf, fk, prof, os.path.join(args.out, name + '_overview.png'))
        rows = [('success: case {}'.format(s['case']), P, GOOD)]
        if Pf is not None:
            rows.append(('{}: case {}'.format(fk, f['case']), Pf, CRIT if fk == 'collision' else WARN))
        views(sname, scene, shape, pyr, path, rows, os.path.join(args.out, name + '_views.png'))
        picks[name] = {'success_case': s['case'], 'success_rot_turns': s['rot_len_geo'],
                       'failure_case': None if f is None else f['case'], 'failure_kind': fk,
                       'profile': {'t': prof[0].round(4).tolist(), 'free': prof[1].round(4).tolist()}}
        for k in ('overview', 'views'):
            shutil.copy2(os.path.join(args.out, '{}_{}.png'.format(name, k)), os.path.join(args.report, '{}_{}.png'.format(name, k)))
        print(name, 'success case', s['case'], 'failure', fk, None if f is None else f['case'], flush=True)
    pj = os.path.join(args.out, 'picks.json')
    old = {}
    if os.path.exists(pj):
        with open(pj) as fh:
            old = json.load(fh)
    old.update(picks)                                  # keep the scenes not re-rendered this time
    with open(pj, 'w') as fh:
        json.dump(old, fh, indent=1)


if __name__ == '__main__':
    main()
