"""Figures of the drone-pair link study (static PNGs): results/drone_link/figures/.

    scenes.png             the four scenes (y up), the link drawn at a start pose of each crossing set
    showcase_<scene>.png   one crossing case per scene: the NTFields plan (link snapshots, both drones' tracks) next
                           to the independent drones (each on its shortest path), the link length over time with the
                           allowed band, and the contact timeline of both (green = line of sight and range kept)
    rates.png              success / valid rates per scene and set: NTFields (3 seeds), straight, independent drones

The showcase case is the crossing case, among those NTFields seed 0 solves, where the independent drones keep the
line of sight for the smallest share of time (ties: the longer independent paths), unless --case <scene>=<k>.

    python models/drone_link/scripts/make_figures.py            (container, repo root)
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.drone_link import geometry as G  # noqa: E402

RES = os.path.join(REPO, 'results', 'drone_link')
SETS = os.path.join(REPO, 'testing_data', '3dshape')
SCENES = ('forest', 'windows', 'corridor', 'scaffold')
VIEW = {'forest': (24, -58), 'windows': (24, -32), 'corridor': (74, -90), 'scaffold': (20, -62)}
C_A, C_B, C_OK, C_BAD, C_OBS = '#2f6fb5', '#e08a1e', '#2e9e57', '#d62728', '#8c7b6b'

ap = argparse.ArgumentParser()
ap.add_argument('--case', action='append', default=[], help='<scene>=<case index> showcase override')
ap.add_argument('--out', default=os.path.join(RES, 'figures'))
ap.add_argument('--res', default=RES, help='results root (plan/, baselines/, summary/)')
ap.add_argument('--seed-run', default='ntf_s0')
ap.add_argument('--scenes', default=','.join(SCENES))
args = ap.parse_args()
RES = args.res
SCENES = tuple(s for s in args.scenes.split(',') if os.path.exists(os.path.join(SETS, 'link_{}_cross'.format(s),
                                                                                 'sampled_points.npy')))
OVR = {k: int(v) for k, v in (c.split('=') for c in args.case)}
os.makedirs(args.out, exist_ok=True)


def P3(p):
    """scene (x, y, z), y up -> plot (x, z, y)."""
    p = np.asarray(p)
    return p[..., [0, 2, 1]]


def box_faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [P3(c[list(q)]) for q in f]


def draw_scene(ax, sc, alpha=0.35):
    faces = []
    for lo, hi in zip(sc.box_lo, sc.box_hi):
        faces += box_faces(lo, hi)
    ax.add_collection3d(Poly3DCollection(faces, facecolor=C_OBS, edgecolor='#4d4339', linewidths=0.3, alpha=alpha))
    lo, hi = sc.room_lo, sc.room_hi
    floor = P3(np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]]))
    ax.add_collection3d(Poly3DCollection([floor], facecolor='#e9e4dc', edgecolor='none', alpha=0.5))
    E = []
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for i in range(8):
        for j in range(i + 1, 8):
            if (np.abs(corners[i] - corners[j]) > 1e-9).sum() == 1:
                E.append(P3(corners[[i, j]]))
    ax.add_collection3d(Line3DCollection(E, colors='#9a948c', linewidths=0.5))
    m = P3(np.stack([lo, hi]))
    ax.set_xlim(m[0, 0], m[1, 0])
    ax.set_ylim(m[0, 1], m[1, 1])
    ax.set_zlim(m[0, 2], m[1, 2])
    ax.set_box_aspect(m[1] - m[0], zoom=1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(False)


def link_box(ax, sc, link, a, b, ok, lw=None):
    d = b - a
    ax.plot(*P3(np.stack([a, b])).T, color=C_OK if ok else C_BAD, lw=lw or 3.4, solid_capstyle='round', zorder=5)
    ax.scatter(*P3(a[None]).T, s=16, color=C_A, depthshade=False, zorder=6)
    ax.scatter(*P3(b[None]).T, s=16, color=C_B, depthshade=False, zorder=6)
    return d


def ends_of_chart(P, link):
    t, R, L = G.from_chart(torch.as_tensor(P, dtype=torch.float64))
    a, b = link.ends(t, R, L)
    return a.numpy(), b.numpy(), L.numpy()


def resample(P, k):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    if s[-1] < 1e-12:
        return np.repeat(P[:1], k, 0)
    u = np.linspace(0, s[-1], k)
    return np.stack([np.interp(u, s, P[:, j]) for j in range(3)], 1)


def pair_status(sc, link, pa, pb):
    D = pb - pa
    L = np.linalg.norm(D, axis=1)
    d = D / np.maximum(L, 1e-12)[:, None]
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(np.broadcast_to(ez, d.shape), d)
    s = np.linalg.norm(v, axis=1)
    rv = v / np.maximum(s, 1e-12)[:, None] * np.arctan2(s, d[:, 2])[:, None]
    rv[(s < 1e-9) & (d[:, 2] < 0)] = [math.pi, 0, 0]
    R = G.so3_exp(torch.as_tensor(rv, dtype=torch.float64))
    cs = G.scene_clearance(sc, link, torch.as_tensor(pa), R, torch.as_tensor(np.maximum(L, 1e-6))).numpy()
    return cs, L


def load(p):
    with open(p) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- scenes overview
fig = plt.figure(figsize=(3.5 * max(1, len(SCENES)), 3.9))
for i, scn in enumerate(SCENES):
    s = 'link_{}_cross'.format(scn)
    meta = load(os.path.join(SETS, s, 'meta.json'))
    sc, link = G.load_scene(meta, REPO), G.load_link(meta)
    ax = fig.add_subplot(1, len(SCENES), i + 1, projection='3d')
    draw_scene(ax, sc)
    X = np.load(os.path.join(SETS, s, 'sampled_points.npy'))
    for x in X[:6]:
        for half in (x[:7], x[7:]):
            a, b, _ = ends_of_chart(half[None], link)
            link_box(ax, sc, link, a[0], b[0], True, lw=2.0)
    ax.view_init(*VIEW[scn])
    ax.set_title(scn, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'scenes.png'), dpi=130, bbox_inches='tight')
plt.close(fig)
print('scenes.png')

# ----------------------------------------------------------------------------- showcases
for scn in SCENES:
    s = 'link_{}_cross'.format(scn)
    ev_p = os.path.join(RES, 'plan', '{}_{}'.format(scn, args.seed_run), s + '.json')
    base_p = os.path.join(RES, 'baselines', s + '.json')
    if not (os.path.exists(ev_p) and os.path.exists(base_p)):
        print('skip', scn, '(no evaluation / baseline yet)')
        continue
    meta = load(os.path.join(SETS, s, 'meta.json'))
    sc, link = G.load_scene(meta, REPO), G.load_link(meta)
    ev, base = load(ev_p), load(base_p)
    pz = np.load(ev_p[:-5] + '_paths.npz')
    iz = np.load(base_p[:-5] + '_independent.npz', allow_pickle=True)
    ind_rec = {r['case']: r['independent'] for r in base['records']}
    ok_cases = [r['case'] for r in ev['records'] if r['success_se3']]
    if scn in OVR:
        k = OVR[scn]
    else:
        cand = [c for c in ok_cases if ind_rec[c].get('solved')]
        k = min(cand, key=lambda c: (ind_rec[c]['contact_frac'], -(ind_rec[c]['len_a'] + ind_rec[c]['len_b'])))
    rec = next(r for r in ev['records'] if r['case'] == k)
    j = int(np.nonzero(pz['case'] == k)[0][0])
    P = pz['path'][j]
    k3 = int(pz['iters3'][j])
    P = P[:k3 + 2] if k3 >= 0 else P[~np.isnan(P[:, 0])]
    a, b, L = ends_of_chart(P, link)
    clr = pz['clear'][j][:max(1, len(P) - 1)].reshape(-1)
    ji = int(np.nonzero(iz['case'] == k)[0][0])
    Pa, Pb = np.asarray(iz['a'][ji], float), np.asarray(iz['b'][ji], float)
    T = 400
    ra, rb = resample(Pa, T), resample(Pb, T)
    cs_i, L_i = pair_status(sc, link, ra, rb)
    ok_i = (cs_i >= 0) & (L_i <= link.L_max) & (L_i >= link.L_min)

    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[4.2, 1.0, 0.45], hspace=0.12, wspace=0.0)
    for col, (title, A, B, OK) in enumerate((
            ('NTFields, 7 DOF (link planned as one body)', a, b, None),
            ('independent drones (each on its shortest path)', ra, rb, ok_i))):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        draw_scene(ax, sc, alpha=0.18 if scn in ('corridor', 'windows') else 0.28)
        ax.plot(*P3(A).T, color=C_A, lw=1.4)
        ax.plot(*P3(B).T, color=C_B, lw=1.4)
        n = len(A)
        snaps = np.unique(np.linspace(0, n - 1, 9).round().astype(int))
        if OK is not None:   # add the worst moments of the independent drones
            bad = np.nonzero(~OK)[0]
            if len(bad):
                snaps = np.unique(np.concatenate([snaps[::2], bad[np.linspace(0, len(bad) - 1, min(4, len(bad))).astype(int)]]))
        for i in snaps:
            good = True if OK is None else bool(OK[i])
            link_box(ax, sc, link, A[i], B[i], good)
        ax.view_init(*VIEW[scn])
        ax.set_title(title, fontsize=11, y=1.04)
    # link length over time
    ax = fig.add_subplot(gs[1, :])
    u_n = np.linspace(0, 1, len(L))
    u_i = np.linspace(0, 1, T)
    ax.axhspan(link.L_min, link.L_max, color='#2e9e57', alpha=0.08, lw=0)
    ax.axhline(link.L_min, color='#2e9e57', lw=0.8, ls='--')
    ax.axhline(link.L_max, color='#2e9e57', lw=0.8, ls='--')
    ax.plot(u_n, L, color='#333333', lw=1.8, label='NTFields: link length L')
    ax.plot(u_i, L_i, color='#a05cc8', lw=1.5, label='independent: |b - a|')
    ax.set_ylabel('length')
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    ax.set_xticklabels([])
    # contact timelines
    ax = fig.add_subplot(gs[2, :])
    c_n = clr >= 0
    for row, (u, ok) in enumerate(((np.linspace(0, 1, len(c_n)), c_n), (u_i, ok_i))):
        cols = np.where(ok, C_OK, C_BAD)
        ax.scatter(u, np.full(len(u), 1 - row), c=cols, marker='|', s=90, linewidths=1.2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(['NTFields', 'independent'], fontsize=9)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel('normalized time  (green: drones see each other within range;  red: line of sight blocked or out of range)',
                  fontsize=9)
    fig.suptitle('{}: crossing case {}  (NTFields contact {:.0f} %, independent drones {:.0f} %)'.format(
        scn, k, 100 * float(c_n.mean()), 100 * float(ok_i.mean())), fontsize=12, y=0.995)
    fig.savefig(os.path.join(args.out, 'showcase_{}.png'.format(scn)), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('showcase_{}.png case {}'.format(scn, k))

# ----------------------------------------------------------------------------- rates
summ = os.path.join(RES, 'summary', 'summary.json')
if os.path.exists(summ):
    S = load(summ)['sets']
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, v in zip(axes, ('std', 'cross')):
        xs = np.arange(len(SCENES))
        w = 0.26
        ntf = [S.get('link_{}_{}'.format(sc, v), {}).get('ntf', {}) for sc in SCENES]
        mean = [100 * (n.get('success_rate_se3') or 0) for n in ntf]
        sd = [100 * float(np.std(n.get('success_per_seed') or [0])) for n in ntf]
        st = [100 * (S.get('link_{}_{}'.format(sc, v), {}).get('baselines', {}).get('straight', {}).get('success_rate') or 0)
              for sc in SCENES]
        ind = [100 * (S.get('link_{}_{}'.format(sc, v), {}).get('baselines', {}).get('independent', {}).get('valid_rate') or 0)
               for sc in SCENES]
        ax.bar(xs - w, mean, w, yerr=sd, color='#2f6fb5', label='NTFields 7-DOF plan (3 seeds)', capsize=2)
        ax.bar(xs, ind, w, color='#a05cc8', label='independent drones')
        ax.bar(xs + w, st, w, color='#b8b2a7', label='straight 7-d motion')
        for x, m in zip(xs - w, mean):
            ax.text(x, m + 1.5, '{:.0f}'.format(m), ha='center', fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(SCENES)
        ax.set_title('{} set: cases solved with contact kept the whole way (%)'.format(v), fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'rates.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('rates.png')
