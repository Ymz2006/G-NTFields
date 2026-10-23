#!/usr/bin/env python3
"""Build the report page (index.html next to this file) from the aggregated results and the figures.

    python3 models/constrained_movement/scripts/aggregate_cmove.py
    python models/constrained_movement/scripts/make_figures_cmove.py --seed 0          # report/img/*.png
    python3 models/constrained_movement/report/build_report.py [--status "..."] [--notes notes.json]

notes.json (optional): {"lede": "...", "scenes": {"forest": "...", ...}, "findings": ["...", ...]} written after
looking at the numbers; numbers themselves always come from summary.json.
"""
import argparse
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(REPO, 'results/constrained_movement')
SCENES = ['forest', 'city', 'corridor2', 'env1', 'env4']        # runs; corridor2 = the corridor, longer training pairs
SET_OF = {'corridor2': 'corridor'}
LABEL = {'corridor2': 'corridor'}
FLIGHT = {'forest': 'low between the trunks, around one tree',
          'city': 'over a low roof, past a tall building, across a street',
          'corridor': 'the whole serpentine, two U-turns',
          'env1': 'diagonally down between the upper and lower slabs',
          'env4': 'from above one box down past two others'}
PICK = {'forest': 5, 'city': 1, 'corridor': 13, 'env1': 8, 'env4': 13}

ap = argparse.ArgumentParser()
ap.add_argument('--status', default='')
ap.add_argument('--notes', default=os.path.join(HERE, 'notes.json'))
ap.add_argument('--summary', default='summary', help='results/constrained_movement/<summary>/summary.json')
ap.add_argument('--figures', default='figures', help='results/constrained_movement/<figures>/picks.json')
args = ap.parse_args()


def load(p, default=None):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


S = load(os.path.join(RES, args.summary, 'summary.json'), {'cells': {}, 'sets': {}, 'train': {}})
picks = load(os.path.join(RES, args.figures, 'picks.json'), {})
notes = load(args.notes, {})
CORR = notes.get('corridor_run', 'corridor2')
SCENES = [CORR if sc == 'corridor2' else sc for sc in SCENES]
SET_OF[CORR] = 'corridor'
LABEL[CORR] = 'corridor'
esc = html.escape


def cell(scene, variant, mode='forward'):
    return S['cells'].get('{}|{}|{}'.format(scene, variant, mode))


def pct(v, d=1):
    return '–' if v is None else '{:.{}f}'.format(100 * v, d)


def mean(c, k):
    return None if c is None else c['agg'][k]['mean']


def pm(c, k):
    if c is None:
        return '–'
    a = c['agg'][k]
    return pct(a['mean']) + ('<span class="pm"> ± {}</span>'.format(pct(a['std'])) if a['n_seeds'] > 1 else '')


def path_stats(scene):
    sc = SET_OF.get(scene, scene)
    j = load(os.path.join(REPO, 'datasets/3dshape/paths/cm_{}_cand.json'.format(sc)))
    return None if j is None else j['candidates'][PICK[sc]]


# ----------------------------------------------------------------------------- diagram
def diagram():
    """Side view: the bird on its path, the pyramid anchored at its centre in two orientations, an obstacle."""
    ax_, ay_ = 330.0, 215.0                        # anchor (bird centre)
    D, half = 150.0, 50.0                          # apex distance, half base (drawn scale)

    def pyr(deg):
        a = math.radians(deg)
        u = (math.cos(a), -math.sin(a))            # anchor -> camera (screen y down)
        n = (-u[1], u[0])
        return ((ax_ + D * u[0], ay_ + D * u[1]), (ax_ + half * n[0], ay_ + half * n[1]),
                (ax_ - half * n[0], ay_ - half * n[1]))

    def pts(*p):
        return ' '.join('{:.1f},{:.1f}'.format(x, y) for x, y in p)

    (a1, b11, b12), (a2, b21, b22) = pyr(150), pyr(50)
    r = 74.0
    arc = 'M {:.1f} {:.1f} A {r:.0f} {r:.0f} 0 0 1 {:.1f} {:.1f}'.format(
        ax_ + r * math.cos(math.radians(140)), ay_ - r * math.sin(math.radians(140)),
        ax_ + r * math.cos(math.radians(62)), ay_ - r * math.sin(math.radians(62)), r=r)
    lab = (ax_ + 124 * math.cos(math.radians(100)), ay_ - 124 * math.sin(math.radians(100)))
    low = b11 if b11[1] > b12[1] else b12
    return '''<svg viewBox="0 0 700 320" role="img" aria-label="The camera pyramid's base is centred on the bird and moves with it along the path; the pyramid can only rotate about that centre, and a pose whose pyramid touches an obstacle is not allowed.">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker>
    <marker id="ahc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--cam)"/></marker>
  </defs>
  <g style="color: var(--ink-2)">
    <path d="M 40 270 C 150 262, 240 222, 330 215 S 540 236, 640 272" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="7 5" marker-end="url(#ah)"/>
  </g>
  <text x="40" y="308" class="m">the bird's path p(t) from the RRT; t = arc length, 0 to L</text>
  <text x="652" y="300" class="m" text-anchor="end">t grows</text>
  <rect x="404" y="52" width="120" height="58" rx="3" fill="var(--rule)" stroke="var(--muted)" stroke-width="1"/>
  <text x="464" y="86" text-anchor="middle" class="m">obstacle</text>
  <polygon points="{p2}" fill="var(--crit)" fill-opacity="0.10" stroke="var(--crit)" stroke-width="1.5" stroke-dasharray="5 4"/>
  <text x="534" y="70" fill="var(--crit)">touches the obstacle:</text>
  <text x="534" y="88" fill="var(--crit)">this R is not allowed</text>
  <polygon points="{p1}" fill="var(--cam)" fill-opacity="0.16" stroke="var(--cam)" stroke-width="1.8"/>
  <line x1="{b11x:.1f}" y1="{b11y:.1f}" x2="{b12x:.1f}" y2="{b12y:.1f}" stroke="var(--cam)" stroke-width="4" stroke-linecap="round"/>
  <circle cx="{a1x:.1f}" cy="{a1y:.1f}" r="5" fill="var(--cam)"/>
  <text x="{a1lx:.1f}" y="{a1ly:.1f}" fill="var(--cam)" text-anchor="end">camera (apex)</text>
  <path d="{arc}" fill="none" stroke="var(--cam)" stroke-width="1.6" marker-end="url(#ahc)"/>
  <text x="{lx:.1f}" y="{ly:.1f}" fill="var(--cam)" text-anchor="middle">R: rotation about the anchor</text>
  <ellipse cx="{ax:.1f}" cy="{ay:.1f}" rx="40" ry="9" fill="var(--bird)"/>
  <path d="M 298 211 L 280 198 L 290 213 Z" fill="var(--bird)"/>
  <circle cx="{ax:.1f}" cy="{ay:.1f}" r="4.5" fill="var(--ink)"/>
  <text x="372" y="276" fill="var(--ink)">anchor = bird's centre = base centre</text>
  <text x="{blx:.1f}" y="{bly:.1f}" class="m" text-anchor="end">base 48 × 22</text>
</svg>'''.format(p1=pts(a1, b11, b12), p2=pts(a2, b21, b22), a1x=a1[0], a1y=a1[1], a1lx=a1[0] - 10, a1ly=a1[1] - 10,
                 b11x=b11[0], b11y=b11[1], b12x=b12[0], b12y=b12[1], arc=arc, lx=lab[0], ly=lab[1], ax=ax_, ay=ay_,
                 blx=low[0] - 10, bly=low[1] + 4)


# ----------------------------------------------------------------------------- outcome bars
def bars():
    """Stacked outcome bars per scene and set (forward mode): success, collision, stall without collision."""
    rows = []
    for sc in SCENES:
        for v in ('full', 'seg'):
            c = cell(sc, v)
            if c is not None:
                rows.append((sc, v, mean(c, 'success'), mean(c, 'collision'), mean(c, 'stall_clean'),
                             S['sets'].get('cm_{}_a_{}'.format(SET_OF.get(sc, sc), v), {}).get('direct_short_ok')))
    if not rows:
        return '<p class="muted">No evaluations yet.</p>'
    W, lab, top, rh = 760.0, 150.0, 26.0, 30.0
    bw = W - lab - 70
    H = top + rh * len(rows) + 30
    out = ['<svg viewBox="0 0 {:.0f} {:.0f}" role="img" aria-label="Outcome of every NTFields plan, forward mode: '
           'success, collision, or stall, per scene and test set, with the straight-motion rate as a tick.">'.format(W, H)]
    for g in range(0, 101, 25):
        x = lab + bw * g / 100
        out.append('<line class="grid" x1="{0:.1f}" y1="{1:.1f}" x2="{0:.1f}" y2="{2:.1f}"/>'.format(x, top - 6, H - 26))
        out.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle">{}%</text>'.format(x, H - 10, g))
    for i, (sc, v, s, c, st, straight) in enumerate(rows):
        y = top + i * rh
        out.append('<text class="rowlab" x="{:.1f}" y="{:.1f}" text-anchor="end">{} · {}</text>'.format(
            lab - 10, y + 15, LABEL.get(sc, sc), 'whole path' if v == 'full' else 'sub-paths'))
        x = lab
        for val, col, name in ((s, 'var(--good)', 'success'), (c, 'var(--crit)', 'collision'),
                               (st, 'var(--warn)', 'stall')):
            if val is None or val <= 0:
                continue
            w = bw * val
            out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="18" fill="{}" rx="2"><title>{} · {}: {} {}%'
                       '</title></rect>'.format(x, y + 2, max(w - 2, 0.5), col, sc, v, name, pct(val)))
            x += w
        if straight is not None:
            xs = lab + bw * straight
            out.append('<line x1="{0:.1f}" y1="{1:.1f}" x2="{0:.1f}" y2="{2:.1f}" stroke="var(--ink)" stroke-width="2">'
                       '<title>straight motion collision-free: {3}%</title></line>'.format(xs, y - 1, y + 23, pct(straight)))
        out.append('<text x="{:.1f}" y="{:.1f}" class="val">{}%</text>'.format(lab + bw + 8, y + 15, pct(s)))
    out.append('</svg>')
    return '\n'.join(out)


# ----------------------------------------------------------------------------- page
def results_table(mode):
    head = ('<thead><tr><th>scene</th><th>set</th><th>L</th><th>straight ok</th><th>NTFields success</th>'
            '<th>collision</th><th>stall, no collision</th><th>seeds</th></tr></thead>')
    body = []
    for sc in SCENES:
        for v in ('full', 'seg'):
            c = cell(sc, v, mode)
            if c is None:
                continue
            st = S['sets'].get('cm_{}_a_{}'.format(SET_OF.get(sc, sc), v), {})
            body.append('<tr><td>{}</td><td>{}</td><td class="num">{}</td><td class="num">{}</td><td class="num strong">{}'
                        '</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td></tr>'.format(
                            LABEL.get(sc, sc), 'whole path' if v == 'full' else 'sub-paths',
                            '{:.2f}'.format(st['L']) if st.get('L') else '–', pct(st.get('direct_short_ok')),
                            pm(c, 'success'), pm(c, 'collision'), pm(c, 'stall_clean'), c['agg']['success']['n_seeds']))
    if not body:
        return ''
    return '<div class="tbl"><table>{}<tbody>{}</tbody></table></div>'.format(head, ''.join(body))


def path_table():
    rows = []
    for sc in SCENES:
        c = path_stats(sc)
        if c is None:
            continue
        rows.append('<tr><td>{}</td><td class="txt">{}</td><td class="num">{:.2f}</td><td class="num">{:.2f} / {:.2f}</td>'
                    '<td class="num">{}</td><td class="num">{}</td></tr>'.format(
                        LABEL.get(sc, sc), esc(FLIGHT[SET_OF.get(sc, sc)]), c['L'], c['free_mean'], c['free_min'], pct(c['straight_ok']), c['rrt_iters']))
    return ('<div class="tbl"><table><thead><tr><th>scene</th><th class="txt">the bird flies</th><th>L</th>'
            '<th>camera room mean / min</th><th>straight ok</th><th>RRT iterations</th></tr></thead><tbody>{}</tbody>'
            '</table></div>'.format(''.join(rows)))


def tiles():
    out = []
    for sc in SCENES:
        c = cell(sc, 'full')
        st = S['sets'].get('cm_{}_a_full'.format(SET_OF.get(sc, sc)), {})
        out.append('<div class="tile"><div class="lab">{}</div><div class="big num">{}<span class="unit">%</span></div>'
                   '<div class="sub">straight motion {}%</div></div>'.format(
                       LABEL.get(sc, sc), pct(mean(c, 'success')) if c else '…', pct(st.get('direct_short_ok')) if st else '–'))
    return '<div class="tiles">{}</div>'.format(''.join(out))


def scene_sections():
    out = []
    for sc in SCENES:
        if not os.path.exists(os.path.join(HERE, 'img', sc + '_overview.png')):
            continue
        cf, cs = cell(sc, 'full'), cell(sc, 'seg')
        pk = picks.get(sc, {})
        auto = 'Whole path: {}% success ({}% collision, {}% stall){}.'.format(
            pct(mean(cf, 'success')), pct(mean(cf, 'collision')), pct(mean(cf, 'stall_clean')),
            '; sub-paths: {}%'.format(pct(mean(cs, 'success'))) if cs else '')
        txt = notes.get('scenes', {}).get(sc, '')
        out.append('''<section class="scene" id="{sc}">
  <h3>{label}</h3>
  <p class="col">{auto} {txt}</p>
  <figure><img src="img/{sc}_overview.png" alt="{sc}: the scene, the bird's path and the camera pyramid along a successful NTFields plan, with the camera room and the clearance along the plans" loading="lazy">
  <figcaption>Seed 0. Left: the bird's RRT waypoints (dashed) and smoothed path, the bird and the camera pyramid at points of a successful plan (case {s}), the camera's track (blue). Right: the fraction of camera rotations that are free along the path, and the pyramid's clearance along the successful plan and along a failed one ({fk}, case {f}).</figcaption></figure>
  <figure><img src="img/{sc}_views.png" alt="{sc}: what the camera sees along the successful and the failed plan" loading="lazy">
  <figcaption>What the camera sees, rendered with a 2.5× wider lens; the dashed rectangle is the pyramid's base around the bird, the part of the view a plan must keep clear (red where an obstacle is inside it).</figcaption></figure>
</section>'''.format(sc=sc, label=LABEL.get(sc, sc), auto=auto, txt=esc(txt), s=pk.get('success_case', '–'), f=pk.get('failure_case', '–'),
                     fk=pk.get('failure_kind', '–')))
    return '\n'.join(out)


CORR_EVALS = {('corridor', 2000): 'plan_steps860_e2000', ('corridor', 4000): 'plan_steps860_e4000',
              ('corridor', 8000): 'plan_steps860', ('corridor2', 2000): 'plan_e2000', ('corridor2', 4000): 'plan_e4000',
              ('corridor2', 8000): 'plan', ('corridor3', 2000): 'plan_e2000', ('corridor3', 4000): 'plan_e4000',
              ('corridor3', 8000): 'plan'}


def corridor_rate(run, epoch, variant='full'):
    J = load(os.path.join(RES, CORR_EVALS[(run, epoch)], run + '_ntf_s0',
                          'cm_corridor_a_{}__base_forward.json'.format(variant)))
    return None if J is None or J['summary']['steps'] < 800 else J['summary']['success_rate_se3']


def ablation():
    """The corridor runs at equal epochs (860 MPPI steps): how the training pairs were drawn, and training length."""
    runs = [('corridor', 'isotropic in (t, r), length up to 2.0 (the prism study\'s bound)'),
            ('corridor2', 'isotropic in (t, r), length up to sqrt(L² + 3) = 2.75'),
            ('corridor3', 't part stretched by L = 2.14, length up to 2 in (t / L, r)')]
    rows = []
    for run, how in runs:
        vals = [corridor_rate(run, e) for e in (2000, 4000, 8000)]
        if all(v is None for v in vals):
            continue
        sub = next((corridor_rate(run, e, 'seg') for e in (8000, 4000, 2000) if corridor_rate(run, e, 'seg') is not None),
                   None)
        rows.append('<tr><td>{}</td><td class="txt">{}</td>{}<td class="num">{}</td></tr>'.format(
            run + (' (shown above)' if run == CORR else ''), esc(how),
            ''.join('<td class="num">{}</td>'.format(pct(v) if v is not None else 'training') for v in vals), pct(sub)))
    if not rows:
        return ''
    return ('<h3>Corridor: pair range and training length</h3><p class="col">{}</p><div class="tbl"><table>'
            '<thead><tr><th>run</th><th class="txt">training pair displacement</th><th>whole path, epoch 2000</th>'
            '<th>4000</th><th>8000</th><th>sub-paths, latest</th></tr></thead><tbody>{}</tbody></table></div>'.format(
                esc(notes.get('corridor', '')), ''.join(rows)))


def epochs_table():
    """Whole-path success of the seed-0 runs at epochs 2000, 4000 and 8000."""
    sums = [(e, load(os.path.join(RES, d, 'summary.json'))) for e, d in
            ((2000, 'summary_e2000'), (4000, 'summary_e4000'), (8000, 'summary'))]
    rows = []
    for sc in SCENES:
        if sc == CORR:
            continue
        vals = []
        for e, sm in sums:
            c = None if sm is None else sm['cells'].get('{}|full|forward'.format(sc))
            v = None if c is None else c['seeds'].get('0', {}).get('success')
            vals.append(pct(v))
        if any(v != '–' for v in vals):
            rows.append('<tr><td>{}</td>{}</tr>'.format(LABEL.get(sc, sc), ''.join('<td class="num">{}</td>'.format(v)
                                                                                for v in vals)))
    if not rows:
        return ''
    return ('<h3>Training length</h3><p class="col">Whole-path success of the seed-0 runs at three checkpoints. It '
            'moves by up to 9 points between checkpoints, in both directions, so a single run carries several points '
            'of noise; a second seed will show how much of it is seed-to-seed variation.</p><div class="tbl"><table><thead><tr><th>scene</th><th>epoch 2000</th>'
            '<th>epoch 4000</th><th>epoch 8000</th></tr></thead><tbody>{}</tbody></table></div>'.format(''.join(rows)))


def failure_profiles():
    """Small multiples per scene: the camera room along the path (fraction of rotations free) above a histogram
    of where the failed whole-path plans collide (first blocked pose) or stall (where they end)."""
    panels = []
    for sc in SCENES:
        c, pk = cell(sc, 'full'), picks.get(sc)
        if c is None or not pk:
            continue
        r = next(iter(c['seeds'].values()))
        if 'collision_where' not in r:
            continue
        panels.append((sc, pk['profile'], r['collision_where'], r['stall_where'], r['n']))
    if not panels:
        return ''
    W, H, x0, x1 = 236.0, 178.0, 30.0, 226.0
    top0, top1, bot0, bot1 = 22.0, 82.0, 100.0, 150.0
    nb = 20
    hist = []
    for sc, prof, cw, sw, n in panels:
        hc = [0] * nb
        hs = [0] * nb
        for w in cw:
            hc[min(nb - 1, max(0, int(w * nb)))] += 1
        for w in sw:
            hs[min(nb - 1, max(0, int(w * nb)))] += 1
        hist.append((hc, hs, n))
    out = []
    for (sc, prof, cw, sw, n), (hc, hs, _) in zip(panels, hist):
        vmax = max(a + b for a, b in zip(hc, hs)) / n
        vmax = max(0.02, math.ceil(vmax * 50) / 50)            # each panel its own scale, labelled
        X = lambda t: x0 + (x1 - x0) * t  # noqa: E731
        Yp = lambda f: top1 - (top1 - top0) * f  # noqa: E731
        Yh = lambda v: bot1 - (bot1 - bot0) * v / vmax  # noqa: E731
        pts = ' '.join('{:.1f},{:.1f}'.format(X(t), Yp(f)) for t, f in zip(prof['t'], prof['free']))
        area = '{:.1f},{:.1f} {} {:.1f},{:.1f}'.format(X(0), Yp(0), pts, X(1), Yp(0))
        g = ['<svg viewBox="0 0 {:.0f} {:.0f}" role="img" aria-label="{}: camera room along the path and where the '
             'failed plans collide or stall">'.format(W, H, LABEL.get(sc, sc))]
        g.append('<text x="{:.0f}" y="13" class="ttl">{}</text>'.format(x0, LABEL.get(sc, sc)))
        for f in (0, 0.5, 1):
            g.append('<line class="grid" x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}"/>'.format(x0, Yp(f), x1, Yp(f)))
            g.append('<text x="{:.1f}" y="{:.1f}" text-anchor="end">{:.0%}</text>'.format(x0 - 4, Yp(f) + 4, f))
        g.append('<polygon points="{}" fill="var(--cam)" fill-opacity="0.16"/>'.format(area))
        g.append('<polyline points="{}" fill="none" stroke="var(--cam)" stroke-width="1.6"/>'.format(pts))
        bw = (x1 - x0) / nb
        for i in range(nb):
            vc, vs = hc[i] / n, hs[i] / n
            if vc > 0:
                g.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="var(--crit)"><title>{}: '
                         'collisions at t/L {:.2f}-{:.2f}: {:.1%} of cases</title></rect>'.format(
                             X(i / nb) + 0.6, Yh(vc), bw - 1.2, bot1 - Yh(vc), LABEL.get(sc, sc), i / nb, (i + 1) / nb, vc))
            if vs > 0:
                g.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="var(--warn)"><title>{}: '
                         'stalls ending at t/L {:.2f}-{:.2f}: {:.1%} of cases</title></rect>'.format(
                             X(i / nb) + 0.6, Yh(vc + vs), bw - 1.2, Yh(vc) - Yh(vc + vs), LABEL.get(sc, sc), i / nb,
                             (i + 1) / nb, vs))
        g.append('<line x1="{0:.1f}" y1="{1:.1f}" x2="{2:.1f}" y2="{1:.1f}" class="axis"/>'.format(x0, bot1, x1))
        g.append('<text x="{:.1f}" y="{:.1f}" text-anchor="end">{:.0%}</text>'.format(x0 - 4, Yh(vmax) + 4, vmax))
        for t in (0, 0.5, 1):
            g.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle">{:g}</text>'.format(X(t), bot1 + 14, t))
        g.append('</svg>')
        out.append('<div class="sm">{}</div>'.format(''.join(g)))
    return ('<h3>Where the plans fail</h3><p class="col">Top of each panel: the camera room along the bird\'s path '
            '(the fraction of random camera rotations that are free). Bottom: where the failed whole-path plans '
            'collide (first blocked pose, red) or stall (where the plan ends, amber), as a share of all cases (seed 0), '
            'each panel on its own scale; the x axis is the bird\'s progress t / L.</p><div class="smgrid">{}</div>'.format(''.join(out)))


def findings():
    items = notes.get('findings', [])
    if not items:
        return ''
    return '<h2>What the numbers say</h2><ul class="col">{}</ul>'.format(''.join('<li>{}</li>'.format(esc(i)) for i in items))


PAGE = '''<title>Constrained Movement</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a lab-notebook page: one reading column (~68ch), figures and tables run the full 1080px. */
:root {{
  --bg: #f4f6f8; --surface: #ffffff; --ink: #14181c; --ink-2: #4b535c; --muted: #6f7882; --rule: #dce1e6;
  --cam: #2a78d6; --bird: #eb6834; --good: #0ca30c; --crit: #d03b3b; --warn: #e3a008;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #111417; --surface: #1a1e22; --ink: #eceff2; --ink-2: #b8bfc6; --muted: #8d969f; --rule: #2b3137;
    --cam: #3987e5; --bird: #d95926; --good: #0ca30c; --crit: #e66767; --warn: #fab219; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #111417; --surface: #1a1e22; --ink: #eceff2; --ink-2: #b8bfc6; --muted: #8d969f; --rule: #2b3137;
  --cam: #3987e5; --bird: #d95926; --good: #0ca30c; --crit: #e66767; --warn: #fab219; color-scheme: dark;
}}
body {{ background: var(--bg); color: var(--ink); font: 16.5px/1.6 var(--f-body); }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.col {{ max-width: 68ch; }}
h1, h2, h3 {{ font-family: var(--f-display); font-weight: 600; line-height: 1.15; text-wrap: balance; margin: 0; }}
h1 {{ font-size: clamp(2.2rem, 5vw, 3.1rem); letter-spacing: -0.01em; }}
h2 {{ font-size: 1.55rem; margin-top: 56px; padding-top: 16px; border-top: 1px solid var(--rule); }}
h3 {{ font-size: 1.25rem; margin-top: 40px; }}
p, ul {{ margin: 12px 0 0; }}
li + li {{ margin-top: 6px; }}
.eyebrow {{ font-family: var(--f-mono); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }}
.lede {{ font-size: 1.12rem; margin-top: 16px; }}
.status {{ display: inline-block; margin-top: 14px; font-size: 0.85rem; color: var(--ink-2); background: var(--surface); border: 1px solid var(--rule); border-radius: 999px; padding: 3px 12px; }}
code, .mono {{ font-family: var(--f-mono); font-size: 0.86em; }}
code {{ background: var(--surface); border: 1px solid var(--rule); padding: 0 5px; border-radius: 4px; overflow-wrap: anywhere; }}
.num {{ font-family: var(--f-mono); font-variant-numeric: tabular-nums; }}
.pm {{ color: var(--muted); font-size: 0.85em; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 26px; }}
.tile {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 14px; min-width: 0; }}
.tile .lab {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }}
.tile .big {{ font-size: 1.9rem; line-height: 1.15; margin-top: 4px; color: var(--ink); }}
.tile .unit {{ font-size: 1rem; color: var(--muted); margin-left: 2px; }}
.tile .sub {{ font-size: 0.82rem; color: var(--muted); margin-top: 2px; }}
.tiles-cap {{ font-size: 0.88rem; color: var(--muted); margin-top: 8px; }}
.tbl {{ overflow-x: auto; margin-top: 14px; }}
table {{ border-collapse: collapse; font-size: 0.92rem; min-width: 100%; }}
th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
th:first-child, td:first-child, th.txt, td.txt {{ text-align: left; }}
td.txt {{ white-space: normal; min-width: 16ch; }}
thead th {{ font-weight: 600; color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }}
td.strong {{ font-weight: 600; }}
.twocol {{ display: grid; grid-template-columns: 1.05fr 1fr; gap: 20px; align-items: start; margin-top: 18px; }}
@media (max-width: 760px) {{ .twocol {{ grid-template-columns: 1fr; }} }}
.twocol > * {{ min-width: 0; }}
figure {{ margin: 20px 0 0; }}
figure img {{ display: block; width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; background: #fff; }}
figcaption {{ font-size: 0.88rem; color: var(--muted); margin-top: 8px; max-width: 84ch; }}
.diagram {{ margin: 0; overflow-x: auto; }}
.diagram svg {{ display: block; width: 100%; min-width: 520px; height: auto; color: var(--ink); }}
.diagram text {{ font-family: var(--f-body); font-size: 14.5px; fill: var(--ink); }}
.diagram text.m {{ fill: var(--muted); }}
.chart {{ overflow-x: auto; margin-top: 14px; }}
.chart svg {{ display: block; width: 100%; min-width: 560px; height: auto; }}
.chart text {{ font-family: var(--f-body); font-size: 12px; fill: var(--muted); }}
.chart text.rowlab {{ fill: var(--ink); }}
.chart text.val {{ fill: var(--ink); font-family: var(--f-mono); }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.smgrid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px 16px; margin-top: 12px; }}
.sm {{ min-width: 0; }}
.sm svg {{ display: block; width: 100%; height: auto; }}
.sm text {{ font-family: var(--f-mono); font-size: 9.5px; fill: var(--muted); }}
.sm text.ttl {{ font-family: var(--f-body); font-size: 12px; font-weight: 600; fill: var(--ink); }}
.sm .grid {{ stroke: var(--rule); stroke-width: 1; }}
.sm .axis {{ stroke: var(--muted); stroke-width: 1; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.88rem; margin-top: 10px; color: var(--ink-2); }}
.sw {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.tick {{ display: inline-block; width: 2px; height: 13px; background: var(--ink); margin-right: 7px; vertical-align: -2px; }}
.muted {{ color: var(--muted); }}
dl.defs {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 14px 0 0; }}
dl.defs dt {{ font-family: var(--f-mono); font-size: 0.88rem; color: var(--ink-2); }}
dl.defs dd {{ margin: 0; }}
a {{ color: var(--cam); }}
:focus-visible {{ outline: 2px solid var(--cam); outline-offset: 2px; }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">G-NTFields · camera follow · 2026-10-22</div>
  <h1>Constrained Movement</h1>
  <p class="lede col">{lede}</p>
  {status}
  {tiles}
  <p class="tiles-cap">NTFields success on the whole-path query (the bird flies t = 0 → L, the camera starts and ends in given orientations, forward mode), {seeds_note}, 1000 cases per scene; below each, how often the straight motion would do.</p>
</header>

<h2>The model</h2>
<div class="twocol">
  <figure class="diagram">{diagram}
    <figcaption>A pose is (t, R): t moves the bird, and with it the base centre, along the bird's path; R turns the pyramid about the base centre. The apex is the camera, D = 50 units from the anchor. A pose is allowed when the pyramid touches nothing in the scene.</figcaption>
  </figure>
  <figure><img src="img/concept_view.png" alt="What the camera sees from behind and above the bird in the forest, the pyramid's base outlined" loading="lazy">
    <figcaption>The camera's own view at one pose in the forest. The dashed rectangle is the base of the pyramid (48 × 22 units through the bird's centre); everything between the camera and that rectangle has to stay free.</figcaption>
  </figure>
</div>
<dl class="defs col">
  <dt>shape</dt><dd>The bird of the camera + object test, flying at a fixed heading. Its path is an RRT path of its centre (exact bird clearance at least 0.01), shortcut and smoothed; the camera's base rectangle (48 × 22 units) is centred on it, the camera 50 units away.</dd>
  <dt>poses</dt><dd>M = [0, L] × SO(3): arc length t of the bird's centre along its path × the pyramid's rotation. A 4-d slice of SE(3) with the usual metric dt² + |ω / 2π|².</dd>
  <dt>model input</dt><dd>(t − L/2, 0, 0, rotvec / 2π): the path unrolled onto the x axis, so the NTFields network and planner run unchanged; no xyz.</dd>
  <dt>training</dt><dd>800k pose pairs sampled along the path (pyramid clearance band 0.001–0.05), speeds from the exact pyramid clearance and its angular clearance; Eikonal residual with the gradient projected onto M; the bundle's metric NTFields, 8000 epochs.</dd>
  <dt>planning</dt><dd>MPPI on (t, R), forward mode: t never decreases (the bird does not fly backwards). Success = reaching the goal within 0.01 with the pyramid collision-free at every waypoint and 4 sub-steps between waypoints (exact geometry).</dd>
</dl>

<h2>Scenes and bird paths</h2>
<p class="col">The followed shape is the bird of the camera + object test (48 × 18 × 48 units), the camera the same 51° × 25° camera, now 50 units from the bird's centre. One RRT path per scene, picked from 4–24 candidates so the camera always has some room (at least 15 % of rotations free) while the straight camera motion mostly fails.</p>
{path_table}

<h2>Results</h2>
<p class="col">Forward mode, seed 0 unless more seeds are listed. Whole path: t from 0 to L with given start and end orientations; sub-paths: random t0 &lt; t1 at least 0.25 apart. 1000 RRT-verified cases per set.</p>
<div class="chart">{bars}</div>
<div class="legend"><span><i class="sw" style="background:var(--good)"></i>success</span><span><i class="sw" style="background:var(--crit)"></i>collision (camera hits the scene or the view is blocked)</span><span><i class="sw" style="background:var(--warn)"></i>stall (no collision, goal not reached)</span><span><i class="tick"></i>straight motion collision-free</span></div>
{table_forward}
{failure_profiles}
{ablation}
{epochs}
<h3>Free mode (t may move backwards)</h3>
{table_free}
{findings}

<h2>Per scene</h2>
{scenes}

<h2>Files</h2>
<ul class="col">
  <li>Code: <code>models/constrained_movement/</code> (geometry, renderer, scripts), tests <code>tests/test_constrained_movement.py</code>; training <code>train/train_metric_path.py</code>.</li>
  <li>Bird paths: <code>datasets/3dshape/paths/cm_&lt;scene&gt;_a.npz</code>; cells <code>datasets/3dshape/data/cm_&lt;scene&gt;_a</code>; test sets <code>testing_data/3dshape/cm_&lt;scene&gt;_a_{{full,seg}}</code>.</li>
  <li>Results: <code>results/constrained_movement/</code> (plans, <code>summary/summary.md</code>, figures); log <code>models/constrained_movement/EXPERIMENTS.md</code>.</li>
</ul>
</div>
'''


def main():
    lede = notes.get('lede') or ('A camera follows a bird through five scenes. The camera is a rectangle-based pyramid '
                                 'whose base is centred on the bird: it moves with the bird along the bird\'s RRT path '
                                 'and can only turn about the bird\'s centre. Planning happens in t + rotation, and the '
                                 'baseline is the original metric NTFields.')
    status = '<div class="status">{}</div>'.format(esc(args.status)) if args.status else ''
    ns = sorted({c['agg']['success']['n_seeds'] for k, c in S['cells'].items() if k.endswith('|full|forward')})
    seeds_note = 'seed 0' if ns in ([], [1]) else ('mean of seeds 0 and 1 (corridor: seed 0)' if ns == [1, 2]
                                                  else 'mean over seeds')
    page = PAGE.format(lede=esc(lede), status=status, tiles=tiles(), seeds_note=seeds_note, diagram=diagram(), path_table=path_table(),
                       bars=bars(), table_forward=results_table('forward'), ablation=ablation(),
                       failure_profiles=failure_profiles(), epochs=epochs_table(),
                       table_free=results_table('free') or '<p class="muted">Pending.</p>', findings=findings(),
                       scenes=scene_sections())
    with open(os.path.join(HERE, 'index.html'), 'w') as fh:
        fh.write(page)
    print('wrote', os.path.join(HERE, 'index.html'), len(page), 'bytes')


if __name__ == '__main__':
    main()
