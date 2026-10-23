"""Build the report page of the drone-pair link study: models/drone_link/report/index.html (+ img/).

Reads results/drone_link/summary/summary.json (scripts/aggregate_link.py), the per-case evaluations and baselines for
the distance breakdown, and copies results/drone_link/figures/*.png (scripts/make_figures.py) into img/.

    python3 models/drone_link/scripts/aggregate_link.py && python3 models/drone_link/report/build_report.py
"""
import html
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(REPO, 'results', 'drone_link')
SETS = os.path.join(REPO, 'testing_data', '3dshape')
SCENES = ('forest', 'windows', 'corridor', 'scaffold')
VARIANTS = ('std', 'cross')
SEEDS = (0, 1, 2)
BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 2.0))


def load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def pct(x, nd=1):
    return '–' if x is None else '{:.{}f}'.format(100 * x, nd)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


SUMMARY = load(os.path.join(RES, 'summary', 'summary.json'))
S, TRAIN = SUMMARY['sets'], SUMMARY.get('training', {})


# ----------------------------------------------------------------------------- data for the charts
def main_rows():
    rows = []
    for sc in SCENES:
        for v in VARIANTS:
            r = S.get('link_{}_{}'.format(sc, v), {})
            n, b = r.get('ntf', {}), r.get('baselines', {})
            per = n.get('success_per_seed') or []
            rows.append(dict(scene=sc, set=v, ntf=None if not per else 100 * float(np.mean(per)),
                             ntf_sd=None if len(per) < 2 else 100 * float(np.std(per)), seeds=len(per),
                             col=None if n.get('collision_rate_se3') is None else 100 * n['collision_rate_se3'],
                             stall=None if n.get('nonconv_rate_se3') is None else 100 * n['nonconv_rate_se3'],
                             ind=None if not b else 100 * b['independent']['valid_rate'],
                             dir=None if not b else 100 * b['straight']['success_rate']))
    return rows


def distance_curve():
    """Success % per start-goal distance bin (drone a), pooled over scenes, sets and evaluated seeds."""
    out = {k: [[0, 0] for _ in BINS] for k in ('ntf', 'dir', 'ind')}
    for sc in SCENES:
        for v in VARIANTS:
            s = 'link_{}_{}'.format(sc, v)
            ver = load(os.path.join(SETS, s, 'verification.json'))
            base = load(os.path.join(RES, 'baselines', s + '.json'))
            if not ver:
                continue
            dist = [c['dist_t'] for c in ver['cases']]
            b_of = lambda x: next(i for i, (lo, hi) in enumerate(BINS) if lo <= x < hi)  # noqa: E731
            for seed in SEEDS:
                ev = load(os.path.join(RES, 'plan', '{}_ntf_s{}'.format(sc, seed), s + '.json'))
                if ev:
                    for r in ev['records']:
                        i = b_of(dist[r['case']])
                        out['ntf'][i][0] += r['success_se3']
                        out['ntf'][i][1] += 1
            if base:
                for r in base['records']:
                    i = b_of(dist[r['case']])
                    out['dir'][i][0] += r['straight']['success']
                    out['dir'][i][1] += 1
                    if r['independent'].get('solved'):
                        out['ind'][i][0] += r['independent']['valid']
                        out['ind'][i][1] += 1
    return [dict(bin='{:.1f}–{:.1f}'.format(lo, hi) if hi < 2 else '≥ {:.1f}'.format(lo),
                 **{k: (None if out[k][i][1] < 20 else 100 * out[k][i][0] / out[k][i][1]) for k in out},
                 n=out['ntf'][i][1]) for i, (lo, hi) in enumerate(BINS)]


ROWS = main_rows()
DIST = distance_curve()
STD = [r for r in ROWS if r['set'] == 'std']
CROSS = [r for r in ROWS if r['set'] == 'cross']
ntf_std, dir_std = mean([r['ntf'] for r in STD]), mean([r['dir'] for r in STD])
ntf_cross, dir_cross = mean([r['ntf'] for r in CROSS]), mean([r['dir'] for r in CROSS])
ind_cross = mean([r['ind'] for r in CROSS])
los_cross = mean([S['link_{}_cross'.format(sc)].get('baselines', {}).get('independent', {}).get('rate_los_ever_blocked')
                  for sc in SCENES if 'link_{}_cross'.format(sc) in S])
short_cross = mean([S['link_{}_cross'.format(sc)]['ntf'].get('frac_success_using_short_link') for sc in SCENES])


def f1(x):
    return '–' if x is None else '{:.0f}'.format(x)


# ----------------------------------------------------------------------------- tables
def table(head, rows, cls=None):
    th = ''.join('<th>{}</th>'.format(h) for h in head)
    body = ''
    for row in rows:
        body += '<tr>' + ''.join('<td{}>{}</td>'.format('' if i == 0 else ' class="num"', c) for i, c in enumerate(row)) + '</tr>\n'
    return '<div class="tbl"><table{}><thead><tr>{}</tr></thead><tbody>\n{}</tbody></table></div>'.format(
        ' class="{}"'.format(cls) if cls else '', th, body)


def pm(per):
    per = [x for x in per if x is not None]
    if not per:
        return '–'
    return '{:.1f}'.format(100 * np.mean(per)) + ('' if len(per) < 2 else ' ± {:.1f}'.format(100 * np.std(per)))


main_tbl = table(['Scene · set', 'Direct motion', 'Independent drones', 'NTFields (7 DOF)', 'Collision', 'Stall',
                  'Seeds'],
                 [['{} · {}'.format(r['scene'], 'random' if r['set'] == 'std' else 'crossing'), f1(r['dir']), f1(r['ind']),
                   pm(S['link_{}_{}'.format(r['scene'], r['set'])]['ntf'].get('success_per_seed') or []),
                   f1(r['col']), f1(r['stall']), str(r['seeds'])] for r in ROWS])

contact_rows = []
for sc in SCENES:
    for v in VARIANTS:
        b = S.get('link_{}_{}'.format(sc, v), {}).get('baselines', {}).get('independent')
        if b:
            contact_rows.append(['{} · {}'.format(sc, 'random' if v == 'std' else 'crossing'), pct(b['mean_contact_frac'], 0),
                                 pct(b['rate_los_ever_blocked'], 0), pct(b['rate_ever_too_far'], 0),
                                 pct(b.get('rate_ever_too_close'), 0), pct(b['valid_rate'], 0)])
contact_tbl = table(['Scene · set', 'Contact, % of time', 'Line of sight lost, % of cases', 'Out of range, % of cases',
                     'Too close, % of cases', 'Link valid all the way, % of cases'], contact_rows)

var_rows = []
for sc in SCENES:
    for v in VARIANTS:
        r = S.get('link_{}_{}'.format(sc, v), {})
        if not r:
            continue
        lr = load(os.path.join(RES, 'plan', '{}_ntflr_s0'.format(sc), 'link_{}_{}.json'.format(sc, v)))
        var_rows.append(['{} · {}'.format(sc, 'random' if v == 'std' else 'crossing'),
                         pm(r['ntf'].get('success_per_seed') or []), pm(r.get('ntf4k') or []),
                         pct(r.get('ntfb')), pct((r.get('ntf_s0') or {}).get('success_rate_se3')),
                         pct((r.get('ntfsb') or {}).get('success_rate_se3')),
                         pct(lr['summary']['success_rate_se3']) if lr else '–',
                         pm((r.get('ntflr20_per_seed') or {}).get('success_rate_se3') or [])])
var_tbl = table(['Scene · set', '8000 epochs (seeds)', '4000 epochs (seeds)', 'Without TD / normal terms (seed 0)',
                 'Recipe, seed 0', 'Band on scene clearance (seed 0)', 'Long-range pairs, half (seed 0)',
                 'Long-range pairs, a fifth (seeds)'], var_rows)

SEED_ROWS = [dict(scene=sc, set=v,
                  ntf=[100 * x for x in (S.get('link_{}_{}'.format(sc, v), {}).get('ntf') or {}).get('success_per_seed') or []],
                  lr=[100 * x for x in (S.get('link_{}_{}'.format(sc, v), {}).get('ntflr20_per_seed') or {}).get('success_rate_se3')
                      or []]) for sc in SCENES for v in VARIANTS]
DATA = dict(main=ROWS, dist=DIST, seeds=SEED_ROWS)


# ----------------------------------------------------------------------------- page
CSS = """
/* Layout: one reading column (~70ch); figures and charts may run the full 1060px width. */
:root {
  --bg: #fbfcfd; --surface: #f1f4f7; --fg: #16202a; --muted: #56626e; --rule: #dce2e9; --accent: #2a78d6;
  --s-ntf: #2a78d6; --s-ind: #eb6834; --s-dir: #1baf7a; --s-lr: #eda100;
  --good: #1f7a4d; --bad: #c62f2f;
  --obs: #d9d2c7; --obs-edge: #8f8579; --link: #1f7a4d; --link-fill: rgba(31, 122, 77, 0.16); --drone-a: #2a78d6;
  --drone-b: #eb6834; --band: rgba(31, 122, 77, 0.18); --muted-line: #8a96a3; --clear: #c62f2f;
  --f-display: "Newsreader", Georgia, "Times New Roman", serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12171c; --surface: #1b2228; --fg: #e8ecf0; --muted: #9eabb7; --rule: #2c3640; --accent: #3987e5;
    --s-ntf: #3987e5; --s-ind: #d95926; --s-dir: #199e70; --s-lr: #c98500; --good: #4cc08a; --bad: #ef6b6b;
    --obs: #3a3631; --obs-edge: #8f8579; --link: #4cc08a; --link-fill: rgba(76, 192, 138, 0.18); --drone-a: #3987e5;
    --drone-b: #d95926; --band: rgba(76, 192, 138, 0.22); --muted-line: #73808c; --clear: #ef6b6b; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12171c; --surface: #1b2228; --fg: #e8ecf0; --muted: #9eabb7; --rule: #2c3640; --accent: #3987e5;
  --s-ntf: #3987e5; --s-ind: #d95926; --s-dir: #199e70; --s-lr: #c98500; --good: #4cc08a; --bad: #ef6b6b;
  --obs: #3a3631; --obs-edge: #8f8579; --link: #4cc08a; --link-fill: rgba(76, 192, 138, 0.18); --drone-a: #3987e5;
  --drone-b: #d95926; --band: rgba(76, 192, 138, 0.22); --muted-line: #73808c; --clear: #ef6b6b; color-scheme: dark;
}
body { background: var(--bg); color: var(--fg); font: 17px/1.6 var(--f-body); }
.wrap { max-width: 1060px; margin: 0 auto; padding-block: 40px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.col { max-width: 70ch; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; line-height: 1.2; text-wrap: balance; margin: 0; }
h1 { font-size: clamp(2rem, 4.5vw, 2.8rem); }
h2 { font-size: 1.6rem; margin-top: 56px; padding-top: 18px; border-top: 1px solid var(--rule); }
h3 { font-size: 1.2rem; margin-top: 28px; }
p, ul { margin: 12px 0 0; }
li + li { margin-top: 6px; }
.byline { color: var(--muted); font-size: 0.95rem; margin-top: 10px; }
.lede { font-size: 1.15rem; margin-top: 18px; }
code, .mono { font-family: var(--f-mono); font-size: 0.86em; }
code { background: var(--surface); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
.math { font-family: var(--f-mono); font-size: 0.9em; background: var(--surface); padding: 10px 14px; border-radius: 6px;
  overflow-x: auto; white-space: pre; margin-top: 12px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-top: 28px; }
.tile { background: var(--surface); border-radius: 8px; padding: 14px 16px; min-width: 0; }
.tile .lab { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.tile .pair { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: 1.45rem; margin-top: 6px; line-height: 1.2; }
.tile .pair span.vs { color: var(--muted); font-size: 0.9rem; margin: 0 6px; }
.tile .sub { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }
.k-ntf { color: var(--s-ntf); } .k-ind { color: var(--s-ind); } .k-dir { color: var(--s-dir); }
.tbl { overflow-x: auto; margin-top: 16px; }
table { border-collapse: collapse; font-size: 0.92rem; min-width: 100%; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font-weight: 600; color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; white-space: normal; vertical-align: bottom; }
td.num { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
figure { margin: 22px 0 0; }
figure img { display: block; width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 6px; background: #fff; }
figcaption { font-size: 0.9rem; color: var(--muted); margin-top: 8px; max-width: 80ch; }
.chart { margin-top: 18px; position: relative; overflow-x: auto; }
.chart svg { display: block; width: 100%; min-width: 560px; height: auto; }
.chart text { fill: var(--muted); font-family: var(--f-body); font-size: 12px; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .rowlab { fill: var(--fg); }
.legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 0.9rem; margin-top: 12px; color: var(--fg); }
.tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font-size: 0.82rem; padding: 6px 9px; border-radius: 5px; white-space: nowrap; }
.diagram { margin-top: 20px; overflow-x: auto; }
.diagram svg { display: block; width: 100%; min-width: 560px; height: auto; }
.diagram text { font-family: var(--f-body); font-size: 13px; fill: var(--fg); }
.diagram .m { fill: var(--muted); }
.diagram .c { fill: var(--clear); }
.note { background: var(--surface); border-radius: 8px; padding: 12px 16px; margin-top: 18px; font-size: 0.95rem; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""

JS = r"""
const SERIES = [
  { key: 'dir', name: 'direct 7-DOF motion', color: 'var(--s-dir)' },
  { key: 'ind', name: 'independent drones', color: 'var(--s-ind)' },
  { key: 'ntf', name: 'NTFields, 7 DOF', color: 'var(--s-ntf)' },
];
const SETNAME = { std: 'random', cross: 'crossing' };
const ns = 'http://www.w3.org/2000/svg';
function mk(svg, tag, attrs, text) { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; svg.appendChild(e); return e; }
function tipper(el) {
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; el.appendChild(tip);
  return { show(ev, html) { tip.innerHTML = html; tip.hidden = false; const b = el.getBoundingClientRect();
      let tx = ev.clientX - b.left + el.scrollLeft + 14; const ty = ev.clientY - b.top + 14;
      if (tx + tip.offsetWidth > el.scrollWidth) tx = ev.clientX - b.left + el.scrollLeft - tip.offsetWidth - 14;
      tip.style.left = tx + 'px'; tip.style.top = ty + 'px'; }, hide() { tip.hidden = true; } };
}
const fmt = v => v == null ? '–' : v.toFixed(1);
function dotplot(el, rows, label) {
  const lo = 0, hi = 100, W = 900, left = 170, right = 24, rowH = 24, top = 30, gapH = 12;
  let y = top, prev = null;
  rows.forEach(r => { if (prev !== null && r.scene !== prev) y += gapH; r._y = y; y += rowH; prev = r.scene; });
  const H = y + 30, x = v => left + (v - lo) / (hi - lo) * (W - left - right);
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label);
  for (let v = lo; v <= hi; v += 20) {
    mk(svg, 'line', { x1: x(v), x2: x(v), y1: top - 8, y2: H - 26, class: 'grid' });
    mk(svg, 'text', { x: x(v), y: H - 8, 'text-anchor': 'middle' }, v + '%');
  }
  mk(svg, 'text', { x: left, y: 14 }, 'cases solved with the link kept valid the whole way');
  prev = null;
  const tp = tipper(el);
  rows.forEach(r => {
    const cy = r._y + rowH / 2;
    mk(svg, 'text', { x: left - 12, y: cy + 4, 'text-anchor': 'end', class: r.scene !== prev ? 'rowlab' : '' }, (r.scene !== prev ? r.scene + ' · ' : '') + SETNAME[r.set]);
    prev = r.scene;
    const vals = SERIES.map(s => r[s.key]).filter(v => v != null);
    if (vals.length > 1) mk(svg, 'line', { x1: x(Math.min(...vals)), x2: x(Math.max(...vals)), y1: cy, y2: cy, stroke: 'var(--rule)', 'stroke-width': 2 });
    if (r.ntf != null && r.ntf_sd) mk(svg, 'line', { x1: x(Math.max(0, r.ntf - r.ntf_sd)), x2: x(Math.min(100, r.ntf + r.ntf_sd)), y1: cy, y2: cy, stroke: 'var(--s-ntf)', 'stroke-width': 2, 'stroke-opacity': 0.45 });
    SERIES.forEach(s => { if (r[s.key] != null) mk(svg, 'circle', { cx: x(r[s.key]), cy, r: 5.5, fill: s.color, stroke: 'var(--bg)', 'stroke-width': 2 }); });
    const hit = mk(svg, 'rect', { x: 0, y: r._y, width: W, height: rowH, fill: 'transparent' });
    hit.addEventListener('mousemove', ev => tp.show(ev, `<b>${r.scene}, ${SETNAME[r.set]} set</b><br>NTFields ${fmt(r.ntf)}${r.ntf_sd ? ' ± ' + r.ntf_sd.toFixed(1) : ''} (collision ${fmt(r.col)}, stall ${fmt(r.stall)})<br>independent drones ${fmt(r.ind)} · direct motion ${fmt(r.dir)}`));
    hit.addEventListener('mouseleave', tp.hide);
  });
  el.appendChild(svg);
}
function linechart(el, rows, label) {
  const W = 900, H = 330, left = 60, right = 24, top = 30, bottom = 46;
  const n = rows.length, x = i => left + (i + 0.5) / n * (W - left - right), y = v => top + (100 - v) / 100 * (H - top - bottom);
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label);
  for (let v = 0; v <= 100; v += 20) {
    mk(svg, 'line', { x1: left, x2: W - right, y1: y(v), y2: y(v), class: 'grid' });
    mk(svg, 'text', { x: left - 8, y: y(v) + 4, 'text-anchor': 'end' }, v + '%');
  }
  rows.forEach((r, i) => mk(svg, 'text', { x: x(i), y: H - 24, 'text-anchor': 'middle' }, r.bin));
  mk(svg, 'text', { x: (left + W - right) / 2, y: H - 6, 'text-anchor': 'middle' }, 'distance drone a travels from start to goal (normalized room units; the room is 0.90 × 0.63 × 0.99)');
  mk(svg, 'text', { x: left, y: 16 }, 'cases solved with the link kept valid the whole way');
  SERIES.forEach(s => {
    const pts = rows.map((r, i) => r[s.key] == null ? null : [x(i), y(r[s.key])]);
    let d = '';
    pts.forEach((p, i) => { if (p) d += (d && pts[i - 1] ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1); });
    mk(svg, 'path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2 });
    pts.forEach(p => { if (p) mk(svg, 'circle', { cx: p[0], cy: p[1], r: 4.5, fill: s.color, stroke: 'var(--bg)', 'stroke-width': 2 }); });
  });
  const labs = SERIES.filter(s => rows[0][s.key] != null).map(s => ({ s, y: y(rows[0][s.key]) - 8 })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < labs.length; i++) if (labs[i].y - labs[i - 1].y < 15) labs[i].y = labs[i - 1].y + 15;
  labs.forEach(l => mk(svg, 'text', { x: x(0) + 12, y: l.y, class: 'rowlab' }, l.s.name));
  const tp = tipper(el);
  rows.forEach((r, i) => {
    const hit = mk(svg, 'rect', { x: x(i) - (W - left - right) / n / 2, y: top, width: (W - left - right) / n, height: H - top - bottom, fill: 'transparent' });
    hit.addEventListener('mousemove', ev => tp.show(ev, `<b>distance ${r.bin}</b> (${r.n} NTFields plans)<br>NTFields ${fmt(r.ntf)} · independent drones ${fmt(r.ind)} · direct motion ${fmt(r.dir)}`));
    hit.addEventListener('mouseleave', tp.hide);
  });
  el.appendChild(svg);
}
const SEEDSER = [
  { key: 'ntf', name: 'recipe', color: 'var(--s-ntf)', dy: -7 },
  { key: 'lr', name: 'a fifth long-range', color: 'var(--s-lr)', dy: 7 },
];
function seedplot(el, rows, label) {
  const lo = 0, hi = 100, W = 900, left = 170, right = 24, rowH = 34, top = 30, gapH = 12;
  let y = top, prev = null;
  rows.forEach(r => { if (prev !== null && r.scene !== prev) y += gapH; r._y = y; y += rowH; prev = r.scene; });
  const H = y + 30, x = v => left + (v - lo) / (hi - lo) * (W - left - right);
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label);
  for (let v = lo; v <= hi; v += 20) {
    mk(svg, 'line', { x1: x(v), x2: x(v), y1: top - 8, y2: H - 26, class: 'grid' });
    mk(svg, 'text', { x: x(v), y: H - 8, 'text-anchor': 'middle' }, v + '%');
  }
  mk(svg, 'text', { x: left, y: 14 }, 'cases solved with the link kept valid the whole way, one dot per training seed');
  prev = null;
  const tp = tipper(el), list = v => v.length ? v.map(fmt).join(' / ') : '–';
  const avg = v => v.length ? fmt(v.reduce((a, b) => a + b, 0) / v.length) : '–';
  rows.forEach((r, i) => {
    const cy = r._y + rowH / 2;
    mk(svg, 'text', { x: left - 12, y: cy + 4, 'text-anchor': 'end', class: r.scene !== prev ? 'rowlab' : '' }, (r.scene !== prev ? r.scene + ' · ' : '') + SETNAME[r.set]);
    prev = r.scene;
    SEEDSER.forEach(s => {
      const v = r[s.key], yy = cy + s.dy;
      if (!v.length) return;
      const a = Math.min(...v), b = Math.max(...v);
      if (v.length > 1) mk(svg, 'line', { x1: x(a), x2: x(b), y1: yy, y2: yy, stroke: s.color, 'stroke-width': 2, 'stroke-opacity': 0.45 });
      v.forEach(val => mk(svg, 'circle', { cx: x(val), cy: yy, r: 5, fill: s.color, stroke: 'var(--bg)', 'stroke-width': 2 }));
      if (i === 0) {        // direct labels on the first row, beside the dots
        const right = x(b) + 10 + 7 * s.name.length < W;
        mk(svg, 'text', { x: right ? x(b) + 10 : x(a) - 10, y: yy + 4, 'text-anchor': right ? 'start' : 'end', class: 'rowlab' }, s.name);
      }
    });
    const hit = mk(svg, 'rect', { x: 0, y: r._y, width: W, height: rowH, fill: 'transparent' });
    hit.addEventListener('mousemove', ev => tp.show(ev, `<b>${r.scene}, ${SETNAME[r.set]} set</b><br>recipe, seeds 0 / 1 / 2: ${list(r.ntf)} (mean ${avg(r.ntf)})<br>a fifth long-range, seeds 0 / 1 / 2: ${list(r.lr)} (mean ${avg(r.lr)})`));
    hit.addEventListener('mouseleave', tp.hide);
  });
  el.appendChild(svg);
}
const chartSeeds = document.getElementById('chart-seeds');
if (chartSeeds) seedplot(chartSeeds, DATA.seeds, 'Dot plot of the share of cases solved per scene and test set, one dot per training seed, for the NTFields recipe and for a fifth of the training pairs long-range.');
dotplot(document.getElementById('chart-main'), DATA.main, 'Dot plot of the share of cases solved with the link valid all the way, per scene and test set, for NTFields, independent drones and the direct motion.');
linechart(document.getElementById('chart-dist'), DATA.dist, 'Line chart of the share of cases solved against the distance between start and goal, for NTFields, independent drones and the direct motion.');
"""


def page(body):
    head = ('<title>Stretchable Drone Link</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;'
            '6..72,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">\n')
    body = body.replace(' %', '&nbsp;%')        # keep a number and its unit on one line
    return head + '<style>' + CSS + '</style>\n' + body + '\n<script>\nconst DATA = ' + json.dumps(DATA) + ';\n' + JS + '</script>\n'


def copy_figures():
    src = os.path.join(RES, 'figures')
    dst = os.path.join(HERE, 'img')
    os.makedirs(dst, exist_ok=True)
    if os.path.isdir(src):
        for f in os.listdir(src):
            if f.endswith('.png'):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    return sorted(os.listdir(dst))


def diagram():
    """The link box between the drones, drone a's frame, the scene clearance and the allowed lengths (inline SVG)."""
    ax, ay, bx, by = 130.0, 230.0, 560.0, 140.0
    L = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    ux, uy = (bx - ax) / L, (by - ay) / L            # link direction R e_z
    nx, ny = uy, -ux                                 # in-plane normal (pointing up-right)
    hw = 9.0
    box = [(ax + nx * hw, ay + ny * hw), (bx + nx * hw, by + ny * hw), (bx - nx * hw, by - ny * hw), (ax - nx * hw, ay - ny * hw)]
    pts = ' '.join('{:.1f},{:.1f}'.format(*p) for p in box)
    # obstacle above the middle of the link, and the clearance segment to it
    mx, my = ax + 0.62 * (bx - ax), ay + 0.62 * (by - ay)
    sx, sy = mx + nx * hw, my + ny * hw              # closest link point
    ex, ey = sx + nx * 34, sy + ny * 34              # closest obstacle point
    ob = (ex - 70, ey - 74, 120, 74)
    ruler_y, r0, r1 = 300.0, 160.0, 760.0
    lmin, lmax = r0 + (r1 - r0) * 0.06 / 0.36, r0 + (r1 - r0) * 0.30 / 0.36
    lcur = r0 + (r1 - r0) * 0.21 / 0.36
    return """<div class="diagram" role="img" aria-label="Diagram: drone a with its frame, the link box along the frame's z axis to drone b at distance L, an obstacle near the box at clearance c_scene, and a ruler of lengths with the allowed band from L_min = 0.06 to L_max = 0.30.">
<svg viewBox="0 0 900 340">
  <defs><marker id="dg-ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--fg)"/></marker></defs>
  <rect x="{ox:.1f}" y="{oy:.1f}" width="{ow:.1f}" height="{oh:.1f}" rx="3" fill="var(--obs)" stroke="var(--obs-edge)"/>
  <text x="{otx:.1f}" y="{oty:.1f}" text-anchor="middle" class="m">obstacle</text>
  <polygon points="{pts}" fill="var(--link-fill)" stroke="var(--link)" stroke-width="1.6"/>
  <line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="var(--clear)" stroke-width="1.6" stroke-dasharray="4 3"/>
  <text x="{ctx:.1f}" y="{cty:.1f}" class="c">c_scene</text>
  <line x1="{ax:.1f}" y1="{ay:.1f}" x2="{fzx:.1f}" y2="{fzy:.1f}" stroke="var(--fg)" stroke-width="1.5" marker-end="url(#dg-ar)"/>
  <line x1="{ax:.1f}" y1="{ay:.1f}" x2="{fyx:.1f}" y2="{fyy:.1f}" stroke="var(--fg)" stroke-width="1.5" marker-end="url(#dg-ar)"/>
  <text x="{tzx:.1f}" y="{tzy:.1f}">R e<tspan baseline-shift="sub" font-size="10">z</tspan></text>
  <text x="{tyx:.1f}" y="{tyy:.1f}" class="m">R e<tspan baseline-shift="sub" font-size="10">y</tspan></text>
  <circle cx="{ax:.1f}" cy="{ay:.1f}" r="8" fill="var(--drone-a)" stroke="var(--bg)" stroke-width="2"/>
  <circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="var(--drone-b)" stroke="var(--bg)" stroke-width="2"/>
  <text x="{ax:.1f}" y="{aly:.1f}" text-anchor="middle">drone a = t</text>
  <text x="{blx:.1f}" y="{bly:.1f}">drone b = t + L R e<tspan baseline-shift="sub" font-size="10">z</tspan></text>
  <text x="{wx:.1f}" y="{wy:.1f}" class="m">width w = 0.02</text>
  <line x1="{r0:.1f}" y1="{ry:.1f}" x2="{r1:.1f}" y2="{ry:.1f}" stroke="var(--muted-line)"/>
  <rect x="{lmin:.1f}" y="{ryb:.1f}" width="{lw:.1f}" height="12" fill="var(--band)"/>
  <line x1="{lmin:.1f}" y1="{ryt:.1f}" x2="{lmin:.1f}" y2="{ryd:.1f}" stroke="var(--fg)" stroke-width="1.4"/>
  <line x1="{lmax:.1f}" y1="{ryt:.1f}" x2="{lmax:.1f}" y2="{ryd:.1f}" stroke="var(--fg)" stroke-width="1.4"/>
  <circle cx="{lcur:.1f}" cy="{ry:.1f}" r="5" fill="var(--link)"/>
  <text x="{lmin:.1f}" y="{rlab:.1f}" text-anchor="middle">L_min 0.06</text>
  <text x="{lmax:.1f}" y="{rlab:.1f}" text-anchor="middle">L_max 0.30</text>
  <text x="{lcur:.1f}" y="{rlab:.1f}" text-anchor="middle" class="m">L</text>
  <text x="{r0:.1f}" y="{rlab:.1f}" class="m">length</text>
  <text x="600" y="60">pose (t, R, L): 6 DOF of SE(3) + 1 length</text>
  <text x="600" y="84" class="m">clearance c = min(c_scene, L − L_min, L_max − L)</text>
  <text x="600" y="108" class="m">c ≥ 0: drones clear, line of sight clear,</text>
  <text x="600" y="128" class="m">length allowed</text>
</svg>
</div>""".format(
        ox=ob[0], oy=ob[1], ow=ob[2], oh=ob[3], otx=ob[0] + ob[2] / 2, oty=ob[1] + ob[3] / 2 + 4, pts=pts,
        sx=sx, sy=sy, ex=ex, ey=ey, ctx=ex + 10, cty=(sy + ey) / 2 + 4,
        ax=ax, ay=ay, fzx=ax + ux * 70, fzy=ay + uy * 70, fyx=ax + nx * 46, fyy=ay + ny * 46,
        tzx=ax + ux * 74 - 4, tzy=ay + uy * 74 + 22, tyx=ax + nx * 50 - 30, tyy=ay + ny * 50 - 6,
        bx=bx, by=by, aly=ay + 30, blx=bx + 14, bly=by + 26, wx=ax + 0.3 * (bx - ax) + 10, wy=ay + 0.3 * (by - ay) + 34,
        r0=r0, r1=r1, ry=ruler_y, ryb=ruler_y - 6, ryt=ruler_y - 12, ryd=ruler_y + 12, lmin=lmin, lmax=lmax,
        lw=lmax - lmin, lcur=lcur, rlab=ruler_y + 30)


def fig(name, alt, cap):
    if not os.path.exists(os.path.join(HERE, 'img', name)):
        return ''
    return '<figure><img src="img/{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(name, html.escape(alt), cap)


def body():
    sel = lambda sc, v, *k: (lambda d: d)(S.get('link_{}_{}'.format(sc, v), {}))  # noqa: E731
    pairs_long = 3.0          # % of training pairs whose drone-a move exceeds 0.7 (link_forest, measured)
    ind_std = mean([r['ind'] for r in STD])
    lede = ('Two drones fly through a scene and must keep their radio link: see each other and stay in range. '
            'We plan them as one body, a box from drone a to drone b that stretches and shrinks between 0.06 and 0.30 '
            '(room units), so a pose is drone a\'s SE(3) pose plus one length. Regular metric NTFields, trained on '
            'speeds re-sampled in these 7 DOF, solves {} % of random link queries with the link valid the whole way, '
            'against {} % for independent drones and {} % for the direct motion. On the long crossing tasks it '
            'reaches only {} %: on long queries the planner stalls in local minima of the learned field.').format(
                f1(ntf_std), f1(ind_std), f1(dir_std), f1(ntf_cross))
    lr = ABL.get('ntflr') or {}
    if LR20_DONE and L20_GAIN:
        their = 'their' if len(L20_GAIN) > 1 else 'its'
        lede += (' Making a fifth of the training pairs span the room fixes much of that in the {}, in all {} seeds: '
                 '{} crossings rise from {}, {} random queries from {}.').format(
            and_list(L20_GAIN), WORDS[len(L20_SEEDS)], their,
            ' and from '.join('{} to {} %'.format(f1(MREC(sc, 'cross')), f1(M20(sc, 'cross'))) for sc in L20_GAIN), their,
            ' and from '.join('{} to {} %'.format(f1(MREC(sc, 'std')), f1(M20(sc, 'std'))) for sc in L20_GAIN))
        for sc in L20_MIXED:
            lede += ' In the {} it {} (random {} → {} %, crossings {} → {} %).'.format(
                sc, 'trades random queries for crossings' if M20(sc, 'std') < MREC(sc, 'std') else 'helps less',
                f1(MREC(sc, 'std')), f1(M20(sc, 'std')), f1(MREC(sc, 'cross')), f1(M20(sc, 'cross')))
        for sc in L20_UNSTABLE:
            worst = max(x for v in VARIANTS for s_, x in zip(L20_SEEDS, seeds_of(sc, v, 'ntflr20')) if s_ in L20_BROKEN[sc])
            lede += (' In the {} scene training fails in {} of the {} seeds (at most {} % solved), so the share is not '
                     'yet safe to make part of the recipe.').format(sc, WORDS[len(L20_BROKEN[sc])], WORDS[len(L20_SEEDS)],
                                                                    pct(worst))
    elif ('corridor', 'cross') in lr and LR_DOWN:
        lede += (' Training pairs that span the room lift the corridor\'s crossings from {} to {} % but break the '
                 'field in the {} scenes.').format(pct(lr[('corridor', 'cross')][1], 0), pct(lr[('corridor', 'cross')][0], 0),
                                        and_list(LR_DOWN))
    tiles = """
<div class="tiles" aria-label="Headline results">
  <div class="tile"><div class="lab">Random queries</div><div class="pair"><span class="k-ntf">{a}</span><span class="vs">vs</span><span class="k-ind">{b}</span></div><div class="sub">% solved with the link valid all the way: NTFields vs independent drones, mean of 4 scenes</div></div>
  <div class="tile"><div class="lab">Crossing tasks</div><div class="pair"><span class="k-ntf">{c}</span><span class="vs">vs</span><span class="k-ind">{d}</span></div><div class="sub">the same, on each scene's crossing task (long, through the obstacles)</div></div>
  <div class="tile"><div class="lab">Line of sight lost</div><div class="pair"><span class="k-ind">{e}</span></div><div class="sub">% of crossing cases where independent drones lose sight of each other at least once</div></div>
  <div class="tile"><div class="lab">Length used</div><div class="pair"><span class="k-ntf">{f}</span></div><div class="sub">% of solved crossings where NTFields shortens the link below both endpoint lengths</div></div>
</div>
<div class="legend" aria-hidden="true">
  <span><span class="sw" style="background:var(--s-ntf)"></span>NTFields, 7 DOF (link planned as one body)</span>
  <span><span class="sw" style="background:var(--s-ind)"></span>independent drones (each on its shortest path)</span>
  <span><span class="sw" style="background:var(--s-dir)"></span>direct 7-DOF motion</span>
</div>""".format(a=f1(ntf_std), b=f1(ind_std), c=f1(ntf_cross), d=f1(ind_cross), e=f1(None if los_cross is None else 100 * los_cross),
                 f=f1(None if short_cross is None else 100 * short_cross))
    return """<div class="wrap">
<header class="col">
  <h1>Stretchable Drone Link</h1>
  <p class="byline">A box that stretches along one axis, planned in 7 DOF with metric NTFields · results of 2026-10-22 · repo <code>models/drone_link/</code></p>
  <p class="lede">{lede}</p>
</header>
{tiles}

<section>
<h2>The shape</h2>
<div class="col">
<p>The link is the box <i>[−w/2, w/2]² × [0, L]</i> in its own frame. Drone a sits at the body origin, the centre of the end face <i>z</i> = 0; drone b sits at <i>(0, 0, L)</i>, the centre of the far face. A pose is <i>(t, R, L)</i>: <i>t</i> is drone a's position, <i>R</i> turns <i>e<sub>z</sub></i> onto the direction from a to b, and <i>L</i> = |b − a| is the length from a outwards. The SE(3) part embeds drone a and the direction b − a; the seventh coordinate is the length.</p>
<p>The box has width <i>w</i> = 0.02 (4.4 room units, a drone's size and the width of the line-of-sight beam). The length is limited to <i>[0.06, 0.30]</i>: the drones stay at least three widths apart and within radio range. If the box never touches the scene, neither drone hits anything and nothing ever blocks their line of sight. The limits act as walls of the seventh coordinate, exactly as the room walls do for the translation, and the planner also clamps <i>L</i> into them.</p>
</div>
{diagram}
</section>

<section>
<h2>The speed, re-sampled in 7 DOF</h2>
<div class="col">
<p>The networks read the chart <code>x = (t, rotvec(R) / 2π, L)</code>: the SE(3) chart used by every model in this repository plus the length in room units, so stretching the link by d<i>L</i> costs as much as moving drone a by d<i>L</i>. The clearance of a pose is exact (oriented box against axis-aligned boxes) and folds in the length limits:</p>
</div>
<div class="math">c       = min(c_scene(t, R, L), L − L_min, L_max − L)
s_dist  = clip(c / 0.05, 0.02, 1)
s_angle = smallest angle about drone a between a link point and a scene point
          in the same radial shell (shells of the box of length L), divided by π
speed   = (s_dist + s_angle) / 2            the speed metric NTFields trains on</div>
<div class="col">
<p>These are the formulas of the existing SE(3) preprocessing, with the box of each pose's own length. The clearance normal, used by NTFields' normal-alignment term, is the 7-d gradient of <i>c</i>; its length component is <i>ζ n·Re<sub>z</sub></i> for a witness point at fraction <i>ζ</i> of the box (checked against finite differences). Per scene, 800k training pairs: <i>x0</i> uniform (drone a in the room, <i>R</i> uniform, <i>L</i> uniform) inside the band 0.001 &lt; <i>c</i> &lt; 0.05, and <i>x1</i> = <i>x0</i> moved in the 6-d SE(3) chart as in the existing sampler, with a fresh uniform length. 28–44 % of the band samples sit at a length limit rather than near an obstacle.</p>
<p>The model is the original metric NTFields, unchanged except that its input is 7-d: the full recipe of the camera test (TD and normal-alignment terms 1e-3), 8000 epochs, three seeds. Planning uses the same MPPI as every SE(3) evaluation here (50 samples, horizon 5, step 0.015), with <i>L</i> clamped to its limits.</p>
</div>
</section>

<section>
<h2>Four scenes</h2>
<div class="col">
<p>All four use the room of env3 (198 × 138 × 218 units). Each breaks the line of sight between two drones in a different way.</p>
<ul>
  <li><b>forest</b>: the forest from the camera test, unchanged. Four trees with 12-unit trunks and 30-unit crowns between heights 90 and 120.</li>
  <li><b>windows</b>: a wall across the room with two 40 × 40 windows, one low and one high. The two rooms see each other only through the windows.</li>
  <li><b>corridor</b>: two full-height walls make a serpentine corridor of three 67-unit legs. Line of sight has to go around the corners.</li>
  <li><b>scaffold</b>: a lattice of 3 × 3 columns and beams across the room at two heights.</li>
</ul>
</div>
{fig_scenes}
<div class="col">
<p>Two test sets per scene, 1000 cases each, both ends at least 0.02 from the scene and from the length limits. <b>Random</b> draws both poses uniformly. <b>Crossing</b> is the scene's signature task: both drones start in one region and end in the other (in front of the trees and behind them; front room and back room; first leg and last leg; the two sides of the lattice). Every case is certified solvable in the 7-d space with the link kept valid: a 60k-node roadmap per scene, built on the GPU with the exact clearance, solved all 8000 cases. The certified paths are the proof that a link-keeping plan exists, not a competitor.</p>
<p><b>Independent drones</b> is the link-unaware baseline. Each drone flies its own shortest collision-free path; both leave together and arrive together. We then score the link along the way. The <b>direct motion</b> moves drone a in a straight line, slerps the orientation and interpolates the length.</p>
</div>
</section>

<section>
<h2>Results</h2>
<div class="col"><p>Share of cases solved with the link valid the whole way: the drones reach their goals, never touch the scene, always see each other and keep an allowed distance. NTFields is the mean of {nseeds} with a ± one-standard-deviation whisker. Hover a row for the failure split.</p></div>
<div class="legend" aria-hidden="true"><span><span class="sw" style="background:var(--s-ntf)"></span>NTFields, 7 DOF</span><span><span class="sw" style="background:var(--s-ind)"></span>independent drones</span><span><span class="sw" style="background:var(--s-dir)"></span>direct 7-DOF motion</span></div>
<div class="chart" id="chart-main"></div>
{main_tbl}
{results_notes}
</section>

<section>
<h2>Signal contact without the link</h2>
<div class="col"><p>How independent shortest paths lose the link. Contact means the drones see each other within range; the last column also requires the minimum separation.</p></div>
{contact_tbl}
{contact_notes}
</section>

<section>
<h2>What the plans look like</h2>
<div class="col"><p>One crossing case per scene that NTFields solves, chosen where the independent drones keep the link for the shortest time. Left: the NTFields plan, with the link drawn at evenly spaced moments (drone a blue, drone b orange). Right: the independent drones, with the link red wherever the drones cannot see each other. Below: the link length over time against the allowed band, and the contact timeline of both.</p></div>
{showcases}
</section>

<section>
<h2>Why NTFields fails the crossings</h2>
<div class="col"><p>Success falls with the distance drone a has to travel, pooled over scenes, sets and seeds. Hover a point for the numbers.</p></div>
<div class="legend" aria-hidden="true"><span><span class="sw" style="background:var(--s-ntf)"></span>NTFields, 7 DOF</span><span><span class="sw" style="background:var(--s-ind)"></span>independent drones</span><span><span class="sw" style="background:var(--s-dir)"></span>direct 7-DOF motion</span></div>
<div class="chart" id="chart-dist"></div>
{why_notes}
</section>

<section>
<h2>Ablations</h2>
<div class="col"><p>The variant that changes the most, a fifth of the training pairs long-range, against the recipe: one dot per training seed, the same cases for both. Hover a row for the numbers; the table below lists every variant.</p></div>
<div class="legend" aria-hidden="true"><span><span class="sw" style="background:var(--s-ntf)"></span>NTFields recipe</span><span><span class="sw" style="background:var(--s-lr)"></span>a fifth of the training pairs long-range</span></div>
<div class="chart" id="chart-seeds"></div>
{var_tbl}
{ablation_notes}
</section>

<section>
<h2>Next steps</h2>
{next_steps}
</section>

<section>
<h2>Files</h2>
<div class="col">
<ul>
  <li>Code: <code>models/drone_link/</code> (geometry, scripts, README, experiment log), scenes <code>dataprocessing/make_link_scenes.py</code>, trainer <code>train/train_metric_link.py</code>, tests <code>tests/test_drone_link.py</code> (8 tests).</li>
  <li>Data: <code>datasets/3dshape/data/link_&lt;scene&gt;</code>, test sets <code>testing_data/3dshape/link_&lt;scene&gt;_{{std,cross}}</code>, runs <code>Experiments/drone_link/</code>, results <code>results/drone_link/</code> (summary in <code>summary/summary.md</code>).</li>
</ul>
</div>
</section>
</div>""".format(lede=lede, tiles=tiles, diagram=diagram(), main_tbl=main_tbl, contact_tbl=contact_tbl, var_tbl=var_tbl,
                 nseeds=NSEEDS_TEXT,
                 fig_scenes=fig('scenes.png', 'The four scenes in the same room: forest (four trees), windows (a wall with two windows), corridor (two walls forming a serpentine), scaffold (a lattice of columns and beams), each with a few example link poses drawn as green segments between a blue and an orange drone.',
                                'The four scenes, each with a few start poses of its crossing set (green: the link; blue: drone a; orange: drone b).'),
                 showcases=showcases(), results_notes=RESULTS_NOTES, contact_notes=CONTACT_NOTES, why_notes=WHY_NOTES,
                 ablation_notes=ABLATION_NOTES, next_steps=NEXT_STEPS)


# ----------------------------------------------------------------------------- data-driven notes
def runs_of(sc, v, kind='ntf'):
    out = []
    for seed in SEEDS:
        ev = load(os.path.join(RES, 'plan', '{}_{}_s{}'.format(sc, kind, seed), 'link_{}_{}.json'.format(sc, v)))
        if ev:
            out.append(ev)
    return out


def failure_split(sc, v):
    recs = [r for ev in runs_of(sc, v) for r in ev['records']]
    fail = [r for r in recs if not r['success_se3']]
    if not fail:
        return None
    return dict(stall=float(np.mean([(not r['converged_se3']) and not r['collision_se3'] for r in fail])),
                coll=float(np.mean([r['converged_se3'] and r['collision_se3'] for r in fail])),
                both=float(np.mean([(not r['converged_se3']) and r['collision_se3'] for r in fail])))


def rng_txt(vals, nd=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return '–'
    lo, hi = ('{:.{n}f}'.format(x, n=nd) for x in (min(vals), max(vals)))
    return lo if lo == hi else '{}–{}'.format(lo, hi)


nseeds = sorted({r['seeds'] for r in ROWS if r['seeds']})
NSEEDS_TEXT = ('{} seeds'.format(nseeds[0]) if len(nseeds) == 1 else '{}–{} seeds'.format(nseeds[0], nseeds[-1])) if nseeds else 'the seeds'
row = {(r['scene'], r['set']): r for r in ROWS}
gap_ind = [r['ntf'] - r['ind'] for r in STD if r['ntf'] is not None and r['ind'] is not None]
gap_dir = [r['ntf'] - r['dir'] for r in STD if r['ntf'] is not None and r['dir'] is not None]
cross_vals = {r['scene']: r['ntf'] for r in CROSS if r['ntf'] is not None}
splits = {(sc, v): failure_split(sc, v) for sc in SCENES for v in VARIANTS}
shrink_all = [S['link_{}_{}'.format(sc, v)]['ntf'].get('frac_success_using_short_link') for sc in SCENES for v in VARIANTS
              if 'link_{}_{}'.format(sc, v) in S]

RESULTS_NOTES = '<div class="col"><ul>\n'
RESULTS_NOTES += ('<li><b>Random queries.</b> NTFields solves {} % of the cases in each scene: {} points more than independent '
                  'drones and {} more than the direct motion.</li>\n').format(
    rng_txt([r['ntf'] for r in STD]), rng_txt(gap_ind), rng_txt(gap_dir))
if cross_vals:
    lo_sc, hi_sc = min(cross_vals, key=cross_vals.get), max(cross_vals, key=cross_vals.get)
    RESULTS_NOTES += ('<li><b>Crossings.</b> NTFields drops to {} % ({} {:.0f} % to {} {:.0f} %). '
                      'Where the line of sight must go around corners it still beats ignoring the link by far '
                      '(corridor: {} % against {} % for independent drones and {} % for the direct motion). In the '
                      'open forest the independent drones do better ({} % against {} %): the drones rarely need '
                      'each other there, and NTFields stalls on the long trips.</li>\n').format(
        rng_txt(list(cross_vals.values())), lo_sc, cross_vals[lo_sc], hi_sc, cross_vals[hi_sc],
        f1(row[('corridor', 'cross')]['ntf']), f1(row[('corridor', 'cross')]['ind']), f1(row[('corridor', 'cross')]['dir']),
        f1(row[('forest', 'cross')]['ind']), f1(row[('forest', 'cross')]['ntf']))
sp = [x for x in splits.values() if x]
if sp:
    RESULTS_NOTES += ('<li><b>How it fails.</b> A collision is a plan whose link crosses an obstacle somewhere; a stall is '
                      'a plan that does not converge within 600 steps. They overlap: {} % of all failures are stalls '
                      'without a collision (most of the forest\'s), {} % cross an obstacle and then stall (most of the '
                      'corridor\'s: through a partition wall), {} % reach the goal through an obstacle (windows: '
                      'through a window frame).</li>\n').format(
        f1(100 * np.mean([x['stall'] for x in sp])), f1(100 * np.mean([x['both'] for x in sp])),
        f1(100 * np.mean([x['coll'] for x in sp])))
RESULTS_NOTES += ('<li><b>It uses the length.</b> In {} % of the solved cases the link gets shorter than at both ends by more '
                  'than 0.02 somewhere along the way, most often to squeeze past an obstacle or a window frame.</li>\n').format(
    rng_txt([100 * x for x in shrink_all if x is not None]))
RESULTS_NOTES += '</ul></div>'

ib = [S['link_{}_{}'.format(sc, v)].get('baselines', {}).get('independent') for sc in SCENES for v in VARIANTS
      if 'link_{}_{}'.format(sc, v) in S]
ib = [x for x in ib if x]
CONTACT_NOTES = '<div class="col"><ul>\n'
if ib:
    CONTACT_NOTES += ('<li>On average, independent drones keep contact {} % of the time, which looks fine. But they '
                      'lose sight of each other at least once in {} % of the cases, and come closer than the minimum '
                      'separation in {} % of them. Range matters mostly in the windows scene, where the two drones '
                      'often pick different windows (out of range in {} % of its crossings).</li>\n').format(
        rng_txt([100 * x['mean_contact_frac'] for x in ib]), rng_txt([100 * x['rate_los_ever_blocked'] for x in ib]),
        rng_txt([100 * x['rate_ever_too_close'] for x in ib if x.get('rate_ever_too_close') is not None]),
        pct(S.get('link_windows_cross', {}).get('baselines', {}).get('independent', {}).get('rate_ever_too_far'), 0))
CONTACT_NOTES += ('<li>A plan NTFields completes keeps contact 100 % of the time by construction, since the link is part '
                  'of the body it plans. Its failed plans are not a usable alternative: they cross an obstacle or stop '
                  'short, so the planner should report failure there rather than fly them.</li>\n')
CONTACT_NOTES += ('<li>The certified roadmap paths exist for every case: a link-keeping motion was always available. '
                  'Their drone paths are {} longer than the independent shortest paths (median, crossing sets): '
                  'that is the price of the link.</li>\n').format(
    rng_txt([100 * (S['link_{}_cross'.format(sc)]['baselines']['rrt']['median_len_a']
                    / S['link_{}_cross'.format(sc)]['baselines']['independent']['median_len_a'] - 1)
             for sc in SCENES if S.get('link_{}_cross'.format(sc), {}).get('baselines')]) + ' %')
CONTACT_NOTES += '</ul></div>'

def showcases():
    out = ''
    for sc in SCENES:
        if not os.path.exists(os.path.join(HERE, 'img', 'showcase_{}.png'.format(sc))):
            continue
        out += '<h3>{}</h3>'.format(sc) + fig(
        'showcase_{}.png'.format(sc),
            'Two 3-D views of the {} scene with one crossing case: the NTFields plan with green link segments, and the '
            'independent drones with red link segments where they lose sight; below, the link length over time and the '
            'contact timelines.'.format(sc), SHOWCASE_CAPTIONS.get(sc, 'Crossing case in the {} scene.'.format(sc)))
    return out


SHOWCASE_CAPTIONS = {
    'forest': 'Forest. The NTFields plan keeps the link clear of trunks and crowns all the way. The independent drones '
              'fly one under and one over the canopy, and the crowns come between them twice.',
    'windows': 'Windows. NTFields shortens the link from 0.27 to about 0.13 while the pair passes the wall, then '
               'stretches it again. The independent drones drift apart (up to 0.42, beyond the 0.30 range) and lose '
               'each other for most of the crossing.',
    'corridor': 'Corridor. Both plans go around the two walls. NTFields shortens the link to about 0.09 and keeps both '
                'drones on the same side of every corner; the independent drones cut each corner separately, so a wall '
                'comes between them twice.',
    'scaffold': 'Scaffold. NTFields squeezes the link down to its 0.06 minimum to thread the lattice. The independent '
                'drones lose sight of each other behind columns and beams for more than half of the flight.',
}

DIAG = {sc: load(os.path.join(RES, 'diagnostics', 'field_{}_ntf_s0_link_{}_cross.json'.format(sc, sc))) for sc in SCENES}
MPPI = {c: load(os.path.join(RES, 'diagnostics', 'mppi_forest_ntf_s0_link_forest_cross_{}.json'.format(c)))
        for c in ('50_5', '200_5', '200_15')}


def stall_progress(sc, v):
    """Median share of the start-goal translation covered by the stalled plans (seed 0 paths)."""
    p = os.path.join(RES, 'plan', '{}_ntf_s0'.format(sc), 'link_{}_{}'.format(sc, v))
    if not os.path.exists(p + '_paths.npz'):
        return None
    ev, z = load(p + '.json'), np.load(p + '_paths.npz')
    X = np.load(os.path.join(SETS, 'link_{}_{}'.format(sc, v), 'sampled_points.npy'))
    rec = {r['case']: r for r in ev['records']}
    prog = []
    for j, c in enumerate(z['case']):
        if rec[int(c)]['converged_se3']:
            continue
        P = z['path'][j]
        P = P[~np.isnan(P[:, 0])]
        d0 = np.linalg.norm(X[c, 7:10] - X[c, :3])
        prog.append(1 - np.linalg.norm(X[c, 7:10] - P[-1, :3]) / max(d0, 1e-6))
    return float(np.median(prog)) if prog else None


def ablation(kind):
    """{(scene, set): success} of a seed-0 ablation run, and the matching seed-0 recipe numbers."""
    out = {}
    for sc in SCENES:
        for v in VARIANTS:
            ev = load(os.path.join(RES, 'plan', '{}_{}_s0'.format(sc, kind), 'link_{}_{}.json'.format(sc, v)))
            base = load(os.path.join(RES, 'plan', '{}_ntf_s0'.format(sc), 'link_{}_{}.json'.format(sc, v)))
            if ev and base:
                out[(sc, v)] = (ev['summary']['success_rate_se3'], base['summary']['success_rate_se3'])
    return out


WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}


def poss(sc):
    return sc + ("'" if sc.endswith('s') else "'s")


def and_list(xs):
    return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + ' and ' + xs[-1]


ABL = {k: ablation(k) for k in ('ntfb', 'ntfsb', 'ntflr', 'ntflr20')}
LR_DOWN = [sc for sc in SCENES if (sc, 'std') in ABL['ntflr'] and ABL['ntflr'][(sc, 'std')][0] < ABL['ntflr'][(sc, 'std')][1] - 0.1]
LR_POINTER = ''


def seeds_of(sc, v, kind='ntf', key='success_rate_se3'):
    """Per-seed values (seed order) of the recipe ('ntf', success only) or of a fifth long-range ('ntflr20')."""
    d = S.get('link_{}_{}'.format(sc, v), {})
    if kind == 'ntf':
        return (d.get('ntf') or {}).get('success_per_seed') or []
    return (d.get('ntflr20_per_seed') or {}).get(key) or []


def m100(xs):
    return None if not xs else 100 * float(np.mean(xs))


def escapes(kind, scenes, seeds):
    """First epochs below the loss plateau (aggregate_link.PLATEAU) of the runs that left it, and the runs that never did."""
    runs = [TRAIN.get('{}_{}_s{}'.format(sc, kind, s)) for sc in scenes for s in seeds]
    return [r['escape'] for r in runs if r and r.get('escape') is not None], \
        [r for r in runs if r and 'escape' in r and r['escape'] is None]


L20_SEEDS = sorted(set.intersection(*[set((S.get('link_{}_{}'.format(sc, v), {}).get('ntflr20_per_seed') or {}).get('seeds')
                                          or []) for sc in SCENES for v in VARIANTS]))
LR20_DONE = L20_SEEDS == list(SEEDS)
L20_BROKEN = {sc: [s for s, x in zip(L20_SEEDS, seeds_of(sc, 'std', 'ntflr20')) if x < 0.1] for sc in SCENES}
L20_UNSTABLE = [sc for sc in SCENES if LR20_DONE and L20_BROKEN[sc]]
L20_GAIN = [sc for sc in SCENES if LR20_DONE and sc not in L20_UNSTABLE
            and all(min(seeds_of(sc, v, 'ntflr20')) > max(seeds_of(sc, v)) for v in VARIANTS)]
L20_MIXED = [sc for sc in SCENES if LR20_DONE and sc not in L20_UNSTABLE + L20_GAIN]
M20 = lambda sc, v: m100(seeds_of(sc, v, 'ntflr20'))  # noqa: E731
MREC = lambda sc, v: m100(seeds_of(sc, v))  # noqa: E731
if LR20_DONE and L20_GAIN:
    LR_POINTER = ('Making a fifth of them long-range lifts the crossings in the {} ({} seeds each); making half of them '
                  'long-range breaks the field in the {} scenes (ablations below).').format(
        ', in the '.join('{} from {} to {} %'.format(sc, f1(MREC(sc, 'cross')), f1(M20(sc, 'cross'))) for sc in L20_GAIN),
        WORDS[len(L20_SEEDS)], and_list(LR_DOWN))
elif ('corridor', 'cross') in ABL['ntflr']:
    LR_POINTER = ('Making half of them long-range (ablations below) lifts the corridor\'s crossings from {} to {} %{}.').format(
        pct(ABL['ntflr'][('corridor', 'cross')][1], 0), pct(ABL['ntflr'][('corridor', 'cross')][0], 0),
        ' but breaks the field in the {} scenes'.format(and_list(LR_DOWN)) if LR_DOWN else '')

WHY_NOTES = '<div class="col"><ul>\n'
fp = stall_progress('forest', 'cross')
if fp is not None:
    WHY_NOTES += ('<li><b>Stalls, not slow plans.</b> A stalled forest crossing covers a median {} % of the distance to '
                  'its goal and then jitters in place for the rest of its 600 steps: the planner sits in a local minimum '
                  'of the learned travel time.</li>\n').format(f1(100 * fp))
d_f, d_s = DIAG.get('forest'), DIAG.get('scaffold')
d_c, d_w = DIAG.get('corridor'), DIAG.get('windows')
if d_f and d_s and d_c and d_w:
    s_ = lambda d, k, key: 100 * d['summary'][k][key]  # noqa: E731
    WHY_NOTES += ('<li><b>The field is right on average but bumpy.</b> Along each case\'s certified path, the learned '
                  'travel time tracks the remaining path length (correlation {}). In the forest and the scaffold, '
                  'stalled cases have {} % and {} % of the steps along their certified path going uphill, against '
                  '{} % and {} % for solved cases: bumps a local descent cannot cross. In the corridor and the '
                  'windows the field along the valid route is as smooth for failed cases as for solved ones ({} % '
                  'and {} % uphill); there the descent leaves the route through a wall, where the field leaks, and '
                  'stalls on the other side.</li>\n').format(
        rng_txt([d['summary'][k]['corr'] for d in (d_f, d_s, d_c, d_w) for k in ('solved', 'stalled')], 2),
        f1(s_(d_f, 'stalled', 'frac_up')), f1(s_(d_s, 'stalled', 'frac_up')), f1(s_(d_f, 'solved', 'frac_up')),
        f1(s_(d_s, 'solved', 'frac_up')), f1(s_(d_c, 'stalled', 'frac_up')), f1(s_(d_w, 'stalled', 'frac_up')))
    tp = [d['summary']['training_pairs'] for d in (d_f, d_s, d_c, d_w)]
    WHY_NOTES += ('<li><b>The training pairs barely reach crossing distances.</b> Drone a moves a median {} between '
                  'the two poses of a training pair, and more than 0.7 in only {} % of the pairs: the SE(3) '
                  'displacement of the sampler spends half its length on rotation, and long moves leave the room. '
                  'The crossings need 0.7–1.2. {}</li>\n').format(
        rng_txt([x['drone_a_move_median'] for x in tp], 2), rng_txt([100 * x['drone_a_move_gt_0.7'] for x in tp], 1),
        LR_POINTER)
if all(MPPI.values()):
    WHY_NOTES += ('<li><b>The planner is not the bottleneck.</b> On 200 forest crossings, four times the MPPI samples '
                  'and three times the horizon move success from {} % to {} %.</li>\n').format(
        pct(MPPI['50_5']['summary']['success_rate_se3']), pct(MPPI['200_15']['summary']['success_rate_se3']))
k4c = mean([mean(S['link_{}_cross'.format(sc)].get('ntf4k') or [None]) for sc in SCENES])
k8c = mean([mean(S['link_{}_cross'.format(sc)]['ntf'].get('success_per_seed') or [None]) for sc in SCENES])
k4s = mean([mean(S['link_{}_std'.format(sc)].get('ntf4k') or [None]) for sc in SCENES if 'link_{}_std'.format(sc) in S])
k8s = mean([mean(S['link_{}_std'.format(sc)]['ntf'].get('success_per_seed') or [None]) for sc in SCENES if 'link_{}_std'.format(sc) in S])
if None not in (k4c, k8c, k4s, k8s):
    WHY_NOTES += ('<li><b>Training longer helps the long queries most.</b> From 4000 to 8000 epochs the crossings go '
                  'from {} % to {} % on average, the random sets from {} % to {} %.</li>\n').format(
        pct(k4c), pct(k8c), pct(k4s), pct(k8s))
WHY_NOTES += '</ul></div>'


def long_success(kind, keys, lo=0.7, seeds=(0,)):
    """Success % of a run kind on the queries where drone a travels at least `lo`, pooled over the given sets and
    seeds, and the number of such queries."""
    hit = n = nq = 0
    for sc, v in keys:
        s = 'link_{}_{}'.format(sc, v)
        ver = load(os.path.join(SETS, s, 'verification.json'))
        if not ver:
            continue
        dist = [c['dist_t'] for c in ver['cases']]
        nq += sum(d >= lo for d in dist)
        for seed in seeds:
            ev = load(os.path.join(RES, 'plan', '{}_{}_s{}'.format(sc, kind, seed), s + '.json'))
            for r in (ev or {}).get('records', []):
                if dist[r['case']] >= lo:
                    hit, n = hit + r['success_se3'], n + 1
    return (100 * hit / n if n else None), nq


def field_stats(run, s):
    """Medians over the sampled cases of field_diag_link.py: travel time per unit of certified path, uphill share."""
    d = load(os.path.join(RES, 'diagnostics', 'field_{}_{}.json'.format(run, s)))
    if not d:
        return None
    cs = [c for k in ('solved', 'stalled') for c in d['cases'].get(k, [])]
    return dict(t_len=float(np.median([c['T_start'] / c['cert_len'] for c in cs])),
                up=100 * float(np.median([c['frac_up'] for c in cs])), corr=float(np.median([c['corr'] for c in cs])))


E3 = '&nbsp;×&nbsp;10<sup>−3</sup>'
E2 = '&nbsp;×&nbsp;10<sup>−2</sup>'


def ntfb_note(a):
    name = 'Without the TD and normal-alignment terms'
    if not a:
        return '<li><b>{}</b> (the bundle\'s default metric NTFields): still training.</li>\n'.format(name)
    broke = [sc for sc in SCENES if (sc, 'std') in a and a[(sc, 'std')][0] < 0.1]
    fine = [sc for sc in SCENES if (sc, 'std') in a and a[(sc, 'std')][0] >= 0.1]
    eik = lambda sc, k: (TRAIN.get('{}_{}_s0'.format(sc, k)) or {}).get('eik')  # noqa: E731
    sci = lambda v: '–' if v is None else '{:.1f}'.format(1e3 * v)  # noqa: E731
    out = '<li><b>{}</b> (the bundle\'s default metric NTFields). '.format(name)
    if broke:
        fb = [field_stats('{}_ntfb_s0'.format(sc), 'link_{}_std'.format(sc)) for sc in broke]
        ff = [field_stats('{}_ntf_s0'.format(sc), 'link_{}_std'.format(sc)) for sc in broke]
        out += ('Training fails in the {} scenes: those fields solve {} % of the random cases and {} % of the crossings, against '
                '{} % and {} % with the terms. Their Eikonal residual stops at {}{e3}, against {}{e3} with the terms. ').format(
            and_list(broke), rng_txt([100 * a[(sc, 'std')][0] for sc in broke], 1),
            rng_txt([100 * a[(sc, 'cross')][0] for sc in broke if (sc, 'cross') in a], 1),
            rng_txt([100 * a[(sc, 'std')][1] for sc in broke]),
            rng_txt([100 * a[(sc, 'cross')][1] for sc in broke if (sc, 'cross') in a]),
            rng_txt([1e3 * eik(sc, 'ntfb') for sc in broke if eik(sc, 'ntfb')], 1),
            rng_txt([1e3 * eik(sc, 'ntf') for sc in broke if eik(sc, 'ntf')], 1), e3=E3)
        if all(fb) and all(ff):
            out += ('The field comes out slow and bumpy: along the certified paths of random cases its travel time is '
                    '{} per unit of path length (with the terms {}), as if the whole room were next to an obstacle, '
                    'and {} % of the steps go uphill (with the terms {} %). The planner jitters in place: almost '
                    'nothing converges. ').format(rng_txt([x['t_len'] for x in fb]), rng_txt([x['t_len'] for x in ff], 1),
                                                  rng_txt([x['up'] for x in fb]), rng_txt([x['up'] for x in ff]))
    for sc in fine:
        out += ('In the {} the plain loss trains as well as the full one (residual {}{e3} against {}{e3}) and plans '
                'almost as well ({} % against {} % random, {} % against {} % crossings). ').format(
            sc, sci(eik(sc, 'ntfb')), sci(eik(sc, 'ntf')), pct(a[(sc, 'std')][0]), pct(a[(sc, 'std')][1]),
            pct(a.get((sc, 'cross'), (None, None))[0]), pct(a.get((sc, 'cross'), (None, None))[1]), e3=E3)
    if broke:
        out += 'In 7 DOF the two extra terms are what makes the training work at all.'
    return out + '</li>\n'


def plain_note(kind, name, what, a):
    if not a:
        return '<li><b>{}</b> ({}): still training.</li>\n'.format(name, what)
    d_std = [100 * (x - y) for (sc, v), (x, y) in a.items() if v == 'std']
    d_cr = [100 * (x - y) for (sc, v), (x, y) in a.items() if v == 'cross']
    sgn = lambda x: '{}{:.1f}'.format('+' if x >= 0 else '−', abs(x))  # noqa: E731
    span = lambda d: '–' if not d else (sgn(min(d)) if abs(max(d) - min(d)) < 0.05 else  # noqa: E731
                                         'from {} to {}'.format(sgn(min(d)), sgn(max(d))))
    out = '<li><b>{}</b> ({}): random sets {} points, crossings {} points against the recipe'.format(
        name, what, span(d_std), span(d_cr))
    lo, n = long_success(kind, list(a))
    base, _ = long_success('ntf', list(a))
    if lo is not None and base is not None:
        out += '; on the {} queries where drone a travels 0.7 or more, {} % against {} %'.format(n, f1(lo), f1(base))
    out += '.' if len(a) == 8 else ' ({} of 8 sets so far).'.format(len(a))
    return out + '</li>\n'


def lr_note(kind, name, what, a):
    """Long-range pairs: the scenes whose crossings it lifts, and those whose field it breaks."""
    if not a:
        return '<li><b>{}</b> ({}): still training.</li>\n'.format(name, what)
    got = lambda sc, v: a.get((sc, v), (None, None))  # noqa: E731
    up = [sc for sc in SCENES if got(sc, 'cross')[0] is not None and got(sc, 'cross')[0] > got(sc, 'cross')[1] + 0.05]
    down = [sc for sc in SCENES if got(sc, 'std')[0] is not None and got(sc, 'std')[0] < got(sc, 'std')[1] - 0.1]
    rest = [sc for sc in SCENES if sc not in up + down and got(sc, 'std')[0] is not None]
    out = '<li><b>{}</b> ({}). '.format(name, what)
    for sc in up:
        lo, n = long_success(kind, [(sc, 'std'), (sc, 'cross')])
        base, _ = long_success('ntf', [(sc, 'std'), (sc, 'cross')])
        out += ('In the {} the crossings go from {} to {} % (random set {} → {} %); on its {} queries where drone a '
                'travels 0.7 or more, from {} to {} %. ').format(
            sc, pct(got(sc, 'cross')[1]), pct(got(sc, 'cross')[0]), pct(got(sc, 'std')[1]), pct(got(sc, 'std')[0]), n,
            f1(base), f1(lo))
    for sc in rest:
        seeds = [r['summary']['success_rate_se3'] for r in runs_of(sc, 'cross') if r]
        out += 'In the {}: random {} % (recipe {} %), crossings {} % (recipe {} %{}). '.format(
            sc, pct(got(sc, 'std')[0]), pct(got(sc, 'std')[1]), pct(got(sc, 'cross')[0]), pct(got(sc, 'cross')[1]),
            '; its seeds 0–2 span {} %'.format(rng_txt([100 * x for x in seeds], 1)) if len(seeds) > 1 else '')
    if up and not down:
        fs = {sc: field_stats('{}_{}_s0'.format(sc, kind), 'link_{}_std'.format(sc)) for sc in up + rest}
        ff = {sc: field_stats('{}_ntf_s0'.format(sc), 'link_{}_std'.format(sc)) for sc in up + rest}
        if all(fs.values()) and all(ff.values()):
            ups = [fs[sc]['up'] for sc in up]
            out += ('The fields stay clean: along the certified paths of random cases {} (recipe up to {} %), and the '
                    'travel time is {} per unit of path (recipe {}). ').format(
                'no step goes uphill' if max(ups) == 0 else '{} % of the steps go uphill'.format(rng_txt(ups)),
                f1(max(ff[sc]['up'] for sc in up)),
                rng_txt([fs[sc]['t_len'] for sc in up], 1), rng_txt([ff[sc]['t_len'] for sc in up], 1))
            slow = [sc for sc in rest if fs[sc]['t_len'] > 1.5 * ff[sc]['t_len']]
            for sc in slow:
                out += ('The {} field is the exception, {} per unit of path against {}: a mild form of the '
                        'collapse above. ').format(poss(sc), '{:.1f}'.format(fs[sc]['t_len']), '{:.1f}'.format(ff[sc]['t_len']))
    if down:
        fs = [field_stats('{}_{}_s0'.format(sc, kind), 'link_{}_std'.format(sc)) for sc in down]
        ff = [field_stats('{}_ntf_s0'.format(sc), 'link_{}_std'.format(sc)) for sc in down]
        out += ('In the {} it breaks the field: random sets {} % (recipe {} %), crossings {} % (recipe {} %). ').format(
            and_list(down), rng_txt([100 * got(sc, 'std')[0] for sc in down], 1), rng_txt([100 * got(sc, 'std')[1] for sc in down]),
            rng_txt([100 * got(sc, 'cross')[0] for sc in down if got(sc, 'cross')[0] is not None], 1),
            rng_txt([100 * got(sc, 'cross')[1] for sc in down if got(sc, 'cross')[1] is not None]))
        if all(fs) and all(ff):
            out += ('Their fields come out slow, as without the TD and normal terms: travel time {} per unit of '
                    'certified path, against {} for the recipe. ').format(
                rng_txt([x['t_len'] for x in fs]), rng_txt([x['t_len'] for x in ff], 1))
    if len(a) < 8:
        out += '({} of 8 sets so far.)'.format(len(a))
    return out + '</li>\n'


def lr20_note():
    """A fifth long-range over its seeds against the recipe's seeds: where it helps, where it trades, where it breaks."""
    name, what = 'A fifth of the pairs long-range', ('the same sampler with 20 % of the x1 uniform in the room, to see '
                                                     'if a smaller share keeps the near field')
    if not LR20_DONE:
        return lr_note('ntflr20', name, what, ABL['ntflr20'])
    p1 = lambda x: '–' if x is None else '{:.1f}'.format(x)  # noqa: E731
    out = '<li><b>{}</b> ({}; seeds 0–{} against the recipe\'s seeds 0–{}). '.format(name, what, L20_SEEDS[-1], SEEDS[-1])
    if L20_GAIN:
        keys = [(sc, v) for sc in L20_GAIN for v in VARIANTS]
        lo, nq = long_success('ntflr20', keys, seeds=L20_SEEDS)
        base, _ = long_success('ntf', keys, seeds=SEEDS)
        out += ('In the {} every seed beats every recipe seed on both sets: crossings {}, random sets {} (means over '
                'the seeds); on their {} queries where drone a travels 0.7 or more, {} % against {} %. ').format(
            and_list(L20_GAIN), ' and '.join('{} → {} %'.format(p1(MREC(sc, 'cross')), p1(M20(sc, 'cross'))) for sc in L20_GAIN),
            ' and '.join('{} → {} %'.format(p1(MREC(sc, 'std')), p1(M20(sc, 'std'))) for sc in L20_GAIN), nq, f1(lo), f1(base))
        fs = [field_stats('{}_ntflr20_s{}'.format(sc, s), 'link_{}_std'.format(sc)) for sc in L20_GAIN for s in L20_SEEDS]
        ff = [field_stats('{}_ntf_s0'.format(sc), 'link_{}_std'.format(sc)) for sc in L20_GAIN]
        if all(fs) and all(ff):
            ups = [x['up'] for x in fs]
            out += ('The fields come out alike in every seed and clean: along the certified paths of random cases {} '
                    '(recipe seed 0: up to {} %), at a travel time of {} per unit of path (recipe {}). ').format(
                'no step goes uphill' if max(ups) == 0 else '{} % of the steps go uphill'.format(rng_txt(ups, 1)),
                p1(max(x['up'] for x in ff)), rng_txt([x['t_len'] for x in fs], 1), rng_txt([x['t_len'] for x in ff], 1))
    for sc in L20_MIXED:
        every = lambda v, f: all(f(a, b) for a in seeds_of(sc, v, 'ntflr20') for b in seeds_of(sc, v))  # noqa: E731
        col20 = m100(seeds_of(sc, 'std', 'ntflr20', 'collision_rate_se3'))
        colr = S['link_{}_std'.format(sc)]['ntf'].get('collision_rate_se3')
        fs = [field_stats('{}_ntflr20_s{}'.format(sc, s), 'link_{}_std'.format(sc)) for s in L20_SEEDS]
        ff = field_stats('{}_ntf_s0'.format(sc), 'link_{}_std'.format(sc))
        trade = every('cross', lambda a, b: a > b) and every('std', lambda a, b: a < b)
        out += ('In the {} {}: crossings {} → {} %, random set {} → {} %, with more collisions ({} → {} %){}. ').format(
            sc, 'every seed trades random queries for crossings' if trade else 'the result is mixed',
            p1(MREC(sc, 'cross')), p1(M20(sc, 'cross')), p1(MREC(sc, 'std')), p1(M20(sc, 'std')),
            p1(None if colr is None else 100 * colr), p1(col20),
            '; its field is slower, {} per unit of path against {}'.format(rng_txt([x['t_len'] for x in fs], 1),
                                                                           '{:.1f}'.format(ff['t_len']))
            if all(fs) and ff else '')
    out += '</li>\n'
    for sc in L20_UNSTABLE:
        bad, ok = L20_BROKEN[sc], [s for s in L20_SEEDS if s not in L20_BROKEN[sc]]
        val = lambda v, ss: [x for s, x in zip(L20_SEEDS, seeds_of(sc, v, 'ntflr20')) if s in ss]  # noqa: E731
        seed_txt = lambda ss: ('seed {}' if len(ss) == 1 else 'seeds {}').format(and_list([str(x) for x in ss]))  # noqa: E731
        rec_std, rec_cr = seeds_of(sc, 'std'), seeds_of(sc, 'cross')
        out += '<li><b>In the {} scene a fifth long-range is unstable.</b> '.format(sc)
        if ok:
            inside = all(min(rec) <= x <= max(rec) for rec, v in ((rec_std, 'std'), (rec_cr, 'cross')) for x in val(v, ok))
            out += ('{} {} ({} % random, {} % crossings; recipe seeds {} % and {} %); '.format(
                seed_txt(ok).capitalize(), 'stays in the recipe\'s range' if inside else 'trains',
                rng_txt([100 * x for x in val('std', ok)], 1), rng_txt([100 * x for x in val('cross', ok)], 1),
                rng_txt([100 * x for x in rec_std], 1), rng_txt([100 * x for x in rec_cr], 1)))
        out += '{} solve {} % of the random cases and {} % of the crossings. '.format(
            seed_txt(bad) if ok else seed_txt(bad).capitalize(), rng_txt([100 * x for x in val('std', bad)], 1),
            rng_txt([100 * x for x in val('cross', bad)], 1))
        e_rec, _ = escapes('ntf', SCENES, SEEDS)
        e_l20, _ = escapes('ntflr20', [x for x in SCENES if x not in L20_UNSTABLE], L20_SEEDS)
        e_ok, _ = escapes('ntflr20', [sc], ok)
        first = [r['first'] for r in (TRAIN.get('{}_{}_s{}'.format(x, k, s)) for x in SCENES for k in ('ntf', 'ntflr20')
                                      for s in SEEDS) if r and 'first' in r]
        last = lambda ss: [TRAIN['{}_ntflr20_s{}'.format(sc, s)]['loss'] for s in ss]  # noqa: E731
        if e_rec and e_l20 and first:
            out += ('The training loss shows where it goes wrong. Every run of the recipe and of this variant starts on a plateau (total '
                    'loss {}{e2}) and drops by an order of magnitude once the field forms: the recipe by epoch {} in '
                    'every scene and seed, this variant by epoch {} in the other scenes. In the {} scene {}{} {} still '
                    'on the plateau at epoch 8000 (loss {}{e2}{}). ').format(
                rng_txt([1e2 * x for x in first], 1), rng_txt(e_rec), rng_txt(e_l20), sc,
                '{} drops only at epoch {}, and '.format(seed_txt(ok), rng_txt(e_ok)) if ok and e_ok else '',
                seed_txt(bad), 'is' if len(bad) == 1 else 'are', rng_txt([1e2 * x for x in last(bad)], 1),
                ', against {}{e3} for {}'.format(rng_txt([1e3 * x for x in last(ok)], 1), seed_txt(ok), e3=E3) if ok else '',
                e2=E2)
        fb = [field_stats('{}_ntflr20_s{}'.format(sc, s), 'link_{}_std'.format(sc)) for s in bad]
        fo = [field_stats('{}_ntflr20_s{}'.format(sc, s), 'link_{}_std'.format(sc)) for s in ok]
        fh = field_stats('{}_ntflr_s0'.format(sc), 'link_{}_std'.format(sc))
        if all(fb) and fh:
            out += ('Their fields look like the collapsed field of the half share: {} per unit of path with {} % of the '
                    'steps uphill (half share {} and {} %{}).').format(
                rng_txt([x['t_len'] for x in fb], 1), rng_txt([x['up'] for x in fb], 1), '{:.1f}'.format(fh['t_len']),
                '{:.1f}'.format(fh['up']), '; {}: {} and {}'.format(
                    seed_txt(ok), rng_txt([x['t_len'] for x in fo], 1),
                    'none' if max(x['up'] for x in fo) == 0 else rng_txt([x['up'] for x in fo], 1) + ' %')
                if ok and all(fo) else '')
        out += '</li>\n'
    return out


ABLATION_NOTES = ('<div class="col"><p>Against the recipe on the same cases: seed 0 against seed 0, except a fifth of '
                  'the pairs long-range, which was trained with all three seeds.</p><ul>\n')
ABLATION_NOTES += ntfb_note(ABL['ntfb'])
ABLATION_NOTES += lr_note('ntflr', 'Long-range pairs', 'half of the x1 drawn uniformly in the room instead of near x0, '
                          'so drone a moves up to the room size within a pair', ABL['ntflr'])
if ABL['ntflr']:
    ABLATION_NOTES += lr20_note()
ABLATION_NOTES += plain_note('ntfsb', 'Band on the scene clearance only', 'every x0 within 0.05 of an obstacle, none '
                             'placed at a length limit', ABL['ntfsb'])
ABLATION_NOTES += '</ul></div>'

LR20_TXT = 'A fifth of the pairs long-range is training.' if not ABL['ntflr20'] else ''
NEXT_LR = ('  <li><b>Cover long queries in the data.</b> The long-range ablation above tests the cheapest fix inside '
           'NTFields: training pairs that span the room.</li>\n')
if ('corridor', 'cross') in ABL['ntflr']:
    NEXT_LR = ('  <li><b>Long-range pairs without losing the near field.</b> Half of the pairs long-range lifts the '
               'corridor\'s crossings from {} to {} % but collapses the field in the {} scenes. {} Other ways to keep '
               'the near field: keep every near pair and add the long ones on top, or raise the share during '
               'training.</li>\n').format(pct(ABL['ntflr'][('corridor', 'cross')][1], 0),
                                          pct(ABL['ntflr'][('corridor', 'cross')][0], 0), and_list(LR_DOWN), LR20_TXT)
if LR20_DONE and L20_GAIN and L20_UNSTABLE:
    _e_rec, _ = escapes('ntf', SCENES, SEEDS)
    _e_un, _ = escapes('ntf', L20_UNSTABLE, SEEDS)
    _e_ok, _ = escapes('ntflr20', L20_UNSTABLE, L20_SEEDS)
    _bad = sum(len(L20_BROKEN[sc]) for sc in L20_UNSTABLE)
    _nok = len(L20_SEEDS) * len(L20_UNSTABLE) - _bad
    NEXT_LR = ('  <li><b>Make the long-range share stable.</b> A fifth of the pairs long-range is the best variant in the '
               '{} (crossings {}, over three seeds), but in the {} scene it delays the drop off the initial loss plateau '
               'from epoch {} (recipe) to {} in {} seed{} and past 8000 in the other {}. Two cheap fixes target exactly '
               'that: switch the long-range pairs on only after the field has formed (the recipe forms it by epoch {} '
               'in every scene), or fine-tune a trained recipe checkpoint with them. Half of the pairs breaks the field '
               'in {} scenes, so 10 % and 30 % would then locate the best share.</li>\n').format(
        and_list(L20_GAIN), ' and '.join('{} → {} %'.format(f1(MREC(sc, 'cross')), f1(M20(sc, 'cross'))) for sc in L20_GAIN),
        and_list(L20_UNSTABLE), rng_txt(_e_un), rng_txt(_e_ok), WORDS[_nok], '' if _nok == 1 else 's', WORDS[_bad],
        max(_e_rec),
        WORDS.get(len(LR_DOWN), len(LR_DOWN)))
NEXT_STEPS = """<div class="col"><ul>
""" + NEXT_LR + """  <li><b>The homotopy head in 7 DOF.</b> In SE(3) it removed most of NTFields' stalls on this repository's scenes, because its analytic cone has no spurious minima. The length would enter its encoder as one more coordinate, with the same clearance and speed as here.</li>
  <li><b>Short queries along a coarse route.</b> NTFields solves most queries shorter than 0.3. Chaining short link-aware queries along a coarse global route (a low-resolution roadmap of the link) would use it where it is reliable.</li>
  <li><b>Soft contact when the hard link is impossible.</b> Here every case has a link-keeping solution, so contact is a hard constraint and a plan either keeps it the whole way or fails. To maximize contact time where it cannot be kept, give blocked line of sight a low but non-zero speed instead of treating it as an obstacle; the travel time then trades path length against time out of contact.</li>
</ul></div>"""


if __name__ == '__main__':
    print('figures:', copy_figures())
    html_ = page(body())
    with open(os.path.join(HERE, 'index.html'), 'w') as fh:
        fh.write(html_)
    print('wrote', os.path.join(HERE, 'index.html'), len(html_), 'bytes')
