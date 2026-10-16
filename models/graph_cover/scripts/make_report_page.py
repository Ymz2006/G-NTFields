#!/usr/bin/env python3
"""Build the HTML report page of the cover-decoder campaign from the result files (standard library only).

    python3 models/graph_cover/scripts/make_report_page.py --out /path/to/cover_report.html

Reads results/graph_cover/summary/plan_summary.json (aggregate_cover.py), results/graph_cover/diag/*.json,
ridge_free/ridge_free.json, ridge_plan/ridge_plan.json and the figures in results/graph_cover/figures.
"""
import argparse
import base64
import glob
import html
import json
import os
import re
import statistics

HOST = '/media/jeffrey/G-Shape-NTFields'
R = os.path.join(HOST, 'results/graph_cover')

LABEL = {
    'eu2|cover': 'cover, Euclidean residual, 2 learned covers', 'eu0|cover': 'cover, Euclidean, rotation routes only',
    'eu2w|cover': 'cover, Euclidean, wide features', 'eu2r|cover': 'cover, Euclidean, residual x2',
    'eu2L|cover': 'cover, Euclidean, Lipschitz cap 4', 'lse2|cover': 'cover, logsumexp residual (ablation)',
    'lp2|cover': 'cover, l8 seminorm, 2 covers', 'lp0|cover': 'cover, l8 seminorm, rotation routes only',
    'lp4|cover': 'cover, l8 seminorm, 4 covers', 'lp2nf|cover': 'cover, l8 seminorm, no T_free',
    'lp2p2|cover': 'cover, l2 group seminorm, 2 covers', 'lp2soft|cover': 'cover, l8 seminorm, soft min 0.02',
    'eu2c|cover': 'cover, Euclidean + conformal cone', 'lp2p2c|cover': 'cover, l2 group seminorm + conformal cone',
    'gmetric_iso_lie|lie': 'graph_metric (logsumexp head, reference)', 'trieq_iso_lie|lie': 'graph_tri_metric, equivariant V (previous guaranteed head)',
    'ggram|branch': 'graph_gram (fixed (a, b) decoder)', 'br1|branch': 'branch: T_free + |dF|^2 (other campaign)',
    'bc4|branch': 'branch: conformal, K4 (other campaign)',
}
METRIC = {'eu2', 'eu0', 'eu2w', 'eu2r', 'eu2L', 'lp2', 'lp0', 'lp4', 'lp2nf', 'lp2p2', 'trieq_iso_lie'}
ORDER = ['eu2|cover', 'eu0|cover', 'eu2w|cover', 'eu2r|cover', 'eu2L|cover', 'lse2|cover', 'lp2|cover', 'lp0|cover',
         'lp4|cover', 'lp2nf|cover', 'lp2p2|cover', 'lp2soft|cover', 'eu2c|cover', 'lp2p2c|cover', 'ggram|branch',
         'gmetric_iso_lie|lie', 'trieq_iso_lie|lie', 'br1|branch', 'bc4|branch']
SCENE_SET = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}


def ms(vals, f=100.0, d=1):
    if not vals:
        return '-'
    if len(vals) == 1:
        return '{:.{d}f}'.format(vals[0] * f, d=d)
    return '{:.{d}f} ± {:.{d}f}'.format(statistics.mean(vals) * f, statistics.stdev(vals) * f, d=d)


def load_plan():
    with open(os.path.join(R, 'summary/plan_summary.json')) as fh:
        raw = json.load(fh)
    out = {}
    for k, v in raw.items():
        scene, tag, src = k.split('|')
        out[(scene, tag, src)] = {t: {int(s): r for s, r in d.items()} for t, d in v.items()}
    return out


def load_diag():
    out = {}
    for f in glob.glob(os.path.join(R, 'diag/*.json')):
        run = os.path.basename(f)[:-5]
        m = re.match(r'^([TLA])_(.+)_s(\d+)$', run)
        if not m:
            continue
        d = json.load(open(f))
        out.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = d
    return out


def b64(path):
    with open(path, 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()


def table_scene(plan, diag, scene):
    tset = SCENE_SET[scene]
    rows = []
    for key in ORDER:
        tag, src = key.split('|')
        rec = plan.get((scene, tag, src))
        if not rec or tset not in rec:
            continue
        v = list(rec[tset].values())
        dg = diag.get((scene, tag), {})
        dex = [d['dex_bins'] for d in dg.values()]
        low = [d['speed_calibration'].get('0.001-0.002', {}).get('median_pred_over_true') for d in dg.values()]
        low = [x for x in low if x]
        tri = [d['triangle']['attack_y'] for d in dg.values()]
        tri_s = '-'
        if tri:
            fr = max(t['viol_frac'] for t in tri)
            mx = max(t['viol_rel_max'] for t in tri)
            tri_s = '0' if mx <= 1e-6 else '{:.0f} % / {:.0f} %'.format(100 * fr, 100 * mx)
        extra = {}
        for s in rec:
            if s != tset:
                vv = list(rec[s].values())
                extra[s.replace(tset + '_', '')] = '{} / {}'.format(ms([x['success_rate_se3'] for x in vv]), ms([x['collision_rate_se3'] for x in vv]))
        guar = 'metric' if tag in METRIC else ('≈ metric' if tag == 'lp2soft' else 'none')
        rows.append(dict(label=LABEL.get(key, key), guar=guar, n=len(v),
                         succ=ms([x['success_rate_se3'] for x in v]), coll=ms([x['collision_rate_se3'] for x in v]),
                         nonconv=ms([x['nonconv_rate_se3'] for x in v]),
                         big=extra.get('bigangle', '-'), rot=extra.get('rotheavy', '-'),
                         dex=ms(dex, 1.0, 2) if dex else '-', low='{:.0f}×'.format(min(low)) + ('-{:.0f}×'.format(max(low)) if len(low) > 1 and max(low) - min(low) > 2 else '') if low else '-',
                         tri=tri_s, tag=tag, metric=tag in METRIC,
                         best=float(statistics.mean([x['success_rate_se3'] for x in v]))))
    return rows


def scatter_svg(rows):
    pts = []
    for r in rows:
        if r['dex'] == '-':
            continue
        dex = float(r['dex'].split(' ')[0])
        pts.append((dex, r['best'] * 100, r['tag'], r['metric']))
    if not pts:
        return ''
    W, H, L, B, T_, Rm = 640, 380, 56, 46, 16, 16
    x0, x1 = 0.5, 1.0
    y0, y1 = 35, 100
    def X(x): return L + (x - x0) / (x1 - x0) * (W - L - Rm)
    def Y(y): return T_ + (y1 - y) / (y1 - y0) * (H - T_ - B)
    s = ['<svg viewBox="0 0 {} {}" role="img" aria-label="planning success versus speed calibration error">'.format(W, H)]
    for gx in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        s.append('<line x1="{0:.1f}" y1="{1}" x2="{0:.1f}" y2="{2}" class="grid"/>'.format(X(gx), T_, H - B))
        s.append('<text x="{:.1f}" y="{}" class="tick" text-anchor="middle">{:g}</text>'.format(X(gx), H - B + 16, gx))
    for gy in (40, 50, 60, 70, 80, 90, 100):
        s.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" class="grid"/>'.format(L, Y(gy), W - Rm))
        s.append('<text x="{}" y="{:.1f}" class="tick" text-anchor="end">{}</text>'.format(L - 6, Y(gy) + 4, gy))
    s.append('<text x="{:.1f}" y="{}" class="axis" text-anchor="middle">speed calibration error, dex (lower is better)</text>'.format((L + W - Rm) / 2, H - 6))
    s.append('<text transform="translate(14 {:.1f}) rotate(-90)" class="axis" text-anchor="middle">SE(3) success, %</text>'.format((T_ + H - B) / 2))
    for dex, succ, tag, metric in sorted(pts):
        cls = 'pt-metric' if metric else 'pt-free'
        s.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" class="{}"/>'.format(X(dex), Y(succ), cls))
        s.append('<text x="{:.1f}" y="{:.1f}" class="lbl">{}</text>'.format(X(dex) + 7, Y(succ) - 5, html.escape(tag)))
    s.append('<circle cx="{}" cy="{}" r="5" class="pt-metric"/><text x="{}" y="{}" class="lbl">pseudometric by construction</text>'.format(L + 12, T_ + 10, L + 22, T_ + 14))
    s.append('<circle cx="{}" cy="{}" r="5" class="pt-free"/><text x="{}" y="{}" class="lbl">no guarantee</text>'.format(L + 12, T_ + 28, L + 22, T_ + 32))
    s.append('</svg>')
    return '\n'.join(s)


def ridge_table():
    d = json.load(open(os.path.join(R, 'ridge_free/ridge_free.json')))['results']
    names = [('hardmin', 'hard min (exact field)'), ('softmin_tau0.01', 'soft min, τ 0.01'), ('softmin_tau0.03', 'soft min, τ 0.03'),
             ('softmin_tau0.1', 'soft min, τ 0.1'), ('graph_gram', 'graph_gram (decoder corner)'), ('tri_eq', 'triangle head, equivariant V'),
             ('graph_sqdist', 'graph_sqdist'), ('metric_best', 'metric head, best recipe'), ('metric_ntf', 'metric head, NTFields loss'),
             ('graph_metric', 'graph_metric'), ('scalar', 'scalar network')]
    rows = []
    for k, lab in names:
        if k not in d:
            continue
        r = d[k]
        fired = r['fired_frac']
        bias = '{:.3f}'.format(r['bias_deg']['median']) if fired > 0 else '-'
        band = r['band_deg']['median']
        band_s = 'whole window' if band >= 79 else '{:.2f}'.format(band)
        lam = '{:.1f}'.format(r['lam_at_ridge']['median']) if fired > 0 else '-'
        rows.append('<tr><td>{}</td><td>{:.0f} %</td><td>{}</td><td>{}</td><td>{}</td><td>{:+.1f} %</td></tr>'.format(
            lab, 100 * fired, bias, band_s, lam, 100 * r['T_err_pi']['median']))
    return '\n'.join(rows)


def plan_table():
    d = json.load(open(os.path.join(R, 'ridge_plan/ridge_plan.json')))['fields']
    rows = []
    for tag, lab in (('softmin_tau0.1', 'soft min, τ 0.1'), ('softmin_tau0.03', 'soft min, τ 0.03'), ('metric_best', 'metric head, best recipe'), ('graph_metric', 'graph_metric')):
        if tag not in d:
            continue
        for b in ('170-179', '179-180'):
            B = d[tag]['bins'][b]
            cells = ''.join('<td>{:.1f} / {:.4f} / {:.4f}</td>'.format(100 * B[m]['success'], B[m]['cost_ratio_p50'], B[m]['cost_ratio_p95']) for m in ('gradient', 'ridge_sample', 'ridge_sharpen'))
            rows.append('<tr><td>{}</td><td>{}°</td>{}<td>{:.1f}</td></tr>'.format(lab, b, cells, B['ridge_sample']['fired_mean']))
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = load_plan()
    diag = load_diag()
    rowsT = table_scene(plan, diag, 'T')
    def trs(rows):
        out = []
        for r in rows:
            cls = ' class="hi"' if r['tag'] in ('eu2', 'eu2c') else ''
            out.append('<tr{}><td>{}</td><td><span class="chip {}">{}</span></td><td>{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td></tr>'.format(
                cls, html.escape(r['label']), 'ok' if r['guar'] == 'metric' else ('mid' if r['guar'] == '≈ metric' else 'no'), r['guar'], r['n'], r['succ'], r['coll'], r['nonconv'], r['big'], r['rot'], r['dex'], r['low'], r['tri']))
        return '\n'.join(out)
    scene_tables = ''
    for sc in 'LA':
        rows = table_scene(plan, diag, sc)
        if rows:
            scene_tables += '<h3>{} scene ({})</h3><div class="tw"><table><thead><tr><th>head</th><th>guarantee</th><th>n</th><th>success</th><th>collision</th><th>non-conv</th><th>bigangle</th><th>rotheavy</th><th>dex</th><th>lowest decade</th><th>triangle</th></tr></thead><tbody>{}</tbody></table></div>'.format(sc, SCENE_SET[sc], trs(rows))
    figs = {k: b64(os.path.join(R, 'figures', k + '.png')) for k in ('ridge_free', 'slice_T_lp2_s0_goal0', 'slice_T_eu2_s0_goal0', 'slice_T_gmetric_iso_lie_s0_goal0')}
    eu2 = next(r for r in rowsT if r['tag'] == 'eu2')
    eu2c = next((r for r in rowsT if r['tag'] == 'eu2c'), None)
    gg = next(r for r in rowsT if r['tag'] == 'ggram')
    gm = next(r for r in rowsT if r['tag'] == 'gmetric_iso_lie')
    tq = next(r for r in rowsT if r['tag'] == 'trieq_iso_lie')
    page = PAGE.format(
        rowsT=trs(rowsT), scene_tables=scene_tables, scatter=scatter_svg(rowsT), ridge_rows=ridge_table(), plan_rows=plan_table(),
        eu2=eu2['succ'], eu2c=eu2c['succ'] if eu2c else '-', eu2c_coll=eu2c['coll'] if eu2c else '-', eu2c_n=eu2c['n'] if eu2c else 0,
        gg=gg['succ'], gm=gm['succ'], tq=tq['succ'], eu2n=eu2['n'], **figs)
    with open(a.out, 'w') as fh:
        fh.write(page)
    print('wrote', a.out, len(page) // 1024, 'KB')


PAGE = r'''<title>Cover Decoder Report</title>
<style>
/* layout: a single reading column of 72ch with full-bleed tables and figures; summary tiles first, then one section per question */
:root {{
  --bg: #f7f6f2; --fg: #1d2430; --mute: #5b6472; --line: #d9d6cc; --accent: #0f5c73; --accent-2: #b4521d;
  --ok: #1f7a4d; --no: #9a3b2b; --mid: #8a6d1f; --tile: #ecebe4; --hi: #e5eef1;
  --display: "Fraunces", Georgia, "Times New Roman", serif; --body: "Source Sans 3", "Segoe UI", Helvetica, Arial, sans-serif; --mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #15191f; --fg: #e8e6df; --mute: #a6adb8; --line: #333a44; --accent: #6fc3d8; --accent-2: #e59a63; --ok: #5fc48f; --no: #e07a66; --mid: #d6b45a; --tile: #1e242c; --hi: #1c2b32; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #15191f; --fg: #e8e6df; --mute: #a6adb8; --line: #333a44; --accent: #6fc3d8; --accent-2: #e59a63; --ok: #5fc48f; --no: #e07a66; --mid: #d6b45a; --tile: #1e242c; --hi: #1c2b32; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.5; padding-block: 32px 64px; padding-inline: 16px; }}
main {{ max-width: 78ch; margin: 0 auto; }}
h1, h2, h3 {{ font-family: var(--display); font-weight: 600; text-wrap: balance; line-height: 1.15; }}
h1 {{ font-size: 2.1rem; margin: 0 0 .4rem; }}
h2 {{ font-size: 1.45rem; margin: 2.6rem 0 .8rem; padding-top: .8rem; border-top: 1px solid var(--line); }}
h3 {{ font-size: 1.1rem; margin: 1.6rem 0 .5rem; }}
p {{ margin: 0 0 .9rem; max-width: 72ch; }}
.eyebrow {{ font-family: var(--mono); font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--mute); margin-bottom: .6rem; }}
.lede {{ font-size: 1.08rem; color: var(--mute); max-width: 72ch; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1.4rem 0; }}
.tile {{ background: var(--tile); padding: 12px 14px; border-radius: 6px; min-width: 0; }}
.tile .v {{ font-family: var(--display); font-size: 1.8rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
.tile .k {{ font-size: .82rem; color: var(--mute); }}
.tile.acc .v {{ color: var(--accent); }}
ul {{ padding-left: 1.2rem; max-width: 72ch; }} li {{ margin-bottom: .45rem; }}
.tw {{ overflow-x: auto; margin: .6rem 0 1.2rem; }}
table {{ border-collapse: collapse; font-size: .86rem; min-width: 100%; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; white-space: nowrap; }}
th {{ font-family: var(--mono); font-size: .72rem; letter-spacing: .05em; text-transform: uppercase; color: var(--mute); font-weight: 500; }}
td.num {{ font-variant-numeric: tabular-nums; }}
tr.hi td {{ background: var(--hi); }}
.chip {{ font-family: var(--mono); font-size: .7rem; padding: 1px 6px; border-radius: 3px; border: 1px solid currentColor; }}
.chip.ok {{ color: var(--ok); }} .chip.no {{ color: var(--no); }} .chip.mid {{ color: var(--mid); }}
figure {{ margin: 1rem 0 1.4rem; }} figure img {{ max-width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 4px; }}
figcaption {{ font-size: .84rem; color: var(--mute); margin-top: .4rem; max-width: 72ch; }}
svg {{ max-width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--line); stroke-width: 1; }} .tick {{ fill: var(--mute); font-size: 11px; font-family: var(--mono); }} .axis {{ fill: var(--fg); font-size: 12px; }}
.lbl {{ fill: var(--fg); font-size: 10.5px; font-family: var(--mono); }} .pt-metric {{ fill: var(--accent); }} .pt-free {{ fill: var(--accent-2); }}
code {{ font-family: var(--mono); font-size: .85em; }}
.formula {{ font-family: var(--mono); font-size: .86rem; background: var(--tile); padding: 10px 12px; border-radius: 4px; overflow-x: auto; margin: .6rem 0 1rem; }}
.note {{ border-left: 3px solid var(--accent-2); padding-left: 12px; color: var(--mute); font-size: .92rem; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<main>
<div class="eyebrow">G-Shape-NTFields · 2026-10-13 · Tshape3d_env1, iso clearance speed, Lie-gradient Eikonal, per-case forward MPPI on 1000 RRT-verified cases</div>
<h1>Cover-quotient decoder, triangle inequality and ridge detection</h1>
<p class="lede">Four questions from the decoder-properties note: can a decoder beat the fixed (a, b) decoder of graph_gram, can the triangle inequality be made to work, do "peaks only" and "movable ridges" hold by construction, and does the staged ridge-detection idea pay off. Every number comes from <code>results/graph_cover/</code>; the written report is <code>results/graph_cover/REPORT.md</code>.</p>

<div class="tiles">
  <div class="tile acc"><div class="v">{eu2}</div><div class="k">cover decoder, Euclidean residual: SE(3) success, pseudometric by construction ({eu2n} seeds)</div></div>
  <div class="tile"><div class="v">{gg}</div><div class="k">graph_gram, same pipeline (3 seeds)</div></div>
  <div class="tile"><div class="v">{tq}</div><div class="k">previous guaranteed head, graph_tri_metric (5 seeds)</div></div>
  <div class="tile"><div class="v">{gm}</div><div class="k">graph_metric, logsumexp, no guarantee (3 seeds)</div></div>
  <div class="tile acc"><div class="v">{eu2c}</div><div class="k">cover decoder + conformal cone ({eu2c_n} seed(s)), collisions {eu2c_coll} %</div></div>
</div>

<h2>1. What was found</h2>
<ul>
<li><strong>The guaranteed decoder beats graph_gram on every scene.</strong> The cover-quotient head with a Euclidean residual is a pseudometric for every parameter value and reaches {eu2} % on the T scene against graph_gram's {gg} %, with a third of the non-convergences, and 96.1 / 94.0 % on the L / A scenes against 85.8 / 90.2 %; it beats the previous guaranteed head ({tq} % on T) by 9-20 points. On L and A it is level with graph_metric (96.2 / 94.4 %); on T it stays about 4 points below it, all of it collisions.</li>
<li><strong>The seminorm, not the inequality, sank the earlier guaranteed heads.</strong> Every variant built on the group l<sub>p</sub> seminorm of standardised embeddings (p = 8 or 2, with 0, 2 or 4 learned covers, with or without the free-space bound, hard or soft minimum) collapses to 41-57 % with 40-58 % collisions, exactly like graph_tri_metric. The Euclidean residual on the same cover structure gives {eu2} %, and its logsumexp twin with no guarantee gives the same, so among these heads the guarantee is free.</li>
<li><strong>Success is a monotone function of low-speed calibration.</strong> Across all nineteen diagnosed heads, planning success tracks how well the field reproduces the slow clearance speeds (chart below). The Eikonal equation asks for a slope up to 1000 at contact; a metric's slope is bounded by the embedding's Lipschitz constant, which wider features, a larger residual or a 4× Lipschitz cap did not move.</li>
<li><strong>The conformal cone supplies the missing slope.</strong> Scaling the cone at a pose by a learned 1 + ρ(p) (the branch decoder's term) lifts the same head to {eu2c} % with {eu2c_coll} % collisions on T and to 97.6 / 98.0 % on L / A, above graph_metric on all three scenes and level with the concurrent campaign's best branch head (95.4 / 97.5 / 97.9 %); it rescues even the l<sub>2</sub> seminorm (41.7 → 91.8 %). It gives up the exact inequality but violates it mildly (7.8 % of attacked in-domain triples, at most 24 %, against 57 % / 11.5 % for graph_metric).</li>
<li><strong>Peaks only holds; movable ridges are learned.</strong> A float64 kink census finds no valley kink in any decoder field in free space and none attributable to the cover heads among obstacles; the cover head's element switches register as ridges only. The l<sub>8</sub> head's learned ridges separate the two corridors around the central obstacle in a translation slice.</li>
<li><strong>The ridge detector works exactly where a ridge exists.</strong> On a soft-min model it fires on every anchor with zero location bias and curvature −1/τ to the digit, and route recovery (stage 3) removes the rounding cost at the half-turn. The trained metric heads turn out to have no ridge at the half-turn at all, only a 100°-wide plateau, so the detector has nothing to find there and neither planning stage helps them.</li>
</ul>

<h2>2. The decoder</h2>
<p>Property 3 of the note allows the triangle inequality for branches that are copies of one metric under a symmetry group, T = min<sub>g</sub> d(x, g·y). The group does not have to come from the geometry: any finite group acting freely by isometries on a covering space gives a quotient pseudometric. The head therefore builds covers with learned branch points. Per pose, from the shared G-Shape encoder embedding e(x) of G(x) = (t, vec R):</p>
<div class="formula">even      E(x)   = U e(x)
q-odd     Q(x~)  = q_x ⊗ W e(x)                      flips sign under q → −q (rotation double cover)
cut-odd   O_i(x~) = w(z_i(x)) ⊗ A_i e(x),  i = 1..m   z_i = Z_i e(x) ∈ R²,  w(z) = |z|^(1/2) sqrt(z)  (complex root)

T(x, y)² = min over g in Z_2^(1+m) of [ T_free,σ(g)(x, y)² + ρ( F(x~) − g·F(y~) ) ],   F = [E; Q; O_1; …; O_m]
T_free,σ² = |Δt|²/s_t² + ψ_σ²/s_r²,   ψ_+ = 2 atan2(|v|, q_x·q_y),   ψ_− = 2π − ψ_+</div>
<p>A loop around a zero of z<sub>i</sub> flips the sign of the complex square root, so O<sub>i</sub> is a function on a double cover branched at a learned point, meant to sit inside an obstacle. The group flips the signs of the odd blocks (an isometry of every l<sub>p</sub> norm) and swaps the two quaternion routes. Because sqrt(T_free² + ρ) is an invariant pseudometric on the cover, its minimum over the group is a pseudometric on the base. For every parameter value: T ≥ 0 with T(x, x) = 0, symmetry, the triangle inequality, T ≥ T_free, a sqrt(Q) cone at the goal, only min and sqrt on the output path (peaks only), ridges wherever two elements tie (movable), and no cap. The branch is selected without autograd and only the selected element is recomputed with gradients, so the far point of the unused rotation route cannot inject NaN into the double backward. Nine property tests pass in float64 at random parameters, including a 40-step adversarial triangle attack (≤ 1e-9 for every metric variant, > 1e-4 for the logsumexp ablation).</p>
<p>Residual options: Euclidean |d|² (metric), the group l<sub>p</sub> seminorm of graph_tri_metric (metric), graph_metric's logsumexp group max (ablation, no guarantee); switches for dropping T_free, a soft minimum of width τ over the learned covers, and the branch decoder's conformal cone term T_free → T_free + ρ̄λ̄ tanh(T_free/λ̄) with per-pose softplus heads (no guarantee). Training is the original metric NTFields pipeline (iso clearance speed, Lie-gradient Eikonal residual, AdamW with accept/reject, 4000 epochs); evaluation is the per-case forward MPPI planner at seed 0 with SE(3) convergence.</p>

<h2>3. Obstacle results</h2>
<h3>T scene (Tshape3d_env1)</h3>
<p>T scene, 1000 cases. Success, collision and non-convergence in % (SE(3) criterion); bigangle and rotheavy give success / collision on the rotation-heavy test sets; dex is the mean |log10(predicted / true speed)| over nine clearance decades at 16 384 training endpoints (lower is better calibrated); "lowest decade" is how many times too fast the field is where the true speed is 0.001-0.002; "triangle" is the share of in-domain triples violated after a 300-step adversarial ascent of the middle pose, and the largest violation relative to T(x, z).</p>
<div class="tw"><table><thead><tr><th>head</th><th>guarantee</th><th>n</th><th>success</th><th>collision</th><th>non-conv</th><th>bigangle</th><th>rotheavy</th><th>dex</th><th>lowest decade</th><th>triangle</th></tr></thead><tbody>
{rowsT}
</tbody></table></div>
<figure>{scatter}<figcaption>SE(3) planning success against the speed-calibration error of each head (T scene, seed means). Every head that plans well predicts the slowest decade 27-46× too fast; the l<sub>p</sub> heads 74-125×. The two cover heads with the conformal cone (eu2c, lp2p2c) and the other campaign's conformal branch head (bc4) sit at the calibrated end.</figcaption></figure>
{scene_tables}

<h2>4. Ridge detection, stage 1</h2>
<p>Detector: body-frame gradient c and Hessian of T(x exp(ξ), y) at the start; a point is on the learned ridge when S|c| &lt; 1 noticeably, the Hessian has a strongly negative eigenvalue λ<sub>min</sub> with eigenvector v, and the cross derivative r = c·v vanishes. Validation on the free-space half-turn: 48 anchors, start x(φ) = (p + 0.2 m, R<sub>g</sub> exp(φ n)), φ scanned over 180 ± 40°, band S|c| &lt; 0.95, λ<sub>min</sub> &lt; −5.</p>
<div class="tw"><table><thead><tr><th>field</th><th>fires on</th><th>location bias, deg (median)</th><th>band width, deg</th><th>λ<sub>min</sub> at ridge</th><th>T error at exact π</th></tr></thead><tbody>
{ridge_rows}
</tbody></table></div>
<p>On the soft-min model the derivation holds to the digit: the two rotation routes are opposite in the rotation block, so λ<sub>min</sub> = −(1 − p<sub>1</sub>·p<sub>2</sub>)/(2τ) = −1/τ, the band scales with τ, and |c| on the ridge is the translation share cos(α/2). The decoder-corner fields have sub-degree bands from the 1e-6 range margin of φ(b). The metric and scalar heads never fire: with loosened thresholds (band 0.9, κ 0.5, ±60°) they fire on 85-98 % of anchors with a median bias below 1.6°, but the "band" is 106-120° wide. Their 36-49 % value error at the half-turn is a plateau with |c| ≈ 0.8 and curvature ≈ −1, not a rounded corner.</p>
<figure><img src="{ridge_free}" alt="T, gradient norm, minimum Hessian eigenvalue and cross derivative along a rotation scan through the half-turn for the hard and soft minimum fields and the trained free-space heads"><figcaption>One anchor: T(φ), |c|, λ<sub>min</sub> and the cross derivative r along the rotation scan. The soft-min fields (orange, green, red) dip to |c| ≈ 0.06 with a curvature spike of −1/τ exactly at 180°; graph_gram and the triangle head (pink, dark blue) show the decoder corner; the metric and scalar heads (purple, brown, grey, cyan) have |c| ≈ 0.5-0.8 and a slowly varying r across the whole window.</figcaption></figure>

<h2>5. Stages 2 and 3: planning with the detector</h2>
<p>Gradient planner (step 0.02, backtracking, sampling fallback) against the same planner with the detector (band 0.9, κ 0.5). Stage 2 takes a 128-candidate sampling step wherever the detector fires. Stage 3 probes x exp(±0.3 v) on both sides of the ridge, extrapolates each side back to x and descends the gradient of the lower branch. 300 pairs per relative-rotation bin; cells are success % / cost ratio p50 / p95 (path time over the exact time).</p>
<div class="tw"><table><thead><tr><th>field</th><th>bin</th><th>gradient</th><th>stage 2 (sample in band)</th><th>stage 3 (recover routes)</th><th>firings per path</th></tr></thead><tbody>
{plan_rows}
</tbody></table></div>
<p>Where the field is a genuine rounded min of two routes, stage 3 recovers the exact-field cost (the residual 1.0002 is the planner's own discretisation) and stage 2 gives nothing: a sampling step inside the band is as blind as the gradient. On the trained metric heads the detector fires 50-70 times per path, sampling changes nothing and sharpening costs 1-2 points: without a corner there are no two routes to recover. The staged plan is the right tool for a branch or cover head with a soft minimum and the wrong tool for the plain metric heads.</p>

<h2>6. Peaks only, and ridges among obstacles</h2>
<p>Kink census: 256 random geodesic lines of ±0.3 scaled-twist units per field, spacing 2·10<sup>−4</sup>, float64, a kink being a slope change above 30 % of the local slope within one spacing. Free-space decoder fields, metric head and scalar field: 0 ridge and 0 valley kinks on 154 units of path. Among obstacles the smooth logsumexp heads register 112-153 ridge-like and 52-92 valley-like events near contact (slopes of hundreds changing within the spacing), which is the noise floor there; the cover heads are at or below it, with 18-50 of their ridge events at a branch switch and 0-2 valley events at a switch. A float32 pass was discarded because rounding of T over 2·10<sup>−4</sup> fakes about 1500 kink pairs.</p>
<figure><img src="{slice_T_lp2_s0_goal0}" alt="translation slice of the l8 cover head: travel time, speed-corrected gradient norm, minimum curvature and active group element"><figcaption>l<sub>8</sub> cover head, translation slice through test goal 0 at the goal's rotation. Right: the active group element; its boundaries are the field's exact ridges and separate the upper and lower corridor around the central block. Left to centre: T, S|c| (1 = Eikonal), λ<sub>min</sub>; magenta is the zero set of the cross derivative, black the detector band.</figcaption></figure>
<figure><img src="{slice_T_eu2_s0_goal0}" alt="translation slice of the Euclidean cover head"><figcaption>Euclidean cover head, same slice: a single element is active in the plane (its learned covers act on 4-18 % of pairs, mostly rotation-heavy ones), so the field in this slice is smooth like a metric head's.</figcaption></figure>
<figure><img src="{slice_T_gmetric_iso_lie_s0_goal0}" alt="translation slice of graph_metric"><figcaption>graph_metric, same slice: the detector band (black) covers 40 % of the free area and the r = 0 contour is scattered; a smooth head has no ridge to find.</figcaption></figure>

<h2>7. Caveats</h2>
<ul>
<li>One seed for most variants (three for the Euclidean cover head, graph_gram, graph_metric and the conformal branch head; five for graph_tri_metric). The evaluator is deterministic at seed 0; differences of 1-2 points are within seed spread.</li>
<li>The machine was shared with two other campaigns throughout (load 27-38 on 36 cores); the other campaign's seed-0 checkpoints are used as comparison points, read-only.</li>
<li>The l<sub>p</sub> failure is established empirically over five variants; the Lipschitz explanation is consistent with every calibration number but was not isolated by a controlled experiment.</li>
</ul>
<p class="note">Code: <code>models/graph_cover/</code> (README, EXPERIMENTS.md K0-K5), head in <code>models/graph_method_triangle/cover_decoder.py</code>, tests in <code>tests/test_graph_cover.py</code>; tables from <code>scripts/aggregate_cover.py</code>, this page from <code>scripts/make_report_page.py</code>.</p>
</main>
'''

if __name__ == '__main__':
    main()
