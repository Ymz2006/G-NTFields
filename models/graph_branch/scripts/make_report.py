#!/usr/bin/env python3
"""Build the campaign report page (HTML) from the summary JSONs of models/graph_branch.

    python3 models/graph_branch/scripts/make_report.py --out <path.html>

Reads results/graph_branch/summary/{plan_summary,synth_summary,properties,conf_diag}.json (run the aggregate_*.py
scripts first) and embeds the slice figures. Python 3 standard library only.
"""
import argparse
import base64
import html
import json
import math
import os
import statistics as st

HOST = '/media/jeffrey/G-Shape-NTFields'
SUM = os.path.join(HOST, 'results/graph_branch/summary')
FIG = os.path.join(HOST, 'results/graph_branch')

HEAD_LABEL = {
    'orig': 'metric baseline (original NTFields)', 'ggram': 'graph_gram', 'gmetric': 'graph_metric', 'br1': 'branch K=1 strict', 'br4': 'branch K=4 strict',
    'bc1e': 'branch K=1 conformal + end', 'bc4e': 'branch K=4 conformal + end', 'bc4': 'branch K=4 conformal',
    'br4L': 'branch K=4 lse strict', 'bc4Le': 'branch K=4 lse conformal + end', 'ggc': 'graph_gram + conformal term',
}
HEAD_COLOR = {'orig': 'var(--c-orig)', 'ggram': 'var(--c-gram)', 'gmetric': 'var(--c-metric)', 'br1': 'var(--c-strict)', 'br4': 'var(--c-strict)',
              'bc4': 'var(--c-branch)', 'bc4e': 'var(--c-branch2)', 'bc1e': 'var(--c-branch2)', 'ggc': 'var(--c-gram2)'}


def load(name):
    p = os.path.join(SUM, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def esc(s):
    return html.escape(str(s))


def ms(vals, scale=100.0, digits=1):
    v = [x * scale for x in vals if x is not None]
    if not v:
        return '–', None, None
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else None
    txt = ('{:.%df}' % digits).format(m) + (' ± {:.{d}f}'.format(sd, d=digits) if sd is not None else '')
    return txt, m, sd


def img(path, alt):
    if not os.path.exists(path):
        return '<p class="muted">figure pending</p>'
    b = base64.b64encode(open(path, 'rb').read()).decode()
    return '<figure><img src="data:image/png;base64,{}" alt="{}" loading="lazy"><figcaption>{}</figcaption></figure>'.format(
        b, esc(alt), esc(alt))


def bar_chart(groups, heads, plan, key='success_rate_se3', title='', ymin=40, ymax=100):
    """Grouped bars: groups = [(set, label)], heads = [variant]; mean +- sd over seeds."""
    W, H, L, B, T = 640, 250, 40, 34, 18
    gw = (W - L - 10) / max(1, len(groups))
    bw = min(22, (gw - 16) / max(1, len(heads)))
    out = ['<svg viewBox="0 0 {} {}" role="img" aria-label="{}" class="chart">'.format(W, H, esc(title))]

    def y(v):
        return T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin))
    for t in range(ymin, ymax + 1, 10):
        out.append('<line x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}" class="grid"/>'.format(L, W - 10, y(t), y(t)))
        out.append('<text x="{}" y="{:.1f}" class="tick" text-anchor="end">{}</text>'.format(L - 6, y(t) + 4, t))
    for gi, (s, lab) in enumerate(groups):
        x0 = L + gi * gw + (gw - bw * len(heads)) / 2
        for hi, h in enumerate(heads):
            d = plan.get(s, {}).get(h)
            if not d:
                continue
            txt, m, sd = ms(d.get(key, []))
            if m is None:
                continue
            x = x0 + hi * bw
            out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="{}" rx="2"><title>{}: {}</title></rect>'.format(
                x + 2, y(max(m, ymin)), bw - 4, y(ymin) - y(max(m, ymin)), HEAD_COLOR.get(h, 'var(--muted)'),
                esc(HEAD_LABEL.get(h, h)), txt))
            if sd:
                out.append('<line x1="{:.1f}" x2="{:.1f}" y1="{:.1f}" y2="{:.1f}" class="err"/>'.format(
                    x + bw / 2, x + bw / 2, y(min(m + sd, ymax)), y(max(m - sd, ymin))))
            out.append('<text x="{:.1f}" y="{:.1f}" class="val" text-anchor="middle">{:.0f}</text>'.format(
                x + bw / 2, y(max(m, ymin)) - 4 - (8 if sd else 0), m))
        out.append('<text x="{:.1f}" y="{}" class="glab" text-anchor="middle">{}</text>'.format(L + gi * gw + gw / 2, H - 12, esc(lab)))
    out.append('</svg>')
    legend = '<div class="legend">' + ''.join('<span><i style="background:{}"></i>{}</span>'.format(
        HEAD_COLOR.get(h, 'var(--muted)'), esc(HEAD_LABEL.get(h, h))) for h in heads) + '</div>'
    return '<div class="chartbox">' + ''.join(out) + legend + '</div>'


def plan_table(plan, sets, heads, ref='ggram'):
    cols = [('success_rate_se3', 'SE(3) success %', 100), ('collision_rate_se3', 'collision %', 100),
            ('nonconv_rate_se3', 'non-conv. %', 100), ('success_rate', 'chart success %', 100),
            ('median_rotgeo_ratio_success_se3', 'rot. path ratio', 1)]
    rows = []
    for s, lab in sets:
        first = True
        for h in heads:
            d = plan.get(s, {}).get(h)
            if not d:
                continue
            cells = []
            for k, _, sc in cols:
                txt, m, _ = ms(d.get(k, []), sc, 1 if sc == 100 else 2)
                cells.append('<td class="num">{}</td>'.format(txt))
            difs = []
            for r_ in (ref, 'gmetric'):
                dif = ''
                if h not in (r_, 'orig') and plan.get(s, {}).get(r_):   # orig: other trainer, its seeds do not pair
                    g = plan[s][r_]
                    common = [sd for sd in d['seeds'] if sd in g['seeds']]
                    if common:
                        dv = [100 * (d['success_rate_se3'][d['seeds'].index(c)] - g['success_rate_se3'][g['seeds'].index(c)])
                              for c in common]
                        dif = '{:+.1f}'.format(st.mean(dv))
                difs.append('<td class="num diff">{}</td>'.format(dif))
            rows.append('<tr class="{}"><td>{}</td><td>{}</td><td class="num">{}</td>{}{}</tr>'.format(
                'grp' if first else '', esc(lab) if first else '', esc(HEAD_LABEL.get(h, h)), len(d['seeds']), ''.join(cells),
                ''.join(difs)))
            first = False
    head = '<tr><th>test set</th><th>head</th><th class="num">seeds</th>' + ''.join(
        '<th class="num">{}</th>'.format(c[1]) for c in cols) + '<th class="num">Δ vs graph_gram</th><th class="num">Δ vs graph_metric</th></tr>'
    return '<div class="tablewrap"><table>' + head + ''.join(rows) + '</table></div>'


def synth_table(syn, probs, heads):
    cols = [('ridge_rel_median', 'near-ridge error %', 100), ('rot_peak_height_rel_err_median', 'ridge height error %', 100),
            ('rot_ridge_err_deg_median', 'ridge position error °', 1), ('uniform_rel_median', 'whole-field error %', 100),
            ('plan_cost_ratio_median', 'gradient-planner cost', 1)]
    rows = []
    for p, lab in probs:
        first = True
        for h in heads:
            d = syn.get(p, {}).get(h)
            if not d:
                continue
            cells = ''.join('<td class="num">{}</td>'.format(ms(d.get(k, []), sc, 2 if sc == 100 else 2)[0]) for k, _, sc in cols)
            rows.append('<tr class="{}"><td>{}</td><td>{}</td><td class="num">{}</td>{}</tr>'.format(
                'grp' if first else '', esc(lab) if first else '', esc(SYN_LABEL.get(h, h)), len(d.get(cols[0][0], [])), cells))
            first = False
    head = '<tr><th>problem</th><th>head</th><th class="num">seeds</th>' + ''.join('<th class="num">{}</th>'.format(c[1]) for c in cols) + '</tr>'
    return '<div class="tablewrap"><table>' + head + ''.join(rows) + '</table></div>'


ATTRIBUTION_TEXT = ('Each change alone is worth about eleven points on the T-shape standard set, through different failures. '
                    'The decoder structure alone (branch K=1, strict cone) removes the non-convergence: 7.5% to 1.7%, with collisions '
                    'still at 8.2%. The conformal goal term alone, bolted onto graph_gram, removes the collisions: 14.7% to 1.9%, through its '
                    'far-field escape-time part, while non-convergence stays at 7.0%. Together they give the best head on every set. On the '
                    'exact-truth problems trained by the physics loss alone, the structure (T ≥ T_free, exact routes) is what separates a '
                    'correct field from a collapsed one.')

SYN_LABEL = {'morig': 'metric baseline net (original, chart input)', 'morig05': 'metric baseline net (original, chart input)',
             'ggram': 'graph_gram', 'gmetric': 'graph_metric', 'gmet05': 'graph_metric', 'br1': 'branch K=1',
             'br1s': 'branch K=1', 'br4': 'branch K=4', 'br4s': 'branch K=4', 'br4P': 'branch K=4, ℓp residual',
             'br4Ps': 'branch K=4, ℓp residual', 'br4L': 'branch K=4, logsumexp residual', 'br4H': 'branch K=4, Huber',
             'br4Hs': 'branch K=4, Huber residual', 'bc4': 'branch K=4, conformal'}


def prop_rows(props):
    """Rows of the property table: (#, property, graph_gram architecture, branch architecture, then the measured value on
    the trained T-scene fields of the metric baseline, graph_gram, graph_metric, branch conformal, branch conformal + end)."""
    heads = ['T_orig', 'T_ggram', 'T_gmetric', 'T_bc4', 'T_bc4e']

    def get(var, key):
        v = props.get(var, {}).get(key)
        return st.mean(v) if v else None

    def pct(v):
        return '–' if v is None else '{:.2f}%'.format(100 * v)

    def num(v, d=2):
        return '–' if v is None else ('{:.%df}' % d).format(v)

    def meas(key, f):
        return [f(get(h, key)) for h in heads]
    return [
        ('1', 'Non-negative, zero exactly at the goal', 'pseudometric: zero off the goal possible', 'yes, T ≥ T_free > 0 off the goal',
         ['max |T(p,p)| ' + ('{:.1e}'.format(v) if v else '0') for v in [get(h, 'P1_goal_max_abs') for h in heads]]),
        ('2', 'Symmetry', 'yes', 'yes (bitwise)', ['max error ' + num(get(h, 'P2_sym_rel_max'), 1) for h in heads]),
        ('3', 'Triangle inequality', 'only for a rank-1 lift', 'not guaranteed (measured)',
         [pct(v) + ' of triples violate' for v in [get(h, 'P3_triangle_viol_rate') for h in heads]]),
        ('4', 'Speed-limit lower bound T ≥ T_free', 'no', 'yes',
         ['{} below; min T/T_free {}'.format(pct(get(h, 'P4_frac_below_Tfree')), num(get(h, 'P4_min_T_over_Tfree'))) for h in heads]),
        ('5', 'Cone at the goal, both endpoints', 'yes', 'yes',
         ['slope change 1e-3→1e-4: ' + pct(get(h, 'P5_cone_conv_e0')) for h in heads]),
        ('6', 'The cone has the metric’s shape', 'no (learned quadratic form)', 'yes (free metric, or scale 1 + ρ(p))',
         ['anisotropy {}; S × slope {}'.format(num(get(h, 'P6_anisotropy_median_e0')), num(get(h, 'P6_eikonal_at_goal_median_e0')))
          for h in heads]),
        ('7', 'Peaks only (no V-shaped kink)', 'yes', 'yes',
         ['V-kinks {} / jumps {} of 200 lines'.format(num(get(h, 'P7_lines_with_V_kink'), 0), num(get(h, 'P7_lines_with_jump'), 0))
          for h in heads]),
        ('8', 'Movable ridges', 'one surface per start, where b = 1', 'yes: wherever two routes or branches tie',
         ['ridges on {} of 200 lines'.format(num(get(h, 'P7_lines_with_ridge'), 0)) for h in heads]),
        ('9', 'No cap on time', 'rotation term ≤ 1/4 (turns²)', 'yes',
         ['pure-rotation slice max T (median) ' + num(get(h, 'P9_rot_slice_Tmax_median')) for h in heads]),
    ]


CSS = """
:root {
  /* layout: one reading column, tables and charts break out to the column's full width */
  --bg: #f6f7f9; --panel: #ffffff; --fg: #18202b; --muted: #5a6574; --rule: #d8dde5;
  --c-branch: #0e7a71; --c-branch2: #2aa198; --c-strict: #3b63ad; --c-gram: #b54a3c; --c-gram2: #d98c7c; --c-metric: #b5822b; --c-orig: #6b4c9a;
  --good: #11704d; --bad: #a33a2c;
  --f-display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --f-body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
  --f-data: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #11151b; --panel: #171c24; --fg: #e3e8ef; --muted: #9aa5b4; --rule: #2b323d;
  --c-branch: #3cc0b3; --c-branch2: #82d8cf; --c-strict: #82a3e6; --c-gram: #e27f70; --c-gram2: #f0b3a6; --c-metric: #e0b25e; --c-orig: #b39ddb;
  --good: #5fd0a0; --bad: #f08a7a; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #11151b; --panel: #171c24; --fg: #e3e8ef; --muted: #9aa5b4; --rule: #2b323d;
  --c-branch: #3cc0b3; --c-branch2: #82d8cf; --c-strict: #82a3e6; --c-gram: #e27f70; --c-gram2: #f0b3a6; --c-metric: #e0b25e; --c-orig: #b39ddb;
  --good: #5fd0a0; --bad: #f08a7a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--f-body); font-size: 16px; line-height: 1.6; }
main { max-width: 980px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px; }
h1, h2, h3 { font-family: var(--f-display); font-weight: 600; text-wrap: balance; line-height: 1.2; }
h1 { font-size: 2.3rem; margin: 0 0 6px; }
h2 { font-size: 1.55rem; margin: 48px 0 10px; padding-top: 12px; border-top: 1px solid var(--rule); }
h3 { font-size: 1.15rem; margin: 26px 0 6px; }
p, li { max-width: 72ch; }
.lede { font-size: 1.07rem; color: var(--fg); }
.muted, figcaption { color: var(--muted); font-size: 0.9rem; }
.meta { color: var(--muted); font-size: 0.85rem; letter-spacing: 0.02em; }
code, .num, .mono { font-family: var(--f-data); font-size: 0.86em; font-variant-numeric: tabular-nums; }
.eq { font-family: var(--f-data); font-size: 0.9rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px;
      padding: 12px 14px; overflow-x: auto; white-space: pre; line-height: 1.5; }
.strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 22px 0 8px; }
.stat { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 12px 14px; min-width: 0; }
.stat b { display: block; font-family: var(--f-data); font-size: 1.45rem; font-weight: 600; color: var(--c-branch); }
.stat span { color: var(--muted); font-size: 0.86rem; }
.tablewrap { overflow-x: auto; margin: 10px 0 6px; border: 1px solid var(--rule); border-radius: 8px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.03em; text-transform: uppercase; white-space: nowrap; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr.grp td { border-top: 2px solid var(--rule); }
td.diff { color: var(--good); font-weight: 600; }
.yes { color: var(--good); font-weight: 600; } .no { color: var(--bad); font-weight: 600; } .part { color: var(--c-metric); font-weight: 600; }
figure { margin: 14px 0; } figure img { width: 100%; border-radius: 6px; border: 1px solid var(--rule); background: #fff; }
.chartbox { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 10px 12px; margin: 12px 0; }
svg.chart { width: 100%; height: auto; display: block; }
svg .grid { stroke: var(--rule); stroke-width: 1; } svg .tick, svg .glab { fill: var(--muted); font: 11px var(--f-data); }
svg .val { fill: var(--fg); font: 10px var(--f-data); } svg .err { stroke: var(--fg); stroke-width: 1.3; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.82rem; color: var(--muted); margin-top: 4px; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.two > * { min-width: 0; }
.note { border-left: 3px solid var(--c-branch); padding: 4px 0 4px 14px; margin: 14px 0; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    plan = load('plan_summary.json')
    syn = load('synth_summary.json')
    props = load('properties.json')
    conf = load('conf_diag.json')
    sup, phy = syn.get('supervised', {}), syn.get('physics', {})
    main_heads = ['orig', 'ggram', 'gmetric', 'bc4', 'bc4e']
    # headline numbers
    def mean_of(s, h, k='success_rate_se3'):
        d = plan.get(s, {}).get(h)
        return ms(d[k])[1] if d else None
    head_stats = []
    names = {'Tshape3d_env1': 'T-shape', 'Lshape3d_env3': 'L-shape', 'Ashape3d_env4': 'A-shape',
             'Tshape3d_env1_bigangle': 'T big-angle set'}
    for s in ('Tshape3d_env1', 'Lshape3d_env3', 'Ashape3d_env4', 'Tshape3d_env1_bigangle'):
        nb, ne, ng = mean_of(s, 'bc4'), mean_of(s, 'bc4e'), mean_of(s, 'ggram')
        nm, no = mean_of(s, 'gmetric'), mean_of(s, 'orig')
        if nb is not None and ng is not None:
            head_stats.append('<div class="stat"><b>{:.1f}%</b><span>{}: branch conformal (with end constraint {:.1f}). graph_metric '
                              '{:.1f}, graph_gram {:.1f}, metric baseline {:.1f}</span></div>'.format(
                                  nb, names[s], ne or float('nan'), nm or float('nan'), ng, no or float('nan')))
    pr = prop_rows(props)
    prop_html = ''.join(
        '<tr><td class="num">{}</td><td>{}</td><td>{}</td><td>{}</td>{}</tr>'.format(
            *[esc(x) for x in r[:4]], ''.join('<td>{}</td>'.format(esc(x)) for x in r[4])) for r in pr)
    parts = []
    parts.append('<title>SE(3) Branch Decoder</title>')
    parts.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    parts.append('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=JetBrains+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap">')
    parts.append('<style>' + CSS + '</style>')
    parts.append('<main>')
    parts.append('<p class="meta">G-Shape-NTFields · models/graph_branch · 2026-10-13</p>')
    parts.append('<h1>SE(3) Branch Decoder</h1>')
    parts.append('<p class="lede">graph_gram decodes its travel time with a fixed formula, T = √(a + φ(b)/s_r²), whose only rotational ridge sits '
                 'where the learned b reaches 1 and whose rotation term is capped at a quarter turn². This study replaces that decoder '
                 'with a minimum over smooth branches, built so that the nine properties of “Decoder properties that hold with or '
                 'without obstacles” hold by construction (the triangle inequality is measured instead), adds constraints at the goal '
                 'end of the field, tests every property, and compares planning among obstacles against graph_gram, graph_metric and the '
                 'original metric NTFields (the metric baseline) on three scenes and three seeds. The full methodology, with proofs, is in '
                 'the LaTeX report results/graph_branch/report/branch_decoder_methodology.pdf.</p>')
    parts.append('<div class="strip">' + ''.join(head_stats) + '</div>')
    parts.append('<p class="muted">Primary metric: SE(3) success of the per-case forward MPPI (1000 RRT-verified cases per scene; '
                 'the pose must be reached on SE(3), with no collision on the way). The metric baseline is the original metric NTFields '
                 '(chart input, combined speed, chart gradient; its own trainer, published checkpoints, seeds 1–3), scored by the same planner. '
                 'Same trainer, loss, data and planner for every other head: '
                 'graph_lie pipeline, Lie-gradient Eikonal residual, isotropic clearance speed, 4000 epochs.</p>')

    # 1 analysis
    parts.append('<h2>1. What the current decoder does</h2>')
    parts.append('<p>Every graph head has three layers. A fixed pose embedding G(x) = (t, vec R) ∈ ℝ¹² keeps the pair map an embedding. '
                 'A siamese G-Shape encoder turns each pose into features e(x) (random Fourier features of G, sine-gated residual blocks, '
                 'input skip). A pair head then builds a smooth latent from the two feature vectors and hands it to a fixed decoder:</p>')
    parts.append('<div class="eq">a = |A (e_x − e_y)|²                       Euclidean embedding distance (graph_gram)\n'
                 'b = (1 − ε) (1 − ⟨P_x, P_y⟩),  P = LLᵀ/|LLᵀ|   PSD Gram lift; P = qqᵀ gives b = sin²(θ/2)\n'
                 'T = √(a/s_t² + φ(b)/s_r²),   φ(b) = 4 asin²√b     exact in free space</div>')
    parts.append('<p>All non-smoothness lives in the decoder: the goal cone (the square root at Q = 0) and one ridge where b → 1. '
                 'Three consequences follow among obstacles.</p><ul>'
                 '<li>The rotational ridge can only be where the two Gram lifts become orthogonal, the half turn of the lift. '
                 'For a rank-1 lift that is a surface; for any higher-rank lift the condition ⟨P_x, P_y⟩ = 0 has codimension 4, '
                 'so generically no rotation slice reaches it and the ridge disappears.</li>'
                 '<li>The rotation term is capped: φ(b)/s_r² ≤ π²/s_r² = 1/4 with rotations in turns, so a pure rotation costs at most 0.5. '
                 'Extra rotational time must go into a, an embedding distance, which has no ridges.</li>'
                 '<li>Nothing keeps T above the speed-limit time T_free, and the cone at the goal is a learned quadratic form rather '
                 'than the metric’s shape.</li></ul>')
    gb = props.get('T_ggram', {})
    if gb.get('P8_bmax_median'):
        parts.append('<div class="note"><b>Measured on the trained T-scene field:</b> along full-turn rotation slices graph_gram’s b peaks at a median of '
                     '{:.2f} and reaches the kink (b > 0.999) on {:.0f}% of slices; its rotation term never exceeds {:.2f}. '
                     'Among obstacles the lift collapses and rotation is carried by the smooth a-latent, so the field has no ridge at all: '
                     'none of 200 random lines crosses one. The forced kink is not misplaced; it is lost.</div>'.format(
                         st.mean(gb['P8_bmax_median']), 100 * st.mean(gb.get('P8_frac_slices_with_kink_b_gt_0.999', [0])),
                         st.mean(gb.get('P9_rotation_term_max', [0]))))

    # 2 decoder
    parts.append('<h2>2. The branch decoder</h2>')
    parts.append('<div class="eq">T(x, y) = min over σ ∈ {+, −}, k = 1..K of   T_free,σ(x, y) + C_σ(x, y) + δ_σ,k(x, y)\n\n'
                 'T_free,σ = √(|t_x − t_y|²/s_t² + ψ_σ²/s_r²),   ψ_+ = 2 atan2(|v|, q_x·q_y),  ψ_− = 2π − ψ_+\n'
                 'δ_σ,k   = |F_k(x̃) − σ F_k(ỹ)|²,   F_k(x̃) = [U_k e(x),  Σ_i q_i W_k,i e(x)]   (even | odd in the quaternion)\n'
                 'C_σ     = ρ̄ λ̄ tanh(T_free,σ / λ̄),   ρ̄ = (ρ_x + ρ_y)/2 ≥ 0   (conformal cone; strict form: C = 0)</div>')
    parts.append('<ul><li><b>Two rotation routes.</b> With unit quaternions of either sign, route + turns by ψ_+ ∈ [0, 2π] and route − by 2π − ψ_+. '
                 'They are the two copies of the ℝ³ × S³ metric under q → −q, and each route keeps its identity through the half turn, '
                 'so it can stay selected as the long way round. Flipping the sign of either lift swaps the routes and their residuals, '
                 'so T is a function on SE(3) × SE(3) whatever signs the quaternion extraction picks.</li>'
                 '<li><b>Route residuals.</b> δ ≥ 0, unbounded, quadratic at the goal, and different for the two routes because part of the '
                 'feature map is odd in the lift. The ridge is wherever two branches tie, so its position and height are learned.</li>'
                 '<li><b>Hard minimum.</b> A soft minimum would break T(x, x) = 0 and T ≥ T_free. The branch is chosen without autograd and only '
                 'the chosen branch is recomputed with gradients, so the unused routes’ singular points cannot put NaN into the Eikonal loss.</li></ul>')
    parts.append(img(os.path.join(FIG, 'figures/fig_decoders.png'),
                     '(a) Exact pure-rotation travel time of the slow-sector problem (A = 4, start yaw 90°): the minimum of two route '
                     'times, each T_free of its route plus a smooth non-negative residual, which is the form of the branch decoder. The '
                     'ridge sits where the routes tie, 70° before the half turn and 0.90 high; graph_gram can only kink at the half '
                     'turn and cannot exceed 0.5. (b) The conformal term near the goal: slope 1 + ρ, saturating to an offset ρλ.'))
    parts.append('<h3>Constraints at the ends of the field</h3>')
    parts.append('<p>At the goal (either endpoint, by symmetry) the decoder gives T(x, x) = 0 exactly, a cone with T = √Q, Q smooth and ∇Q = 0, '
                 'and the cone’s shape is the metric’s: T(p·exp ξ, p) = (1 + ρ(p))·√g(ξ, ξ) + O(|ξ|²). The strict form (ρ = 0) is the note’s '
                 'statement: the free metric’s cone. The note assumes hard obstacles and constant speed; in this pipeline obstacles enter '
                 'through a clearance speed S &lt; 1 (median 0.39 at the training endpoints, 23% below 0.1), where the true '
                 'cone is the speed-weighted metric g/S². The conformal form keeps the metric’s shape and lets the scale follow the speed. '
                 'The <b>end constraint</b> ties that scale to the data: in the limit x → y the Eikonal equation S|c| = 1 becomes '
                 'S(p)(1 + ρ(p)) = 1, imposed at both endpoints of every training pair with the residual form of the Eikonal loss.</p>')

    # 3 properties
    parts.append('<h2>3. The nine properties</h2>')
    parts.append('<p>Architecture columns: what holds for every parameter value (checked at random float64 weights by '
                 '<code>tests/test_branch_decoder.py</code>, one test group per property). Measured columns: the trained T-scene fields '
                 'of the metric baseline, graph_gram, graph_metric and the two conformal branch heads (mean over three seeds), from '
                 '<code>decoder_properties.py</code>. A jump is a step that does not shrink under refinement: the metric baseline’s chart '
                 'seam; every graph head is continuous.</p>')
    parts.append('<div class="tablewrap"><table><tr><th class="num">#</th><th>property</th><th>graph_gram, architecture</th>'
                 '<th>branch, architecture</th><th>metric baseline, trained</th><th>graph_gram, trained</th><th>graph_metric, trained</th>'
                 '<th>branch conformal, trained</th><th>conformal + end, trained</th></tr>' + prop_html + '</table></div>')
    parts.append('<h3>Why the triangle inequality is measured, not built in</h3>')
    parts.append('<p>Properties 3, 4 and the strict form of 6 cannot all hold by construction for a field that sees obstacles. '
                 'Split the free geodesic from x to y into n steps: by 6 each step costs L/n + O(1/n²), by 3 T(x, y) is at most their sum, '
                 'which tends to L = T_free(x, y), and by 4 T ≥ T_free, so T = T_free everywhere. The network is defined on all of SE(3), '
                 'poses inside obstacles included, so obstacles do not interrupt the chain. A strict-cone head with a lower bound therefore '
                 'has to give up 3 to represent anything. With the conformal cone the same chain only bounds T by ∫ 1/S along paths, which is '
                 'the speed-weighted distance itself.</p>')

    # 4 synthetic
    parts.append('<h2>4. Ridges with exact ground truth</h2>')
    parts.append('<p>Among obstacles there is no ground truth for where a ridge belongs, so the ridge and the cap are tested on synthetic '
                 'SE(2)-in-SE(3) problems with exact travel times: a slow yaw sector (A = 4 or 20) puts the rotational ridge 61–96° away from '
                 'the half turn and makes pure rotations cost up to 0.90 or 2.49, above graph_gram’s 0.5 cap; a hard disc gives a '
                 'translational ridge. Every head is fitted to exact labels (representation test) with the same network and budget.</p>')
    parts.append(synth_table(sup, [('sector_A20', 'slow sector, A = 20'), ('sector_A4', 'slow sector, A = 4'),
                                   ('disc_A0', 'hard disc'), ('both_A4', 'disc + sector')],
                             ['morig', 'ggram', 'gmetric', 'br1', 'br4', 'br4L', 'br4P', 'bc4']))
    parts.append(img(os.path.join(FIG, 'figures/fig_synth_fit.png'),
                     'Label fit, slow sector A = 4 (top) and A = 20 (bottom), start yaw 0° and 90°: pure-rotation slices of each fitted head '
                     'against the exact time (black). Every head finds the ridge position; graph_gram (red) rounds it off below its true '
                     'height. The metric baseline net is the original metric NTFields network (chart input) fitted with the same budget.'))
    if phy:
        parts.append('<h3>Trained by the physics loss alone</h3>')
        parts.append('<p>The same sector problems written as NTFields datasets and trained with the Eikonal loss only (anisotropic planar '
                     'speed, no labels), then scored against the exact time.</p>')
        parts.append(synth_table(phy, [('sector_A20', 'slow sector, A = 20'), ('sector_A4', 'slow sector, A = 4')],
                                 ['morig05', 'ggram', 'gmet05', 'br4s', 'br4Hs']))
        parts.append('<p>On the milder sector (A = 4) both earlier heads collapse to the rotation-blind solution: the Eikonal residual '
                     'is almost zero (about 1e-6) because a field that ignores rotation satisfies it, a projection-metric degeneracy of the '
                     'physics-only loss. Their pure-rotation slices are nearly flat and 96–97% of the gradient plans fail. The branch heads '
                     'cannot collapse, since T ≥ T_free already contains the exact rotation routes; they recover the field to 0.6–0.7% and '
                     'place the ridge 70° away from the half turn where the routes tie.</p>')
        parts.append('<p>Without labels graph_gram converges to a field with a false goal: on the slices below its time drops to '
                     'about 0.08 at a goal yaw near 110–135° that is not the start, where the true time is about 0.6, and steepest descent '
                     'stops there. That is the almost-everywhere Eikonal solution with a fake second goal; graph_gram admits it because '
                     'its decoder has no lower bound and can reach zero away from the diagonal. The branch heads cannot (T ≥ T_free > 0 '
                     'off the goal, properties 1 and 4); they put the ridge where the routes tie and every gradient plan reaches the goal.</p>')
        parts.append('<p>The original metric NTFields network, trained with the same loss (its own combined, isotropic speed cannot '
                     'express this anisotropic problem), does not collapse either: its raw rotation-vector input with σ = 1 Fourier '
                     'features is strongly rotation-dependent from the start, whereas the G(x) heads use σ_r = 0.1. Its median error is low '
                     '(1.0% and 0.8%), but its 95th percentile (22% on A = 4) and ridge-height error (33% and 12%) come from the jump at its '
                     'chart seam, which in this problem lies in the middle of the slow sector. On A = 20 it is more accurate than the branch '
                     'heads in the median and the 95th percentile.</p>')
        parts.append(img(os.path.join(FIG, 'figures/fig_synth_phys.png'),
                         'Physics-only training (no labels), slow sector A = 4 (top) and A = 20 (bottom): graph_gram and graph_metric '
                         'collapse to nearly rotation-blind fields (graph_gram has a false minimum near 110–135° for A = 20); the branch heads '
                         'follow the exact time; the metric baseline net follows it except at its chart seam (absolute yaw π, here 90° from '
                         'the start in the right column), where it jumps.'))

    # 5 planning
    parts.append('<h2>5. Planning among obstacles</h2>')
    parts.append(bar_chart([('Tshape3d_env1', 'T-shape'), ('Lshape3d_env3', 'L-shape'), ('Ashape3d_env4', 'A-shape'),
                            ('Tshape3d_env1_bigangle', 'T big-angle'), ('Tshape3d_env1_rotheavy', 'T rotation-heavy')],
                           main_heads, plan, title='SE(3) success by test set', ymin=30))
    parts.append(plan_table(plan, [('Tshape3d_env1', 'T-shape, 1000 cases'), ('Lshape3d_env3', 'L-shape, 1000 cases'),
                                   ('Ashape3d_env4', 'A-shape, 1000 cases'), ('Tshape3d_env1_bigangle', 'T big-angle (191)'),
                                   ('Tshape3d_env1_rotheavy', 'T rotation-heavy (196)')],
                            ['orig', 'ggram', 'gmetric', 'br1', 'br4', 'ggc', 'bc4', 'bc4e', 'bc1e']))
    parts.append('<p class="muted">Δ is the mean paired difference to graph_gram or graph_metric over the seeds both heads have (none for '
                 'the metric baseline, whose checkpoints come from its own trainer). graph_metric rows are '
                 'the existing graph_lie runs (same pipeline, iso / Lie). Rotation path ratio: geodesic rotation of successful paths over '
                 'the straight start–goal angle.</p>')
    # attribution: decoder structure vs the conformal goal constraint
    def cell(s, h, k='success_rate_se3'):
        d = plan.get(s, {}).get(h)
        return ms(d[k])[0] if d else '–'
    att_rows = ''
    for h, lab in [('orig', 'metric baseline (original NTFields)'), ('ggram', 'graph_gram'), ('gmetric', 'graph_metric'),
                   ('ggc', 'graph_gram + conformal term (ablation)'), ('br1', 'branch K=1, strict cone'),
                   ('br4', 'branch K=4, strict cone'), ('bc4', 'branch K=4, conformal cone'), ('bc4e', 'branch K=4, conformal cone + end constraint')]:
        att_rows += '<tr><td>{}</td>{}</tr>'.format(esc(lab), ''.join(
            '<td class="num">{} / {} / {}</td>'.format(cell(s, h), cell(s, h, 'collision_rate_se3'), cell(s, h, 'nonconv_rate_se3'))
            for s in ('Tshape3d_env1', 'Tshape3d_env1_bigangle', 'Tshape3d_env1_rotheavy')))
    parts.append('<h3>Where the gain comes from</h3>')
    parts.append('<p>Two changes separate graph_gram from the best branch head: the decoder structure (exact ±q routes, T ≥ T_free, '
                 'metric-shaped cone, route residuals) and the conformal goal constraint (the cone scale that follows the speed, which far from '
                 'the goal acts as an escape-time cost). The ablation adds only the conformal term to graph_gram’s own decoder.</p>')
    parts.append('<div class="tablewrap"><table><tr><th>head (T-shape)</th><th class="num">standard: success / collision / non-conv. %</th>'
                 '<th class="num">big-angle</th><th class="num">rotation-heavy</th></tr>' + att_rows + '</table></div>')
    parts.append('<p>{}</p>'.format(esc(ATTRIBUTION_TEXT)))
    parts.append(img(os.path.join(FIG, 'figures/fig_failures.png'),
                     'Outcome partition per case on the T-shape sets (mean over seeds): success (bar colour = head), collision, or '
                     'non-convergence without collision.'))
    if conf:
        c = list(conf.values())[-1]
        parts.append('<h3>Why the conformal cone matters here</h3>')
        parts.append('<p>The strict cone has slope 1 at every goal; the Eikonal equation asks for 1/S, and S is small almost everywhere in these '
                     'scenes, so the strict head violates the equation at every slow goal by construction. The conformal head learns the scale '
                     'from the speed: the Spearman correlation between ρ(p) and 1/S(p) − 1 is {:.2f}. Far from the goal the same term saturates '
                     'at ρ̄λ̄, an additive cost that grows when either endpoint is near an obstacle, i.e. a learned escape-time potential, '
                     'the change that cut collisions in the earlier speed-range study.</p>'.format(c.get('spearman_rho_vs_inv_speed', float('nan'))))
        parts.append('<p>The cone calibration below shows that the conformal head learns the cone scale from the Eikonal loss alone '
                     '(median speed × slope 1.01; within 20% of 1/S for S ≳ 0.2). The end constraint changes it little: it raises the slow-goal '
                     'bins by 25–35% (0.16 → 0.23 at S ≈ 0.02) but they stay far below 1, and it costs 1–1.5 points of planning success. '
                     'graph_metric and the metric baseline lose their cone below the logsumexp smoothing scale (0.72 and 0.63 at 1e-2, '
                     '0.20 and 0.06 at 1e-4).</p>')
        parts.append(img(os.path.join(FIG, 'figures/fig_cone.png'),
                         'Speed × cone slope at the goal by local speed (T-shape, 4000 goals, mean of per-seed medians; band: quartiles). '
                         '1 = the Eikonal limit at the goal. Each field uses its own speed model (metric baseline: combined speed and chart '
                         'directions). Left: slope measured 1e-2 from the goal; right: 1e-4.'))
    parts.append(img(os.path.join(FIG, 'figures/T_slices_all_s0.png'),
                     'Trained T-scene fields along rotation slices (top) and translation slices (bottom); dotted: the speed-limit time T_free. '
                     'The conformal branch heads rise where the start passes obstacles; graph_gram and the strict branch head stay low; the '
                     'metric baseline (purple) jumps where a rotation slice crosses its chart seam and its times are two to five times those '
                     'of the graph heads.'))

    # 6 caveats
    parts.append('<h2>6. What is and is not established</h2><ul>'
                 '<li>Established by construction and checked numerically: properties 1, 2, 4, 5, 6, 7, 8 and 9 for every parameter value '
                 'of the branch head; the quaternion lift and the Lie gradient through it are exact.</li>'
                 '<li>Empirical (this machine, these scenes, 3 seeds): the planning gains over graph_gram, graph_metric and the metric '
                 'baseline, the synthetic ridge results, and the mechanism of the conformal cone. The L-shape gain over graph_metric '
                 '(+1.3) is within noise.</li>'
                 '<li>The end constraint is the weakest part: the conformal head already learns the cone scale without it, it leaves '
                 'slow goals too shallow, and it costs 1–1.5 points of success.</li>'
                 '<li>Not guaranteed: the triangle inequality (a squared residual added to T_free is not a metric; violations are rare, '
                 'about 0.06% of random triples). A min of branches rules out V-kinks but not smooth dips, so false minima are not '
                 'excluded by the architecture.</li>'
                 '<li>Among obstacles there is no ground truth for ridge positions; the exact-truth evidence for property 8 comes from the '
                 'synthetic problems. Extra residual branches (K = 4) did not learn the two routes around the hard disc better than K = 1.</li>'
                 '<li>A pure gradient follower is sensitive to every branch switch: on the synthetic problems steepest descent costs up to 18% '
                 'more on branch fields than on graph_metric. The obstacle results use the sampling MPPI.</li>'
                 '<li>On the physics-only synthetic problems the original metric network also avoids the rotation-blind collapse (through its '
                 'rotation-dependent chart features), at the price of its seam.</li></ul>')
    parts.append('<h2>Reproduce</h2><div class="eq">python -m unittest tests.test_branch_decoder\n'
                 'python train/train_graph_branch.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_bc4e_s0 --seed 0 \\\n'
                 '    --arch graph_branch_conf --grad lie --speed iso --end-weight 1 --set n_branch=4\n'
                 'python3 models/graph_branch/scripts/run_branch_queue.py --phase main --gpus 0,1,2\n'
                 'python3 models/graph_branch/scripts/aggregate_branch.py</div>')
    parts.append('<p class="muted">Code: models/graph_method_triangle/branch_decoder.py (heads), models/graph_branch (trainer, end constraint, '
                 'scripts), tests/test_branch_decoder.py. Log: models/graph_branch/EXPERIMENTS.md.</p>')
    parts.append('</main>')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write('\n'.join(parts))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'KB')


if __name__ == '__main__':
    main()
