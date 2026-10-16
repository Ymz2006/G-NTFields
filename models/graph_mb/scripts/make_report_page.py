#!/usr/bin/env python3
"""Results page of the learned-cut campaign (HTML, standard library only), from
results/graph_mb/summary/plan_summary.json (aggregate_mb.py), the branch logs (Experiments/graph_mb/*/branch_log.json),
the diagnostics (results/graph_mb/diag/*.json) and the stage-0 curves.

    python3 models/graph_mb/scripts/make_report_page.py   # -> results/graph_mb/report/learned-cut.html
"""
import html
import json
import math
import os
import statistics as st
import sys

HOST = '/media/jeffrey/G-Shape-NTFields'
SUM = os.path.join(HOST, 'results/graph_mb/summary/plan_summary.json')
EXP = os.path.join(HOST, 'Experiments/graph_mb')
DIAG = os.path.join(HOST, 'results/graph_mb/diag')
OUT = os.path.join(HOST, 'results/graph_mb/report/learned-cut.html')

SETS = [('Tshape3d_env1', 'T standard'), ('Lshape3d_env3', 'L standard'), ('Ashape3d_env4', 'A standard'),
        ('Tshape3d_env1_bigangle', 'T big-angle'), ('Tshape3d_env1_rotheavy', 'T rotation-heavy')]
ROWS = [('orig', 'metric baseline (original NTFields)'), ('gmetric', 'graph_metric'),
        ('bc4', 'branch K=4 conformal (previous)'), ('mb4', 'learned cut, spec training (this work)'),
        ('bc4e', 'branch K=4 conformal + end (previous)'), ('mb4e', 'learned cut + end (this work)'),
        ('mb4F', 'spec free heads'), ('mb4A', 'ablation: active branch only'), ('mb4p', 'proactive splits K=4'),
        ('mb2p', 'proactive splits K=2')]


def esc(x):
    return html.escape(str(x))


def msd(vals, scale=100):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return '', None, 0
    if len(vals) == 1:
        return '{:.1f}'.format(vals[0]), vals[0], 1
    return '{:.1f} ± {:.1f}'.format(st.mean(vals), st.stdev(vals)), st.mean(vals), len(vals)


def success_table(summary):
    head = '<tr><th>SE(3) success %</th>' + ''.join('<th>{}</th>'.format(esc(l)) for _, l in SETS) + '</tr>'
    body = []
    for key, label in ROWS:
        cells = []
        for s, _ in SETS:
            d = summary.get(s, {}).get(key)
            if not d:
                cells.append('<td class="na">–</td>')
                continue
            txt, m, n = msd(d['success_rate_se3'])
            cls = ' class="this"' if key in ('mb4', 'mb4e') else ''
            cells.append('<td{}>{}<span class="n">n={}</span></td>'.format(cls, txt, n))
        body.append('<tr><th scope="row">{}</th>{}</tr>'.format(esc(label), ''.join(cells)))
    return '<table class="num">' + head + ''.join(body) + '</table>'


def detail_table(summary, s):
    keys = [('success_rate_se3', 'SE(3) succ.', 100), ('collision_rate_se3', 'collision', 100),
            ('nonconv_rate_se3', 'non-conv.', 100), ('median_iters_success_se3', 'iters', 1),
            ('median_rotgeo_ratio_success_se3', 'rot. ratio', 1)]
    head = '<tr><th>{}</th>'.format(esc(dict(SETS)[s])) + ''.join('<th>{}</th>'.format(l) for _, l, _ in keys) + '</tr>'
    body = []
    for key, label in ROWS:
        d = summary.get(s, {}).get(key)
        if not d:
            continue
        cells = []
        for k, _, sc in keys:
            vals = [v * sc for v in d.get(k, []) if v is not None]
            if not vals:
                cells.append('<td class="na">–</td>')
            elif len(vals) == 1:
                cells.append('<td>{}</td>'.format('{:.1f}'.format(vals[0]) if sc == 100 else '{:.3g}'.format(vals[0])))
            else:
                cells.append('<td>{}</td>'.format('{:.1f} ± {:.1f}'.format(st.mean(vals), st.stdev(vals)) if sc == 100
                                                  else '{:.3g} ± {:.2g}'.format(st.mean(vals), st.stdev(vals))))
        body.append('<tr><th scope="row">{} <span class="n">n={}</span></th>{}</tr>'.format(
            esc(label), len(d['seeds']), ''.join(cells)))
    return '<table class="num small">' + head + ''.join(body) + '</table>'


def paired(summary, s, a, b):
    da, db = summary.get(s, {}).get(a), summary.get(s, {}).get(b)
    if not da or not db:
        return None
    common = sorted(set(da['seeds']) & set(db['seeds']))
    if not common:
        return None
    ia = {sd: i for i, sd in enumerate(da['seeds'])}
    ib = {sd: i for i, sd in enumerate(db['seeds'])}
    ds = [100 * (da['success_rate_se3'][ia[c]] - db['success_rate_se3'][ib[c]]) for c in common]
    return st.mean(ds), common, ds


def usage_chart(runs, title):
    """Inline SVG: usage of the top unit vs epoch for several runs (from branch_log.json)."""
    W, H, L, B, T_, R = 640, 220, 48, 34, 18, 12
    series = []
    for run, label, col in runs:
        p = os.path.join(EXP, run, 'branch_log.json')
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            log = json.load(fh)
        pts = [(r['epoch'], max(r['usage_unit']), r['events']) for r in log]
        series.append((label, col, pts))
    if not series:
        return ''
    xmax = max(e for _, _, pts in series for e, _, _ in pts)
    def X(e):
        return L + (W - L - R) * e / xmax
    def Y(u):
        return T_ + (H - T_ - B) * (1 - u)
    out = ['<svg viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, esc(title))]
    for u in (0.25, 0.5, 0.75, 1.0):
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" class="grid"/>'.format(L, Y(u), W - R, Y(u)))
        out.append('<text x="{}" y="{:.1f}" class="tick" text-anchor="end">{:.0f} %</text>'.format(L - 6, Y(u) + 4, 100 * u))
    for e in range(0, xmax + 1, 1000):
        out.append('<text x="{:.1f}" y="{}" class="tick" text-anchor="middle">{}</text>'.format(X(e), H - B + 16, e))
    out.append('<text x="{:.1f}" y="{}" class="tick" text-anchor="middle">epoch</text>'.format((L + W - R) / 2, H - 4))
    for label, col, pts in series:
        path = ' '.join('{}{:.1f},{:.1f}'.format('M' if i == 0 else 'L', X(e), Y(u)) for i, (e, u, _) in enumerate(pts))
        out.append('<path d="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(path, col))
        for e, u, ev in pts:
            kinds = {x['event'] for x in ev}
            if 'split' in kinds or 'revive' in kinds:
                out.append('<circle cx="{:.1f}" cy="{:.1f}" r="3.2" fill="{}"/>'.format(X(e), Y(u), col))
            if 'prune' in kinds or 'merge' in kinds:
                out.append('<rect x="{:.1f}" y="{:.1f}" width="6" height="6" fill="{}"/>'.format(X(e) - 3, Y(u) - 3, col))
    lx = L + 8
    for label, col, _ in series:
        out.append('<rect x="{}" y="{}" width="14" height="3" fill="{}"/>'.format(lx, T_ + 4, col))
        out.append('<text x="{}" y="{}" class="legend">{}</text>'.format(lx + 18, T_ + 8, esc(label)))
        lx += 18 + 7 * len(label) + 22
    out.append('</svg>')
    return ''.join(out)


def circle_chart():
    """Inline SVG of the stage-0 single-head field vs the truth (zigzag)."""
    import numpy as np   # the npz needs numpy; the host python has it
    p = os.path.join(HOST, 'results/graph_mb/stage0_split_test/0a_K1_s0_curves.npz')
    if not os.path.exists(p):
        return ''
    z = np.load(p)
    d, V, T = np.degrees(z['d']), z['V'][:, 0], z['T']
    W, H, L, B, T_, R = 640, 230, 44, 34, 14, 12
    def X(a):
        return L + (W - L - R) * a / 360.0
    def Y(v):
        return T_ + (H - T_ - B) * (1 - v / 3.3)
    step = max(1, len(d) // 400)
    out = ['<svg viewBox="0 0 {} {}" role="img" aria-label="circle field vs truth">'.format(W, H)]
    for v in (1, 2, 3):
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" class="grid"/>'.format(L, Y(v), W - R, Y(v)))
        out.append('<text x="{}" y="{:.1f}" class="tick" text-anchor="end">{}</text>'.format(L - 6, Y(v) + 4, v))
    for a in range(0, 361, 60):
        out.append('<text x="{:.1f}" y="{}" class="tick" text-anchor="middle">{}°</text>'.format(X(a), H - B + 16, a))
    tp = ' '.join('{}{:.1f},{:.1f}'.format('M' if i == 0 else 'L', X(d[j]), Y(T[j])) for i, j in enumerate(range(0, len(d), step)))
    vp = ' '.join('{}{:.1f},{:.1f}'.format('M' if i == 0 else 'L', X(d[j]), Y(V[j])) for i, j in enumerate(range(0, len(d), step)))
    out.append('<path d="{}" fill="none" stroke="var(--fg-muted)" stroke-width="1.5" stroke-dasharray="5 4"/>'.format(tp))
    out.append('<path d="{}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>'.format(vp))
    out.append('<text x="{}" y="{}" class="legend">learned single head (slope ±1 almost everywhere, smooth turning points = false goals)</text>'.format(L + 8, T_ + 10))
    out.append('<text x="{}" y="{}" class="legend">dashed: truth min(δ, 360° − δ)</text>'.format(L + 8, T_ + 24))
    out.append('</svg>')
    return ''.join(out)


def diag_table():
    rows = []
    order = ['T_bc4_s0', 'T_bc4_s1', 'T_bc4_s2', 'L_bc4_s0', 'A_bc4_s0', 'T_mb4_s0', 'T_mb4_s1', 'T_mb4_s2',
             'T_mb4e_s0', 'T_mb4p_s0', 'T_mb4F_s0']
    for r in order:
        p = os.path.join(DIAG, r + '.json')
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        rows.append('<tr><th scope="row">{}</th><td>{}</td><td>{}</td><td>{:.1f} %</td><td>{}</td><td>{:.2f} %</td></tr>'.format(
            esc(r), d['surviving_units'], ' '.join('{:.2f}'.format(u) for u in d['usage_unit']), 100 * d['frac_near_eta'],
            '{:.0f} %'.format(100 * d['near_tie_dup_frac']) if 'near_tie_dup_frac' in d else '–',
            100 * d['real_ridge_frac']))
    return ('<table class="num small"><tr><th>checkpoint</th><th>units &gt; 1 %</th><th>usage per unit</th>'
            '<th>pairs within η of a tie</th><th>duplicates among them</th><th>pairs at a real ridge</th></tr>'
            + ''.join(rows) + '</table>')


def main():
    with open(SUM) as fh:
        summary = json.load(fh)
    pair_T = paired(summary, 'Tshape3d_env1', 'mb4', 'bc4')
    pair_rows = []
    for s, label in SETS:
        for a, b in (('mb4', 'bc4'), ('mb4e', 'bc4e'), ('mb4', 'gmetric')):
            pr = paired(summary, s, a, b)
            if pr:
                pair_rows.append('<tr><th scope="row">{}</th><td>{} − {}</td><td>{:+.1f}</td><td>{}</td></tr>'.format(
                    esc(label), a, b, pr[0], ' / '.join('{:+.1f}'.format(x) for x in pr[2])))
    pair_table = ('<table class="num small"><tr><th>set</th><th>pair</th><th>mean paired difference (points)</th>'
                  '<th>per seed</th></tr>' + ''.join(pair_rows) + '</table>')
    css = """
:root{
  /* layout: one 68ch column of running text; tables and charts may run to 880px */
  --bg:#f7f6f2; --surface:#efede6; --fg:#1d2126; --fg-muted:#5b6269; --rule:#d9d6cc;
  --accent:#1f6f78; --accent-soft:#d8ebec; --warn:#a85d1c; --this:#e4f0ea;
  --display:'Source Serif 4', Georgia, 'Times New Roman', serif;
  --body:'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  --mono:'IBM Plex Mono', Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#15181b; --surface:#1d2125; --fg:#e6e4dd; --fg-muted:#a3a79f; --rule:#343a40;
  --accent:#6fc0c8; --accent-soft:#1f3a3e; --warn:#e0a060; --this:#1f332a; color-scheme:dark } }
:root[data-theme="dark"]{
  --bg:#15181b; --surface:#1d2125; --fg:#e6e4dd; --fg-muted:#a3a79f; --rule:#343a40;
  --accent:#6fc0c8; --accent-soft:#1f3a3e; --warn:#e0a060; --this:#1f332a; color-scheme:dark }
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:15.5px;line-height:1.55;
  padding-block:32px 64px;padding-inline:16px}
main{max-width:880px;margin:0 auto}
h1,h2,h3{font-family:var(--display);font-weight:600;text-wrap:balance;line-height:1.2}
h1{font-size:2.1rem;margin:0 0 .3rem}
h2{font-size:1.45rem;margin:2.6rem 0 .8rem;padding-top:.6rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.6rem 0 .5rem}
p,li{max-width:68ch}
.lede{font-size:1.1rem;color:var(--fg-muted);margin:0 0 1.4rem;max-width:68ch}
.eyebrow{font-family:var(--mono);font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin:0 0 .6rem}
.verdict{background:var(--surface);border-left:3px solid var(--accent);padding:14px 18px;margin:1.2rem 0 1.6rem;max-width:72ch}
.verdict p{margin:.3rem 0}
.wrap{overflow-x:auto;margin:.8rem 0 1.2rem}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92rem;min-width:520px}
table.small{font-size:.85rem}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;color:var(--fg-muted);font-family:var(--body)}
th[scope=row]{text-align:left;color:var(--fg);font-weight:500}
tr:first-child th{text-align:right}
tr:first-child th:first-child{text-align:left}
td.this{background:var(--this);font-weight:600}
td.na{color:var(--fg-muted)}
.n{display:block;font-size:.7rem;color:var(--fg-muted);font-family:var(--mono);font-weight:400}
svg{width:100%;height:auto;max-width:880px;display:block;margin:.6rem 0 1rem}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .tick{fill:var(--fg-muted);font-size:11px;font-family:var(--mono)}
svg .legend{fill:var(--fg);font-size:11.5px;font-family:var(--body)}
code,.mono{font-family:var(--mono);font-size:.88em;background:var(--surface);padding:1px 4px;border-radius:3px}
figcaption{font-size:.85rem;color:var(--fg-muted);max-width:68ch;margin-top:-.4rem}
ul{padding-left:1.2rem}
li{margin:.3rem 0}
.warn{color:var(--warn);font-weight:600}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.35rem 1.2rem;max-width:72ch}
dt{font-family:var(--mono);font-size:.85rem;color:var(--fg-muted)}
dd{margin:0}
@media (max-width:520px){dl{grid-template-columns:1fr}}
"""
    tT = summary.get('Tshape3d_env1', {})
    def m(key, s='Tshape3d_env1'):
        d = summary.get(s, {}).get(key)
        return msd(d['success_rate_se3'])[0] if d else '–'
    page = f"""<title>Learned Cut Decoder</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{css}</style>
<main>
<p class="eyebrow">G-Shape-NTFields · SE(3) travel-time fields · 2026-10-14</p>
<h1>Multi-branch decoder that learns the cut</h1>
<p class="lede">The spec's K-branch decoder with band-limited sheet losses, distinctness at ties and branch management, built on the previous campaign's best head (folded min-of-branches with the ±q rotation routes and the conformal cone), trained in the graph_lie pipeline and scored with the per-case MPPI planner on the T, L and A scenes.</p>

<div class="verdict">
<p><strong>The cut is not learned.</strong> In every SE(3) run (24 of 24: three seeds on T, L and A with and without the end constraint, plus the T variants) the detour units die or stay duplicates and the learned branch count is one unit with its two rotation routes. The spec's defaults reach winner-take-all by epoch 300; its revival lands outside the band; without offsets the extra units are duplicates; a gradient-sign split revival produces duplicates that decay. Stage 0 (circle, free heads) does not pass its single-branch prerequisite in an Eikonal-only pipeline.</p>
<p><strong>Planning is unchanged.</strong> With the spec's training procedure the folded conformal head scores {m('mb4')} % SE(3) success on T against {m('bc4')} % for the same head trained winner-only, {m('mb4', 'Lshape3d_env3')} vs {m('bc4', 'Lshape3d_env3')} on L and {m('mb4', 'Ashape3d_env4')} vs {m('bc4', 'Ashape3d_env4')} on A (3 seeds); big-angle {m('mb4', 'Tshape3d_env1_bigangle')} vs {m('bc4', 'Tshape3d_env1_bigangle')}, rotation-heavy {m('mb4', 'Tshape3d_env1_rotheavy')} vs {m('bc4', 'Tshape3d_env1_rotheavy')}. With the end constraint the procedure recovers the 1–1.5 points the constraint used to cost ({m('mb4e')} vs {m('bc4e')} on T, {m('mb4e', 'Tshape3d_env1_bigangle')} vs {m('bc4e', 'Tshape3d_env1_bigangle')} big-angle). Both remain well above graph_metric ({m('gmetric')} on T) and the metric baseline ({m('orig')}).</p>
</div>

<h2>1. Planning results (per-case forward MPPI, batch 125, SE(3) success, mean ± sd over seeds)</h2>
<div class="wrap">{success_table(summary)}</div>
<figcaption>Rows marked in green are this campaign. The previous campaign's heads (bc4, bc4e) and graph_metric are the graph_lie pipeline's own runs with the same seeds; the metric baseline is the original NTFields checkpoints (seeds 1–3, other trainer). Full tables with collision, non-convergence, iterations and path ratios: <code>results/graph_mb/summary/plan_summary.md</code>.</figcaption>

<h3>Paired differences on common seeds</h3>
<div class="wrap">{pair_table}</div>

<h3>Per-set detail</h3>
{''.join('<div class="wrap">' + detail_table(summary, s) + '</div>' for s, _ in SETS)}

<h2>2. What was built</h2>
<dl>
<dt>heads</dt><dd><code>graph_mb</code>: branch = rotation route σ ∈ {{+, −}} × unit k, T = T_free,σ + C_σ + |F_k(x̃) − σF_k(ỹ)|² + o_k with o_0 = 0 (the direct unit carries the goal cone exactly) and detour offsets o_k = softplus β_k; conformal cone C on by default. <code>graph_mb_free</code>: the spec's free heads (direct sqrt(Q_0) with Q_0 = (|A_0Δe|² + ρ|ΔG|²)e^{{2 tanh g_0}}, detours softplus f_k on symmetric features).</dd>
<dt>losses</dt><dd>every branch within Δ of the minimum is evaluated once more on replicated rows, so one autograd call gives all in-band endpoint Lie gradients; decoded Eikonal on the winner where the gap exceeds 3τ, sheet Eikonal on every band row, distinctness hinge on the gradient cosine at ties, optional end constraint. NTFields residual (√(S|c|) − 1)² with the exp(−τ/2) weight.</dd>
<dt>training</dt><dd>graph_lie loop; τ annealed 0.1 → 10⁻³ τ_ref over 60 % (τ_ref = 0.3, the median travel time); 20 % of each batch from the smallest gaps of a second batch; usage checks every 500 steps with revival (copy of the max-residual unit), pruning after two revivals, merging of duplicates; every event in <code>branch_log.json</code>; the <code>alive</code> buffer carries the learned count into the checkpoint.</dd>
<dt>not built</dt><dd>Bellman and sheet-consistency terms (need the speed at arbitrary poses; the datasets carry speeds at sampled pairs only), route anchors, stages 1–2 (fast marching). The soft minimum only enters through the 3τ exclusion zone: without a Bellman target it has no other role.</dd>
<dt>tests</dt><dd>12 unit tests (permutation invariance, exact zero at the goal with positive detours, symmetry, soft→hard min, per-branch gradients vs finite differences, detached masks, revival with optimiser reset, dead units excluded).</dd>
</dl>

<h2>3. What the previous K = 4 heads had learned</h2>
<p>Before any new training, the branch structure of the previous campaign's conformal K = 4 checkpoints was measured on 20 000 training pairs (η = 0.006, Δ = 0.03, γ = 0.95). One unit holds 66–98 % of the pairs, 26–61 % of all pairs are within η of a tie, and 97–99 % of those ties are duplicates (parallel gradients). Real ridges, ties with distinct gradients, are under 1 % of the pairs and are route switches (+q vs −q). This is the spec's "collapse / fake cut" picture, and it explains why the previous campaign found K = 1 ≥ K = 4.</p>
<div class="wrap">{diag_table()}</div>
<figcaption>The new runs (mb4, mb4e, mb4p, mb4F) end with one unit; their near-tie fraction is 0.3–0.4 % and the remaining ties are the route switches.</figcaption>

<h2>4. What happened to the branches</h2>
<figure>
{usage_chart([('T_mb4_s0', 'spec defaults (mb4)', 'var(--accent)'), ('T_mb4z_s0', 'no offsets, revive without raise (mb4z)', 'var(--warn)'), ('T_mb4p_s0', 'proactive gradient-sign splits (mb4p)', 'var(--fg-muted)')], 'top-unit usage vs epoch')}
<figcaption>Share of the held-out pool won by the dominant unit, at every usage check (T scene, seed 0). Dots: revival or split; squares: prune or merge. The spec's defaults reach 100 % by epoch 300 and cycle revive → revive → prune on every detour. Without offsets the second unit survives as a duplicate until the merge rule removes it. Each gradient-sign split holds 20–40 % during its forced warm-up and decays within 200–300 epochs.</figcaption>
</figure>
<p>Three mechanisms were traced:</p>
<ul>
<li><strong>Offsets suppress competition.</strong> A detour at o_k above the direct unit only wins where its residual is smaller by o_k. With random initial residuals the direct unit's residual shrinks fastest and every detour leaves the band within 200 epochs.</li>
<li><strong>The spec's revival is dead on arrival.</strong> A copy raised by 0.2 τ_ref sits outside a band of 0.1 τ_ref, receives no gradient at all, and is pruned after two attempts. A raise inside the band, or none, produces duplicates instead (74 % of pairs within η, 100 % parallel), which the merge rule then masks.</li>
<li><strong>Nothing makes a loser descend.</strong> A copy trained on its band rounds a ridge exactly like its source. The committed two-sheet configuration, each branch continuing its own family through the ridge and leaving the band on the far side, is a zero-loss minimum, but gradient descent from a duplicate never reaches it: the band loss keeps training the loser on the side it should leave. A split by the sign of the source's gradient along a random direction (this campaign's addition) separates the families only where a ridge exists, under 1 % of the pairs; elsewhere the halves are the same family and the pair collapses by a rich-get-richer dynamic.</li>
</ul>

<h2>5. Stage 0 (circle) with free heads</h2>
<figure>
{circle_chart()}
<figcaption>Goal at 0°, unit speed. The single free head converges to a zigzag with slope ±1 almost everywhere and smooth turning points (false goals at 60°, 80°, 180°, 220°, 260°, 320°): median error 25 %, Eikonal residual 0.02 in the ridge band. The spec's residual form (|c|² − 1)² instead leaves a flat plateau (zero slope is a stationary point of that residual). The 1-D Eikonal equation has infinitely many such solutions; the folded SE(3) head excludes them by T ≥ T_free, the spec by its Bellman term, and stage 0 has neither, so its gates (two branches, cut within 1° of 180°) could not be evaluated.</figcaption>
</figure>

<h2>6. Reading of the result</h2>
<p>The spec's premise is that ridges move and multiply among obstacles and that a fixed decoder loses them. On these scenes the measurable ridges are the ±q route switches, which the folded head already represents exactly; the geometric ridges (over versus under a bar) occupy too small a share of the training pairs to sustain a sheet under an Eikonal-only loss, and the loss itself has no term that rewards a branch for taking one side. The procedure around the branches (sampling near ties, training the losing route on its band, masking the decoded term at ties) is neutral on its own and recovers the end constraint's cost; a single seed (T seed 0, 97.7 %) looked like a gain of three points and the other two seeds took it back.</p>
<p>What would change the picture: a Bellman term with a collision oracle (the spec's own remedy for both the false goals of stage 0 and the level of a losing sheet), or route anchors from a planner for a supervised warm start, both of which need information the current datasets do not carry.</p>

<h2>7. Deviations from the spec</h2>
<ul>
<li>Hard minimum throughout; τ only sets the 3τ exclusion zone of the decoded Eikonal.</li>
<li>Revival raise 0.05 τ_ref instead of 0.2 τ_ref (inside the band); split revival and proactive splits are additions; the merge rule is off when splits are used.</li>
<li>NTFields residual and causality weight instead of (|c|² − 1)².</li>
<li>Management events reset the trainer's accept/reject reference and rollback queue (the original loop otherwise rejects every epoch after a split).</li>
<li>Stage 0: the blocked arc is a slow region (S = 0.05) sampled like the rest, as obstacles are in the NTFields data; stages 1–2 not run.</li>
</ul>
<p class="mono" style="font-size:.8rem;color:var(--fg-muted)">Code: models/graph_method_triangle/multibranch.py, models/graph_mb/ (README.md, EXPERIMENTS.md M0–M5), tests/test_multibranch.py. Results: results/graph_mb/.</p>
</main>
"""
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as fh:
        fh.write(page)
    print(OUT)


if __name__ == '__main__':
    sys.exit(main())
