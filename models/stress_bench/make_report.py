"""Build the stress-benchmark report: results/stress_bench/REPORT.md and report.html (inline SVG charts).

    python3 models/stress_bench/make_report.py
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_svg as sv  # noqa: E402
from aggregate_stress import MODELS, SCENES, SET_LABEL  # noqa: E402

HOST = '/media/jeffrey/G-Shape-NTFields'
SUM = os.path.join(HOST, 'results/stress_bench/summary/stress_summary.json')
SUM_MD = os.path.join(HOST, 'results/stress_bench/summary/stress_summary.md')
OUT_DIR = os.path.join(HOST, 'results/stress_bench')
SHORT = OrderedDict([('orig', 'original NTFields'), ('ggram', 'graph_gram'), ('gmetric', 'graph_metric'),
                     ('psi', 'graph_metric + Ψ'), ('eu2', 'cover euclid'), ('eu2c', 'cover + cone'), ('bc4', 'branch + cone')])

METHODS = OrderedDict([
    ('orig', ('Original metric NTFields (baseline)',
              'Siamese Fourier-feature encoder on the rotation-vector chart (t, rotvec / 2π); travel time from the '
              'NTFields logsumexp-metric decoder on the two latent codes; Eikonal loss with the chart gradient and the '
              'combined (translation + rotation) speed target. Lie-Shape-NTFields metric_baseline, seeds 1-3, epoch 4000.',
              'Symmetric, T(x,x)=0; no triangle inequality; discontinuous at the chart seam |rotvec| = π.')),
    ('ggram', ('graph_gram (first graph decoder)',
               'G-Shape encoder with a linear input skip, a PSD-Gram rotation latent and an analytic decoder: '
               'T² = a + b with a Euclidean position latent and the Gram rotation term; relative-angle curriculum, '
               'cosine LR. Exact in free space (0.1 % error), the weakest head among obstacles.',
               'Metric by construction (symmetric, T(x,x)=0, triangle inequality).')),
    ('gmetric', ('graph_metric (hybrid metric head)',
                 'graph_gram\'s rotation latent with a = (logsumexp metric-head distance)² for the position part; '
                 'trained in the graph_lie pipeline with the body-frame (Lie) gradient and the isotropic speed. The '
                 'reference head of the previous campaigns (T 90.8 / L 96.2 / A 94.4 % SE(3) success).',
                 'Symmetric, T(x,x)=0; the logsumexp part breaks the triangle inequality.')),
    ('psi', ('graph_metric + escape-time potential (sf_psi = fixed)',
             'graph_metric plus an additive potential T = D + (Ψ(x) + Ψ(y))(1 − κ) with Ψ an escape-time potential of '
             'the clearance (gain and ρ trained). The one keeper of the speed-range campaign: cut collisions on the '
             'standard sets from 4-6 % to 0.4-0.6 %.',
             'Symmetric, T(x,x)=0; non-metric.')),
    ('eu2', ('cover euclid, 2 cuts (pseudometric)',
             'Cover-quotient decoder T² = min over g ∈ Z₂^(1+m) of [T_free,σ(g)² + ρ(F_x − g·F_y)], F = [even; q-odd; '
             'm = 2 cut-odd blocks with learned branch points], ρ Euclidean. The only new head with a metric guarantee '
             'for every parameter value.',
             'Pseudometric by construction: symmetry, T(x,x)=0 and the triangle inequality hold exactly.')),
    ('eu2c', ('cover euclid + conformal cone',
              'The cover-quotient decoder with a per-pose conformal factor (cone) that scales the free-space time by a '
              'learned function of the endpoints. Non-metric ablation of eu2 that recovered the slope the metric head '
              'cannot represent.',
              'Symmetric, T(x,x)=0; conformal factor breaks the triangle inequality.')),
    ('bc4', ('branch K=4, conformal cone',
             'Min-of-branches decoder of the "Decoder properties" PDF: K = 4 routes (± q and learned cuts), the time is '
             'the minimum over routes of a conformal cone whose scale is a learned per-pose factor; no end constraint. '
             'Best head of the previous campaigns (T 95.4 / L 97.5 / A 97.9 % SE(3)).',
             'Symmetric, T(x,x)=0; non-metric (min of cones).')),
])

TESTS = [
    ('standard', 'Standard 1000-case sets (endpoint offset 0.02). The baseline test; results reused from the campaigns where '
     'the evaluator settings were identical.', 'reference'),
    ('tight_3d', '1000 cases with endpoint offset 0.005 anywhere in the scene (Lie-Shape-NTFields "tight" sets).', 'tight space'),
    ('tightstart', '300 cases, start clearance in [0.005, 0.015], goal > 0.02, |Δp| in [0.1, 0.4] (graph_lie bench).', 'tight space'),
    ('tightgoal', '300 cases, goal clearance in [0.005, 0.015] (the field\'s source pose), start > 0.02.', 'tight space'),
    ('tightboth', '300 cases, both endpoints in [0.005, 0.015].', 'tight space'),
    ('tightrot', '300 cases, same position (|Δp| ≤ 0.02), both endpoints in [0.005, 0.015], relative rotation ≥ 60°: rotate in place inside clutter.', 'tight space'),
    ('ctrl', '300 cases, endpoints > 0.02, |Δp| in [0.2, 0.5], the straight SE(3) interpolation stays > 0.02 from everything (control).', 'narrow passage'),
    ('squeeze', 'Same endpoint distribution; the straight interpolation is free but passes a gap of clearance ≤ 0.008.', 'narrow passage'),
    ('blocked', 'Same endpoint distribution; ≥ 20 % of the straight interpolation collides (an obstacle sits between start and goal).', 'narrow passage'),
    ('tightgoal_rev', 'tightgoal with start and goal swapped (the same 300 problems planned in the other direction).', 'direction'),
    ('blocked_rev', 'blocked with start and goal swapped.', 'direction'),
    ('longtrans_far', '300 cases, |Δp| in [0.5, 1.2], both |rotvec| in [135°, 180°) (long path inside the chart-distorted region).', 'rotation / length'),
    ('bigangle', '191 cases, both endpoints 135-180° from the identity (rotation cusp stress).', 'rotation / length'),
    ('rotheavy', '200 cases, rotation-dominated moves.', 'rotation / length'),
]


def load():
    return json.load(open(SUM))


def cell(J, S, suffix, m, key):
    k = '{}_{}_{}'.format(S, suffix or 'standard', m)
    if k not in J or not J[k][key]:
        return None, None, 0
    v = np.asarray(J[k][key], dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else None), int(v.size)


def heat(J, S, suffixes, key='success_se3', title='', reverse=False, vmax=100.0):
    rows, vals, subs = [], [], []
    for m in MODELS:
        if not any(cell(J, S, s, m, key)[0] is not None for s in suffixes):
            continue
        rows.append(SHORT[m])
        r, sb = [], []
        for s in suffixes:
            mu, sd, n = cell(J, S, s, m, key)
            r.append(None if mu is None else mu * 100)
            sb.append('' if n <= 1 else '')
        vals.append(r)
        subs.append(sb)
    return sv.heatmap(rows, [SET_LABEL[s] for s in suffixes], vals, title=title, reverse=reverse, vmax=vmax)


def bars(J, S, suffixes, key='success_se3', title='', ymax=100.0, ylabel='%'):
    series = [m for m in MODELS if any(cell(J, S, s, m, key)[0] is not None for s in suffixes)]
    vals = [[(None if cell(J, S, s, m, key)[0] is None else cell(J, S, s, m, key)[0] * 100) for s in suffixes] for m in series]
    errs = [[(cell(J, S, s, m, key)[1] or 0) * 100 for s in suffixes] for m in series]
    return sv.grouped_bars([SET_LABEL[s] for s in suffixes], [SHORT[m] for m in series], vals, errs, ymax=ymax, title=title, ylabel=ylabel) + sv.legend([SHORT[m] for m in series])


def speed_chart(J, S='T'):
    edges = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    series, vals = [], []
    for m in MODELS:
        d = J['field'].get('{}_{}'.format(m, S), {}).get('speed')
        if not d:
            continue
        acc = {e: [] for e in edges}
        for run in d:
            for b in run['bins']:
                acc[round(b['lo'], 3)].append(b['median_ratio'])
        series.append(SHORT[m])
        vals.append([float(np.mean(acc[e])) if acc[e] else None for e in edges])
    if not series:
        return ''
    return sv.log_lines(['{:g}'.format(e) for e in edges], series, vals, title='Predicted / true speed at training endpoints (T scene, median per bin)') + sv.legend(series)


def md_section(md_text, header):
    """Extract one '### header' table block from the summary markdown."""
    lines = md_text.split('\n')
    out, on = [], False
    for ln in lines:
        if ln.startswith('### ') and header in ln:
            on = True
            continue
        if on and ln.startswith('#'):
            break
        if on:
            out.append(ln)
    return '\n'.join(out).strip()


def md_table_to_html(md):
    rows = [r for r in md.split('\n') if r.startswith('|')]
    if len(rows) < 2:
        return ''
    head = [c.strip() for c in rows[0].strip('|').split('|')]
    body = [[c.strip() for c in r.strip('|').split('|')] for r in rows[2:]]
    h = '<div class="tbl"><table><thead><tr>' + ''.join('<th>{}</th>'.format(c) for c in head) + '</tr></thead><tbody>'
    for r in body:
        h += '<tr>' + ''.join('<td>{}</td>'.format(c) for c in r) + '</tr>'
    return h + '</tbody></table></div>'


def main():
    J = load()
    md = open(SUM_MD).read()
    findings = json.load(open(os.path.join(OUT_DIR, 'findings.json'))) if os.path.exists(os.path.join(OUT_DIR, 'findings.json')) else {}
    tight = ['', 'tight_3d', 'tightstart', 'tightgoal', 'tightboth', 'tightrot']
    narrow = ['ctrl', 'squeeze', 'blocked']
    rot = ['', 'longtrans_far', 'bigangle', 'rotheavy']
    all_sets = ['', 'tight_3d', 'tightstart', 'tightgoal', 'tightboth', 'tightrot', 'ctrl', 'squeeze', 'blocked', 'longtrans_far', 'bigangle', 'rotheavy']

    css = """
<style>
/* layout: one reading column (72ch) with full-width chart bands; data-first, lab-notebook treatment */
:root{--bg:#f7f5f0;--fg:#1d1b17;--mute:#6b665c;--line:#d9d4c8;--card:#fffdf8;--accent:#8a3b12;--heat:#2b6b5a;
 --c0:#5c5c5c;--c1:#b08a3c;--c2:#2b6b5a;--c3:#5aa28a;--c4:#7a4ea0;--c5:#c05a8a;--c6:#c2441d;--good:#2b6b5a;--bad:#b03a2e;
 --fd:'Fraunces',Georgia,serif;--fb:'Source Sans 3','Segoe UI',system-ui,sans-serif;--fm:'JetBrains Mono',ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17160f;--fg:#ece7da;--mute:#a39d8f;--line:#3a382f;--card:#201f17;--accent:#e2905f;--heat:#5fc0a4;
 --c0:#a9a9a9;--c1:#e0b45a;--c2:#5fc0a4;--c3:#8fd8bf;--c4:#b08ad8;--c5:#e58bb5;--c6:#ef7a4e;--good:#5fc0a4;--bad:#e07a6f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#17160f;--fg:#ece7da;--mute:#a39d8f;--line:#3a382f;--card:#201f17;--accent:#e2905f;--heat:#5fc0a4;
 --c0:#a9a9a9;--c1:#e0b45a;--c2:#5fc0a4;--c3:#8fd8bf;--c4:#b08ad8;--c5:#e58bb5;--c6:#ef7a4e;--good:#5fc0a4;--bad:#e07a6f;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--fb);font-size:16px;line-height:1.5;padding-block:32px 80px;padding-inline:clamp(16px,4vw,48px)}
main{max-width:1100px;margin:0 auto}
h1{font-family:var(--fd);font-weight:600;font-size:clamp(28px,4vw,40px);line-height:1.1;margin:0 0 8px;text-wrap:balance}
h2{font-family:var(--fd);font-weight:600;font-size:24px;margin:48px 0 12px;text-wrap:balance;border-top:1px solid var(--line);padding-top:20px}
h3{font-size:17px;margin:28px 0 8px;letter-spacing:.01em}
p,li{max-width:72ch}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--mute);margin-bottom:4px}
.lead{font-size:18px;color:var(--mute);max-width:72ch}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);padding:12px 16px;min-width:160px;flex:1 1 160px}
.kpi b{display:block;font-size:26px;font-family:var(--fd);font-variant-numeric:tabular-nums}
.kpi span{font-size:13px;color:var(--mute)}
.chart{width:100%;max-width:100%;height:auto;display:block;margin:12px 0 4px;overflow:visible}
.ct{font-size:13px;fill:var(--fg);font-weight:600}
.cl{font-size:11px;fill:var(--mute)}
.cv{font-size:11px;fill:var(--fg);font-variant-numeric:tabular-nums}
.cv-dark{fill:#fff}
.cell-na{fill:var(--line);fill-opacity:.4}
.grid{stroke:var(--line);stroke-width:1}
.axis{stroke:var(--mute);stroke-width:1}
.err{stroke:var(--fg);stroke-width:1.2;opacity:.6}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--mute);margin:4px 0 12px}
.legend i{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-1px}
.tbl{overflow-x:auto;margin:10px 0 18px;border:1px solid var(--line);background:var(--card)}
table{border-collapse:collapse;font-size:13px;min-width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;vertical-align:top}
th{font-weight:600;background:var(--bg)}
td:first-child,th:first-child{white-space:normal;min-width:180px}
.methods{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.m{background:var(--card);border:1px solid var(--line);padding:14px 16px;min-width:0}
.m h4{margin:0 0 6px;font-size:15px}
.m p{font-size:14px;margin:0 0 6px;color:var(--fg)}
.m .g{font-size:13px;color:var(--mute);font-family:var(--fm)}
.verdict{border-left:3px solid var(--accent);padding:4px 16px;margin:14px 0;background:var(--card)}
.verdict h4{margin:6px 0 4px}
.verdict p{margin:4px 0 8px;font-size:15px}
.tag{display:inline-block;font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:1px 8px;border:1px solid var(--line);border-radius:10px;color:var(--mute);margin-left:6px}
code{font-family:var(--fm);font-size:.9em}
.note{font-size:13px;color:var(--mute)}
a{color:var(--accent)}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
"""
    H = ['<title>SE(3) Field Stress Benchmark</title>', css, '<main>']
    H.append('<div class="eyebrow">G-Shape-NTFields · 2026-10-13 · forward MPPI, SE(3) criterion</div>')
    H.append('<h1>Stress benchmark of the travel-time fields</h1>')
    H.append('<p class="lead">{}</p>'.format(findings.get('lead', 'Six new travel-time heads and the original metric NTFields baseline on 14 planning sets per scene built to expose their weak spots: tight spaces, narrow passages, reversed queries, rotation cusps and long paths, plus field-level consistency checks.')))
    for k in findings.get('kpis', []):
        pass
    if findings.get('kpis'):
        H.append('<div class="kpis">' + ''.join('<div class="kpi"><b>{}</b><span>{}</span></div>'.format(v, l) for v, l in findings['kpis']) + '</div>')

    H.append('<h2>Algorithms under test</h2>')
    H.append('<div class="methods">')
    for m, (name, desc, guar) in METHODS.items():
        H.append('<div class="m"><h4>{} <code>{}</code></h4><p>{}</p><div class="g">{}</div></div>'.format(name, m, desc, guar))
    H.append('</div>')
    H.append('<p class="note">All heads are evaluated with the same per-case planner (the official forward MPPI of Shape-NTFields: 50 samples, horizon 5, step 0.015, 200 iterations, batch 125, torch seed 0) and scored on SE(3) convergence (product geodesic distance to the goal < 0.01); collision = an environment surface point inside the placed shape at any waypoint. T scene: 3 seeds per head (mean ± sd); L and A scenes: seed 0.</p>')

    H.append('<h2>Test design</h2>')
    H.append('<div class="tbl"><table><thead><tr><th>set</th><th>probe</th><th>construction</th></tr></thead><tbody>')
    for s, d, p in TESTS:
        H.append('<tr><td><code>{}</code></td><td>{}</td><td style="white-space:normal">{}</td></tr>'.format(s, p, d))
    H.append('</tbody></table></div>')
    H.append('<p>Field-level checks (no planner): T(x,x); symmetry |T(x,y) − T(y,x)|; the triangle inequality at free midpoints of the ctrl pairs and at random third points; the detour factor, i.e. the median T / d_SE3 on blocked pairs divided by the same on ctrl pairs (1 = the field does not see the obstacle); and the predicted Eikonal speed 1/|∇T| against the training speed label per clearance decade.</p>')

    H.append('<h2>Headline: SE(3) success on every set (T scene)</h2>')
    H.append(heat(J, 'T', all_sets, title='SE(3) success (%)'))
    H.append(heat(J, 'T', all_sets, key='collision_se3', title='collision (%)', reverse=True, vmax=60))
    if findings.get('headline'):
        H.append('<p>{}</p>'.format(findings['headline']))

    H.append('<h2>Tight spaces</h2>')
    H.append(bars(J, 'T', tight, title='SE(3) success (%), T scene'))
    H.append(bars(J, 'T', tight, key='collision_se3', title='collision (%), T scene', ymax=60))
    for s in ('tight_3d', 'tightboth', 'tightrot'):
        H.append('<h3>T / {}</h3>'.format(SET_LABEL[s]) + md_table_to_html(md_section(md, 'T / ' + SET_LABEL[s])))
    if findings.get('tight'):
        H.append('<p>{}</p>'.format(findings['tight']))

    H.append('<h2>Narrow passages</h2>')
    H.append(bars(J, 'T', narrow, title='SE(3) success (%), T scene'))
    H.append(bars(J, 'T', narrow, key='collision_se3', title='collision (%), T scene', ymax=40))
    H.append('<h3>blocked: success (collision) by colliding fraction of the straight line</h3>' + md_table_to_html(md_section(md, 'T / blocked set: SE(3) success')))
    if findings.get('narrow'):
        H.append('<p>{}</p>'.format(findings['narrow']))

    H.append('<h2>Direction asymmetry (same problems, reversed)</h2>')
    for fwd in ('tightgoal', 'blocked'):
        H.append('<h3>{} vs reversed</h3>'.format(fwd) + md_table_to_html(md_section(md, 'T / direction asymmetry: ' + fwd)))
    if findings.get('direction'):
        H.append('<p>{}</p>'.format(findings['direction']))

    H.append('<h2>Rotation cusps and long paths</h2>')
    H.append(bars(J, 'T', rot, title='SE(3) success (%), T scene'))
    H.append(bars(J, 'T', rot, key='collision_se3', title='collision (%), T scene', ymax=60))
    if findings.get('rotation'):
        H.append('<p>{}</p>'.format(findings['rotation']))

    H.append('<h2>L and A scenes (seed 0)</h2>')
    la_sets = ['', 'tight_3d', 'tightstart', 'tightgoal', 'tightboth', 'tightrot', 'ctrl', 'squeeze', 'blocked', 'longtrans_far']
    for S in ('L', 'A'):
        H.append(heat(J, S, la_sets, title='{} scene: SE(3) success (%)'.format(S)))
        H.append(heat(J, S, la_sets, key='collision_se3', title='{} scene: collision (%)'.format(S), reverse=True, vmax=60))
    if findings.get('la'):
        H.append('<p>{}</p>'.format(findings['la']))

    H.append('<h2>Field-level checks</h2>')
    H.append(md_table_to_html(md_section(md, 'T scene: field consistency')))
    H.append(speed_chart(J))
    H.append(md_table_to_html(md_section(md, 'T scene: predicted vs true speed')))
    if findings.get('field'):
        H.append('<p>{}</p>'.format(findings['field']))

    H.append('<h2>Flaws exposed, per algorithm</h2>')
    for m in MODELS:
        v = findings.get('verdict', {}).get(m)
        if v:
            H.append('<div class="verdict"><h4>{} <span class="tag">{}</span></h4><p>{}</p></div>'.format(METHODS[m][0], m, v))
    if findings.get('closing'):
        H.append('<p>{}</p>'.format(findings['closing']))
    H.append('<h2>Reproduction</h2><p class="note">Sets: <code>models/stress_bench/make_stress_sets.py</code> → <code>datasets/3dshape/data/bench/&lt;scene&gt;_&lt;subset&gt;</code>. Queue: <code>models/stress_bench/run_stress.py</code> → <code>results/stress_bench/plan/&lt;model&gt;_&lt;S&gt;_s&lt;seed&gt;/&lt;set&gt;.json</code>. Field checks: <code>field_checks.py</code>, speed calibration: <code>models/graph_lie/scripts/speed_range_eval.py</code>. Tables: <code>aggregate_stress.py</code> → <code>results/stress_bench/summary/stress_summary.md</code>.</p>')
    H.append('</main>')
    with open(os.path.join(OUT_DIR, 'report.html'), 'w') as fh:
        fh.write('\n'.join(H))

    # markdown report = findings + the summary tables
    R = ['# Stress benchmark of the SE(3) travel-time fields (2026-10-13)', '', findings.get('lead', ''), '']
    if findings.get('kpis'):
        R += ['| ' + ' | '.join(l for _, l in findings['kpis']) + ' |', '|' + '---|' * len(findings['kpis']),
              '| ' + ' | '.join(str(v) for v, _ in findings['kpis']) + ' |', '']
    R += ['## Algorithms under test', '']
    for m, (name, desc, guar) in METHODS.items():
        R += ['* **{}** (`{}`): {} *{}*'.format(name, m, desc, guar)]
    R += ['', '## Test design', '', '| set | probe | construction |', '|---|---|---|']
    R += ['| `{}` | {} | {} |'.format(s, p, d) for s, d, p in TESTS]
    for key, title in (('headline', 'Headline'), ('tight', 'Tight spaces'), ('narrow', 'Narrow passages'), ('direction', 'Direction asymmetry'),
                       ('rotation', 'Rotation cusps and long paths'), ('la', 'L and A scenes'), ('field', 'Field-level checks')):
        if findings.get(key):
            R += ['', '## ' + title, '', findings[key]]
    if findings.get('verdict'):
        R += ['', '## Flaws exposed, per algorithm', '']
        for m in MODELS:
            if m in findings['verdict']:
                R += ['* **{}**: {}'.format(METHODS[m][0], findings['verdict'][m])]
    if findings.get('closing'):
        R += ['', findings['closing']]
    R += ['', '---', '', '# Full tables', '', md]
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w') as fh:
        fh.write('\n'.join(R) + '\n')
    print('wrote', os.path.join(OUT_DIR, 'report.html'), os.path.join(OUT_DIR, 'REPORT.md'))


if __name__ == '__main__':
    main()
