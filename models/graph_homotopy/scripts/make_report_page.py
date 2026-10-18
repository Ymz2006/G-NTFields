#!/usr/bin/env python3
"""Self-contained HTML report page of the campaign (tables from the aggregated results, figures embedded).

    python3 models/graph_homotopy/scripts/make_report_page.py --out results/graph_homotopy/report/one-cone-one-ridge.html

The narrative (findings) lives in results/graph_homotopy/report/findings.json, written at the end of the campaign:
{"headline": [...], "findings": [[title, text], ...], "variants": {tag: label}, "main_tags": [...], "exploration_tags": [...]}
Standard library only.
"""
import argparse
import base64
import glob
import html
import json
import os
import re
import statistics
from collections import defaultdict

SCENE_NAMES = {'Tshape3d_env1': 'T', 'Lshape3d_env3': 'L', 'Ashape3d_env4': 'A'}


def b64(path):
    with open(path, 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()


def load_runs(res):
    rows = defaultdict(dict)
    for f in glob.glob(os.path.join(res, '*', '*__*.json')):
        run = os.path.basename(os.path.dirname(f))
        if run == 'test':
            continue
        scene, planner = os.path.basename(f)[:-5].split('__')
        try:
            rows[(run, scene)][planner] = json.load(open(f))['summary']
        except Exception:
            pass
    return rows


def ms(vals, digits=1):
    v = [x for x in vals if x is not None]
    if not v:
        return '&ndash;'
    m = 100 * statistics.mean(v)
    if len(v) == 1:
        return '{:.{d}f}'.format(m, d=digits)
    return '{:.{d}f} <span class="pm">&plusmn; {:.{d}f}</span>'.format(m, 100 * statistics.pstdev(v), d=digits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/graph_homotopy/plan')
    ap.add_argument('--fig', default='results/graph_homotopy/figures')
    ap.add_argument('--notes', default='results/graph_homotopy/report/findings.json')
    ap.add_argument('--out', default='results/graph_homotopy/report/one-cone-one-ridge.html')
    a = ap.parse_args()
    notes = json.load(open(a.notes)) if os.path.exists(a.notes) else {}
    labels = notes.get('variants', {})
    rows = load_runs(a.res)
    groups = defaultdict(list)
    for (run, scene), d in rows.items():
        m = re.match(r'([TLA])_(.+)_s(\d+)$', run)
        if m and 'base' in d:
            groups[(m.group(2), SCENE_NAMES.get(scene, scene))].append(d)

    def group_table(tags, scenes=('T', 'L', 'A'), with_prev=None):
        out = ['<div class="tablewrap"><table><thead><tr><th>head</th><th>scene</th><th>seeds</th><th>base planner</th>'
               '<th>collision</th><th>non-conv.</th><th>cover, one plan</th><th>cover, both classes</th><th>shorter-path rule</th><th>either (oracle)</th></tr></thead><tbody>']
        for tag in tags:
            for sc in scenes:
                recs = groups.get((tag, sc), [])
                if not recs:
                    continue
                du = [r.get('dual') for r in recs]
                out.append('<tr><td>{}</td><td>{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td><td class="num">{}</td>'
                           '<td class="num">{}</td><td class="num hi">{}</td><td class="num">{}</td><td class="num dim">{}</td></tr>'.format(
                               html.escape(labels.get(tag, tag)), sc, len(recs),
                               ms([r['base']['success_rate_se3'] for r in recs]),
                               ms([r['base']['collision_rate_se3'] for r in recs]),
                               ms([r['base']['nonconv_rate_se3'] for r in recs]),
                               ms([d and d.get('success_se3_cover') for d in du]),
                               ms([d and d.get('success_rate_se3') for d in du]),
                               ms([d and d.get('success_se3_shorter') for d in du]),
                               ms([d and d.get('success_se3_either') for d in du])))
        out.append('</tbody></table></div>')
        return '\n'.join(out)

    main_tags = notes.get('main_tags', ['hminL', 'hmin'])
    expl_tags = notes.get('exploration_tags', [])
    figs = {}
    for name in ('report/success_by_angle.png', 'report/planners.png'):
        p = os.path.join(a.fig, name)
        if os.path.exists(p):
            figs[name] = b64(p)
    slice_fig = notes.get('slice_figure')
    if slice_fig and os.path.exists(slice_fig):
        figs['slice'] = b64(slice_fig)
    headline = notes.get('headline', [])
    tiles = ''.join('<div class="tile"><div class="tile-v">{}</div><div class="tile-l">{}</div></div>'.format(html.escape(v), html.escape(l))
                    for v, l in headline)
    findings = ''.join('<li><b>{}</b> {}</li>'.format(html.escape(t), html.escape(x)) for t, x in notes.get('findings', []))
    page = """<title>One Cone, One Ridge</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column (68ch) with full-width tables; figures sit in the column */
:root { --bg:#f7f8f8; --fg:#1c2426; --muted:#5b6b70; --line:#d6dedf; --card:#ffffff; --accent:#0e6f6a; --accent-ink:#ffffff; --hi:#e3f2ef; --dim:#8a9a9e;
  --display:'Source Serif 4', Georgia, 'Times New Roman', serif; --body:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; --mono:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141a1c; --fg:#e4eaeb; --muted:#9db0b4; --line:#2b373a; --card:#1b2326; --accent:#4fc1b8; --accent-ink:#0f1a1b; --hi:#1f3331; --dim:#7a8d91; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#141a1c; --fg:#e4eaeb; --muted:#9db0b4; --line:#2b373a; --card:#1b2326; --accent:#4fc1b8; --accent-ink:#0f1a1b; --hi:#1f3331; --dim:#7a8d91; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); line-height: 1.5; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }
main { max-width: 68ch; margin: 0 auto; }
h1 { font-family: var(--display); font-weight: 600; font-size: 2.2rem; line-height: 1.1; margin: 0 0 .3rem; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 1.35rem; margin: 2.2rem 0 .6rem; text-wrap: balance; }
.eyebrow { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: .6rem; }
.lede { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.2rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1.2rem 0; }
.tile { background: var(--card); border: 1px solid var(--line); padding: 12px 14px; }
.tile-v { font-family: var(--mono); font-size: 1.5rem; font-weight: 500; font-variant-numeric: tabular-nums; }
.tile-l { font-size: .8rem; color: var(--muted); margin-top: 2px; }
.tablewrap { overflow-x: auto; margin: .6rem 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; min-width: 640px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; font-size: .78rem; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.hi { background: var(--hi); font-weight: 500; }
td.dim, .pm { color: var(--dim); }
.pm { font-size: .8em; }
figure { margin: 1rem 0; }
figure img { width: 100%; max-width: 100%; display: block; background: #fff; border: 1px solid var(--line); }
figcaption { font-size: .82rem; color: var(--muted); margin-top: .4rem; }
ul.findings { padding-left: 1.1rem; } ul.findings li { margin-bottom: .55rem; }
code, pre { font-family: var(--mono); font-size: .84rem; }
pre { background: var(--card); border: 1px solid var(--line); padding: 10px 12px; overflow-x: auto; }
p { margin: .6rem 0; }
a { color: var(--accent); }
@media (max-width: 480px) { h1 { font-size: 1.7rem; } }
</style>
<main>
<div class="eyebrow">graph_homotopy &middot; 2026-10-17 &middot; one RTX 5070, 8 hours</div>
<h1>One cone, one ridge</h1>
<p class="lede">The SE(3) travel-time field defined on the double cover R&sup3; &times; S&sup3;, deck-invariant by construction, with the time between poses the minimum over the two homotopy classes of paths. Planning descends one sheet of the cover and never meets the ridge.</p>
<div class="tiles">""" + tiles + """</div>
<h2>Findings</h2>
<ul class="findings">""" + findings + """</ul>
<h2>Planning on the cover vs the base field</h2>
<p>SE(3) success (%) of the forward MPPI on the 1000-case RRT-verified standard sets, seed means &plusmn; population std. <i>Base planner</i>: the official planner on T = min(T&#771;(x&#771;, y&#771;), T&#771;(x&#771;, &minus;y&#771;)). <i>Cover, one plan</i>: towards the lift the field prefers at the start. <i>Cover, both classes</i>: both lifted goals planned, the converged plan kept (fewer iterations); the <i>shorter-path rule</i> breaks ties by path length. <i>Either</i>: at least one sheet plan succeeded (an oracle bound).</p>
""" + group_table(main_tags) + ("""<figure><img src=\"""" + figs['report/success_by_angle.png'] + """\" alt="SE(3) success against the relative rotation angle between start and goal for the base planner and the cover planners"><figcaption>Success against the relative rotation between start and goal (T scene, pooled seeds). The base planner loses its cases where the two homotopy classes tie; the cover planner is flat.</figcaption></figure>""" if 'report/success_by_angle.png' in figs else '') + """
<h2>What was tried on the T scene (seed 0)</h2>
""" + group_table(expl_tags, scenes=('T',)) + (("""<figure><img src=\"""" + figs['slice'] + """\" alt="Rotation slice of a trained field showing the two sheets of the cover, their minimum and the ridge"><figcaption>""" + html.escape(notes.get('slice_caption', '')) + """</figcaption></figure>""") if 'slice' in figs else '') + """
<h2>Where things are</h2>
<pre>models/graph_method_triangle/homotopy_decoder.py   head (arch graph_homotopy)
models/graph_homotopy/                              trainer, cover loss, scripts, README.md, EXPERIMENTS.md
train/train_graph_homotopy.py                       training entry point
tests/test_graph_homotopy.py                        architectural property tests
results/graph_homotopy/{plan,summary,diag,figures}  per-case evaluations, tables, field diagnostics, figures</pre>
</main>
"""
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write(page)
    print('written', a.out, len(page) // 1024, 'kB')


if __name__ == '__main__':
    main()
