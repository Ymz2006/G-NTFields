#!/usr/bin/env python3
"""Self-contained HTML report page of the expanded comparison (tables from expand_summary.json, figures embedded).

    python3 models/graph_homotopy/scripts/make_expand_report.py --out results/graph_homotopy/report/expanded-tests.html

Narrative: results/graph_homotopy/report/expand_findings.json
{"headline": [[value, label], ...], "findings": [[title, text], ...], "lede": "...", "notes": [...]}. Standard library only.
"""
import argparse
import base64
import html
import json
import os
import statistics

SETS = {'std': 'standard', 'long': 'long (|dp| &ge; 0.6)', 'bigrot': 'big rotation (&ge; 150&deg;)', 'tight': 'tight (clearance 0.005-0.02)',
        'hard': 'hard (tight + long + &ge; 135&deg;)'}
SCENES = {'T': 'T-shape, env1', 'L': 'L-shape, env3', 'A': 'A-shape, env4'}


def b64(path):
    with open(path, 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()


def ms(vals, digits=1):
    v = [x for x in (vals or []) if x is not None]
    if not v:
        return '<span class="dim">&ndash;</span>'
    if len(v) == 1:
        return '{:.{}f}'.format(100 * v[0], digits)
    return '{:.{}f} <span class="pm">&plusmn; {:.{}f}</span>'.format(100 * statistics.mean(v), digits, 100 * statistics.pstdev(v), digits)


def n_of(vals):
    return len([x for x in (vals or []) if x is not None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/graph_homotopy/summary/expand_summary.json')
    ap.add_argument('--fig', default='results/graph_homotopy/figures/expand')
    ap.add_argument('--notes', default='results/graph_homotopy/report/expand_findings.json')
    ap.add_argument('--out', default='results/graph_homotopy/report/expanded-tests.html')
    a = ap.parse_args()
    S = json.load(open(a.summary))
    notes = json.load(open(a.notes)) if os.path.exists(a.notes) else {}
    figs = {n: b64(os.path.join(a.fig, n + '.png')) for n in ('sets', 'bins', 'holes') if os.path.exists(os.path.join(a.fig, n + '.png'))}

    def cell(v, cls='num'):
        return '<td class="{}">{}</td>'.format(cls, v)

    # ---- table 1: targeted + standard
    t1 = ['<div class="tablewrap"><table><thead><tr><th>scene</th><th>set</th><th>homotopy, base planner</th><th>homotopy, cover planner (dual)</th>'
          '<th>NTFields 8000 ep.</th><th>NTFields 4000 ep.</th><th>collision % (dual / NTFields 8k)</th><th>non-convergence % (dual / NTFields 8k)</th></tr></thead><tbody>']
    for r in S['targeted']:
        t1.append('<tr><td>{}</td><td>{}</td>{}{}{}{}{}{}</tr>'.format(
            r['scene'], SETS.get(r['set'], r['set']), cell(ms(r['ours base'])), cell(ms(r['ours dual']), 'num hi'), cell(ms(r['NTFields 8k'])),
            cell(ms(r['NTFields 4k']), 'num dim'), cell('{} / {}'.format(ms(r['coll_ours']), ms(r['coll_mb']))),
            cell('{} / {}'.format(ms(r['nonconv_ours']), ms(r['nonconv_mb'])))))
    t1.append('</tbody></table></div>')
    # ---- table 2: paired
    t2 = ['<div class="tablewrap"><table><thead><tr><th>set</th><th>paired cases</th><th>both succeed</th><th>homotopy only</th><th>NTFields only</th><th>net</th><th>p (exact McNemar)</th>'
          '<th>base planner: homotopy only / NTFields only / p</th></tr></thead><tbody>']
    for r in S['paired']:
        t2.append('<tr><td>{}</td>{}{}{}{}{}{}{}</tr>'.format(html.escape(r['set']), cell(r['n']), cell(r['both']), cell(r['ours_only']), cell(r['theirs_only']),
                                                          cell('{:+d}'.format(r['ours_only'] - r['theirs_only']), 'num hi'), cell('{:.2g}'.format(r['p'])),
                                                          cell('{} / {} / {:.2g}'.format(r['base_ours_only'], r['base_theirs_only'], r['base_p']))))
    t2.append('</tbody></table></div>')
    # ---- table 3: wall
    t3 = ['<div class="tablewrap"><table><thead><tr><th>shape</th><th>hole side</th><th>clearance face-on</th><th>homotopy, base</th><th>homotopy, dual</th>'
          '<th>NTFields 8000 ep.</th><th>NTFields 4000 ep.</th><th>collision % (dual / NTFields 8k)</th><th>non-convergence % (dual / NTFields 8k)</th></tr></thead><tbody>']
    for r in S['wall']:
        t3.append('<tr><td>{}</td><td>{}</td>{}{}{}{}{}{}{}</tr>'.format(
            r['scene'], r['hole'], cell('&ndash;' if r.get('passage_clear') is None else '{:.3f}'.format(r['passage_clear'])),
            cell(ms(r['ours base'])), cell(ms(r['ours dual']), 'num hi'), cell(ms(r['NTFields 8k'])), cell(ms(r['NTFields 4k']), 'num dim'),
            cell('{} / {}'.format(ms(r['coll_ours']), ms(r['coll_mb']))), cell('{} / {}'.format(ms(r['nonconv_ours']), ms(r['nonconv_mb'])))))
    t3.append('</tbody></table></div>')

    tiles = ''.join('<div class="tile"><div class="tile-v">{}</div><div class="tile-l">{}</div></div>'.format(html.escape(v), html.escape(l))
                    for v, l in notes.get('headline', []))
    findings = ''.join('<li><b>{}</b> {}</li>'.format(html.escape(t), html.escape(x)) for t, x in notes.get('findings', []))
    notes_list = ''.join('<li>{}</li>'.format(html.escape(x)) for x in notes.get('notes', []))

    def fig(name, alt, cap):
        if name not in figs:
            return ''
        return '<figure><img src="{}" alt="{}"><figcaption>{}</figcaption></figure>'.format(figs[name], html.escape(alt), cap)

    page = """<title>Expanded Tests</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root { --bg:#f7f8f8; --fg:#1c2426; --muted:#5b6b70; --line:#d6dedf; --card:#ffffff; --accent:#0e6f6a; --accent-ink:#ffffff; --hi:#e3f2ef; --dim:#8a9a9e;
  --display:'Source Serif 4', Georgia, 'Times New Roman', serif; --body:'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; --mono:'IBM Plex Mono', ui-monospace, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#141a1c; --fg:#e4eaeb; --muted:#9db0b4; --line:#2b373a; --card:#1b2326; --accent:#4fc1b8; --accent-ink:#0f1a1b; --hi:#1f3331; --dim:#7a8d91; color-scheme: dark } }
:root[data-theme="dark"] { --bg:#141a1c; --fg:#e4eaeb; --muted:#9db0b4; --line:#2b373a; --card:#1b2326; --accent:#4fc1b8; --accent-ink:#0f1a1b; --hi:#1f3331; --dim:#7a8d91; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); line-height: 1.5; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 32px 64px; }
main { max-width: 72ch; margin: 0 auto; }
h1 { font-family: var(--display); font-weight: 600; font-size: 2.2rem; line-height: 1.1; margin: 0 0 .3rem; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 1.35rem; margin: 2.2rem 0 .6rem; text-wrap: balance; }
.eyebrow { font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin-bottom: .6rem; }
.lede { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.2rem; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 1.2rem 0; }
.tile { background: var(--card); border: 1px solid var(--line); padding: 12px 14px; }
.tile-v { font-family: var(--mono); font-size: 1.5rem; font-weight: 500; font-variant-numeric: tabular-nums; }
.tile-l { font-size: .8rem; color: var(--muted); margin-top: 2px; }
.tablewrap { overflow-x: auto; margin: .6rem 0 1rem; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; min-width: 640px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-weight: 600; font-size: .76rem; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
td.num { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
td.hi { background: var(--hi); font-weight: 500; }
td.dim, .dim, .pm { color: var(--dim); }
.pm { font-size: .8em; }
figure { margin: 1rem 0; }
figure img { width: 100%; max-width: 100%; display: block; background: #fff; border: 1px solid var(--line); }
figcaption { font-size: .82rem; color: var(--muted); margin-top: .4rem; }
ul.findings { padding-left: 1.1rem; } ul.findings li { margin-bottom: .55rem; }
code, pre { font-family: var(--mono); font-size: .84rem; }
pre { background: var(--card); border: 1px solid var(--line); padding: 10px 12px; overflow-x: auto; }
p { margin: .6rem 0; }
@media (max-width: 480px) { h1 { font-size: 1.7rem; } }
</style>
<main>
<div class="eyebrow">graph_homotopy &middot; 2026-10-17 &middot; expanded test sets, metric baseline, walls with holes</div>
<h1>Expanded tests</h1>
<p class="lede">""" + html.escape(notes.get('lede', 'The 8000-epoch double-cover head against the original metric NTFields on targeted test sets (long paths, big rotations, tight endpoints) and on rooms divided by a wall with a shrinking hole.')) + """</p>
<div class="tiles">""" + tiles + """</div>
<h2>Findings</h2>
<ul class="findings">""" + findings + """</ul>
<h2>Targeted and standard sets</h2>
<p>SE(3) success (%) of the forward MPPI, 1000 RRT-verified cases per set, seed means &plusmn; population std (homotopy: 5 seeds; NTFields: 3 seeds). <i>Base planner</i>: the official planner on the base field. <i>Cover planner (dual)</i>: both lifted goals planned, the converged plan kept. NTFields is retrained here with its own trainer on the same 800k-pair arrays, 8000 epochs (its usual 4000-epoch checkpoint alongside).</p>
""" + '\n'.join(t1) + fig('sets', 'SE(3) success per scene and set for the homotopy head and NTFields', 'Success per scene and set. Error bars: population std over seeds.') + """
<h2>Per-case comparison</h2>
<p>Seed-matched pairs (homotopy seed s against NTFields seed s, s = 0, 1, 2, 8000 epochs both), exact McNemar test on the discordant cases.</p>
""" + '\n'.join(t2) + fig('bins', 'SE(3) success by relative rotation, distance and endpoint clearance', 'Success by case property, targeted sets pooled over scenes and seeds.') + """
<h2>Wall with a hole</h2>
<p>The room of env1 divided by a slab (thickness 8 units) with a centred square hole of side 90, 75 or 66 units; the shapes are 60 &times; 10 &times; 60 (T), 35 &times; 10 &times; 60 (L), 54 &times; 10 &times; 60 (A), so face-on the clearance in the hole is 0.071 / 0.036 / 0.014 (normalized; training margin 0.05). Both models are trained per cell on 200k pairs of the same pre-filtered preprocessing and scored on 1000 start-goal pairs on opposite sides of the slab (verified by an aligned passage); a success is a collision-free plan through the hole.</p>
""" + '\n'.join(t3) + fig('holes', 'Success, collision and non-convergence against hole size per shape and model', 'Success, collisions and non-convergence against the hole side (shrinking to the right).') + """
<h2>Notes</h2>
<ul class="findings">""" + notes_list + """</ul>
<h2>Where things are</h2>
<pre>models/graph_homotopy/scripts/make_test_sets.py      targeted / through-the-hole sets (RRT-verified)
models/graph_homotopy/scripts/make_wall_envs.py      hole90 / hole75 / hole66 environments
models/graph_homotopy/scripts/preprocess_hole.py     training data of the hole cells (pre-filtered preprocessing)
train/train_metric_baseline.py                       the original metric NTFields (bundle /home/jeffrey/export_metric)
models/graph_homotopy/scripts/run_expand_queue.py    GPU-memory-aware queue of the whole campaign
results/graph_homotopy/plan_expand/                  per-case evaluations; summary/expand_summary.md; figures/expand/
testing_data/3dshape/, datasets/3dshape/             the new sets and cells (README.md there)</pre>
</main>
"""
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write(page)
    print('written', a.out, len(page) // 1024, 'kB')


if __name__ == '__main__':
    main()
