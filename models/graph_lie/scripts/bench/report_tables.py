#!/usr/bin/env python3
"""Compact tables for results/graph_lie/bench/REPORT.md from summary/bench_summary.json (host, numpy only).

    python3 models/graph_lie/scripts/bench/report_tables.py > results/graph_lie/bench/summary/report_tables.md
"""
import json
import numpy as np

J = json.load(open('results/graph_lie/bench/summary/bench_summary.json'))
SP = [('comb', 'combined'), ('iso', 'iso'), ('aniso', 'aniso')]
SUBS = ['nearid', 'bothfar', 'startfar', 'goalfar', 'perp', 'par', 'longtrans_near', 'longtrans_far', 'tightstart',
        'standard300', 'bigangle', 'rotheavy', 'transonly']
P = J['plan']


print('## T1. SE(3) success: chart-trained / Lie-trained field (%), Δ = Lie − chart; bold Δ: sign test p < 0.01\n')
for scene in ('T', 'L', 'A'):
    samplers = [s for s in ('chart', 'lie') if any(k.startswith('{}|{}|'.format(scene, s)) for k in P)]
    if not samplers:
        continue
    for sa in samplers:
        print('### {} scene, {} MPPI\n'.format(scene, 'official chart' if sa == 'chart' else 'Lie body-twist'))
        print('| subset | ' + ' | '.join('{}: chart / lie (Δ)'.format(lab) for _, lab in SP) + ' | non-conv chart / lie (comb, iso, aniso) |')
        print('|---|' + '---|' * (len(SP) + 1))
        for sub in SUBS:
            cells, nc = [], []
            for sp, lab in SP:
                c = P.get('{}|{}|{}|{}|chart'.format(scene, sa, sub, sp))
                l = P.get('{}|{}|{}|{}|lie'.format(scene, sa, sub, sp))
                if not c or not l:
                    cells.append('-')
                    nc.append('-')
                    continue
                d = l.get('delta') or {}
                star = '**' if d.get('p_case') is not None and d['p_case'] < 0.01 else ''
                cells.append('{:.1f} / {:.1f} ({s}{:+.1f}{s})'.format(c['succ'], l['succ'], d.get('d', float('nan')), s=star))
                nc.append('{:.0f}/{:.0f}'.format(c['nonconv'], l['nonconv']))
            if all(x == '-' for x in cells):
                continue
            print('| {} | {} | {} |'.format(sub, ' | '.join(cells), ', '.join(nc)))
        print()

print('## T2. Predicted travel time of the Lie-trained field relative to the chart-trained field, same query (paired median; p10-p90)\n')
for scene in ('T', 'L', 'A'):
    sa = 'chart' if any(k.startswith(scene + '|chart|') for k in P) else 'lie'
    if not any(k.startswith('{}|{}|'.format(scene, sa)) for k in P):
        continue
    print('### {} scene\n'.format(scene))
    print('| subset | ' + ' | '.join(lab for _, lab in SP) + ' |')
    print('|---|' + '---|' * len(SP))
    for sub in SUBS:
        cells = []
        for sp, lab in SP:
            l = P.get('{}|{}|{}|{}|lie'.format(scene, sa, sub, sp))
            d = (l or {}).get('delta') or {}
            cells.append('{:.3f} ({:.2f}-{:.2f})'.format(d['tau_ratio'], *d['tau_ratio_q']) if d.get('tau_ratio') else '-')
        if all(x == '-' for x in cells):
            continue
        print('| {} | {} |'.format(sub, ' | '.join(cells)))
    rows = [(sp, lab, g, P.get('{}|{}|perp_over_par|{}|{}'.format(scene, sa, sp, g))) for sp, lab in SP for g in ('chart', 'lie')]
    if any(r[3] for r in rows):
        print('\nperp / par (same start, same geodesic angle; chart-segment ratio {:.2f}): '.format(next(r[3]['chart_ratio'] for r in rows if r[3]))
              + '; '.join('{} {}-trained {:.3f}'.format(lab, g, e['tau_ratio']) for sp, lab, g, e in rows if e))
    print()

print('## T3. Saturated-clearance pairs: predicted / exact travel time by start |rotvec| (median over pairs, mean over seeds)\n')
for pf, F in J.get('free', {}).items():
    sc = pf[0]
    print('### {} scene (chart segment / geodesic time of the pairs: {})\n'.format(sc, ', '.join('{:.2f}'.format(x) for x in F['T_chart_over_T_geo'])))
    print('| field | ' + ' | '.join(F['labels']) + ' | paired log(tau_Lie/tau_chart) % |')
    print('|---|' + '---|' * (len(F['labels']) + 1))
    for sp, lab in SP:
        for g in ('chart', 'lie'):
            e = F.get('{}|{}'.format(sp, g))
            if not e:
                continue
            pr = F.get('paired|{}'.format(sp))
            print('| {} / {}-trained | {} | {} |'.format(lab, g, ' | '.join('{:.3f}'.format(x) for x in e['tau_over_Tgeo']),
                                                       ', '.join('{:+.0f}'.format(100 * np.log(x) if x > 0 else float('nan')) for x in pr) if (pr and g == 'lie') else ''))
    print()

print('## T4. Training endpoints: median Eikonal value N with the Lie gradient (1/N = predicted / true speed), by |rotvec|\n')
ts = J.get('train_subsets', {})
bins = ['theta_0_45', 'theta_45_90', 'theta_90_135', 'theta_135_160', 'theta_160_180', 'speed_ge0.5_theta_lt45', 'speed_ge0.5_theta_ge150']
print('| scene | field | ' + ' | '.join(b.replace('theta_', '').replace('_', '-') for b in bins) + ' |')
print('|---|---|' + '---|' * len(bins))
for scene in ('T', 'L', 'A'):
    for sp, lab in SP:
        for g in ('chart', 'lie'):
            vals = [ts.get('{}|{}|{}|{}'.format(scene, b, sp, g), {}).get('N_lie') for b in bins]
            if all(v is None for v in vals):
                continue
            print('| {} | {} / {}-trained | {} |'.format(scene, lab, g, ' | '.join('{:.3f}'.format(v) if v is not None else '-' for v in vals)))
