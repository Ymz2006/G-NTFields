#!/usr/bin/env python3
"""LaTeX table bodies for the methodology report, generated from the summary JSONs (no hand-copied numbers).

    python3 models/graph_branch/scripts/make_latex_tables.py   # -> results/graph_branch/report/latex/tables/*.tex

Inputs: results/graph_branch/summary/{plan_summary,properties,synth_summary}.json and cone_calibration_T.json
(aggregate_branch.py, aggregate_props.py, aggregate_synth.py, cone_calibration.py). Python 3 standard library only.
"""
import json
import os
import statistics as st

HOST = '/media/jeffrey/G-Shape-NTFields'
SUM = os.path.join(HOST, 'results/graph_branch/summary')
OUT = os.path.join(HOST, 'results/graph_branch/report/latex/tables')

NAME = {'orig': 'metric baseline', 'ggram': 'graph\\_gram', 'gmetric': 'graph\\_metric', 'br1': 'branch $K{=}1$, strict',
        'br4': 'branch $K{=}4$, strict', 'ggc': 'graph\\_gram + conf.', 'bc4': 'branch, conformal',
        'bc4e': 'branch, conf.\\ + end', 'bc1e': 'branch $K{=}1$, conf.\\ + end', 'morig': 'metric baseline net',
        'morig05': 'metric baseline net', 'gmet05': 'graph\\_metric', 'br4s': 'branch $K{=}4$, strict',
        'br4Hs': 'branch $K{=}4$, Huber', 'br4L': 'branch $K{=}4$, logsumexp', 'br4P': 'branch $K{=}4$, group-$\\ell_p$'}
SETS = [('Tshape3d_env1', 'T-shape (1000)'), ('Lshape3d_env3', 'L-shape (1000)'), ('Ashape3d_env4', 'A-shape (1000)'),
        ('Tshape3d_env1_bigangle', 'T big-angle (191)'), ('Tshape3d_env1_rotheavy', 'T rotation-heavy (196)')]


def load(name):
    return json.load(open(os.path.join(SUM, name)))


def ms(vals, scale=100.0, d=1, sd=True):
    v = [x * scale for x in vals if x is not None]
    if not v:
        return '--'
    m = st.mean(v)
    if sd and len(v) > 1:
        return '{:.{d}f} $\\pm$ {:.{d}f}'.format(m, st.stdev(v), d=d)
    return '{:.{d}f}'.format(m, d=d)


def best_bold(cells, values):
    """Bold the largest mean in a row."""
    if not values:
        return cells
    i = max(range(len(values)), key=lambda k: -1e9 if values[k] is None else values[k])
    cells[i] = '\\textbf{' + cells[i] + '}'
    return cells


def tab_plan(plan):
    heads = ['orig', 'ggram', 'gmetric', 'br1', 'br4', 'ggc', 'bc4', 'bc4e']
    lines = []
    for s, lab in SETS:
        cells, vals = [], []
        for h in heads:
            d = plan.get(s, {}).get(h)
            if not d:
                cells.append('--')
                vals.append(None)
                continue
            cells.append(ms(d['success_rate_se3']) + ('$^{\\dagger}$' if len(d['seeds']) < 3 else ''))
            vals.append(st.mean(d['success_rate_se3']))
        lines.append(lab + ' & ' + ' & '.join(best_bold(cells, vals)) + ' \\\\')
    return '\n'.join(lines) + '\n'


def tab_plan_detail(plan, s):
    heads = ['orig', 'ggram', 'gmetric', 'br1', 'br4', 'ggc', 'bc4', 'bc4e']
    keys = [('success_rate_se3', 100, 1), ('collision_rate_se3', 100, 1), ('nonconv_rate_se3', 100, 1),
            ('success_rate', 100, 1), ('median_iters_success_se3', 1, 0), ('median_trans_ratio_success_se3', 1, 2),
            ('median_rotgeo_ratio_success_se3', 1, 2)]
    lines = []
    for h in heads:
        d = plan.get(s, {}).get(h)
        if not d:
            continue
        cells = [ms(d[k], sc, dd, sd=(sc == 100)) for k, sc, dd in keys]
        lines.append(NAME[h] + ' & ' + str(len(d['seeds'])) + ' & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines) + '\n'


def tab_props(props, scene='T'):
    heads = ['orig', 'ggram', 'gmetric', 'br4', 'bc4', 'bc4e']

    def g(h, k, f):
        v = props.get('{}_{}'.format(scene, h), {}).get(k)
        if not v:
            return '--'
        return f(st.mean(v))

    pct = lambda v: '{:.2f}\\%'.format(100 * v)                   # noqa: E731
    pct1 = lambda v: '{:.1f}\\%'.format(100 * v)                  # noqa: E731
    num = lambda d: (lambda v: '{:.{d}f}'.format(v, d=d))         # noqa: E731
    sci = lambda v: '0' if v == 0 else '{:.1e}'.format(v)         # noqa: E731
    rows = [
        ('P1', '$\\max_p |T(p,p)|$', 'P1_goal_max_abs', sci),
        ('P3', 'triangle violations (random triples)', 'P3_triangle_viol_rate', pct),
        ('P4', 'pairs with $T < T_{\\mathrm{free}}$', 'P4_frac_below_Tfree', pct),
        ('P4', '$\\min T/T_{\\mathrm{free}}$', 'P4_min_T_over_Tfree', num(2)),
        ('P5', 'cone: slope change $10^{-3}\\to10^{-4}$', 'P5_cone_conv_e0', pct1),
        ('P6', 'cone anisotropy $\\max_\\xi s/\\min_\\xi s$', 'P6_anisotropy_median_e0', num(2)),
        ('P6', '$S_d\\times$ slope at the goal (median)', 'P6_eikonal_at_goal_median_e0', num(2)),
        ('P7', 'lines with a V-kink (of 200)', 'P7_lines_with_V_kink', num(1)),
        ('P7', 'lines with a jump (of 200)', 'P7_lines_with_jump', num(1)),
        ('P8', 'lines with a ridge (of 200)', 'P7_lines_with_ridge', num(1)),
        ('P8', 'rotation slices reaching $b>0.999$', 'P8_frac_slices_with_kink_b_gt_0.999', pct1),
        ('P8', 'route switches $>10^\\circ$ off the half turn', 'P8_frac_switches_beyond_10deg', pct1),
        ('P9', 'largest rotation term $\\varphi(b)/s_r^2$', 'P9_rotation_term_max', num(3)),
        ('P9', 'pure-rotation slice: median $\\max T$', 'P9_rot_slice_Tmax_median', num(2)),
    ]
    lines = []
    for p, lab, k, f in rows:
        lines.append('{} & {} & {} \\\\'.format(p, lab, ' & '.join(g(h, k, f) for h in heads)))
    return '\n'.join(lines) + '\n'


def tab_props_LA(props):
    heads = ['orig', 'ggram', 'gmetric', 'bc4', 'bc4e']
    rows = [('pairs with $T<T_{\\mathrm{free}}$', 'P4_frac_below_Tfree', lambda v: '{:.2f}\\%'.format(100 * v)),
            ('cone anisotropy', 'P6_anisotropy_median_e0', lambda v: '{:.2f}'.format(v)),
            ('$S_d\\times$ slope at the goal', 'P6_eikonal_at_goal_median_e0', lambda v: '{:.2f}'.format(v)),
            ('lines with a ridge / jump', None, None)]
    lines = []
    for sc, scn in (('L', 'L-shape'), ('A', 'A-shape')):
        for i, (lab, k, f) in enumerate(rows):
            cells = []
            for h in heads:
                d = props.get('{}_{}'.format(sc, h), {})
                if k is None:
                    r = d.get('P7_lines_with_ridge')
                    j = d.get('P7_lines_with_jump')
                    cells.append('{:.0f} / {}'.format(st.mean(r), '{:.0f}'.format(st.mean(j)) if j else '--') if r else '--')
                else:
                    v = d.get(k)
                    cells.append(f(st.mean(v)) if v else '--')
            lines.append('{} & {} & {} \\\\'.format(scn if i == 0 else '', lab, ' & '.join(cells)))
        lines.append('\\addlinespace')
    return '\n'.join(lines[:-1]) + '\n'


def tab_synth(syn, kind):
    if kind == 'fit':
        probs = [('sector_A4', 'sector $A{=}4$'), ('sector_A20', 'sector $A{=}20$'), ('disc_A0', 'hard disc'),
                 ('both_A4', 'disc + sector')]
        heads = ['morig', 'ggram', 'gmetric', 'br1', 'br4', 'bc4']
        S = syn['supervised']
    else:
        probs = [('sector_A4', 'sector $A{=}4$'), ('sector_A20', 'sector $A{=}20$')]
        heads = ['morig05', 'ggram', 'gmet05', 'br4s', 'br4Hs']
        S = syn['physics']
    keys = [('uniform_rel_median', 100, 2), ('uniform_rel_p95', 100, 1), ('ridge_rel_median', 100, 2),
            ('rot_ridge_err_deg_median', 1, 1), ('rot_peak_height_rel_err_median', 100, 1), ('plan_success', 100, 1),
            ('plan_cost_ratio_median', 1, 3)]
    lines = []
    for p, lab in probs:
        first = True
        for h in heads:
            d = S.get(p, {}).get(h)
            if not d:
                continue
            n = len(d['uniform_rel_median'])
            cells = [ms(d.get(k, []), sc, dd, sd=False) for k, sc, dd in keys]
            lines.append('{} & {} & {} & {} \\\\'.format(lab if first else '', NAME[h], n, ' & '.join(cells)))
            first = False
        lines.append('\\addlinespace')
    return '\n'.join(lines[:-1]) + '\n'


def tab_cone(cone):
    heads = ['orig', 'ggram', 'gmetric', 'br4', 'bc4', 'bc4e']
    lines = []
    for h in heads:
        runs = [v for k, v in cone['models'].items() if k.rsplit('_s', 1)[0] == h]
        if not runs:
            continue
        cells = []
        for eps in ('0.01', '0.001', '0.0001'):
            cells.append('{:.2f}'.format(st.mean(r['eps_' + eps]['median_all'] for r in runs)))
        lo = [r['eps_0.0001']['bins'][1] for r in runs if r['eps_0.0001']['bins'][1]]
        cells.append('{:.2f}'.format(st.mean(b['median'] for b in lo)) if lo else '--')
        lines.append('{} & {} & {} \\\\'.format(NAME[h], len(runs), ' & '.join(cells)))
    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(OUT, exist_ok=True)
    plan = load('plan_summary.json')
    props = load('properties.json')
    syn = load('synth_summary.json')
    files = {'tab_plan.tex': tab_plan(plan), 'tab_plan_T.tex': tab_plan_detail(plan, 'Tshape3d_env1'),
             'tab_plan_big.tex': tab_plan_detail(plan, 'Tshape3d_env1_bigangle'),
             'tab_props_T.tex': tab_props(props, 'T'), 'tab_props_LA.tex': tab_props_LA(props),
             'tab_synth_fit.tex': tab_synth(syn, 'fit'), 'tab_synth_phys.tex': tab_synth(syn, 'phys')}
    p = os.path.join(SUM, 'cone_calibration_T.json')
    if os.path.exists(p):
        files['tab_cone.tex'] = tab_cone(json.load(open(p)))
    for k, v in files.items():
        with open(os.path.join(OUT, k), 'w') as fh:
            fh.write('% generated by models/graph_branch/scripts/make_latex_tables.py\n' + v)
        print('wrote', k)


if __name__ == '__main__':
    main()
