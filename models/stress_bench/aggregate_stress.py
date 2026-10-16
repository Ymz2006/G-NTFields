"""Aggregate the stress benchmark: results/stress_bench/{plan,field,speed} -> results/stress_bench/summary/
stress_summary.{json,md}. mean +- sd over seeds; paired direction analysis on the *_rev sets; blocked-set
success binned by how much of the straight line collides.

    python3 models/stress_bench/aggregate_stress.py
"""
import glob
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np

HOST = '/media/jeffrey/G-Shape-NTFields'
PLAN = os.path.join(HOST, 'results/stress_bench/plan')
OUT = os.path.join(HOST, 'results/stress_bench/summary')
SCENES = OrderedDict([('T', 'Tshape3d_env1'), ('L', 'Lshape3d_env3'), ('A', 'Ashape3d_env4')])
MODELS = OrderedDict([
    ('orig', 'original metric NTFields (baseline)'),
    ('ggram', 'graph_gram (first graph decoder)'),
    ('gmetric', 'graph_metric (hybrid metric head)'),
    ('psi', 'graph_metric + escape-time potential'),
    ('eu2', 'cover euclid, 2 cuts (pseudometric)'),
    ('eu2c', 'cover euclid + conformal cone'),
    ('bc4', 'branch K=4, conformal cone'),
])
SET_ORDER = ['', 'tight_3d', 'tightstart', 'tightgoal', 'tightboth', 'tightrot', 'ctrl', 'squeeze', 'blocked',
             'tightgoal_rev', 'blocked_rev', 'longtrans_far', 'bigangle', 'rotheavy']
SET_LABEL = {'': 'standard', 'tight_3d': 'tight_3d (offset 0.005)', 'tightstart': 'tightstart', 'tightgoal': 'tightgoal',
             'tightboth': 'tightboth', 'tightrot': 'tightrot', 'ctrl': 'ctrl (line free)', 'squeeze': 'squeeze (gap <= 0.008)',
             'blocked': 'blocked (line collides)', 'tightgoal_rev': 'tightgoal reversed', 'blocked_rev': 'blocked reversed',
             'longtrans_far': 'longtrans_far', 'bigangle': 'bigangle', 'rotheavy': 'rotheavy'}


def load_all():
    """{(model, S, seed, suffix): json}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(PLAN, '*', '*.json'))):
        run = os.path.basename(os.path.dirname(p))
        model, S, seed = run.rsplit('_', 2)
        seed = int(seed[1:])
        sname = os.path.basename(p)[:-5]
        suffix = sname[len(SCENES[S]):].lstrip('_')
        try:
            out[(model, S, seed, suffix)] = json.load(open(p))
        except Exception:
            pass
    return out


def ms(v, pct=True, nd=1):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size == 0:
        return '-'
    f = 100.0 if pct else 1.0
    if v.size == 1:
        return '{:.{nd}f}'.format(v[0] * f, nd=nd)
    return '{:.{nd}f} ± {:.{nd}f}'.format(v.mean() * f, v.std(ddof=1) * f, nd=nd)


def summary_rows(R):
    """per (S, suffix, model): list over seeds of summary dicts"""
    rows = defaultdict(list)
    for (model, S, seed, suffix), d in R.items():
        s = d['summary']
        rec = d['records']
        s = dict(s)
        s['nonconv_only_se3'] = float(np.mean([(not r['success_se3']) and (not r['collision_se3']) for r in rec])) if rec else None
        s['frac_tight_path'] = float(np.mean([r.get('frac_clear_lt_0p01', 0.0) for r in rec if r['success_se3']])) if rec else None
        rows[(S, suffix, model)].append(s)
    return rows


def table(rows, S, suffix):
    keys = [m for m in MODELS if (S, suffix, m) in rows]
    if not keys:
        return ''
    lines = ['| model | n seeds | n cases | SE(3) success | chart success | collision | non-conv (no coll.) | iters | trans ratio | rot ratio | min clearance |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for m in keys:
        L = rows[(S, suffix, m)]
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            MODELS[m], len(L), L[0]['n_endpoints_ok'], ms([s['success_rate_se3'] for s in L]),
            ms([s['success_rate'] for s in L]), ms([s['collision_rate_se3'] for s in L]),
            ms([s['nonconv_only_se3'] for s in L]), ms([s['median_iters_success_se3'] for s in L], False, 0),
            ms([s['median_trans_ratio_success_se3'] for s in L], False, 2),
            ms([s['median_rotgeo_ratio_success_se3'] for s in L], False, 2),
            ms([s['median_min_clearance_success'] for s in L], False, 4)))
    return '\n'.join(lines)


def matrix(rows, S, key='success_rate_se3', pct=True):
    suffixes = [s for s in SET_ORDER if any((S, s, m) in rows for m in MODELS)]
    lines = ['| model | ' + ' | '.join(SET_LABEL[s] for s in suffixes) + ' |', '|---|' + '---|' * len(suffixes)]
    for m in MODELS:
        if not any((S, s, m) in rows for s in suffixes):
            continue
        cells = [ms([x[key] for x in rows[(S, s, m)]], pct) if (S, s, m) in rows else '-' for s in suffixes]
        lines.append('| {} | '.format(MODELS[m]) + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def paired_direction(R, S, fwd, rev):
    """Same cases planned start->goal and goal->start."""
    out = OrderedDict()
    for m in MODELS:
        seeds = sorted({k[2] for k in R if k[0] == m and k[1] == S and k[3] == fwd and (m, S, k[2], rev) in R})
        if not seeds:
            continue
        both, fonly, ronly, neither, sf, sr, n = [], [], [], [], [], [], 0
        for seed in seeds:
            a = {r['case']: r for r in R[(m, S, seed, fwd)]['records']}
            b = {r['case']: r for r in R[(m, S, seed, rev)]['records']}
            cases = sorted(set(a) & set(b))
            fa = np.array([a[c]['success_se3'] for c in cases])
            fb = np.array([b[c]['success_se3'] for c in cases])
            both.append((fa & fb).mean()); fonly.append((fa & ~fb).mean()); ronly.append((~fa & fb).mean())
            neither.append((~fa & ~fb).mean()); sf.append(fa.mean()); sr.append(fb.mean()); n = len(cases)
        out[m] = {'n_seeds': len(seeds), 'n': n, 'fwd': ms(sf), 'rev': ms(sr), 'both': ms(both), 'fwd_only': ms(fonly),
                  'rev_only': ms(ronly), 'neither': ms(neither)}
    return out


def blocked_bins(R, S):
    """SE(3) success on the blocked set binned by the colliding fraction of the straight line."""
    p = os.path.join(HOST, 'datasets/3dshape/bench', SCENES[S] + '_blocked', 'line_stats.npy')
    if not os.path.exists(p):
        return None, None
    ls = np.load(p)
    edges = [0.2, 0.4, 0.6, 0.8, 1.01]
    out = OrderedDict()
    for m in MODELS:
        seeds = sorted({k[2] for k in R if k[0] == m and k[1] == S and k[3] == 'blocked'})
        if not seeds:
            continue
        per_bin = defaultdict(list)
        coll_bin = defaultdict(list)
        for seed in seeds:
            rec = R[(m, S, seed, 'blocked')]['records']
            f = np.array([ls[r['case'], 0] for r in rec])
            s = np.array([r['success_se3'] for r in rec])
            c = np.array([r['collision_se3'] for r in rec])
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (f >= lo) & (f < hi)
                if sel.sum():
                    per_bin[(lo, hi)].append(s[sel].mean())
                    coll_bin[(lo, hi)].append(c[sel].mean())
        out[m] = OrderedDict(('{:.0f}-{:.0f}%'.format(lo * 100, min(hi, 1) * 100), (ms(per_bin[(lo, hi)]), ms(coll_bin[(lo, hi)]),
                              int(((ls[:, 0] >= lo) & (ls[:, 0] < hi)).sum()))) for lo, hi in zip(edges[:-1], edges[1:]))
    return out, edges


def field_tables():
    F = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HOST, 'results/stress_bench/field', '*.json'))):
        model, S, seed = os.path.basename(p)[:-5].rsplit('_', 2)
        F[(model, S)].append(json.load(open(p)))
    SP = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HOST, 'results/stress_bench/speed', '*.json'))):
        model, S, seed = os.path.basename(p)[:-5].rsplit('_', 2)
        d = json.load(open(p))
        if 'groups' in d:
            d = list(d['groups'].values())[0]
        SP[(model, S)].append(d)
    lines, J = [], OrderedDict()
    for S in SCENES:
        if not any((m, S) in F for m in MODELS):
            continue
        lines += ['', '### {} scene: field consistency (per checkpoint, mean over seeds)'.format(S), '',
                  '| model | n | T(x,x) / median T | asym. median / p95 / max (%) | tri. viol. midpoints (%) | tri. p95 excess (%) | tri. viol. random y (%) | detour factor blocked/ctrl | T/d ctrl | T/d blocked | T/d tightboth | T/d tightrot |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for m in MODELS:
            if (m, S) not in F:
                continue
            L = F[(m, S)]
            g = lambda k, kk: [x[k][kk] for x in L if k in x]  # noqa: E731
            lines.append('| {} | {} | {} | {} / {} / {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                MODELS[m], len(L), ms(g('identity', 'rel_to_median_T'), False, 4), ms(g('symmetry', 'median')),
                ms(g('symmetry', 'p95')), ms(g('symmetry', 'max')), ms(g('tri_mid', 'frac_viol')), ms(g('tri_mid', 'p95_rel')),
                ms(g('tri_rand', 'frac_viol')), ms(g('detour', 'factor'), False, 3), ms(g('detour', 'ctrl_T_over_d_median'), False, 2),
                ms(g('detour', 'blocked_T_over_d_median'), False, 2), ms(g('T_over_d_tightboth', 'median'), False, 2),
                ms(g('T_over_d_tightrot', 'median'), False, 2)))
            J['{}_{}'.format(m, S)] = {'field': L}
        if any((m, S) in SP for m in MODELS):
            lines += ['', '### {} scene: predicted vs true speed at training endpoints (Lie gradient; 1 = calibrated)'.format(S), '',
                      '| model | n | dex bins | dex low (S<0.05) | dex high | N med | median pred/true per true-speed bin (0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5-1) |',
                      '|---|---|---|---|---|---|---|']
            for m in MODELS:
                if (m, S) not in SP:
                    continue
                L = SP[(m, S)]
                bins = defaultdict(list)
                for d in L:
                    for b in d['bins']:
                        bins[round(b['lo'], 3)].append(b['median_ratio'])
                cells = ' '.join('{:.2f}'.format(np.mean(bins[e])) if e in bins else '-' for e in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
                lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                    MODELS[m], len(L), ms([d['dex_bins'] for d in L], False, 3), ms([d['dex_low'] for d in L], False, 3),
                    ms([d['dex_high'] for d in L], False, 3), ms([d['N_med'] for d in L], False, 2), cells))
                J.setdefault('{}_{}'.format(m, S), {})['speed'] = L
    return '\n'.join(lines), J


def main():
    os.makedirs(OUT, exist_ok=True)
    R = load_all()
    rows = summary_rows(R)
    md = ['# Stress benchmark: per-case forward MPPI (plan_eval_graph.py, batch 125, seed 0), SE(3) criterion', '',
          'Models: ' + '; '.join('`{}` = {}'.format(k, v) for k, v in MODELS.items()), '',
          'Sets: standard / tight_3d / bigangle / rotheavy = Lie-Shape-NTFields testing_data; tightstart / longtrans_far = '
          'graph_lie bench; the rest = models/stress_bench/make_stress_sets.py (300 cases, endpoint offset 0.005).', '']
    J = OrderedDict()
    for S in SCENES:
        if not any(k[0] == S for k in rows):
            continue
        md += ['', '## {} scene ({})'.format(S, SCENES[S]), '', '### SE(3) success (%) by set', '', matrix(rows, S), '',
               '### collision (%) by set', '', matrix(rows, S, 'collision_rate_se3'), '',
               '### non-convergence without collision (%) by set', '', matrix(rows, S, 'nonconv_only_se3')]
        for suffix in SET_ORDER:
            t = table(rows, S, suffix)
            if t:
                md += ['', '### {} / {}'.format(S, SET_LABEL[suffix]), '', t]
        for fwd, rev in (('tightgoal', 'tightgoal_rev'), ('blocked', 'blocked_rev')):
            P = paired_direction(R, S, fwd, rev)
            if P:
                md += ['', '### {} / direction asymmetry: {} (start -> goal) vs reversed (same cases)'.format(S, fwd), '',
                       '| model | seeds | n | success fwd | success rev | both | fwd only | rev only | neither |', '|---|---|---|---|---|---|---|---|---|']
                for m, v in P.items():
                    md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(MODELS[m], v['n_seeds'], v['n'], v['fwd'], v['rev'], v['both'], v['fwd_only'], v['rev_only'], v['neither']))
                J['paired_{}_{}'.format(S, fwd)] = P
        B, edges = blocked_bins(R, S)
        if B:
            cols = list(next(iter(B.values())).keys())
            md += ['', '### {} / blocked set: SE(3) success (collision) by colliding fraction of the straight line'.format(S), '',
                   '| model | ' + ' | '.join('{} (n={})'.format(c, next(iter(B.values()))[c][2]) for c in cols) + ' |', '|---|' + '---|' * len(cols)]
            for m, v in B.items():
                md.append('| {} | '.format(MODELS[m]) + ' | '.join('{} ({})'.format(v[c][0], v[c][1]) for c in cols) + ' |')
            J['blocked_bins_{}'.format(S)] = B
    ft, FJ = field_tables()
    md += ['', '## Field-level checks (no planner)', ft]
    with open(os.path.join(OUT, 'stress_summary.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    # json: per (S, suffix, model) means
    for (S, suffix, m), L in rows.items():
        J['{}_{}_{}'.format(S, suffix or 'standard', m)] = {
            'n_seeds': len(L), 'n_cases': L[0]['n_endpoints_ok'],
            'success_se3': [s['success_rate_se3'] for s in L], 'success_chart': [s['success_rate'] for s in L],
            'collision_se3': [s['collision_rate_se3'] for s in L], 'nonconv_only_se3': [s['nonconv_only_se3'] for s in L],
            'iters': [s['median_iters_success_se3'] for s in L], 'trans_ratio': [s['median_trans_ratio_success_se3'] for s in L],
            'rot_ratio': [s['median_rotgeo_ratio_success_se3'] for s in L], 'min_clearance': [s['median_min_clearance_success'] for s in L],
            'reused': [s.get('reused_from') for s in L]}
    J['field'] = FJ
    with open(os.path.join(OUT, 'stress_summary.json'), 'w') as fh:
        json.dump(J, fh, indent=1)
    print('\n'.join(md[:60]))


if __name__ == '__main__':
    main()
