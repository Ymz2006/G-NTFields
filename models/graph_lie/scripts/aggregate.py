"""Aggregate graph_lie planning results and field diagnostics into tables.

Reads results/graph_lie/plan/<run>/<set>.json (plan_eval_graph.py records; run names
<env>_<head>_<speed>_<grad>_s<seed>) and results/graph_lie/diag/<run>.json (field_diag.py), writes
results/graph_lie/summary/{summary.md, summary.json, runs.csv}.

Primary metric: SE(3) success (the robot reaches the goal pose on SE(3), sqrt(|dp|^2 + (theta / 2 pi)^2)
< 0.01, without collision); the official chart criterion is reported alongside. Chart success
undercounts seam-free fields such as the graph heads (results/graph_method_cusp/REPORT.md, Section 4.2).
Lie vs chart comparisons are paired: same head, speed formulation, scene and seed (identical
initialisation and data order), same test cases and planner noise. Per pair the difference in success
rate is taken; across seeds a paired t-test; per case, discordant successes are pooled over seeds and
compared with an exact two-sided sign (McNemar) test.

    python models/graph_lie/scripts/aggregate.py   (inside pytorchserver, from the repo root)
"""
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = 'results/graph_lie'
PLAN = os.path.join(ROOT, 'plan')
DIAG = os.path.join(ROOT, 'diag')
OUT = os.path.join(ROOT, 'summary')
NAME = re.compile(r'^(?P<env>[TLA])_(?P<head>gmetric|trieq)_(?P<speed>comb|iso|aniso)_(?P<grad>chart|lie)_s(?P<seed>\d+)$')
STD = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
EXTRA = {'T': ['Tshape3d_env1_bigangle', 'Tshape3d_env1_rotheavy', 'Tshape3d_env1_transonly', 'Tshape3d_env1_tight_3d'],
         'L': ['Lshape3d_env3_tight_3d'], 'A': ['Ashape3d_env4_tight_3d']}
HEAD_LABEL = {'gmetric': 'graph_metric', 'trieq': 'triangle (graph_tri_metric, eq. V)'}
SPEED_LABEL = {'comb': 'combined S|g|=1', 'iso': 'iso S_d|g|=1', 'aniso': 'aniso (S_d,S_a)'}
ANGLE_BINS = [(0, 90), (90, 150), (150, 180.01)]


def load(plan_dir=PLAN):
    runs = {}
    for d in sorted(glob.glob(os.path.join(plan_dir, '*'))):
        m = NAME.match(os.path.basename(d))
        if not m:
            continue
        key = (m['env'], m['head'], m['speed'], m['grad'], int(m['seed']))
        sets = {}
        for f in glob.glob(os.path.join(d, '*.json')):
            J = json.load(open(f))
            sets[os.path.basename(f)[:-5]] = J
        runs[key] = sets
    diag = {}
    for f in glob.glob(os.path.join(DIAG, '*.json')):
        m = NAME.match(os.path.basename(f)[:-5])
        if m:
            diag[(m['env'], m['head'], m['speed'], m['grad'], int(m['seed']))] = json.load(open(f))
    return runs, diag


def pct(x):
    return 100.0 * x


def ms(v, digits=1):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    if len(v) == 1:
        return '{:.{d}f}'.format(v[0], d=digits)
    return '{:.{d}f} ± {:.{d}f}'.format(np.mean(v), np.std(v, ddof=1), d=digits)


def angle_of(rec):
    return rec['straight_rot_geo'] * 360.0


def set_metrics(J):
    s, R = J['summary'], J['records']
    out = {'succ_se3': pct(s['success_rate_se3']), 'succ_chart': pct(s['success_rate']),
           'coll_se3': pct(s['collision_rate_se3']), 'coll': pct(s['collision_rate']),
           'nc_se3': pct(s['nonconv_rate_se3']), 'nc': pct(s['nonconv_rate']),
           'iters_se3': s['median_iters_success_se3'], 'trans_ratio': s['median_trans_ratio_success_se3'],
           'rot_ratio': s['median_rotgeo_ratio_success_se3'], 'n': s['n_endpoints_ok']}
    for lo, hi in ANGLE_BINS:
        sel = [r for r in R if lo <= angle_of(r) < hi]
        out['succ_se3_{}_{}'.format(lo, int(hi))] = pct(np.mean([r['success_se3'] for r in sel])) if sel else None
    return out


def paired(runs, env, head, speed, sset, metric='succ_se3', case_key='success_se3'):
    """Lie - chart over seeds with both runs evaluated on sset."""
    diffs, lie_only, chart_only, seeds = [], 0, 0, []
    for (e, h, sp, g, seed), sets in runs.items():
        if (e, h, sp, g) != (env, head, speed, 'lie') or sset not in sets:
            continue
        other = runs.get((env, head, speed, 'chart', seed), {}).get(sset)
        if other is None:
            continue
        a, b = set_metrics(sets[sset]), set_metrics(other)
        diffs.append(a[metric] - b[metric])
        seeds.append(seed)
        ra = {r['case']: r[case_key] for r in sets[sset]['records']}
        rb = {r['case']: r[case_key] for r in other['records']}
        for c in set(ra) & set(rb):
            lie_only += int(ra[c] and not rb[c])
            chart_only += int(rb[c] and not ra[c])
    if not diffs:
        return None
    p_t = float(stats.ttest_1samp(diffs, 0.0).pvalue) if len(diffs) > 1 and np.std(diffs) > 0 else None
    n_disc = lie_only + chart_only
    p_case = float(stats.binomtest(lie_only, n_disc, 0.5).pvalue) if n_disc else None
    return {'n': len(diffs), 'seeds': sorted(seeds), 'mean': float(np.mean(diffs)),
            'sd': float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None, 'diffs': diffs, 'p_t': p_t,
            'lie_only': lie_only, 'chart_only': chart_only, 'p_case': p_case}


def fmt_p(p):
    if p is None:
        return '-'
    return '{:.3f}'.format(p) if p >= 0.001 else '{:.0e}'.format(p)


def main():
    runs, diag = load()
    os.makedirs(OUT, exist_ok=True)
    L = ['# models/graph_lie: Lie gradient vs chart (partial-derivative) gradient', '',
         'Generated by `models/graph_lie/scripts/aggregate.py` from `results/graph_lie/plan/*/*.json` and '
         '`results/graph_lie/diag/*.json`. Success in %, mean ± sd over seeds (n). SE(3) = goal reached on SE(3) '
         '(primary); chart = official chart-distance criterion. Δ = Lie − chart, paired by seed; p_t = paired '
         't-test over seeds; cases = per-case discordant successes pooled over seeds (Lie-only / chart-only) '
         'with an exact sign test p_case.', '']
    configs = [(h, sp) for h in ('gmetric', 'trieq') for sp in ('comb', 'iso', 'aniso')]
    summary = {'runs': {}, 'paired': {}, 'diag': {}}
    # ---------------------------------------------------------------- per-run CSV
    with open(os.path.join(OUT, 'runs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        cols = ['succ_se3', 'succ_chart', 'coll_se3', 'nc_se3', 'nc', 'iters_se3', 'trans_ratio', 'rot_ratio', 'n']
        w.writerow(['env', 'head', 'speed', 'grad', 'seed', 'set'] + cols)
        for key in sorted(runs):
            for sset, J in sorted(runs[key].items()):
                m = set_metrics(J)
                w.writerow(list(key) + [sset] + [m[c] for c in cols])
                summary['runs']['{}_{}_{}_{}_s{}|{}'.format(*key, sset)] = m
    # ---------------------------------------------------------------- standard sets
    L += ['# Planner: official forward MPPI (samples in the rotation-vector chart)', '']
    for env in ('T', 'L', 'A'):
        sset = STD[env]
        L += ['## {} — standard 1000-case set (`{}`)'.format(env, sset), '',
              '| head | speed formulation | gradient | n | SE(3) success | chart success | collision | SE(3) non-conv | iters | trans ratio | rot ratio |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for h, sp in configs:
            for g in ('chart', 'lie'):
                M = [set_metrics(s[sset]) for (e, hh, ss, gg, _), s in sorted(runs.items())
                     if (e, hh, ss, gg) == (env, h, sp, g) and sset in s]
                if not M:
                    continue
                L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    HEAD_LABEL[h], SPEED_LABEL[sp], g, len(M), ms([m['succ_se3'] for m in M]),
                    ms([m['succ_chart'] for m in M]), ms([m['coll_se3'] for m in M]), ms([m['nc_se3'] for m in M]),
                    ms([m['iters_se3'] for m in M], 0), ms([m['trans_ratio'] for m in M], 2),
                    ms([m['rot_ratio'] for m in M], 2)))
        L.append('')
        L += ['**Lie − chart, paired** (`{}`)'.format(sset), '',
              '| head | speed formulation | n pairs | Δ SE(3) success | p_t | cases Lie-only / chart-only | p_case | Δ chart success | Δ collision |',
              '|---|---|---|---|---|---|---|---|---|']
        for h, sp in configs:
            P = paired(runs, env, h, sp, sset)
            if P is None:
                continue
            Pc = paired(runs, env, h, sp, sset, 'succ_chart', 'success')
            Pk = paired(runs, env, h, sp, sset, 'coll_se3', 'collision_se3')
            summary['paired']['{}|{}|{}|{}'.format(env, h, sp, sset)] = {'se3': P, 'chart': Pc, 'coll': Pk}
            L.append('| {} | {} | {} | {:+.1f} ± {} | {} | {} / {} | {} | {:+.1f} | {:+.1f} |'.format(
                HEAD_LABEL[h], SPEED_LABEL[sp], P['n'], P['mean'], '{:.1f}'.format(P['sd']) if P['sd'] is not None else '-',
                fmt_p(P['p_t']), P['lie_only'], P['chart_only'], fmt_p(P['p_case']), Pc['mean'], Pk['mean']))
        L.append('')
    # ---------------------------------------------------------------- extra sets
    for env in ('T', 'L', 'A'):
        sets_here = [s for s in EXTRA[env] if any(s in v for k, v in runs.items() if k[0] == env)]
        if not sets_here:
            continue
        L += ['## {} — additional test sets (SE(3) success; Δ = Lie − chart, paired)'.format(env), '',
              '| head | speed | gradient | ' + ' | '.join('`{}`'.format(s.replace(STD[env] + '_', '')) for s in sets_here) + ' |',
              '|---|---|---|' + '---|' * len(sets_here)]
        for h, sp in configs:
            for g in ('chart', 'lie'):
                cells = []
                for sset in sets_here:
                    M = [set_metrics(s[sset])['succ_se3'] for (e, hh, ss, gg, _), s in runs.items()
                         if (e, hh, ss, gg) == (env, h, sp, g) and sset in s]
                    cells.append('{} ({})'.format(ms(M), len(M)) if M else '-')
                if any(c != '-' for c in cells):
                    L.append('| {} | {} | {} | {} |'.format(HEAD_LABEL[h], SPEED_LABEL[sp], g, ' | '.join(cells)))
            cells = []
            for sset in sets_here:
                P = paired(runs, env, h, sp, sset)
                if P:
                    summary['paired']['{}|{}|{}|{}'.format(env, h, sp, sset)] = {'se3': P}
                cells.append('{:+.1f} (p_case {})'.format(P['mean'], fmt_p(P['p_case'])) if P else '-')
            if any(c != '-' for c in cells):
                L.append('| {} | {} | **Δ** | {} |'.format(HEAD_LABEL[h], SPEED_LABEL[sp], ' | '.join(cells)))
        L.append('')
    # ---------------------------------------------------------------- by angle
    L += ['## SE(3) success by start-goal relative rotation (standard sets)', '',
          '| env | head | speed | gradient | 0-90° | 90-150° | 150-180° |', '|---|---|---|---|---|---|---|']
    for env in ('T', 'L', 'A'):
        for h, sp in configs:
            for g in ('chart', 'lie'):
                M = [set_metrics(s[STD[env]]) for (e, hh, ss, gg, _), s in runs.items()
                     if (e, hh, ss, gg) == (env, h, sp, g) and STD[env] in s]
                if M:
                    L.append('| {} | {} | {} | {} | {} |'.format(env, HEAD_LABEL[h], SPEED_LABEL[sp], g, ' | '.join(
                        ms([m['succ_se3_{}_{}'.format(lo, int(hi))] for m in M]) for lo, hi in ANGLE_BINS)))
    L.append('')
    # ---------------------------------------------------------------- overall
    L += ['## Mean over the three scenes (standard sets, mean of per-scene seed means)', '',
          '| head | speed | gradient | SE(3) success | chart success | collision | scenes |', '|---|---|---|---|---|---|---|']
    for h, sp in configs:
        for g in ('chart', 'lie'):
            per = []
            for env in ('T', 'L', 'A'):
                M = [set_metrics(s[STD[env]]) for (e, hh, ss, gg, _), s in runs.items()
                     if (e, hh, ss, gg) == (env, h, sp, g) and STD[env] in s]
                if M:
                    per.append((np.mean([m['succ_se3'] for m in M]), np.mean([m['succ_chart'] for m in M]),
                                np.mean([m['coll_se3'] for m in M])))
            if per:
                L.append('| {} | {} | {} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
                    HEAD_LABEL[h], SPEED_LABEL[sp], g, *np.mean(per, 0), len(per)))
    L.append('')
    # ---------------------------------------------------------------- field diagnostics
    if diag:
        L += ['## Field diagnostics (which Eikonal equation does the trained field satisfy?)', '',
              'Median N of the run\'s own speed formulation evaluated with the Lie gradient (physical, chart-free) and '
              'with the chart gradient, binned by the endpoint rotation angle |rotvec|; N = 1 is the Eikonal equation. '
              'Mean over seeds. `res` = median |N − 1| over all endpoints.', '']
        bins = None
        rows = []
        for env in ('T', 'L', 'A'):
            for h, sp in configs:
                for g in ('chart', 'lie'):
                    D = [d for (e, hh, ss, gg, _), d in diag.items() if (e, hh, ss, gg) == (env, h, sp, g)]
                    if not D:
                        continue
                    own = D[0]['speed_model']
                    bins = [(b['lo'], b['hi']) for b in D[0]['by_angle']]
                    nl = np.mean([[b['N_lie'] for b in d['by_angle']] for d in D], 0)
                    nc = np.mean([[b['N_chart'] for b in d['by_angle']] for d in D], 0)
                    rl = np.mean([d['res_{}_lie'.format(own)]['med_abs'] for d in D])
                    rc = np.mean([d['res_{}_chart'.format(own)]['med_abs'] for d in D])
                    rs = np.mean([d['rot_share'] for d in D])
                    Ds = [d for d in D if 'by_speed' in d]
                    nsp = np.mean([[b['N_lie'] if b['N_lie'] is not None else np.nan for b in d['by_speed']]
                                   for d in Ds], 0) if Ds else None
                    summary['diag']['{}|{}|{}|{}'.format(env, h, sp, g)] = {
                        'n': len(D), 'N_lie': nl.tolist(), 'N_chart': nc.tolist(), 'res_lie': rl, 'res_chart': rc,
                        'rot_share': rs, 'bins': bins, 'N_lie_by_speed': nsp.tolist() if nsp is not None else None,
                        'n_by_speed': len(Ds)}
                    rows.append('| {} | {} | {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                        env, HEAD_LABEL[h], SPEED_LABEL[sp], g, len(D), ' '.join('{:.2f}'.format(x) for x in nl),
                        ' '.join('{:.2f}'.format(x) for x in nc),
                        ' '.join('{:.2f}'.format(x) for x in nsp) + ' ({})'.format(len(Ds)) if nsp is not None else '-',
                        rl, rc, rs))
        hdr = ' '.join('{}-{}'.format(int(lo), int(hi)) for lo, hi in bins)
        L += ['| env | head | speed | trained with | n | N Lie by |rotvec| ({}°) | N chart by |rotvec| | N Lie by speed (<0.01, 0.01-0.1, 0.1-0.5, 0.5-1) | res Lie | res chart | rot share |'.format(hdr),
              '|---|---|---|---|---|---|---|---|---|---|---|'] + rows + ['']
    # ---------------------------------------------------------------- Lie MPPI sampler
    runs_l, _ = load(os.path.join(ROOT, 'plan_liemppi'))
    if runs_l:
        L += ['# Planner: Lie MPPI (unit body twists, Lie-Shape-NTFields controllers/lie_mppi.py)', '',
              'Same checkpoints, cases, step, samples, horizon, temperature and budget; convergence on SE(3). '
              'The chart sampler measures rotation in the chart metric (matched to chart-gradient fields), the Lie '
              'sampler in the left-invariant metric (matched to Lie-gradient fields).', '']
        summary['paired_liemppi'] = {}
        for env in ('T', 'L', 'A'):
            for sset in [STD[env]] + [x for x in EXTRA[env] if any(x in v for v in runs_l.values())]:
                if not any(sset in v for k, v in runs_l.items() if k[0] == env):
                    continue
                L += ['## {} — `{}`, Lie MPPI'.format(env, sset), '',
                      '| head | speed | gradient | n | SE(3) success | collision | non-conv | iters | trans ratio | rot ratio |',
                      '|---|---|---|---|---|---|---|---|---|---|']
                for h, sp in configs:
                    for g in ('chart', 'lie'):
                        M = [set_metrics(v[sset]) for (e, hh, ss, gg, _), v in sorted(runs_l.items())
                             if (e, hh, ss, gg) == (env, h, sp, g) and sset in v]
                        if M:
                            L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                                HEAD_LABEL[h], SPEED_LABEL[sp], g, len(M), ms([m['succ_se3'] for m in M]),
                                ms([m['coll_se3'] for m in M]), ms([m['nc_se3'] for m in M]),
                                ms([m['iters_se3'] for m in M], 0), ms([m['trans_ratio'] for m in M], 2),
                                ms([m['rot_ratio'] for m in M], 2)))
                L += ['', '| head | speed | n pairs | Δ SE(3) success (Lie − chart field) | p_t | cases Lie-only / chart-only | p_case |',
                      '|---|---|---|---|---|---|---|']
                for h, sp in configs:
                    P = paired(runs_l, env, h, sp, sset)
                    if P:
                        summary['paired_liemppi']['{}|{}|{}|{}'.format(env, h, sp, sset)] = {'se3': P}
                        L.append('| {} | {} | {} | {:+.1f} ± {} | {} | {} / {} | {} |'.format(
                            HEAD_LABEL[h], SPEED_LABEL[sp], P['n'], P['mean'],
                            '{:.1f}'.format(P['sd']) if P['sd'] is not None else '-', fmt_p(P['p_t']),
                            P['lie_only'], P['chart_only'], fmt_p(P['p_case'])))
                L.append('')
        # 2 x 2: field gradient x planner, SE(3) success, seeds evaluated by both planners
        L += ['## Field gradient × planner (SE(3) success, mean over seeds evaluated with both planners)', '',
              '| scene / set | head | speed | chart field, chart MPPI | chart field, Lie MPPI | Lie field, chart MPPI | Lie field, Lie MPPI |',
              '|---|---|---|---|---|---|---|']
        summary['two_by_two'] = {}
        for env in ('T', 'L', 'A'):
            for sset in [STD[env]] + EXTRA[env]:
                for h, sp in configs:
                    cells, rec = [], {}
                    for g in ('chart', 'lie'):
                        for planner, R_ in (('chart', runs), ('lie', runs_l)):
                            seeds = [k[4] for k, v in runs_l.items() if k[:4] == (env, h, sp, g) and sset in v
                                     and sset in runs.get(k, {})]
                            M = [set_metrics(R_[(env, h, sp, g, sd)][sset])['succ_se3'] for sd in seeds]
                            rec['{}_{}'.format(g, planner)] = M
                            cells.append('{} ({})'.format(ms(M), len(M)) if M else '-')
                    if any(c != '-' for c in cells):
                        summary['two_by_two']['{}|{}|{}|{}'.format(env, sset, h, sp)] = rec
                        L.append('| {} `{}` | {} | {} | {} |'.format(env, sset.replace(STD[env] + '_', '') if sset != STD[env] else 'standard',
                                                                       HEAD_LABEL[h], SPEED_LABEL[sp], ' | '.join(cells)))
        L.append('')
    # ---------------------------------------------------------------- obstacle-free ground truth
    free = {}
    for f in glob.glob(os.path.join(ROOT, 'free', '*.json')):
        nm = os.path.basename(f)[:-5]
        m = re.match(r'^(F2?)_(gmetric|trieq)_iso_(chart|lie)_s(\d+)$', nm)
        if m:
            free[(m[1], m[2], m[3], int(m[4]))] = json.load(open(f))
        elif nm == 'oracle':
            free[('oracle', 'oracle', 'exact', 0)] = json.load(open(f))
    if free:
        L += ['## Obstacle-free SE(3), exact ground truth (`datasets/3dshape/data/free_se3`, all speeds 1)', '',
              'T* = sqrt(|dt|^2 + (theta / 2 pi)^2) solves the Lie Eikonal exactly. Relative error |tau − T*| / T* on '
              '20k held-out pairs (T* > 0.05), by the larger endpoint |rotvec|; bias = median (tau − T*) / T*; '
              'planning = forward MPPI without obstacles on 1000 pairs, path cost = SE(3) path length / T*. '
              'Mean ± sd over seeds. F2 = unweighted Eikonal residual (the control); F = the NTFields exp(-tau/2) '
              'weight, which collapses to the degenerate inflated field for both gradients (EXPERIMENTS.md L5).', '']
        rv_bins = None
        rows = []
        summary['free'] = {}
        for var, h, g in [('oracle', 'oracle', 'exact')] + [(v, h, g) for v in ('F2', 'F') for h in ('gmetric', 'trieq')
                                                             for g in ('chart', 'lie')]:
            F = [v for (vv, hh, gg, _), v in free.items() if (vv, hh, gg) == (var, h, g)]
            if not F:
                continue
            rv_bins = [(b['lo'], b['hi']) for b in F[0]['by_rotvec']]
            rec = {'n': len(F), 'rel_p50': [x['rel_p50'] * 100 for x in F], 'rel_p95': [x['rel_p95'] * 100 for x in F],
                   'bias': [x['bias_p50'] * 100 for x in F],
                   'by_rotvec': np.mean([[b['med'] * 100 for b in x['by_rotvec']] for x in F], 0).tolist(),
                   'bias_by_rotvec': np.mean([[b['med'] * 100 for b in x['bias_by_rotvec']] for x in F], 0).tolist(),
                   'succ': [x['plan']['succ_se3'] for x in F], 'cost': [x['plan']['cost_p50'] for x in F if x['plan']['cost_p50']]}
            summary['free']['{}|{}|{}'.format(var, h, g)] = rec
            rows.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                var, HEAD_LABEL.get(h, 'exact T*'), g, len(F), ms(rec['rel_p50'], 2), ms(rec['rel_p95'], 2),
                ms(rec['bias'], 2), ' / '.join('{:.2f}'.format(x) for x in rec['by_rotvec']),
                ' / '.join('{:+.2f}'.format(x) for x in rec['bias_by_rotvec']), ms(rec['succ']), ms(rec['cost'], 3)))
        hdr = ' / '.join('{}-{}'.format(int(lo), int(hi)) for lo, hi in rv_bins)
        L += ['| variant | head | gradient | n | rel. error p50 (%) | p95 (%) | bias (%) | rel. error p50 by max |rotvec| ({}°) | bias by max |rotvec| | MPPI SE(3) success | path cost p50 |'.format(hdr),
              '|---|---|---|---|---|---|---|---|---|---|---|'] + rows + ['']
    with open(os.path.join(OUT, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    with open(os.path.join(OUT, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, default=float)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
