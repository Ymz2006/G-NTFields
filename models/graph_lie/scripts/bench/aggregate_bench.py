#!/usr/bin/env python3
"""Tables for the chart-vs-Lie benchmark (results/graph_lie/bench): planning outcomes per targeted subset,
the path audit (predicted vs true speed, tau vs physical cost) and the free-pair exact travel times.

    python3 models/graph_lie/scripts/bench/aggregate_bench.py [--root results/graph_lie/bench]

Writes <root>/summary/bench_summary.md and bench_summary.json. numpy only (runs on the host).
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

RUN = re.compile(r'^([TLA])_gmetric_(comb|iso|aniso)_(chart|lie)_s(\d+)$')
RUN_GN = re.compile(r'^([TLA])_gmetric_(chart|lie)_s(\d+)$')   # models/graph_native names (combined speed)


def match_run(name):
    m = RUN.match(name)
    if m:
        return m[1], m[2], m[3], int(m[4])
    m = RUN_GN.match(name)
    if m:
        return m[1], 'comb', m[2], int(m[3])
    return None
SPEED_LABEL = {'comb': 'combined S|g|=1', 'iso': 'iso S_d|g|=1', 'aniso': 'aniso (S_d,S_a)'}
SUBSET_ORDER = ['nearid', 'bothfar', 'startfar', 'goalfar', 'perp', 'par', 'longtrans_near', 'longtrans_far',
                'tightstart', 'standard300', 'bigangle', 'rotheavy', 'transonly', 'standard']
SUBSET_DESC = {
    'nearid': 'both |rotvec| < 45°, rel. rot. >= 45° (control: chart = Lie)',
    'bothfar': 'both |rotvec| in [135, 180)°',
    'startfar': 'start |rotvec| >= 150°, goal < 45°',
    'goalfar': 'start |rotvec| < 45°, goal >= 150°',
    'perp': 'start 120-165°, goal = rotation 60-120° about a body axis ⊥ rotvec (max chart distortion)',
    'par': 'same cases as perp, rotation about the rotvec axis toward the identity (no distortion)',
    'longtrans_near': '|dp| >= 0.5, both |rotvec| < 45°',
    'longtrans_far': '|dp| >= 0.5, both |rotvec| in [135, 180)°, rel. rot. < 90°',
    'tightstart': 'start clearance <= 0.015 (S_d <= 0.26), goal clearance > 0.02',
    'standard300': 'first 300 cases of the standard RRT-verified set',
    'bigangle': 'Lie repo: both |rotvec| in [135, 180)° (200)',
    'rotheavy': 'Lie repo: |dp| <= 0.05, rel. rot. >= 90° (200)',
    'transonly': 'Lie repo: identical rotations, |dp| >= 0.3 (200)',
    'standard': 'standard 1000-case set',
}
THETA_BINS = [0, 45, 90, 135, 160, 180.01, 1000]
THETA_LABELS = ['0-45', '45-90', '90-135', '135-160', '160-180', '>180 (chart MPPI leaves the ball)']


def binom_two_sided(k, n):
    if n == 0:
        return None
    p_le = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    p_ge = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(p_le, p_ge))


def paired_t(d):
    d = np.asarray(d, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return None
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    # two-sided p from Student t with n-1 dof via the regularized incomplete beta (numpy only)
    n = len(d) - 1
    x = n / (n + t * t)
    return float(_betainc(n / 2.0, 0.5, x))


def _betainc(a, b, x, steps=20000):
    """Regularized incomplete beta I_x(a, b) by midpoint integration (adequate for p-values)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    t = (np.arange(steps) + 0.5) / steps * x
    f = t ** (a - 1) * (1 - t) ** (b - 1)
    B = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return float(f.sum() * x / steps / B)


def fmt(m, s=None, d=1):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return '-'
    if s is None:
        return '{:.{d}f}'.format(m, d=d)
    return '{:.{d}f} ± {:.{d}f}'.format(m, s, d=d)


def load_plan(root):
    """-> {(scene, speed, grad, seed, sampler, subset): (summary, records)}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, 'plan', '*', '*', '*.json'))):
        run, sampler, subset = f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-5]
        m = RUN.match(run)
        if not m:
            continue
        try:
            J = json.load(open(f))
        except Exception:
            continue
        out[(m[1], m[2], m[3], int(m[4]), sampler, subset)] = (J['summary'], J['records'])
    return out


def load_audit(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, 'plan', '*', '*', '*.audit.npz'))):
        run, sampler, subset = f.split(os.sep)[-3], f.split(os.sep)[-2], os.path.basename(f)[:-len('.audit.npz')]
        m = RUN.match(run)
        if not m:
            continue
        out[(m[1], m[2], m[3], int(m[4]), sampler, subset)] = np.load(f)
    return out


def rate(recs, key):
    return 100.0 * float(np.mean([r[key] for r in recs])) if recs else float('nan')


def med(recs, key, cond=lambda r: r['success_se3']):
    v = [r[key] for r in recs if cond(r) and r.get(key) is not None]
    return float(np.median(v)) if v else float('nan')


def med_ratio(recs, num, den, cond=lambda r: r['success_se3']):
    v = [r[num] / r[den] for r in recs if cond(r) and r.get(num) is not None and r.get(den) and r[den] > 0]
    return float(np.median(v)) if v else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/graph_lie/bench')
    a = ap.parse_args()
    plan = load_plan(a.root)
    audit = load_audit(a.root)
    L = []
    J = {'plan': {}, 'audit_by_theta': {}, 'free': {}}
    L.append('# Chart vs Lie gradient: targeted benchmark (`results/graph_lie/bench`)\n')
    L.append('Generated by `models/graph_lie/scripts/bench/aggregate_bench.py`. graph_metric head, original NTFields trainer, '
             'T scene unless stated; mean ± sd over seeds (n). SE(3) success = goal pose reached within 0.01 on SE(3) and no '
             'collision. Δ = Lie − chart paired by seed (same init / data order); p_case = exact sign test on per-case '
             'discordant successes pooled over seeds; p_t = paired t-test over seeds; tau_0 Lie / chart = predicted travel time of the '
             'Lie-trained field over that of the chart-trained field for the same start-goal query (paired by case and seed, median '
             'and p10-p90 over cases).\n')
    L.append('Subsets (300 cases each unless stated; endpoints free with clearance > 0.005):\n')
    for s in SUBSET_ORDER:
        L.append('* `{}`: {}'.format(s, SUBSET_DESC[s]))
    L.append('')
    scenes = sorted({k[0] for k in plan}, key='TLA'.index)
    samplers = [s for s in ('chart', 'lie') if any(k[4] == s for k in plan)]
    # ------------------------------------------------------------------ A. planning outcomes
    for scene in scenes:
        for sampler in samplers:
            L.append('# {} scene, {} MPPI sampler\n'.format(scene, 'official chart' if sampler == 'chart' else 'Lie (body-twist)'))
            L.append('## A. Planning outcomes per subset\n')
            L.append('| subset | speed | gradient | n | SE(3) success | collision | non-conv | iters | trans ratio | rot ratio | Δ SE(3) success | p_t | Lie-only / chart-only | p_case | tau_0 Lie / chart (paired median, p10-p90) |')
            L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
            for subset in SUBSET_ORDER:
                for speed in ('comb', 'iso', 'aniso'):
                    rows = {}
                    for grad in ('chart', 'lie'):
                        seeds = sorted(k[3] for k in plan if k[:3] == (scene, speed, grad) and k[4:] == (sampler, subset))
                        if not seeds:
                            continue
                        per = []
                        for sd in seeds:
                            S, R = plan[(scene, speed, grad, sd, sampler, subset)]
                            per.append(dict(seed=sd, succ=rate(R, 'success_se3'), coll=rate(R, 'collision_se3'),
                                            nonconv=100.0 * float(np.mean([not r['converged_se3'] for r in R])),
                                            iters=med(R, 'iters_se3'), tr=med_ratio(R, 'trans_len_se3', 'straight_trans'),
                                            rr=med_ratio(R, 'rot_len_geo_se3', 'straight_rot_geo'),
                                            cases={r['case']: r['success_se3'] for r in R},
                                            taus={r['case']: r['tau_start'] for r in R if r.get('tau_start')}))
                        rows[grad] = per
                    if not rows:
                        continue
                    delta = {}
                    if 'chart' in rows and 'lie' in rows:
                        common = sorted(set(p['seed'] for p in rows['chart']) & set(p['seed'] for p in rows['lie']))
                        ds, lo, co, tr = [], 0, 0, []
                        for sd in common:
                            pc = next(p for p in rows['chart'] if p['seed'] == sd)
                            pl = next(p for p in rows['lie'] if p['seed'] == sd)
                            ds.append(pl['succ'] - pc['succ'])
                            tr += [pl['taus'][c] / pc['taus'][c] for c in pl['taus'] if c in pc['taus'] and pc['taus'][c] > 0]
                            for c, v in pl['cases'].items():
                                if c in pc['cases']:
                                    lo += int(v and not pc['cases'][c])
                                    co += int(pc['cases'][c] and not v)
                        if ds:
                            delta = dict(d=float(np.mean(ds)), sd=float(np.std(ds, ddof=1)) if len(ds) > 1 else None,
                                         p_t=paired_t(ds), lo=lo, co=co, p_case=binom_two_sided(lo, lo + co), n=len(ds),
                                         tau_ratio=float(np.median(tr)) if tr else None,
                                         tau_ratio_q=[float(np.percentile(tr, q)) for q in (10, 90)] if tr else None)
                    for grad in ('chart', 'lie'):
                        if grad not in rows:
                            continue
                        per = rows[grad]
                        arr = {k: np.array([p[k] for p in per], float) for k in ('succ', 'coll', 'nonconv', 'iters', 'tr', 'rr')}
                        sdv = lambda k: (float(np.nanstd(arr[k], ddof=1)) if len(per) > 1 else None)  # noqa: E731
                        dcol = ''
                        if grad == 'lie' and delta:
                            dcol = '{} | {} | {} / {} | {} | {}'.format(
                                fmt(delta['d'], delta['sd']) if delta['sd'] is not None else fmt(delta['d']),
                                fmt(delta['p_t'], d=3) if delta['p_t'] is not None else '-', delta['lo'], delta['co'],
                                '{:.3g}'.format(delta['p_case']) if delta['p_case'] is not None else '-',
                                '{:.3f} ({:.2f}-{:.2f})'.format(delta['tau_ratio'], *delta['tau_ratio_q']) if delta['tau_ratio'] else '-')
                        else:
                            dcol = ' | | | | '
                        L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                            subset if grad == 'chart' else '', SPEED_LABEL[speed] if grad == 'chart' else '', grad, len(per),
                            fmt(float(np.nanmean(arr['succ'])), sdv('succ')), fmt(float(np.nanmean(arr['coll'])), sdv('coll')),
                            fmt(float(np.nanmean(arr['nonconv'])), sdv('nonconv')), fmt(float(np.nanmean(arr['iters'])), None, 0),
                            fmt(float(np.nanmean(arr['tr'])), None, 2), fmt(float(np.nanmean(arr['rr'])), None, 2), dcol))
                        J['plan']['{}|{}|{}|{}|{}'.format(scene, sampler, subset, speed, grad)] = {
                            'n': len(per), 'succ': float(np.nanmean(arr['succ'])), 'coll': float(np.nanmean(arr['coll'])),
                            'nonconv': float(np.nanmean(arr['nonconv'])), 'delta': delta if grad == 'lie' else None,
                            'per_seed': [{k: p[k] for k in ('seed', 'succ', 'coll', 'nonconv', 'iters', 'tr', 'rr')} for p in per]}
            L.append('')
            # ------------------------------------------------------------ A2. perp vs par predicted time
            rows2 = []
            for speed in ('comb', 'iso', 'aniso'):
                for grad in ('chart', 'lie'):
                    per = []
                    for sd in (0, 1, 2):
                        kp, kq = (scene, speed, grad, sd, sampler, 'perp'), (scene, speed, grad, sd, sampler, 'par')
                        if kp not in plan or kq not in plan:
                            continue
                        Rp = {r['case']: r for r in plan[kp][1] if r.get('tau_start')}
                        Rq = {r['case']: r for r in plan[kq][1] if r.get('tau_start')}
                        cs = sorted(set(Rp) & set(Rq))
                        if not cs:
                            continue
                        per.append(dict(
                            tau=float(np.median([Rp[c]['tau_start'] / Rq[c]['tau_start'] for c in cs])),
                            chart=float(np.median([Rp[c]['straight_rot_chart'] / Rq[c]['straight_rot_chart'] for c in cs])),
                            geo=float(np.median([Rp[c]['straight_rot_geo'] / Rq[c]['straight_rot_geo'] for c in cs])),
                            succ=(100 * float(np.mean([Rp[c]['success_se3'] for c in cs])), 100 * float(np.mean([Rq[c]['success_se3'] for c in cs]))),
                            n=len(cs)))
                    if per:
                        rows2.append((speed, grad, per))
            if rows2:
                L.append('### A2. perp vs par: same start, goals at the same geodesic angle (predicted time ratio tau_perp / tau_par)\n')
                L.append('An exact Lie-metric field gives a ratio of 1 (equal geodesic angle; the translations are identical); a field that '
                         'measures chart length gives the chart-length ratio of the two segments. Median over the shared cases, mean ± sd over seeds.\n')
                L.append('| speed | gradient | n seeds | tau_perp / tau_par | chart-segment ratio | geodesic ratio | success perp / par |')
                L.append('|---|---|---|---|---|---|---|')
                for speed, grad, per in rows2:
                    t = np.array([x['tau'] for x in per])
                    L.append('| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.1f} / {:.1f} |'.format(
                        SPEED_LABEL[speed], grad, len(per), fmt(float(t.mean()), float(t.std(ddof=1)) if len(t) > 1 else None, 3),
                        float(np.mean([x['chart'] for x in per])), float(np.mean([x['geo'] for x in per])),
                        float(np.mean([x['succ'][0] for x in per])), float(np.mean([x['succ'][1] for x in per]))))
                    J['plan']['{}|{}|perp_over_par|{}|{}'.format(scene, sampler, speed, grad)] = {
                        'tau_ratio': float(t.mean()), 'tau_ratio_sd': float(t.std(ddof=1)) if len(t) > 1 else None,
                        'chart_ratio': float(np.mean([x['chart'] for x in per])), 'per_seed': [x['tau'] for x in per]}
                L.append('')
            # ------------------------------------------------------------ B. audit per subset
            L.append('## B. Path audit (successful SE(3) cases; medians over cases, mean over seeds)\n')
            L.append('tau_0 / cost: predicted travel time at the start over the physical cost of the executed path '
                     '(Lie: body-twist lengths / true speed; chart: chart lengths / true speed). slope: median over the path of '
                     '−Δtau / Δcost (1 = the field measures time in that metric). N = own-formulation Eikonal value along the path '
                     'with the Lie / chart gradient (1 = Eikonal satisfied; 1/N = predicted / true speed). len ratio = Lie path '
                     'length / chart path length of the same executed path (< 1 when the chart inflates rotations).\n')
            L.append('| subset | speed | gradient | n | tau_0 / cost_Lie | tau_0 / cost_chart | slope Lie | slope chart | N Lie | N chart | len Lie / chart | rot share of steps |')
            L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
            for subset in SUBSET_ORDER:
                for speed in ('comb', 'iso', 'aniso'):
                    for grad in ('chart', 'lie'):
                        keys = sorted(k for k in plan if k[:3] == (scene, speed, grad) and k[4:] == (sampler, subset))
                        vals = defaultdict(list)
                        for k in keys:
                            S, R = plan[k]
                            vals['tl'].append(med_ratio(R, 'tau_start', 'cost_lie'))
                            vals['tc'].append(med_ratio(R, 'tau_start', 'cost_chart'))
                            vals['sl'].append(med(R, 'slope_lie_med'))
                            vals['sc'].append(med(R, 'slope_chart_med'))
                            vals['nl'].append(med(R, 'N_lie_med'))
                            vals['nc'].append(med(R, 'N_chart_med'))
                            vals['lr'].append(med_ratio(R, 'len_lie', 'len_chart'))
                            vals['rs'].append(med(R, 'rotshare_med'))
                        if not keys:
                            continue
                        L.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                            subset if (grad == 'chart' and speed == 'comb') or True else '', SPEED_LABEL[speed], grad, len(keys),
                            *[fmt(float(np.nanmean(vals[x])), None, 3) for x in ('tl', 'tc', 'sl', 'sc', 'nl', 'nc', 'lr')],
                            fmt(float(np.nanmean(vals['rs'])), None, 2)))
            L.append('')
            # ------------------------------------------------------------ C. waypoint-level by |rotvec|
            L.append('## C. Waypoint-level audit by |rotvec| of the waypoint (all subsets pooled, free waypoints, seeds pooled)\n')
            L.append('Median N (own formulation) with the Lie and the chart gradient, and the local slope −Δtau / Δcost in both '
                     'metrics, for translation-dominated steps (rotation share < 0.3) and rotation-dominated steps (share > 0.7). '
                     'A field that measures time in the Lie metric has slope_Lie = 1 at every angle; one that measures chart time has '
                     'slope_chart = 1; the two differ only on rotation-dominated steps at large |rotvec|.\n')
            L.append('| speed | gradient | quantity | ' + ' | '.join(THETA_LABELS) + ' |')
            L.append('|---|---|---|' + '---|' * len(THETA_LABELS))
            for speed in ('comb', 'iso', 'aniso'):
                for grad in ('chart', 'lie'):
                    keys = [k for k in audit if k[:3] == (scene, speed, grad) and k[4] == sampler]
                    if not keys:
                        continue
                    cols = defaultdict(list)
                    for k in keys:
                        A = audit[k]
                        for c in ('theta_deg', 'free', 'N_lie', 'N_chart', 'dtau', 'dc_lie', 'dc_chart', 'rotshare', 'Sd', 'cw', 'gw', 'cv', 'ds_lie', 'ds_chart'):
                            cols[c].append(A[c])
                    D = {c: np.concatenate(v) for c, v in cols.items()}
                    fr = D['free'].astype(bool)
                    # steps longer than the MPPI step are artefacts (goal row appended by the Lie sampler, seam flips)
                    step = np.isfinite(D['dtau']) & (D['dc_lie'] > 0) & fr & (D['ds_lie'] < 0.03) & (D['ds_chart'] < 0.03)
                    out = {}
                    for q, sel, val in (
                            ('N Lie', fr, D['N_lie']), ('N chart', fr, D['N_chart']),
                            ('slope Lie, trans steps', step & (D['rotshare'] < 0.3), -D['dtau'] / D['dc_lie']),
                            ('slope chart, trans steps', step & (D['rotshare'] < 0.3), -D['dtau'] / D['dc_chart']),
                            ('slope Lie, rot steps', step & (D['rotshare'] > 0.7), -D['dtau'] / D['dc_lie']),
                            ('slope chart, rot steps', step & (D['rotshare'] > 0.7), -D['dtau'] / D['dc_chart']),
                            ('|c_w| / |g_w|', fr & (D['gw'] > 1e-6), D['cw'] / np.maximum(D['gw'], 1e-6)),
                            ('rot share of gradient |c_w|²/|c|²', fr, D['cw'] ** 2 / np.maximum(D['cw'] ** 2 + D['cv'] ** 2, 1e-12))):
                        vals = []
                        for lo, hi in zip(THETA_BINS[:-1], THETA_BINS[1:]):
                            s = sel & (D['theta_deg'] >= lo) & (D['theta_deg'] < hi)
                            vals.append(float(np.median(val[s])) if s.sum() >= 30 else float('nan'))
                        out[q] = vals
                        L.append('| {} | {} | {} | {} |'.format(SPEED_LABEL[speed], grad, q, ' | '.join(fmt(v, None, 3) for v in vals)))
                    J['audit_by_theta']['{}|{}|{}|{}'.format(scene, sampler, speed, grad)] = out
            L.append('')
    # ------------------------------------------------------------------ D. free pairs
    pairs_files = glob.glob(os.path.join('datasets/3dshape/bench', '*_freepairs.npz'))
    for pf in pairs_files:
        P = np.load(pf)
        scene_name = os.path.basename(pf).split('_')[0][0]
        L.append('# D. Exact travel time on saturated-clearance pairs ({}, {} pairs)\n'.format(os.path.basename(pf), len(P['T_geo'])))
        L.append('Pairs whose endpoints and interior (SE(3) geodesic and chart segment) have translational clearance >= margin '
                 '(S_d = 1). Exact iso travel time: T_geo (Lie) vs T_chart (chart segment). Per bin of the start |rotvec|: median '
                 'T_chart / T_geo of the pairs, then per field the median tau / T_geo and tau / T_chart, and the endpoint gradient '
                 'norms |c| (Lie) and |g| (chart) (exact Lie solution: |c| = 1; exact chart solution: |g| = 1). The combined speed never '
                 'saturates, so for `combined` only the trend is meaningful.\n')
        bins_lo = sorted(set(P['bin_lo'].tolist()))
        labels = ['{:.0f}-{:.0f}'.format(lo, hi) for lo, hi in zip(bins_lo, bins_lo[1:] + [180])]
        sel = {lo: P['bin_lo'] == lo for lo in bins_lo}
        L.append('| field | quantity | ' + ' | '.join(labels) + ' |')
        L.append('|---|---|' + '---|' * len(labels))
        L.append('| pairs | n | ' + ' | '.join(str(int(sel[lo].sum())) for lo in bins_lo) + ' |')
        L.append('| pairs | median T_chart / T_geo | ' + ' | '.join(fmt(float(np.median((P['T_chart'] / P['T_geo'])[sel[lo]])), None, 3) for lo in bins_lo) + ' |')
        L.append('| pairs | p90 T_chart / T_geo | ' + ' | '.join(fmt(float(np.percentile((P['T_chart'] / P['T_geo'])[sel[lo]], 90)), None, 3) for lo in bins_lo) + ' |')
        L.append('| pairs | median rotation share of T_geo | ' + ' | '.join(fmt(float(np.median(P['rotshare'][sel[lo]])), None, 2) for lo in bins_lo) + ' |')
        groups = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(a.root, 'freepairs', '*.json'))):
            m = match_run(os.path.basename(f)[:-5])
            if m and m[0] == scene_name:
                groups[(m[1], m[2])].append(json.load(open(f)))
        fp = {}
        for speed in ('comb', 'iso', 'aniso'):
            for grad in ('chart', 'lie'):
                G = groups.get((speed, grad), [])
                if not G:
                    continue
                tau = np.array([g['tau_fwd'] for g in G])          # (seeds, pairs)
                cN = np.array([g['c_start'] for g in G])
                gN = np.array([g['g_start'] for g in G])
                tag = '{} / {} ({} seeds)'.format(SPEED_LABEL[speed], grad, len(G))
                r_geo = [float(np.median((tau / P['T_geo'][None])[:, sel[lo]])) for lo in bins_lo]
                r_ch = [float(np.median((tau / P['T_chart'][None])[:, sel[lo]])) for lo in bins_lo]
                cn = [float(np.median(cN[:, sel[lo]])) for lo in bins_lo]
                gn = [float(np.median(gN[:, sel[lo]])) for lo in bins_lo]
                L.append('| {} | tau / T_geo | {} |'.format(tag, ' | '.join(fmt(v, None, 3) for v in r_geo)))
                L.append('| {} | tau / T_chart | {} |'.format(tag, ' | '.join(fmt(v, None, 3) for v in r_ch)))
                L.append('| {} | endpoint norm Lie grad |c| | {} |'.format(tag, ' | '.join(fmt(v, None, 3) for v in cn)))
                L.append('| {} | endpoint norm chart grad |g| | {} |'.format(tag, ' | '.join(fmt(v, None, 3) for v in gn)))
                fp['{}|{}'.format(speed, grad)] = {'tau_over_Tgeo': r_geo, 'tau_over_Tchart': r_ch, 'c': cn, 'g': gn,
                                                   'tau_median': [float(np.median(tau[:, sel[lo]])) for lo in bins_lo]}
        # paired Lie - chart (same seed) on log tau
        L.append('')
        L.append('Paired Lie − chart, same seed, median over pairs of log(tau_Lie / tau_chart) in % (positive = the Lie field predicts a longer time):\n')
        L.append('| speed | ' + ' | '.join(labels) + ' |')
        L.append('|---|' + '---|' * len(labels))
        for speed in ('comb', 'iso', 'aniso'):
            Gc, Gl = groups.get((speed, 'chart'), []), groups.get((speed, 'lie'), [])
            bys = {}
            for g in Gc:
                bys.setdefault(re.search(r'_s(\d+)/', g['checkpoint'])[1], {})['chart'] = np.array(g['tau_fwd'])
            for g in Gl:
                bys.setdefault(re.search(r'_s(\d+)/', g['checkpoint'])[1], {})['lie'] = np.array(g['tau_fwd'])
            pairs = [v for v in bys.values() if 'chart' in v and 'lie' in v]
            if not pairs:
                continue
            d = np.array([np.log(v['lie'] / v['chart']) for v in pairs])
            L.append('| {} | {} |'.format(SPEED_LABEL[speed], ' | '.join(
                '{:+.1f} ± {:.1f}'.format(100 * float(np.median(d[:, sel[lo]])), 100 * float(np.std([np.median(dd[sel[lo]]) for dd in d])))
                for lo in bins_lo)))
            fp['paired|{}'.format(speed)] = [float(np.median(d[:, sel[lo]])) for lo in bins_lo]
        J['free'][os.path.basename(pf)] = {'labels': labels, 'T_chart_over_T_geo': [float(np.median((P['T_chart'] / P['T_geo'])[sel[lo]])) for lo in bins_lo], **fp}
        L.append('')
    # ------------------------------------------------------------------ E. training-pair subsets
    ts = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.root, 'train_subsets', '*.json'))):
        Jt = json.load(open(f))
        sc = {'Tshape3d_env1': 'T', 'Lshape3d_env3': 'L', 'Ashape3d_env4': 'A'}.get(Jt['scene'], Jt['scene'])
        sp = {'combined': 'comb', 'iso': 'iso', 'aniso': 'aniso'}[Jt['speed_model']]
        ts[(sc, sp, Jt['grad_mode'])].append(Jt)
    if ts:
        L.append('# E. Predicted vs true speed on subsets of the TRAINING pairs (200k pairs per scene, both endpoints)\n')
        L.append('Median over endpoints of the own-formulation Eikonal value N with the Lie gradient (physical) and the chart '
                 'gradient; 1/N = predicted / true speed; N < 1 = the field predicts a speed above the clearance speed. '
                 '|c_w|/|g_w| = the chart distortion of the field\'s own rotational gradient (1 at the identity, pi/2 at a half '
                 'turn for gradients orthogonal to the rotation vector); perp = share of the rotational chart gradient '
                 'orthogonal to the rotation vector. Mean over seeds of the per-seed medians.\n')
        subs = ['all', 'theta_0_45', 'theta_45_90', 'theta_90_135', 'theta_135_160', 'theta_160_180', 'pair_both_lt45',
                'pair_both_ge150', 'pair_start_ge150_goal_lt45', 'pair_rotation_dominated', 'pair_rotdom_theta_lt45',
                'pair_rotdom_theta_ge135', 'pair_translation_dominated', 'speed_0_0.01', 'speed_0.01_0.1', 'speed_0.1_0.5',
                'speed_0.5_1.0', 'speed_ge0.5_theta_lt45', 'speed_ge0.5_theta_ge150']
        E = {}
        for sc in sorted({k[0] for k in ts}, key='TLA'.index):
            L.append('## {} scene\n'.format(sc))
            L.append('| subset | n endpoints | ' + ' | '.join('{} / {}: N Lie, N chart, |c_w|/|g_w|'.format(SPEED_LABEL[sp], g)
                                                            for sp in ('comb', 'iso', 'aniso') for g in ('chart', 'lie') if (sc, sp, g) in ts) + ' |')
            L.append('|---|---|' + '---|' * sum(1 for sp in ('comb', 'iso', 'aniso') for g in ('chart', 'lie') if (sc, sp, g) in ts))
            for sub in subs:
                cells, n_ep = [], None
                for sp in ('comb', 'iso', 'aniso'):
                    for g in ('chart', 'lie'):
                        if (sc, sp, g) not in ts:
                            continue
                        G = ts[(sc, sp, g)]
                        vals = {k: [x['subsets'][sub][k]['p50'] for x in G if x['subsets'].get(sub) and x['subsets'][sub][k]] for k in ('N_lie', 'N_chart', 'ratio', 'perp')}
                        n_ep = G[0]['subsets'][sub]['n'] if G[0]['subsets'].get(sub) else 0
                        if not vals['N_lie']:
                            cells.append('-')
                            continue
                        cells.append('{:.3f}, {:.3f}, {:.2f}'.format(np.mean(vals['N_lie']), np.mean(vals['N_chart']), np.mean(vals['ratio'])))
                        E['{}|{}|{}|{}'.format(sc, sub, sp, g)] = {k: float(np.mean(v)) for k, v in vals.items() if v}
                L.append('| {} | {} | {} |'.format(sub, n_ep, ' | '.join(cells)))
            L.append('')
            # paired Lie - chart per subset (same seed) on N_lie
            L.append('Paired Lie − chart (same seed) of the median N_lie, i.e. how much more the Lie-trained field over-predicts the physical speed (negative = Lie field predicts higher speed than the chart field):\n')
            L.append('| subset | ' + ' | '.join(SPEED_LABEL[sp] for sp in ('comb', 'iso', 'aniso') if (sc, sp, 'chart') in ts and (sc, sp, 'lie') in ts) + ' |')
            L.append('|---|' + '---|' * sum(1 for sp in ('comb', 'iso', 'aniso') if (sc, sp, 'chart') in ts and (sc, sp, 'lie') in ts))
            for sub in subs:
                cells = []
                for sp in ('comb', 'iso', 'aniso'):
                    if not ((sc, sp, 'chart') in ts and (sc, sp, 'lie') in ts):
                        continue
                    by = {}
                    for g in ('chart', 'lie'):
                        for x in ts[(sc, sp, g)]:
                            sd = re.search(r'_s(\d+)/', x['checkpoint'])[1]
                            if x['subsets'].get(sub) and x['subsets'][sub]['N_lie']:
                                by.setdefault(sd, {})[g] = x['subsets'][sub]['N_lie']['p50']
                    d = [v['lie'] - v['chart'] for v in by.values() if 'lie' in v and 'chart' in v]
                    cells.append('{:+.3f} ± {:.3f} (n={})'.format(np.mean(d), np.std(d, ddof=1) if len(d) > 1 else 0, len(d)) if d else '-')
                L.append('| {} | {} |'.format(sub, ' | '.join(cells)))
            L.append('')
        J['train_subsets'] = E
    os.makedirs(os.path.join(a.root, 'summary'), exist_ok=True)
    with open(os.path.join(a.root, 'summary', 'bench_summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    with open(os.path.join(a.root, 'summary', 'bench_summary.json'), 'w') as fh:
        json.dump(J, fh, indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
