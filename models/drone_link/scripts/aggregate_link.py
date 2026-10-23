"""Summary tables of the drone-pair link study: results/drone_link/summary/summary.{md,json}.

Reads the evaluations results/drone_link/plan/<run>/<set>.json (plan_eval_link.py), the baselines
results/drone_link/baselines/<set>.json (baseline_link.py) and the test sets' verification.json; runs are
<scene>_ntf_s<seed> (full metric NTFields, 8000 epochs), <scene>_ntf4k_s<seed> (its epoch-4000 checkpoint),
<scene>_ntfb_s0 (the bundle's NTFields without the TD / normal terms), <scene>_ntfsb_s0 (x0 band on the scene
clearance only), <scene>_ntflr_s0 (half of the x1 uniform in the room) and <scene>_ntflr20_s<seed> (a fifth, seeds 0-2).
Training losses come from the logs Experiments/drone_link/logs/<run>.out.

    python3 models/drone_link/scripts/aggregate_link.py
"""
import json
import math
import os
import re

import numpy as np

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(HOST, 'results', 'drone_link')
SETS = os.path.join(HOST, 'testing_data', '3dshape')
SCENES = ('forest', 'windows', 'corridor', 'scaffold')
VARIANTS = ('std', 'cross')
SEEDS = (0, 1, 2)
RUN_KINDS = ('ntf', 'ntfb', 'ntfsb', 'ntflr', 'ntflr20')
SEEDED = ('ntf', 'ntflr20')     # kinds trained with seeds 0-2; the other ablations have seed 0 only
PLATEAU = 4e-3                  # runs start at a total loss of ~2-4e-2 and drop below this once the field forms
EPOCH_RE = re.compile(r'Epoch = (\d+) -- Loss = ([0-9.e+-]+) -- Alpha = [0-9.e+-]+(?: -- Eik = ([0-9.e+-]+))?')


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def pct(x):
    return '-' if x is None else '{:.1f}'.format(100 * x)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    if len(v) == 1:
        return '{:.1f}'.format(100 * v[0])
    return '{:.1f} +- {:.1f}'.format(100 * np.mean(v), 100 * np.std(v))


def train_loss(run, last=100):
    """(epochs logged, mean total loss, mean Eikonal residual) over the last `last` epochs of a run's log, the loss
    of its first epoch and the first epoch whose total loss is below PLATEAU (None if it never is)."""
    try:
        with open(os.path.join(HOST, 'Experiments', 'drone_link', 'logs', run + '.out'), errors='replace') as fh:
            m = EPOCH_RE.findall(fh.read())
    except OSError:
        return None
    if not m:
        return None
    ep = int(m[-1][0])
    tail = [x for x in m if int(x[0]) > ep - last]
    eik = [float(x[2] or x[1]) for x in tail]   # without TD / normal terms the loss is the Eikonal residual
    esc = next((int(x[0]) for x in m if float(x[1]) < PLATEAU), None)
    return dict(epoch=ep, loss=float(np.mean([float(x[1]) for x in tail])), eik=float(np.mean(eik)), first=float(m[0][1]),
                escape=esc)


def mcnemar(a, b):
    """Exact two-sided McNemar p for paired booleans a, b; returns (a only, b only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def main():
    out = {'sets': {}}
    lines = ['# Drone-pair link study: summary', '',
             'Link: box of width 0.02 between drone a and drone b, length L in [0.06, 0.30] (normalized room units; '
             'the room is 0.90 x 0.63 x 0.99). Percent of 1000 certified-solvable cases per set (section 6); NTFields = the full metric '
             'NTFields (TD + normal terms 1e-3) in the 7-d chart (t, rotvec / 2 pi, L), 8000 epochs, mean +- std over '
             'the seeds evaluated so far (0-2). Contact % of time averages every plan, failed ones included (a failed '
             'NTFields plan usually crosses an obstacle, so it cannot be flown). Success = converged (0.01 in sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2)) with the exact link '
             'clearance >= 0 at every waypoint and 4 sub-steps between waypoints (drones clear of the scene, line of '
             'sight never blocked, length within its limits).', '']
    hdr = ('| scene | set | straight | independent drones: valid | independent: contact % of time | NTFields success | '
           'collision | stall | NTFields contact % of time |')
    lines += ['## 1. Main table', '', hdr, '|' + '---|' * 9]
    for sc in SCENES:
        for v in VARIANTS:
            s = 'link_{}_{}'.format(sc, v)
            ver = load(os.path.join(SETS, s, 'verification.json'))
            base = load(os.path.join(RES, 'baselines', s + '.json'))
            ev = {seed: load(os.path.join(RES, 'plan', '{}_ntf_s{}'.format(sc, seed), s + '.json')) for seed in SEEDS}
            ev4 = {seed: load(os.path.join(RES, 'plan', '{}_ntf4k_s{}'.format(sc, seed), s + '.json')) for seed in SEEDS}
            evb = load(os.path.join(RES, 'plan', '{}_ntfb_s0'.format(sc), s + '.json'))
            evs = load(os.path.join(RES, 'plan', '{}_ntfsb_s0'.format(sc), s + '.json'))
            evl = load(os.path.join(RES, 'plan', '{}_ntflr_s0'.format(sc), s + '.json'))
            evl20 = {seed: load(os.path.join(RES, 'plan', '{}_ntflr20_s{}'.format(sc, seed), s + '.json')) for seed in SEEDS}
            rec = {'set': s}
            if ver:
                rec['test'] = ver['summary']
            if base:
                rec['baselines'] = base['summary']
            S = {k: [e['summary'][k] for e in ev.values() if e] for k in
                 ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3', 'mean_contact_frac', 'full_contact_rate',
                  'drone_hit_rate', 'median_trans_ratio_a_success', 'median_trans_ratio_b_success',
                  'median_L_travel_success', 'frac_success_using_short_link', 'median_iters_se3')}
            rec['ntf'] = {k: (float(np.mean(x)) if x else None) for k, x in S.items()}
            rec['ntf']['success_per_seed'] = S['success_rate_se3']
            rec['ntf4k'] = [e['summary']['success_rate_se3'] for e in ev4.values() if e]
            rec['ntfb'] = evb['summary']['success_rate_se3'] if evb else None
            rec['ntfsb'] = {k: evs['summary'][k] for k in ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3')} \
                if evs else None
            rec['ntflr'] = {k: evl['summary'][k] for k in ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3')} \
                if evl else None
            rec['ntflr20'] = {k: evl20[0]['summary'][k] for k in ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3')} \
                if evl20[0] else None
            rec['ntflr20_per_seed'] = {k: [e['summary'][k] for e in evl20.values() if e] for k in
                                       ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3')}
            rec['ntflr20_per_seed']['seeds'] = [seed for seed, e in evl20.items() if e]
            rec['ntf_s0'] = {k: ev[0]['summary'][k] for k in ('success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3')} \
                if ev.get(0) else None
            # paired: NTFields seed s success vs independent drones valid (same cases)
            if base and any(ev.values()):
                ind = {r['case']: r['independent'].get('valid', False) for r in base['records']}
                st = {r['case']: r['straight']['success'] for r in base['records']}
                pair_ind, pair_st = [], []
                for seed, e in ev.items():
                    if not e:
                        continue
                    a = [r['success_se3'] for r in e['records']]
                    pair_ind.append(mcnemar(a, [ind[r['case']] for r in e['records']]))
                    pair_st.append(mcnemar(a, [st[r['case']] for r in e['records']]))
                rec['paired_vs_independent'] = pair_ind
                rec['paired_vs_straight'] = pair_st
                # contact of the NTFields plans on the cases where the independent drones lose contact
                lost = [r['case'] for r in base['records'] if r['independent'].get('solved') and not r['independent']['full_contact']]
                rec['ntf_success_on_lost_cases'] = [float(np.mean([r['success_se3'] for r in e['records'] if r['case'] in set(lost)]))
                                                    for e in ev.values() if e and lost]
                rec['n_lost_cases'] = len(lost)
            out['sets'][s] = rec
            b = rec.get('baselines', {})
            lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                sc, v, pct(b.get('straight', {}).get('success_rate')), pct(b.get('independent', {}).get('valid_rate')),
                pct(b.get('independent', {}).get('mean_contact_frac')), ms(S['success_rate_se3']),
                ms(S['collision_rate_se3']), ms(S['nonconv_rate_se3']), ms(S['mean_contact_frac'])))
    lines += ['', 'straight = the direct 7-d motion (drone a linear, R slerp, L linear) is valid; independent drones = '
              'each drone flies its own shortest path (point drone, clearance 0.015), leaving and arriving together; '
              'valid = the whole motion keeps line of sight, range (L <= 0.30) and separation (L >= 0.06).', '']
    lines += ['## 2. How the independent drones lose contact (percent of time, percent of cases)', '',
              '| scene | set | contact % of time | cases with full contact | line of sight blocked (time / cases) | '
              'too far (time / cases) | too close (time) |', '|' + '---|' * 7]
    for s, rec in out['sets'].items():
        b = rec.get('baselines', {}).get('independent')
        if not b:
            continue
        sc, v = s.split('_')[1:]
        lines.append('| {} | {} | {} | {} | {} / {} | {} / {} | {} |'.format(
            sc, v, pct(b['mean_contact_frac']), pct(b['full_contact_rate']), pct(b['mean_los_blocked_frac']),
            pct(b['rate_los_ever_blocked']), pct(b['mean_too_far_frac']), pct(b['rate_ever_too_far']),
            pct(b['mean_too_close_frac'])))
    lines += ['', '## 3. NTFields variants (SE(3) + length success %)', '',
              '| scene | set | full NTFields 8000 (3 seeds) | full NTFields 4000 (3 seeds) | NTFields without TD / normal, '
              '8000 (seed 0) | full NTFields, seed 0: success / collision / stall | x0 band on the scene clearance only, '
              'seed 0: success / collision / stall | long-range pairs (half), seed 0: success / collision / stall | '
              'long-range pairs (a fifth), seeds 0 / 1 / 2: success (mean +- std) |', '|' + '---|' * 9]
    f3 = lambda d: '-' if not d else '{} / {} / {}'.format(pct(d['success_rate_se3']), pct(d['collision_rate_se3']),  # noqa: E731
                                                          pct(d['nonconv_rate_se3']))
    for s, rec in out['sets'].items():
        sc, v = s.split('_')[1:]
        l20 = rec['ntflr20_per_seed']['success_rate_se3']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            sc, v, ms(rec['ntf']['success_per_seed']), ms(rec['ntf4k']), pct(rec['ntfb']), f3(rec['ntf_s0']),
            f3(rec['ntfsb']), f3(rec['ntflr']), '-' if not l20 else '{} ({})'.format(' / '.join(pct(x) for x in l20), ms(l20))))
    lines += ['', '## 4. Paired per case: NTFields (each seed) vs the baselines', '',
              '| scene | set | NTFields only / independent only (p), per seed | NTFields only / straight only (p) | '
              'NTFields success on the cases the independent drones lose contact |', '|' + '---|' * 5]
    for s, rec in out['sets'].items():
        if 'paired_vs_independent' not in rec:
            continue
        sc, v = s.split('_')[1:]
        f = lambda P: ', '.join('+{} / -{} ({:.0e})'.format(*p) for p in P)  # noqa: E731
        lines.append('| {} | {} | {} | {} | {} of {} cases |'.format(sc, v, f(rec['paired_vs_independent']),
                                                                    f(rec['paired_vs_straight']),
                                                                    ms(rec['ntf_success_on_lost_cases']),
                                                                    rec['n_lost_cases']))
    lines += ['', '## 5. Plans: lengths and use of the length coordinate (NTFields successes, median, mean of seeds)', '',
              '| scene | set | drone a path / straight | drone b path / straight | L travel | plans shorter than both '
              'endpoint links (-0.02) | median iterations | independent a / b length | RRT a / b length |',
              '|' + '---|' * 9]
    for s, rec in out['sets'].items():
        sc, v = s.split('_')[1:]
        n = rec['ntf']
        b = rec.get('baselines', {})
        f2 = lambda x: '-' if x is None else '{:.2f}'.format(x)  # noqa: E731
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} / {} | {} / {} |'.format(
            sc, v, f2(n['median_trans_ratio_a_success']), f2(n['median_trans_ratio_b_success']),
            f2(n['median_L_travel_success']), pct(n['frac_success_using_short_link']), f2(n['median_iters_se3']),
            f2(b.get('independent', {}).get('median_len_a')), f2(b.get('independent', {}).get('median_len_b')),
            f2(b.get('rrt', {}).get('median_len_a')), f2(b.get('rrt', {}).get('median_len_b'))))
    lines += ['', '## 6. Test sets', '', '| set | certificate | direct motion valid | certificate path length (median) | '
              'start-goal distance (median) | relative rotation deg (median) | cases drawn for 1000 (unsolved) |',
              '|' + '---|' * 7]
    for s, rec in out['sets'].items():
        t = rec.get('test')
        if t:
            cert = t.get('certificate', 'rrt')
            if cert == 'prm':
                cert = 'PRM ({} nodes, {} component{})'.format(t['roadmap_nodes'], t['roadmap_components'],
                                                             '' if t['roadmap_components'] == 1 else 's')
            lines.append('| {} | {} | {} | {:.2f} | {:.2f} | {:.0f} | {} ({}) |'.format(
                s, cert, pct(t['direct_ok']), t['rrt_len_median'], t['dist_t_median'], t['angle_deg_median'],
                t['tried'], t['unsolved']))
    out['training'] = {'{}_{}_s{}'.format(sc, k, seed): train_loss('{}_{}_s{}'.format(sc, k, seed))
                       for sc in SCENES for k in RUN_KINDS for seed in (SEEDS if k in SEEDED else (0,))}
    lines += ['', '## 7. Training: Eikonal residual, mean of the last 100 epochs (total loss in brackets)', '',
              '| scene | full NTFields, seeds 0 / 1 / 2 | without TD / normal | scene band | long-range pairs (half) | '
              'long-range pairs (a fifth), seeds 0 / 1 / 2 |', '|' + '---|' * 6]
    fl = lambda r: '-' if not r else '{:.2e} ({:.2e}){}'.format(r['eik'], r['loss'], '' if r['epoch'] >= 8000 else  # noqa: E731
                                                                 ' at epoch {}'.format(r['epoch']))
    for sc in SCENES:
        T = out['training']
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            sc, ' / '.join(fl(T['{}_ntf_s{}'.format(sc, seed)]) for seed in SEEDS), fl(T['{}_ntfb_s0'.format(sc)]),
            fl(T['{}_ntfsb_s0'.format(sc)]), fl(T['{}_ntflr_s0'.format(sc)]),
            ' / '.join(fl(T['{}_ntflr20_s{}'.format(sc, seed)]) for seed in SEEDS)))
    first = [T['{}_{}_s{}'.format(sc, k, seed)]['first'] for sc in SCENES for k in SEEDED for seed in SEEDS
             if T['{}_{}_s{}'.format(sc, k, seed)]]
    lines += ['', 'These runs start at a total loss of {:.1e} to {:.1e} and drop by an order of magnitude once the field '
              'forms. First epoch with a total loss below {:.0e}:'.format(min(first), max(first), PLATEAU), '',
              '| scene | full NTFields, seeds 0 / 1 / 2 | long-range pairs (a fifth), seeds 0 / 1 / 2 |', '|---|---|---|']
    esc = lambda r: '-' if not r else ('never' if r['escape'] is None else str(r['escape']))  # noqa: E731
    for sc in SCENES:
        lines.append('| {} | {} | {} |'.format(sc, *(' / '.join(esc(T['{}_{}_s{}'.format(sc, k, seed)]) for seed in SEEDS)
                                                       for k in SEEDED)))
    os.makedirs(os.path.join(RES, 'summary'), exist_ok=True)
    with open(os.path.join(RES, 'summary', 'summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(RES, 'summary', 'summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
