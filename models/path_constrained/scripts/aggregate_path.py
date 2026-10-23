#!/usr/bin/env python3
"""Tables of the path-constrained comparison (results/path_constrained/plan -> results/path_constrained/summary).

Run names: <cell code>[w]_<model>[4k]_s<seed>: e1a = prism_env1_a (main path of env1), e1b = second path,
e1s = prism30_env1_a (stress: prism length 0.3); w = world chart; model hom (graph_homotopy) / ntf (metric
NTFields); 4k = the epoch-4000 checkpoint. Files: <run>/<test set>__<planner>_<mode>.json (plan_eval_path.py).
Every number is the SE(3) success (converged within 0.01, no collision at any waypoint or sub-step) in %, mean +-
population std over seeds; paired tests are exact McNemar on the seed-matched cases.

    python3 models/path_constrained/scripts/aggregate_path.py
"""
import glob
import json
import math
import os
import re

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(HOST, 'results/path_constrained/plan')
OUT = os.path.join(HOST, 'results/path_constrained/summary')
SETS = os.path.join(HOST, 'testing_data/3dshape')
CELLS = {'e1a': 'prism_env1_a', 'e3a': 'prism_env3_a', 'e4a': 'prism_env4_a',
         'e1b': 'prism_env1_b', 'e3b': 'prism_env3_b', 'e4b': 'prism_env4_b',
         'e1s': 'prism30_env1_a', 'e3s': 'prism30_env3_a', 'e4s': 'prism30_env4_a'}
MAIN, SECOND, STRESS = ('e1a', 'e3a', 'e4a'), ('e1b', 'e3b', 'e4b'), ('e1s', 'e3s', 'e4s')
VARIANTS = ('full', 'seg', 'tight', 'bigrot', 'hard')
VDESC = {'full': 'whole path', 'seg': 'sub-paths', 'tight': 'whole path, tight ends', 'bigrot': 'whole path, >= 150 deg',
         'hard': 'whole path, tight ends, >= 135 deg'}
MODELS = (('hom', 'dual'), ('hom', 'base'), ('ntf', 'base'))
MNAME = {('hom', 'dual'): 'homotopy dual', ('hom', 'base'): 'homotopy base', ('ntf', 'base'): 'NTFields'}


def load():
    rows = []
    for f in glob.glob(os.path.join(RES, '*', '*.json')):
        run = os.path.basename(os.path.dirname(f))
        m = re.match(r'(e\d[abs])(w?)_(hom|ntf)(4k)?_s(\d+)$', run)
        b = re.match(r'(.+)_(full|seg|tight|bigrot|hard)__(base|dual)_(free|forward)\.json$', os.path.basename(f))
        if not m or not b:
            continue
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        rows.append(dict(run=run, code=m.group(1), world=bool(m.group(2)), model=m.group(3), k4=bool(m.group(4)),
                         seed=int(m.group(5)), set=b.group(1) + '_' + b.group(2), variant=b.group(2),
                         planner=b.group(3), mode=b.group(4), summary=d['summary'],
                         ok={r['case']: bool(r['success_se3']) for r in d['records']}, records=d['records']))
    return rows


def mean(v):
    return sum(v) / len(v) if v else float('nan')


def ms(v, n=True):
    if not v:
        return '-'
    m = mean(v)
    if len(v) == 1:
        return '{:.1f}'.format(100 * m)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return '{:.1f} +- {:.1f}'.format(100 * m, 100 * s)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (a only, b only, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def verification(cell, variant):
    try:
        return json.load(open(os.path.join(SETS, '{}_{}'.format(cell, variant), 'verification.json')))['summary']
    except (OSError, ValueError):
        return {}


def succ(rows, model, planner, mode, **kw):
    return [r['summary']['success_rate_se3'] for r in pick(rows, model=model, planner=planner, mode=mode, **kw)]


def paired(sel, planner_h, mode):
    """Seed-matched McNemar homotopy (planner_h) vs NTFields over all seeds of sel."""
    a, b = [], []
    for rn in pick(sel, model='ntf', planner='base', mode=mode):
        rh = pick(sel, model='hom', planner=planner_h, mode=mode, seed=rn['seed'])
        if not rh:
            continue
        cases = sorted(set(rh[0]['ok']) & set(rn['ok']))
        a += [rh[0]['ok'][c] for c in cases]
        b += [rn['ok'][c] for c in cases]
    return mcnemar(a, b) + (len(a),) if a else None


def fmt_p(t):
    if t is None:
        return '-'
    w, l_, p, n = t
    return '+{} / -{} (p = {:.0e})'.format(w, l_, p) if p > 0 else '+{} / -{} (p < 1e-300)'.format(w, l_)


def main():
    rows = [r for r in load() if not r['k4'] and not r['world']]
    allrows = load()
    os.makedirs(OUT, exist_ok=True)
    J = {}
    L = ['# Path-constrained prism (camera follow): homotopy head vs metric NTFields', '',
         'SE(3) success % = converged within 0.01 in sqrt(dl^2 + (theta / 2 pi)^2) with the exact prism clearance >= 0 '
         'at every waypoint and at 4 sub-steps between waypoints (line of sight never blocked, camera never touching '
         'the scene). Mean +- std over seeds (3 on the main cells, 1 elsewhere). homotopy = graph_homotopy, '
         'NTFields = the original metric NTFields, both 8000 epochs on M with the projected Eikonal residual '
         '(arclength chart). dual = both lifted goals planned on the double cover (homotopy only), base = MPPI on T. '
         'forward = the object only moves forward (camera follow); free = l anywhere on the path. straight = the direct '
         'motion (l linear, R slerp) collision-free, short way / either way.', '']

    # ---------------------------------------------------------------- main table
    L += ['## 1. Main cells (prism 0.02 x 0.2, path a of each scene, 3 seeds), forward mode', '',
          '| scene | set | straight (short / either) | homotopy dual | homotopy base | NTFields |', '|---|---|---|---|---|---|']
    for code in MAIN:
        cell = CELLS[code]
        for v in VARIANTS:
            sel = pick(rows, code=code, variant=v)
            if not sel:
                continue
            ver = verification(cell, v)
            cols = [ms(succ(sel, mo, pl, 'forward')) for mo, pl in MODELS]
            L.append('| {} | {} | {:.1f} / {:.1f} | {} | {} | {} |'.format(
                cell.replace('prism_', ''), v, 100 * ver.get('direct_short_ok', float('nan')),
                100 * ver.get('direct_either_ok', float('nan')), *cols))
            for mo, pl in MODELS:
                for mode in ('forward', 'free'):
                    xs = pick(sel, model=mo, planner=pl, mode=mode)
                    if xs:
                        J['{}|{}|{}_{}_{}'.format(cell, v, mo, pl, mode)] = dict(
                            {k: mean([r['summary'][k] for r in xs]) for k in (
                                'success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3', 'success_rate_se3_monotone',
                                'monotone_rate', 'mean_frac_blocked', 'median_min_clearance_success')},
                            n=len(xs), seeds=[r['summary']['success_rate_se3'] for r in xs])
    # pooled over the three scenes
    L += ['', 'Pooled over the three scenes (mean of the per-scene means):', '',
          '| set | homotopy dual | homotopy base | NTFields |', '|---|---|---|---|']
    for v in VARIANTS:
        cols = []
        for mo, pl in MODELS:
            per = [mean(succ(pick(rows, code=c, variant=v), mo, pl, 'forward')) for c in MAIN]
            per = [x for x in per if x == x]
            cols.append('{:.1f}'.format(100 * mean(per)) if per else '-')
        L.append('| {} | {} | {} | {} |'.format(v, *cols))
    L.append('')

    # ---------------------------------------------------------------- failure modes
    L += ['## 2. Failure modes (main cells, forward mode): collision % / not converged % / mean fraction of the plan with '
          'the line of sight blocked %', '', '| scene | set | homotopy dual | homotopy base | NTFields |', '|---|---|---|---|---|']
    for code in MAIN:
        for v in VARIANTS:
            sel = pick(rows, code=code, variant=v, mode='forward')
            if not sel:
                continue
            cols = []
            for mo, pl in MODELS:
                xs = pick(sel, model=mo, planner=pl)
                cols.append('{:.1f} / {:.1f} / {:.2f}'.format(*(100 * mean([r['summary'][k] for r in xs]) for k in (
                    'collision_rate_se3', 'nonconv_rate_se3', 'mean_frac_blocked'))) if xs else '-')
            L.append('| {} | {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), v, *cols))
    L.append('')

    # ---------------------------------------------------------------- free mode
    L += ['## 3. Free mode (l anywhere on the path): success, success with a monotone plan, monotone plans %', '',
          'A plan is monotone if it never moves the object back along the path by more than 0.005.', '',
          '| scene | set | homotopy dual | homotopy base | NTFields |', '|---|---|---|---|---|']
    for code in MAIN:
        for v in ('full', 'seg'):
            sel = pick(rows, code=code, variant=v, mode='free')
            if not sel:
                continue
            cols = []
            for mo, pl in MODELS:
                xs = pick(sel, model=mo, planner=pl)
                cols.append('{:.1f} / {:.1f} / {:.1f}'.format(*(100 * mean([r['summary'][k] for r in xs]) for k in (
                    'success_rate_se3', 'success_rate_se3_monotone', 'monotone_rate'))) if xs else '-')
            L.append('| {} | {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), v, *cols))
    L.append('')

    # ---------------------------------------------------------------- paired tests
    L += ['## 4. Paired tests, homotopy vs NTFields (seed-matched cases, exact McNemar): cases only homotopy solves / only '
          'NTFields solves', '', '| scene | set | mode | homotopy dual vs NTFields | homotopy base vs NTFields |',
          '|---|---|---|---|---|']
    for code in MAIN + SECOND + STRESS:
        for v in VARIANTS:
            for mode in ('forward', 'free'):
                sel = pick(rows, code=code, variant=v, mode=mode)
                if not pick(sel, model='ntf'):
                    continue
                L.append('| {} | {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), v, mode,
                                                           fmt_p(paired(sel, 'dual', mode)), fmt_p(paired(sel, 'base', mode))))
    L.append('')

    # ---------------------------------------------------------------- second paths and stress
    for title, codes, note in (('5. Second path per scene (path b, prism 0.2, seed 0)', SECOND, ''),
                               ('6. Stress: camera farther away (prism length 0.3, path a, seed 0)', STRESS, '')):
        L += ['## ' + title, '', '| scene | set | straight (short / either) | mode | homotopy dual | homotopy base | NTFields |',
              '|---|---|---|---|---|---|---|']
        for code in codes:
            for v in VARIANTS:
                for mode in ('forward', 'free'):
                    sel = pick(rows, code=code, variant=v, mode=mode)
                    if not sel:
                        continue
                    ver = verification(CELLS[code], v)
                    cols = [ms(succ(sel, mo, pl, mode)) for mo, pl in MODELS]
                    L.append('| {} | {} | {:.1f} / {:.1f} | {} | {} | {} | {} |'.format(
                        CELLS[code].replace('prism_', ''), v, 100 * ver.get('direct_short_ok', float('nan')),
                        100 * ver.get('direct_either_ok', float('nan')), mode, *cols))
                    for mo, pl in MODELS:
                        xs = pick(sel, model=mo, planner=pl)
                        if xs:
                            J['{}|{}|{}_{}_{}'.format(CELLS[code], v, mo, pl, mode)] = dict(
                                {k: mean([r['summary'][k] for r in xs]) for k in (
                                    'success_rate_se3', 'collision_rate_se3', 'nonconv_rate_se3', 'mean_frac_blocked')},
                                n=len(xs))
        L.append('')

    # ---------------------------------------------------------------- chart ablation
    L += ['## 7. Chart ablation (main cells, seed 0): path unrolled onto the x axis (arclength) / real anchor position '
          '(world)', '', '| scene | set | mode | homotopy dual: arc / world | homotopy base: arc / world | NTFields: arc / world |',
          '|---|---|---|---|---|---|']
    for code in MAIN:
        for v in ('full', 'seg'):
            for mode in ('forward', 'free'):
                cols = []
                for mo, pl in MODELS:
                    a = pick(allrows, code=code, variant=v, model=mo, planner=pl, mode=mode, world=False, seed=0, k4=False)
                    w = pick(allrows, code=code, variant=v, model=mo, planner=pl, mode=mode, world=True, seed=0, k4=False)
                    f = lambda x: '{:.1f}'.format(100 * x[0]['summary']['success_rate_se3']) if x else '-'  # noqa: E731
                    cols.append('{} / {}'.format(f(a), f(w)))
                L.append('| {} | {} | {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), v, mode, *cols))
    L.append('')

    # ---------------------------------------------------------------- training length
    L += ['## 8. Training length (main cells, forward mode, 3 seeds): epoch 4000 / epoch 8000', '',
          '| scene | set | homotopy dual | NTFields |', '|---|---|---|---|']
    for code in MAIN:
        for v in ('full', 'seg'):
            cols = []
            for mo, pl in (('hom', 'dual'), ('ntf', 'base')):
                a = succ(pick(allrows, code=code, variant=v, world=False, k4=True), mo, pl, 'forward')
                b = succ(pick(allrows, code=code, variant=v, world=False, k4=False), mo, pl, 'forward')
                cols.append('{} / {}'.format(ms(a), ms(b)))
            L.append('| {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), v, *cols))
    L.append('')

    # ---------------------------------------------------------------- homotopy sheets
    L += ['## 9. Homotopy dual planner: plan per homotopy class (main cells, forward mode)', '',
          '| scene | set | + sheet | - sheet | either (oracle) | dual (converged plan kept) |', '|---|---|---|---|---|---|']
    for code in MAIN + STRESS:
        for v in VARIANTS:
            xs = pick(rows, code=code, variant=v, mode='forward', model='hom', planner='dual')
            if xs:
                L.append('| {} | {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} |'.format(CELLS[code].replace('prism_', ''), v, *(
                    100 * mean([r['summary'][k] for r in xs]) for k in (
                        'success_se3_plus', 'success_se3_minus', 'success_se3_either', 'success_rate_se3'))))
    L.append('')

    # ---------------------------------------------------------------- plan quality
    L += ['## 10. Plans that succeed (main cells, full set, forward mode, 3 seeds): median smallest clearance, median '
          'rotation (turns), median iterations', '', '| scene | homotopy dual | homotopy base | NTFields |', '|---|---|---|---|']
    for code in MAIN:
        sel = pick(rows, code=code, variant='full', mode='forward')
        cols = []
        for mo, pl in MODELS:
            xs = pick(sel, model=mo, planner=pl)
            cols.append('{:.3f} / {:.2f} / {:.0f}'.format(*(mean([r['summary'][k] for r in xs]) for k in (
                'median_min_clearance_success', 'median_rot_len_success', 'median_iters_se3'))) if xs else '-')
        L.append('| {} | {} | {} | {} |'.format(CELLS[code].replace('prism_', ''), *cols))
    L.append('')

    open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(J, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
