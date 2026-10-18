#!/usr/bin/env python3
"""Aggregate the expanded comparison (targeted sets, standard sets, wall-with-hole cells): the 8000-epoch homotopy
head (base planner / dual cover planner) against the original metric NTFields (8000 epochs and its epoch-4000
checkpoint). Standard library only.

    python3 models/graph_homotopy/scripts/aggregate_expand.py

Reads results/graph_homotopy/plan_expand/<run>/<set>__<planner>.json (and the standard-set results of the
homotopy runs in results/graph_homotopy/plan/), the sets' verification.json, and writes
results/graph_homotopy/summary/expand_summary.{md,json}: seed means of SE(3) success / collision / non-convergence,
seed-matched paired McNemar tests (dual vs baseline), and success by relative angle / distance / endpoint
clearance bins.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import defaultdict

SCENES = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
SHAPES = {'T': 'Tshape3d', 'L': 'Lshape3d', 'A': 'Ashape3d'}
VARIANTS = ('std', 'long', 'bigrot', 'tight', 'hard')
HOLES = ('90', '75', '66')
ANG_BINS = [(0, 60), (60, 120), (120, 150), (150, 180.01)]
DP_BINS = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 2.0)]
CL_BINS = [(0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 1.0)]


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def pct(x, d=1):
    return '-' if x is None else '{:.{}f}'.format(100 * x, d)


def ms(vals, d=1):
    v = [x for x in vals if x is not None]
    if not v:
        return '-'
    if len(v) == 1:
        return '{:.{}f}'.format(100 * v[0], d)
    return '{:.{}f} +- {:.{}f} ({})'.format(100 * statistics.mean(v), d, 100 * statistics.pstdev(v), d, len(v))


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (1 << n)   # exact integers: no float overflow
    return min(1.0, 2.0 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='results/graph_homotopy/plan_expand')
    ap.add_argument('--res-std', default='results/graph_homotopy/plan')
    ap.add_argument('--sets', default='testing_data/3dshape')
    ap.add_argument('--out', default='results/graph_homotopy/summary')
    a = ap.parse_args()

    # ---- collect results: R[(run, set, planner)] = (summary, records)
    R = {}
    for root in (a.res, a.res_std):
        for f in glob.glob(os.path.join(root, '*', '*__*.json')):
            run = os.path.basename(os.path.dirname(f))
            if run == 'test' or not re.match(r'^[TLA](\d+)?_(hminL|mb|mb4k)_s\d+$', run):
                continue
            setname, planner = os.path.basename(f)[:-5].split('__')
            d = load(f)
            if d and 'summary' in d:
                R[(run, setname, planner)] = (d['summary'], d['records'])

    def model_of(run):
        m = re.match(r'^([TLA])(\d+)?_(hminL|mb|mb4k)_s(\d+)$', run)
        return m.group(1), m.group(2), m.group(3), int(m.group(4))

    # ---- per-set table
    def set_name(sc, variant):
        return SCENES[sc] if variant == 'std' else '{}_{}'.format(SCENES[sc], variant)

    def seeds(sc, hole, model, setname, planner):
        out = []
        for (run, s, p), (summ, _) in R.items():
            msc, mh, mm, seed = model_of(run)
            if msc == sc and mh == hole and mm == model and s == setname and p == planner:
                out.append((seed, summ))
        return sorted(out)

    columns = [('ours base', 'hminL', 'base', 'success_rate_se3'), ('ours dual', 'hminL', 'dual', 'success_rate_se3'),
               ('ours dual-shorter', 'hminL', 'dual', 'success_se3_shorter'), ('ours either', 'hminL', 'dual', 'success_se3_either'),
               ('NTFields 8k', 'mb', 'base', 'success_rate_se3'), ('NTFields 4k', 'mb4k', 'base', 'success_rate_se3')]
    lines = ['# Expanded comparison: 8000-epoch homotopy head vs the original metric NTFields', '',
             'SE(3) success % (mean +- std over seeds, n seeds in parentheses). Sets: std = the standard 1000-case set; long '
             '(|dp| >= 0.6), bigrot (relative rotation >= 150 deg), tight (endpoint clearance 0.005-0.02), hard '
             '(clearance 0.005-0.02, |dp| >= 0.5, rotation >= 135 deg); all 1000 cases, RRT-verified. Planners: base = the '
             'official MPPI on the base field; dual = both lifted goals planned, converged plan kept; dual-shorter = converged '
             'then shorter-path rule; either = oracle over the two sheets. NTFields 8k / 4k = the metric baseline at 8000 '
             'epochs and its epoch-4000 checkpoint.', '',
             '## Targeted and standard sets', '',
             '| scene | set | ' + ' | '.join(c[0] for c in columns) + ' | ours dual coll % | NTFields 8k coll % | ours dual nonconv % | NTFields 8k nonconv % |',
             '|' + '---|' * (2 + len(columns) + 4)]
    table = []
    for sc in SCENES:
        for variant in VARIANTS:
            sn = set_name(sc, variant)
            row = dict(scene=sc, set=variant)
            cells = []
            for label, model, planner, key in columns:
                vals = [summ.get(key) for _, summ in seeds(sc, None, model, sn, planner)]
                row[label] = vals
                cells.append(ms(vals))
            coll_o = [summ.get('collision_rate_se3') for _, summ in seeds(sc, None, 'hminL', sn, 'dual')]
            coll_b = [summ.get('collision_rate_se3') for _, summ in seeds(sc, None, 'mb', sn, 'base')]
            nc_o = [summ.get('nonconv_rate_se3') for _, summ in seeds(sc, None, 'hminL', sn, 'dual')]
            nc_b = [summ.get('nonconv_rate_se3') for _, summ in seeds(sc, None, 'mb', sn, 'base')]
            row.update(coll_ours=coll_o, coll_mb=coll_b, nonconv_ours=nc_o, nonconv_mb=nc_b)
            if any(row[c[0]] for c in columns):
                table.append(row)
                lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(sc, variant, ' | '.join(cells), ms(coll_o), ms(coll_b), ms(nc_o), ms(nc_b)))

    # ---- wall track
    lines += ['', '## Wall with a hole (room of env1, slab with a centred square hole of side 90 / 75 / 66 units; the shapes are 60 x 10 x 60)', '',
              'Through-the-hole sets: start and goal on opposite sides of the slab (1000 cases each, verified by an aligned passage).', '',
              '| shape | hole | clearance face-on | ' + ' | '.join(c[0] for c in columns) + ' | ours dual coll % | NTFields 8k coll % | ours dual nonconv % | NTFields 8k nonconv % |',
              '|' + '---|' * (3 + len(columns) + 4)]
    wall = []
    for hole in HOLES:
        for sc in SCENES:
            sn = '{}_hole{}_through'.format(SHAPES[sc], hole)
            row = dict(scene=sc, hole=hole, set=sn)
            cells = []
            for label, model, planner, key in columns:
                vals = [summ.get(key) for _, summ in seeds(sc, hole, model, sn, planner)]
                row[label] = vals
                cells.append(ms(vals))
            coll_o = [summ.get('collision_rate_se3') for _, summ in seeds(sc, hole, 'hminL', sn, 'dual')]
            coll_b = [summ.get('collision_rate_se3') for _, summ in seeds(sc, hole, 'mb', sn, 'base')]
            nc_o = [summ.get('nonconv_rate_se3') for _, summ in seeds(sc, hole, 'hminL', sn, 'dual')]
            nc_b = [summ.get('nonconv_rate_se3') for _, summ in seeds(sc, hole, 'mb', sn, 'base')]
            row.update(coll_ours=coll_o, coll_mb=coll_b, nonconv_ours=nc_o, nonconv_mb=nc_b)
            ver = load(os.path.join(a.sets, sn, 'verification.json'))
            row['passage_clear'] = ver['passage']['min_clear'] if ver and ver.get('passage') else None
            if any(row[c[0]] for c in columns):
                wall.append(row)
                lines.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    sc, hole, '-' if row['passage_clear'] is None else '{:.4f}'.format(row['passage_clear']), ' | '.join(cells),
                    ms(coll_o), ms(coll_b), ms(nc_o), ms(nc_b)))

    # ---- paired tests: ours dual (seed s) vs NTFields 8k (seed s), s = 0..2, pooled per set
    lines += ['', '## Paired per-case comparison (seed-matched: homotopy seed s vs NTFields 8k seed s, s = 0, 1, 2; exact McNemar)', '',
              '| set | n pairs | both | ours only | NTFields only | net | p | ours base vs NTFields 8k: ours only / NTFields only / p |',
              '|---|---|---|---|---|---|---|---|']
    paired = []
    all_sets = [(sc, None, set_name(sc, v)) for sc in SCENES for v in VARIANTS] + \
               [(sc, h, '{}_hole{}_through'.format(SHAPES[sc], h)) for h in HOLES for sc in SCENES]
    for sc, hole, sn in all_sets:
        rows = {}
        for key_name, model, planner in (('dual', 'hminL', 'dual'), ('base', 'hminL', 'base')):
            both = ours = theirs = n = 0
            for s in (0, 1, 2):
                ro = R.get(('{}{}_{}_s{}'.format(sc, hole or '', model, s), sn, planner))
                rb = R.get(('{}{}_mb_s{}'.format(sc, hole or '', s), sn, 'base'))
                if ro is None or rb is None:
                    continue
                B = {r['case']: r['success_se3'] for r in rb[1]}
                for r in ro[1]:
                    if r['case'] not in B:
                        continue
                    n += 1
                    o, b = bool(r['success_se3']), bool(B[r['case']])
                    both += o and b
                    ours += o and not b
                    theirs += b and not o
            rows[key_name] = (n, both, ours, theirs)
        n, both, ours, theirs = rows['dual']
        if n:
            nb, _, ob, tb = rows['base']
            p = binom_two_sided(theirs, ours + theirs)
            pb = binom_two_sided(tb, ob + tb)
            paired.append(dict(set=sn, n=n, both=both, ours_only=ours, theirs_only=theirs, p=p, base_ours_only=ob, base_theirs_only=tb, base_p=pb))
            lines.append('| {} | {} | {} | {} | {} | {:+d} | {:.2g} | {} / {} / {:.2g} |'.format(sn, n, both, ours, theirs, ours - theirs, p, ob, tb, pb))

    # ---- bins (pooled over seeds): success by relative angle, |dp|, min endpoint clearance; ours dual vs NTFields 8k
    def binned(sc, hole, model, planner, sets_):
        acc = defaultdict(lambda: [0, 0])
        for sn in sets_:
            ver = load(os.path.join(a.sets, sn, 'verification.json'))
            clear = None
            if ver:
                clear = [min(d['clear_start'], d['clear_goal']) for d in ver['per_pair']]
            for (run, s, p), (_, recs) in R.items():
                msc, mh, mm, seed = model_of(run)
                if not (msc == sc and mh == hole and mm == model and s == sn and p == planner):
                    continue
                for r in recs:
                    ang = r['straight_rot_geo'] * 360.0
                    dp = r['straight_trans']
                    for lo_, hi_ in ANG_BINS:
                        if lo_ <= ang < hi_:
                            acc[('ang', lo_)][0] += r['success_se3']; acc[('ang', lo_)][1] += 1
                    for lo_, hi_ in DP_BINS:
                        if lo_ <= dp < hi_:
                            acc[('dp', lo_)][0] += r['success_se3']; acc[('dp', lo_)][1] += 1
                    if clear is not None and r['case'] < len(clear):
                        c = clear[r['case']]
                        for lo_, hi_ in CL_BINS:
                            if lo_ <= c < hi_:
                                acc[('cl', lo_)][0] += r['success_se3']; acc[('cl', lo_)][1] += 1
        return acc

    lines += ['', '## Success by case property (targeted sets pooled per scene, all seeds; SE(3) success %, n cases in parentheses)', '']
    bins_out = {}
    for sc in SCENES:
        sets_ = [set_name(sc, v) for v in VARIANTS if v != 'std']
        bo = binned(sc, None, 'hminL', 'dual', sets_)
        bb = binned(sc, None, 'mb', 'base', sets_)
        bbase = binned(sc, None, 'hminL', 'base', sets_)
        lines += ['| {} | bin | ours base | ours dual | NTFields 8k |'.format(sc), '|---|---|---|---|---|']
        for kind, bins, fmt in (('ang', ANG_BINS, 'rotation {:.0f}-{:.0f} deg'), ('dp', DP_BINS, '|dp| {:.1f}-{:.1f}'), ('cl', CL_BINS, 'clearance {:.3f}-{:.3f}')):
            for lo_, hi_ in bins:
                k = (kind, lo_)
                cells = []
                for acc in (bbase, bo, bb):
                    s_, n_ = acc.get(k, [0, 0])
                    cells.append('-' if n_ == 0 else '{:.1f} ({})'.format(100.0 * s_ / n_, n_))
                    bins_out[(sc, kind, lo_)] = [acc.get(k, [0, 0]) for acc in (bbase, bo, bb)]
                lines.append('| | {} | {} |'.format(fmt.format(lo_, min(hi_, 180)), ' | '.join(cells)))
        lines.append('')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'expand_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(a.out, 'expand_summary.json'), 'w') as fh:
        json.dump({'targeted': table, 'wall': wall, 'paired': paired,
                   'bins': [{'scene': k[0], 'kind': k[1], 'lo': k[2], 'ours_base': v[0], 'ours_dual': v[1], 'ntfields8k': v[2]} for k, v in bins_out.items()]},
                  fh, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
