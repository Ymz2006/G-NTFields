#!/usr/bin/env python3
"""Table of the nine decoder properties measured on trained fields (decoder_properties.py JSONs in
results/graph_branch/properties/). One row per run, grouped by scene; mean +- sd per variant at the end.

    python3 models/graph_branch/scripts/aggregate_props.py   # -> results/graph_branch/summary/properties.{md,json}
"""
import glob
import json
import os
import re
import statistics as st

HOST = '/media/jeffrey/G-Shape-NTFields'
PROP = os.path.join(HOST, 'results/graph_branch/properties')
OUT = os.path.join(HOST, 'results/graph_branch/summary')
COLS = [
    ('P1_goal_max_abs', 'P1 max abs T(p,p)', '{:.1e}'),
    ('P2_sym_rel_max', 'P2 max sym. err.', '{:.1e}'),
    ('P3_triangle_viol_rate', 'P3 viol. rate %', 'pct'),
    ('P3_triangle_viol_max_rel', 'P3 max viol. %', 'pct'),
    ('P4_frac_below_Tfree', 'P4 T<T_free %', 'pct'),
    ('P4_min_T_over_Tfree', 'P4 min T/T_free', '{:.3f}'),
    ('P5_cone_conv_e0', 'P5 cone conv. (1e-3 vs 1e-4)', '{:.3f}'),
    ('P5_cone_conv_1e-2_e0', 'P5 cone conv. (1e-2)', '{:.3f}'),
    ('P6_anisotropy_median_e0', 'P6 anisotropy', '{:.2f}'),
    ('P6_eikonal_at_goal_median_e0', 'P6 S slope at goal', '{:.3f}'),
    ('P6_eikonal_at_goal_median_lowS_e0', 'P6 S slope (S<0.05)', '{:.3f}'),
    ('P6_eikonal_at_goal_comb_median_e0', 'P6 S_comb slope', '{:.3f}'),
    ('P7_lines_with_V_kink', 'P7 V-kink lines /200', '{:.0f}'),
    ('P7_lines_with_ridge', 'P7 ridge lines /200', '{:.0f}'),
    ('P7_lines_with_jump', 'P7 jump lines /200', '{:.0f}'),
    ('P8_decoder_kink_offset_from_halfturn_deg_median', 'P8 b-kink off half turn (deg)', '{:.1f}'),
    ('P8_frac_slices_with_kink_b_gt_0.999', 'P8 slices with b-kink %', 'pct'),
    ('P8_route_switch_offset_from_halfturn_deg_median', 'P8 route switch off half turn (deg)', '{:.1f}'),
    ('P8_frac_switches_beyond_10deg', 'P8 switches >10 deg %', 'pct'),
    ('P8_route_switches_per_slice_mean', 'P8 switches / slice', '{:.2f}'),
    ('P9_rotation_term_max', 'P9 rot. term max', '{:.3f}'),
    ('P9_rot_slice_Tmax_median', 'P9 rot. slice T max', '{:.2f}'),
    ('branch_long_way_frac', 'long-way pairs %', 'pct'),
]


def f(v, fmt):
    if v is None:
        return '-'
    return '{:.2f}'.format(100 * v) if fmt == 'pct' else fmt.format(v)


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(PROP, '*.json'))):
        d = json.load(open(p))
        d['_run'] = os.path.basename(p)[:-5]
        rows.append(d)
    lines = ['# Decoder properties measured on trained fields', '',
             'decoder_properties.py on endpoints of the scene training pairs (collision-free poses with stored clearance',
             'speeds). P5/P6 at eps = 1e-4 unless noted; "S slope at goal" = median S_d(p) x cone slope (1 = the cone matches',
             'the Eikonal equation at the goal); P8 offsets are measured on rotation slices at the goal position.', '',
             '| run | ' + ' | '.join(c[1] for c in COLS) + ' |', '|' + '---|' * (len(COLS) + 1)]
    for d in rows:
        lines.append('| {} | {} |'.format(d['_run'], ' | '.join(f(d.get(c[0]), c[2]) for c in COLS)))
    # per-variant means
    groups = {}
    for d in rows:
        m = re.match(r'^(ref_|orig_)?([TLA])_(.+?)(?:_iso_lie)?_s(\d+)$', d['_run'])
        if m and m.group(1) == 'orig_':    # the original metric NTFields checkpoints (decoder_properties.py --orig)
            key = '{}_orig'.format(m.group(2))
        else:
            key = '{}_{}'.format(m.group(2), m.group(3)) if m else d['_run']
        groups.setdefault(key, []).append(d)
    lines += ['', '## Per variant (mean over seeds)', '', '| variant | n | ' + ' | '.join(c[1] for c in COLS) + ' |',
              '|' + '---|' * (len(COLS) + 2)]
    out = {}
    for k in sorted(groups):
        ds = groups[k]
        cells = []
        for c in COLS:
            vals = [d.get(c[0]) for d in ds if d.get(c[0]) is not None]
            cells.append(f(st.mean(vals), c[2]) if vals else '-')
            out.setdefault(k, {})[c[0]] = vals
        lines.append('| {} | {} | {} |'.format(k, len(ds), ' | '.join(cells)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'properties.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'properties.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
