"""Aggregate Shape-NTFields evaluator outputs (success_rate.txt) into a markdown table.

    python models/graph_method/scripts/aggregate_shape.py --groups \
        'graph_gram=results/graph_method_stageE/stageE_graph_gram_s*_c1000' \
        'metric head=results/graph_method_stageE/stageE_metric_s*_c1000' --out results/graph_method_stageE/summary.md
"""
import argparse
import glob
import math
import re

ROWS = [('regular/forward', 'forward'), ('flipped/reverse', 'reverse'), ('either/or', 'either'),
        ('alt/alternate', 'alternate')]


def parse(path):
    txt = open(path).read()
    out = {}
    for key, short in ROWS:
        m = re.search(re.escape(key) + r'\s*:\s*([0-9.]+)', txt)
        if m:
            out[short] = 100 * float(m.group(1))
    for key in ('collision', 'no_converge'):
        m = re.search(r'^\s+' + key + r'\s*:\s*(\d+)', txt, re.M)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r'episodes\s*:\s*(\d+)', txt)
    out['episodes'] = int(m.group(1)) if m else None
    return out


def ms(v):
    if not v:
        return 'n/a'
    m = sum(v) / len(v)
    if len(v) == 1:
        return '{:.1f}'.format(m)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return '{:.1f} ± {:.1f}'.format(m, s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--groups', nargs='+', required=True)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    lines = ['| model | seeds | cases | forward (%) | reverse (%) | either (%) | alternate (%) | collision | no-converge |',
             '|---|---|---|---|---|---|---|---|---|']
    for g in a.groups:
        label, pat = g.split('=', 1)
        files = sorted(glob.glob(pat + '/success_rate.txt')) or sorted(glob.glob(pat))
        rs = [parse(f) for f in files]
        if not rs:
            continue
        col = lambda k: [r[k] for r in rs if k in r]
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            label, len(rs), rs[0].get('episodes'), ms(col('forward')), ms(col('reverse')), ms(col('either')),
            ms(col('alternate')), ms(col('collision')), ms(col('no_converge'))))
    text = '\n'.join(lines)
    print(text)
    if a.out:
        open(a.out, 'w').write(text + '\n')


if __name__ == '__main__':
    main()
