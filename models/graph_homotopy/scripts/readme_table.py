#!/usr/bin/env python3
"""Markdown table of the main variants for the README (seed means +- population std of the SE(3) success %)."""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

SC = {'Tshape3d_env1': 'T', 'Lshape3d_env3': 'L', 'Ashape3d_env4': 'A'}
LABELS = {'hminL': 'graph_homotopy, 8000 epochs', 'hmin': 'graph_homotopy, 4000 epochs', 'bc4': 'branch decoder K = 4 (reference)'}


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    return '{:.1f}'.format(100 * v[0]) if len(v) == 1 else '{:.1f} +- {:.1f}'.format(100 * statistics.mean(v), 100 * statistics.pstdev(v))


tags = sys.argv[1].split(',') if len(sys.argv) > 1 else ['hminL', 'hmin', 'bc4']
g = defaultdict(list)
for f in glob.glob('results/graph_homotopy/plan/*/*__base.json'):
    run = os.path.basename(os.path.dirname(f))
    m = re.match(r'([TLA])_(.+)_s(\d+)$', run)
    if not m or m.group(2) not in tags:
        continue
    b = json.load(open(f))['summary']
    fd = f.replace('__base', '__dual')
    d = json.load(open(fd))['summary'] if os.path.exists(fd) else {}
    g[(m.group(2), m.group(1))].append((b, d))
print('| head | scene | seeds | base planner | collision | non-conv. | cover, one plan (sigma*) | cover, both classes | + shorter-path rule | either (oracle) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for tag in tags:
    for sc in 'TLA':
        recs = g.get((tag, sc), [])
        if not recs:
            continue
        print('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            LABELS.get(tag, tag), sc, len(recs), ms([b['success_rate_se3'] for b, _ in recs]), ms([b['collision_rate_se3'] for b, _ in recs]),
            ms([b['nonconv_rate_se3'] for b, _ in recs]), ms([d.get('success_se3_cover') for _, d in recs]),
            ms([d.get('success_rate_se3') for _, d in recs]), ms([d.get('success_se3_shorter') for _, d in recs]),
            ms([d.get('success_se3_either') for _, d in recs])))
