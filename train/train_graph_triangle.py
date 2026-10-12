"""Train an SE(3) smooth-graph travel-time field (triangle-inequality heads or baselines).

    python train/train_graph_triangle.py --config configs/graph_method_triangle/graph_tri.json --seed 0
    python train/train_graph_triangle.py --set arch=graph_tri_metric mode=physics steps=12000 --name tri_metric_s0

Every key of models.graph_method_triangle.model_function_graph.default_config can be
set through the JSON config or `--set key=value` (later wins).
"""

import argparse
import ast
import json
import sys

sys.path.append('.')

from models.graph_method_triangle import model_function_graph as mf
from models.graph_method_triangle import model_train_graph as mt


def parse_value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None)
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    p.add_argument('--name', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--out_root', default='./Experiments/graph_method_triangle')
    a = p.parse_args()
    over = {}
    if a.config:
        with open(a.config) as f:
            over.update(json.load(f))
    for kv in a.set:
        k, v = kv.split('=', 1)
        over[k] = parse_value(v)
    if a.seed is not None:
        over['seed'] = a.seed
    if a.device is not None:
        over['device'] = a.device
    if a.name is not None:
        over['name'] = a.name
    elif 'name' in over and a.seed is not None:
        over['name'] = '{}_s{}'.format(over['name'], a.seed)
    if over.get('mode') == 'shape':
        from models.graph_method_triangle.model_function_shape import shape_config
        cfg = shape_config(**over)
    else:
        cfg = mf.default_config(**over)
    mt.Model(cfg, out_root=a.out_root).train()


if __name__ == '__main__':
    main()
