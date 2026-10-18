"""Score physics-trained fields on the synthetic exact-truth problems (datasets/3dshape/data/synth_*) with the same
metrics as synth_fit.py (field error, rotational ridge location, cap, planning in the problem metric).

    python models/graph_branch/scripts/synth_eval.py --out results/graph_branch/synth_phys \
        Experiments/graph_branch/S4_br4s_aniso_s0/latest.pt ...

The problem (kind, A, ...) is read from the dataset's meta.json named in the run's config.json.
"""
import argparse
import json
import os
import sys

import torch

sys.path.append('.')

from models.graph_branch import synth  # noqa: E402
from models.graph_branch import chart_metric  # noqa: E402,F401  (arch 'metric_chart')
from models.graph_branch.scripts import synth_fit  # noqa: E402
from models.graph_native import model_train_metric as gn_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('ckpts', nargs='+')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ck in a.ckpts:
        run = os.path.basename(os.path.dirname(ck))
        dst = os.path.join(a.out, run + '.json')
        if os.path.exists(dst):
            print('skip', run)
            continue
        cfg = json.load(open(os.path.join(os.path.dirname(ck), 'config.json')))
        meta = json.load(open(os.path.join(cfg['DataPath'], 'meta.json')))
        prob = synth.Problem(meta['kind'], A=meta['A'], w=meta['w'], psi0=meta['psi0'], r0=meta.get('r0', 0.15),
                             half=meta['half'])
        m = gn_train.Model('/tmp/se', cfg['DataPath'], 6, [0.0] * 6, device=a.device)
        m.load(ck)
        m.network.eval()
        res = synth_fit.evaluate(m.network, prob, a.device)
        res.update({'run': run, 'arch': m.network.graph['arch'], 'kind': meta['kind'], 'A': meta['A'],
                    'graph': {k: v for k, v in m.network.graph.items() if isinstance(v, (int, float, str, bool))}})
        with open(dst, 'w') as fh:
            json.dump(res, fh, indent=1)
        print(run, json.dumps(res))


if __name__ == '__main__':
    main()
