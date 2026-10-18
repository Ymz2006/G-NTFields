"""Train a graph head inside the original metric NTFields pipeline (models/graph_native).

    python train/train_graph_native.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_gn_trieq_chart_s0 \
        --device cuda:0 --seed 0 --arch graph_tri_metric --grad chart --set rot_mode=equivariant

--set KEY=VALUE pairs go to the head (models/graph_native/model_network_metric.DEFAULT_GRAPH keys).
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--out_root', default='./Experiments/graph_native')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_tri_metric')
    ap.add_argument('--grad', default='chart', choices=['chart', 'lie'])
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    a = ap.parse_args()
    from models.graph_native import model_train_metric as md
    m = md.Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    m.Params['Training']['Number of Epochs'] = a.epochs
    m.Params['Training']['Seed'] = a.seed
    m.Params['Training']['Grad mode'] = a.grad
    g = {'arch': a.arch}
    for kv in a.set:
        k, v = kv.split('=', 1)
        try:
            v = eval(v, {}, {})
        except Exception:
            pass
        g[k] = v
    m.Params['Network']['Graph'] = g
    t0 = time.time()
    m.train()
    print('Training time: {:.1f}s'.format(time.time() - t0))


if __name__ == '__main__':
    main()
