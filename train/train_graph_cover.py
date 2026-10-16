"""Train the cover-quotient metric head (models/graph_method_triangle/cover_decoder.py) in the graph_lie pipeline.

    python train/train_graph_cover.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_cov_lp2_s0 \
        --seed 0 --grad lie --speed iso --set n_cut=2 res_kind="'lp'" res_scale=0.4

Identical to train/train_graph_lie.py (same trainer, loss, random-number sequence) apart from the default
architecture and output root; any arch of the triangle package can be trained through it.
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/Tshape3d_env1')
    ap.add_argument('--out_root', default='./Experiments/graph_cover')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_cover')
    ap.add_argument('--grad', default='lie', choices=['chart', 'lie'])
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'])
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    a = ap.parse_args()
    from models.graph_lie import model_train_lie as md
    m = md.Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    P = m.Params['Training']
    P['Number of Epochs'] = a.epochs
    P['Seed'] = a.seed
    P['Grad mode'] = a.grad
    P['Speed model'] = a.speed
    P['Weight'] = a.weight
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
