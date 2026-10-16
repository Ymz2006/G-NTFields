"""Train the multi-branch decoder that learns the cut (models/graph_mb) in the graph_lie pipeline.

    python train/train_graph_mb.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_mb4_s0 --seed 0 \
        --arch graph_mb --set n_branch=4 --spec delta=0.1 w_dist=0.1
    python train/train_graph_mb.py ... --arch graph_mb --end-weight 1          # + end constraint
    python train/train_graph_mb.py ... --arch graph_mb_free --set n_branch=4   # the spec's free heads

--set KEY=VALUE pairs go to the head, --spec KEY=VALUE pairs to the loss / management
(models/graph_mb/model_function_mb.DEFAULT_SPEC lists them; python literals).
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def kv(items):
    out = {}
    for s in items:
        k, v = s.split('=', 1)
        try:
            v = eval(v, {}, {})
        except Exception:
            pass
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/Tshape3d_env1')
    ap.add_argument('--out_root', default='./Experiments/graph_mb')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_mb')
    ap.add_argument('--grad', default='lie', choices=['chart', 'lie'])
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'])
    ap.add_argument('--end-weight', type=float, default=0.0)
    ap.add_argument('--planar', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE')
    ap.add_argument('--spec', nargs='*', default=[], help='loss / management options KEY=VALUE')
    a = ap.parse_args()
    from models.graph_mb import model_train_mb as md
    m = md.Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    P = m.Params['Training']
    P['Number of Epochs'] = a.epochs
    P['Seed'] = a.seed
    P['Grad mode'] = a.grad
    P['Speed model'] = a.speed
    P['Weight'] = a.weight
    P['End weight'] = a.end_weight
    P['Planar'] = a.planar
    P['Spec'] = kv(a.spec)
    g = {'arch': a.arch}
    g.update(kv(a.set))
    m.Params['Network']['Graph'] = g
    t0 = time.time()
    m.train()
    print('Training time: {:.1f}s'.format(time.time() - t0))


if __name__ == '__main__':
    main()
