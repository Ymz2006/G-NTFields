"""Train a graph head with the Lie-gradient (or chart-gradient) Eikonal loss (models/graph_lie).

    python train/train_graph_lie.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_gmetric_iso_lie_s0 \
        --device cuda:0 --seed 0 --arch graph_metric --grad lie --speed iso
    python train/train_graph_lie.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_trieq_aniso_lie_s0 \
        --seed 0 --arch graph_tri_metric --grad lie --speed aniso --set rot_mode=equivariant

--grad   chart (partial derivatives d tau / d(t, rotvec / 2 pi)) | lie (body-frame Lie gradient)
--speed  combined (S |g| = 1) | iso (S_d |g| = 1) | aniso (sqrt(S_d^2 |g_v|^2 + S_a^2 |g_w|^2) = 1)
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
    ap.add_argument('--out_root', default='./Experiments/graph_lie')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_metric')
    ap.add_argument('--grad', default='lie', choices=['chart', 'lie'])
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'],
                    help='residual weight: exp(-tau/2) as NTFields (default), with tau detached, or 1 '
                         '(the obstacle-free control uses none)')
    a = ap.parse_args()
    from models.graph_lie import model_train_lie as md
    m = md.Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    m.Params['Training']['Number of Epochs'] = a.epochs
    m.Params['Training']['Seed'] = a.seed
    m.Params['Training']['Grad mode'] = a.grad
    m.Params['Training']['Speed model'] = a.speed
    m.Params['Training']['Weight'] = a.weight
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
