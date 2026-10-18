"""Train the double-cover (homotopy) head in the graph_lie pipeline with lifted training pairs.

    python train/train_graph_homotopy.py --dataPath datasets/3dshape/Tshape3d_env1 --name T_hm_s0 --seed 0 \
        --lift random --lift-far 0.5 --set res_even=32 res_odd=32 pair_dim=64

--set KEY=VALUE pairs go to the head (python literals). --lift min reproduces a K = 1 conformal branch head
trained on the base field (same loss, same random-number sequence up to the head).
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/Tshape3d_env1')
    ap.add_argument('--out_root', default='./Experiments/graph_homotopy')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_homotopy')
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'])
    ap.add_argument('--lift', default='min', choices=['min', 'random', 'both'])
    ap.add_argument('--lift-far', type=float, default=0.5)
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    a = ap.parse_args()
    from models.graph_homotopy import model_train_homotopy as md
    m = md.Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    P = m.Params['Training']
    P['Number of Epochs'] = a.epochs
    P['Seed'] = a.seed
    P['Grad mode'] = 'lie'
    P['Speed model'] = a.speed
    P['Weight'] = a.weight
    P['Lift'] = a.lift
    P['Lift far'] = a.lift_far
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
