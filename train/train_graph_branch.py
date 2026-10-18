"""Train a branch-decoder head (or any graph head) in the graph_lie pipeline with the endpoint constraint.

    python train/train_graph_branch.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_branch_s0 \
        --seed 0 --arch graph_branch --grad lie --speed iso --set n_branch=4
    python train/train_graph_branch.py --dataPath datasets/3dshape/data/Tshape3d_env1 --name T_bconf_s0 \
        --seed 0 --arch graph_branch_conf --grad lie --speed iso --end-weight 1 --set n_branch=4

--set KEY=VALUE pairs go to the head (python literals). With --end-weight 0 and a non-branch head this is
exactly train/train_graph_lie.py (same random-number sequence).
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', default='datasets/3dshape/data/Tshape3d_env1')
    ap.add_argument('--out_root', default='./Experiments/graph_branch')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--arch', default='graph_branch')
    ap.add_argument('--grad', default='lie', choices=['chart', 'lie'])
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'])
    ap.add_argument('--end-weight', type=float, default=0.0, help='endpoint cone-slope constraint weight (0 = off)')
    ap.add_argument('--planar', action='store_true', help='in-plane Eikonal norm (synthetic SE(2) benchmark)')
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    a = ap.parse_args()
    from models.graph_branch import model_train_branch as md
    from models.graph_branch import chart_metric  # noqa: F401  (registers arch 'metric_chart', the original metric net)
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
