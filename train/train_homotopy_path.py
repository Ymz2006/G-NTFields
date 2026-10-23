"""Train the double-cover (homotopy) head on a path-constrained cell (models/path_constrained): the graph_lie
pipeline of train_graph_homotopy.py with the Eikonal residual formed on M = path x SO(3) (the translation block of
the Lie gradient projected onto the path tangent, models/path_constrained/losses.py).

    python train/train_homotopy_path.py --dataPath datasets/3dshape/data/prism_env1_a --name env1a_hom_s0 --seed 0 \
        --epochs 8000 --lift min

Checkpoints keep the graph_native contract, so the evaluator loads them as every homotopy checkpoint.
"""
import argparse
import os
import sys
import time

sys.path.append('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataPath', required=True)
    ap.add_argument('--out_root', default='./Experiments/path_constrained')
    ap.add_argument('--name', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8000)
    ap.add_argument('--arch', default='graph_homotopy')
    ap.add_argument('--speed', default='iso', choices=['combined', 'iso', 'aniso'])
    ap.add_argument('--weight', default='exp', choices=['exp', 'exp_detached', 'none'])
    ap.add_argument('--lift', default='min', choices=['min', 'random', 'both'])
    ap.add_argument('--lift-far', type=float, default=0.5)
    ap.add_argument('--set', nargs='*', default=[], help='head options KEY=VALUE (python literals)')
    a = ap.parse_args()
    from models.graph_homotopy import model_train_homotopy as md
    from models.path_constrained.losses import ChartTangent, homotopy_function_class

    Fn = homotopy_function_class()
    Fn.tangent = ChartTangent(a.dataPath)

    class Model(md.Model):
        def _build(self, B):
            super()._build(B)
            P = self.Params['Training']
            self.function = Fn(self.folder, self.Params['Device'], self.network, self.dim,
                               grad_mode=P['Grad mode'], speed_model=P['Speed model'],
                               weight_mode=P.get('Weight', 'exp'), lift=P.get('Lift', 'min'),
                               lift_far=P.get('Lift far', 0.5), seed=P.get('Seed', 0))

    m = Model(a.out_root, a.dataPath, 6, [0.0] * 6, device=a.device)
    m.folder = os.path.join(a.out_root, a.name)
    P = m.Params['Training']
    P['Number of Epochs'] = a.epochs
    P['Save Every * Epoch'] = 4000          # keeps Model_Epoch_04000_*.pt (and the final one) next to latest.pt
    P['Seed'] = a.seed
    P['Grad mode'] = 'lie'
    P['Speed model'] = a.speed
    P['Weight'] = a.weight
    P['Lift'] = a.lift
    P['Lift far'] = a.lift_far
    P['Constraint'] = {'chart': Fn.tangent.chart, 'path_L': Fn.tangent.path.L}
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
