"""Stage E: Eikonal |c| by speed bin, per endpoint, on held-out dataset pairs."""
import sys
sys.path.append('.')
import torch
from models.graph_method import model_train_graph as mt
from models.graph_method.model_function_shape import shape_config, ShapeFunction

ck = torch.load(sys.argv[1], map_location='cpu')
cfg = dict(ck['config']); cfg['device'] = 'cuda:0'
f = ShapeFunction(cfg)
f.network.load_state_dict(ck['model_state_dict']); f.network.eval()
D = f.data
idx = f.hold_idx[:8192]
T, cx, cy = f.tangent_c(D['g0'][idx], D['g1'][idx], D['Sd'][idx, 0], D['Sa'][idx, 0], D['Sd'][idx, 1], D['Sa'][idx, 1], create_graph=False)
for name, c, S in (('x0 (band)', cx, D['Sd'][idx, 0]), ('x1 (random)', cy, D['Sd'][idx, 1])):
    n = torch.linalg.norm(c, dim=-1)
    print(name)
    for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.99), (0.99, 1.01)):
        m = (S >= lo) & (S < hi)
        if m.sum() > 10:
            print('  S in [%.2f,%.2f): n=%5d  |c| p10 %.3f p50 %.3f p90 %.3f' % (lo, hi, m.sum(), n[m].quantile(.1), n[m].median(), n[m].quantile(.9)))
print('T p50 %.3f' % T.median())
