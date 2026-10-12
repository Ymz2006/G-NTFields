"""Pure-translation pairs: T/|dt| vs |dt| and vs how far the pair straddles the box centre."""
import sys
sys.path.append('.')
import torch
from models.graph_method import model_train_graph as mt
from models.graph_method.data.poses import PoseSampler
from models.graph_method.geometry import se3

f, ck = mt.load_function(sys.argv[1], device='cuda:0')
s = PoseSampler(seed=9, device='cuda:0')
tx, Rx, ty, Ry = s.pure_translation_pairs(20000)
with torch.no_grad():
    T = f.network.value(se3.features(tx, Rx), se3.features(ty, Ry))
d = torch.linalg.norm(tx - ty, dim=-1)
r = T / d
mid = torch.linalg.norm(0.5 * (tx + ty), dim=-1)          # pair midpoint distance to centre
cosang = ((tx * ty).sum(-1) / (torch.linalg.norm(tx, dim=-1) * torch.linalg.norm(ty, dim=-1)))
print('|dt| bins:')
for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3.5)):
    m = (d >= lo) & (d < hi)
    print('  [%.2f,%.2f) n=%5d T/|dt| p5 %.3f p50 %.3f p95 %.3f' % (lo, hi, m.sum(), r[m].quantile(.05), r[m].median(), r[m].quantile(.95)))
print('pairs with |dt|>1, by cos(angle between t_x and t_y):')
for lo, hi in ((-1, -.5), (-.5, 0), (0, .5), (.5, 1.01)):
    m = (d > 1) & (cosang >= lo) & (cosang < hi)
    print('  cos in [%.1f,%.1f) n=%5d T/|dt| p5 %.3f p50 %.3f' % (lo, hi, m.sum(), r[m].quantile(.05), r[m].median()))
