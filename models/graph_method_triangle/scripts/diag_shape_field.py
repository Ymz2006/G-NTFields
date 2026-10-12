"""Stage E field sanity: T vs free-space distance, and obstacle-agnostic descent to the goal."""
import sys, math
sys.path.append('.')
import numpy as np
import torch
from models.graph_method_triangle import model_train_graph as mt
from models.graph_method_triangle.model_function_shape import coords_to_features
from models.graph_method_triangle.geometry import se3
from models.graph_method_triangle.planning import rollout

f, ck = mt.load_function(sys.argv[1], device='cuda:0')
pts = torch.from_numpy(np.load(sys.argv[2])).float().cuda()[:300]
g0, g1 = coords_to_features(pts[:, :6]), coords_to_features(pts[:, 6:])
with torch.no_grad():
    T = f.network.value(g0, g1)
t0, R0 = se3.split_features(g0); t1, R1 = se3.split_features(g1)
ang = se3.rotation_angle(R0.transpose(-1, -2) @ R1)
Tfree = torch.sqrt(((t0 - t1) ** 2).sum(-1) + (ang / (2 * math.pi)) ** 2)
r = (T / Tfree)
print('T/Tfree  p5 %.3f p50 %.3f p95 %.3f   T p50 %.3f  Tfree p50 %.3f' % (
    r.quantile(.05), r.median(), r.quantile(.95), T.median(), Tfree.median()))
res = rollout.rollout(f.network, t0, R0, t1, R1, s_t=1.0, s_r=2 * math.pi, mode='gradient', h=0.01,
                      max_steps=800)
print('obstacle-agnostic gradient descent success %.3f  fallback frac %.3f  cost/T p50 %.3f' % (
    res['success'].float().mean(), (res['fallback'] > 0).float().mean(),
    (res['cost'] / T)[res['success']].median() if res['success'].any() else float('nan')))
print('final err_t p50 %.3f err_r p50 %.3f' % (res['err_t'].median(), res['err_r'].median()))
