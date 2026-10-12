"""Eikonal residual / error / latent by relative angle for a checkpoint."""
import sys, math
sys.path.append('.')
import torch
from models.graph_method_triangle import model_train_graph as mt
from models.graph_method_triangle.data.poses import PoseSampler
from models.graph_method_triangle.fields import oracle
from models.graph_method_triangle.geometry import se3, tangents

f, ck = mt.load_function(sys.argv[1], device='cuda:0')
s = PoseSampler(seed=5, device='cuda:0')
tx, Rx, ty, Ry = s.pure_rotation_pairs(20000)
ty = tx + 0.3 * s.unit_vectors(20000, 3)  # small fixed translation offset
gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
T, cx, cy = tangents.tangent_gradients(f.network.value, gx, gy, 1.0, 1.0, create_graph=False)
Ts = oracle.time_star(tx, Rx, ty, Ry, 1.0, 1.0)
ang = oracle.relative_angle(Rx, Ry) * 180 / math.pi
nx = (cx ** 2).sum(-1)
ct, cr = (cx[:, :3] ** 2).sum(-1), (cx[:, 3:] ** 2).sum(-1)
# true split
dt = 0.3
ct_true = dt ** 2 / Ts ** 2
with torch.no_grad():
    a, b = f.network.predict_latent(gx, gy) if f.network.has_latent else (T * 0, T * 0)
_, bs = oracle.latent_star(tx, Rx, ty, Ry)
print('angle   |c|^2   |c_t|^2 (true)   |c_r|^2 (true)   T/T*    b    b*')
for lo in range(0, 180, 15):
    m = (ang >= lo) & (ang < lo + 15)
    print('%3d-%3d  %.3f   %.3f (%.3f)   %.3f (%.3f)   %.3f  %.3f  %.3f' % (
        lo, lo + 15, nx[m].mean(), ct[m].mean(), ct_true[m].mean(), cr[m].mean(), (1 - ct_true[m]).mean(),
        (T / Ts)[m].mean(), b[m].mean(), bs[m].mean()))
