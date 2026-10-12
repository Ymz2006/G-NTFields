"""Time each component of one physics training step."""
import sys, time
sys.path.append('.')
import torch
from models.graph_method import model_function_graph as mf
from models.graph_method.geometry import se3
from models.graph_method.losses import bellman, eikonal

arch = sys.argv[1] if len(sys.argv) > 1 else 'graph_gram'
cfg = mf.default_config(arch=arch, mode='physics', device='cuda:0', fourier_sigma=0.03)
f = mf.Function(cfg)
opt = torch.optim.Adam(f.network.parameters(), 1e-3)

def timeit(name, fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = fn()
    torch.cuda.synchronize(); print('%-28s %7.2f ms' % (name, (time.time() - t) / n * 1e3)); return r

c = cfg
batch = f.sampler.training_pairs(c['batch_size'])
tx, Rx, ty, Ry, fam = batch
gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
timeit('sample', lambda: f.sampler.training_pairs(c['batch_size']))
timeit('mask', lambda: eikonal.regular_mask(tx, Rx, ty, Ry, 1., 1., 1., 1e-3, 1e-4))
def eik():
    _, cx, cy = eikonal.eikonal_terms(f.network, gx, gy, create_graph=True)
    return eikonal.endpoint_residual(cx, cy).mean()
timeit('eikonal fwd (create_graph)', eik)
timeit('eikonal fwd+bwd', lambda: eik().backward())
timeit('value fwd', lambda: f.network.value(gx, gy))

u = torch.nn.functional.normalize(torch.randn(gx.shape[0], 6, device=gx.device), dim=-1)
U = bellman.candidate_controls(u, 16, 15, 0.3, f.gen)
timeit('bellman target', lambda: bellman.bellman_target(f.target, tx, Rx, ty, Ry, U, 0.05, 1., 1.))
timeit('goal', lambda: f._goal_loss())
timeit('anchor fwd', lambda: f._anchor_loss())
def full():
    loss, logs = f.loss(0); opt.zero_grad(); loss.backward(); opt.step(); f.update_target()
timeit('FULL step', full)
