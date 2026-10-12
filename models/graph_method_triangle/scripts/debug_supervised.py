"""Debug: how well does each head fit the analytic latents with oracle labels?"""
import sys, time
sys.path.append('.')
import torch
from models.graph_method_triangle import model_function_graph as mf

def run(name, steps=1500, **kw):
    cfg = mf.default_config(mode='supervised', device='cuda:0', steps=steps, **kw)
    f = mf.Function(cfg)
    opt = torch.optim.Adam(f.network.parameters(), lr=cfg['lr'])
    t0 = time.time()
    for s in range(1, steps + 1):
        loss, logs = f.loss(s)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(f.network.parameters(), cfg['clip'])
        opt.step()
        if s % 500 == 0:
            print(name, s, {k: round(v, 5) for k, v in logs.items()}, '%.1fs' % (time.time() - t0), flush=True)

which = sys.argv[1]
if which == 'a':
    run('gram_default')
    run('gram_lr3e-4', lr=3e-4)
    run('gram_skip', skip_input=True)
    run('gram_sigma0.2', fourier_sigma=0.2)
elif which == 'b':
    run('gram_norm', normalize=True)
    run('sqdist', arch='graph_sqdist')
    run('joint', arch='graph_joint')
    run('scalar', arch='scalar')
    run('metric', arch='metric')
