import sys, time
sys.path.append('.')
import torch
from models.graph_method import model_function_graph as mf
for arch in ('graph_gram', 'metric'):
    for bs in (256, 1024):
        cfg = mf.default_config(arch=arch, mode='physics', device='cuda:0', fourier_sigma=0.03, batch_size=bs)
        f = mf.Function(cfg)
        opt = torch.optim.Adam(f.network.parameters(), 1e-3)
        def full():
            loss, logs = f.loss(0); opt.zero_grad(); loss.backward()
            g = torch.nn.utils.clip_grad_norm_(f.network.parameters(), 1.0); bool(torch.isfinite(g))
            opt.step(); f.update_target()
        for _ in range(5): full()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(30): full()
        torch.cuda.synchronize(); print(arch, bs, '%.1f ms/step' % ((time.time() - t) / 30 * 1e3), flush=True)
