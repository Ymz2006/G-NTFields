"""Quick sanity check of every head: symmetry, diagonal value, finite output."""
import sys
sys.path.append('.')
from models.graph_method import model_network_graph as nets
from models.graph_method.geometry import se3
from models.graph_method.data import poses

tx, Rx, ty, Ry, _ = poses.PoseSampler(seed=1).training_pairs(64)
gx, gy = se3.features(tx, Rx), se3.features(ty, Ry)
for arch in nets.ARCHITECTURES:
    net = nets.build_network(arch, fourier_sigma=0.3, fourier_sigma_t=0.5, fourier_sigma_r=0.1)
    print('%-13s sym %.2e  diag %.2e  T mean %.3f' % (arch, (net.value(gx, gy) - net.value(gy, gx)).abs().max().item(),
                                                    net.value(gx, gx).abs().max().item(), net.value(gx, gy).mean().item()))
