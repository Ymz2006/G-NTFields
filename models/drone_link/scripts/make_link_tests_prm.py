"""Test sets for a stretchable-link cell, every case certified by a probabilistic roadmap (PRM) built on the GPU.

The cases are those of make_link_tests.py (models/drone_link/cases.py: case k of a set is drawn from the same random
stream whatever the certificate, so the endpoints of case k are the same). Only the certificate differs: on a machine
whose CPUs are shared with many other jobs the RRT-Connect (one CPU process per case) crawled, so one roadmap per
scene answers all queries:

    nodes   --nodes poses (t uniform in the room, R uniform on SO(3), L uniform) with exact clearance >= --margin
    edges   each node to its --k nearest nodes in the 13-d embedding (t, L, vec(R) / (2 pi sqrt 2)), whose distance
            is the pose metric sqrt(|dt|^2 + (theta / 2 pi)^2 + dL^2) to second order in theta; kept if the motion
            (t linear, R slerp the short way, L linear) has exact clearance >= --margin at resolution --res in the pose
            metric (the RRT's motion check)
    query   the direct motion if it is valid, else start and goal joined to their --kq nearest nodes by valid motions
            and Dijkstra on the pose metric
    path    greedy shortcut: from each kept waypoint to the farthest later one it reaches by a valid motion

Outputs as make_link_tests.py; the certificate path is stored where the RRT's is (cases.npz rrt_flat / rrt_len,
verification.json rrt_len, rrt_waypoints) and meta.json / verification.json record 'certificate': 'prm'.

    python models/drone_link/scripts/make_link_tests_prm.py --cell datasets/3dshape/data/link_forest --variant std
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.drone_link import geometry as G  # noqa: E402
from models.drone_link.cases import REGIONS, draw_case  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cell', required=True)
ap.add_argument('--variant', required=True, choices=['std', 'cross'])
ap.add_argument('--n', type=int, default=1000)
ap.add_argument('--out', default='testing_data/3dshape')
ap.add_argument('--name', default='')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--offset', type=float, default=0.02)
ap.add_argument('--res', type=float, default=0.003)
ap.add_argument('--margin', type=float, default=0.002)
ap.add_argument('--nodes', type=int, default=60000)
ap.add_argument('--k', type=int, default=12)
ap.add_argument('--kq', type=int, default=40)
ap.add_argument('--max-edge', type=float, default=0.35)
ap.add_argument('--workers', type=int, default=6, help='CPU processes drawing the endpoints')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--chunk', type=int, default=2048, help='poses per exact-clearance batch (GPU memory)')
args = ap.parse_args()

TWO_PI = 2 * math.pi
dev = args.device
G.CHUNK = args.chunk
meta = json.load(open(os.path.join(args.cell, 'meta.json')))
scene = G.load_scene(meta, REPO)
link = G.load_link(meta)
SC = meta['link']['scene']
name = args.name or '{}_{}'.format(os.path.basename(args.cell.rstrip('/')), args.variant)


# ----------------------------------------------------------------------------- torch pose helpers (nodes (t, q, L))
def quat_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def slerp(q0, q1, u):
    """Shortest-arc slerp of unit quaternions (N, 4) at fractions u (N,)."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    th = torch.arccos(d)
    s = torch.sin(th)
    u = u[:, None]
    small = s < 1e-6
    w0 = torch.where(small, 1 - u, torch.sin((1 - u) * th) / s.clamp_min(1e-12))
    w1 = torch.where(small, u, torch.sin(u * th) / s.clamp_min(1e-12))
    q = w0 * q0 + w1 * q1
    return q / q.norm(dim=-1, keepdim=True)


def clearance(X):
    return G.clearance(scene, link, X[:, :3], quat_mat(X[:, 3:7]), X[:, 7])


def pose_dist(A, B):
    th = 2 * torch.arccos((A[:, 3:7] * B[:, 3:7]).sum(-1).abs().clamp(max=1.0))
    return torch.sqrt(((A[:, :3] - B[:, :3]) ** 2).sum(-1) + (th / TWO_PI) ** 2 + (A[:, 7] - B[:, 7]) ** 2)


def embed(X):
    return torch.cat([X[:, :3], X[:, 7:8], quat_mat(X[:, 3:7]).reshape(-1, 9) / (TWO_PI * math.sqrt(2.0))], 1)


def motions_ok(A, B, margin, max_pts=50000):
    """(E,) bool: the motions A -> B (E, 8) keep exact clearance >= margin at resolution args.res (u in (0, 1])."""
    E = A.shape[0]
    ok = torch.ones(E, dtype=torch.bool, device=dev)
    if E == 0:
        return ok
    K = torch.ceil(pose_dist(A, B) / args.res).clamp(min=1).long()
    csum = torch.cumsum(K, 0)
    s = 0
    while s < E:
        base = int(csum[s - 1]) if s > 0 else 0
        e = max(s + 1, int(torch.searchsorted(csum, torch.tensor(base + max_pts, device=dev), right=True)))
        Kc = K[s:e]
        n = int(Kc.sum())
        loc = torch.repeat_interleave(torch.arange(e - s, device=dev), Kc)
        first = torch.cumsum(Kc, 0) - Kc
        j = torch.arange(n, device=dev) - first[loc] + 1
        u = j.float() / Kc[loc].float()
        a, b = A[s:e][loc], B[s:e][loc]
        X = torch.cat([a[:, :3] + u[:, None] * (b[:, :3] - a[:, :3]), slerp(a[:, 3:7], b[:, 3:7], u),
                       (a[:, 7] + u * (b[:, 7] - a[:, 7]))[:, None]], 1)
        bad = (clearance(X) < margin).float()
        ok[s:e] = torch.zeros(e - s, device=dev).index_add_(0, loc, bad) == 0
        s = e
    return ok


def haar(n):
    u1, u2, u3 = torch.rand(n, device=dev), torch.rand(n, device=dev), torch.rand(n, device=dev)
    q = torch.stack([torch.sqrt(u1) * torch.cos(TWO_PI * u3), torch.sqrt(1 - u1) * torch.sin(TWO_PI * u2),
                     torch.sqrt(1 - u1) * torch.cos(TWO_PI * u2), torch.sqrt(u1) * torch.sin(TWO_PI * u3)], 1)
    return q * torch.where(q[:, :1] < 0, -1.0, 1.0)


def knn(Q, P, k, chunk=1024, exclude_self=False):
    """Indices (len(Q), k) of the k nearest rows of P (embedding distance)."""
    eq, ep = embed(Q), embed(P)
    out = []
    for i in range(0, len(eq), chunk):
        d = torch.cdist(eq[i:i + chunk], ep)
        if exclude_self:
            d[torch.arange(d.shape[0], device=dev), torch.arange(i, i + d.shape[0], device=dev)] = math.inf
        out.append(d.topk(k, dim=1, largest=False).indices)
    return torch.cat(out)


# ----------------------------------------------------------------------------- endpoints (CPU, as make_link_tests.py)
def endpoints(k):
    torch.set_num_threads(1)
    a, c0, b, c1, _ = draw_case(scene, link, SC, args.variant, k, args.seed, args.offset)
    return k, a, c0, b, c1


if __name__ == '__main__':
    t_start = time.time()
    torch.manual_seed(args.seed)
    n_try = int(args.n * 1.1) + 4
    with mp.get_context('fork').Pool(args.workers) as pool:
        draws = sorted(pool.map(endpoints, range(n_try), chunksize=8))
    print('{} cases drawn ({:.0f}s)'.format(len(draws), time.time() - t_start), flush=True)

    # ------------------------------------------------------------------ roadmap
    lo = torch.as_tensor(scene.room_lo, dtype=torch.float32, device=dev)
    hi = torch.as_tensor(scene.room_hi, dtype=torch.float32, device=dev)
    nodes, cnt = [], 0
    while cnt < args.nodes:
        B = 16384
        X = torch.cat([lo + (hi - lo) * torch.rand(B, 3, device=dev), haar(B),
                       (link.L_min + (link.L_max - link.L_min) * torch.rand(B, device=dev))[:, None]], 1)
        X = X[clearance(X) >= args.margin]
        nodes.append(X)
        cnt += len(X)
    V = torch.cat(nodes)[:args.nodes]
    nb = knn(V, V, args.k, exclude_self=True)
    I = torch.arange(len(V), device=dev)[:, None].expand_as(nb).reshape(-1)
    J = nb.reshape(-1)
    pair = torch.unique(torch.stack([torch.minimum(I, J), torch.maximum(I, J)], 1), dim=0)
    w = pose_dist(V[pair[:, 0]], V[pair[:, 1]])
    pair, w = pair[w <= args.max_edge], w[w <= args.max_edge]
    t1 = time.time()
    ok = torch.cat([motions_ok(V[pair[i:i + 50000, 0]], V[pair[i:i + 50000, 1]], args.margin)
                    for i in range(0, len(pair), 50000)])
    pair, w = pair[ok].cpu().numpy(), w[ok].cpu().numpy()
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components, dijkstra
    N = len(V)
    Gr = sp.coo_matrix((w, (pair[:, 0], pair[:, 1])), shape=(N, N)).tocsr()
    ncomp, lab = connected_components(Gr, directed=False)
    big = np.bincount(lab).max()
    print('roadmap: {} nodes, {} candidate edges, {} valid ({:.0f}s for the motions), {} components, largest {}'.format(
        N, int(ok.numel()), len(pair), time.time() - t1, ncomp, big), flush=True)

    # ------------------------------------------------------------------ queries
    S = torch.as_tensor(np.stack([d[1] for d in draws]), dtype=torch.float32, device=dev)
    Gl = torch.as_tensor(np.stack([d[3] for d in draws]), dtype=torch.float32, device=dev)
    direct0 = motions_ok(S, Gl, 0.0).cpu().numpy()
    direct_m = motions_ok(S, Gl, args.margin).cpu().numpy()
    links = []
    for X in (S, Gl):
        nn_ = knn(X, V, args.kq)
        A = X[:, None, :].expand(-1, args.kq, -1).reshape(-1, 8)
        Bn = V[nn_.reshape(-1)]
        okq = motions_ok(A, Bn, args.margin).reshape(-1, args.kq).cpu().numpy()
        dq = pose_dist(A, Bn).reshape(-1, args.kq).cpu().numpy()
        links.append((nn_.cpu().numpy(), okq, dq))
    Vn = V.cpu().numpy()
    paths, solved = [], []
    for ci in range(len(draws)):
        if direct_m[ci]:
            paths.append(np.stack([draws[ci][1], draws[ci][3]]))
            solved.append(True)
            continue
        rows, cols, ws = [], [], []
        for side, node in ((0, N), (1, N + 1)):
            nn_, okq, dq = links[side]
            sel = okq[ci]
            rows += [node] * int(sel.sum())
            cols += list(nn_[ci][sel])
            ws += list(dq[ci][sel])
        if not rows or N not in rows or N + 1 not in rows:
            paths.append(None)
            solved.append(False)
            continue
        Ga = sp.bmat([[Gr, sp.coo_matrix((ws, (np.array(cols), np.array(rows) - N)), shape=(N, 2))],
                      [None, sp.coo_matrix((2, 2))]]).tocsr()
        dist_, pred = dijkstra(Ga, directed=False, indices=N, return_predecessors=True)
        if not np.isfinite(dist_[N + 1]):
            paths.append(None)
            solved.append(False)
            continue
        chain = [N + 1]
        while chain[-1] != N:
            chain.append(pred[chain[-1]])
        chain = chain[::-1]
        P = np.stack([draws[ci][1]] + [Vn[i] for i in chain[1:-1]] + [draws[ci][3]])
        paths.append(P)
        solved.append(True)
    print('queries: {} / {} solved ({:.0f}s)'.format(sum(solved), len(draws), time.time() - t_start), flush=True)

    def plen(P):
        P = torch.as_tensor(P, dtype=torch.float32)
        return float(pose_dist(P[:-1], P[1:]).sum()) if len(P) > 1 else 0.0

    def shortcut(P):
        Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
        out, i = [P[0]], 0
        while i < len(P) - 1:
            later = Pt[i + 1:]
            okl = motions_ok(Pt[i:i + 1].expand(len(later), -1), later, args.margin).cpu().numpy()
            j = i + 1 + (int(np.nonzero(okl)[0].max()) if okl.any() else 0)
            out.append(P[j])
            i = j
        return np.stack(out)

    recs, starts, goals, cert = [], [], [], []
    for ci, (k, a, c0, b, c1) in enumerate(draws):
        rec = {'k': k, 'clear0': c0, 'clear1': c1, 'solved': bool(solved[ci]),
               'dist_t': float(np.linalg.norm(b[:3] - a[:3])), 'angle_deg': float(np.degrees(G.quat_angle(a[3:7], b[3:7]))),
               'L0': float(a[7]), 'L1': float(b[7]), 'direct_ok': bool(direct0[ci]), 'certificate': 'prm'}
        if solved[ci] and len(recs) < args.n:
            P = paths[ci]
            rec['rrt_len_raw'] = plen(P)
            P = shortcut(P)
            rec['rrt_len'] = plen(P)
            rec['rrt_waypoints'] = int(len(P))
            recs.append(rec)
            starts.append(a)
            goals.append(b)
            cert.append(P)
    if len(recs) < args.n:
        print('only {} solved cases of {} drawn; rerun with more nodes or a larger --n budget'.format(len(recs), len(draws)))
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    Sx, Gx = np.asarray(starts), np.asarray(goals)
    x0 = G.to_chart(Sx[:, :3], G.rotvec_from_quat(Sx[:, 3:7]), Sx[:, 7])
    x1 = G.to_chart(Gx[:, :3], G.rotvec_from_quat(Gx[:, 3:7]), Gx[:, 7])
    np.save(os.path.join(out, 'sampled_points.npy'), np.concatenate([x0, x1], 1).astype(np.float32))
    np.savez(os.path.join(out, 'cases.npz'), start=Sx, goal=Gx, rrt_flat=np.concatenate(cert),
             rrt_len=np.array([len(p) for p in cert]))
    m = dict(meta)
    m['testing_data'] = True
    m['offset'] = args.offset
    m['variant'] = args.variant
    m['certificate'] = 'prm'
    m['test_settings'] = dict(vars(args))
    if args.variant == 'cross':
        m['regions_raw'] = REGIONS[SC]
    json.dump(m, open(os.path.join(out, 'meta.json'), 'w'), indent=1)
    tried = int(np.nonzero([d[0] == recs[-1]['k'] for d in draws])[0][0]) + 1
    summ = {'n': len(recs), 'tried': tried, 'unsolved': int(sum(not s for s in solved[:tried])),
            'certificate': 'prm', 'roadmap_nodes': N, 'roadmap_edges': int(len(pair)), 'roadmap_components': int(ncomp),
            'seconds': time.time() - t_start, 'direct_ok': float(np.mean([r['direct_ok'] for r in recs])),
            'rrt_len_median': float(np.median([r['rrt_len'] for r in recs])),
            'rrt_waypoints_median': float(np.median([r['rrt_waypoints'] for r in recs])),
            'dist_t_median': float(np.median([r['dist_t'] for r in recs])),
            'angle_deg_median': float(np.median([r['angle_deg'] for r in recs]))}
    json.dump({'summary': summ, 'cases': recs}, open(os.path.join(out, 'verification.json'), 'w'), indent=1)
    print(json.dumps(summ))
