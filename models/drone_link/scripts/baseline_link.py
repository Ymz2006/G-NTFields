"""Link-unaware baselines on a stretchable-link test set: how much signal contact do the drones keep?

For every case (start and goal poses of the link, i.e. both drones' start and goal positions) three motions are
scored at --samples poses uniform in time:

    straight     the direct 7-d motion (drone a linear, R slerp the short way, L linear): the link-aware
                 interpolation that ignores the scene
    independent  each drone flies its own shortest collision-free path, the link ignored: a point drone with
                 clearance >= --drone-r on a 26-connected grid of spacing --grid (GPU wavefront from the goal cell,
                 the steepest-descent cell path from the start cell, the straight segment when it is clear), then
                 string-pulled with the exact point clearance; both drones at constant speed, leaving and arriving
                 together
    certificate  the test set's own solution (RRT or PRM, link-aware): its drone path lengths

At each pose the link is the box between the drones (geometry.scene_clearance with L = |b - a| and R the shortest
rotation taking e_z to (b - a) / L). Contact = the box is clear of the scene (line of sight) and L <= L_max (in radio
range); the planner's constraint adds L >= L_min (the drones keep their separation). Per case: the fractions of time
with contact, with the line of sight blocked, too far, too close; whether the whole motion keeps contact; the
drones' travel lengths.

    python models/drone_link/scripts/baseline_link.py --dataPath testing_data/3dshape/link_forest_std \
        --out results/drone_link/baselines/link_forest_std.json --save-paths
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from models.drone_link import geometry as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--dataPath', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--grid', type=float, default=0.01)
ap.add_argument('--drone-r', type=float, default=0.015, help='clearance of a drone (>= the half diagonal of the '
                                                                'link cross-section, w / sqrt 2)')
ap.add_argument('--samples', type=int, default=400)
ap.add_argument('--cases', type=int, default=0)
ap.add_argument('--batch', type=int, default=64, help='drone queries per wavefront batch')
ap.add_argument('--chunk', type=int, default=2048, help='poses per exact-clearance batch (GPU memory)')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--save-paths', action='store_true')
args = ap.parse_args()

TWO_PI = 2 * math.pi
dev = args.device
G.CHUNK = args.chunk
meta = json.load(open(os.path.join(args.dataPath, 'meta.json')))
scene = G.load_scene(meta, REPO)
link = G.Link(meta['link']['width'], meta['link']['L_min'], meta['link']['L_max'])
arr = np.load(os.path.join(args.dataPath, 'sampled_points.npy')).astype(np.float64)
n = len(arr) if args.cases <= 0 else min(args.cases, len(arr))
cz = np.load(os.path.join(args.dataPath, 'cases.npz'))
cert_off = np.concatenate([[0], np.cumsum(cz['rrt_len'])])
R_D = args.drone_r


def pclear(P):
    return scene.point_clearance(P)


def seg_free(p, Q, r, res=0.004):
    """(m,) bool: point clearance >= r along the segments p (3,) -> Q (m, 3)."""
    k = max(2, int(math.ceil(float((Q - p).norm(dim=1).max()) / res)) + 1)
    u = torch.linspace(0, 1, k, device=dev, dtype=Q.dtype)
    S = p[None, None, :] + u[None, :, None] * (Q - p)[:, None, :]
    return (pclear(S.reshape(-1, 3)).reshape(len(Q), k) >= r).all(1)


# ----------------------------------------------------------------------------- grid
lo = torch.as_tensor(scene.room_lo, dtype=torch.float32, device=dev)
hi = torch.as_tensor(scene.room_hi, dtype=torch.float32, device=dev)
h = args.grid
dims = [int(x) for x in np.floor((scene.room_hi - scene.room_lo) / h)]
axes = [lo[k] + h * (0.5 + torch.arange(dims[k], device=dev, dtype=torch.float32)) for k in range(3)]
CEN = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
FREE = (pclear(CEN) >= R_D).reshape(dims)
FREE_IDX = torch.nonzero(FREE.reshape(-1)).squeeze(1)
OFFS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
OFF_T = torch.tensor(OFFS, device=dev)
OFF_W = torch.tensor([h * math.sqrt(i * i + j * j + k * k) for i, j, k in OFFS], device=dev)
INF = 1e9


def nearest_free(P):
    """Flat index of the nearest free cell centre for each point (m, 3)."""
    out = []
    for i in range(0, len(P), 64):
        d = torch.cdist(P[i:i + 64], CEN[FREE_IDX])
        out.append(FREE_IDX[d.argmin(1)])
    return torch.cat(out)


def wavefront(goal):
    """Grid distance fields (B, nx, ny, nz) to the goal cells (B,) (flat indices) through the free cells."""
    B = goal.shape[0]
    D = torch.full((B, *dims), INF, device=dev)
    D.view(B, -1)[torch.arange(B, device=dev), goal] = 0.0
    nx, ny, nz = dims
    for it in range(4000):
        Dp = F.pad(D, (1, 1, 1, 1, 1, 1), value=INF)
        best = D.clone()
        for (dx, dy, dz), w in zip(OFFS, OFF_W):
            torch.minimum(best, Dp[:, 1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny, 1 + dz:1 + dz + nz] + w, out=best)
        best = torch.where(FREE[None], best, torch.full_like(best, INF))
        if it % 16 == 15:
            if not bool((best < D - 1e-7).any()):
                return best
        D = best
    return D


def descend(D, start):
    """Steepest-descent cell paths (lists of flat indices) from the start cells (B,) to D's zero cells."""
    B = D.shape[0]
    Df = D.view(B, -1)
    nx, ny, nz = dims
    cur = start.clone()
    paths = [[int(c)] for c in start]
    live = Df[torch.arange(B, device=dev), cur] > 0
    for _ in range(20000):
        if not bool(live.any()):
            break
        ix, iy, iz = cur // (ny * nz), (cur // nz) % ny, cur % nz
        jx, jy, jz = ix[:, None] + OFF_T[:, 0], iy[:, None] + OFF_T[:, 1], iz[:, None] + OFF_T[:, 2]
        inb = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny) & (jz >= 0) & (jz < nz)
        nb = (jx.clamp(0, nx - 1) * ny + jy.clamp(0, ny - 1)) * nz + jz.clamp(0, nz - 1)
        dn = torch.where(inb, torch.gather(Df, 1, nb), torch.full_like(OFF_W.expand(B, -1), INF))
        a = (dn + OFF_W).argmin(1)                          # the Bellman minimizer: D[c] = D[nb] + w
        here = Df[torch.arange(B, device=dev), cur]
        step = live & (torch.gather(dn, 1, a[:, None]).squeeze(1) < here - 1e-9)
        nxt = torch.where(step, torch.gather(nb, 1, a[:, None]).squeeze(1), cur)
        for b in torch.nonzero(step).squeeze(1).tolist():
            paths[b].append(int(nxt[b]))
        cur = nxt
        live = step & (Df[torch.arange(B, device=dev), cur] > 0)
    return paths


def string_pull(P):
    """Greedy shortcut of a polyline (k, 3) with point clearance >= drone radius along each kept segment."""
    out, i = [P[0]], 0
    while i < len(P) - 1:
        ok = seg_free(P[i], P[i + 1:], R_D)
        j = i + 1 + (int(torch.nonzero(ok).max()) if bool(ok.any()) else 0)
        out.append(P[j])
        i = j
    return torch.stack(out)


def shortest_paths(P0, P1):
    """Shortest point-drone paths P0 -> P1 (m, 3) each: list of (k, 3) tensors (None if unreachable)."""
    m = len(P0)
    res = [None] * m
    direct = torch.stack([seg_free(P0[i], P1[i:i + 1], R_D)[0] for i in range(m)])
    for i in torch.nonzero(direct).squeeze(1).tolist():
        res[i] = torch.stack([P0[i], P1[i]])
    todo = torch.nonzero(~direct).squeeze(1)
    for s in range(0, len(todo), args.batch):
        idx = todo[s:s + args.batch]
        g, st = nearest_free(P1[idx]), nearest_free(P0[idx])
        D = wavefront(g)
        reach = D.view(len(idx), -1)[torch.arange(len(idx), device=dev), st] < INF / 2
        cells = descend(D, st)
        for b, i in enumerate(idx.tolist()):
            if not bool(reach[b]):
                continue
            Pc = torch.cat([P0[i:i + 1], CEN[torch.as_tensor(cells[b], device=dev)], P1[i:i + 1]])
            res[i] = string_pull(Pc)
        del D
    return res


# ----------------------------------------------------------------------------- scoring
def resample(P, k):
    seg = (P[1:] - P[:-1]).norm(dim=1)
    s = torch.cat([torch.zeros(1, device=dev, dtype=P.dtype), torch.cumsum(seg, 0)])
    if float(s[-1]) < 1e-12:
        return P[:1].expand(k, 3)
    u = torch.linspace(0, float(s[-1]), k, device=dev, dtype=P.dtype)
    j = torch.searchsorted(s, u, right=True).clamp(1, len(s) - 1)
    w = ((u - s[j - 1]) / (s[j] - s[j - 1]).clamp_min(1e-12))[:, None]
    return P[j - 1] * (1 - w) + P[j] * w


def rot_to(d):
    """Rotations (k, 3, 3) taking e_z to the unit vectors d (k, 3) (the shortest rotation)."""
    ez = torch.zeros_like(d)
    ez[:, 2] = 1.0
    v = torch.cross(ez, d, dim=1)
    s = v.norm(dim=1)
    rv = v / s.clamp_min(1e-12)[:, None] * torch.atan2(s, d[:, 2])[:, None]
    flip = (s < 1e-9) & (d[:, 2] < 0)
    rv[flip] = torch.tensor([math.pi, 0.0, 0.0], device=dev, dtype=d.dtype)
    return G.so3_exp(rv)


def score_pairs(PA, PB):
    """Link status along synchronized drone positions PA, PB (m, k, 3): per-case dicts."""
    m, k, _ = PA.shape
    D = (PB - PA).reshape(-1, 3)
    L = D.norm(dim=1)
    R = rot_to(D / L.clamp_min(1e-12)[:, None])
    cs = G.scene_clearance(scene, link, PA.reshape(-1, 3), R, L.clamp_min(1e-6)).reshape(m, k)
    L = L.reshape(m, k)
    los, far, close = cs >= 0, L > link.L_max, L < link.L_min
    contact = los & ~far
    valid = contact & ~close
    out = []
    for i in range(m):
        out.append(dict(contact_frac=float(contact[i].float().mean()), full_contact=bool(contact[i].all()),
                        valid_frac=float(valid[i].float().mean()), valid=bool(valid[i].all()),
                        los_blocked_frac=float((~los[i]).float().mean()), too_far_frac=float(far[i].float().mean()),
                        too_close_frac=float(close[i].float().mean()), min_L=float(L[i].min()), max_L=float(L[i].max())))
    return out


def chart_ends(X):
    t, R, L = G.from_chart(X)
    return link.ends(t, R, L)


def straight_all(X0, X1, k):
    q0 = torch.as_tensor(G.quat_from_rotvec(X0[:, 3:6].cpu().numpy() * TWO_PI), dtype=torch.float32, device=dev)
    q1 = torch.as_tensor(G.quat_from_rotvec(X1[:, 3:6].cpu().numpy() * TWO_PI), dtype=torch.float32, device=dev)
    d = (q0 * q1).sum(1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    th = torch.arccos(d.abs().clamp(max=1.0))
    u = torch.linspace(0, 1, k, device=dev)[None, :, None]
    s = torch.sin(th)[:, None, :]
    th3 = th[:, None, :]
    w0 = torch.where(s < 1e-6, 1 - u, torch.sin((1 - u) * th3) / s.clamp_min(1e-12))
    w1 = torch.where(s < 1e-6, u, torch.sin(u * th3) / s.clamp_min(1e-12))
    q = w0 * q0[:, None] + w1 * q1[:, None]
    q = (q / q.norm(dim=-1, keepdim=True)).reshape(-1, 4)
    t = (X0[:, None, :3] + u * (X1[:, None, :3] - X0[:, None, :3])).reshape(-1, 3)
    L = (X0[:, None, 6:7] + u * (X1[:, None, 6:7] - X0[:, None, 6:7])).reshape(-1)
    R = torch.as_tensor(G.quat_matrix(q.cpu().numpy()), dtype=torch.float32, device=dev)
    c, cs, _ = G.clearance(scene, link, t, R, L, parts=True)
    c, cs = c.reshape(-1, k), cs.reshape(-1, k)
    pa = t.reshape(-1, k, 3)
    pb = (t + R[:, :, 2] * L[:, None]).reshape(-1, k, 3)
    la = (pa[:, 1:] - pa[:, :-1]).norm(dim=2).sum(1)
    lb = (pb[:, 1:] - pb[:, :-1]).norm(dim=2).sum(1)
    return [dict(success=bool((c[i] >= 0).all()), contact_frac=float((cs[i] >= 0).float().mean()),
                 full_contact=bool((cs[i] >= 0).all()), len_a=float(la[i]), len_b=float(lb[i])) for i in range(len(c))]


def cert_case(i):
    W = cz['rrt_flat'][cert_off[i]:cert_off[i + 1]]
    R = G.quat_matrix(W[:, 3:7])
    pa, pb = W[:, :3], W[:, :3] + R[:, :, 2] * W[:, 7:8]
    return dict(len_a=float(np.linalg.norm(np.diff(pa, axis=0), axis=1).sum()),
                len_b=float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum()))


if __name__ == '__main__':
    t0 = time.time()
    with torch.no_grad():
        print('grid {} cells ({} free) ({:.0f}s)'.format(int(np.prod(dims)), int(FREE.sum()), time.time() - t0), flush=True)
        X = torch.as_tensor(arr[:n], dtype=torch.float32, device=dev)
        st = []
        for s in range(0, n, 250):
            st += straight_all(X[s:s + 250, :7], X[s:s + 250, 7:], args.samples)
        A0, B0 = chart_ends(X[:, :7])
        A1, B1 = chart_ends(X[:, 7:])
        PA = shortest_paths(A0, A1)
        print('drone a paths ({:.0f}s)'.format(time.time() - t0), flush=True)
        PB = shortest_paths(B0, B1)
        print('drone b paths ({:.0f}s)'.format(time.time() - t0), flush=True)
        ok = [i for i in range(n) if PA[i] is not None and PB[i] is not None]
        ind = {}
        for s in range(0, len(ok), 250):
            ids = ok[s:s + 250]
            ra = torch.stack([resample(PA[i], args.samples) for i in ids])
            rb = torch.stack([resample(PB[i], args.samples) for i in ids])
            sc = score_pairs(ra, rb)
            dmin = torch.minimum(pclear(ra.reshape(-1, 3)), pclear(rb.reshape(-1, 3))).reshape(len(ids), -1).min(1).values
            for j, i in enumerate(ids):
                sc[j].update(solved=True, len_a=float((PA[i][1:] - PA[i][:-1]).norm(dim=1).sum()),
                             len_b=float((PB[i][1:] - PB[i][:-1]).norm(dim=1).sum()), drone_min_clear=float(dmin[j]),
                             waypoints_a=int(len(PA[i])), waypoints_b=int(len(PB[i])))
                ind[i] = sc[j]
    recs = [{'case': i, 'straight': st[i], 'rrt': cert_case(i), 'independent': ind.get(i, dict(solved=False))}
            for i in range(n)]

    def mean(method, key, cond=None):
        v = [r[method][key] for r in recs if key in r[method] and (cond is None or cond(r))]
        return float(np.mean(v)) if v else None

    ind_ok = lambda r: r['independent'].get('solved', False)  # noqa: E731
    summary = {
        'dataPath': args.dataPath, 'n_cases': n, 'samples': args.samples, 'grid': args.grid, 'drone_r': args.drone_r,
        'link': meta['link'], 'certificate': meta.get('certificate', 'rrt'),
        'straight': {'success_rate': mean('straight', 'success'), 'mean_contact_frac': mean('straight', 'contact_frac'),
                     'full_contact_rate': mean('straight', 'full_contact')},
        'independent': {'solved_rate': float(np.mean([ind_ok(r) for r in recs])),
                        'mean_contact_frac': mean('independent', 'contact_frac', ind_ok),
                        'full_contact_rate': mean('independent', 'full_contact', ind_ok),
                        'mean_valid_frac': mean('independent', 'valid_frac', ind_ok),
                        'valid_rate': mean('independent', 'valid', ind_ok),
                        'mean_los_blocked_frac': mean('independent', 'los_blocked_frac', ind_ok),
                        'mean_too_far_frac': mean('independent', 'too_far_frac', ind_ok),
                        'mean_too_close_frac': mean('independent', 'too_close_frac', ind_ok),
                        'rate_los_ever_blocked': float(np.mean([r['independent']['los_blocked_frac'] > 0
                                                                for r in recs if ind_ok(r)])),
                        'rate_ever_too_far': float(np.mean([r['independent']['too_far_frac'] > 0
                                                            for r in recs if ind_ok(r)])),
                        'rate_ever_too_close': float(np.mean([r['independent']['too_close_frac'] > 0
                                                              for r in recs if ind_ok(r)])),
                        'median_len_a': float(np.median([r['independent']['len_a'] for r in recs if ind_ok(r)])),
                        'median_len_b': float(np.median([r['independent']['len_b'] for r in recs if ind_ok(r)]))},
        'rrt': {'median_len_a': float(np.median([r['rrt']['len_a'] for r in recs])),
                'median_len_b': float(np.median([r['rrt']['len_b'] for r in recs]))},
        'seconds': time.time() - t0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.tmp', 'w') as fh:
        json.dump({'summary': summary, 'records': recs}, fh, indent=1)
    os.replace(args.out + '.tmp', args.out)
    if args.save_paths:
        keep = sorted(ind)
        np.savez_compressed(args.out[:-5] + '_independent.npz', case=np.array(keep),
                            a=np.array([PA[i].cpu().numpy() for i in keep], dtype=object),
                            b=np.array([PB[i].cpu().numpy() for i in keep], dtype=object))
    print(json.dumps(summary))
