"""preprocess_obj.py --2d for the moving-sofa halls, with grid-guided proposals.

Same command line and outputs as preprocess_obj_fast.py (which it builds on), and
the same distribution of pairs, but feasible in a corridor: there only 0.06 - 1.3 %
of uniform (x, y, rz) placements are free with enough clearance, and the original
sampler needs two of them per pair (test: both endpoints; training: x0 in the band,
then x1 = x0 + a long random displacement), so it makes a few pairs an hour.

Here placements are proposed uniformly over the cells of check_sofa2d.py's
(x, y, theta) grid (0.5 units, 1 degree) that can hold a valid placement, and every
pair is still accepted by the original exact tests on the original arrays:

    testing    x0, x1 ~ U(S(offset)) independently; keep iff both free, clearance > offset
    training   x0 ~ U(S(offset) minus cells beyond margin), delta as the original,
               x1 = x0 + delta; drop x1 outside the bbox or outside S(offset);
               keep iff x0 free with offset < clearance < margin, x1 free with
               clearance > offset (exact band mask of preprocess_obj_fast.py, then
               the original evaluate_placements, whose values are stored)

S(c) = cells free of the walls grown by c - DELTA.  The grid's own error (cell
half-diagonal, half a degree at the sofa's radius, the wall and footprint rasters)
is below 2 units, but the exact test sees the walls only through their 10k surface
points (about 1.4 units apart), so a valid placement can reach a little into a wall
between points: the selftest finds valid placements up to 3 units outside the grid
set.  With DELTA = 5 every valid placement lies in S, and since
the proposal is uniform on S and the rejection tests are the original ones, the
accepted pairs have the original density (uniform proposal x indicator of the
original acceptance).  ``--selftest`` checks the cover: it draws placements from a
wider grid shell around S, evaluates them exactly and reports any valid one
outside S (must be none), with the margin actually needed.  Pairs are i.i.d., so
only the torch random stream differs from the original, not the distribution.

    python dataprocessing/preprocess_obj_2dgrid.py <preprocess_obj.py flags incl. --2d>
    python dataprocessing/preprocess_obj_2dgrid.py --selftest 200000 --env ... --shape ...
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess_obj_fast as pf  # noqa: E402  (loads preprocess_obj.py with the exact x0 band mask)
import check_sofa2d as cs  # noqa: E402
import make_sofa2d as mk  # noqa: E402

po = pf.po
DELTA = 5.0              # env units
CANDIDATES = 400         # training candidate rows drawn per accepted-batch row


class Grid:
    """check_sofa2d's grid for one hall + sofa, in the normalized frame of preprocess_obj.py --2d."""

    def __init__(self, env_path, shape_path, device):
        env = os.path.splitext(os.path.basename(env_path))[0]
        shape = os.path.splitext(os.path.basename(shape_path))[0]
        assert env in mk.HALLS and shape in mk.SOFA, 'grid proposals need a make_sofa2d hall + sofa: {} {}'.format(env, shape)
        self.W, self.boxes = mk.HALLS[env], mk.sofa_boxes(**mk.SOFA[shape])
        V, _, _ = po.load_obj(env_path)
        V[:, 2] = 0.0                                            # as preprocess_obj.py --2d
        lo, hi = V.min(0), V.max(0)
        self.center, self.scale = 0.5 * (lo + hi), float((hi - lo).max())
        self.device = device
        self.x0 = float(cs.XS[0])
        self.n = len(cs.XS)

    def cells(self, clear):
        return torch.from_numpy(cs.free_grid(self.W, self.boxes, clear)).to(self.device)

    def propose(self, idx, n):
        """n placements uniform over the cells listed in idx (flat indices), normalized units, rotvec radians."""
        k = idx[torch.randint(len(idx), (n,), device=self.device)]
        th, rest = k // (self.n * self.n), k % (self.n * self.n)
        i, j = rest // self.n, rest % self.n
        u = torch.rand(n, 3, device=self.device)
        x = self.x0 + (i + u[:, 0] - 0.5) * cs.RES
        y = self.x0 + (j + u[:, 1] - 0.5) * cs.RES
        a = (th + u[:, 2] - 0.5) * (2 * np.pi / cs.NTH)
        c = torch.zeros(n, 6, device=self.device)
        c[:, 0] = (x - self.center[0]) / self.scale
        c[:, 1] = (y - self.center[1]) / self.scale
        c[:, 5] = torch.remainder(a + np.pi, 2 * np.pi) - np.pi
        return c

    def lookup(self, mask, cfg):
        """mask value at the cell of each placement (False outside the grid)."""
        x = (cfg[:, 0] * self.scale + self.center[0] - self.x0) / cs.RES
        y = (cfg[:, 1] * self.scale + self.center[1] - self.x0) / cs.RES
        i, j = torch.round(x).long(), torch.round(y).long()
        th = torch.round(torch.remainder(cfg[:, 5], 2 * np.pi) / (2 * np.pi) * cs.NTH).long() % cs.NTH
        ok = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        out = torch.zeros(len(cfg), dtype=torch.bool, device=cfg.device)
        out[ok] = mask[th[ok], i[ok], j[ok]]
        return out


GRID = None              # set in main from --env / --shape


def grid_generate_valid_pairs(number_pairs, tet_verts_local, face_verts_local, env_points, half_extent,
                              margin, offset, rad_points, rad_bins, batch_size=256, device='cuda',
                              testing=False, yrot=False, two_d=False, track_angle_pts=False):
    """preprocess_obj.generate_valid_pairs (same arguments and returns) with grid proposals."""
    assert two_d and not yrot, 'grid proposals are for --2d without --yrot'
    env_t = torch.tensor(env_points, dtype=torch.float32).to(device)
    tet_verts_local = tet_verts_local.to(device)
    face_verts_local = face_verts_local.to(device)
    rad_points = rad_points.to(device)
    rad_bins = rad_bins.to(device)
    hx, hy, hz = float(half_extent[0]), float(half_extent[1]), float(half_extent[2])
    s = GRID.scale
    lo = GRID.cells(offset * s - DELTA)                          # S(offset)
    if testing:
        prop = torch.nonzero(lo.reshape(-1)).squeeze(1)
    else:
        hi = GRID.cells(margin * s + DELTA)                      # certainly clearance >= margin
        prop = torch.nonzero((lo & ~hi).reshape(-1)).squeeze(1)
        del hi
    print('[2dgrid] proposal cells {} ({:.3%} of the grid), DELTA {} units'.format(
        len(prop), len(prop) / lo.numel(), DELTA), flush=True)

    out = {k: torch.zeros(number_pairs, w) for k, w in
           (('pairs', 12), ('dists', 2), ('angles', 2), ('normals', 12), ('trans', 12), ('rot', 12))}
    ang_shape = torch.zeros(number_pairs, 3) if track_angle_pts else None
    ang_env = torch.zeros(number_pairs, 3) if track_angle_pts else None
    count = 0
    two_pi = 2.0 * np.pi

    def evaluate(cfg, angle_pts=False):
        return po.evaluate_placements(cfg, tet_verts_local, face_verts_local, env_t, rad_points, rad_bins,
                                      device, return_angle_pts=angle_pts, two_d=True)

    def in_bbox(c):
        return (c[:, 0].abs() <= hx) & (c[:, 1].abs() <= hy) & (c[:, 2].abs() <= hz)

    def sample_disp(n):                                          # preprocess_obj's _sample_disp, two_d
        d = torch.rand(n, 6, device=device) - 0.5
        d[:, 2:5] = 0
        d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + po.EPS)
        return d * (torch.rand(n, 1, device=device) * float(np.sqrt(6.0)))

    while count < number_pairs:
        if testing:
            x0, x1 = GRID.propose(prop, batch_size), GRID.propose(prop, batch_size)
            ok = in_bbox(x0) & in_bbox(x1)
            x0, x1 = x0[ok], x1[ok]
        else:
            n = batch_size * CANDIDATES
            x0 = GRID.propose(prop, n)
            delta = sample_disp(n)
            x1 = x0.clone()
            x1[:, 0:3] = x0[:, 0:3] + delta[:, 0:3]
            x1[:, 3:6] = po.wrap_rotvec(x0[:, 3:6] + delta[:, 3:6] * two_pi)
            ok = in_bbox(x0) & in_bbox(x1)
            ok[ok.clone()] = GRID.lookup(lo, x1[ok])
            x0, x1 = x0[ok][:batch_size], x1[ok][:batch_size]
            if len(x0) == 0:
                continue
            band = pf.x0_band_mask(x0.cpu(), tet_verts_local, face_verts_local, env_t, margin, offset, device)
            x0, x1 = x0[band.to(device)], x1[band.to(device)]
        if len(x0) == 0:
            continue
        x0, x1 = x0.cpu(), x1.cpu()
        free1, dist1, angle1, normal1, trans1, rot1 = evaluate(x1)
        pre = free1 & (dist1 > offset)
        if int(pre.sum()) == 0:
            continue
        x0, x1 = x0[pre], x1[pre]
        free1, dist1, angle1, normal1, trans1, rot1 = (t[pre] for t in (free1, dist1, angle1, normal1, trans1, rot1))
        r0 = evaluate(x0, angle_pts=track_angle_pts)
        free0, dist0, angle0, normal0, trans0, rot0 = r0[:6]
        if testing:
            keep = free0 & (dist0 > offset)
        else:
            keep = free0 & (dist0 > offset) & (dist0 < margin)
        nk = int(keep.sum())
        if nk == 0:
            continue
        take = min(nk, number_pairs - count)
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)[:take]
        sl = slice(count, count + take)
        out['pairs'][sl, 0:6], out['pairs'][sl, 6:12] = x0[idx], x1[idx]
        out['dists'][sl, 0], out['dists'][sl, 1] = dist0[idx], dist1[idx]
        out['angles'][sl, 0], out['angles'][sl, 1] = angle0[idx], angle1[idx]
        out['normals'][sl, 0:6], out['normals'][sl, 6:12] = normal0[idx], normal1[idx]
        out['trans'][sl, 0:6], out['trans'][sl, 6:12] = trans0[idx], trans1[idx]
        out['rot'][sl, 0:6], out['rot'][sl, 6:12] = rot0[idx], rot1[idx]
        if track_angle_pts:
            ang_shape[sl], ang_env[sl] = r0[6][idx], r0[7][idx]
        count += take
        print('generated: {} / {}'.format(count, number_pairs), flush=True)
    return (out['pairs'], out['dists'], out['angles'], out['normals'], out['trans'], out['rot'],
            ang_shape, ang_env)


def selftest(argv):
    """Cover check: placements uniform over the cells free at c - DELTA - 3 (wider than
    S(c) = cells free at c - DELTA), evaluated exactly; every valid one (free, clearance > c) must
    lie in S(c).  Reported per c: valid placements found, and how many fall outside
    the grid sets free at c - m for m = 0 .. DELTA."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', type=int, required=True)
    ap.add_argument('--env', required=True)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--clear', default='0.001,0.02', help='normalized clearances to test (training offset, test offset)')
    ap.add_argument('--batch_size', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    global GRID
    GRID = Grid(a.env, a.shape, a.device)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    V, F, names = po.load_obj(a.env)
    V[:, 2] = 0.0
    Vn = (V - GRID.center) / GRID.scale
    keep = np.array(['null' not in str(n).lower() for n in names])
    env_t = torch.tensor(po.sample_surface_points(Vn, F[keep], po.DEFAULT_ENV_POINTS), dtype=torch.float32).to(a.device)
    Vs, Fs, _ = po.load_obj(a.shape)
    Vs = (Vs - 0.5 * (Vs.min(0) + Vs.max(0))) / GRID.scale
    z = Vs[:, 2]
    Vs[:, 2] = (z - 0.5 * (z.max() + z.min())) * (po.TWOD_SHAPE_THICKNESS / float(z.max() - z.min()))
    TV, TT, TF = po.tetrahedralize_shape(Vs, Fs)
    tets = torch.tensor(TV[TT], dtype=torch.float32).to(a.device)
    faces = torch.tensor(TV[TF], dtype=torch.float32).to(a.device)
    rp, rb = po.generate_radius_surface_points(Vs, Fs, 1000, 10)
    rp, rb = rp.to(a.device), rb.to(a.device)
    ok_all = True
    for c in (float(v) for v in a.clear.split(',')):
        cw = c * GRID.scale
        wide = GRID.cells(cw - DELTA - 3.0)
        sets = {m: GRID.cells(cw - m) for m in (0.0, 1.0, 2.0, 3.0, 4.0, DELTA)}
        prop = torch.nonzero(wide.reshape(-1)).squeeze(1)
        n_valid, outside = 0, {m: 0 for m in sets}
        t0 = time.time()
        for s0 in range(0, a.selftest, a.batch_size):
            x = GRID.propose(prop, min(a.batch_size, a.selftest - s0))
            free, dist = po.evaluate_placements(x.cpu(), tets, faces, env_t, rp, rb, a.device, two_d=True)[:2]
            valid = (free & (dist > c)).to(a.device)
            n_valid += int(valid.sum())
            for m, S in sets.items():
                outside[m] += int((valid & ~GRID.lookup(S, x)).sum())
        print('clearance {:.3f} ({:.2f} units): {} placements from S(c-{:g}), {} valid; valid outside '
              'grid-free at c-m: {}  ({:.0f}s)'.format(c, cw, a.selftest, DELTA + 3, n_valid,
                                                     ', '.join('m={:g}: {}'.format(m, v) for m, v in outside.items()),
                                                     time.time() - t0), flush=True)
        ok_all &= outside[DELTA] == 0 and n_valid > 0
    print('SELFTEST', 'PASSED: every valid placement lies in S' if ok_all else 'FAILED')
    return ok_all


def main():
    global GRID
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--env', required=True)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--device', default='cuda')
    a, _ = ap.parse_known_args()
    assert '--2d' in sys.argv, 'preprocess_obj_2dgrid.py is for --2d runs'
    GRID = Grid(a.env, a.shape, a.device)
    po.generate_valid_pairs = grid_generate_valid_pairs
    po.main()


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest(sys.argv[1:]) else 1)
    main()
