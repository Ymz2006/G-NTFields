"""preprocess_obj.py with an exact early rejection of training x0 candidates.

Same command line and outputs as the original (Lie-Shape-NTFields
dataprocessing/preprocess_obj.py, path in $PREPROCESS_OBJ), and the same
samples: for a fixed numpy / torch seed it writes the same arrays.

Why: in training mode every x0 candidate gets the full evaluate_placements
(clearance, collision, rotational clearance, normals) against every env point
within ``min_dist + 2 R`` of its centre, and only those with
``free & offset < clearance < margin`` survive.  With a large shape in a tight
room most candidates collide, so almost all of that work is thrown away.

What changes: one block is inserted into the original generate_valid_pairs,
right after its ``m0 - R <= margin`` gate.  It computes the survivor test
``free & offset < clearance < margin`` exactly, but only on the env points
that can affect it -- a point inside a tet lies within R of the centre, and a
point closer than ``margin`` to the surface lies within ``R + margin`` -- and
drops the candidates that fail.  The original evaluate_placements then runs on
the survivors only, so every stored value comes from the original code, and
the survivors (hence the torch random stream) are the same as without the block.

    python dataprocessing/preprocess_obj_fast.py <preprocess_obj.py flags>
    python dataprocessing/preprocess_obj_fast.py --selftest 2000 --env ... --shape ...  # compare with the original
"""
import importlib.util
import inspect
import os
import sys
import textwrap

import numpy as np
import torch

ORIG = os.environ.get('PREPROCESS_OBJ', '/lieshape/dataprocessing/preprocess_obj.py')
_spec = importlib.util.spec_from_file_location('preprocess_obj', ORIG)
po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(po)

ANCHOR = '            x0 = x0[feas0]\n'
INSERT = '''            band = x0_band_mask(x0, tet_verts_local, face_verts_local, env_t, margin, offset, device)
            if int(band.sum()) == 0:
                continue
            x0 = x0[band]
'''


def x0_band_mask(x0, tet_verts_local, face_verts_local, env_t, margin, offset, device):
    """``free & (offset < clearance < margin)`` per candidate, exactly as
    evaluate_placements would compute it, on the env points that can matter."""
    cfg = x0.to(device)
    B = cfg.shape[0]
    t = cfg[:, 0:3]
    R = po.rotvec_to_matrix(cfg[:, 3:6])
    # The same transforms as evaluate_placements.
    tets = tet_verts_local.reshape(1, -1, 3)
    tets = torch.einsum('bij,nj->bni', R, tets.squeeze(0)) + t.unsqueeze(1)
    tets = tets.reshape(B, -1, 4, 3)
    faces = face_verts_local.reshape(-1, 3)
    faces = torch.einsum('bij,nj->bni', R, faces) + t.unsqueeze(1)
    faces = faces.reshape(B, -1, 3, 3)

    R_shape = face_verts_local.reshape(-1, 3).norm(dim=1).max()
    to_env_dist = torch.linalg.norm(env_t.unsqueeze(0) - t.unsqueeze(1), dim=-1)   # (B,E)
    sorted_d, sort_idx = torch.sort(to_env_dist, dim=1)

    # Collision: a point inside a tet is within R of the centre (slack for the
    # -1e-6 barycentric tolerance of points_inside_tets).
    r_col = R_shape * (1 + 1e-4) + 1e-6
    n_col = int((sorted_d <= r_col).sum(dim=1).max().clamp(min=1))
    free = po.points_inside_tets(tets, env_t[sort_idx[:, :n_col]])
    band = free.clone()
    fi = torch.nonzero(free, as_tuple=False).squeeze(1)
    if len(fi):
        # Clearance < margin needs a point within R + margin of the centre; on that
        # set the minimum is the true clearance whenever it is below margin, and
        # stays >= margin otherwise.
        r_clr = R_shape + margin * (1 + 1e-4) + 1e-6
        n_clr = int((sorted_d[fi] <= r_clr).sum(dim=1).max().clamp(min=1))
        dist, _, _ = po.calculate_dist(faces[fi], env_t[sort_idx[fi, :n_clr]], t[fi])
        band[fi] = (dist > offset) & (dist < margin)
    return band.cpu()


def _fast_generate_valid_pairs():
    src = textwrap.dedent(inspect.getsource(po.generate_valid_pairs))
    assert src.count(ANCHOR) == 1, 'anchor not found once in preprocess_obj.generate_valid_pairs'
    src = src.replace(ANCHOR, ANCHOR + INSERT)
    ns = po.__dict__
    ns['x0_band_mask'] = x0_band_mask
    exec(compile(src, ORIG + ' [+ x0_band_mask]', 'exec'), ns)
    return ns['generate_valid_pairs']


ORIGINAL_GENERATE = po.generate_valid_pairs
FAST_GENERATE = _fast_generate_valid_pairs()   # also installs itself as po.generate_valid_pairs


def selftest(argv):
    """Run original and fast generate_valid_pairs on the same scene and seeds and
    compare every returned array."""
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', type=int, required=True)
    ap.add_argument('--env', required=True)
    ap.add_argument('--shape', required=True)
    ap.add_argument('--margin', type=float, default=0.05)
    ap.add_argument('--offset', type=float, default=0.001)
    ap.add_argument('--batch_size', type=int, default=250)
    ap.add_argument('--num_env_points', type=int, default=po.DEFAULT_ENV_POINTS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    np.random.seed(a.seed)
    V, F, names = po.load_obj(a.env)
    lo, hi = V.min(0), V.max(0)
    scale = float((hi - lo).max())
    Vn = (V - 0.5 * (lo + hi)) / scale
    keep = np.array(['null' not in str(n).lower() for n in names])
    env = po.sample_surface_points(Vn, F[keep], a.num_env_points).astype(np.float32)
    half = (hi - lo) / scale * 0.5 - 0.01
    Vs, Fs, _ = po.load_obj(a.shape)
    Vs = (Vs - 0.5 * (Vs.min(0) + Vs.max(0))) / scale
    TV, TT, TF = po.tetrahedralize_shape(Vs, Fs)
    tets = torch.tensor(TV[TT], dtype=torch.float32)
    faces = torch.tensor(TV[TF], dtype=torch.float32)
    rp, rb = po.generate_radius_surface_points(Vs, Fs, 1000, 10)
    out = {}
    for name, fn in (('original', ORIGINAL_GENERATE), ('fast', FAST_GENERATE)):
        torch.manual_seed(a.seed)
        t0 = time.time()
        res = fn(a.selftest, tets, faces, env, half, margin=a.margin, offset=a.offset,
                 rad_points=rp, rad_bins=rb, batch_size=a.batch_size, device=a.device,
                 track_angle_pts=True)
        out[name] = res
        print('{:8s} {:.1f}s'.format(name, time.time() - t0), flush=True)
    labels = ['pairs', 'dists', 'angles', 'normals', 'trans_n', 'rot_n', 'ang_shape', 'ang_env']
    same = True
    for lab, x, y in zip(labels, out['original'], out['fast']):
        eq = torch.equal(x, y)
        same &= eq
        print('  {:9s} identical {}  max |diff| {:.3g}'.format(lab, eq, float((x - y).abs().max())))
    print('SELFTEST', 'PASSED: identical outputs' if same else 'FAILED')
    return same


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest(sys.argv[1:]) else 1)
    po.main()
