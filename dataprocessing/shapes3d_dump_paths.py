"""Run Lie-Shape-NTFields analysis/plan_eval.py and also save every planned path.

plan_eval.py keeps the MPPI paths and per-waypoint clearances only in memory;
this runs its source with two lines added, so the planner, filter and collision
test are the original ones, and writes next to ``--out`` (json as usual):

    <out>.npz   case, path (object array of (T, 6) chart poses), clear (per-waypoint
                signed clearance, < 0 = collision), converged, collision, success

    cd /lieshape && python /workspace/dataprocessing/shapes3d_dump_paths.py <plan_eval.py flags>
"""
import os
import sys

import numpy as np

ORIG = os.environ.get('PLAN_EVAL', '/lieshape/analysis/plan_eval.py')
KEEP = "        rec['_path'] = P\n"
DUMP = "for k, rec in enumerate(records):\n    P = rec.pop('_path')\n"

src = open(ORIG).read()
assert src.count(KEEP) == 1 and src.count(DUMP) == 1, 'plan_eval.py changed: anchors not found'
src = src.replace(KEEP, KEEP + "        rec['_clear'] = clear\n")
src = src.replace(DUMP, "_dump_paths(records, args.out)\n" + DUMP)


def _dump_paths(records, out):
    clear = [r.pop('_clear') for r in records]
    np.savez_compressed(
        os.path.splitext(out)[0] + '.npz',
        case=np.array([r['case'] for r in records]),
        path=np.array([r['_path'].astype(np.float32) for r in records] + [None], dtype=object)[:-1],
        clear=np.array([c.astype(np.float32) for c in clear] + [None], dtype=object)[:-1],
        converged=np.array([r['converged'] for r in records]),
        collision=np.array([r['collision'] for r in records]),
        success=np.array([r['success'] for r in records]))


sys.argv[0] = ORIG
# In this module's own namespace, so the worker pool can pickle path_clearance as __main__.path_clearance.
exec(compile(src, ORIG + ' [+ path dump]', 'exec'), globals())
