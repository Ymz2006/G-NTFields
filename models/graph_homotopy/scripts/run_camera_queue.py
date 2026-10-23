#!/usr/bin/env python3
"""Host-side queue of the camera + object test (2026-10-21): the bodies and scenes of dataprocessing/make_camera3d.py.

Cells (shape_env): bird3d / birdcam3d x forest / city.  Per cell: 4 x 100k training pairs (preprocess_sharded.py as
the shapes cells, which have 800k; the gated preprocess_obj_fast.py at batch 2000), then the three models of the comparison, seeds 0-2, 8000 epochs each:

  hminL   the homotopy head (graph_homotopy, --lift min)
  hlift   the lifted-encoder homotopy head (graph_homotopy_lift)
  mbtd    the full metric NTFields (TD + normal terms 1e-3)

Test sets: the two birdcam3d cells get a 1000-pair RRT-verified standard set (make_test_sets.py --variant std, named
as the cell).  Every model is scored on them with plan_eval_homotopy.py --camera --save-paths: the birdcam3d models
(the planner knows the camera's view volume) and the bird3d models of the same scene (the rigid follower that plans
the bird alone; its plans are checked against the whole body on the same pairs).  Homotopy heads: dual and base
planners; NTFields: base.  Results: results/graph_homotopy/plan_camera/<run>/<set>__<planner>.json.

    python3 models/graph_homotopy/scripts/run_camera_queue.py --gpus 0,1,2,3        # run / resume
    python3 models/graph_homotopy/scripts/run_camera_queue.py --dry-run

Containers as run_shapes_queue.py (gsp_ / gss_ / gst_ / gse_); status results/graph_homotopy/queue_status_camera.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_shapes_queue as rq  # noqa: E402

HOST, DATA, SETS, EXP_H, EXP_M = rq.HOST, rq.DATA, rq.SETS, rq.EXP_H, rq.EXP_M
RES = 'results/graph_homotopy/plan_camera'
CAMERA = '/workspace/datasets/3dshape/obj/birdcam3d_camera.json'
ENVS = ('forest', 'city')
BODIES = ('birdcam3d', 'bird3d')
SEEDS = (0, 1, 2)
MODELS = ('hminL', 'hlift', 'mbtd')
MEM = dict(rq.MEM, shard=10.0)     # the gated preprocess_obj_fast.py at batch 2000 reserves ~11 GB (3.5 GB allocated)
# 400k training pairs per cell (4 x 100k; the shapes cells have 800k): the body's radius makes every exact
# placement test 10-30x the cost of a tripod's, see the shard rates in the log
rq.TRAIN_N = 400000
rq.TRAIN_BS = 2000


def eval_done(run, setname, planner):
    return rq.complete_json(os.path.join(HOST, RES, run, '{}__{}.json'.format(setname, planner)))


def launch_set(cname, cell, workers):
    os.makedirs(os.path.join(HOST, SETS, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/models/graph_homotopy/scripts/make_test_sets.py --scene {c} --variant std --name {c} '
           '--n 1000 --workers {w} --out /workspace/{sets} > /workspace/{sets}/logs/{c}_std.out 2>&1').format(
        c=cell, w=workers, sets=SETS)
    rq.docker(cname, '/lieshape', cmd, threads=1)


def launch_eval(cname, run, kind, ckpt, setname, planner, workers, gpu):
    out = '/workspace/{}/{}'.format(RES, run)
    os.makedirs(os.path.join(HOST, RES, run), exist_ok=True)
    cmd = ('python -u /workspace/models/graph_homotopy/scripts/plan_eval_homotopy.py --planner {p} --kind {k} '
           '--checkpoint {ck} --dataPath /workspace/{sets}/{s} --out {o}/{s}__{p}.json --device cuda:0 --batch 125 '
           '--workers {w} --seed 0 --metric-root /export --camera {cam} --save-paths > {o}/{s}__{p}.log 2>&1').format(
        p=planner, k=kind, ck=ckpt, sets=SETS, o=out, s=setname, w=workers, cam=CAMERA)
    rq.docker(cname, '/lieshape', cmd, gpu=gpu, threads=1)


def build_jobs(a):
    J = []
    add = lambda *args, **kw: J.append(rq.Job(*args, **kw))  # noqa: E731
    pre = {}
    for ci, env in enumerate(ENVS):
        for body in BODIES:
            cell = '{}_{}'.format(body, env)
            pre[cell] = ()
            if not rq.cell_done(cell):
                for i in range(rq.SHARDS):
                    add('shard:{}:{}'.format(cell, i), 'shard', 0 + 0.01 * i + 0.001 * ci,
                        (lambda c=cell, ii=i: rq.shard_done(c, ii)), (lambda cn, g, c=cell, ii=i: rq.launch_shard(cn, c, ii, g)))
                add('merge:' + cell, 'merge', 0, (lambda c=cell: rq.cell_done(c)), (lambda cn, g, c=cell: rq.launch_merge(cn, c)),
                    deps=tuple('shard:{}:{}'.format(cell, i) for i in range(rq.SHARDS)))
                pre[cell] = ('merge:' + cell,)
        setname = 'birdcam3d_' + env
        add('set:' + setname, 'set', 0.5, (lambda s=setname: rq.set_done(s)),
            (lambda cn, g, c=setname: launch_set(cn, c, a.set_workers)), deps=pre[setname])
        for si, seed in enumerate(SEEDS):
            tier = 2 + si + 0.001 * ci
            for body in BODIES:
                cell = '{}_{}'.format(body, env)
                runs = []
                for mi, model in enumerate(MODELS):
                    name = '{}_{}_s{}'.format(cell, model, seed)
                    if model == 'mbtd':
                        add('mb:' + name, 'mb', tier + 0.0001 * mi, (lambda n=name: rq.train_done(EXP_M, n)),
                            (lambda cn, g, n=name, c=cell, ss=seed: rq.launch_train_mb(
                                cn, n, c, ss, g, '--td-weight 1e-3 --normal-weight 1e-3 --keep 4000')), deps=pre[cell])
                        runs.append((name, 'metric', '/workspace/{}/{}/latest.pt'.format(EXP_M, name), 'base', 'mb:' + name))
                    else:
                        arch = 'graph_homotopy' if model == 'hminL' else 'graph_homotopy_lift'
                        add('hom:' + name, 'hom', tier + 0.0001 * mi, (lambda n=name: rq.train_done(EXP_H, n)),
                            (lambda cn, g, n=name, c=cell, ss=seed, ar=arch: rq.launch_train_h(cn, n, c, ss, g, ar)), deps=pre[cell])
                        runs += [(name, 'graph', '/workspace/{}/{}/latest.pt'.format(EXP_H, name), p, 'hom:' + name)
                                 for p in ('dual', 'base')]
                for run, kind, ck, p, dep in runs:
                    add('eval:{}:{}:{}'.format(run, setname, p), 'eval', 1 + 0.001 * ci,
                        (lambda r=run, ss=setname, pp=p: eval_done(r, ss, pp)),
                        (lambda cn, g, r=run, k=kind, c=ck, ss=setname, pp=p: launch_eval(cn, r, k, c, ss, pp, a.eval_workers, g)),
                        deps=(dep, 'set:' + setname))
    return sorted(J, key=lambda j: (j.prio, j.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--reserve', type=float, default=1.5, help='GB left free on a GPU after a launch')
    ap.add_argument('--max-train-per-gpu', type=int, default=10)
    ap.add_argument('--max-eval', type=int, default=4)
    ap.add_argument('--max-shard', type=int, default=16)
    ap.add_argument('--max-set', type=int, default=2)
    ap.add_argument('--eval-workers', type=int, default=4)
    ap.add_argument('--set-workers', type=int, default=12)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(',')]
    if a.dry_run:
        jobs = build_jobs(a)
        for j in jobs:
            print('{:6.3f} {:5s} {:60s} {}'.format(j.prio, j.kind, j.name, 'done' if j.done() else ''))
        print(len(jobs), 'jobs,', sum(j.done() for j in jobs), 'done')
        return
    attempts, failed, launched = {}, {}, {}
    t0 = time.time()
    while True:
        jobs = build_jobs(a)
        by_name = {j.name: j for j in jobs}
        live = rq.containers()
        for n, cname in list(launched.items()):
            if cname not in live:
                j = by_name.get(n)
                if j is not None and not j.done():
                    attempts[n] = attempts.get(n, 0) + 1
                    if attempts[n] >= a.attempts:
                        failed[n] = 'exited without result after {} attempts'.format(attempts[n])
                del launched[n]
        kinds = {j.cname: j.kind for j in jobs}
        n_train = {g: 0 for g in gpus}
        mem_est = {g: 0.0 for g in gpus}
        n_eval = n_shard = n_set = 0
        for c, g in live.items():
            k = kinds.get(c, 'eval' if c.startswith('gse_') else 'hom')
            if g in mem_est:
                mem_est[g] += MEM[k]
                n_train[g] += k in ('hom', 'mb')
            n_eval += k == 'eval'
            n_shard += k == 'shard'
            n_set += k == 'set'
        done_of = lambda n: by_name[n].done() if n in by_name else True  # noqa: E731
        pend = [j for j in jobs if not j.done() and j.cname not in live and j.name not in failed and j.ready(done_of)]
        free = {g: min(f, rq.GPU_GB - mem_est.get(g, 0.0)) for g, f in rq.gpu_free_gb().items()}
        for j in pend:
            g = None
            if j.kind == 'set':
                if n_set >= a.max_set:
                    continue
            elif j.kind != 'merge':
                if j.kind == 'eval' and n_eval >= a.max_eval:
                    continue
                if j.kind == 'shard' and n_shard >= a.max_shard:
                    continue
                need = MEM[j.kind]
                ok = [x for x in gpus if free.get(x, 0) >= need + a.reserve
                      and not (j.kind in ('hom', 'mb') and n_train[x] >= a.max_train_per_gpu)]
                if not ok:
                    continue
                g = max(ok, key=lambda x: (free[x] - mem_est[x], -n_train[x]))
            j.launch(j.cname, g)
            launched[j.name] = j.cname
            print('[{:6.0f}s] launch {} gpu {}'.format(time.time() - t0, j.name, g), flush=True)
            if j.kind == 'set':
                n_set += 1
            elif g is not None:
                free[g] -= MEM[j.kind]
                mem_est[g] += MEM[j.kind]
                n_train[g] += j.kind in ('hom', 'mb')
                n_eval += j.kind == 'eval'
                n_shard += j.kind == 'shard'
            time.sleep(2)
        status = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': time.time() - t0,
                  'done': sum(j.done() for j in jobs), 'total': len(jobs),
                  'by_kind': {k: [sum(j.done() for j in jobs if j.kind == k), sum(1 for j in jobs if j.kind == k)]
                              for k in ('shard', 'merge', 'set', 'hom', 'mb', 'eval')},
                  'running': sorted(live), 'gpu_free_gb': rq.gpu_free_gb(), 'failed': failed}
        try:
            dst = os.path.join(HOST, 'results/graph_homotopy/queue_status_camera.json')
            with open(dst + '.tmp', 'w') as fh:
                json.dump(status, fh, indent=1)
            os.replace(dst + '.tmp', dst)
        except OSError as e:
            print('status not written: {}'.format(e), flush=True)
        if all(j.done() or j.name in failed for j in jobs) and not live:
            print('all done; failed: {}'.format(failed), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
