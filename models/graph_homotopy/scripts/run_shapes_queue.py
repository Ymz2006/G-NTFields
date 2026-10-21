#!/usr/bin/env python3
"""Host-side queue for the third comparison of the 8000-epoch homotopy head against the original metric NTFields
(2026-10-17, third request), on this machine's three RTX 3090, GPU-memory-aware, one GPU per job. Two grids:

  shapes   the five shapes of dataprocessing/make_shapes3d.py (tripod3d, crank3d, loop3d, twistI3d, chair3d) on
           env4..env9 (30 cells, brought over from G-Shape-NTFields-old: datasets/3dshape/data/<cell>, standard test
           sets testing_data/3dshape/<cell>, 1000 pairs each);
  tla      the T / L / A plates on env1 and env3 (6 cells; Tshape3d_env1 and Lshape3d_env3 existed, the other four
           are preprocessed here with dataprocessing/preprocess_sharded.py, 4 x 200k pairs, as the shapes cells);
           test sets: the standard sets (Lie repo, copied to testing_data/3dshape/<cell>) and the targeted variants
           long / bigrot / tight / hard (scripts/make_test_sets.py; generated here for the four new cells).
  gibson   (--grid gibson, run on its own; --grid all = both grids in one queue) the cube and the small rectangle in the Gibson scene eastville
           (cube_eastville, rect_small_eastville: preprocessed in G-Shape-NTFields-old, 800k pairs; standard test
           sets testing_data/3dshape/<cell>), seeds 0-5 as the six seeds of the old metric runs (results/gibson_metric).

Per cell and seed (0, 1, 2; seed tiers run in order): both models trained 8000 epochs (homotopy: train_graph_homotopy.py
--lift min --lift-far 0.5; NTFields: train_metric_baseline.py with the bundle /media/4tbssd/jeffrey/export_metric,
epoch-4000 checkpoint kept) and scored with plan_eval_homotopy.py on every set of the cell: homotopy base and dual
planners, NTFields 8000 and 4000 epochs (base planner).

    python3 models/graph_homotopy/scripts/run_shapes_queue.py            # run everything (resumable)
    python3 models/graph_homotopy/scripts/run_shapes_queue.py --dry-run  # list jobs and their state

Containers: gsp_ preprocessing shard / merge, gss_ set generation (CPU), gst_ training, gse_ evaluation; running
ones are adopted on restart. The lifted-encoder head (hlift, arch graph_homotopy_lift) is trained and scored
next to hminL on every cell and seed. Status: results/graph_homotopy/queue_status_shapes.json. Standard library only.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LIE = '/media/jeffrey/Lie-Shape-NTFields'
EXPORT = '/media/4tbssd/jeffrey/export_metric'
IMAGE = 'pytorchserver:latest'
EXP_H = 'Experiments/graph_homotopy'
EXP_M = 'Experiments/metric_baseline'
RES = 'results/graph_homotopy/plan_shapes'
SETS = 'testing_data/3dshape'
DATA = 'datasets/3dshape/data'
OBJ = 'datasets/3dshape/obj'
SHAPES = ('tripod3d', 'crank3d', 'loop3d', 'twistI3d', 'chair3d')
ENVS = ('env4', 'env5', 'env6', 'env7', 'env8', 'env9')
TLA = {'T': 'Tshape3d', 'L': 'Lshape3d', 'A': 'Ashape3d'}
TLA_ENVS = ('env1', 'env3')
VARIANTS = ('long', 'bigrot', 'tight', 'hard')
SEEDS = (0, 1, 2)
GIBSON = ('cube_eastville', 'rect_small_eastville')
GIBSON_SEEDS = (0, 1, 2, 3, 4, 5)
SHARDS = 4
TRAIN_N = 800000
TRAIN_BS = 500          # at 1000 the caching allocator swings between 3 and 14 GB next to the trainings
EPOCHS = 8000
MEM = {'shard': 8.0, 'merge': 0.0, 'set': 0.0, 'hom': 0.8, 'mb': 3.2, 'eval': 1.0}   # GB per job kind
GPU_GB = 24.0
PREFIX = ('gsp_', 'gss_', 'gst_', 'gse_')


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def containers():
    """{name: gpu index or None} of our running containers."""
    out = {}
    for line in sh("docker ps --format '{{.Names}} {{.Label \"gpu\"}}'").splitlines():
        parts = line.split()
        if parts and parts[0].startswith(PREFIX):
            out[parts[0]] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return out


def gpu_free_gb():
    free = {}
    for line in sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits').strip().splitlines():
        i, u, t = (float(x) for x in line.split(','))
        free[int(i)] = (t - u) / 1024.0
    return free


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def exists(*parts):
    return os.path.exists(os.path.join(HOST, *parts))


def docker(cname, workdir, cmd, gpu=None, threads=2, extra_env=''):
    sh(('docker run -d --rm {g} --label gpu={gl} -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp '
        '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} {ee} -v {host}:/workspace -v {lie}:/lieshape:ro '
        '-v {exp}:/export:ro -w {wd} --name {c} {img} /bin/bash -c "{cmd}"').format(
        g='--gpus device={}'.format(gpu) if gpu is not None else '', gl='' if gpu is None else gpu,
        uid=os.getuid(), gid=os.getgid(), th=threads, ee=extra_env, host=HOST, lie=LIE, exp=EXPORT, wd=workdir,
        c=cname, img=IMAGE, cmd=cmd))


# ----------------------------------------------------------------------------- job table
class Job:
    def __init__(self, name, kind, prio, done, launch, deps=()):
        self.name, self.kind, self.prio, self.done, self.launch, self.deps = name, kind, prio, done, launch, deps
        self.cname = {'shard': 'gsp_', 'merge': 'gsp_', 'set': 'gss_', 'hom': 'gst_', 'mb': 'gst_',
                      'eval': 'gse_'}[kind] + name.split(':', 1)[1].replace(':', '__')

    def ready(self, done_of):
        return all(done_of(d) for d in self.deps)


def cell_done(cell):
    return all(exists(DATA, cell, f) for f in ('sampled_points.npy', 'speed.npy', 'normal.npy', 'meta.json'))


def shard_done(cell, i):
    return cell_done(cell) or exists(DATA, cell, 'shards', 'shard_{}'.format(i), 'meta.json')


def set_done(setname):
    return exists(SETS, setname, 'sampled_points.npy') and exists(SETS, setname, 'meta.json')


def train_done(exp, name):
    return exists(exp, name, 'train_summary.json') and exists(exp, name, 'latest.pt')


def eval_done(run, setname, planner):
    return complete_json(os.path.join(HOST, RES, run, '{}__{}.json'.format(setname, planner)))


def launch_shard(cname, cell, i, gpu):
    shape, env = cell.split('_')
    os.makedirs(os.path.join(HOST, DATA, 'logs'), exist_ok=True)
    cmd = ('PREPROCESS_OBJ=/lieshape/dataprocessing/preprocess_obj.py python -u dataprocessing/preprocess_sharded.py run '
           '--script dataprocessing/preprocess_obj_fast.py --shard {i} --shards {S} --env_seed 1 -- '
           '--env {o}/{env}.obj --shape {o}/{shape}.obj --out {d}/{cell} --num_samples {n} --batch_size {bs} '
           '--device cuda:0 > {d}/logs/pre_{cell}_{i}.out 2>&1').format(
        i=i, S=SHARDS, o=OBJ, env=env, shape=shape, d=DATA, cell=cell, n=TRAIN_N, bs=TRAIN_BS)
    docker(cname, '/workspace', cmd, gpu=gpu, extra_env='-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')


def launch_merge(cname, cell):
    cmd = ('PREPROCESS_OBJ=/lieshape/dataprocessing/preprocess_obj.py python -u dataprocessing/preprocess_sharded.py merge '
           '--script /lieshape/dataprocessing/preprocess_obj.py --shards {S} --env_seed 1 --out {d}/{cell} '
           '> {d}/logs/merge_{cell}.out 2>&1').format(S=SHARDS, d=DATA, cell=cell)
    docker(cname, '/workspace', cmd)


def launch_set(cname, scene, variant, workers):
    os.makedirs(os.path.join(HOST, SETS, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/models/graph_homotopy/scripts/make_test_sets.py --scene {sc} --variant {v} --n 1000 '
           '--workers {w} --out /workspace/{sets} > /workspace/{sets}/logs/{sc}_{v}.out 2>&1').format(
        sc=scene, v=variant, w=workers, sets=SETS)
    docker(cname, '/lieshape', cmd, threads=1)


def launch_train_h(cname, name, cell, seed, gpu, arch='graph_homotopy'):
    os.makedirs(os.path.join(HOST, EXP_H, 'logs'), exist_ok=True)
    cmd = ('python -u train/train_graph_homotopy.py --dataPath {d}/{cell} --name {n} --seed {s} --epochs {ep} '
           '--lift min --lift-far 0.5 --arch {a} > {exp}/logs/{n}.out 2>&1').format(
        d=DATA, cell=cell, n=name, s=seed, ep=EPOCHS, a=arch, exp=EXP_H)
    docker(cname, '/workspace', cmd, gpu=gpu)


def launch_train_mb(cname, name, cell, seed, gpu, extra=''):
    os.makedirs(os.path.join(HOST, EXP_M, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/train/train_metric_baseline.py --dataPath /workspace/{d}/{cell} --name {n} --seed {s} '
           '--epochs {ep} --bundle /export --out_root /workspace/{exp} {x} > /workspace/{exp}/logs/{n}.out 2>&1').format(
        d=DATA, cell=cell, n=name, s=seed, ep=EPOCHS, exp=EXP_M, x=extra)
    docker(cname, '/export', cmd, gpu=gpu)


def launch_eval(cname, run, kind, ckpt, setname, planner, workers, gpu):
    out = '/workspace/{}/{}'.format(RES, run)
    os.makedirs(os.path.join(HOST, RES, run), exist_ok=True)
    cmd = ('python -u /workspace/models/graph_homotopy/scripts/plan_eval_homotopy.py --planner {p} --kind {k} '
           '--checkpoint {ck} --dataPath /workspace/{sets}/{s} --out {o}/{s}__{p}.json --device cuda:0 --batch 125 '
           '--workers {w} --seed 0 --metric-root /export > {o}/{s}__{p}.log 2>&1').format(
        p=planner, k=kind, ck=ckpt, sets=SETS, o=out, s=setname, w=workers)
    docker(cname, '/lieshape', cmd, gpu=gpu, threads=1)


def cells(which='main'):
    """(cell, run code, grid) in launch order: the T / L / A cells first, then the shapes, narrowest env last."""
    if which == 'gibson':
        return [(c, c, 'gibson') for c in GIBSON]
    if which == 'all':
        return cells('main') + cells('gibson')
    out = [('{}_{}'.format(TLA[k], e), '{}{}'.format(k, e[3:]), 'tla') for e in TLA_ENVS for k in TLA]
    out += [('{}_{}'.format(s, e), '{}_{}'.format(s, e), 'shapes') for e in ENVS for s in SHAPES]
    return out


def sets_of(cell, grid):
    return [cell] + (['{}_{}'.format(cell, v) for v in VARIANTS] if grid == 'tla' else [])


def build_jobs(a):
    J = []
    add = lambda *args, **kw: J.append(Job(*args, **kw))  # noqa: E731
    for ci, (cell, code, grid) in enumerate(cells(a.grid)):
        pre = []
        if not cell_done(cell):
            for i in range(SHARDS):
                add('shard:{}:{}'.format(cell, i), 'shard', 0 + 0.01 * i, (lambda c=cell, ii=i: shard_done(c, ii)),
                    (lambda cn, g, c=cell, ii=i: launch_shard(cn, c, ii, g)))
            add('merge:' + cell, 'merge', 0, (lambda c=cell: cell_done(c)), (lambda cn, g, c=cell: launch_merge(cn, c)),
                deps=tuple('shard:{}:{}'.format(cell, i) for i in range(SHARDS)))
            pre = ('merge:' + cell,)
        set_deps = {}
        for s in sets_of(cell, grid)[1:]:
            v = s.rsplit('_', 1)[1]
            add('set:' + s, 'set', 0.5, (lambda ss=s: set_done(ss)), (lambda cn, g, c=cell, vv=v: launch_set(cn, c, vv, a.set_workers)))
            set_deps[s] = ('set:' + s,)
        for si, seed in enumerate(GIBSON_SEEDS if grid == 'gibson' else SEEDS):
            tier = 2 + si + 0.001 * ci
            nh, nm = '{}_hminL_s{}'.format(code, seed), '{}_mb_s{}'.format(code, seed)
            add('hom:' + nh, 'hom', tier, (lambda n=nh: train_done(EXP_H, n)),
                (lambda cn, g, n=nh, c=cell, ss=seed: launch_train_h(cn, n, c, ss, g)), deps=pre)
            add('mb:' + nm, 'mb', tier + 0.0005, (lambda n=nm: train_done(EXP_M, n)),
                (lambda cn, g, n=nm, c=cell, ss=seed: launch_train_mb(cn, n, c, ss, g)), deps=pre)
            for s in sets_of(cell, grid):
                sd = set_deps.get(s, ())
                runs = [(nh, 'graph', '/workspace/{}/{}/latest.pt'.format(EXP_H, nh), p, 'hom:' + nh) for p in ('base', 'dual')]
                runs += [(nm, 'metric', '/workspace/{}/{}/latest.pt'.format(EXP_M, nm), 'base', 'mb:' + nm),
                         ('{}_mb4k_s{}'.format(code, seed), 'metric', '/workspace/{}/{}/epoch4000.pt'.format(EXP_M, nm), 'base', 'mb:' + nm)]
                # the full metric NTFields: TD and normal-alignment terms on (1e-3), as G-Shape-NTFields-old models/metric
                # (the old Gibson runs); scored at 8000 epochs and at 5000 (Gibson, as the old runs) / 4000 (as mb4k)
                nt, kk = '{}_mbtd_s{}'.format(code, seed), (5000 if grid == 'gibson' else 4000)
                if s == cell:
                    add('mb:' + nt, 'mb', tier + 0.0007, (lambda n=nt: train_done(EXP_M, n)),
                        (lambda cn, g, n=nt, c=cell, ss=seed: launch_train_mb(
                            cn, n, c, ss, g, '--td-weight 1e-3 --normal-weight 1e-3 --keep 4000,5000')), deps=pre)
                runs += [(nt, 'metric', '/workspace/{}/{}/latest.pt'.format(EXP_M, nt), 'base', 'mb:' + nt),
                         ('{}_mbtd{}k_s{}'.format(code, kk // 1000, seed), 'metric',
                          '/workspace/{}/{}/epoch{}.pt'.format(EXP_M, nt, kk), 'base', 'mb:' + nt)]
                if grid == 'gibson':
                    # the old repo's own checkpoints (G-Shape-NTFields-old, 5000 epochs, seeds 1-6), copied over
                    oc = '/workspace/Experiments/gibson_old_ntf/{}_s{}/epoch5000.pt'.format(cell, seed + 1)
                    runs += [('{}_oldntf_s{}'.format(code, seed + 1), 'metric', oc, 'base', None)]
                # the lifted-encoder homotopy head (arch graph_homotopy_lift, 2026-10-19): the encoder reads (t, +-q) on
                # the double cover instead of G(x) = (t, vec R); everything else as hminL
                nl = '{}_hlift_s{}'.format(code, seed)
                if s == cell:
                    add('hom:' + nl, 'hom', tier + 0.0009, (lambda n=nl: train_done(EXP_H, n)),
                        (lambda cn, g, n=nl, c=cell, ss=seed: launch_train_h(cn, n, c, ss, g, 'graph_homotopy_lift')),
                        deps=pre)
                runs += [(nl, 'graph', '/workspace/{}/{}/latest.pt'.format(EXP_H, nl), p, 'hom:' + nl) for p in ('base', 'dual')]
                for run, kind, ck, p, dep in runs:
                    add('eval:{}:{}:{}'.format(run, s, p), 'eval', 1 + 0.001 * ci, (lambda r=run, ss=s, pp=p: eval_done(r, ss, pp)),
                        (lambda cn, g, r=run, k=kind, c=ck, ss=s, pp=p: launch_eval(cn, r, k, c, ss, pp, a.eval_workers, g)),
                        deps=((dep,) if dep else ()) + sd)
    return sorted(J, key=lambda j: (j.prio, j.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='main', choices=('main', 'gibson', 'all'))
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--reserve', type=float, default=1.5, help='GB left free on a GPU after a launch')
    ap.add_argument('--max-train-per-gpu', type=int, default=8)
    ap.add_argument('--max-eval', type=int, default=4)
    ap.add_argument('--max-shard', type=int, default=4)
    ap.add_argument('--max-set', type=int, default=2)
    ap.add_argument('--eval-workers', type=int, default=4)
    ap.add_argument('--set-workers', type=int, default=8)
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
        live = containers()
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
        # a container that has just started has not allocated yet: trust the smaller of the reading and the estimate
        free = {g: min(f, GPU_GB - mem_est.get(g, 0.0)) for g, f in gpu_free_gb().items()}
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
                  'running': sorted(live), 'gpu_free_gb': gpu_free_gb(), 'failed': failed}
        try:
            dst = os.path.join(HOST, 'results/graph_homotopy/queue_status_{}.json'.format('shapes' if a.grid == 'main' else a.grid))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
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
