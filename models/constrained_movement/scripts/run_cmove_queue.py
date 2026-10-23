#!/usr/bin/env python3
"""Host-side queue of the constrained-movement test (metric NTFields), GPU-memory aware, one GPU per job,
resumable (a job is done when its output exists; running containers are adopted).

Per scene (path a of the followed shape, datasets/3dshape/paths/cm_<scene>_a.npz):

    pre    cell datasets/3dshape/data/cm_<scene>_a (preprocess_cmove.py, 800k pairs)
    set    testing_data/3dshape/cm_<scene>_a_{full, seg} (make_cmove_tests.py, 1000 RRT-verified cases)
    ntf    Experiments/constrained_movement/<scene>_ntf_s<seed> (train/train_metric_path.py, 8000 epochs)
    eval   results/constrained_movement/plan/<run>/<set>__base_{forward, free}.json (plan_eval_cmove.py)

    python3 models/constrained_movement/scripts/run_cmove_queue.py --dry-run
    python3 models/constrained_movement/scripts/run_cmove_queue.py --gpus 0,1,2,3 --seeds 0,1,2

Containers: cmq_p (preprocessing), cmq_s (sets, CPU), cmq_t (trainings), cmq_e (evaluations).
Status: results/constrained_movement/queue_status.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPORT = '/media/4tbssd/jeffrey/export_metric'
IMAGE = 'pytorchserver:latest'
EXP = 'Experiments/constrained_movement'
RES = 'results/constrained_movement/plan'
SETS = 'testing_data/3dshape'
DATA = 'datasets/3dshape/data'
PATHS = 'datasets/3dshape/paths'
EPOCHS = 8000
MEM = {'pre': 2.0, 'set': 0.0, 'ntf': 2.0, 'eval': 1.5}
GPU_GB = 24.0
PREFIX = {'pre': 'cmq_p_', 'set': 'cmq_s_', 'ntf': 'cmq_t_', 'eval': 'cmq_e_'}
VARIANTS = ('full', 'seg')
MODES = ('forward', 'free')


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def containers():
    out = {}
    for line in sh("docker ps --format '{{.Names}} {{.Label \"gpu\"}}'").splitlines():
        parts = line.split()
        if parts and parts[0].startswith(tuple(PREFIX.values())):
            out[parts[0]] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return out


def gpu_free_gb():
    free = {}
    for line in sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits').strip().splitlines():
        i, u, t = (float(x) for x in line.split(','))
        free[int(i)] = (t - u) / 1024.0
    return free


def exists(*parts):
    return os.path.exists(os.path.join(HOST, *parts))


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def docker(cname, workdir, cmd, gpu=None, threads=2):
    sh(('docker run -d --rm {g} --label gpu={gl} -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e NTF_NO_PLOT=1 '
        '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} -v {host}:/workspace -v {exp}:/export:ro -w {wd} '
        '--name {c} {img} /bin/bash -c "{cmd}"').format(
        g='--gpus device={}'.format(gpu) if gpu is not None else '', gl='' if gpu is None else gpu, uid=os.getuid(),
        gid=os.getgid(), th=threads, host=HOST, exp=EXPORT, wd=workdir, c=cname, img=IMAGE, cmd=cmd))


class Job:
    def __init__(self, name, kind, prio, done, launch, deps=()):
        self.name, self.kind, self.prio, self.done, self.launch, self.deps = name, kind, prio, done, launch, deps
        self.cname = PREFIX[kind] + name.split(':', 1)[1].replace(':', '__')

    def ready(self, done_of):
        return all(done_of(d) for d in self.deps)


def cell_done(cell):
    return exists(DATA, cell, 'meta.json')


def set_done(s):
    return exists(SETS, s, 'cases.npz') and exists(SETS, s, 'verification.json')


def train_done(name):
    return exists(EXP, name, 'latest.pt') and exists(EXP, name, 'train_summary.json')


def eval_out(run, s, mode):
    return os.path.join(HOST, RES, run, '{}__base_{}.json'.format(s, mode))


def launch_pre(cname, scene, gpu):
    os.makedirs(os.path.join(HOST, DATA, 'logs'), exist_ok=True)
    cmd = ('python -u models/constrained_movement/scripts/preprocess_cmove.py --path {p}/cm_{s}_a.npz '
           '--out {d}/cm_{s}_a --num_samples 800000 --batch 16384 > {d}/logs/pre_cm_{s}_a.out 2>&1').format(
        p=PATHS, d=DATA, s=scene)
    docker(cname, '/workspace', cmd, gpu=gpu)


def launch_set(cname, cell, variant, workers):
    os.makedirs(os.path.join(HOST, SETS, 'logs'), exist_ok=True)
    cmd = ('python -u models/constrained_movement/scripts/make_cmove_tests.py --cell {d}/{c} --variant {v} --n 1000 '
           '--workers {w} --out {s} > {s}/logs/{c}_{v}.out 2>&1').format(d=DATA, c=cell, v=variant, w=workers, s=SETS)
    docker(cname, '/workspace', cmd, threads=1)


def launch_ntf(cname, name, cell, seed, gpu):
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/train/train_metric_path.py --dataPath /workspace/{d}/{c} --name {n} --seed {s} '
           '--epochs {ep} --bundle /export --out_root /workspace/{e} --keep 2000,4000 > /workspace/{e}/logs/{n}.out 2>&1').format(
        d=DATA, c=cell, n=name, s=seed, ep=EPOCHS, e=EXP)
    docker(cname, '/export', cmd, gpu=gpu)


def launch_eval(cname, run, s, mode, gpu):
    out = eval_out(run, s, mode)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rel = os.path.relpath(out, HOST)
    cmd = ('python -u models/constrained_movement/scripts/plan_eval_cmove.py --kind metric '
           '--checkpoint /workspace/{e}/{r}/latest.pt --dataPath {sets}/{s} --mode {m} --out {o} --device cuda:0 '
           '--save-paths > {lo} 2>&1').format(e=EXP, r=run, sets=SETS, s=s, m=mode, o=rel, lo=rel[:-5] + '.log')
    docker(cname, '/workspace', cmd, gpu=gpu, threads=2)


def build_jobs(a):
    J = []
    add = lambda *x, **kw: J.append(Job(*x, **kw))  # noqa: E731
    for si, scene in enumerate(a.scenes):
        cell = 'cm_{}_a'.format(scene)
        add('pre:' + cell, 'pre', 0, (lambda c=cell: cell_done(c)), (lambda cn, g, sc=scene: launch_pre(cn, sc, g)))
        for v in VARIANTS:
            s = '{}_{}'.format(cell, v)
            add('set:' + s, 'set', 1 if v == 'full' else 1.5, (lambda ss=s: set_done(ss)),
                (lambda cn, g, c=cell, vv=v: launch_set(cn, c, vv, a.set_workers)), deps=('pre:' + cell,))
        for seed in a.seeds:
            if seed > 0 and a.extra_seed_scenes and scene not in a.extra_seed_scenes:
                continue
            run = '{}_ntf_s{}'.format(scene, seed)
            add('ntf:' + run, 'ntf', 2 + seed, (lambda n=run: train_done(n)),
                (lambda cn, g, n=run, c=cell, ss=seed: launch_ntf(cn, n, c, ss, g)), deps=('pre:' + cell,))
            for v in VARIANTS:
                s = '{}_{}'.format(cell, v)
                for m in MODES:
                    add('eval:{}:{}:{}'.format(run, s, m), 'eval', 0.5 + 0.1 * seed + 0.01 * (m == 'free'),
                        (lambda r=run, ss=s, mm=m: complete_json(eval_out(r, ss, mm))),
                        (lambda cn, g, r=run, ss=s, mm=m: launch_eval(cn, r, ss, mm, g)),
                        deps=('ntf:' + run, 'set:' + s))
    return sorted(J, key=lambda j: (j.prio, j.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='forest,city,corridor,env4')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--extra-seed-scenes', default='', help='only these scenes get seeds > 0 (default all)')
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--reserve', type=float, default=2.0)
    ap.add_argument('--max-train-per-gpu', type=int, default=4)
    ap.add_argument('--max-train', type=int, default=5, help='trainings at once (seed order)')
    ap.add_argument('--max-eval', type=int, default=4)
    ap.add_argument('--max-set', type=int, default=2)
    ap.add_argument('--max-pre', type=int, default=4)
    ap.add_argument('--set-workers', type=int, default=10)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.scenes = [s for s in a.scenes.split(',') if s]
    a.seeds = [int(s) for s in a.seeds.split(',') if s]
    a.extra_seed_scenes = [s for s in a.extra_seed_scenes.split(',') if s]
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
        for nm, cname in list(launched.items()):
            if cname not in live:
                j = by_name.get(nm)
                if j is not None and not j.done():
                    attempts[nm] = attempts.get(nm, 0) + 1
                    if attempts[nm] >= a.attempts:
                        failed[nm] = 'exited without result after {} attempts'.format(attempts[nm])
                del launched[nm]
        kinds = {j.cname: j.kind for j in jobs}
        n_train = {g: 0 for g in gpus}
        mem_est = {g: 0.0 for g in gpus}
        count = {k: 0 for k in PREFIX}
        for c, g in live.items():
            k = kinds.get(c, next(kk for kk, p in PREFIX.items() if c.startswith(p)))
            if g in mem_est:
                mem_est[g] += MEM[k]
                n_train[g] += k == 'ntf'
            count[k] += 1
        done_of = lambda nm: by_name[nm].done() if nm in by_name else True  # noqa: E731
        pend = [j for j in jobs if not j.done() and j.cname not in live and j.name not in failed and j.ready(done_of)]
        free = {g: min(f, GPU_GB - mem_est.get(g, 0.0)) for g, f in gpu_free_gb().items() if g in gpus}
        cap = {'pre': a.max_pre, 'set': a.max_set, 'eval': a.max_eval, 'ntf': a.max_train}
        for j in pend:
            if count[j.kind] >= cap[j.kind]:
                continue
            g = None
            if j.kind != 'set':
                ok = [x for x in gpus if free.get(x, 0) >= MEM[j.kind] + a.reserve
                      and not (j.kind == 'ntf' and n_train[x] >= a.max_train_per_gpu)]
                if not ok:
                    continue
                g = max(ok, key=lambda x: (-n_train[x], free[x]))
            j.launch(j.cname, g)
            launched[j.name] = j.cname
            print('[{:6.0f}s] launch {} gpu {}'.format(time.time() - t0, j.name, g), flush=True)
            count[j.kind] += 1
            if g is not None:
                free[g] -= MEM[j.kind]
                mem_est[g] += MEM[j.kind]
                n_train[g] += j.kind == 'ntf'
            time.sleep(2)
        status = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': time.time() - t0,
                  'done': sum(j.done() for j in jobs), 'total': len(jobs),
                  'by_kind': {k: [sum(j.done() for j in jobs if j.kind == k), sum(1 for j in jobs if j.kind == k)]
                              for k in PREFIX},
                  'running': sorted(live), 'gpu_free_gb': gpu_free_gb(), 'failed': failed}
        dst = os.path.join(HOST, 'results/constrained_movement/queue_status.json')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst + '.tmp', 'w') as fh:
            json.dump(status, fh, indent=1)
        os.replace(dst + '.tmp', dst)
        if all(j.done() or j.name in failed for j in jobs) and not live:
            print('all done; failed: {}'.format(failed), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
