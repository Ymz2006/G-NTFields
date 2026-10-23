#!/usr/bin/env python3
"""Host-side queue of the path-constrained comparison (homotopy head vs the original metric NTFields), GPU-memory
aware, one GPU per job, resumable (a job is done when its output exists; running containers are adopted).

Cells (datasets/3dshape/data, models/path_constrained/scripts/preprocess_path.py): prism_<env>_<path> in the
arclength chart, <cell>_world the same poses in the world chart. Grid:

    main    prism_env{1,3,4}_a x seeds 0, 1, 2 x {homotopy (--lift min, 8000 epochs), NTFields (8000 epochs)}
    world   prism_env{1,3,4}_a_world x seed 0 x both models (the chart ablation)
    paths   prism_env{1,3,4}_b x seed 0 x both models (a second path per scene)

Test sets (make_path_tests.py, 1000 RRT-verified cases): <cell>_full and <cell>_seg for every arclength cell (the
world runs are scored on the same cases); the targeted full-path variants <cell>_{tight, bigrot, hard} for the main
cells, scored in the forward mode. Evaluations (plan_eval_path.py): homotopy {base, dual} x {free, forward},
NTFields base x {free, forward}, every run on both sets of its cell; the epoch-4000 checkpoints of both models with
the forward mode (dual / base).

    python3 models/path_constrained/scripts/run_path_queue.py --dry-run
    python3 models/path_constrained/scripts/run_path_queue.py --gpus 0,1,2,3

Containers: pcq_s (sets, CPU), pcq_t (trainings), pcq_e (evaluations). Status: results/path_constrained/queue_status.json.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HOST = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXPORT = '/media/4tbssd/jeffrey/export_metric'
IMAGE = 'pytorchserver:latest'
EXP = 'Experiments/path_constrained'
RES = 'results/path_constrained/plan'
SETS = 'testing_data/3dshape'
DATA = 'datasets/3dshape/data'
EPOCHS = 8000
MEM = {'set': 0.0, 'hom': 1.0, 'ntf': 3.3, 'eval': 1.5, 'pre': 4.0}
GPU_GB = 24.0
PREFIX = ('pcq_s', 'pcq_t', 'pcq_e', 'pcq_p')
TARGETED = ('tight', 'bigrot', 'hard')    # full-path variants (make_path_tests.py), forward mode, main cells
CODE = {'prism_env1_a': 'e1a', 'prism_env3_a': 'e3a', 'prism_env4_a': 'e4a',
        'prism_env1_b': 'e1b', 'prism_env3_b': 'e3b', 'prism_env4_b': 'e4b',
        'prism30_env1_a': 'e1s', 'prism30_env3_a': 'e3s', 'prism30_env4_a': 'e4s'}
STRESS = {'prism30_env1_a': 'env1_a', 'prism30_env3_a': 'env3_a', 'prism30_env4_a': 'env4_a'}   # h = 0.3, seed 0


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def containers():
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


def exists(*parts):
    return os.path.exists(os.path.join(HOST, *parts))


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def docker(cname, workdir, cmd, gpu=None, threads=2, extra=''):
    sh(('docker run -d --rm {g} --label gpu={gl} -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e NTF_NO_PLOT=1 '
        '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} {x} -v {host}:/workspace -v {exp}:/export:ro -w {wd} '
        '--name {c} {img} /bin/bash -c "{cmd}"').format(
        g='--gpus device={}'.format(gpu) if gpu is not None else '', gl='' if gpu is None else gpu, uid=os.getuid(),
        gid=os.getgid(), th=threads, x=extra, host=HOST, exp=EXPORT, wd=workdir, c=cname, img=IMAGE, cmd=cmd))


class Job:
    def __init__(self, name, kind, prio, done, launch, deps=()):
        self.name, self.kind, self.prio, self.done, self.launch, self.deps = name, kind, prio, done, launch, deps
        self.cname = {'set': 'pcq_s_', 'hom': 'pcq_t_', 'ntf': 'pcq_t_', 'eval': 'pcq_e_', 'pre': 'pcq_p_'}[kind] + \
            name.split(':', 1)[1].replace(':', '__')

    def ready(self, done_of):
        return all(done_of(d) for d in self.deps)


def set_done(s):
    return exists(SETS, s, 'cases.npz') and exists(SETS, s, 'verification.json')


def train_done(name):
    return exists(EXP, name, 'latest.pt') and exists(EXP, name, 'train_summary.json')


def ckpt4k(name, kind):
    if kind == 'ntf':
        return '/workspace/{}/{}/epoch4000.pt'.format(EXP, name)
    c = sorted(glob.glob(os.path.join(HOST, EXP, name, 'Model_Epoch_04000_*.pt')))
    return '/workspace/{}/{}/{}'.format(EXP, name, os.path.basename(c[0])) if c else None


def eval_out(run, s, planner, mode):
    return os.path.join(HOST, RES, run, '{}__{}_{}.json'.format(s, planner, mode))


def launch_pre(cname, cell, path, gpu):
    os.makedirs(os.path.join(HOST, DATA, 'logs'), exist_ok=True)
    cmd = ('python -u models/path_constrained/scripts/preprocess_path.py --path datasets/3dshape/paths/{p}.npz '
           '--out {d}/{c} --length 0.3 --no-world --num_samples 800000 --batch 16384 > {d}/logs/pre_{c}.out 2>&1').format(
        p=path, d=DATA, c=cell)
    docker(cname, '/workspace', cmd, gpu=gpu)


def launch_set(cname, cell, variant, workers):
    os.makedirs(os.path.join(HOST, SETS, 'logs'), exist_ok=True)
    cmd = ('python -u models/path_constrained/scripts/make_path_tests.py --cell {d}/{c} --variant {v} --n 1000 '
           '--workers {w} --out {s} > {s}/logs/{c}_{v}.out 2>&1').format(d=DATA, c=cell, v=variant, w=workers, s=SETS)
    docker(cname, '/workspace', cmd, threads=1)


def launch_hom(cname, name, cell, seed, gpu):
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    cmd = ('python -u train/train_homotopy_path.py --dataPath {d}/{c} --name {n} --seed {s} --epochs {ep} --lift min '
           '> {e}/logs/{n}.out 2>&1').format(d=DATA, c=cell, n=name, s=seed, ep=EPOCHS, e=EXP)
    docker(cname, '/workspace', cmd, gpu=gpu)


def launch_ntf(cname, name, cell, seed, gpu):
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/train/train_metric_path.py --dataPath /workspace/{d}/{c} --name {n} --seed {s} '
           '--epochs {ep} --bundle /export --out_root /workspace/{e} > /workspace/{e}/logs/{n}.out 2>&1').format(
        d=DATA, c=cell, n=name, s=seed, ep=EPOCHS, e=EXP)
    docker(cname, '/export', cmd, gpu=gpu)


def launch_eval(cname, run, kind, ckpt, s, planner, mode, chart, gpu):
    out = eval_out(run, s, planner, mode)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rel = os.path.relpath(out, HOST)
    cmd = ('python -u models/path_constrained/scripts/plan_eval_path.py --kind {k} --checkpoint {ck} '
           '--dataPath {sets}/{s} --planner {p} --mode {m} --chart {ch} --out {o} --device cuda:0 --save-paths '
           '> {lo} 2>&1').format(k=kind, ck=ckpt, sets=SETS, s=s, p=planner, m=mode, ch=chart, o=rel,
                                 lo=rel[:-5] + '.log')
    docker(cname, '/workspace', cmd, gpu=gpu, threads=2)


def sets_of(cell):
    if cell in STRESS:
        return ('full', 'hard')
    return ('full', 'seg') + (TARGETED if cell.endswith('_a') else ())


def runs_of(grid):
    """(run name, kind, training cell, seed, test cell, chart, priority tier)."""
    out = []
    if grid in ('main', 'all'):
        for seed in (0, 1, 2):
            for cell in ('prism_env1_a', 'prism_env3_a', 'prism_env4_a'):
                for kind in ('hom', 'ntf'):
                    out.append(('{}_{}_s{}'.format(CODE[cell], kind, seed), kind, cell, seed, cell, 'arclength',
                                2 + 2 * seed))
    if grid in ('stress', 'all'):
        for cell in STRESS:
            for kind in ('hom', 'ntf'):
                out.append(('{}_{}_s0'.format(CODE[cell], kind), kind, cell, 0, cell, 'arclength', 5))
    if grid in ('extra', 'all'):
        for cell in ('prism_env1_a', 'prism_env3_a', 'prism_env4_a'):
            for kind in ('hom', 'ntf'):
                out.append(('{}w_{}_s0'.format(CODE[cell], kind), kind, cell + '_world', 0, cell, 'world', 3))
        for cell in ('prism_env1_b', 'prism_env3_b', 'prism_env4_b'):
            for kind in ('hom', 'ntf'):
                out.append(('{}_{}_s0'.format(CODE[cell], kind), kind, cell, 0, cell, 'arclength', 3.5))
    return out


def build_jobs(a):
    J = []
    add = lambda *x, **kw: J.append(Job(*x, **kw))  # noqa: E731
    if a.grid in ('stress', 'all'):
        for cell, pth in STRESS.items():
            add('pre:' + cell, 'pre', 0.2, (lambda c=cell: exists(DATA, c, 'meta.json')),
                (lambda cn, g, c=cell, pp=pth: launch_pre(cn, c, pp, g)))
    ready = lambda c: exists(DATA, c, 'meta.json')  # noqa: E731  (cells still being preprocessed wait)
    runs = [r for r in runs_of(a.grid) if ready(r[2]) and ready(r[4])]
    test_cells = sorted({r[4] for r in runs})
    for cell in test_cells:
        for v in sets_of(cell):
            s = '{}_{}'.format(cell, v)
            add('set:' + s, 'set', 0 if cell.endswith('_a') else 0.5, (lambda ss=s: set_done(ss)),
                (lambda cn, g, c=cell, vv=v: launch_set(cn, c, vv, a.set_workers)))
    for name, kind, cell, seed, tcell, chart, tier in runs:
        if kind == 'hom':
            add('hom:' + name, 'hom', tier, (lambda n=name: train_done(n)),
                (lambda cn, g, n=name, c=cell, ss=seed: launch_hom(cn, n, c, ss, g)))
            evals = [(name, 'graph', '/workspace/{}/{}/latest.pt'.format(EXP, name), p, m)
                     for p in ('base', 'dual') for m in ('free', 'forward')]
            k4 = ckpt4k(name, 'hom')
            if k4:
                evals.append((name.replace('_hom_', '_hom4k_'), 'graph', k4, 'dual', 'forward'))
        else:
            add('ntf:' + name, 'ntf', tier + 0.1, (lambda n=name: train_done(n)),
                (lambda cn, g, n=name, c=cell, ss=seed: launch_ntf(cn, n, c, ss, g)))
            evals = [(name, 'metric', '/workspace/{}/{}/latest.pt'.format(EXP, name), 'base', m)
                     for m in ('free', 'forward')]
            evals.append((name.replace('_ntf_', '_ntf4k_'), 'metric', ckpt4k(name, 'ntf'), 'base', 'forward'))
        for v in sets_of(tcell):
            s = '{}_{}'.format(tcell, v)
            for run, k, ck, p, m in evals:
                if v in TARGETED and (m != 'forward' or '4k' in run or run[3] == 'w'):
                    continue
                if tcell in STRESS and m == 'free' and v != 'full':
                    continue
                add('eval:{}:{}:{}_{}'.format(run, s, p, m), 'eval', 1 + 0.01 * tier,
                    (lambda r=run, ss=s, pp=p, mm=m: complete_json(eval_out(r, ss, pp, mm))),
                    (lambda cn, g, r=run, kk=k, c=ck, ss=s, pp=p, mm=m, ch=chart: launch_eval(cn, r, kk, c, ss, pp, mm, ch, g)),
                    deps=('{}:{}'.format(kind, name), 'set:' + s))
    return sorted(J, key=lambda j: (j.prio, j.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='all', choices=('main', 'extra', 'stress', 'all'))
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--reserve', type=float, default=2.0)
    ap.add_argument('--max-train-per-gpu', type=int, default=7)
    ap.add_argument('--max-eval', type=int, default=4)
    ap.add_argument('--max-set', type=int, default=2)
    ap.add_argument('--set-workers', type=int, default=12)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--attempts', type=int, default=3)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(',')]
    if a.dry_run:
        jobs = build_jobs(a)
        for j in jobs:
            print('{:6.3f} {:5s} {:70s} {}'.format(j.prio, j.kind, j.name, 'done' if j.done() else ''))
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
        n_eval = n_set = 0
        for c, g in live.items():
            k = kinds.get(c, 'eval' if c.startswith('pcq_e') else ('set' if c.startswith('pcq_s') else
                                                               ('pre' if c.startswith('pcq_p') else 'hom')))
            if g in mem_est:
                mem_est[g] += MEM[k]
                n_train[g] += k in ('hom', 'ntf')
            n_eval += k == 'eval'
            n_set += k == 'set'
        done_of = lambda nm: by_name[nm].done() if nm in by_name else True  # noqa: E731
        pend = [j for j in jobs if not j.done() and j.cname not in live and j.name not in failed and j.ready(done_of)]
        free = {g: min(f, GPU_GB - mem_est.get(g, 0.0)) for g, f in gpu_free_gb().items() if g in gpus}
        for j in pend:
            g = None
            if j.kind == 'set':
                if n_set >= a.max_set:
                    continue
            else:
                if j.kind == 'eval' and n_eval >= a.max_eval:
                    continue
                need = MEM[j.kind]
                ok = [x for x in gpus if free.get(x, 0) >= need + a.reserve
                      and not (j.kind in ('hom', 'ntf') and n_train[x] >= a.max_train_per_gpu)]
                if not ok:
                    continue
                g = max(ok, key=lambda x: (-n_train[x], free[x]))
            j.launch(j.cname, g)
            launched[j.name] = j.cname
            print('[{:6.0f}s] launch {} gpu {}'.format(time.time() - t0, j.name, g), flush=True)
            if j.kind == 'set':
                n_set += 1
            else:
                free[g] -= MEM[j.kind]
                mem_est[g] += MEM[j.kind]
                n_train[g] += j.kind in ('hom', 'ntf')
                n_eval += j.kind == 'eval'
            time.sleep(2)
        status = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': time.time() - t0,
                  'done': sum(j.done() for j in jobs), 'total': len(jobs),
                  'by_kind': {k: [sum(j.done() for j in jobs if j.kind == k), sum(1 for j in jobs if j.kind == k)]
                              for k in ('pre', 'set', 'hom', 'ntf', 'eval')},
                  'running': sorted(live), 'gpu_free_gb': gpu_free_gb(), 'failed': failed}
        dst = os.path.join(HOST, 'results/path_constrained/queue_status.json')
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
