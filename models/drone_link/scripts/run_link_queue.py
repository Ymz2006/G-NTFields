#!/usr/bin/env python3
"""Host-side queue of the drone-pair link study (models/drone_link), GPU-memory aware, one GPU per job, resumable (a
job is done when its output exists; running containers are adopted on restart).

Cells link_<scene> (scene in forest, windows, corridor, scaffold; 800k pairs, preprocess_link.py). Per scene:

    sets        link_<scene>_std and link_<scene>_cross (make_link_tests_prm.py on a GPU: 1000 cases each, certified by
                a roadmap; the CPU RRT of make_link_tests.py crawled on the shared machine)
    ntf         the full metric NTFields (TD and normal-alignment terms 1e-3, as the camera test's mbtd), 7-d chart,
                8000 epochs, seeds 0 / 1 / 2 (train_metric_link.py)
    ntfb        the bundle's default metric NTFields (TD and normal terms off), seed 0 (ablation)
    ntfsb       the full NTFields trained on link_<scene>_sb, the cell whose x0 band is on the scene clearance only
                (no x0 bound by a length limit; preprocess_link.py --band scene), seed 0 (ablation)
    ntflr       the full NTFields trained on link_<scene>_lr, the cell where half of the x1 are uniform poses in the
                room (long-range pairs; preprocess_link.py --x1-uniform 0.5), seed 0 (ablation)
    ntflr20     the same with a fifth of the x1 uniform in the room (link_<scene>_lr20, --x1-uniform 0.2), seed 0;
                seeds 1 / 2 added once seed 0 lifted the crossings (30 -> 47 % mean)
    eval        plan_eval_link.py of every run on both sets of its scene (final checkpoint; ntf also epoch 4000)
    baselines   baseline_link.py on every set, GPU (straight 7-d motion, independent drones, the certificate)

    python3 models/drone_link/scripts/run_link_queue.py --dry-run
    python3 models/drone_link/scripts/run_link_queue.py --gpus 0,1,2,3

Containers dlq_p (preprocessing), dlq_s (sets), dlq_b (baselines), dlq_t (trainings), dlq_e (evaluations).
Status: results/drone_link/queue_status.json.
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
EXP = 'Experiments/drone_link'
RES = 'results/drone_link'
SETS = 'testing_data/3dshape'
DATA = 'datasets/3dshape/data'
SCENES = ('forest', 'windows', 'corridor', 'scaffold')
VARIANTS = ('std', 'cross')
EPOCHS = 8000
MEM = {'pre': 1.0, 'set': 1.5, 'base': 1.5, 'ntf': 1.5, 'eval': 2.0}
GPU_GB = 24.0
PREFIX = ('dlq_p_', 'dlq_s_', 'dlq_b_', 'dlq_t_', 'dlq_e_')


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


def docker(cname, workdir, cmd, gpu=None, threads=2):
    sh(('docker run -d --rm {g} --label gpu={gl} -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp -e NTF_NO_PLOT=1 '
        '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} -v {host}:/workspace -v {exp}:/export:ro -w {wd} '
        '--name {c} {img} /bin/bash -c "{cmd}"').format(
        g='--gpus device={}'.format(gpu) if gpu is not None else '', gl='' if gpu is None else gpu, uid=os.getuid(),
        gid=os.getgid(), th=threads, host=HOST, exp=EXPORT, wd=workdir, c=cname, img=IMAGE, cmd=cmd))


class Job:
    def __init__(self, name, kind, prio, done, launch, deps=()):
        self.name, self.kind, self.prio, self.done, self.launch, self.deps = name, kind, prio, done, launch, deps
        self.cname = {'pre': 'dlq_p_', 'set': 'dlq_s_', 'base': 'dlq_b_', 'ntf': 'dlq_t_', 'eval': 'dlq_e_'}[kind] + \
            name.split(':', 1)[1].replace(':', '__')

    def ready(self, done_of):
        return all(done_of(d) for d in self.deps)


def cell_done(cell):
    return exists(DATA, cell, 'meta.json')


def set_done(s):
    return exists(SETS, s, 'verification.json') and exists(SETS, s, 'sampled_points.npy')


def train_done(name):
    return exists(EXP, name, 'latest.pt') and exists(EXP, name, 'train_summary.json')


def launch_pre(cname, cell, gpu):
    os.makedirs(os.path.join(HOST, DATA, 'logs'), exist_ok=True)
    scene = cell.split('_')[1]
    band = {'sb': '--band scene', 'lr': '--x1-uniform 0.5', 'lr20': '--x1-uniform 0.2'}.get(cell.rsplit('_', 1)[1], '')
    cmd = ('python -u models/drone_link/scripts/preprocess_link.py --scene {s} --out {d}/{c} --num_samples 800000 '
           '--batch 16384 --chunk 2048 {b} '
           '> {d}/logs/pre_{c}.out 2>&1').format(s=scene, d=DATA, c=cell, b=band)
    docker(cname, '/workspace', cmd, gpu=gpu)


def launch_set(cname, scene, variant, gpu):
    os.makedirs(os.path.join(HOST, SETS, 'logs'), exist_ok=True)
    cmd = ('python -u models/drone_link/scripts/make_link_tests_prm.py --cell {d}/link_{sc} --variant {v} --n 1000 '
           '--out {s} --device cuda:0 > {s}/logs/link_{sc}_{v}.out 2>&1').format(d=DATA, sc=scene, v=variant, s=SETS)
    docker(cname, '/workspace', cmd, gpu=gpu, threads=2)


def launch_base(cname, s, gpu):
    out = '{}/baselines/{}.json'.format(RES, s)
    os.makedirs(os.path.join(HOST, RES, 'baselines'), exist_ok=True)
    cmd = ('python -u models/drone_link/scripts/baseline_link.py --dataPath {sets}/{s} --out {o} --device cuda:0 '
           '--save-paths > {lo} 2>&1').format(sets=SETS, s=s, o=out, lo=out[:-5] + '.log')
    docker(cname, '/workspace', cmd, gpu=gpu, threads=2)


def launch_ntf(cname, name, cell, seed, gpu, extra):
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    cmd = ('python -u /workspace/train/train_metric_link.py --dataPath /workspace/{d}/{c} --name {n} --seed {s} '
           '--epochs {ep} --bundle /export --out_root /workspace/{e} {x} > /workspace/{e}/logs/{n}.out 2>&1').format(
        d=DATA, c=cell, n=name, s=seed, ep=EPOCHS, e=EXP, x=extra)
    docker(cname, '/export', cmd, gpu=gpu)


def eval_out(run, s):
    return os.path.join(HOST, RES, 'plan', run, s + '.json')


def launch_eval(cname, run, ckpt, s, gpu):
    out = eval_out(run, s)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rel = os.path.relpath(out, HOST)
    cmd = ('python -u models/drone_link/scripts/plan_eval_link.py --checkpoint {ck} --dataPath {sets}/{s} --out {o} '
           '--device cuda:0 --save-paths > {lo} 2>&1').format(ck=ckpt, sets=SETS, s=s, o=rel, lo=rel[:-5] + '.log')
    docker(cname, '/workspace', cmd, gpu=gpu, threads=2)


def runs():
    """(run name, scene, training cell, seed, trainer flags, tier, keep the 4000-epoch evaluation)."""
    out = []
    full = '--td-weight 1e-3 --normal-weight 1e-3 --keep 4000'
    for seed in (0, 1, 2):
        for sc in SCENES:
            out.append(('{}_ntf_s{}'.format(sc, seed), sc, 'link_' + sc, seed, full, 2 + seed, True))
    for sc in SCENES:
        out.append(('{}_ntfb_s0'.format(sc), sc, 'link_' + sc, 0, '--keep 4000', 2.5, False))
        out.append(('{}_ntfsb_s0'.format(sc), sc, 'link_{}_sb'.format(sc), 0, full, 4.5, False))
        out.append(('{}_ntflr_s0'.format(sc), sc, 'link_{}_lr'.format(sc), 0, full, 4.2, False))
        out.append(('{}_ntflr20_s0'.format(sc), sc, 'link_{}_lr20'.format(sc), 0, full, 4.3, False))
    for seed in (1, 2):
        for sc in SCENES:
            out.append(('{}_ntflr20_s{}'.format(sc, seed), sc, 'link_{}_lr20'.format(sc), seed, full, 4.3 + 0.1 * seed, False))
    return out


def build_jobs(a):
    J = []
    add = lambda *x, **kw: J.append(Job(*x, **kw))  # noqa: E731
    for sc in SCENES:
        for cell in ('link_' + sc, 'link_{}_sb'.format(sc), 'link_{}_lr'.format(sc), 'link_{}_lr20'.format(sc)):
            add('pre:' + cell, 'pre', 0, (lambda c=cell: cell_done(c)), (lambda cn, g, c=cell: launch_pre(cn, c, g)))
        for v in VARIANTS:
            s = 'link_{}_{}'.format(sc, v)
            add('set:' + s, 'set', 0.5, (lambda ss=s: set_done(ss)),
                (lambda cn, g, c=sc, vv=v: launch_set(cn, c, vv, g)), deps=('pre:link_' + sc,))
            add('base:' + s, 'base', 1.5, (lambda ss=s: complete_json(os.path.join(HOST, RES, 'baselines', ss + '.json'))),
                (lambda cn, g, ss=s: launch_base(cn, ss, g)), deps=('set:' + s,))
    for name, sc, cell, seed, flags, tier, k4 in runs():
        # the scene-band ablation waits for the long-range one to start (it matters less, and slots are scarce)
        after = ('started:{}_ntflr_s0'.format(sc),) if '_ntfsb_' in name else ()
        add('ntf:' + name, 'ntf', tier, (lambda n=name: train_done(n)),
            (lambda cn, g, n=name, c=cell, ss=seed, x=flags: launch_ntf(cn, n, c, ss, g, x)), deps=('pre:' + cell,) + after)
        evals = [(name, '/workspace/{}/{}/latest.pt'.format(EXP, name))]
        if k4:
            evals.append((name.replace('_ntf_', '_ntf4k_'), '/workspace/{}/{}/epoch4000.pt'.format(EXP, name)))
        for v in VARIANTS:
            s = 'link_{}_{}'.format(sc, v)
            for run, ck in evals:
                add('eval:{}:{}'.format(run, s), 'eval', 1 + 0.01 * tier, (lambda r=run, ss=s: complete_json(eval_out(r, ss))),
                    (lambda cn, g, r=run, c=ck, ss=s: launch_eval(cn, r, c, ss, g)), deps=('ntf:' + name, 'set:' + s))
    return sorted(J, key=lambda j: (j.prio, j.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--reserve', type=float, default=1.0)
    ap.add_argument('--max-train-per-gpu', type=int, default=5)
    ap.add_argument('--max-eval', type=int, default=4)
    ap.add_argument('--max-aux', type=int, default=4, help='set and baseline jobs at a time')
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
        n_eval = n_aux = 0
        for c, g in live.items():
            k = kinds.get(c, {'dlq_p_': 'pre', 'dlq_s_': 'set', 'dlq_b_': 'base', 'dlq_t_': 'ntf', 'dlq_e_': 'eval'}[c[:6]])
            if g in mem_est:
                mem_est[g] += MEM[k]
                n_train[g] += k == 'ntf'
            n_eval += k == 'eval'
            n_aux += k in ('set', 'base')
        def done_of(nm):
            if nm.startswith('started:'):     # a training that is running or finished
                r = nm.split(':', 1)[1]
                return train_done(r) or ('dlq_t_' + r) in live or exists(EXP, r)
            return by_name[nm].done() if nm in by_name else True
        pend = [j for j in jobs if not j.done() and j.cname not in live and j.name not in failed and j.ready(done_of)]
        free = {g: min(f, GPU_GB - mem_est.get(g, 0.0)) for g, f in gpu_free_gb().items() if g in gpus}
        for j in pend:
            if j.kind in ('set', 'base') and n_aux >= a.max_aux:
                continue
            if j.kind == 'eval' and n_eval >= a.max_eval:
                continue
            ok = [x for x in gpus if free.get(x, 0) >= MEM[j.kind] + a.reserve
                  and not (j.kind == 'ntf' and n_train[x] >= a.max_train_per_gpu)]
            if not ok:
                continue
            g = max(ok, key=lambda x: (-n_train[x], free[x]))
            j.launch(j.cname, g)
            launched[j.name] = j.cname
            print('[{:6.0f}s] launch {} gpu {}'.format(time.time() - t0, j.name, g), flush=True)
            free[g] -= MEM[j.kind]
            mem_est[g] += MEM[j.kind]
            n_train[g] += j.kind == 'ntf'
            n_eval += j.kind == 'eval'
            n_aux += j.kind in ('set', 'base')
            time.sleep(2)
        status = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': time.time() - t0,
                  'done': sum(j.done() for j in jobs), 'total': len(jobs),
                  'by_kind': {k: [sum(j.done() for j in jobs if j.kind == k), sum(1 for j in jobs if j.kind == k)]
                              for k in ('pre', 'set', 'base', 'ntf', 'eval')},
                  'running': sorted(live), 'gpu_free_gb': gpu_free_gb(), 'failed': failed}
        dst = os.path.join(HOST, RES, 'queue_status.json')
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
