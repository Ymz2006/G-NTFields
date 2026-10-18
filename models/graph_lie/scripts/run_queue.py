#!/usr/bin/env python3
"""Host-side job queue for models/graph_lie: trains runs in pytorchserver containers on all GPUs and
scores every run with the per-case MPPI evaluator (plan_eval_graph.py --kind graph_native).

    python3 models/graph_lie/scripts/run_queue.py --phase all --per-gpu 4 --eval-slots 3
    python3 models/graph_lie/scripts/run_queue.py --phase T --dry-run      # print the plan
    python3 models/graph_lie/scripts/run_queue.py --phase all --no-train --sampler lie --eval-slots 2
        # evaluation only, with the Lie MPPI sampler (scripts/plan_eval_lie.py), into results/graph_lie/plan_liemppi

Run names: <env>_<head>_<speed>_<grad>_s<seed>, e.g. T_gmetric_iso_lie_s0 (speed comb | iso | aniso).
Runs with speed 'combined' that models/graph_native already trained (bitwise-identical training,
EXPERIMENTS.md L0) are reused: their checkpoints are evaluated in place and existing result files
from results/graph_method_cusp/plan/GN_<run> are copied instead of recomputed.

Resumable: a run with train_summary.json is not retrained, a result JSON is not recomputed, and
containers still running from an earlier invocation (glie_g<gpu>_<run>, gleval_g<gpu>_<run>__<set>)
are adopted. Status: results/graph_lie/queue_status.json. Python 3 standard library only.
"""
import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time

HOST = '/media/jeffrey/G-Shape-NTFields'
LIE = '/media/jeffrey/Lie-Shape-NTFields'
IMAGE = 'pytorchserver:latest'
EXP = 'Experiments/graph_lie'
RES = 'results/graph_lie/plan'          # chart (official) MPPI; set to plan_liemppi by --sampler lie
GN_EXP = 'Experiments/graph_native'
GN_RES = 'results/graph_method_cusp/plan'
ENVS = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4', 'F': 'free_se3', 'F2': 'free_se3'}
EXTRA_ARGS = {'F2': '--weight none'}  # obstacle-free control with the unweighted Eikonal residual (EXPERIMENTS.md L5)
# (set, priority): standard sets first, rotation / control suites next, tight sets last
SETS = {'T': [('Tshape3d_env1', 1), ('Tshape3d_env1_bigangle', 2), ('Tshape3d_env1_rotheavy', 2),
              ('Tshape3d_env1_transonly', 2), ('Tshape3d_env1_tight_3d', 3)],
        'L': [('Lshape3d_env3', 1), ('Lshape3d_env3_tight_3d', 4)],   # 4: skipped at the default --max-priority 3
        'A': [('Ashape3d_env4', 1), ('Ashape3d_env4_tight_3d', 4)],   # (CPU budget; T's tight set covers the question)
        'F': [], 'F2': []}  # obstacle-free controls: scored against the exact geodesic by scripts/free_eval.py
SETS_LIE = {'T': [('Tshape3d_env1', 1), ('Tshape3d_env1_bigangle', 2), ('Tshape3d_env1_rotheavy', 2)],
            'L': [('Lshape3d_env3', 1)], 'A': [('Ashape3d_env4', 1)], 'F': [], 'F2': []}
EVAL_PREFIX = 'gleval'                   # container prefix; glelie for the Lie sampler
SAMPLER = 'chart'
HEADS = {'gmetric': ('graph_metric', []), 'trieq': ('graph_tri_metric', ['rot_mode=equivariant'])}
SPEED_TAG = {'combined': 'comb', 'iso': 'iso', 'aniso': 'aniso'}
# runs models/graph_native trained with the identical procedure: (env, grad) -> seeds
GN_DONE = {('T', 'chart'): [0, 1, 2], ('L', 'chart'): [0, 1], ('A', 'chart'): [0, 1], ('T', 'lie'): [0]}


def run_name(env, head, speed, grad, seed):
    return '{}_{}_{}_{}_s{}'.format(env, head, SPEED_TAG[speed], grad, seed)


def plan(phase):
    """Ordered list of runs; seeds outermost so complete (chart, lie) pairs finish first."""
    blocks = {'T': [('T', [0, 1, 2], ['iso', 'aniso', 'combined'])],
              'LA': [(e, [0, 1], ['iso', 'aniso', 'combined']) for e in ('L', 'A')],
              'LA3': [(e, [2], ['iso', 'aniso', 'combined']) for e in ('L', 'A')],
              'free': [('F', [0, 1, 2], ['iso'])],  # all speeds 1: the three formulations coincide
              'free2': [('F2', [0, 1, 2], ['iso'])],  # same, unweighted residual (see EXTRA_ARGS)
              'T5': [('T', [3, 4], ['iso', 'aniso', 'combined'], ['trieq'])]}  # triangle head: 5 seeds on T
    blocks['all'] = blocks['T'] + blocks['LA'] + blocks['LA3']
    if phase not in blocks:
        raise SystemExit('unknown phase {}; one of {}'.format(phase, sorted(blocks)))
    runs = []
    for env, seeds, speeds, *heads in blocks[phase]:
        for seed, speed, head, grad in itertools.product(seeds, speeds, heads[0] if heads else HEADS, ('chart', 'lie')):
            name = run_name(env, head, speed, grad, seed)
            reuse = speed == 'combined' and seed in GN_DONE.get((env, grad), [])
            gn = '{}_{}_{}_s{}'.format(env, head, grad, seed)
            runs.append(dict(env=env, head=head, speed=speed, grad=grad, seed=seed, name=name, train=not reuse,
                             ckpt=os.path.join(GN_EXP if reuse else EXP, gn if reuse else name, 'latest.pt'),
                             gn_res=os.path.join(GN_RES, 'GN_' + gn) if reuse else None))
    return runs


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def running_containers():
    return [n for n in sh("docker ps --format '{{.Names}}'").split() if n.startswith(('glie_', 'gleval_', 'glelie_'))]


def gpu_of(cname):
    return int(cname.split('_')[1][1:])


def docker_prefix(gpu, cname, workdir, extra_mounts='', threads=2):
    return ('docker run -d --rm --gpus \'"device={g}"\' -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp '
            '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} -v {host}:/workspace {mounts} -w {wd} '
            '--name {c} {img} ').format(g=gpu, uid=os.getuid(), gid=os.getgid(), th=threads, host=HOST,
                                         mounts=extra_mounts, wd=workdir, c=cname, img=IMAGE)


def train_done(r):
    if not r['train']:
        return os.path.exists(os.path.join(HOST, r['ckpt']))
    return os.path.exists(os.path.join(HOST, EXP, r['name'], 'train_summary.json'))


def complete_json(path):
    """Exists and ends with '}' (a result written while the disk filled up is truncated)."""
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def free_gb():
    return shutil.disk_usage(HOST).free / 1e9


def eval_done(r, s):
    out = os.path.join(HOST, RES, r['name'], s + '.json')
    if SAMPLER == 'chart' and not os.path.exists(out) and r['gn_res'] and \
            os.path.exists(os.path.join(HOST, r['gn_res'], s + '.json')):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(os.path.join(HOST, r['gn_res'], s + '.json'), out)  # same checkpoint, same evaluator
    return complete_json(out)


def launch_train(r, gpu, epochs):
    arch, sets = HEADS[r['head']]
    cname = 'glie_g{}_{}'.format(gpu, r['name'])
    args = ('python -u train/train_graph_lie.py --dataPath datasets/3dshape/data/{env} --name {n} --seed {s} '
            '--arch {a} --grad {g} --speed {sp} --epochs {ep}').format(env=ENVS[r['env']], n=r['name'], s=r['seed'],
                                                                     a=arch, g=r['grad'], sp=r['speed'], ep=epochs)
    if r['env'] in EXTRA_ARGS:
        args += ' ' + EXTRA_ARGS[r['env']]
    if sets:
        args += ' --set ' + ' '.join(sets)
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    sh(docker_prefix(gpu, cname, '/workspace') + 'bash -c "{} > {}/logs/{}.out 2>&1"'.format(args, EXP, r['name']))


def launch_eval(r, s, gpu, workers):
    cname = '{}_g{}_{}__{}'.format(EVAL_PREFIX, gpu, r['name'], s)
    out = '/workspace/{}/{}'.format(RES, r['name'])
    os.makedirs(os.path.join(HOST, RES, r['name']), exist_ok=True)
    script = ('models/graph_method_triangle/scripts/plan_eval_graph.py' if SAMPLER == 'chart'
              else 'models/graph_lie/scripts/plan_eval_lie.py --sampler lie')
    args = ('python -u /workspace/{sc} --kind graph_native '
            '--checkpoint /workspace/{ck} --dataPath testing_data/3dshape/{s} --out {o}/{s}.json '
            '--device cuda:0 --batch 125 --workers {w} --seed 0').format(sc=script, ck=r['ckpt'], s=s, o=out, w=workers)
    sh(docker_prefix(gpu, cname, '/lieshape', '-v {}:/lieshape:ro'.format(LIE), threads=1)
       + 'bash -c "{} > {}/{}.log 2>&1"'.format(args, out, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--per-gpu', type=int, default=4)
    ap.add_argument('--eval-slots', type=int, default=3)
    ap.add_argument('--eval-workers', type=int, default=6)
    ap.add_argument('--max-priority', type=int, default=3, help='evaluate sets up to this priority')
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-train', action='store_true', help='evaluate only (never launch training)')
    ap.add_argument('--min-free-gb', type=float, default=3.0, help='launch nothing while the disk has less free')
    ap.add_argument('--sampler', default='chart', choices=['chart', 'lie'])
    a = ap.parse_args()
    global RES, EVAL_PREFIX, SAMPLER, SETS
    if a.sampler == 'lie':
        RES, EVAL_PREFIX, SAMPLER, SETS = 'results/graph_lie/plan_liemppi', 'glelie', 'lie', SETS_LIE
    gpus = [int(g) for g in a.gpus.split(',')]
    runs = plan(a.phase)
    order = {r['name']: i for i, r in enumerate(runs)}
    if a.dry_run:
        for r in runs:
            print('{:28s} {:6s} {}'.format(r['name'], 'train' if r['train'] else 'reuse',
                                           'done' if train_done(r) else ''))
        print(len(runs), 'runs,', sum(r['train'] for r in runs), 'to train')
        return
    launched_train, launched_eval, failed = set(), set(), {}
    t0 = time.time()
    while True:
        live = running_containers()
        live_train = [c for c in live if c.startswith('glie_')]
        live_eval = [c for c in live if c.startswith(EVAL_PREFIX + '_')]
        live_names = {c.split('_', 2)[2] for c in live_train}
        live_evals = {c.split('_', 2)[2] for c in live_eval}
        for r in runs:
            if r['name'] in launched_train and r['name'] not in live_names and not train_done(r):
                failed.setdefault(r['name'], 'training exited without train_summary.json')
        for key in list(launched_eval):
            if key not in live_evals:
                rn, s = key.split('__')
                if not complete_json(os.path.join(HOST, RES, rn, s + '.json')):
                    failed.setdefault(key, 'evaluation exited without JSON')
                launched_eval.discard(key)
        pending_train = [] if a.no_train else [r for r in runs if r['train'] and not train_done(r)
                                               and r['name'] not in live_names and r['name'] not in failed]
        pending_eval = sorted(((p, order[r['name']], r, s) for r in runs if train_done(r)
                               for s, p in SETS[r['env']] if p <= a.max_priority and not eval_done(r, s)
                               and '{}__{}'.format(r['name'], s) not in live_evals
                               and '{}__{}'.format(r['name'], s) not in failed), key=lambda x: x[:2])
        low_disk = free_gb() < a.min_free_gb  # shared disk: never fill it (it happened once, EXPERIMENTS.md L4)
        if low_disk:
            print('[{:6.0f}s] only {:.1f} GB free: not launching'.format(time.time() - t0, free_gb()), flush=True)
        launch_train_list = [] if low_disk else pending_train
        launch_eval_list = [] if low_disk else pending_eval
        load = {g: sum(1 for c in live_train if gpu_of(c) == g) for g in gpus}
        while launch_train_list:
            g = min(gpus, key=lambda k: load[k])
            if load[g] >= a.per_gpu:
                break
            r = launch_train_list.pop(0)
            launch_train(r, g, a.epochs)
            launched_train.add(r['name'])
            load[g] += 1
            print('[{:6.0f}s] train {} on gpu {}'.format(time.time() - t0, r['name'], g), flush=True)
        eload = {g: sum(1 for c in live_eval if gpu_of(c) == g) for g in gpus}
        n_eval = len(live_eval)
        while launch_eval_list and n_eval < a.eval_slots:
            g = min(gpus, key=lambda k: (eload[k], load[k]))
            _, _, r, s = launch_eval_list.pop(0)
            launch_eval(r, s, g, a.eval_workers)
            launched_eval.add('{}__{}'.format(r['name'], s))
            eload[g] += 1
            n_eval += 1
            print('[{:6.0f}s] eval  {} / {} on gpu {}'.format(time.time() - t0, r['name'], s, g), flush=True)
        status = {'phase': a.phase, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'train_done': sum(train_done(r) for r in runs if r['train']),
                  'train_total': sum(r['train'] for r in runs),
                  'eval_done': sum(eval_done(r, s) for r in runs for s, p in SETS[r['env']] if p <= a.max_priority),
                  'eval_total': sum(1 for r in runs for s, p in SETS[r['env']] if p <= a.max_priority),
                  'running': sorted(running_containers()), 'failed': failed}
        os.makedirs(os.path.join(HOST, 'results/graph_lie'), exist_ok=True)
        status['free_gb'] = round(free_gb(), 2)
        try:
            dst = os.path.join(HOST, 'results/graph_lie/queue_status_{}_{}.json'.format(a.phase, a.sampler))
            with open(dst + '.tmp', 'w') as fh:
                json.dump(status, fh, indent=1)
            os.replace(dst + '.tmp', dst)
        except OSError as e:
            print('status not written: {}'.format(e), flush=True)
        mine = [c for c in running_containers() if c.startswith(('glie_', EVAL_PREFIX + '_'))]
        if not pending_train and not pending_eval and not mine:
            print('all done; failed: {}'.format(failed), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
