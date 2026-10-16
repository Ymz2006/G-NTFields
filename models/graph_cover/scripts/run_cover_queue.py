#!/usr/bin/env python3
"""Host-side queue for the cover-quotient decoder campaign: trains runs in pytorchserver containers on the GPUs
and scores each finished run with the per-case MPPI evaluator (plan_eval_graph.py --kind graph_native).

    python3 models/graph_cover/scripts/run_cover_queue.py --phase explore --gpus 0,1,2 --per-gpu 2
    python3 models/graph_cover/scripts/run_cover_queue.py --phase explore --dry-run

Adapted from models/graph_branch/scripts/run_branch_queue.py (same guards: GPU memory, disk, resumable; container
prefixes gct_ / gce_ so the two campaigns never adopt each other's containers). Status:
results/graph_cover/queue_status_<phase>.json. Python 3 standard library only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HOST = '/media/jeffrey/G-Shape-NTFields'
LIE = '/media/jeffrey/Lie-Shape-NTFields'
IMAGE = 'pytorchserver:latest'
EXP = 'Experiments/graph_cover'
RES = 'results/graph_cover/plan'
SCENES = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
SETS = {'T': [('Tshape3d_env1', 1), ('Tshape3d_env1_bigangle', 2), ('Tshape3d_env1_rotheavy', 2),
              ('Tshape3d_env1_transonly', 3), ('Tshape3d_env1_tight_3d', 3)],
        'L': [('Lshape3d_env3', 1)], 'A': [('Ashape3d_env4', 1)]}

# head variants: tag -> (arch, --set options)
LP = ["res_kind='lp'", 'res_scale=0.2']
VARIANTS = {
    'lp2': ('graph_cover', ['n_cut=2'] + LP),                        # the proposal: 2 learned covers + q routes
    'lp0': ('graph_cover', ['n_cut=0'] + LP),                        # q routes only (no learned branch points)
    'lp4': ('graph_cover', ['n_cut=4'] + LP),                        # 4 learned covers (16 group elements)
    'eu2': ('graph_cover', ['n_cut=2', "res_kind='euclid'", 'res_scale=0.05']),
    'lse2': ('graph_cover', ['n_cut=2', "res_kind='lse'", 'res_scale=0.2']),   # non-metric ablation
    'lp2nf': ('graph_cover', ['n_cut=2', 'tfree=False'] + LP),      # no free-space lower bound
    'lp2soft': ('graph_cover', ['n_cut=2', 'soft_tau=0.02'] + LP),  # soft minimum over the learned cuts (width 0.02)
    'gmetric': ('graph_metric', []),                                 # reference head (graph_lie runs exist)
    'lp2p2': ('graph_cover', ['n_cut=2', "res_kind='lp'", 'lp_p=2.0', 'res_scale=0.2']),   # group-Euclidean seminorm
    'eu0': ('graph_cover', ['n_cut=0', "res_kind='euclid'", 'res_scale=0.05']),           # euclid, q routes only
    'eu2w': ('graph_cover', ['n_cut=2', "res_kind='euclid'", 'res_scale=0.05', 'me=128', 'mq=32', 'mo=32']),  # wider
    'eu2r': ('graph_cover', ['n_cut=2', "res_kind='euclid'", 'res_scale=0.1']),           # stronger residual init
    'eu2c': ('graph_cover', ['n_cut=2', "res_kind='euclid'", 'res_scale=0.05', 'conf=True']),   # conformal cone (non-metric ablation)
    'eu2L': ('graph_cover', ['n_cut=2', "res_kind='euclid'", 'res_scale=0.05', 'lip_cap=4.0']),  # 4x Lipschitz capacity
    'lp2p2c': ('graph_cover', ['n_cut=2', "res_kind='lp'", 'lp_p=2.0', 'res_scale=0.2', 'conf=True']),
}
PHASES = {
    'explore': [('T', 0, v) for v in ('lp2', 'lp0', 'eu2', 'lse2', 'lp2nf', 'lp4', 'lp2soft')],
    'explore2': [('T', 0, v) for v in ('lp2p2', 'eu0', 'eu2w', 'eu2r')] + [('T', s, 'eu2') for s in (1, 2)],
    'explore3': [('T', 0, v) for v in ('eu2c', 'eu2L', 'lp2p2c')],
    'seeds': [('T', s, v) for s in (1, 2) for v in ('lp2', 'lse2')],
    'final': [('T', s, 'eu2c') for s in (1, 2)] + [(sc, 0, v) for sc in ('L', 'A') for v in ('eu2', 'eu2c')],
    'scenes': [(sc, s, v) for sc in ('L', 'A') for s in (0, 1) for v in ('lp2', 'lse2')],
}


def runs_of(phase):
    out = []
    for item in PHASES[phase]:
        scene, seed, tag = item[:3]
        speed = item[3] if len(item) > 3 else 'iso'
        arch, sets = VARIANTS[tag]
        name = '{}_{}_s{}'.format(scene, tag, seed) if speed == 'iso' else '{}_{}_{}_s{}'.format(scene, tag, speed, seed)
        out.append(dict(name=name, scene=scene, seed=seed, tag=tag, arch=arch, sets=sets, speed=speed,
                        ckpt=os.path.join(EXP, name, 'latest.pt')))
    return out


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def containers():
    return [n for n in sh("docker ps --format '{{.Names}}'").split() if n.startswith(('gct_', 'gce_'))]


def gpu_free_gb():
    out = sh('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits')
    free = {}
    for line in out.strip().splitlines():
        i, used, tot = [int(x) for x in line.split(',')]
        free[i] = (tot - used) / 1024.0
    return free


def disk_free_gb():
    return shutil.disk_usage(HOST).free / 1e9


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def train_done(r):
    return os.path.exists(os.path.join(HOST, EXP, r['name'], 'train_summary.json'))


def eval_done(r, s):
    return complete_json(os.path.join(HOST, RES, r['name'], s + '.json'))


def docker_prefix(gpu, cname, workdir, mounts='', threads=2):
    return ('docker run -d --rm --gpus \'"device={g}"\' -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp '
            '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} -v {host}:/workspace {m} -w {wd} --name {c} {img} '
            ).format(g=gpu, uid=os.getuid(), gid=os.getgid(), th=threads, host=HOST, m=mounts, wd=workdir, c=cname,
                     img=IMAGE)


def launch_train(r, gpu, epochs):
    cname = 'gct_g{}_{}'.format(gpu, r['name'])
    args = ('python -u train/train_graph_cover.py --dataPath datasets/3dshape/{sc} --name {n} --seed {s} --arch {a} '
            '--grad lie --speed {sp} --epochs {ep}').format(
        sc=SCENES[r['scene']], n=r['name'], s=r['seed'], a=r['arch'], sp=r['speed'], ep=epochs)
    if r['sets']:
        args += ' --set ' + ' '.join(r['sets'])
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    sh(docker_prefix(gpu, cname, '/workspace') + '/bin/bash -c "{} > {}/logs/{}.out 2>&1"'.format(
        args.replace('"', '\\"'), EXP, r['name']))


def launch_eval(r, s, gpu, workers):
    cname = 'gce_g{}_{}__{}'.format(gpu, r['name'], s)
    out = '/workspace/{}/{}'.format(RES, r['name'])
    os.makedirs(os.path.join(HOST, RES, r['name']), exist_ok=True)
    args = ('python -u /workspace/models/graph_method_triangle/scripts/plan_eval_graph.py --kind graph_native '
            '--checkpoint /workspace/{ck} --dataPath testing_data/3dshape/{s} --out {o}/{s}.json '
            '--device cuda:0 --batch 125 --workers {w} --seed 0').format(ck=r['ckpt'], s=s, o=out, w=workers)
    sh(docker_prefix(gpu, cname, '/lieshape', '-v {}:/lieshape:ro'.format(LIE), threads=1)
       + '/bin/bash -c "{} > {}/{}.log 2>&1"'.format(args, out, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--per-gpu', type=int, default=2, help='max training containers per GPU')
    ap.add_argument('--eval-slots', type=int, default=2, help='max evaluation containers in total')
    ap.add_argument('--eval-workers', type=int, default=6)
    ap.add_argument('--max-priority', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--min-free-gb', type=float, default=4.0, help='disk: launch nothing below this')
    ap.add_argument('--min-free-mem-gb', type=float, default=4.0, help='GPU memory needed to launch a training run')
    ap.add_argument('--eval-min-free-mem-gb', type=float, default=6.0, help='GPU memory needed to launch an eval')
    ap.add_argument('--no-train', action='store_true')
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    phases = a.phase.split(',')
    runs = [r for p in phases for r in runs_of(p)]
    order = {r['name']: i for i, r in enumerate(runs)}
    gpus = [int(g) for g in a.gpus.split(',')]
    if a.dry_run:
        for r in runs:
            print('{:24s} {:18s} {:50s} {}'.format(r['name'], r['arch'], ' '.join(r['sets']),
                                                   'trained' if train_done(r) else ''))
        return
    launched_t, launched_e, failed = set(), set(), {}
    t0 = time.time()
    while True:
        live = containers()
        live_t = {c.split('_', 2)[2]: c for c in live if c.startswith('gct_')}
        live_e = {c.split('_', 2)[2]: c for c in live if c.startswith('gce_')}
        for n in list(launched_t):
            if n not in live_t:
                r = next(x for x in runs if x['name'] == n)
                if not train_done(r):
                    failed.setdefault(n, 'training exited without train_summary.json')
                launched_t.discard(n)
        for key in list(launched_e):
            if key not in live_e:
                rn, s = key.split('__')
                if not complete_json(os.path.join(HOST, RES, rn, s + '.json')):
                    failed.setdefault(key, 'evaluation exited without JSON')
                launched_e.discard(key)
        pend_t = [] if a.no_train else [r for r in runs if not train_done(r) and r['name'] not in live_t
                                       and r['name'] not in failed]
        pend_e = [] if a.no_eval else sorted(
            ((p, order[r['name']], r, s) for r in runs if train_done(r) for s, p in SETS[r['scene']]
             if p <= a.max_priority and not eval_done(r, s) and '{}__{}'.format(r['name'], s) not in live_e
             and '{}__{}'.format(r['name'], s) not in failed), key=lambda x: x[:2])
        low_disk = disk_free_gb() < a.min_free_gb
        if low_disk:
            print('[{:6.0f}s] disk {:.1f} GB free: not launching'.format(time.time() - t0, disk_free_gb()), flush=True)
        free = gpu_free_gb()
        tload = {g: sum(1 for c in live_t.values() if int(c.split('_')[1][1:]) == g) for g in gpus}
        if not low_disk:
            for r in pend_t:
                cand = [g for g in gpus if tload[g] < a.per_gpu and free.get(g, 0) >= a.min_free_mem_gb]
                if not cand:
                    break
                g = max(cand, key=lambda k: (free[k] - 2.0 * tload[k]))
                launch_train(r, g, a.epochs)
                launched_t.add(r['name'])
                tload[g] += 1
                free[g] -= 2.0
                print('[{:6.0f}s] train {} on gpu {}'.format(time.time() - t0, r['name'], g), flush=True)
            n_e = len(live_e)
            for _, _, r, s in pend_e:
                if n_e >= a.eval_slots:
                    break
                cand = [g for g in gpus if free.get(g, 0) >= a.eval_min_free_mem_gb]
                if not cand:
                    break
                g = max(cand, key=lambda k: free[k])
                launch_eval(r, s, g, a.eval_workers)
                launched_e.add('{}__{}'.format(r['name'], s))
                free[g] -= 3.0
                n_e += 1
                print('[{:6.0f}s] eval  {} / {} on gpu {}'.format(time.time() - t0, r['name'], s, g), flush=True)
        status = {'phase': a.phase, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'train_done': sum(train_done(r) for r in runs), 'train_total': len(runs),
                  'eval_done': sum(eval_done(r, s) for r in runs for s, p in SETS[r['scene']] if p <= a.max_priority),
                  'eval_total': sum(1 for r in runs for s, p in SETS[r['scene']] if p <= a.max_priority),
                  'running': sorted(containers()), 'failed': failed, 'disk_free_gb': round(disk_free_gb(), 1),
                  'gpu_free_gb': {k: round(v, 1) for k, v in gpu_free_gb().items()}}
        try:
            os.makedirs(os.path.join(HOST, 'results/graph_cover'), exist_ok=True)
            dst = os.path.join(HOST, 'results/graph_cover/queue_status_{}.json'.format(a.phase.replace(',', '+')))
            with open(dst + '.tmp', 'w') as fh:
                json.dump(status, fh, indent=1)
            os.replace(dst + '.tmp', dst)
        except OSError as e:
            print('status not written: {}'.format(e), flush=True)
        mine = containers()
        if not pend_t and not pend_e and not mine:
            print('all done; failed: {}'.format(failed), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
