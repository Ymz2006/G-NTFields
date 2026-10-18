#!/usr/bin/env python3
"""Host-side queue for the homotopy campaign on one GPU: trains runs in ntfields:rtx5070 containers and
scores each finished run with the per-case evaluator (base field and dual cover planning).

    python3 models/graph_homotopy/scripts/run_homotopy_queue.py --phase explore --train-slots 4 --eval-slots 2
    python3 models/graph_homotopy/scripts/run_homotopy_queue.py --phase explore --dry-run

Resumable: a run with train_summary.json is not retrained, a complete result JSON is not recomputed, running
containers (ght_* / ghe_*) are adopted. Status: results/graph_homotopy/queue_status.json. Standard library only.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HOST = '/home/jeffrey/G-NTFields'
LIE = '/home/jeffrey/Lie-Shape-NTFields'
IMAGE = 'ntfields:rtx5070'
EXP = 'Experiments/graph_homotopy'
RES = 'results/graph_homotopy/plan'
SCENES = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
PLANNERS = ('base', 'dual')

# tag -> (lift, lift_far, --set options)
P64 = ['pair_dim=64', 'res_even=32', 'res_odd=32']
VARIANTS = {
    'hmin': ('min', 0.5, []),                       # base-field training (= conformal branch head, K = 1)
    'hrand': ('random', 0.5, []),                   # uniformly random lift: the cover field everywhere
    'hboth': ('both', 0.5, []),                     # both sheets of every pair
    'hrand02': ('random', 0.2, []),                 # mostly the near sheet, 20 % far
    'hrand32': ('random', 0.5, ['res_even=32', 'res_odd=32']),
    'hrandP': ('random', 0.5, P64),                 # + invariant pair term
    'hminP': ('min', 0.5, P64),
    'hbothP': ('both', 0.5, P64),
    'hrand02P': ('random', 0.2, P64),
    'hrandP128': ('random', 0.5, ['pair_dim=128', 'pair_hidden=256', 'res_even=32', 'res_odd=32']),
    'hrandPs': ('random', 0.5, P64 + ['pair_init=0.01']),
    'hminPs': ('min', 0.5, P64 + ['pair_init=0.01']),
    'hrand64': ('random', 0.5, ['res_even=64', 'res_odd=64']),
    'hrandP16': ('random', 0.5, ['pair_dim=64']),
    'hmin32': ('min', 0.5, ['res_even=32', 'res_odd=32']),
    'hmin64': ('min', 0.5, ['res_even=64', 'res_odd=64']),
    'hminP16': ('min', 0.5, ['pair_dim=64']),
    'hrand01': ('random', 0.1, []),
    'hminL': ('min', 0.5, []),                       # 8000 epochs
    'hminXL': ('min', 0.5, []),                      # 16000 epochs
    'hmin32L': ('min', 0.5, ['res_even=32', 'res_odd=32']),   # 8000 epochs
    'hminPL': ('min', 0.5, P64),                     # 8000 epochs
    'hrand01L': ('random', 0.1, []),                 # 8000 epochs
    'hminPsL': ('min', 0.5, P64 + ['pair_init=0.01']),   # 8000 epochs
    'hrand01Ps': ('random', 0.1, P64 + ['pair_init=0.01']),
    'hrand01_32': ('random', 0.1, ['res_even=32', 'res_odd=32']),
}
VARIANT_EPOCHS = {'hminL': 8000, 'hminPL': 8000, 'hrand01L': 8000, 'hminPsL': 8000, 'hminXL': 16000, 'hmin32L': 8000}
# reference heads trained by their own scripts (base-field planner only)
EXTERNAL = {
    'bc4': ('python -u train/train_graph_branch.py --dataPath datasets/3dshape/data/{sc} --name {n} --seed {s} '
            '--arch graph_branch_conf --grad lie --speed iso --epochs {ep} --set n_branch=4'),
}
PHASES = {
    'explore': [('T', 0, v) for v in ('hmin', 'hrand', 'hrandP', 'hboth', 'hminP', 'hrand32', 'hrand02', 'hbothP')],
    'explore2': [('T', 0, v) for v in ('hrand02P', 'hrandP128', 'hminPs', 'hminP16')],   # random-lift extras dropped (H1)
    'baseline': [('T', 0, 'bc4'), ('T', 1, 'bc4'), ('T', 2, 'bc4')],
    # 3 seeds x T, L, A for the base-field-trained head (cover planning on top); the best cover-trained variant is added
    # as 'main2' once the exploration has decided it
    'main_hmin': [('T', 1, 'hmin'), ('T', 2, 'hmin'), ('L', 0, 'hmin'), ('A', 0, 'hmin'), ('L', 1, 'hmin'), ('A', 1, 'hmin'),
                  ('L', 2, 'hmin'), ('A', 2, 'hmin')],
    # H3: the pair term with a small initial value (hminPs) and the 32/32 residual (hmin32) lead at seed 0
    # H7: hminPs is a wash against hmin at 3 seeds on T (95.4 +- 0.8 vs 95.5 +- 0.6), its L / A seeds 1-2 were cancelled
    'main2': [('T', 1, 'hminPs'), ('T', 2, 'hminPs'), ('L', 0, 'hminPs'), ('A', 0, 'hminPs')],
    # H4: 8000 epochs of the plain head is the best single run (96.7 / 98.5); 3 seeds x 3 scenes of it
    'main3': [('T', 1, 'hminL'), ('T', 2, 'hminL'), ('L', 0, 'hminL'), ('A', 0, 'hminL'), ('L', 1, 'hminL'),
              ('A', 1, 'hminL'), ('L', 2, 'hminL'), ('A', 2, 'hminL')],
    'explore4': [('T', 0, 'hrand01Ps'), ('T', 0, 'hrand01_32')],
    'explore5': [('T', 0, 'hrand01L')],
    'explore6': [('T', 0, 'hminXL')],
    'explore3': [('T', 0, v) for v in ('hmin32', 'hmin64', 'hrand01', 'hminL', 'hminPL')],
    'seeds34': [('T', 3, 'hminL'), ('T', 4, 'hminL')],
    'seeds34LA': [('L', 3, 'hminL'), ('A', 3, 'hminL'), ('L', 4, 'hminL'), ('A', 4, 'hminL')],
    'final32': [('T', 0, 'hmin32L'), ('T', 1, 'hmin32L'), ('T', 2, 'hmin32L')],
}


def runs_of(phase):
    out = []
    for scene, seed, tag in PHASES[phase]:
        name = '{}_{}_s{}'.format(scene, tag, seed)
        if tag in EXTERNAL:
            out.append(dict(name=name, scene=scene, seed=seed, tag=tag, lift='-', far='-', sets=[], external=True,
                            ckpt=os.path.join(EXP, name, 'latest.pt')))
            continue
        lift, far, sets = VARIANTS[tag]
        out.append(dict(name=name, scene=scene, seed=seed, tag=tag, lift=lift, far=far, sets=sets, external=False,
                        ckpt=os.path.join(EXP, name, 'latest.pt')))
    return out


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def containers():
    return [n for n in sh("docker ps --format '{{.Names}}'").split() if n.startswith(('ght_', 'ghe_'))]


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


def train_done(r):
    return os.path.exists(os.path.join(HOST, EXP, r['name'], 'train_summary.json'))


def eval_done(r, s, planner):
    return complete_json(os.path.join(HOST, RES, r['name'], '{}__{}.json'.format(s, planner)))


def docker_prefix(cname, workdir, mounts='', threads=2):
    return ('docker run -d --rm --gpus all -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp '
            '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS={th} -v {host}:/workspace {m} -w {wd} --name {c} {img} '
            ).format(uid=os.getuid(), gid=os.getgid(), th=threads, host=HOST, m=mounts, wd=workdir, c=cname, img=IMAGE)


def launch_train(r, epochs):
    cname = 'ght_' + r['name']
    if r['external']:
        args = EXTERNAL[r['tag']].format(sc=SCENES[r['scene']], n=r['name'], s=r['seed'], ep=epochs)
        args += ' --out_root ./' + EXP
    else:
        args = ('python -u train/train_graph_homotopy.py --dataPath datasets/3dshape/data/{sc} --name {n} --seed {s} '
                '--epochs {ep} --lift {l} --lift-far {f}').format(sc=SCENES[r['scene']], n=r['name'], s=r['seed'],
                                                                 ep=VARIANT_EPOCHS.get(r['tag'], epochs), l=r['lift'], f=r['far'])
        if r['sets']:
            args += ' --set ' + ' '.join(r['sets'])
    os.makedirs(os.path.join(HOST, EXP, 'logs'), exist_ok=True)
    sh(docker_prefix(cname, '/workspace') + '/bin/bash -c "{} > {}/logs/{}.out 2>&1"'.format(args, EXP, r['name']))


def launch_eval(r, s, planner, workers):
    cname = 'ghe_{}__{}__{}'.format(r['name'], s, planner)
    out = '/workspace/{}/{}'.format(RES, r['name'])
    os.makedirs(os.path.join(HOST, RES, r['name']), exist_ok=True)
    args = ('python -u /workspace/models/graph_homotopy/scripts/plan_eval_homotopy.py --planner {p} '
            '--checkpoint /workspace/{ck} --dataPath testing_data/3dshape/{s} --out {o}/{s}__{p}.json '
            '--device cuda:0 --batch 125 --workers {w} --seed 0').format(p=planner, ck=r['ckpt'], s=s, o=out, w=workers)
    sh(docker_prefix(cname, '/lieshape', '-v {}:/lieshape:ro'.format(LIE), threads=1)
       + '/bin/bash -c "{} > {}/{}__{}.log 2>&1"'.format(args, out, s, planner))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--train-slots', type=int, default=4)
    ap.add_argument('--eval-slots', type=int, default=2)
    ap.add_argument('--eval-workers', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--planners', default=','.join(PLANNERS))
    ap.add_argument('--poll', type=float, default=30.0)
    ap.add_argument('--no-train', action='store_true')
    ap.add_argument('--no-eval', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    runs = [r for p in a.phase.split(',') for r in runs_of(p)]
    planners = a.planners.split(',')
    order = {r['name']: i for i, r in enumerate(runs)}
    if a.dry_run:
        for r in runs:
            print('{:22s} lift={:7s} far={} {:50s} {}'.format(r['name'], r['lift'], r['far'], ' '.join(r['sets']),
                                                           'trained' if train_done(r) else ''))
        return
    launched_t, launched_e, failed = set(), set(), {}
    t0 = time.time()
    while True:
        live = containers()
        live_t = {c[4:]: c for c in live if c.startswith('ght_')}
        live_e = {c[4:]: c for c in live if c.startswith('ghe_')}
        for n in list(launched_t):
            if n not in live_t:
                r = next(x for x in runs if x['name'] == n)
                if not train_done(r):
                    failed.setdefault(n, 'training exited without train_summary.json')
                launched_t.discard(n)
        for key in list(launched_e):
            if key not in live_e:
                rn, s, p = key.split('__')
                if not complete_json(os.path.join(HOST, RES, rn, '{}__{}.json'.format(s, p))):
                    failed.setdefault(key, 'evaluation exited without JSON')
                launched_e.discard(key)
        pend_t = [] if a.no_train else [r for r in runs if not train_done(r) and r['name'] not in live_t
                                       and r['name'] not in failed]
        pend_e = [] if a.no_eval else sorted(
            ((order[r['name']], pi, r, SCENES[r['scene']], p) for r in runs if train_done(r)
             for pi, p in enumerate(planners) if not (r['external'] and p != 'base')
             if not eval_done(r, SCENES[r['scene']], p) and '{}__{}__{}'.format(r['name'], SCENES[r['scene']], p) not in live_e
             and '{}__{}__{}'.format(r['name'], SCENES[r['scene']], p) not in failed), key=lambda x: x[:2])
        n_t = len(live_t)
        for r in pend_t:
            if n_t >= a.train_slots:
                break
            launch_train(r, a.epochs)
            launched_t.add(r['name'])
            n_t += 1
            print('[{:6.0f}s] train {}'.format(time.time() - t0, r['name']), flush=True)
        n_e = len(live_e)
        for _, _, r, s, p in pend_e:
            if n_e >= a.eval_slots:
                break
            launch_eval(r, s, p, a.eval_workers)
            launched_e.add('{}__{}__{}'.format(r['name'], s, p))
            n_e += 1
            print('[{:6.0f}s] eval  {} / {} / {}'.format(time.time() - t0, r['name'], s, p), flush=True)
        status = {'phase': a.phase, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'train_done': sum(train_done(r) for r in runs), 'train_total': len(runs),
                  'eval_done': sum(eval_done(r, SCENES[r['scene']], p) for r in runs for p in planners
                                   if not (r['external'] and p != 'base')),
                  'eval_total': sum(1 for r in runs for p in planners if not (r['external'] and p != 'base')),
                  'running': sorted(containers()), 'failed': failed}
        try:
            os.makedirs(os.path.join(HOST, 'results/graph_homotopy'), exist_ok=True)
            dst = os.path.join(HOST, 'results/graph_homotopy/queue_status_{}.json'.format(a.phase.replace(',', '+')))
            with open(dst + '.tmp', 'w') as fh:
                json.dump(status, fh, indent=1)
            os.replace(dst + '.tmp', dst)
        except OSError as e:
            print('status not written: {}'.format(e), flush=True)
        if not pend_t and not pend_e and not containers():
            print('all done; failed: {}'.format(failed), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
