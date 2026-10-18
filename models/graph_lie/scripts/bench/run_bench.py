#!/usr/bin/env python3
"""Host-side runner for the chart-vs-Lie benchmark: every graph_metric checkpoint x targeted subset x MPPI sampler.

One container per (checkpoint, sampler) runs plan_eval_bench.py over all its subsets in sequence; containers
are spread over --gpus with --slots in flight. Resumable (a subset whose JSON is complete is skipped; a container
still running from an earlier invocation is adopted). Python 3 standard library only.

    python3 models/graph_lie/scripts/bench/run_bench.py --scenes T --samplers chart,lie --gpus 2,3 --slots 6
    python3 models/graph_lie/scripts/bench/run_bench.py --dry-run

Results: results/graph_lie/bench/plan/<run>/<sampler>/<set>.json (+ .audit.npz), logs next to them.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HOST = '/media/jeffrey/G-Shape-NTFields'
LIE = '/media/jeffrey/Lie-Shape-NTFields'
IMAGE = 'pytorchserver:latest'
RES = 'results/graph_lie/bench/plan'
SCENES = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}
BENCH_SUBSETS = ['nearid', 'bothfar', 'startfar', 'goalfar', 'perp', 'par', 'longtrans_near', 'longtrans_far', 'tightstart']
# (name, dataPath inside the container, --cases); bench sets live in this repo, Lie suites in the Lie repo
def sets_for(scene, full=False):
    S = SCENES[scene]
    out = [(s, '/workspace/datasets/3dshape/data/bench/{}_{}'.format(S, s), 0) for s in BENCH_SUBSETS]
    out.append(('standard300', 'testing_data/3dshape/{}'.format(S), 300))
    if scene == 'T':
        out += [(s, 'testing_data/3dshape/{}_{}'.format(S, s), 0) for s in ('bigangle', 'rotheavy', 'transonly')]
    if full:
        out.append(('standard', 'testing_data/3dshape/{}'.format(S), 0))
    return out


def checkpoints(scene, seeds=(0, 1, 2)):
    """graph_metric runs: (run name, checkpoint path, speed, grad, seed). combined/chart (and T combined/lie s0)
    were trained by models/graph_native (bitwise the same procedure, graph_lie EXPERIMENTS.md L0)."""
    gn_done = {('T', 'chart'): [0, 1, 2], ('L', 'chart'): [0, 1], ('A', 'chart'): [0, 1], ('T', 'lie'): [0]}
    out = []
    for speed in ('comb', 'iso', 'aniso'):
        for grad in ('chart', 'lie'):
            for seed in seeds:
                name = '{}_gmetric_{}_{}_s{}'.format(scene, speed, grad, seed)
                if speed == 'comb' and seed in gn_done.get((scene, grad), []):
                    ck = 'Experiments/graph_native/{}_gmetric_{}_s{}/latest.pt'.format(scene, grad, seed)
                else:
                    ck = 'Experiments/graph_lie/{}/latest.pt'.format(name)
                if os.path.exists(os.path.join(HOST, ck)):
                    out.append(dict(name=name, ckpt=ck, speed=speed, grad=grad, seed=seed, scene=scene))
    return out


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and r.stderr.strip():
        print('[cmd failed] {}'.format(r.stderr.strip().splitlines()[-1][:300]), flush=True)
    return r.stdout


def running():
    return [n for n in sh("docker ps --format '{{.Names}}'").split() if n.startswith('gbench_')]


def complete_json(path):
    try:
        with open(path, 'rb') as fh:
            fh.seek(max(0, os.path.getsize(path) - 64))
            return fh.read().rstrip().endswith(b'}')
    except OSError:
        return False


SUBSET_FILTER = None


def pending_sets(r, sampler, full):
    return [(s, dp, nc) for s, dp, nc in sets_for(r['scene'], full)
            if (SUBSET_FILTER is None or s in SUBSET_FILTER)
            if not complete_json(os.path.join(HOST, RES, r['name'], sampler, s + '.json'))
            and (not dp.startswith('/workspace') or os.path.exists(os.path.join(HOST, dp[len('/workspace/'):], 'sampled_points.npy')))]


def launch(r, sampler, gpu, workers, full):
    cname = 'gbench_g{}_{}_{}'.format(gpu, r['name'], sampler)
    outdir = os.path.join(RES, r['name'], sampler)
    os.makedirs(os.path.join(HOST, outdir), exist_ok=True)
    cmds = []
    for s, dp, nc in pending_sets(r, sampler, full):
        cmds.append('python -u /workspace/models/graph_lie/scripts/bench/plan_eval_bench.py --kind graph_native '
                    '--checkpoint /workspace/{ck} --dataPath {dp} --out /workspace/{o}/{s}.json --device cuda:0 '
                    '--sampler {sa} --batch 150 --workers {w} --seed 0 --cases {nc} > /workspace/{o}/{s}.log 2>&1'
                    .format(ck=r['ckpt'], dp=dp, o=outdir, s=s, sa=sampler, w=workers, nc=nc))
    if not cmds:
        return None
    docker = ('docker run -d --rm --gpus \'"device={g}"\' -u {uid}:{gid} -e HOME=/tmp -e MPLCONFIGDIR=/tmp '
              '-e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '
              '-v {host}:/workspace -v {lie}:/lieshape:ro '
              '-w /lieshape --name {c} {img} bash -c "{cmd}"').format(
        g=gpu, uid=os.getuid(), gid=os.getgid(), host=HOST, lie=LIE, c=cname, img=IMAGE, cmd='; '.join(cmds))
    sh(docker)
    return cname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='T')
    ap.add_argument('--samplers', default='chart,lie')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--gpus', default='2,3', help='comma list; gpu:cap sets a per-GPU cap on containers (default 2)')
    ap.add_argument('--slots', type=int, default=6)
    ap.add_argument('--retries', type=int, default=3, help='relaunch a job whose container died with subsets pending')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--full', action='store_true', help='also the full standard sets')
    ap.add_argument('--subsets', default='', help='comma list: restrict to these subsets')
    ap.add_argument('--poll', type=float, default=20.0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    global SUBSET_FILTER
    SUBSET_FILTER = set(a.subsets.split(',')) if a.subsets else None
    caps = {int(g.split(':')[0]): (int(g.split(':')[1]) if ':' in g else 2) for g in a.gpus.split(',')}
    gpus = sorted(caps)
    seeds = tuple(int(s) for s in a.seeds.split(','))
    jobs = [(r, sa) for sc in a.scenes.split(',') for r in checkpoints(sc, seeds) for sa in a.samplers.split(',')]
    if a.dry_run:
        for r, sa in jobs:
            print('{:28s} {:5s} {:3d} pending: {}'.format(r['name'], sa, len(pending_sets(r, sa, a.full)),
                                                          ' '.join(s for s, _, _ in pending_sets(r, sa, a.full))))
        return
    t0 = time.time()
    launched = {}   # (name, sampler) -> number of launches (retries after an OOM-killed container)
    while True:
        live = running()
        todo = [(r, sa) for r, sa in jobs if pending_sets(r, sa, a.full)
                and not any(c.endswith('_{}_{}'.format(r['name'], sa)) for c in live)
                and launched.get((r['name'], sa), 0) <= a.retries]
        n_live = len(live)
        while todo and n_live < a.slots:
            load = {g: sum(1 for c in live if c.startswith('gbench_g{}_'.format(g))) for g in gpus}
            free = [g for g in gpus if load[g] < caps[g]]
            if not free:
                break
            r, sa = todo.pop(0)
            g = min(free, key=lambda k: (load[k], k))
            c = launch(r, sa, g, a.workers, a.full)
            launched[(r['name'], sa)] = launched.get((r['name'], sa), 0) + 1
            live.append(c)
            n_live += 1
            print('[{:6.0f}s] launch {} / {} on gpu {} (try {})'.format(time.time() - t0, r['name'], sa, g,
                                                                     launched[(r['name'], sa)]), flush=True)
        done = sum(1 for r, sa in jobs if not pending_sets(r, sa, a.full))
        status = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'jobs_done': done, 'jobs_total': len(jobs),
                  'running': sorted(running())}
        with open(os.path.join(HOST, 'results/graph_lie/bench/run_status.json'), 'w') as fh:
            json.dump(status, fh, indent=1)
        if not todo and not running():
            left = [(r['name'], sa) for r, sa in jobs if pending_sets(r, sa, a.full)]
            print('all done; incomplete: {}'.format(left), flush=True)
            return
        time.sleep(a.poll)


if __name__ == '__main__':
    sys.exit(main())
