"""Host-side queue for the stress benchmark: every (model, scene, seed) x test set through the per-case MPPI
evaluator (models/graph_method_triangle/scripts/plan_eval_graph.py, batch 125, seed 0, forward MPPI), in
Docker, a few containers at a time. Results land in results/stress_bench/plan/<model>_<S>_s<seed>/<set>.json.
Results that an earlier campaign already produced with the identical evaluator settings are copied in
(provenance recorded in the JSON under summary.reused_from) instead of being re-planned.

    python3 models/stress_bench/run_stress.py --gpus 0,1,2 --slots 4 --workers 6          # run everything
    python3 models/stress_bench/run_stress.py --status                                    # progress
    python3 models/stress_bench/run_stress.py --dry-run                                   # list the jobs
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HOST = '/media/jeffrey/G-Shape-NTFields'
LIE = '/media/jeffrey/Lie-Shape-NTFields'
RES = 'results/stress_bench/plan'
EVAL = '/workspace/models/graph_method_triangle/scripts/plan_eval_graph.py'
SCENES = {'T': 'Tshape3d_env1', 'L': 'Lshape3d_env3', 'A': 'Ashape3d_env4'}

# model -> (kind, checkpoint pattern (container path), seeds per scene, reuse pattern (host path) or None)
MODELS = {
    'orig': ('metric', '/lieshape/Experiments/3dshape_metric_baseline/{scene}_metric_baseline_s{seed1}/Model_Epoch_04000_*.pt',
             {'T': [0, 1, 2], 'L': [0, 1, 2], 'A': [0, 1, 2]}, 'results/graph_method_cusp/plan/{S}_native_s{seed1}/{set}.json'),
    'ggram': ('graph_native', '/workspace/Experiments/graph_branch/{S}_ggram_s{seed}/latest.pt',
              {'T': [0, 1, 2], 'L': [0, 1, 2], 'A': [0, 1, 2]}, 'results/graph_branch/plan/{S}_ggram_s{seed}/{set}.json'),
    'gmetric': ('graph_native', '/workspace/Experiments/graph_lie/{S}_gmetric_iso_lie_s{seed}/latest.pt',
                {'T': [0, 1, 2], 'L': [0, 1, 2], 'A': [0, 1, 2]}, 'results/graph_lie/plan/{S}_gmetric_iso_lie_s{seed}/{set}.json'),
    'psi': ('graph_native', '/workspace/Experiments/graph_lie/{S}_gmetric_iso_lie_psi_s{seed}/latest.pt',
            {'T': [0, 1, 2], 'L': [0], 'A': [0]}, None),
    'eu2': ('graph_native', '/workspace/Experiments/graph_cover/{S}_eu2_s{seed}/latest.pt',
            {'T': [0, 1, 2], 'L': [0], 'A': [0]}, 'results/graph_cover/plan/{S}_eu2_s{seed}/{set}.json'),
    'eu2c': ('graph_native', '/workspace/Experiments/graph_cover/{S}_eu2c_s{seed}/latest.pt',
             {'T': [0, 1, 2], 'L': [0], 'A': [0]}, 'results/graph_cover/plan/{S}_eu2c_s{seed}/{set}.json'),
    'bc4': ('graph_native', '/workspace/Experiments/graph_branch/{S}_bc4_s{seed}/latest.pt',
            {'T': [0, 1, 2], 'L': [0, 1, 2], 'A': [0, 1, 2]}, 'results/graph_branch/plan/{S}_bc4_s{seed}/{set}.json'),
}

# (set suffix, priority, where): '' = the scene's standard set
STRESS = ['tightboth', 'tightgoal', 'tightrot', 'blocked', 'squeeze', 'ctrl', 'tightgoal_rev', 'blocked_rev']
SETS = ([(s, 1, 'bench') for s in STRESS] + [('tightstart', 2, 'bench'), ('longtrans_far', 2, 'bench'),
        ('tight_3d', 2, 'lie'), ('bigangle', 2, 'lie'), ('rotheavy', 2, 'lie'), ('', 3, 'lie')])
EXPECT = {'planner': 'mppi', 'wrap': False, 'seed': 0, 'batch': 125, 'steps': 200}


def set_name(S, suffix):
    return SCENES[S] + ('_' + suffix if suffix else '')


def data_path(S, suffix, where):
    """(container dataPath, host dir)"""
    name = set_name(S, suffix)
    if where == 'lie':
        return 'testing_data/3dshape/' + name, os.path.join(LIE, 'testing_data/3dshape', name)
    return '/workspace/datasets/3dshape/bench/' + name, os.path.join(HOST, 'datasets/3dshape/bench', name)


def checkpoint(model, S, seed):
    kind, pat, _, _ = MODELS[model]
    c = pat.format(scene=SCENES[S], S=S, seed=seed, seed1=seed + 1)
    host = c.replace('/lieshape', LIE).replace('/workspace', HOST)
    m = sorted(glob.glob(host))
    if not m:
        return None
    return m[-1].replace(LIE, '/lieshape').replace(HOST, '/workspace')


def jobs(args):
    out = []
    for model in args.models:
        kind, pat, seeds, reuse = MODELS[model]
        for S in args.scenes:
            for seed in seeds[S]:
                if S != 'T' and seed not in args.seeds_la:
                    continue
                ck = checkpoint(model, S, seed)
                if ck is None:
                    continue
                run = '{}_{}_s{}'.format(model, S, seed)
                for suffix, prio, where in SETS:
                    if args.sets and suffix not in args.sets:
                        continue
                    dp, hd = data_path(S, suffix, where)
                    if not os.path.exists(os.path.join(hd, 'sampled_points.npy')):
                        continue
                    sname = set_name(S, suffix)
                    src = None
                    if reuse:
                        cand = os.path.join(HOST, reuse.format(S=S, seed=seed, seed1=seed + 1, set=sname))
                        if os.path.exists(cand):
                            src = cand
                    out.append(dict(run=run, model=model, S=S, seed=seed, kind=kind, ckpt=ck, set=sname, dataPath=dp,
                                    prio=prio, reuse=src, out=os.path.join(HOST, RES, run, sname + '.json')))
    out.sort(key=lambda j: (j['prio'], j['S'] != 'T', j['seed'], j['set'], j['model']))
    return out


def reuse_ok(path):
    try:
        s = json.load(open(path))['summary']
    except Exception:
        return False
    return all(s.get(k) == v for k, v in EXPECT.items()) and s.get('temperature', 50.0) == 50.0


def copy_reused(j):
    os.makedirs(os.path.dirname(j['out']), exist_ok=True)
    d = json.load(open(j['reuse']))
    d['summary']['reused_from'] = os.path.relpath(j['reuse'], HOST)
    with open(j['out'], 'w') as fh:
        json.dump(d, fh)


def done(j):
    if not os.path.exists(j['out']):
        return False
    try:
        json.load(open(j['out']))['summary']['success_rate_se3']
        return True
    except Exception:
        return False


def docker_cmd(gpu, name, j, workers):
    log = j['out'][:-5] + '.log'
    inner = ('python -u {ev} --kind {k} --checkpoint {ck} --dataPath {dp} --out {o} --device cuda:0 --batch 125 '
             '--workers {w} --seed 0 > {log} 2>&1').format(ev=EVAL, k=j['kind'], ck=j['ckpt'], dp=j['dataPath'],
                                                            o=j['out'].replace(HOST, '/workspace'), w=workers,
                                                            log=log.replace(HOST, '/workspace'))
    return ['docker', 'run', '--rm', '--name', name, '--gpus', '"device={}"'.format(gpu), '-u',
            '{}:{}'.format(os.getuid(), os.getgid()), '-e', 'HOME=/tmp', '-e', 'MPLCONFIGDIR=/tmp',
            '-e', 'OMP_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', HOST + ':/workspace',
            '-v', LIE + ':/lieshape:ro', '-w', '/lieshape', 'pytorchserver:latest', '/bin/bash', '-c', inner]


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def gpu_free_mb():
    """{index: free MiB} from nvidia-smi (shared machine: other sessions' containers come and go)."""
    try:
        out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                      timeout=20).decode()
        return {int(a): float(b) for a, b in (ln.split(',') for ln in out.strip().split('\n') if ln.strip())}
    except Exception:
        return {}


def live_containers():
    try:
        out = subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], timeout=20).decode()
        return set(ln.strip() for ln in out.split('\n') if ln.startswith('sb_g'))
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--slots', type=int, default=4, help='concurrent evaluation containers')
    ap.add_argument('--workers', type=int, default=6, help='collision-check processes per container')
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--scenes', default='T,L,A')
    ap.add_argument('--seeds-la', default='0', help='seeds used on the L and A scenes')
    ap.add_argument('--sets', default='', help='comma list of set suffixes ("" = standard)')
    ap.add_argument('--max-priority', type=int, default=3)
    ap.add_argument('--retries', type=int, default=4)
    ap.add_argument('--min-gpu-mb', type=float, default=2500.0, help='free GPU memory needed to launch')
    ap.add_argument('--poll', type=float, default=20.0)
    ap.add_argument('--min-free-gb', type=float, default=3.0)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.models = a.models.split(',')
    a.scenes = a.scenes.split(',')
    a.seeds_la = [int(s) for s in a.seeds_la.split(',')]
    a.sets = a.sets.split(',') if a.sets else None
    gpus = [int(g) for g in a.gpus.split(',')]

    J = [j for j in jobs(a) if j['prio'] <= a.max_priority]
    for j in J:
        if not done(j) and j['reuse'] and reuse_ok(j['reuse']):
            copy_reused(j)
    todo = [j for j in J if not done(j)]
    print('{} jobs, {} done, {} to run'.format(len(J), len(J) - len(todo), len(todo)))
    if a.status or a.dry_run:
        for j in (todo if a.dry_run else []):
            print('  {:18s} {:36s} {}'.format(j['run'], j['set'], 'reuse' if j['reuse'] else ''))
        return

    running = {}   # name -> (Popen, job, gpu, t0)
    tries = {}
    # containers of this job list started by an earlier runner: wait for them instead of relaunching
    external = {}
    for j in list(todo):
        for name in live_containers():
            if name.endswith('_{}__{}'.format(j['run'], j['set'])):
                external[name] = j
                todo.remove(j)
                print('attach {} {}'.format(j['run'], j['set']), flush=True)
    status_path = os.path.join(HOST, 'results/stress_bench/queue_status_{}.json'.format('_'.join(a.scenes)))
    while todo or running:
        for name in list(running):
            p, j, g, t0 = running[name]
            if p.poll() is None:
                continue
            del running[name]
            if done(j):
                print('done  {} {} ({:.0f}s)'.format(j['run'], j['set'], time.time() - t0), flush=True)
            else:
                k = tries.get(j['out'], 0) + 1
                tries[j['out']] = k
                print('FAIL  {} {} (rc {}, try {})'.format(j['run'], j['set'], p.returncode, k), flush=True)
                if k <= a.retries:
                    todo.append(j)
                    time.sleep(30)
        alive = live_containers()
        for name in list(external):
            if name not in alive:
                j = external.pop(name)
                if done(j):
                    print('done  {} {} (attached)'.format(j['run'], j['set']), flush=True)
                else:
                    todo.append(j)
        mem = gpu_free_mb()
        while todo and len(running) + len(external) < a.slots and free_gb(HOST) > a.min_free_gb:
            ok = [g for g in gpus if mem.get(g, 0.0) - 1200.0 * sum(1 for v in running.values() if v[2] == g and time.time() - v[3] < 90) >= a.min_gpu_mb]
            if not ok:
                break
            j = todo.pop(0)
            if done(j):
                continue
            load = {g: sum(1 for v in running.values() if v[2] == g) for g in ok}
            g = max(ok, key=lambda x: (mem.get(x, 0.0) - 1200.0 * load[x]))
            name = 'sb_g{}_{}__{}'.format(g, j['run'], j['set'])
            os.makedirs(os.path.dirname(j['out']), exist_ok=True)
            subprocess.call(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p = subprocess.Popen(docker_cmd(g, name, j, a.workers), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running[name] = (p, j, g, time.time())
            print('start {} {} on gpu {}'.format(j['run'], j['set'], g), flush=True)
        with open(status_path, 'w') as fh:
            json.dump({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'running': sorted(running), 'todo': len(todo),
                       'done': sum(done(j) for j in J), 'total': len(J), 'attached': sorted(external)}, fh, indent=1)
        time.sleep(a.poll)
    print('all done', flush=True)


if __name__ == '__main__':
    main()
