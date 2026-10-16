"""Queue the full pipeline for the shapes of make_shapes3d.py (host side, stdlib only).

Cells are ``<shape>_<env>`` for the five shapes on env5, env6 (tight) and env4
(Ashape3d's env) by default; --shapes / --envs pick others (env7 / env8: env5
with a 60 / 70 window).  Per cell:

  test   1000 start/goal pairs: original preprocess_obj.py --testing_data --offset 0.02
  shard  4 x 200k training pairs: preprocess_obj_fast.py via preprocess_sharded.py
         (same env points in every shard, distribution of one 800k run)
  merge  shards -> datasets/3dshape/<cell>
  train  the metric baseline (Lie-Shape-NTFields train/train_3dshape.py
         --models metric_baseline --epochs 4000), checkpoints every 1000 epochs kept
  eval   Lie-Shape-NTFields analysis/plan_eval.py --kind metric on the test set

Every job is a detached docker container (``s3d_<job>``), so jobs outlive this
script; a job is done when results/shapes3d/jobs/<job>.rc exists (its exit code).
GPU jobs start only on a GPU with enough free memory and fewer than
--per_gpu of our jobs.  Rerun the script to resume.

    python3 dataprocessing/shapes3d_pipeline.py            # run / resume the queue
    python3 dataprocessing/shapes3d_pipeline.py --status   # print job states
    python3 dataprocessing/shapes3d_pipeline.py --shapes tripod3d,loop3d --envs env7,env8
"""
import argparse
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIE = '/media/jeffrey/Lie-Shape-NTFields'
OUT = os.path.join(ROOT, 'results', 'shapes3d')
JOBS = os.path.join(OUT, 'jobs')
LOGS = os.path.join(OUT, 'logs')
IMAGE = 'pytorchserver:latest'

SHAPES = ['tripod3d', 'crank3d', 'loop3d', 'twistI3d', 'chair3d']
ENVS = ['env6', 'env5', 'env4']          # slowest first
SHARDS = 4
TRAIN_N = 800000
TRAIN_BS = 1000
TEST_N = 1000
EPOCHS = 4000
SEED_TAG = 's1'

# MB of free GPU memory a job needs before it may start.
NEED = {'test': 5000, 'shard': 7500, 'train': 3500, 'eval': 10000}


def cells():
    return ['{}_{}'.format(s, e) for e in ENVS for s in SHAPES]


def job_list():
    """(job id, kind, cell, deps) in launch-priority order (within a kind)."""
    jobs = []
    for c in cells():
        jobs.append(('test_' + c, 'test', c, []))
    for c in cells():
        for i in range(SHARDS):
            jobs.append(('shard_{}_{}'.format(c, i), 'shard', c, []))
    for c in cells():
        jobs.append(('merge_' + c, 'merge', c, ['shard_{}_{}'.format(c, i) for i in range(SHARDS)]))
        jobs.append(('train_' + c, 'train', c, ['merge_' + c]))
        jobs.append(('eval_' + c, 'eval', c, ['train_' + c, 'test_' + c]))
    return jobs


KIND_ORDER = {'eval': 0, 'train': 1, 'merge': 2, 'test': 3, 'shard': 4}


def command(job, kind, cell):
    shape, env = cell.split('_')
    log = '/workspace/results/shapes3d/logs/{}.log'.format(job)
    if kind == 'test':
        cmd = ('python -u /lieshape/dataprocessing/preprocess_obj.py --env datasets/3dshape/{env}.obj '
               '--shape datasets/3dshape/{shape}.obj --out testing_data/3dshape/{cell} '
               '--num_samples {n} --testing_data --offset 0.02 --batch_size 250 --visualize --device cuda:0'
               ).format(env=env, shape=shape, cell=cell, n=TEST_N)
    elif kind == 'shard':
        i = int(job.rsplit('_', 1)[1])
        cmd = ('PREPROCESS_OBJ=/lieshape/dataprocessing/preprocess_obj.py python -u dataprocessing/preprocess_sharded.py run '
               '--script dataprocessing/preprocess_obj_fast.py --shard {i} --shards {S} --env_seed 1 -- '
               '--env datasets/3dshape/{env}.obj --shape datasets/3dshape/{shape}.obj '
               '--out datasets/3dshape/{cell} --num_samples {n} --batch_size {bs} --visualize --device cuda:0'
               ).format(i=i, S=SHARDS, env=env, shape=shape, cell=cell, n=TRAIN_N, bs=TRAIN_BS)
    elif kind == 'merge':
        cmd = ('PREPROCESS_OBJ=/lieshape/dataprocessing/preprocess_obj.py python -u dataprocessing/preprocess_sharded.py merge '
               '--script /lieshape/dataprocessing/preprocess_obj.py --shards {S} --env_seed 1 '
               '--out datasets/3dshape/{cell}').format(S=SHARDS, cell=cell)
    elif kind == 'train':
        run = '/workspace/Experiments/shapes3d_metric_baseline/{}_{}'.format(cell, SEED_TAG)
        cmd = ('cd /lieshape && OMP_NUM_THREADS=4 NTF_NO_PLOT=1 python -u train/train_3dshape.py '
               '--dataPath /workspace/datasets/3dshape/{cell} --models metric_baseline --epochs {ep} '
               '--device cuda:0 --modelPath /workspace/Experiments/shapes3d_metric_baseline '
               '--name {cell}_{tag} --no-wandb 2>&1 | grep -v "UserWarning\\|warnings.warn" && '
               'find {run} -name "Model_Epoch_*.pt" ! -name "Model_Epoch_0[1-4]000_*" -delete'
               ).format(cell=cell, ep=EPOCHS, tag=SEED_TAG, run=run)
    elif kind == 'eval':
        run = '/workspace/Experiments/shapes3d_metric_baseline/{}_{}'.format(cell, SEED_TAG)
        cmd = ('cd /lieshape && ck=$(ls {run}/Model_Epoch_0{ep:04d}_*.pt) && OMP_NUM_THREADS=1 python -u '
               'analysis/plan_eval.py --kind metric --checkpoint $ck --dataPath /workspace/testing_data/3dshape/{cell} '
               '--out /workspace/results/shapes3d/plan/{cell}_metric_{tag}.json --device cuda:0 '
               '--batch 125 --workers 8 --seed 0 2>&1 | grep -v "UserWarning\\|warnings.warn"'
               ).format(run=run, ep=EPOCHS, cell=cell, tag=SEED_TAG)
    rc = '/workspace/results/shapes3d/jobs/{}.rc'.format(job)
    return 'set -o pipefail; ( {cmd} ) > {log} 2>&1; echo $? > {rc}'.format(cmd=cmd, log=log, rc=rc)


def docker_run(job, kind, cell, gpu):
    args = ['docker', 'run', '-d', '--rm', '--name', 's3d_' + job,
            '-u', '{}:{}'.format(os.getuid(), os.getgid()),
            '-e', 'HOME=/tmp', '-e', 'MPLCONFIGDIR=/tmp', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-v', ROOT + ':/workspace', '-v', LIE + ':/lieshape:ro', '-w', '/workspace']
    if gpu is not None:
        args += ['--gpus', 'device={}'.format(gpu)]
    args += [IMAGE, 'bash', '-c', command(job, kind, cell)]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)


def running():
    out = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True).stdout
    return {n[4:] for n in out.split() if n.startswith('s3d_')}


def gpu_free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, u, t = (int(x) for x in line.split(','))
        free[i] = t - u
    return free


def rc_of(job):
    p = os.path.join(JOBS, job + '.rc')
    if not os.path.exists(p):
        return None
    try:
        return int(open(p).read().strip() or -1)
    except ValueError:
        return -1


def status(jobs, run_set, placed):
    st = {}
    for job, kind, cell, deps in jobs:
        rc = rc_of(job)
        st[job] = ('running@gpu{}'.format(placed.get(job, '?')) if job in run_set else
                   'done' if rc == 0 else 'failed(rc={})'.format(rc) if rc is not None else 'pending')
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--per_gpu', type=int, default=2, help='max of our GPU jobs per GPU')
    ap.add_argument('--max_jobs', type=int, default=5, help='max of our GPU jobs overall')
    ap.add_argument('--reserve', type=int, default=1500, help='MB left free on a GPU after a job starts')
    ap.add_argument('--settle', type=int, default=90, help='seconds to wait after a GPU launch')
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--shapes', default=','.join(SHAPES), help='comma-separated subset of shapes')
    ap.add_argument('--envs', default=','.join(ENVS), help='comma-separated envs (env4..env8)')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    SHAPES[:] = args.shapes.split(',')
    ENVS[:] = args.envs.split(',')
    for d in (JOBS, LOGS, os.path.join(OUT, 'plan')):
        os.makedirs(d, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(',')]
    jobs = job_list()
    jobs.sort(key=lambda j: KIND_ORDER[j[1]])
    placed_path = os.path.join(OUT, 'placed.json')
    placed = json.load(open(placed_path)) if os.path.exists(placed_path) else {}
    tries = {}

    if args.status:
        st = status(jobs, running(), placed)
        for k in ('running', 'failed', 'done', 'pending'):
            sel = [j for j, s in st.items() if s.startswith(k)]
            print('{:8s} {:3d}  {}'.format(k, len(sel), ' '.join(sel) if k in ('running', 'failed') else ''))
        return

    while True:
        run_set = running()
        st = status(jobs, run_set, placed)
        with open(os.path.join(OUT, 'status.json'), 'w') as fh:
            json.dump(st, fh, indent=0)
        # Retry failed jobs a limited number of times (e.g. a CUDA OOM next to other users' jobs).
        for job, s in st.items():
            if s.startswith('failed') and tries.get(job, 0) < args.retries:
                tries[job] = tries.get(job, 0) + 1
                os.rename(os.path.join(JOBS, job + '.rc'),
                          os.path.join(JOBS, '{}.rc.fail{}'.format(job, tries[job])))
                st[job] = 'pending'
        if all(s == 'done' or (s.startswith('failed')) for s in st.values()):
            print('all jobs finished:', sum(s == 'done' for s in st.values()), 'done,',
                  sum(s.startswith('failed') for s in st.values()), 'failed', flush=True)
            return
        ours = {g: sum(1 for j in run_set if placed.get(j) == g) for g in gpus}
        free = gpu_free()              # one reading per pass, so launches follow the priority order
        launched = False
        for job, kind, cell, deps in jobs:
            if st[job] != 'pending' or any(st[d] != 'done' for d in deps):
                continue
            if kind == 'merge':
                docker_run(job, kind, cell, None)
                placed[job] = None
                print(time.strftime('%H:%M:%S'), 'launch', job, '(cpu)', flush=True)
                continue
            if sum(ours.values()) >= args.max_jobs:
                break
            cand = [g for g in gpus if ours[g] < args.per_gpu and free.get(g, 0) >= NEED[kind] + args.reserve]
            if not cand:
                continue
            g = max(cand, key=lambda x: free[x])
            docker_run(job, kind, cell, g)
            placed[job] = g
            ours[g] += 1
            print(time.strftime('%H:%M:%S'), 'launch', job, 'gpu', g, 'free MB', free[g], flush=True)
            launched = True
            break                      # one GPU launch per settle period
        with open(placed_path, 'w') as fh:
            json.dump(placed, fh)
        time.sleep(args.settle if launched else 30)


if __name__ == '__main__':
    main()
