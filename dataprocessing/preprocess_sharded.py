"""Run preprocess_obj.py as independent shards, then merge them into one dataset.

The only numpy randomness in preprocess_obj.py is the env surface sample
(env.npy) and the shape's radial-shell sample; the placements come from torch.
Every shard seeds numpy with the same ``--env_seed`` -- so all shards see the
same env points and shells -- and torch with ``env_seed * 1000 + shard``, so the
placements differ.  Pairs are drawn i.i.d. given the env points (each batch is
independent; ``--batch_size`` never changes the distribution), so the merged set
has the same distribution as one ``--num_samples`` run, while the shards run in
parallel and each one saves as soon as it finishes.

    # shard i of S (pass preprocess_obj.py's own flags after --; --out is the final dataset dir)
    python dataprocessing/preprocess_sharded.py run --shard i --shards S --env_seed 1 -- \\
        --env datasets/3dshape/obj/env5.obj --shape datasets/3dshape/obj/loop3d.obj \\
        --out datasets/3dshape/data/loop3d_env5 --num_samples 800000 --batch_size 500 --device cuda:0

    # once every shard has finished
    python dataprocessing/preprocess_sharded.py merge --out datasets/3dshape/data/loop3d_env5 --shards S

Shards go to ``<out>/shards/shard_<i>/``; ``merge`` writes the usual files to
``<out>`` (meta.json gains ``shards``, ``env_seed``, ``torch_seeds``) and deletes
the shard arrays unless ``--keep_shards``.
"""
import argparse
import importlib.util
import json
import os
import runpy
import shutil
import sys

import numpy as np

DEFAULT_SCRIPT = os.environ.get('PREPROCESS_OBJ', '/lieshape/dataprocessing/preprocess_obj.py')
PAIR_FILES = ['sampled_points', 'speed', 'speed_angles', 'speed_dists', 'normal', 'trans_n', 'rot_n']


def _flag(argv, name):
    i = argv.index(name)
    return i, argv[i + 1]


def run(args, passthrough):
    import torch
    i_out, out = _flag(passthrough, '--out')
    i_num, num = _flag(passthrough, '--num_samples')
    num = int(num)
    n = num // args.shards + (1 if args.shard < num % args.shards else 0)
    argv = list(passthrough)
    argv[i_out + 1] = os.path.join(out, 'shards', 'shard_{}'.format(args.shard))
    argv[i_num + 1] = str(n)
    np.random.seed(args.env_seed)
    torch.manual_seed(args.env_seed * 1000 + args.shard)
    print('[shard {}/{}] {} of {} pairs, numpy seed {}, torch seed {}'.format(
        args.shard, args.shards, n, num, args.env_seed, args.env_seed * 1000 + args.shard), flush=True)
    sys.argv = [args.script] + argv
    runpy.run_path(args.script, run_name='__main__')


def merge(args):
    dirs = [os.path.join(args.out, 'shards', 'shard_{}'.format(i)) for i in range(args.shards)]
    metas = [json.load(open(os.path.join(d, 'meta.json'))) for d in dirs]
    env = np.load(os.path.join(dirs[0], 'env.npy'))
    for d, m in zip(dirs[1:], metas[1:]):
        assert np.array_equal(np.load(os.path.join(d, 'env.npy')), env), 'env points differ: ' + d
        for k in ('env_scale', 'env_center', 'shape_obj', 'env_obj', 'margin', 'offset', 'num_tets'):
            assert m[k] == metas[0][k], '{} differs in {}'.format(k, d)
    merged = {}
    for f in PAIR_FILES:
        parts = [np.load(os.path.join(d, f + '.npy')) for d in dirs]
        merged[f] = np.concatenate(parts, axis=0)
        np.save(os.path.join(args.out, f), merged[f])
    np.save(os.path.join(args.out, 'env'), env)
    n = len(merged['sampled_points'])
    assert all(len(a) == n for a in merged.values())
    meta = dict(metas[0])
    meta['shards'] = args.shards
    meta['env_seed'] = args.env_seed
    meta['torch_seeds'] = [args.env_seed * 1000 + i for i in range(args.shards)]
    with open(os.path.join(args.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    # Visualizations: the placements view of shard 0, distributions of the merged set.
    if os.path.exists(os.path.join(dirs[0], 'sampled_placements.html')):
        shutil.copy(os.path.join(dirs[0], 'sampled_placements.html'), args.out)
        spec = importlib.util.spec_from_file_location('pre', args.script)
        pre = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pre)
        for f in ('speed', 'speed_angles', 'speed_dists'):
            pre.visual_speed_distribution(merged[f], label=f,
                                          save_path=os.path.join(args.out, f + '_distribution.html'))
    sp = merged['speed']
    print('merged {} shards -> {} pairs in {}'.format(args.shards, n, args.out))
    print('  speed x0 [{:.3f}, {:.3f}]  x1 [{:.3f}, {:.3f}]'.format(
        sp[:, 0].min(), sp[:, 0].max(), sp[:, 1].min(), sp[:, 1].max()))
    if not args.keep_shards:
        for d in dirs:
            for f in PAIR_FILES + ['env']:
                os.remove(os.path.join(d, f + '.npy'))


def main():
    argv = sys.argv[1:]
    passthrough = []
    if '--' in argv:
        k = argv.index('--')
        argv, passthrough = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=['run', 'merge'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, required=True)
    ap.add_argument('--env_seed', type=int, default=1)
    ap.add_argument('--out', help='merge: the dataset dir (run takes --out after --)')
    ap.add_argument('--script', default=DEFAULT_SCRIPT, help='preprocess_obj.py to run')
    ap.add_argument('--keep_shards', action='store_true')
    args = ap.parse_args(argv)
    if args.cmd == 'run':
        run(args, passthrough)
    else:
        merge(args)


if __name__ == '__main__':
    main()
