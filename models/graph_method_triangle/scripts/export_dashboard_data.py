"""Collect the results behind the triangle-head dashboard into CSV files.

    python models/graph_method_triangle/scripts/export_dashboard_data.py --out results/graph_method_triangle_dashboard
    python models/graph_method_triangle/scripts/export_dashboard_data.py --out ... --diagnostics   # + GPU diagnostics

Every row names the result file it was read from (`source`). Rates are fractions
(0-1), failure counts are per 1000 test cases. --diagnostics also evaluates
checkpoints: the share of T^2 carried by the rotation latent on dataset pairs, and the
held-out endpoint |c| = S |grad T| per clearance-speed bin (Eikonal target 1).
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

sys.path.append('.')

SCENES = {'T': ('T-shape', 'Tshape3d_env1'), 'L': ('L-shape', 'Lshape3d_env3'), 'A': ('A-shape', 'Ashape3d_env4')}
R_NEW = 'results/graph_method_triangle_stageE'
E_NEW = 'Experiments/graph_method_triangle_stageE'

# (model_id, label, role, {scene: result-dir glob}, {scene: experiment-dir glob})
STAGE_E = [
    ('tri_metric_eq', 'Triangle head (equivariant V)', 'new',
     {s: R_NEW + '/%s_trimetric_eq_s[0-9]_c1000' % s for s in 'TLA'},
     {s: E_NEW + '/%s_trimetric_eq_s[0-9]' % s for s in 'TLA'}),
    ('tri_metric_free', 'Triangle head (free V)', 'variant',
     {s: R_NEW + '/%s_trimetric_b1g1_s[0-9]_c1000' % s for s in 'TLA'},
     {s: E_NEW + '/%s_trimetric_b1g1_s[0-9]' % s for s in 'TLA'}),
    ('metric', 'G-Shape metric head', 'baseline',
     {'T': 'results/graph_method_stageE/stageE_metric_s[0-9]_c1000',
      'L': 'results/graph_method_stageE2/L_metric_s[0-9]_c1000',
      'A': 'results/graph_method_stageE2/A_metric_s[0-9]_c1000'},
     {'T': 'Experiments/graph_method_stageE/stageE_metric_s[0-9]',
      'L': 'Experiments/graph_method_stageE2/L_metric_s[0-9]',
      'A': 'Experiments/graph_method_stageE2/A_metric_s[0-9]'}),
    ('graph_metric', 'graph_metric (logsumexp a)', 'baseline',
     {'T': 'results/graph_method_stageE/stageE_graph_metric_s[0-9]_c1000',
      'L': 'results/graph_method_stageE2/L_graph_metric_s[0-9]_c1000',
      'A': 'results/graph_method_stageE2/A_graph_metric_s[0-9]_c1000'},
     {'T': 'Experiments/graph_method_stageE/stageE_graph_metric_s[0-9]',
      'L': 'Experiments/graph_method_stageE2/L_graph_metric_s[0-9]',
      'A': 'Experiments/graph_method_stageE2/A_graph_metric_s[0-9]'}),
    ('graph_gram', 'graph_gram (Euclidean a)', 'baseline',
     {'T': 'results/graph_method_stageE/stageE_graph_gram_s[0-9]_c1000',
      'L': 'results/graph_method_stageE2/L_graph_gram_s[0-9]_c1000',
      'A': 'results/graph_method_stageE2/A_graph_gram_s[0-9]_c1000'},
     {'T': 'Experiments/graph_method_stageE/stageE_graph_gram_s[0-9]',
      'L': 'Experiments/graph_method_stageE2/L_graph_gram_s[0-9]',
      'A': 'Experiments/graph_method_stageE2/A_graph_gram_s[0-9]'}),
]

F_NEW = 'results/graph_method_triangle_free'
FREE = [
    ('tri_eq', 'Triangle head (equivariant V)', 'new', F_NEW + '/tri_eq_s[0-9]'),
    ('tri_metric_eq', 'Triangle head, seminorm a (equivariant V)', 'new', F_NEW + '/trimetric_eq_s[0-9]'),
    ('gram_proj', 'graph_gram read by the exact head', 'new', F_NEW + '/gram_proj_s[0-9]'),
    ('gram_tri_sw11k', 'graph_gram, exact head from step 11k', 'variant', F_NEW + '/gramtri_sw11k_s[0-9]'),
    ('tri_free', 'Triangle head (free V)', 'variant', F_NEW + '/tri_[db]*_s[0-9]'),
    ('tri_metric_free', 'Triangle head, seminorm a (free V)', 'variant', F_NEW + '/trimetric_[db]*_s[0-9]'),
    ('tri_eq_noanchor', 'Triangle head (equivariant V), no anchors', 'ablation', F_NEW + '/tri_eq_noanchor_s[0-9]'),
    ('graph_gram', 'graph_gram', 'baseline', 'results/graph_method_final/graph_gram_s[0-9]'),
    ('graph_metric', 'graph_metric (logsumexp a)', 'baseline', 'results/graph_method_final/graph_metric_s[0-9]'),
    ('metric', 'G-Shape metric head', 'baseline', 'results/graph_method_final/metric_best_recipe_s[0-9]'),
]
FREE_KEYS = [('uni_rel_p50', 'uniform_rel_p50'), ('uni_rel_p95', 'uniform_rel_p95'), ('mix_rel_p95', 'mixture_rel_p95'),
             ('rel_179_180_p50', 'angle_179-180_rel_p50'), ('exact_pi_abs', 'exact_pi_abs_mean'),
             ('eik_res_p50', 'uniform_eik_x_regular_p50'), ('latent_b_err', 'uniform_latent_b_mean'),
             ('grad_success', 'plan_gradient_success'), ('grad_cost_p50', 'plan_gradient_cost_ratio_p50'),
             ('samp_success', 'plan_sampling_success')]

# triangle inequality guaranteed by the architecture for every parameter value
GUARANTEED = {'tri_metric_eq': 'yes', 'tri_metric_free': 'yes', 'tri_eq': 'yes', 'gram_proj': 'yes', 'gram_tri_sw11k': 'yes',
              'tri_free': 'yes', 'tri_eq_noanchor': 'yes', 'metric': 'no', 'graph_metric': 'no', 'graph_gram': 'no'}

ATTACKS = [  # (json file, checkpoint path, model, setting, guarantee)
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_final/graph_gram_s0/best.pt', 'graph_gram', 'Obstacle-free', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_triangle_free/gram_proj_s0/best.pt', 'graph_gram read by the exact head', 'Obstacle-free', 'yes'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_triangle_free/tri_eq_s0/best.pt', 'Triangle head (equivariant V)', 'Obstacle-free', 'yes'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_triangle_free/trimetric_eq_s0/best.pt', 'Triangle head, seminorm a (equivariant V)', 'Obstacle-free', 'yes'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_final/metric_best_recipe_s0/best.pt', 'G-Shape metric head', 'Obstacle-free', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_final/graph_metric_s0/best.pt', 'graph_metric (logsumexp a)', 'Obstacle-free', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_final/scalar_s0/best.pt', 'Direct scalar network', 'Obstacle-free', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_stageE/stageE_graph_gram_s0/latest.pt', 'graph_gram (Euclidean a)', 'T-shape', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_stageE/stageE_metric_s0/latest.pt', 'G-Shape metric head', 'T-shape', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_stageE/stageE_graph_metric_s0/latest.pt', 'graph_metric (logsumexp a)', 'T-shape', 'no'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_triangle_stageE/T_trimetric_b1g1_s0/latest.pt', 'Triangle head (free V)', 'T-shape', 'yes'),
    ('results/graph_method_triangle_triangle/final.json', 'Experiments/graph_method_triangle_stageE/T_trimetric_eq_s0/latest.pt', 'Triangle head (equivariant V)', 'T-shape', 'yes'),
]


def parse_success(path):
    txt = open(path).read()
    out = {}
    for key, short in (('regular/forward', 'forward'), ('flipped/reverse', 'reverse'), ('either/or', 'either'),
                       ('alt/alternate', 'alternate')):
        m = re.search(re.escape(key) + r'\s*:\s*([0-9.]+)', txt)
        out[short] = float(m.group(1))
    for key in ('collision', 'no_converge'):
        out[key] = int(re.search(r'^\s+' + key + r'\s*:\s*(\d+)', txt, re.M).group(1))
    out['cases'] = int(re.search(r'episodes\s*:\s*(\d+)', txt).group(1))
    return out


def seed_of(path):
    return int(re.search(r'_s(\d+)(?:_c1000)?/?$', path.rstrip('/')).group(1))


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print('wrote', path, len(rows), 'rows')


def stage_e_rows():
    rows = []
    for mid, label, role, res, _ in STAGE_E:
        for s, (scene, env) in SCENES.items():
            for d in sorted(glob.glob(res[s])):
                f = os.path.join(d, 'success_rate.txt')
                if not os.path.exists(f):
                    continue
                r = parse_success(f)
                rows.append(dict(scene=scene, env=env, model_id=mid, model=label, role=role, seed=seed_of(d),
                                 guarantee=GUARANTEED[mid], key='%s|%s|%d' % (scene, mid, seed_of(d)), source=f, **r))
    return rows


def free_rows():
    rows = []
    for mid, label, role, pat in FREE:
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, 'metrics.json')
            if not os.path.exists(f):
                continue
            m = json.load(open(f))
            r = dict(model_id=mid, model=label, role=role, seed=seed_of(d), run=os.path.basename(d),
                     guarantee=GUARANTEED[mid], key='%s|%s' % (mid, os.path.basename(d)), source=f)
            r.update({k: m.get(src, '') for k, src in FREE_KEYS})
            rows.append(r)
    return rows


def attack_rows():
    rows = []
    cache = {}
    for jf, ck, model, setting, guar in ATTACKS:
        cache.setdefault(jf, json.load(open(jf)))
        r = cache[jf][ck]
        a, y = r['attack'], r['attack_y']
        rows.append(dict(checkpoint=ck, model=model, setting=setting, guarantee=guar,
                         key='%s|%s' % (setting, model),
                         y_violated=y['viol_frac'], y_max=max(y['viol_rel_max'], 0.0), y_p99=max(y['viol_rel_p99'], 0.0),
                         y_in_domain=y['in_domain'] if y['viol_frac'] > 0 else '',
                         all_violated=a['viol_frac'], all_max=max(a['viol_rel_max'], 0.0),
                         all_in_domain=a['in_domain'] if a['viol_frac'] > 0 else '', triples=y['n'], source=jf))
    return rows


def beta_rows():
    bj = json.load(open('results/graph_method_triangle_triangle/beta.json'))
    rows = []
    for b in (10, 30, 100):
        ck = 'Experiments/graph_method_triangle_stageE/T_gmetric_beta%d_s0/latest.pt' % b
        f = R_NEW + '/T_gmetric_beta%d_s0_c1000/success_rate.txt' % b
        r = parse_success(f)
        y = bj[ck]['attack_y']
        rows.append(dict(beta=b, forward=r['forward'], collision=r['collision'], no_converge=r['no_converge'],
                         y_max=max(y['viol_rel_max'], 0.0), y_violated=y['viol_frac'],
                         source=f + ' ; results/graph_method_triangle_triangle/beta.json'))
    return rows


def lp_rows():
    rows = []
    for p, run in ((1, 'T_trimetric_p1_s0'), (2, 'T_trimetric_p2_s0'), (4, 'T_trimetric_p4_s0'),
                   (8, 'T_trimetric_b1g1_s0'), (16, 'T_trimetric_p16_s0'), (32, 'T_trimetric_p32_s0')):
        f = R_NEW + '/%s_c1000/success_rate.txt' % run
        r = parse_success(f)
        rows.append(dict(p=p, forward=r['forward'], collision=r['collision'], no_converge=r['no_converge'], source=f))
    return rows


def rerun_rows():
    a = json.load(open(E_NEW + '/T_trimetric_eq_s0/history.json'))
    b = json.load(open('Experiments/graph_method_triangle_rerun/T_trimetric_eq_s0/history.json'))
    rows = []
    for x, y in zip(a, b):
        same = all(x[k] == y[k] for k in ('val_hold_eik_p50', 'val_hold_eik_mean', 'train_loss'))
        rows.append(dict(step=x['step'], original_eik_p50=x['val_hold_eik_p50'], rerun_eik_p50=y['val_hold_eik_p50'],
                         original_loss=x['train_loss'], rerun_loss=y['train_loss'], identical='yes' if same else 'no',
                         source=E_NEW + '/T_trimetric_eq_s0/history.json ; Experiments/graph_method_triangle_rerun/T_trimetric_eq_s0/history.json'))
    return rows


def diagnostics(out):
    import numpy as np
    import torch
    from models.graph_method_triangle import model_function_graph as mf
    from models.graph_method_triangle.fields import decoder
    from models.graph_method_triangle.geometry import se3
    from models.graph_method_triangle.model_function_shape import ShapeFunction, coords_to_features

    share_rows, grad_rows = [], []
    bins = ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01))
    for mid, label, role, _, exps in STAGE_E:
        for s, (scene, env) in SCENES.items():
            for d in sorted(glob.glob(exps[s])):
                ckp = os.path.join(d, 'latest.pt')
                if not os.path.exists(os.path.join(d, 'train_summary.json')):
                    continue
                ck = torch.load(ckp, map_location='cpu')
                cfg = mf.default_config()
                cfg.update(ck['config'])
                cfg['device'] = 'cuda:0'
                f = ShapeFunction(cfg)
                f.network.load_state_dict(ck['model_state_dict'])
                f.network.eval()
                D = f.data
                idx = f.hold_idx[:8192]
                seed = seed_of(d)
                if f.network.has_latent:
                    with torch.no_grad():
                        a, b = f.network.predict_latent(D['g0'][idx], D['g1'][idx])
                        Qa, Qb = a / f.s_t ** 2, decoder.phi(b) / f.s_r ** 2
                    share = (Qb / (Qa + Qb).clamp_min(1e-12)).mean().item()
                    corr = torch.corrcoef(torch.stack([b, D['bfree'][idx]]))[0, 1].item()
                    share_rows.append(dict(scene=scene, model_id=mid, model=label, role=role, seed=seed,
                                           guarantee=GUARANTEED[mid],
                                           key='%s|%s|%d' % (scene, mid, seed), rotation_share=share,
                                           b_corr=corr, source=ckp))
                if scene != 'T-shape':
                    continue
                T, cx, cy = f.tangent_c(D['g0'][idx], D['g1'][idx], D['Sd'][idx, 0], D['Sa'][idx, 0],
                                        D['Sd'][idx, 1], D['Sa'][idx, 1], create_graph=False)
                n = torch.linalg.norm(cx.detach(), dim=-1)
                S = D['Sd'][idx, 0]
                for lo, hi in bins:
                    m = (S >= lo) & (S < hi)
                    grad_rows.append(dict(model_id=mid, model=label, role=role, seed=seed, guarantee=GUARANTEED[mid],
                                          speed_bin='%.2f-%.2f' % (lo, min(hi, 1.0)), bin_lo=lo,
                                          key='%s|%d|%.2f' % (mid, seed, lo), pairs=int(m.sum()),
                                          c_median=n[m].median().item(), source=ckp))
                print('diag', d, flush=True)
    write_csv(os.path.join(out, 'latent_share.csv'), share_rows,
              ['key', 'scene', 'model_id', 'model', 'role', 'guarantee', 'seed', 'rotation_share', 'b_corr', 'source'])
    write_csv(os.path.join(out, 'near_obstacle_grad.csv'), grad_rows,
              ['key', 'model_id', 'model', 'role', 'guarantee', 'seed', 'speed_bin', 'bin_lo', 'pairs', 'c_median',
               'source'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/graph_method_triangle_dashboard')
    p.add_argument('--diagnostics', action='store_true')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    write_csv(os.path.join(a.out, 'stage_e_seeds.csv'), stage_e_rows(),
              ['key', 'scene', 'env', 'model_id', 'model', 'role', 'guarantee', 'seed', 'forward', 'reverse', 'either', 'alternate',
               'collision', 'no_converge', 'cases', 'source'])
    write_csv(os.path.join(a.out, 'free_seeds.csv'), free_rows(),
              ['key', 'model_id', 'model', 'role', 'guarantee', 'seed', 'run'] + [k for k, _ in FREE_KEYS] + ['source'])
    write_csv(os.path.join(a.out, 'triangle_attack.csv'), attack_rows(),
              ['key', 'model', 'setting', 'guarantee', 'y_violated', 'y_max', 'y_p99', 'y_in_domain', 'all_violated',
               'all_max', 'all_in_domain', 'triples', 'checkpoint', 'source'])
    write_csv(os.path.join(a.out, 'lse_beta.csv'), beta_rows(),
              ['beta', 'forward', 'collision', 'no_converge', 'y_max', 'y_violated', 'source'])
    write_csv(os.path.join(a.out, 'lp_sweep.csv'), lp_rows(), ['p', 'forward', 'collision', 'no_converge', 'source'])
    write_csv(os.path.join(a.out, 'rerun_check.csv'), rerun_rows(),
              ['step', 'original_eik_p50', 'rerun_eik_p50', 'original_loss', 'rerun_loss', 'identical', 'source'])
    if a.diagnostics:
        diagnostics(a.out)


if __name__ == '__main__':
    main()
