"""Generate the LaTeX tables of report.tex directly from saved evaluation outputs.

    python3 models/graph_method/report/make_tables.py      (run from the repo root)

Reads results/graph_method*/**/metrics.json (obstacle-free evaluator),
results/graph_method*/**/success_rate.txt (Shape-NTFields evaluator) and the
training histories of the Fourier-sigma sweep; writes report/tables/*.tex.
Pure standard library, so it runs on the host python.
"""
import glob
import json
import math
import os
import re

ROOT = os.getcwd()
OUT = os.path.join(ROOT, 'models', 'graph_method', 'report', 'tables')
LIE = '/media/jeffrey/Lie-Shape-NTFields/results/output_3d'


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else None
    return m, s, len(vals)


def fmt(m, s, digits, maxd=4):
    """mean +- std with the std shown to two significant digits (digits = fallback, at most maxd)."""
    if m is None:
        return '--'
    if s is not None and s > 0:
        digits = min(maxd, max(0, 1 - int(math.floor(math.log10(s)))))
    f = '{:.%df}' % digits
    if s is None:
        return '$' + f.format(m) + '$'
    return '$' + f.format(m) + r'\pm' + f.format(s) + '$'


def load_metrics(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(ROOT, 'results', 'graph_method_final',
                                                                  pattern, 'metrics.json')))]


def cell(ms, key, scale=1.0, digits=2):
    m, s, _ = mean_std([x.get(key) * scale if x.get(key) is not None else None for x in ms])
    return fmt(m, s, digits)


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(text)
    print('wrote', name)


# --------------------------------------------------------------------------- obstacle-free
MAIN = [
    (r'\texttt{graph\_gram} (proposed)', 'graph_gram_s*'),
    (r'\texttt{graph\_sqdist}', 'graph_sqdist_s*'),
    (r'metric, best recipe', 'metric_best_recipe_s*'),
    (r'metric, our recipe', 'metric_same_recipe_s*'),
    (r'metric, NTFields loss', 'metric_ntfields_loss_s*'),
    (r'scalar $T$', 'scalar_s*'),
]
ABL = [
    (r'\texttt{graph\_gram} (full)', 'graph_gram_s*'),
    (r'w/o curriculum', 'abl_graph_gram_no_curriculum_s*'),
    (r'w/o anchors', 'abl_graph_gram_no_anchor_s*'),
    (r'w/o input skip', 'abl_graph_gram_no_skip_s*'),
    (r'w/o Bellman', 'abl_graph_gram_no_bellman_s*'),
    (r'band $10^{-3}$', 'abl_graph_gram_band1e-3_s*'),
    (r'stage B (labels)', 'stageB*'),
    (r'\texttt{graph\_metric}', 'graph_metric_s*'),
]


def field_table(rows):
    lines = []
    for label, pat in rows:
        ms = load_metrics(pat)
        lines.append(' & '.join([label, str(len(ms)),
                                 cell(ms, 'uniform_rel_p50', 100, 3), cell(ms, 'uniform_rel_p95', 100, 2),
                                 cell(ms, 'mixture_rel_p95', 100, 2), cell(ms, 'angle_179-180_rel_p50', 100, 2),
                                 cell(ms, 'exact_pi_abs_mean', 1, 4), cell(ms, 'uniform_eik_x_regular_p50', 1, 4)])
                     + r' \\')
    return '\n'.join(lines) + '\n'


def plan_table(rows):
    lines = []
    for label, pat in rows:
        ms = load_metrics(pat)
        lines.append(' & '.join([label, str(len(ms)),
                                 cell(ms, 'plan_gradient_success', 100, 1),
                                 cell(ms, 'plan_gradient_cost_ratio_p50', 1, 4),
                                 cell(ms, 'plan_gradient_cost_ratio_p95', 1, 3),
                                 cell(ms, 'plan_sampling_success', 100, 1),
                                 cell(ms, 'plan_sampling_cost_ratio_p50', 1, 3)])
                     + r' \\')
    return '\n'.join(lines) + '\n'


def angle_table():
    bins = ['0-90', '90-150', '150-170', '170-179', '179-180']
    rows = MAIN[:1] + MAIN[2:3] + MAIN[5:6]
    lines = []
    for label, pat in rows:
        ms = load_metrics(pat)
        lines.append(' & '.join([label] + [cell(ms, 'angle_{}_rel_p50'.format(b), 100, 3) for b in bins]) + r' \\')
    return '\n'.join(lines) + '\n'


def sci(v):
    """LaTeX number: plain for moderate values, a \\times 10^{k} for tiny ones."""
    if v == 0 or 1e-3 <= abs(v) < 1e4:
        return '%.5g' % v
    e = int(math.floor(math.log10(abs(v))))
    return '%.2g\\times10^{%d}' % (v / 10 ** e, e)


def analytic_numbers():
    m = json.load(open(os.path.join(ROOT, 'results', 'graph_method', 'analytic', 'metrics.json')))
    keys = {'uniform_rel_p95': 'arelpninetyfive', 'exact_pi_abs_mean': 'aexactpi',
            'uniform_eik_x_regular_p95': 'aeikpninetyfive', 'plan_gradient_cost_ratio_p50': 'agradcost',
            'plan_sampling_cost_ratio_p50': 'asampcost', 'plan_gradient_success': 'agradsucc',
            'plan_sampling_success': 'asampsucc'}
    out = []
    fixed = {'agradcost': '%.5f', 'asampcost': '%.3f'}
    for k, macro in keys.items():
        val = fixed[macro] % m[k] if macro in fixed else sci(m[k])
        out.append('\\newcommand{\\%s}{%s}' % (macro, val))
    return '\n'.join(out) + '\n'


def latent_numbers():
    ms = load_metrics('graph_gram_s*')
    out = []
    for key, macro in (('uniform_latent_a_p50', 'latentA'), ('uniform_latent_b_p50', 'latentB'),
                       ('near_diag_abs_mean', 'neardiag'), ('slice_abs_near_pi_max', 'slicepi')):
        m, s, n = mean_std([x[key] for x in ms])
        out.append('\\newcommand{\\%s}{%s}' % (macro, sci(float('%.2g' % m))))
    return '\n'.join(out) + '\n'


def sweep_table():
    archs = [('graph_gram', r'\texttt{graph\_gram}'), ('graph_sqdist', r'\texttt{graph\_sqdist}'),
             ('graph_joint', r'\texttt{graph\_joint}'), ('scalar', 'scalar'), ('metric', 'metric head')]
    sigmas = ['0.03', '0.08', '0.15', '0.25']
    lines = []
    for a, label in archs:
        cells = []
        for s in sigmas:
            p = os.path.join(ROOT, 'Experiments', 'graph_method_sweep1', 'sup_{}_sig{}'.format(a, s), 'history.json')
            if os.path.exists(p):
                h = json.load(open(p))[-1]
                cells.append('${:.2f}\\,/\\,{:.1f}$'.format(100 * h['val_uniform_rel_p50'], 100 * h['val_mixture_rel_p95']))
            else:
                cells.append('--')
        lines.append(' & '.join([label] + cells) + r' \\')
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- obstacles
def parse_sr(path):
    txt = open(path).read()
    out = {}
    for key, short in (('regular/forward', 'forward'), ('flipped/reverse', 'reverse'),
                       ('either/or', 'either'), ('alt/alternate', 'alternate')):
        m = re.search(re.escape(key) + r'\s*:\s*([0-9.]+)', txt)
        if m:
            out[short] = 100 * float(m.group(1))
    for key in ('collision', 'no_converge'):
        m = re.search(r'^\s+' + key + r'\s*:\s*(\d+)', txt, re.M)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r'episodes\s*:\s*(\d+)', txt)
    out['episodes'] = int(m.group(1)) if m else None
    return out


def sr_rows(groups):
    lines = []
    for label, pattern in groups:
        files = sorted(glob.glob(os.path.join(pattern, 'success_rate.txt')))
        rs = [parse_sr(f) for f in files]
        if not rs:
            continue
        col = lambda k: [r[k] for r in rs if k in r]
        cells = [label, str(len(rs))]
        for k in ('forward', 'reverse', 'either', 'alternate'):
            m, s, _ = mean_std(col(k))
            cells.append(fmt(m, s, 1, maxd=1))
        for k in ('collision', 'no_converge'):
            m, s, _ = mean_std(col(k))
            cells.append(fmt(m, s, 0, maxd=0))
        lines.append(' & '.join(cells) + r' \\')
    return '\n'.join(lines) + '\n'


def stage_e_tables():
    R = os.path.join(ROOT, 'results')
    t = sr_rows([
        (r'metric NTFields baseline, native pipeline$^\dagger$', os.path.join(LIE, 'Tshape3d_env1_metric_baseline_s*')),
        (r'metric head, our pipeline', os.path.join(R, 'graph_method_stageE', 'stageE_metric_s*_c1000')),
        (r'\texttt{graph\_metric} (graph latents)', os.path.join(R, 'graph_method_stageE', 'stageE_graph_metric_s*_c1000')),
        (r'\texttt{graph\_gram}', os.path.join(R, 'graph_method_stageE', 'stageE_graph_gram_s[0-9]_c1000')),
        (r'\texttt{graph\_gram}, $k_a=128$', os.path.join(R, 'graph_method_stageE', 'stageE_graph_gram_ka128_s0_c1000')),
    ])
    write('stageE_T.tex', t)
    for env in ('L', 'A'):
        rows = sr_rows([
            (r'metric head, our pipeline', os.path.join(R, 'graph_method_stageE2', env + '_metric_s*_c1000')),
            (r'\texttt{graph\_metric} (graph latents)', os.path.join(R, 'graph_method_stageE2', env + '_graph_metric_s*_c1000')),
            (r'\texttt{graph\_gram}', os.path.join(R, 'graph_method_stageE2', env + '_graph_gram_s*_c1000')),
        ])
        write('stageE_{}.tex'.format(env), rows)
    # exploration on the first 200 cases
    expl = [
        ('graph\\_gram', 'sq. residual, aniso speeds', r'$1.0/0.3$', 'graph_method_shape1/g_aniso_sq_12k_c200'),
        ('graph\\_gram', 'sq. residual, iso speeds', r'$1.0/0.3$', 'graph_method_shape1/g_iso_sq_12k_c200'),
        ('graph\\_gram', r'$\sqrt{\cdot}$ residual, iso', r'$1.0/0.3$', 'graph_method_shape1/g_iso_sqrt_12k_c200'),
        ('graph\\_gram', r'+ non-detached $e^{-T/2}$ weight', r'$1.0/0.3$', 'graph_method_shape3/g_iso_sqrt_ewg_c200'),
        ('metric', r'+ non-detached $e^{-T/2}$ weight', r'$1.0/0.3$', 'graph_method_shape3/m_iso_sqrt_ewg_c200'),
        ('graph\\_gram', 'no input skip', r'$1.0/0.3$', 'graph_method_shape3/g_noskip_iso_sqrt_ewg_c200'),
        ('graph\\_gram', r'+ speed lower bound $T\ge T_{\mathrm{free}}$', r'$1.0/0.3$', 'graph_method_shape4/g_skip_lb_c200'),
        ('metric', r'+ speed lower bound', r'$1.0/0.3$', 'graph_method_shape4/m_lb_c200'),
        ('graph\\_gram', 'AdamW (wd 0.5)', r'$1.0/0.3$', 'graph_method_shape5/g_adamw_c200'),
        ('graph\\_gram', 'Adam, cosine LR', r'$1.0/0.1$', 'graph_method_shape5/g_sr01_c200'),
        ('graph\\_gram', 'AdamW', r'$1.0/0.1$', 'graph_method_shape5/g_sr01_adamw_c200'),
        ('metric', 'Adam, cosine LR', r'$1.0/0.1$', 'graph_method_shape5/m_sr01_c200'),
        ('metric', 'AdamW', r'$1.0/0.1$', 'graph_method_shape5/m_sr01_adamw_c200'),
        ('graph\\_gram', 'AdamW, aniso speeds', r'$1.0/0.1$', 'graph_method_shape7/g_t1_r01_aniso_c200'),
        ('graph\\_gram', 'AdamW', r'$2.0/0.1$', 'graph_method_shape7/g_t2_r01_c200'),
        ('graph\\_gram', 'AdamW', r'$0.5/0.1$', 'graph_method_shape7/g_t05_r01_c200'),
        ('graph\\_gram', 'AdamW', r'$0.5/0.05$', 'graph_method_shape10/g_t05_r005_c200'),
        ('graph\\_gram', 'AdamW', r'$0.3/0.05$', 'graph_method_shape10/g_t03_r005_c200'),
        ('metric', 'AdamW', r'$0.5/0.1$', 'graph_method_shape10/m_t05_r01_c200'),
        ('metric', r'coordinate replica of baseline', r'$1.0$', 'graph_method_shape6/c_m_c200'),
        ('metric', r'replica + rollback, resampled $B$', r'$1.0$', 'graph_method_shape8/c_m_rb_c200'),
        ('metric', r'replica, no clipping, no mask', r'$1.0$', 'graph_method_shape9/c_m_exact_c200'),
        ('metric baseline', 'native pipeline, checkpoint s1', r'$1.0$', 'stageE_baseline_ref/metric_baseline_s1_c200'),
    ]
    lines = []
    for head, desc, sig, rel in expl:
        p = os.path.join(R, rel, 'success_rate.txt')
        if not os.path.exists(p):
            lines.append('% missing ' + rel)
            continue
        r = parse_sr(p)
        lines.append(r'\texttt{%s} & %s & %s & $%.1f$ & %s & %s \\' % (
            head, desc, sig, r['forward'], r.get('collision', '--'), r.get('no_converge', '--')))
    write('stageE_explore.tex', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    write('main_field.tex', field_table(MAIN))
    write('main_plan.tex', plan_table(MAIN))
    write('ablation_field.tex', field_table(ABL))
    write('ablation_plan.tex', plan_table(ABL))
    write('angle.tex', angle_table())
    write('numbers.tex', analytic_numbers() + latent_numbers())
    write('sweep.tex', sweep_table())
    stage_e_tables()
