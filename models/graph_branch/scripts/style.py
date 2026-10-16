"""Shared labels, colours and line styles of the graph_branch figures (matplotlib)."""

LABEL = {
    'orig': 'metric baseline (original NTFields)',
    'morig': 'metric baseline net (original, chart input)',
    'morig05': 'metric baseline net (original, chart input)',
    'ggram': 'graph_gram',
    'ggc': 'graph_gram + conformal term',
    'gmetric': 'graph_metric',
    'gmet05': 'graph_metric',
    'br1': 'branch K=1, strict cone',
    'br1s': 'branch K=1, strict cone',
    'br4': 'branch K=4, strict cone',
    'br4s': 'branch K=4, strict cone',
    'br4H': 'branch K=4, Huber residual',
    'br4Hs': 'branch K=4, Huber residual',
    'br4L': 'branch K=4, logsumexp residual',
    'br4P': 'branch K=4, group-lp residual',
    'bc4': 'branch K=4, conformal cone',
    'bc4e': 'branch K=4, conformal cone + end constraint',
    'bc1e': 'branch K=1, conformal cone + end constraint',
    'exact': 'exact T*',
    'tfree': 'speed-limit time T_free',
}
SHORT = dict(LABEL, orig='metric baseline', morig='metric baseline net', morig05='metric baseline net',
             br4='branch strict', br4s='branch strict', bc4='branch conformal', bc4e='branch conformal + end',
             ggc='graph_gram + conf.', br1='branch K=1 strict', br1s='branch K=1 strict', br4Hs='branch Huber',
             br4H='branch Huber')
COLOR = {
    'orig': '#6b4c9a', 'morig': '#6b4c9a', 'morig05': '#6b4c9a',
    'ggram': '#b54a3c', 'ggc': '#e0a08f',
    'gmetric': '#c58a1f', 'gmet05': '#c58a1f',
    'br1': '#8fa8db', 'br1s': '#8fa8db', 'br4': '#3b63ad', 'br4s': '#3b63ad',
    'br4H': '#5a7fc4', 'br4Hs': '#5a7fc4', 'br4L': '#8c6d46', 'br4P': '#9a9a9a',
    'bc4': '#0e7a71', 'bc4e': '#3fb8ab', 'bc1e': '#93d3cb',
    'exact': 'black', 'tfree': 'black',
}
DASH = {'bc4e': (0, (5, 1.6)), 'br4Hs': (0, (5, 1.6)), 'br4H': (0, (5, 1.6)), 'tfree': ':', 'morig05': '-',
        'gmet05': '-'}


def rc():
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.2,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'font.family': 'DejaVu Sans',
    })
