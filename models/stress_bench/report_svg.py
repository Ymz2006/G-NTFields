"""Inline-SVG chart helpers for the stress-bench report page (theme colours via CSS variables)."""
import math

PALETTE = ['var(--c0)', 'var(--c1)', 'var(--c2)', 'var(--c3)', 'var(--c4)', 'var(--c5)', 'var(--c6)']


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def heatmap(row_labels, col_labels, values, fmt='{:.0f}', vmin=0.0, vmax=100.0, cell_w=74, cell_h=30, label_w=250,
            title='', reverse=False, sub=None):
    """values[r][c] (None = missing). Colour: light -> accent with the value; reverse flips the scale."""
    top = 64
    W = label_w + cell_w * len(col_labels) + 10
    H = top + cell_h * len(row_labels) + 8
    out = ['<svg class="chart" viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, _esc(title))]
    if title:
        out.append('<text x="0" y="16" class="ct">{}</text>'.format(_esc(title)))
    for c, lab in enumerate(col_labels):
        x = label_w + cell_w * c + cell_w / 2
        out.append('<text x="{:.0f}" y="{}" class="cl" text-anchor="end" transform="rotate(-35 {:.0f} {})">{}</text>'.format(
            x, top - 6, x, top - 6, _esc(lab)))
    for r, lab in enumerate(row_labels):
        y = top + cell_h * r
        out.append('<text x="{}" y="{:.0f}" class="cl" text-anchor="end">{}</text>'.format(label_w - 8, y + cell_h / 2 + 4, _esc(lab)))
        for c in range(len(col_labels)):
            v = values[r][c]
            x = label_w + cell_w * c
            if v is None:
                out.append('<rect x="{}" y="{}" width="{}" height="{}" class="cell-na"/>'.format(x + 1, y + 1, cell_w - 2, cell_h - 2))
                continue
            t = min(1.0, max(0.0, (v - vmin) / (vmax - vmin)))
            if reverse:
                t = 1.0 - t
            out.append('<rect x="{}" y="{}" width="{}" height="{}" rx="3" fill="var(--heat)" fill-opacity="{:.2f}"/>'.format(
                x + 1, y + 1, cell_w - 2, cell_h - 2, 0.08 + 0.82 * t))
            out.append('<text x="{:.0f}" y="{:.0f}" class="cv{}" text-anchor="middle">{}</text>'.format(
                x + cell_w / 2, y + cell_h / 2 + 4, ' cv-dark' if t > 0.55 else '', fmt.format(v) + (sub[r][c] if sub else '')))
    out.append('</svg>')
    return '\n'.join(out)


def grouped_bars(group_labels, series_labels, values, errs=None, ymax=100.0, ylabel='%', title='', height=260,
                 group_w=None, yticks=None):
    """values[s][g] (series s, group g); errs same shape or None."""
    n_g, n_s = len(group_labels), len(series_labels)
    left, right, top, bottom = 44, 10, 36, 70
    gw = group_w or max(90, 14 * n_s + 24)
    W = left + gw * n_g + right
    H = height
    ph = H - top - bottom
    bw = (gw - 16) / n_s
    out = ['<svg class="chart" viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, _esc(title))]
    if title:
        out.append('<text x="0" y="16" class="ct">{}</text>'.format(_esc(title)))
    ticks = yticks or [i * ymax / 5 for i in range(6)]
    for tv in ticks:
        y = top + ph * (1 - tv / ymax)
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" class="grid"/>'.format(left, y, W - right, y))
        out.append('<text x="{}" y="{:.1f}" class="cl" text-anchor="end">{:g}</text>'.format(left - 6, y + 4, tv))
    out.append('<text x="12" y="{:.0f}" class="cl" text-anchor="middle" transform="rotate(-90 12 {:.0f})">{}</text>'.format(
        top + ph / 2, top + ph / 2, _esc(ylabel)))
    for g, gl in enumerate(group_labels):
        x0 = left + gw * g + 8
        for s in range(n_s):
            v = values[s][g]
            if v is None:
                continue
            h = ph * min(v, ymax) / ymax
            x = x0 + bw * s
            out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" fill="{}"><title>{}: {} {:.1f}</title></rect>'.format(
                x, top + ph - h, bw - 2, h, PALETTE[s % len(PALETTE)], _esc(gl), _esc(series_labels[s]), v))
            if errs and errs[s][g]:
                e = ph * errs[s][g] / ymax
                cx = x + (bw - 2) / 2
                out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" class="err"/>'.format(cx, top + ph - h - e, cx, min(top + ph, top + ph - h + e)))
        out.append('<text x="{:.0f}" y="{}" class="cl" text-anchor="end" transform="rotate(-30 {:.0f} {})">{}</text>'.format(
            x0 + (gw - 16) / 2, H - bottom + 16, x0 + (gw - 16) / 2, H - bottom + 16, _esc(gl)))
    out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" class="axis"/>'.format(left, top + ph, W - right, top + ph))
    out.append('</svg>')
    return '\n'.join(out)


def legend(series_labels):
    return '<div class="legend">' + ''.join(
        '<span><i style="background:{}"></i>{}</span>'.format(PALETTE[i % len(PALETTE)], _esc(l)) for i, l in enumerate(series_labels)) + '</div>'


def log_lines(x_labels, series_labels, values, title='', ylabel='predicted / true speed', ymin=0.5, ymax=50.0, height=280):
    """values[s][i] on a log y axis; x evenly spaced categories."""
    n = len(x_labels)
    left, right, top, bottom = 50, 16, 36, 50
    W, H = 560, height
    pw, ph = W - left - right, H - top - bottom
    out = ['<svg class="chart" viewBox="0 0 {} {}" role="img" aria-label="{}">'.format(W, H, _esc(title))]
    if title:
        out.append('<text x="0" y="16" class="ct">{}</text>'.format(_esc(title)))
    lo, hi = math.log10(ymin), math.log10(ymax)
    ypix = lambda v: top + ph * (1 - (math.log10(max(v, ymin)) - lo) / (hi - lo))  # noqa: E731
    for tv in (0.5, 1, 2, 5, 10, 20, 50):
        if ymin <= tv <= ymax:
            y = ypix(tv)
            out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" class="{}"/>'.format(left, y, W - right, y, 'axis' if tv == 1 else 'grid'))
            out.append('<text x="{}" y="{:.1f}" class="cl" text-anchor="end">{:g}x</text>'.format(left - 6, y + 4, tv))
    out.append('<text x="12" y="{:.0f}" class="cl" text-anchor="middle" transform="rotate(-90 12 {:.0f})">{}</text>'.format(
        top + ph / 2, top + ph / 2, _esc(ylabel)))
    xs = [left + pw * (i + 0.5) / n for i in range(n)]
    for i, xl in enumerate(x_labels):
        out.append('<text x="{:.0f}" y="{}" class="cl" text-anchor="middle">{}</text>'.format(xs[i], H - bottom + 18, _esc(xl)))
    out.append('<text x="{:.0f}" y="{}" class="cl" text-anchor="middle">true speed S (training label)</text>'.format(left + pw / 2, H - 6))
    for s, sl in enumerate(series_labels):
        pts = ['{:.1f},{:.1f}'.format(xs[i], ypix(v)) for i, v in enumerate(values[s]) if v is not None]
        out.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2"><title>{}</title></polyline>'.format(' '.join(pts), PALETTE[s % len(PALETTE)], _esc(sl)))
        for i, v in enumerate(values[s]):
            if v is not None:
                out.append('<circle cx="{:.1f}" cy="{:.1f}" r="3" fill="{}"/>'.format(xs[i], ypix(v), PALETTE[s % len(PALETTE)]))
    out.append('</svg>')
    return '\n'.join(out)
