"""Static pictures of constrained movement (numpy + matplotlib): the camera's own view and 3-d overviews.

camera_view renders what the camera at the apex of the pyramid sees: a pinhole camera at t + R (0, 0, D) looking
at the anchor (along -R e_z), image axes R e_x (width W) and R e_y (height H), field of view = the pyramid's, so
the image border is the pyramid's base rectangle around the followed shape. A small z-buffer rasterizer (near-plane
clipping, perspective-correct depth, flat shading) over the room's inner faces, the obstacle boxes and the shape.
"""
import numpy as np

_QUADS = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]   # -x +x -y +y -z +z
_BITS = np.array([[(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)

ROOM = {'floor': (0.62, 0.60, 0.55), 'ceiling': (0.93, 0.93, 0.93), 'wall': (0.85, 0.86, 0.88)}
SHAPE_COLOR = (0.93, 0.45, 0.13)


def box_tris(lo, hi, inward=False):
    """(12, 3, 3) triangles of an axis-aligned box (outward normals, or inward for the room)."""
    V = lo + _BITS * (hi - lo)
    T = []
    for a, b, c, d in _QUADS:
        T += [[V[a], V[b], V[c]], [V[a], V[c], V[d]]]
    T = np.asarray(T)                                                     # _QUADS wind inwards
    return T if inward else T[:, ::-1]


def obstacle_colors(scene):
    """One colour per obstacle box: trunks / crowns (forest), buildings (city), walls and slabs otherwise."""
    raw = scene.boxes_raw
    ext = raw[:, 1::2] - raw[:, 0::2]
    out = []
    for e in ext:
        if scene.name == 'forest':
            out.append((0.45, 0.32, 0.20) if max(e[0], e[2]) < 20 else (0.30, 0.55, 0.28))
        elif scene.name == 'city':
            out.append((0.55, 0.62, 0.72))
        else:
            out.append((0.50, 0.56, 0.66))
    return out


def scene_mesh(scene, shape=None, center=None, room_flags=False):
    """(tris (n, 3, 3), colours (n, 3)) of the room (inner faces), the obstacles and the shape at ``center``
    (a list of centres draws several copies); room_flags adds (n,) bool, True for the room's faces."""
    tris, cols = [], []
    room = box_tris(scene.room_lo, scene.room_hi, inward=True)
    for k in range(12):
        face = _QUADS[k // 2]
        kind = 'floor' if k // 2 == 2 else ('ceiling' if k // 2 == 3 else 'wall')
        tris.append(room[k])
        cols.append(ROOM[kind])
    for (lo, hi), c in zip(zip(scene.box_lo, scene.box_hi), obstacle_colors(scene)):
        T = box_tris(lo, hi)
        tris += list(T)
        cols += [c] * len(T)
    if shape is not None:
        for cen in ([center] if np.ndim(center) == 1 else center):
            for lo, hi in shape.boxes(cen):
                T = box_tris(lo, hi)
                tris += list(T)
                cols += [SHAPE_COLOR] * len(T)
    tris, cols = np.asarray(tris), np.asarray(cols, float)
    if room_flags:
        return tris, cols, np.arange(len(tris)) < 12
    return tris, cols


def _clip_near(P, near):
    """Sutherland-Hodgman clip of a camera-space polygon (k, 3) (depth = -z) to depth >= near."""
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        da, db = -a[2] - near, -b[2] - near
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            u = da / (da - db)
            out.append(a + u * (b - a))
    return np.asarray(out)


def rasterize(tris, cols, cam_pos, cam_R, tan_w, tan_h, width, height, near=1e-3, light=(0.3, 0.8, 0.5),
              cull=None, img=None, zbuf=None, alpha=None):
    """z-buffer image (height, width, 3) of world triangles seen by a pinhole camera at cam_pos with axes cam_R
    (columns: image right, image up, backwards), half-angle tangents tan_w, tan_h. cull (n,) bool: drop those
    triangles when they face away from the camera (the room's near walls seen from outside). Given img / zbuf and
    alpha, the triangles are blended over the image where they are in front, without writing depth."""
    img = np.ones((height, width, 3)) if img is None else img
    zbuf = np.full((height, width), np.inf) if zbuf is None else zbuf
    light = np.asarray(light, float) / np.linalg.norm(light)
    Q = np.einsum('ji,nkj->nki', cam_R, tris - cam_pos)                     # camera space
    ys, xs = np.mgrid[0:height, 0:width]
    px = (xs + 0.5) / width * 2 - 1
    py = 1 - (ys + 0.5) / height * 2
    for n_, (tri_w, tri_c, col) in enumerate(zip(tris, Q, cols)):
        nrm = np.cross(tri_w[1] - tri_w[0], tri_w[2] - tri_w[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-14:
            continue
        nrm /= nn
        if cull is not None and cull[n_] and nrm @ (cam_pos - tri_w.mean(0)) < 0:
            continue
        shade = np.clip(np.asarray(col) * (0.62 + 0.38 * abs(nrm @ light)), 0, 1)
        poly = _clip_near(tri_c, near)
        if len(poly) < 3:
            continue
        d = -poly[:, 2]
        sx = poly[:, 0] / (d * tan_w)
        sy = poly[:, 1] / (d * tan_h)
        for k in range(1, len(poly) - 1):
            idx = [0, k, k + 1]
            x, y, iz = sx[idx], sy[idx], 1.0 / d[idx]
            x0, x1 = np.clip([(x.min() + 1) / 2 * width - 1, (x.max() + 1) / 2 * width + 1], 0, width).astype(int)
            y0, y1 = np.clip([(1 - y.max()) / 2 * height - 1, (1 - y.min()) / 2 * height + 1], 0, height).astype(int)
            if x1 <= x0 or y1 <= y0:
                continue
            PX, PY = px[y0:y1, x0:x1], py[y0:y1, x0:x1]
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-14:
                continue
            w0 = ((y[1] - y[2]) * (PX - x[2]) + (x[2] - x[1]) * (PY - y[2])) / den
            w1 = ((y[2] - y[0]) * (PX - x[2]) + (x[0] - x[2]) * (PY - y[2])) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            depth = 1.0 / (w0 * iz[0] + w1 * iz[1] + w2 * iz[2])
            zb = zbuf[y0:y1, x0:x1]
            m = inside & (depth < zb)
            if alpha is None:
                zb[m] = depth[m]
                img[y0:y1, x0:x1][m] = shade
            else:
                sub = img[y0:y1, x0:x1]
                sub[m] = (1 - alpha) * sub[m] + alpha * np.asarray(col)
    return img, zbuf


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Camera axes (columns: right, up, backwards) of a camera at eye looking at target."""
    back = np.asarray(eye, float) - np.asarray(target, float)
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], 1)


def project(P, cam_pos, cam_R, tan_w, tan_h, width, height):
    """Pixel coordinates (n, 2) and depths (n,) of world points P (n, 3)."""
    Q = (np.asarray(P) - cam_pos) @ cam_R
    d = -Q[:, 2]
    x = (Q[:, 0] / (d * tan_w) + 1) / 2 * width
    y = (1 - Q[:, 1] / (d * tan_h)) / 2 * height
    return np.stack([x, y], 1), d


def visible(xy, d, zbuf, eps=2e-3):
    """Which projected points are in front of the z-buffer (not hidden by the opaque scene)."""
    h, w = zbuf.shape
    xi = np.clip(xy[:, 0].astype(int), 0, w - 1)
    yi = np.clip(xy[:, 1].astype(int), 0, h - 1)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h) & (d > 0)
    return inside & (d <= zbuf[yi, xi] * (1 + eps) + eps)


def camera_view(scene, shape, pyr, center, R, width=352, zoom=1.0):
    """What the camera sees (height = width * H / W): the pyramid's apex looking at the shape's centre. zoom > 1
    widens the lens by that factor; the pyramid's base rectangle is then the centre 1 / zoom of the image."""
    height = max(8, int(round(width * pyr.H / pyr.W)))
    tris, cols = scene_mesh(scene, shape, center)
    apex = np.asarray(center) + R @ np.array([0.0, 0.0, pyr.D])
    img, _ = rasterize(tris, cols, apex, R, zoom * pyr.W / 2 / pyr.D, zoom * pyr.H / 2 / pyr.D, width, height)
    return img


def pyramid_world(pyr, center, R):
    """(5, 3) world vertices (base corners, apex) of the pyramid anchored at center with rotation R."""
    return np.asarray(center) + pyr.verts @ np.asarray(R).T


def pyramid_faces(V):
    """Faces (lists of vertices) of a world pyramid (5, 3) for Poly3DCollection."""
    return [[V[0], V[1], V[2], V[3]]] + [[V[i], V[(i + 1) % 4], V[4]] for i in range(4)]


def box_faces(lo, hi):
    V = lo + _BITS * (hi - lo)
    return [[V[a], V[b], V[c], V[d]] for a, b, c, d in _QUADS]
