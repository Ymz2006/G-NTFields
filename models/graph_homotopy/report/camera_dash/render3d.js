// Painter's-algorithm renderer for the camera-follow scenes (plain 2D canvas, no libraries).
// Units: the normalized scene frame of the test sets (y up). Pose = [x, y, z, rx, ry, rz], r in turns.
(function () {
  const V3 = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s], dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: a => Math.hypot(a[0], a[1], a[2]), norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; },
  };
  function rotMat(r) {  // rotation vector in turns -> 3x3 (rows)
    const th = 2 * Math.PI * V3.len(r);
    if (th < 1e-9) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const [x, y, z] = V3.mul(r, 1 / V3.len(r)), c = Math.cos(th), s = Math.sin(th), t = 1 - c;
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]];
  }
  const apply = (R, t, p) => [R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2]];
  function posed(pose, p) { return apply(rotMat(pose.slice(3, 6)), pose.slice(0, 3), p); }

  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function shade(rgb, k, a) { return 'rgba(' + rgb.map(v => Math.round(Math.min(255, v * k))).join(',') + ',' + a + ')'; }

  function boxTris(lo, hi) {
    const c = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]];
    const q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];   // outward quads
    const tris = [];
    for (const f of q) { tris.push([c[f[0]], c[f[1]], c[f[2]]]); tris.push([c[f[0]], c[f[2]], c[f[3]]]); }
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].map(e => [c[e[0]], c[e[1]]]);
    return { tris, edges };
  }

  // A view: eye, target, vertical fov; projects to canvas pixels.
  function makeCamera(eye, target, fovDeg, W, H) {
    const f = V3.norm(V3.sub(target, eye)), s = V3.norm(V3.cross(f, [0, 1, 0])), u = V3.cross(s, f);
    const fl = 0.5 * H / Math.tan(0.5 * fovDeg * Math.PI / 180);
    return {
      eye, f, s, u,
      proj(p) { const d = V3.sub(p, eye); const z = V3.dot(d, f); return [W / 2 + fl * V3.dot(d, s) / z, H / 2 - fl * V3.dot(d, u) / z, z]; },
    };
  }

  // scene: {tris: [{p:[a,b,c], color, alpha, edge}], lines: [{p:[a,b], color, width, alpha}], points: [{p, r, color}]}
  function render(ctx, W, H, cam, scene, opts) {
    const light = V3.norm(opts.light || [0.4, 1.0, 0.6]);
    const items = [];
    for (const t of scene.tris) {
      const P = t.p.map(cam.proj);
      if (P.some(q => q[2] < 0.02)) continue;
      const n = V3.norm(V3.cross(V3.sub(t.p[1], t.p[0]), V3.sub(t.p[2], t.p[0])));
      let k = 0.55 + 0.45 * Math.max(0, V3.dot(n, light)) + 0.12 * Math.max(0, n[1]);
      if (t.flat) k = 1;
      items.push({ z: (P[0][2] + P[1][2] + P[2][2]) / 3 + (t.bias || 0), kind: 'tri', P, fill: shade(t.rgb, k, t.alpha == null ? 1 : t.alpha), edge: t.edge, edgeColor: t.edgeColor });
    }
    for (const l of scene.lines) {
      const P = l.p.map(cam.proj);
      if (P.some(q => q[2] < 0.02)) continue;
      items.push({ z: (P[0][2] + P[1][2]) / 2 + (l.bias || 0), kind: 'line', P, color: l.color, width: l.width, alpha: l.alpha == null ? 1 : l.alpha, dash: l.dash });
    }
    for (const pt of scene.points || []) {
      const P = cam.proj(pt.p);
      if (P[2] < 0.02) continue;
      items.push({ z: P[2] + (pt.bias || 0), kind: 'point', P, r: pt.r, color: pt.color, stroke: pt.stroke });
    }
    items.sort((a, b) => b.z - a.z);
    for (const it of items) {
      if (it.kind === 'tri') {
        ctx.beginPath(); ctx.moveTo(it.P[0][0], it.P[0][1]); ctx.lineTo(it.P[1][0], it.P[1][1]); ctx.lineTo(it.P[2][0], it.P[2][1]); ctx.closePath();
        ctx.fillStyle = it.fill; ctx.fill();
        if (it.edge) { ctx.strokeStyle = it.edgeColor || it.fill; ctx.lineWidth = 0.6; ctx.stroke(); }
        else { ctx.strokeStyle = it.fill; ctx.lineWidth = 0.5; ctx.stroke(); }   // hide seams
      } else if (it.kind === 'line') {
        ctx.beginPath(); ctx.moveTo(it.P[0][0], it.P[0][1]); ctx.lineTo(it.P[1][0], it.P[1][1]);
        ctx.globalAlpha = it.alpha; ctx.strokeStyle = it.color; ctx.lineWidth = it.width; ctx.lineCap = 'round';
        if (it.dash) ctx.setLineDash(it.dash); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      } else {
        ctx.beginPath(); ctx.arc(it.P[0], it.P[1], it.r, 0, 2 * Math.PI); ctx.fillStyle = it.color; ctx.fill();
        if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = 1.2; ctx.stroke(); }
      }
    }
  }

  // Z-buffer rasterizer for the camera's own view (painter's sorting breaks when big faces pass close to the camera).
  // tris: [{c: [[x, y, z] x 3] in camera coordinates (x right, y up, z forward), rgb: [r, g, b], k: shade, obj: id, kind}]
  // Returns per-pixel colour, depth (z) and object id; near-plane clipping keeps faces the camera flies past.
  function rasterize(W, H, fl, tris, near) {
    const iz = new Float32Array(W * H), id = new Int32Array(W * H).fill(-1), kind = new Uint8Array(W * H);
    const rgbBuf = new Float32Array(W * H * 3);
    const cx = W / 2, cy = H / 2;
    for (const t of tris) {
      // Sutherland-Hodgman against z >= near
      let poly = t.c, out = [];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length], ina = a[2] >= near, inb = b[2] >= near;
        if (ina) out.push(a);
        if (ina !== inb) { const u = (near - a[2]) / (b[2] - a[2]); out.push([a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), near]); }
      }
      if (out.length < 3) continue;
      const P = out.map(q => [cx + fl * q[0] / q[2], cy - fl * q[1] / q[2], 1 / q[2]]);
      for (let j = 1; j + 1 < P.length; j++) {
        const A = P[0], B = P[j], Cq = P[j + 1];
        const area = (B[0] - A[0]) * (Cq[1] - A[1]) - (B[1] - A[1]) * (Cq[0] - A[0]);
        if (Math.abs(area) < 1e-12) continue;
        const x0 = Math.max(0, Math.floor(Math.min(A[0], B[0], Cq[0]))), x1 = Math.min(W - 1, Math.ceil(Math.max(A[0], B[0], Cq[0])));
        const y0 = Math.max(0, Math.floor(Math.min(A[1], B[1], Cq[1]))), y1 = Math.min(H - 1, Math.ceil(Math.max(A[1], B[1], Cq[1])));
        if (x0 > x1 || y0 > y1) continue;
        const inv = 1 / area;
        for (let y = y0; y <= y1; y++) {
          const py = y + 0.5;
          for (let x = x0; x <= x1; x++) {
            const px = x + 0.5;
            const w0 = ((B[0] - px) * (Cq[1] - py) - (B[1] - py) * (Cq[0] - px)) * inv;
            const w1 = ((Cq[0] - px) * (A[1] - py) - (Cq[1] - py) * (A[0] - px)) * inv;
            const w2 = 1 - w0 - w1;
            if (w0 < 0 || w1 < 0 || w2 < 0) continue;
            const z = w0 * A[2] + w1 * B[2] + w2 * Cq[2], i = y * W + x;
            if (z <= iz[i]) continue;
            iz[i] = z; id[i] = t.obj; kind[i] = t.kind;
            rgbBuf[3 * i] = t.rgb[0] * t.k; rgbBuf[3 * i + 1] = t.rgb[1] * t.k; rgbBuf[3 * i + 2] = t.rgb[2] * t.k;
          }
        }
      }
    }
    return { iz, id, kind, rgb: rgbBuf };
  }

  window.R3 = { V3, rotMat, posed, boxTris, makeCamera, render, rasterize, hexToRgb, shade };
})();
