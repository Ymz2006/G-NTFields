// Scene views of the camera-follow demo on top of R3 (render3d.js): the body figure and the plan panels.
(function () {
  const { V3, rotMat, posed, boxTris, makeCamera, render, rasterize, hexToRgb } = window.R3;
  const C = {   // fixed "photo" colors of the renders (the same in both themes)
    floor: '#e9e7e1', wall: '#f3f2ee', grid: '#d6d3cb', trunk: '#8c6b4c', crown: '#6c9a63', building: '#aab3bf',
    bird: '#eb6834', birdEdge: '#b94d22', frustum: '#2a78d6', frustumHit: '#e34948', camPath: '#2a78d6', birdPath: '#eb6834',
    cam: '#1c5cab', goal: '#1baf7a', bg: '#fbfaf7',
  };
  const rgb = Object.fromEntries(Object.entries(C).map(([k, v]) => [k, hexToRgb(v)]));

  function pyramid(D) {   // triangles of the view pyramid in the body frame
    const a = D.camera.apex, b = D.camera.base;
    const t = [];
    for (let i = 0; i < 4; i++) t.push([b[i], b[(i + 1) % 4], a]);
    t.push([b[0], b[1], b[2]]); t.push([b[0], b[2], b[3]]);
    return t;
  }
  function meshTris(M) { return M.F.map(f => [M.V[f[0]], M.V[f[1]], M.V[f[2]]]); }

  function addBody(sc, D, pose, o) {
    const R = rotMat(pose.slice(3, 6)), t = pose.slice(0, 3);
    const P = p => [R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0], R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1], R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2]];
    for (const tri of meshTris(D.bird)) sc.tris.push({ p: tri.map(P), rgb: rgb.bird, alpha: o.alpha, edge: false });
    const fr = o.hit ? rgb.frustumHit : rgb.frustum;
    for (const tri of pyramid(D)) sc.tris.push({ p: tri.map(P), rgb: fr, alpha: o.hit ? 0.42 : 0.22, flat: false, bias: -0.002 });
    sc.points.push({ p: P(D.camera.apex), r: o.camR || 4, color: o.hit ? C.frustumHit : C.cam, stroke: '#ffffff', bias: -0.01 });
  }

  // The body alone: a rear three-quarter view.
  window.drawBody = function (canvas, D, W, H) {
    const ctx = canvas.getContext('2d');
    canvas.width = W; canvas.height = H;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const sc = { tris: [], lines: [], points: [] };
    addBody(sc, D, [0, 0, 0, 0, 0, 0], { alpha: 1, camR: 6 });
    // the camera's lines of sight to the base corners
    for (const b of D.camera.base) sc.lines.push({ p: [D.camera.apex, b], color: C.cam, width: 1, alpha: 0.6, dash: [4, 4] });
    const cam = makeCamera([-0.42, 0.26, 0.5], [-0.02, 0.0, 0.0], 36, W, H);
    render(ctx, W, H, cam, sc, { light: [-0.3, 1, 0.6] });
    return cam;
  };

  // A scene with one plan: room, obstacles, both paths, the body at a few poses, the frame where the view breaks.
  window.drawPlan = function (canvas, D, env, planKey, W, H, opts) {
    opts = opts || {};
    const S = D.scenes[env], plan = S.plans[planKey];
    const ctx = canvas.getContext('2d');
    canvas.width = W; canvas.height = H;
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const lo = S.room.lo, hi = S.room.hi, cen = V3.mul(V3.add(lo, hi), 0.5);
    // view: from the side of the start -> goal direction, elevated
    const d = V3.norm([S.goal[0] - S.start[0], 0, S.goal[2] - S.start[2]]);
    let side = [-d[2], 0, d[0]];
    if (opts.flip) side = V3.mul(side, -1);
    const el = (opts.elev || 34) * Math.PI / 180, dist = opts.dist || 2.05;
    const eye = V3.add(cen, V3.add(V3.mul(side, dist * Math.cos(el)), [0, dist * Math.sin(el), 0]));
    const cam = makeCamera(eye, V3.add(cen, [0, -0.04, 0]), 30, W, H);
    const sc = { tris: [], lines: [], points: [] };
    // floor and the two far walls, painted first
    const fx = eye[0] < cen[0] ? hi[0] : lo[0], fz = eye[2] < cen[2] ? hi[2] : lo[2];
    sc.tris.push({ p: [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]]], rgb: rgb.floor, flat: true, bias: 50 });
    sc.tris.push({ p: [[lo[0], lo[1], lo[2]], [hi[0], lo[1], hi[2]], [lo[0], lo[1], hi[2]]], rgb: rgb.floor, flat: true, bias: 50 });
    sc.tris.push({ p: [[fx, lo[1], lo[2]], [fx, hi[1], lo[2]], [fx, hi[1], hi[2]]], rgb: rgb.wall, flat: true, bias: 60 });
    sc.tris.push({ p: [[fx, lo[1], lo[2]], [fx, hi[1], hi[2]], [fx, lo[1], hi[2]]], rgb: rgb.wall, flat: true, bias: 60 });
    sc.tris.push({ p: [[lo[0], lo[1], fz], [hi[0], lo[1], fz], [hi[0], hi[1], fz]], rgb: rgb.wall, flat: true, bias: 60 });
    sc.tris.push({ p: [[lo[0], lo[1], fz], [hi[0], hi[1], fz], [lo[0], hi[1], fz]], rgb: rgb.wall, flat: true, bias: 60 });
    for (let x = Math.ceil(lo[0] * 10) / 10; x <= hi[0]; x += 0.1) sc.lines.push({ p: [[x, lo[1], lo[2]], [x, lo[1], hi[2]]], color: C.grid, width: 1, bias: 40 });
    for (let z = Math.ceil(lo[2] * 10) / 10; z <= hi[2]; z += 0.1) sc.lines.push({ p: [[lo[0], lo[1], z], [hi[0], lo[1], z]], color: C.grid, width: 1, bias: 40 });
    // room edges (far ones)
    sc.lines.push({ p: [[fx, lo[1], lo[2]], [fx, hi[1], lo[2]]], color: C.grid, width: 1, bias: 39 });
    sc.lines.push({ p: [[fx, lo[1], hi[2]], [fx, hi[1], hi[2]]], color: C.grid, width: 1, bias: 39 });
    sc.lines.push({ p: [[lo[0], lo[1], fz], [lo[0], hi[1], fz]], color: C.grid, width: 1, bias: 39 });
    sc.lines.push({ p: [[hi[0], lo[1], fz], [hi[0], hi[1], fz]], color: C.grid, width: 1, bias: 39 });
    // obstacles
    for (const ob of S.obstacles) ob.boxes.forEach((bx, i) => {
      const col = ob.name.startsWith('tree') ? (i === 0 ? rgb.trunk : rgb.crown) : rgb.building;
      const edgeCol = ob.name.startsWith('tree') ? (i === 0 ? '#6b4f36' : '#4e7548') : '#7e8894';
      const { tris, edges } = boxTris(bx[0], bx[1]);
      for (const t of tris) sc.tris.push({ p: t, rgb: col, edge: false });
      for (const e of edges) sc.lines.push({ p: e, color: edgeCol, width: 1, alpha: 0.9, bias: -0.003 });
    });
    // paths
    const poses = plan.poses;
    for (let i = 1; i < poses.length; i++) {
      sc.lines.push({ p: [poses[i - 1].slice(0, 3), poses[i].slice(0, 3)], color: C.birdPath, width: 3, bias: -0.004 });
      sc.lines.push({ p: [posed(poses[i - 1], D.camera.apex), posed(poses[i], D.camera.apex)], color: C.camPath, width: 2.2, alpha: 0.9, bias: -0.004 });
    }
    // the body at a few poses: start, middle(s), end; the frame where the camera loses the bird in red
    const hit = plan.view_lost_at != null ? plan.view_lost_at : (plan.bird_hit_at != null ? plan.bird_hit_at : plan.body_hit_at);
    const n = poses.length - 1;
    let idx = [0, Math.round(n / 2), n];
    if (opts.ghosts === 'rich') idx = [0, Math.round(n / 3), Math.round(2 * n / 3), n];
    if (hit != null) idx = [0, hit, n].filter((v, i, a) => a.indexOf(v) === i);
    idx.forEach((k, j) => addBody(sc, D, poses[k], { alpha: (k === 0 || k === n) ? 1 : 0.9, hit: k === hit && hit != null }));
    // start and goal markers
    sc.points.push({ p: S.start.slice(0, 3), r: 5, color: C.bird, stroke: '#ffffff', bias: -0.02 });
    sc.points.push({ p: S.goal.slice(0, 3), r: 6, color: C.goal, stroke: '#ffffff', bias: -0.02 });
    render(ctx, W, H, cam, sc, { light: [0.35, 1, 0.55] });
    return cam;
  };

  // ---- camera POV: what the camera sees at one waypoint. The image is the view pyramid's cross-section
  // (51 x 25 degrees); an obstacle (or the floor) nearer than the pyramid's base is inside the pyramid, i.e. it
  // blocks the line of sight, and is tinted red; everything beyond the bird is faded.
  const KIND = { room: 1, obst: 2, bird: 3 };
  const POVC = { sky: [235, 238, 241], floor: [214, 210, 200], wall: [226, 226, 222], ceil: [238, 238, 236], block: [227, 73, 72] };

  function povScene(D, S) {   // static world triangles of a scene (room inner faces + obstacles)
    if (S._pov) return S._pov;
    const tris = [];
    const lo = S.room.lo, hi = S.room.hi;
    const quad = (a, b, c, d, rgb, kind, obj) => { tris.push({ w: [a, b, c], rgb, kind, obj }); tris.push({ w: [a, c, d], rgb, kind, obj }); };
    const X = [lo[0], hi[0]], Y = [lo[1], hi[1]], Z = [lo[2], hi[2]];
    quad([X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[0], Z[1]], [X[0], Y[0], Z[1]], POVC.floor, KIND.room, 1);
    quad([X[0], Y[1], Z[0]], [X[1], Y[1], Z[0]], [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]], POVC.ceil, KIND.room, 2);
    quad([X[0], Y[0], Z[0]], [X[0], Y[1], Z[0]], [X[0], Y[1], Z[1]], [X[0], Y[0], Z[1]], POVC.wall, KIND.room, 3);
    quad([X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[1], Y[1], Z[1]], [X[1], Y[0], Z[1]], POVC.wall, KIND.room, 4);
    quad([X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[0], Y[1], Z[0]], POVC.wall, KIND.room, 5);
    quad([X[0], Y[0], Z[1]], [X[1], Y[0], Z[1]], [X[1], Y[1], Z[1]], [X[0], Y[1], Z[1]], POVC.wall, KIND.room, 6);
    let obj = 10;
    for (const ob of S.obstacles) ob.boxes.forEach((bx, i) => {
      const col = ob.name.startsWith('tree') ? (i === 0 ? rgb.trunk : rgb.crown) : rgb.building;
      for (const t of boxTris(bx[0], bx[1]).tris) tris.push({ w: t, rgb: col, kind: KIND.obst, obj, box: bx });
      obj++;
    });
    S._pov = tris;
    return tris;
  }

  // Draws the camera's view at poses[k] of a plan into canvas (W x H css px, rendered at ss x).
  window.drawPOV = function (canvas, D, env, planKey, k, W, ss) {
    ss = ss || 2;
    const S = D.scenes[env], pose = S.plans[planKey].poses[k];
    const cam = D.camera, tanH = cam.base[2][2] / cam.follow_distance, tanV = cam.base[2][1] / cam.follow_distance;
    const H = Math.round(W * tanV / tanH), w = Math.round(W * ss), h = Math.round(H * ss), fl = (w / 2) / tanH, so = Math.max(1, Math.round(ss));
    const R = rotMat(pose.slice(3, 6)), col = j => [R[0][j], R[1][j], R[2][j]];
    const f = col(0), u = col(1), s = col(2), eye = posed(pose, cam.apex);
    const toCam = p => { const d = V3.sub(p, eye); return [V3.dot(d, s), V3.dot(d, u), V3.dot(d, f)]; };
    const light = V3.norm([0.35, 1, 0.55]);
    const tris = [];
    const inside = bx => eye[0] > bx[0][0] && eye[0] < bx[1][0] && eye[1] > bx[0][1] && eye[1] < bx[1][1] && eye[2] > bx[0][2] && eye[2] < bx[1][2];
    const push = (wt, rgbv, kind, obj, bx) => {
      let n = V3.norm(V3.cross(V3.sub(wt[1], wt[0]), V3.sub(wt[2], wt[0])));
      if (V3.dot(n, V3.sub(eye, wt[0])) < 0) {
        // closed meshes face outward: cull the back faces, unless the camera is inside that box (then show its inside)
        if (kind === KIND.bird || (kind === KIND.obst && !inside(bx))) return;
        n = V3.mul(n, -1);
      }
      const kk = kind === KIND.room ? 0.9 + 0.1 * Math.max(0, V3.dot(n, light))
        : kind === KIND.bird ? 0.78 + 0.22 * Math.max(0, V3.dot(n, light)) + 0.12 * Math.max(0, -V3.dot(n, f))
        : 0.55 + 0.45 * Math.max(0, V3.dot(n, light)) + 0.1 * Math.max(0, n[1]);
      tris.push({ c: wt.map(toCam), rgb: rgbv, k: kk, kind, obj });
    };
    for (const t of povScene(D, S)) push(t.w, t.rgb, t.kind, t.obj, t.box);
    for (const fc of D.bird.F) push(fc.map(i => posed(pose, D.bird.V[i])), rgb.bird, KIND.bird, 999);
    const Rb = rasterize(w, h, fl, tris, 0.002);
    const img = new ImageData(w, h), px = img.data, dBase = cam.follow_distance;
    for (let i = 0; i < w * h; i++) {
      let r, g, b;
      if (Rb.id[i] < 0) { [r, g, b] = POVC.sky; }
      else {
        r = Rb.rgb[3 * i]; g = Rb.rgb[3 * i + 1]; b = Rb.rgb[3 * i + 2];
        const z = 1 / Rb.iz[i];
        if (Rb.kind[i] !== KIND.bird) {
          if (z < dBase) { const a = 0.55; r = r * (1 - a) + POVC.block[0] * a; g = g * (1 - a) + POVC.block[1] * a; b = b * (1 - a) + POVC.block[2] * a; }
          else { const a = 0.45; r = r * (1 - a) + POVC.sky[0] * a; g = g * (1 - a) + POVC.sky[1] * a; b = b * (1 - a) + POVC.sky[2] * a; }
        }
        // outlines where the object changes
        const x = i % w, y = (i / w) | 0;
        const o = Rb.id[i];
        if ((x + so < w && Rb.id[i + so] !== o && Rb.iz[i + so] < Rb.iz[i]) || (y + so < h && Rb.id[i + so * w] !== o && Rb.iz[i + so * w] < Rb.iz[i]) ||
            (x >= so && Rb.id[i - so] !== o && Rb.iz[i - so] < Rb.iz[i]) || (y >= so && Rb.id[i - so * w] !== o && Rb.iz[i - so * w] < Rb.iz[i])) { r *= 0.6; g *= 0.6; b *= 0.6; }
      }
      px[4 * i] = r; px[4 * i + 1] = g; px[4 * i + 2] = b; px[4 * i + 3] = 255;
    }
    const off = document.createElement('canvas'); off.width = w; off.height = h;
    off.getContext('2d').putImageData(img, 0, 0);
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, W, H);
    // the optical axis, faint
    ctx.strokeStyle = 'rgba(28,92,171,0.35)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2 - 6, H / 2); ctx.lineTo(W / 2 + 6, H / 2); ctx.moveTo(W / 2, H / 2 - 6); ctx.lineTo(W / 2, H / 2 + 6); ctx.stroke();
    return H;
  };

  // The waypoints shown in a plan's strip: evenly spaced, with the events swapped in for the nearest ones:
  // the first view break, the worst view, the first bird collision.
  window.povFrames = function (plan, n) {
    const m = plan.poses.length - 1, ks = [];
    for (let j = 0; j < n; j++) ks.push(Math.round(j * m / (n - 1)));
    let worst = 0; plan.view.forEach((v, i) => { if (v < plan.view[worst]) worst = i; });
    const ev = [plan.view_lost_at, plan.view[worst] < 0.99 ? worst : null, plan.bird_hit_at].filter(v => v != null);
    const fixed = new Set([0, n - 1]);
    for (const e of ev) {
      if (ks.includes(e)) { fixed.add(ks.indexOf(e)); continue; }
      let best = -1;
      for (let j = 1; j < n - 1; j++) if (!fixed.has(j) && (best < 0 || Math.abs(ks[j] - e) < Math.abs(ks[best] - e))) best = j;
      if (best < 0) break;
      ks[best] = e; fixed.add(best);
    }
    return [...new Set(ks)].sort((a, b) => a - b);
  };
})();
