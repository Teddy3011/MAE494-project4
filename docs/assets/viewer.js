// Interactive initial-vs-optimized viewer. Reads assets/results.json (written by run_all.py).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
// cividis (sequential, colour-vision-deficiency safe) and a grey ramp for density
const CIVIDIS = [[0, [0, 34, 78]], [0.25, [62, 77, 110]], [0.5, [124, 123, 120]], [0.75, [190, 175, 111]], [1, [253, 231, 55]]];
const GREYS = [[0, [240, 240, 240]], [1, [40, 40, 40]]];
const U_MAX = 1.2; // colour bar range for utilization

function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, a] = stops[i - 1], [t1, b] = stops[i], s = (t - t0) / (t1 - t0);
      return a.map((v, k) => (v + s * (b[k] - v)) / 255);
    }
  }
  return stops[stops.length - 1][1].map((v) => v / 255);
}

const FACES = [ // [neighbour offset, local node indices in hex8 order]
  [[-1, 0, 0], [0, 3, 7, 4]], [[1, 0, 0], [1, 5, 6, 2]],
  [[0, -1, 0], [0, 4, 5, 1]], [[0, 1, 0], [3, 2, 6, 7]],
  [[0, 0, -1], [0, 1, 2, 3]], [[0, 0, 1], [4, 7, 6, 5]],
];

class Pane {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.data = data;
    const [nx, ny, nz] = data.grid.nel, h = data.grid.h;
    this.nel = [nx, ny, nz];
    this.X = [];
    for (let k = 0; k <= nz; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) this.X.push([i * h, j * h, k * h]);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(getComputedStyle(document.body).getPropertyValue("--viewer-bg").trim() || "#f6f8fa");
    const L = data.grid.L, c = new THREE.Vector3(L[0] / 2, L[1] / 2, L[2] / 2);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.001, 10);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(c.x - 0.45, c.y - 0.55, c.z + 0.3);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.copy(c);
    this.controls.update();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(-1, -2, 3);
    this.scene.add(sun);
    this.mesh = null;
    this.overlay = new THREE.Group();
    this.scene.add(this.overlay);
    this.markers();
  }

  markers() {
    const d = this.data, X = this.X;
    const sup = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(d.supports.flatMap((n) => X[n]), 3));
    this.scene.add(new THREE.Points(sup, new THREE.PointsMaterial({ color: 0x1f2328, size: 0.007 })));
    const ctr = d.interface.reduce((a, n) => a.map((v, k) => v + X[n][k] / d.interface.length), [0, 0, 0]);
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, -1), new THREE.Vector3(ctr[0], ctr[1], ctr[2] + 0.07), 0.065, 0xb91c1c, 0.018, 0.012);
    this.scene.add(arrow);
    // torque about y: two opposite vertical arrows at the patch ends
    for (const s of [-1, 1]) {
      const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, s), new THREE.Vector3(ctr[0] + s * 0.05, ctr[1], ctr[2] + 0.04 - s * 0.02), 0.035, 0x7c2d12, 0.01, 0.007);
      this.scene.add(a);
    }
  }

  build(design, threshold, colorBy) {
    const [nx, ny, nz] = this.nel, rho = design.rho;
    const shown = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && rho[i + nx * (j + ny * k)] >= threshold;
    const nodes = [], elem = [], cols = [];
    const over = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (!shown(i, j, k)) continue;
      const e = i + nx * (j + ny * k);
      const n0 = i + (nx + 1) * (j + (ny + 1) * k), sx = 1, sy = nx + 1, sz = (nx + 1) * (ny + 1);
      const conn = [n0, n0 + sx, n0 + sx + sy, n0 + sy, n0 + sz, n0 + sz + sx, n0 + sz + sx + sy, n0 + sz + sy];
      const c = colorBy === "util" ? ramp(CIVIDIS, design.util[e] / U_MAX) : ramp(GREYS, rho[e]);
      if (design.util[e] > 1) over.push(conn);
      for (const [[di, dj, dk], q] of FACES) {
        if (shown(i + di, j + dj, k + dk)) continue;
        for (const t of [0, 1, 2, 0, 2, 3]) { nodes.push(conn[q[t]]); cols.push(...c); }
        elem.push(e, e);
      }
    }
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(nodes.length * 3), 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
    this.mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.8, side: THREE.DoubleSide }));
    this.vnodes = nodes;
    this.triElem = elem;
    this.scene.add(this.mesh);
    // over-limit voxels: outlined boxes in addition to colour (none expected in feasible designs)
    this.overlay.clear();
    this.overNodes = over;
    this.overCount = over.length;
    this.design = design;
  }

  deform(field, scale) {
    const pos = this.mesh.geometry.attributes.position.array, X = this.X, u = field ? field.data : null, s = field ? scale * field.scale : 0;
    for (let v = 0; v < this.vnodes.length; v++) {
      const n = this.vnodes[v];
      for (let a = 0; a < 3; a++) pos[3 * v + a] = X[n][a] + (u ? s * u[3 * n + a] : 0);
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.computeVertexNormals();
    this.mesh.geometry.computeBoundingSphere();
    this.overlay.clear();
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    const pts = [];
    for (const conn of this.overNodes) for (const [a, b] of edges) for (const n of [conn[a], conn[b]]) {
      pts.push(...X[n].map((x, k) => x + (u ? s * u[3 * n + k] : 0)));
    }
    if (pts.length) {
      const lg = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      this.overlay.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xdc2626 })));
    }
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.canvas.width !== Math.floor(w * this.renderer.getPixelRatio())) {
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }
  }

  render() { this.resize(); this.renderer.render(this.scene, this.camera); }
}

function colorbar(colorBy) {
  const cv = $("colorbar"), ctx = cv.getContext("2d");
  const stops = colorBy === "util" ? CIVIDIS : GREYS;
  for (let x = 0; x < cv.width; x++) {
    const c = ramp(stops, x / (cv.width - 1)).map((v) => Math.round(255 * v));
    ctx.fillStyle = `rgb(${c})`;
    ctx.fillRect(x, 0, 1, cv.height);
  }
  if (colorBy === "util") {
    const x1 = (cv.width * 1) / U_MAX;
    ctx.fillStyle = "#1f2328";
    ctx.fillRect(x1, 0, 2, cv.height);
    $("cb-label").textContent = "von Mises utilization σvm/σallow (max over load cases, Gauss points). Bar at 1.0 = limit; voxels above it are also outlined.";
    $("cb-ticks").innerHTML = "<span>0</span><span>0.5</span><span>1.0</span><span>1.2+</span>";
  } else {
    $("cb-label").textContent = "Physical (filtered, projected) density ρ";
    $("cb-ticks").innerHTML = "<span>0</span><span>0.5</span><span>1</span>";
  }
}

function table(design, el) {
  const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toPrecision(4));
  el.innerHTML = `<caption>${design.name}: <b>${design.mass.toFixed(3)} kg</b></caption>
    <tr><th>quantity</th><th>value</th><th>limit</th><th>status</th></tr>` +
    design.table.map((r) => `<tr><td>${r.name}</td><td>${fmt(r.value)} ${r.unit}</td><td>${r.name.startsWith("First") || r.name.startsWith("Buck") ? "≥" : "≤"} ${fmt(r.limit)} ${r.unit}</td>
      <td class="${r.passed ? "ok" : "bad"}">${r.passed ? "✓ pass" : "✗ FAIL"}</td></tr>`).join("");
}

async function main() {
  let data;
  try {
    const r = await fetch("assets/results.json", { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    $("viewer-error").hidden = false;
    $("viewer-error").textContent = `Could not load assets/results.json (${err.message}). Run python run_all.py and serve docs/ over HTTP.`;
    return;
  }
  const left = new Pane($("cv-left"), data), right = new Pane($("cv-right"), data);
  // linked cameras
  let syncing = false;
  for (const [a, b] of [[left, right], [right, left]]) {
    a.controls.addEventListener("change", () => {
      if (syncing) return;
      syncing = true;
      b.camera.position.copy(a.camera.position);
      b.camera.quaternion.copy(a.camera.quaternion);
      b.controls.target.copy(a.controls.target);
      b.controls.update();
      syncing = false;
    });
  }
  const st = { colorBy: "util", lc: "LC2", anim: "none", t0: performance.now() };
  const auto = () => 0.1 * Math.max(...data.grid.L) / data.designs.B.u.LC2.scale; // max deflection = 10 % of size

  function rebuild() {
    const thr = +$("thr").value;
    $("thr-val").textContent = thr.toFixed(2);
    left.build(data.designs.initial, thr, st.colorBy);
    right.build(data.designs[$("design").value], thr, st.colorBy);
    colorbar(st.colorBy);
    table(data.designs.initial, $("tab-left"));
    table(data.designs[$("design").value], $("tab-right"));
    $("title-right").textContent = data.designs[$("design").value].name;
    const n = right.overCount + left.overCount;
    $("over").textContent = n ? `${n} voxel(s) over the stress limit are outlined.` : "No displayed voxel exceeds the stress limit.";
    update();
  }

  function update() {
    const f = +$("exag").value;
    $("exag-val").textContent = f === 0 ? "true geometry (×0)" : `×${Math.round(f * auto())}`;
    for (const p of [left, right]) {
      let field = p.design.u[st.lc], s = f * auto();
      if (st.anim !== "none") {
        field = st.anim === "mode" ? p.design.mode1 : p.design.buck1;
        s = 0.06 * Math.max(...data.grid.L) / field.scale * Math.sin((performance.now() - st.t0) / 300);
      }
      p.deform(field, s);
    }
  }

  $("design").onchange = rebuild;
  $("thr").oninput = rebuild;
  $("color").onchange = (e) => { st.colorBy = e.target.value; rebuild(); };
  $("lc").onchange = (e) => { st.lc = e.target.value; update(); };
  $("exag").oninput = update;
  $("anim").onchange = (e) => {
    st.anim = e.target.value;
    const d = data.designs[$("design").value];
    $("anim-note").textContent = st.anim === "mode" ? `Mode 1: ${data.designs.initial.mode1_Hz.toFixed(1)} Hz (left), ${d.mode1_Hz.toFixed(1)} Hz (right). Amplitude arbitrary.`
      : st.anim === "buck" ? `Buckling mode 1, LC2: α = ${data.designs.initial.buck1_alpha.toFixed(1)} (left), ${d.buck1_alpha.toFixed(1)} (right). Amplitude arbitrary.` : "";
    update();
  };

  // hover / tap inspection
  const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
  for (const p of [left, right]) {
    p.canvas.addEventListener("pointermove", (ev) => {
      const r = p.canvas.getBoundingClientRect();
      mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(mouse, p.camera);
      const hit = ray.intersectObject(p.mesh)[0];
      if (!hit) { $("inspect").textContent = "Hover or tap a voxel to inspect it."; return; }
      const e = p.triElem[hit.faceIndex], [nx, ny] = p.nel, d = p.design;
      const i = e % nx, j = Math.floor(e / nx) % ny, k = Math.floor(e / (nx * ny));
      const n0 = i + (nx + 1) * (j + (ny + 1) * k), u = d.u[st.lc];
      const um = Math.hypot(...[0, 1, 2].map((a) => u.data[3 * n0 + a] * u.scale)) * 1e3;
      $("inspect").innerHTML = `<b>${d.name}</b> · element ${e} (i=${i}, j=${j}, k=${k}) · density ${d.rho[e].toFixed(3)} · utilization ${d.util[e].toFixed(3)} · |u| ${st.lc} at its first node ${um.toFixed(4)} mm`;
    });
  }

  rebuild();
  (function loop() {
    if (st.anim !== "none") update();
    left.render();
    right.render();
    requestAnimationFrame(loop);
  })();
  $("viewer-status").textContent = `Loaded ${data.grid.nel.join("×")} voxels, h = ${data.grid.h * 1000} mm.`;
}

main();
