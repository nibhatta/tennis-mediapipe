/**
 * 3D Pose Lab — Three.js viewer for tennis-mediapipe landmark data.
 *
 * Reads a "tennis-mediapipe/landmarks3d@1" JSON (see export_3d_json.py)
 * or a 33-landmark CSV (see export_landmarks_csv / the analyzer's
 * "Download Full 33-Landmark 3D CSV" button): per-frame MediaPipe
 * 33-landmark 3D positions, rendered as joint dots
 * connected by a skeleton, with a frame scrubber, playback controls,
 * touch orbit controls, camera presets, 3D wrist motion trails and
 * live biomechanical angle readouts. CSVs can be uploaded as files
 * or pasted as text; both flow through the same render path as JSON.
 *
 * Coordinate mapping (MediaPipe normalized -> Three.js world):
 *   X = (x - 0.5) * S          (left-to-right stays left-to-right)
 *   Y = (0.5 - y) * S          (flip: MediaPipe y grows downward)
 *   Z = -z * S                 (MediaPipe -z is toward the camera)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------- palette
const COL = {
  bg: 0x0e131c,
  volt: 0xc6e014,   // center / head / dominant accents
  cyan: 0x1ec8f0,   // left side
  orange: 0xff8c1e, // right side
  grid: 0x2a3448,
  dim: 0x8b96a8,
};
const SCALE = 3.0;          // world units per normalized unit
const TRAIL_LEN = 28;       // wrist trail length in frames
const SPEEDS = [0.25, 0.5, 1, 2];

// MediaPipe landmark indices used for telemetry
const IDX = { L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14, L_WR: 15, R_WR: 16,
              L_HIP: 23, R_HIP: 24, L_KN: 25, R_KN: 26, L_AN: 27, R_AN: 28 };

// Full 33-landmark names (mirrors detector.py LANDMARK_NAMES)
const LANDMARK_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
  "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee",
  "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
];

// Standard skeleton used when loading CSV (which carries no bone list).
// Mirrors detector.py UPPER/LOWER/HEAD_BODY_CONNECTIONS.
const DEFAULT_CONNECTIONS = {
  upper: [[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
          [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
          [11,23],[12,24],[23,24]],
  lower: [[23,25],[25,27],[27,29],[29,31],[27,31],
          [24,26],[26,28],[28,30],[30,32],[28,32]],
  head: [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10]],
};

// ---------------------------------------------------------------- state
let data = null;            // loaded JSON payload
let frames = [];            // frame records
let nFrames = 0;
let fps = 30;
let cur = 0;                // current frame (float while playing)
let playing = false;
let loop = true;
let speedIdx = 2;
let lastT = 0;
let scrubbing = false;
let wasPlaying = false;
let showTrails = true;
let showBones = true;
let dominantHand = 'Right';
let wristWorld = { dom: [], off: [] }; // precomputed per-frame world positions

// ---------------------------------------------------------------- three setup
const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(COL.bg);
scene.fog = new THREE.Fog(COL.bg, 8, 18);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.15, 0);
controls.minDistance = 1.2;
controls.maxDistance = 12;
controls.maxPolarAngle = 1.53;
controls.autoRotateSpeed = 1.6;

const CAMS = {
  front: [0, 0.35, 4.4],
  side: [4.4, 0.35, 0],
  back: [0, 0.35, -4.4],
  top: [0, 5.4, 0.9],
};
function setCam(name) {
  const p = CAMS[name] || CAMS.front;
  camera.position.set(p[0], p[1], p[2]);
  controls.target.set(0, 0.15, 0);
  controls.update();
  document.querySelectorAll('#cams [data-cam]').forEach(b =>
    b.classList.toggle('on', b.dataset.cam === name || (name === 'front' && b.dataset.cam === 'reset' && false)));
}
setCam('front');

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a2233, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(3, 6, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(COL.cyan, 0.5);
rim.position.set(-4, 3, -3);
scene.add(rim);

const grid = new THREE.GridHelper(10, 20, COL.grid, 0x1a2233);
grid.position.y = -0.78;
scene.add(grid);

// ---------------------------------------------------------------- skeleton objects
const NUM_LM = 33;
const jointMesh = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.035, 16, 12),
  new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1 }),
  NUM_LM
);
jointMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
jointMesh.frustumCulled = false;
scene.add(jointMesh);

const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();

function jointColor(i) {
  if (i <= 10) return tmpColor.setHex(COL.volt).clone();       // head
  if (i % 2 === 1) return tmpColor.setHex(COL.cyan).clone();    // left side
  return tmpColor.setHex(COL.orange).clone();                  // right side
}
for (let i = 0; i < NUM_LM; i++) jointMesh.setColorAt(i, jointColor(i));
jointMesh.instanceColor.needsUpdate = true;

let boneLines = null;   // LineSegments, rebuilt per dataset (bone count varies)
let bonePos = null;
function buildBones(connections) {
  if (boneLines) { scene.remove(boneLines); boneLines.geometry.dispose(); }
  const pairs = [];
  for (const group of Object.values(connections)) for (const c of group) pairs.push(c);
  const n = pairs.length;
  bonePos = new Float32Array(n * 2 * 3);
  const boneCol = new Float32Array(n * 2 * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(bonePos, 3).setUsage(THREE.DynamicDrawUsage));
  const cA = new THREE.Color(), cB = new THREE.Color();
  pairs.forEach(([a, b], k) => {
    cA.copy(jointColor(a)); cB.copy(jointColor(b));
    boneCol.set([cA.r, cA.g, cA.b], k * 6);
    boneCol.set([cB.r, cB.g, cB.b], k * 6 + 3);
  });
  geo.setAttribute('color', new THREE.BufferAttribute(boneCol, 3));
  boneLines = new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
  boneLines.frustumCulled = false;
  boneLines.userData.pairs = pairs;
  scene.add(boneLines);
}

// Wrist trails (dominant = volt, off-hand = cyan)
function makeTrail(hex) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(TRAIL_LEN * 3);
  const col = new Float32Array(TRAIL_LEN * 3);
  const c = new THREE.Color(hex);
  for (let i = 0; i < TRAIL_LEN; i++) {
    const f = Math.pow(i / (TRAIL_LEN - 1), 1.6); // fade tail -> head
    col.set([c.r * f, c.g * f, c.b * f], i * 3);
  }
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const line = new THREE.Line(g, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false }));
  line.frustumCulled = false;
  scene.add(line);
  return line;
}
const trailDom = makeTrail(COL.volt);
const trailOff = makeTrail(COL.cyan);

// ---------------------------------------------------------------- data handling
function mpToWorld(lm, out) {
  out.set((lm[0] - 0.5) * SCALE, (0.5 - lm[1]) * SCALE, -lm[2] * SCALE);
  return out;
}
const _v = new THREE.Vector3();

function loadData(payload, label) {
  if (!payload || payload.meta?.schema !== 'tennis-mediapipe/landmarks3d@1')
    throw new Error('Not a landmarks3d@1 file.');
  data = payload;
  frames = payload.frames;
  nFrames = frames.length;
  fps = payload.meta.fps || 30;
  dominantHand = payload.meta.dominant_hand || 'Right';
  cur = 0; playing = false;

  buildBones(payload.connections);

  // precompute wrist world trajectories for trails
  const dIdx = dominantHand === 'Right' ? IDX.R_WR : IDX.L_WR;
  const oIdx = dominantHand === 'Right' ? IDX.L_WR : IDX.R_WR;
  wristWorld = { dom: [], off: [] };
  for (const fr of frames) {
    if (fr.detected && fr.landmarks) {
      wristWorld.dom.push(mpToWorld(fr.landmarks[dIdx], new THREE.Vector3()).clone());
      wristWorld.off.push(mpToWorld(fr.landmarks[oIdx], new THREE.Vector3()).clone());
    } else {
      const l = wristWorld.dom.length;
      wristWorld.dom.push(l ? wristWorld.dom[l - 1].clone() : new THREE.Vector3());
      wristWorld.off.push(l ? wristWorld.off[l - 1].clone() : new THREE.Vector3());
    }
  }

  document.getElementById('scrub').max = nFrames - 1;
  const syn = payload.meta.synthetic;
  document.getElementById('badge').innerHTML =
    `${syn ? '<span class="synthetic">SYNTHETIC DEMO</span>' : '<span class="real">REAL DATA</span>'}<br>${label}<br>${nFrames} frames · ${fps} fps`;
  updatePlayBtn();
  renderFrame(0);
}

// ---------------------------------------------------------------- CSV input
// Parse a 33-landmark CSV (the layout export_landmarks_csv produces:
// frame,timestamp_sec,landmark_index,landmark_name,x,y,z,visibility)
// into a landmarks3d@1-shaped payload, so pasted / uploaded CSVs flow
// through the exact same render path as JSON.
function splitCSVLine(line) {
  const out = [];
  let cur = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++; }
        else quoted = false;
      } else cur += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { out.push(cur); cur = ''; }
    else cur += c;
  }
  out.push(cur);
  return out;
}

function parseLandmarksCSV(text, label) {
  const lines = String(text).split(/\r?\n/);
  let hi = 0;
  while (hi < lines.length && !lines[hi].trim()) hi++;
  if (hi >= lines.length) throw new Error('The pasted text is empty.');

  const header = splitCSVLine(lines[hi]).map(h => h.trim().toLowerCase());
  if (header.includes('hitting_elbow_angle') || header.includes('dom_wrist_x'))
    throw new Error(
      'This looks like the telemetry CSV (joint angles per frame), which has no 3D joint positions. ' +
      'The 3D viewer needs the 33-landmark CSV: in the analyzer use "Download Full 33-Landmark 3D CSV".');

  const col = (...names) => {
    for (const n of names) { const i = header.indexOf(n); if (i >= 0) return i; }
    return -1;
  };
  const cFrame = col('frame');
  const cIdx = col('landmark_index', 'index', 'landmark_id');
  const cName = col('landmark_name', 'name', 'landmark');
  const cX = col('x'), cY = col('y'), cZ = col('z');
  const cVis = col('visibility', 'vis');
  const cT = col('timestamp_sec', 't', 'time', 'timestamp');
  const missing = [];
  if (cFrame < 0) missing.push('"frame"');
  if (cIdx < 0 && cName < 0) missing.push('"landmark_index" or "landmark_name"');
  if (cX < 0 || cY < 0 || cZ < 0) missing.push('"x", "y", "z"');
  if (missing.length) throw new Error(
    'Missing required column(s): ' + missing.join(', ') + '.\n' +
    'Expected a header like:\nframe,timestamp_sec,landmark_index,landmark_name,x,y,z,visibility');

  const nameToIdx = new Map(LANDMARK_NAMES.map((n, i) => [n, i]));
  const byFrame = new Map(); // frame -> { rows: Map(idx -> [x,y,z]), vis: Map(idx -> v), t }
  for (let li = hi + 1; li < lines.length; li++) {
    const line = lines[li];
    if (!line.trim()) continue;
    const cells = splitCSVLine(line);
    const f = parseInt(cells[cFrame], 10);
    if (!Number.isFinite(f) || f < 0) continue;
    let idx = -1;
    if (cIdx >= 0) {
      const v = parseInt(cells[cIdx], 10);
      if (Number.isFinite(v) && v >= 0 && v < 33) idx = v;
    }
    if (idx < 0 && cName >= 0) {
      const v = nameToIdx.get((cells[cName] || '').trim().toLowerCase());
      if (v !== undefined) idx = v;
    }
    const x = parseFloat(cells[cX]), y = parseFloat(cells[cY]), z = parseFloat(cells[cZ]);
    if (idx < 0 || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    let vis = 1.0;
    if (cVis >= 0) { const v = parseFloat(cells[cVis]); if (Number.isFinite(v)) vis = v; }
    let entry = byFrame.get(f);
    if (!entry) { entry = { rows: new Map(), vis: new Map(), t: null }; byFrame.set(f, entry); }
    entry.rows.set(idx, [x, y, z]);
    entry.vis.set(idx, vis);
    if (entry.t === null && cT >= 0) {
      const tv = parseFloat(cells[cT]);
      if (Number.isFinite(tv)) entry.t = tv;
    }
  }
  if (byFrame.size === 0)
    throw new Error('No usable landmark rows found. Check the column layout against the expected header.');

  const maxF = Math.max(...byFrame.keys());
  // Estimate fps from timestamp_sec deltas when available.
  let fps = 30;
  const ts = [];
  for (let f = 0; f <= maxF; f++) {
    const e = byFrame.get(f);
    if (e && e.t !== null) ts.push(e.t);
  }
  if (ts.length >= 3) {
    const deltas = [];
    for (let i = 1; i < ts.length; i++) {
      const d = ts[i] - ts[i - 1];
      if (d > 0 && d < 5) deltas.push(d);
    }
    if (deltas.length) {
      deltas.sort((a, b) => a - b);
      const med = deltas[Math.floor(deltas.length / 2)];
      if (med > 0) {
        fps = Math.round(1000 / med) / 1000;
        // Snap to the nearest standard frame rate when close (timestamps are
        // often rounded to milliseconds, e.g. 0.033s -> 30.303fps for true 30fps).
        let best = null, bestErr = 0.02;
        for (const std of [23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120]) {
          const err = Math.abs(fps - std) / std;
          if (err < bestErr) { bestErr = err; best = std; }
        }
        if (best !== null) fps = best;
      }
    }
  }

  const frames = [];
  let detectedCount = 0;
  for (let f = 0; f <= maxF; f++) {
    const e = byFrame.get(f);
    if (e && e.rows.size === 33) {
      const lms = [], vis = [];
      for (let j = 0; j < 33; j++) { lms.push(e.rows.get(j)); vis.push(e.vis.get(j) ?? 1.0); }
      frames.push({
        frame: f, t: e.t !== null ? e.t : Math.round(f / fps * 1000) / 1000,
        detected: true, landmarks: lms, visibility: vis,
      });
      detectedCount++;
    } else {
      frames.push({
        frame: f, t: Math.round(f / fps * 1000) / 1000,
        detected: false, landmarks: null, visibility: null,
      });
    }
  }
  if (detectedCount === 0)
    throw new Error('No complete frames found: every frame needs all 33 landmarks (indices 0-32).');

  return {
    meta: {
      schema: 'tennis-mediapipe/landmarks3d@1',
      fps, frame_count: frames.length,
      duration_sec: Math.round(frames.length / fps * 1000) / 1000,
      source: label || 'pasted.csv',
      synthetic: /sample/i.test(label || ''),
      dominant_hand: 'Right',
      coordinate_system: 'mediapipe_normalized',
      generated: new Date().toISOString(),
      notes: 'Parsed from a pasted/uploaded 33-landmark CSV in the 3D Pose Lab viewer.',
    },
    landmark_names: LANDMARK_NAMES,
    connections: DEFAULT_CONNECTIONS,
    frames,
  };
}

function frameAngles(fr) {
  // bend angle (0 = straight) at a joint, mirroring detector.py "bend" mode
  const P = fr.landmarks.map(p => mpToWorld(p, new THREE.Vector3()));
  const v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
  function bend3(ai, bi, ci) {
    v1.subVectors(P[ai], P[bi]); v2.subVectors(P[ci], P[bi]);
    const cos = THREE.MathUtils.clamp(v1.normalize().dot(v2.normalize()), -1, 1);
    return 180 - THREE.MathUtils.radToDeg(Math.acos(cos));
  }
  const R = dominantHand === 'Right';
  const elbow = bend3(R ? IDX.R_SH : IDX.L_SH, R ? IDX.R_EL : IDX.L_EL, R ? IDX.R_WR : IDX.L_WR);
  const knee = bend3(R ? IDX.R_HIP : IDX.L_HIP, R ? IDX.R_KN : IDX.L_KN, R ? IDX.R_AN : IDX.L_AN);
  const hip = new THREE.Vector3().addVectors(P[IDX.L_HIP], P[IDX.R_HIP]).multiplyScalar(0.5);
  const sho = new THREE.Vector3().addVectors(P[IDX.L_SH], P[IDX.R_SH]).multiplyScalar(0.5);
  const torso = new THREE.Vector3().subVectors(sho, hip).normalize();
  const tilt = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(torso.y, -1, 1)));
  return { elbow, knee, tilt };
}

// ---------------------------------------------------------------- rendering
const _p = new THREE.Vector3();
function renderFrame(fFloat) {
  const i = Math.max(0, Math.min(nFrames - 1, Math.round(fFloat)));
  const fr = frames[i];
  const ok = fr && fr.detected && fr.landmarks;
  jointMesh.visible = !!ok;
  if (boneLines) boneLines.visible = !!ok && showBones;
  trailDom.visible = trailOff.visible = !!ok && showTrails;
  document.getElementById('nopose').style.display = ok ? 'none' : 'block';

  if (ok) {
    const vis = fr.visibility || [];
    for (let j = 0; j < NUM_LM; j++) {
      mpToWorld(fr.landmarks[j], _p);
      dummy.position.copy(_p);
      const s = (vis[j] ?? 1) < 0.4 ? 0.35 : 1;
      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      jointMesh.setMatrixAt(j, dummy.matrix);
    }
    jointMesh.instanceMatrix.needsUpdate = true;

    const pairs = boneLines.userData.pairs;
    pairs.forEach(([a, b], k) => {
      mpToWorld(fr.landmarks[a], _p);
      bonePos.set([_p.x, _p.y, _p.z], k * 6);
      mpToWorld(fr.landmarks[b], _p);
      bonePos.set([_p.x, _p.y, _p.z], k * 6 + 3);
    });
    boneLines.geometry.attributes.position.needsUpdate = true;

    // trails: trailing window ending at current frame
    for (const [line, arr] of [[trailDom, wristWorld.dom], [trailOff, wristWorld.off]]) {
      const pos = line.geometry.attributes.position.array;
      for (let k = 0; k < TRAIL_LEN; k++) {
        const src = arr[Math.max(0, i - (TRAIL_LEN - 1 - k))];
        pos.set([src.x, src.y, src.z], k * 3);
      }
      line.geometry.attributes.position.needsUpdate = true;
    }

    const { elbow, knee, tilt } = frameAngles(fr);
    document.getElementById('h-elbow').textContent = `${elbow.toFixed(0)}°`;
    document.getElementById('h-knee').textContent = `${knee.toFixed(0)}°`;
    document.getElementById('h-torso').textContent = `${tilt.toFixed(0)}°`;
  }

  if (!scrubbing) document.getElementById('scrub').value = i;
  document.getElementById('timelabel').innerHTML =
    `<b>F ${i}</b> / ${nFrames - 1} · ${(i / fps).toFixed(2)}s`;
}

// ---------------------------------------------------------------- transport
function updatePlayBtn() {
  document.getElementById('b-play').textContent = playing ? '❚❚' : '▶';
}
function setPlaying(p) { playing = p; updatePlayBtn(); }
function step(d) {
  setPlaying(false);
  cur = THREE.MathUtils.clamp(Math.round(cur) + d, 0, nFrames - 1);
  renderFrame(cur);
}

document.getElementById('b-play').onclick = () => setPlaying(!playing);
document.getElementById('b-prev').onclick = () => step(-1);
document.getElementById('b-next').onclick = () => step(1);
document.getElementById('b-first').onclick = () => { setPlaying(false); cur = 0; renderFrame(cur); };
document.getElementById('b-last').onclick = () => { setPlaying(false); cur = nFrames - 1; renderFrame(cur); };
document.getElementById('b-speed').onclick = (e) => {
  speedIdx = (speedIdx + 1) % SPEEDS.length;
  e.target.textContent = `${SPEEDS[speedIdx]}×`;
};
document.getElementById('b-loop').onclick = (e) => {
  loop = !loop; e.target.classList.toggle('on', loop);
};
const scrub = document.getElementById('scrub');
scrub.addEventListener('pointerdown', () => { scrubbing = true; wasPlaying = playing; setPlaying(false); });
window.addEventListener('pointerup', () => {
  if (scrubbing) { scrubbing = false; if (wasPlaying) setPlaying(true); }
});
scrub.addEventListener('input', () => { cur = parseFloat(scrub.value); renderFrame(cur); });

document.querySelectorAll('#cams [data-cam]').forEach(b => {
  b.onclick = () => setCam(b.dataset.cam === 'reset' ? 'front' : b.dataset.cam);
});
document.getElementById('t-trails').onclick = (e) => {
  showTrails = !showTrails; e.target.classList.toggle('on', showTrails); renderFrame(cur);
};
document.getElementById('t-bones').onclick = (e) => {
  showBones = !showBones; e.target.classList.toggle('on', showBones); renderFrame(cur);
};
document.getElementById('t-spin').onclick = (e) => {
  controls.autoRotate = !controls.autoRotate;
  e.target.classList.toggle('on', controls.autoRotate);
};
document.getElementById('loadbtn').onclick = () =>
  document.getElementById('file').click();
document.getElementById('file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { loadData(JSON.parse(rd.result), f.name); }
    catch (err) { showErr(`Could not load that file: ${err.message}`); }
  };
  rd.readAsText(f);
  e.target.value = '';
});

// CSV upload: same 33-landmark CSV the analyzer exports.
document.getElementById('csvbtn').onclick = () =>
  document.getElementById('csvfile').click();
document.getElementById('csvfile').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { loadData(parseLandmarksCSV(rd.result, f.name), f.name); }
    catch (err) { showErr(`Could not parse that CSV:\n\n${err.message}`); }
  };
  rd.readAsText(f);
  e.target.value = '';
});

// CSV paste panel.
const pastePanel = document.getElementById('pastepanel');
document.getElementById('pastebtn').onclick = () => {
  pastePanel.style.display = pastePanel.style.display === 'block' ? 'none' : 'block';
};
document.getElementById('pp-close').onclick = () => { pastePanel.style.display = 'none'; };
document.getElementById('pp-render').onclick = () => {
  const text = document.getElementById('pp-text').value;
  try {
    loadData(parseLandmarksCSV(text, 'pasted.csv'), 'pasted CSV');
    pastePanel.style.display = 'none';
  } catch (err) { showErr(`Could not parse that CSV:\n\n${err.message}`); }
};
document.getElementById('pp-sample').onclick = () => {
  const fill = (t) => { document.getElementById('pp-text').value = t; };
  if (window.__SAMPLE_CSV__) { fill(window.__SAMPLE_CSV__); return; }
  fetch('./sample/sample_landmarks.csv')
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
    .then(fill)
    .catch(err => showErr(`Could not load the sample CSV:\n\n${err.message}`));
};

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
  if (e.code === 'ArrowLeft') step(-1);
  if (e.code === 'ArrowRight') step(1);
});

function showErr(msg) {
  document.getElementById('errmsg').textContent = msg;
  document.getElementById('err').style.display = 'flex';
}

// ---------------------------------------------------------------- main loop
function tick(t) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (t - lastT) / 1000 || 0);
  lastT = t;
  if (playing && nFrames > 0 && !scrubbing) {
    cur += dt * fps * SPEEDS[speedIdx];
    if (cur >= nFrames) {
      if (loop) cur = cur % nFrames;
      else { cur = nFrames - 1; setPlaying(false); }
    }
    renderFrame(cur);
  }
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------------- boot
fetch('./sample/sample_landmarks.json')
  .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then(j => loadData(j, 'sample_landmarks.json'))
  .catch(err => showErr(
    'Could not load the bundled demo data.\n' +
    'Serve this folder over HTTP (e.g. `python3 -m http.server`) or use "Load JSON".\n\n' + err.message));

requestAnimationFrame(tick);
