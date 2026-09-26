/**
 * 3D Pose Lab — Three.js viewer for tennis-mediapipe landmark data.
 *
 * Reads a "tennis-mediapipe/landmarks3d@1" JSON (see export_3d_json.py):
 * per-frame MediaPipe 33-landmark 3D positions, rendered as joint dots
 * connected by a skeleton, with a frame scrubber, playback controls,
 * touch orbit controls, camera presets, 3D wrist motion trails and
 * live biomechanical angle readouts.
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
