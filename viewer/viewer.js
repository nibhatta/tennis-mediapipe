/**
 * 3D Pose Lab — Three.js viewer for tennis-mediapipe landmark data.
 *
 * Reads a "tennis-mediapipe/landmarks3d@1" JSON (see export_3d_json.py)
 * or a 33-landmark CSV (see export_landmarks_csv / the analyzer's
 * "Download Full 33-Landmark 3D CSV" button): per-frame MediaPipe
 * 33-landmark 3D positions, rendered as joint dots connected by a skeleton,
 * with a realistic 3D tennis racket simulation, 3D athletic tennis sports cap,
 * clean eye-level gaze stabilization, full 360-degree orbit camera controls,
 * Savitzky-Golay trajectory smoothing, frame scrubber, playback controls,
 * 3D wrist motion trails, and live biomechanical readouts.
 *
 * Coordinate mapping (MediaPipe normalized -> Three.js world):
 *   X = (x - 0.5) * SCALE          (left-to-right stays left-to-right)
 *   Y = (0.5 - y) * SCALE          (flip: MediaPipe y grows downward)
 *   Z = -z * SCALE                 (MediaPipe -z is toward the camera)
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

// MediaPipe landmark indices
const IDX = {
  NOSE: 0,
  L_EYE_IN: 1, L_EYE: 2, L_EYE_OUT: 3,
  R_EYE_IN: 4, R_EYE: 5, R_EYE_OUT: 6,
  L_EAR: 7, R_EAR: 8,
  MOUTH_L: 9, MOUTH_R: 10,
  L_SH: 11, R_SH: 12,
  L_EL: 13, R_EL: 14,
  L_WR: 15, R_WR: 16,
  L_PINKY: 17, R_PINKY: 18,
  L_INDEX: 19, R_INDEX: 20,
  L_THUMB: 21, R_THUMB: 22,
  L_HIP: 23, R_HIP: 24,
  L_KN: 25, R_KN: 26,
  L_AN: 27, R_AN: 28,
  L_HEEL: 29, R_HEEL: 30,
  L_FOOT: 31, R_FOOT: 32,
};

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

// Clean skeleton connections (Clean head gaze stabilizer + body)
const DEFAULT_CONNECTIONS = {
  upper: [
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 23], [12, 24], [23, 24]
  ],
  lower: [
    [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
    [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]
  ],
  head: [
    [2, 5],     // Eye-to-Eye horizontal gaze stabilizer bar
    [0, 2],     // Nose to left eye
    [0, 5],     // Nose to right eye
  ],
};

// ---------------------------------------------------------------- Savitzky-Golay Smoothing
const SG_COEFFS = {
  5: [-3, 12, 17, 12, -3].map(v => v / 35),
  7: [-2, 3, 6, 7, 6, 3, -2].map(v => v / 21),
  9: [-21, 14, 39, 54, 59, 54, 39, 14, -21].map(v => v / 231),
};

function applySavitzkyGolay1D(arr, windowSize = 7) {
  const n = arr.length;
  if (n < windowSize) return arr.slice();
  const coeffs = SG_COEFFS[windowSize] || SG_COEFFS[7];
  const half = Math.floor(coeffs.length / 2);
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = -half; k <= half; k++) {
      let idx = i + k;
      if (idx < 0) idx = -idx;
      if (idx >= n) idx = 2 * (n - 1) - idx;
      idx = Math.max(0, Math.min(n - 1, idx));
      sum += arr[idx] * coeffs[k + half];
    }
    out[i] = sum;
  }
  return out;
}

function smoothFramesData(rawFrames) {
  const n = rawFrames.length;
  if (n < 5) return rawFrames;

  const smoothed = rawFrames.map(f => ({
    frame: f.frame,
    t: f.t,
    detected: f.detected,
    visibility: f.visibility ? [...f.visibility] : null,
    landmarks: f.landmarks ? f.landmarks.map(p => (p ? [...p] : null)) : null,
  }));

  const detectedIndices = [];
  for (let i = 0; i < n; i++) {
    if (rawFrames[i].detected && rawFrames[i].landmarks) {
      detectedIndices.push(i);
    }
  }

  if (detectedIndices.length < 5) return smoothed;

  const m = detectedIndices.length;
  const win = m >= 9 ? 9 : (m >= 7 ? 7 : 5);

  for (let j = 0; j < 33; j++) {
    for (let d = 0; d < 3; d++) {
      const series = new Float64Array(m);
      for (let k = 0; k < m; k++) {
        const fIdx = detectedIndices[k];
        series[k] = rawFrames[fIdx].landmarks[j][d];
      }
      const smoothSeries = applySavitzkyGolay1D(series, win);
      for (let k = 0; k < m; k++) {
        const fIdx = detectedIndices[k];
        smoothed[fIdx].landmarks[j][d] = smoothSeries[k];
      }
    }
  }

  return smoothed;
}

// ---------------------------------------------------------------- state
let data = null;            // loaded JSON payload
let rawFrames = [];         // raw input frame records
let smoothFrames = [];      // Savitzky-Golay smoothed frame records
let activeFrames = [];      // currently rendered frames (raw or smoothed)
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
let showRacket = true;
let showCap = true;
let showSmooth = true;
let dominantHand = 'Right';
let wristWorldRaw = { dom: [], off: [] };
let wristWorldSmooth = { dom: [], off: [] };
let wristWorld = { dom: [], off: [] };

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
controls.minDistance = 0.8;
controls.maxDistance = 16;
// Allow full 360-degree rotation in all directions (overhead, level, and below horizontal)
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;
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

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a2233, 1.2));
const key = new THREE.DirectionalLight(0xffffff, 1.7);
key.position.set(3, 6, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(COL.cyan, 0.6);
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
  if (i === IDX.L_EYE) return tmpColor.setHex(COL.cyan).clone();
  if (i === IDX.R_EYE) return tmpColor.setHex(COL.orange).clone();
  if (i === IDX.NOSE) return tmpColor.setHex(COL.volt).clone();
  if (i <= 10) return tmpColor.setHex(COL.dim).clone();
  if (i % 2 === 1) return tmpColor.setHex(COL.cyan).clone();    // Left side
  return tmpColor.setHex(COL.orange).clone();                  // Right side
}
for (let i = 0; i < NUM_LM; i++) jointMesh.setColorAt(i, jointColor(i));
jointMesh.instanceColor.needsUpdate = true;

let boneLines = null;
let bonePos = null;
let neckLine = null;
let neckPos = null;

function buildBones(connections) {
  if (boneLines) { scene.remove(boneLines); boneLines.geometry.dispose(); }
  if (neckLine) { scene.remove(neckLine); neckLine.geometry.dispose(); }

  const pairs = [];
  const conn = connections || DEFAULT_CONNECTIONS;
  const upper = conn.upper || DEFAULT_CONNECTIONS.upper;
  const lower = conn.lower || DEFAULT_CONNECTIONS.lower;
  const head = DEFAULT_CONNECTIONS.head;

  for (const c of upper) pairs.push(c);
  for (const c of lower) pairs.push(c);
  for (const c of head) pairs.push(c);

  const n = pairs.length;
  bonePos = new Float32Array(n * 2 * 3);
  const boneCol = new Float32Array(n * 2 * 3);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(bonePos, 3).setUsage(THREE.DynamicDrawUsage));

  const cA = new THREE.Color(), cB = new THREE.Color();
  pairs.forEach(([a, b], k) => {
    cA.copy(jointColor(a)); cB.copy(jointColor(b));
    if ((a === 2 && b === 5) || (a === 5 && b === 2)) {
      cA.setHex(COL.volt); cB.setHex(COL.volt);
    }
    boneCol.set([cA.r, cA.g, cA.b], k * 6);
    boneCol.set([cB.r, cB.g, cB.b], k * 6 + 3);
  });
  geo.setAttribute('color', new THREE.BufferAttribute(boneCol, 3));
  boneLines = new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, linewidth: 2 }));
  boneLines.frustumCulled = false;
  boneLines.userData.pairs = pairs;
  scene.add(boneLines);

  // Neck bone (Eye center to Shoulder center)
  neckPos = new Float32Array(6);
  const neckGeo = new THREE.BufferGeometry();
  neckGeo.setAttribute('position', new THREE.BufferAttribute(neckPos, 3).setUsage(THREE.DynamicDrawUsage));
  const neckCol = new Float32Array(6);
  const cVolt = new THREE.Color(COL.volt);
  neckCol.set([cVolt.r, cVolt.g, cVolt.b, cVolt.r, cVolt.g, cVolt.b], 0);
  neckGeo.setAttribute('color', new THREE.BufferAttribute(neckCol, 3));
  neckLine = new THREE.Line(neckGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 }));
  neckLine.frustumCulled = false;
  scene.add(neckLine);
}

// ---------------------------------------------------------------- Mannequin Face & Exaggerated White Visor
function createWilsonLogoTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 512, 512);

  ctx.fillStyle = '#eb142b'; // Iconic Wilson Red

  // Stylized Wilson 'W' logo
  ctx.beginPath();
  ctx.moveTo(115, 140);
  ctx.bezierCurveTo(125, 120, 155, 120, 165, 140);
  ctx.bezierCurveTo(160, 220, 155, 310, 168, 355);
  ctx.bezierCurveTo(178, 395, 212, 395, 222, 355);
  ctx.bezierCurveTo(230, 310, 230, 220, 232, 140);
  ctx.bezierCurveTo(242, 120, 272, 120, 282, 140);
  ctx.bezierCurveTo(280, 220, 275, 310, 288, 355);
  ctx.bezierCurveTo(298, 395, 332, 395, 342, 355);
  ctx.bezierCurveTo(350, 310, 350, 220, 352, 140);
  ctx.bezierCurveTo(362, 120, 392, 120, 402, 140);
  // Outer loops
  ctx.bezierCurveTo(400, 240, 385, 350, 368, 395);
  ctx.bezierCurveTo(342, 445, 275, 445, 252, 395);
  ctx.bezierCurveTo(240, 370, 235, 340, 232, 310);
  // Center loop
  ctx.bezierCurveTo(228, 340, 222, 370, 210, 395);
  ctx.bezierCurveTo(188, 445, 120, 445, 95, 395);
  ctx.bezierCurveTo(80, 350, 92, 240, 115, 140);
  ctx.closePath();
  ctx.fill();

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 4;
  return tex;
}

function createCapMesh() {
  const group = new THREE.Group();

  // Materials
  const whiteBrimMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.3,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  const whiteBandMat = new THREE.MeshStandardMaterial({
    color: 0xf8fafc,
    roughness: 0.5,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  const rimMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a, // Dark contrast rim border
    roughness: 0.4,
    metalness: 0.1,
  });

  const faceMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b, // Stylized dark slate athletic mannequin
    roughness: 0.55,
    metalness: 0.15,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
  });

  const featureMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8, // Cyan accent for facial landmarks
    roughness: 0.4,
    metalness: 0.2,
  });

  const logoTex = createWilsonLogoTexture();
  const logoMat = new THREE.MeshStandardMaterial({
    map: logoTex,
    transparent: true,
    roughness: 0.35,
    metalness: 0.15,
    side: THREE.DoubleSide,
  });

  const R = 0.088;      // Head radius at forehead
  const bw = 0.125;     // Exaggerated half-width of visor brim (~0.25m broad)
  const bl = 0.33;      // Doubled forward length of brim (~0.33m long projection!)
  const bThick = 0.009; // Solid 3D thickness

  // -------------------------------------------------------------
  // 1. Stylized Athletic Mannequin Head & Face
  // -------------------------------------------------------------
  const headGroup = new THREE.Group();

  // 1A. Cranium / Head Contour
  const craniumGeo = new THREE.SphereGeometry(0.082, 24, 16);
  craniumGeo.scale(0.88, 1.15, 1.0);
  const cranium = new THREE.Mesh(craniumGeo, faceMat);
  cranium.position.set(0, -0.015, -0.02);
  headGroup.add(cranium);

  // 1B. Jawline & Chin
  const jawGeo = new THREE.ConeGeometry(0.065, 0.09, 16);
  jawGeo.scale(0.85, 1.0, 0.95);
  jawGeo.rotateX(Math.PI);
  const jaw = new THREE.Mesh(jawGeo, faceMat);
  jaw.position.set(0, -0.075, 0.01);
  headGroup.add(jaw);

  // 1C. 3D Nose Bridge & Tip (Pointing forward along +Z)
  const noseShape = new THREE.Shape();
  noseShape.moveTo(-0.008, 0.02);
  noseShape.lineTo(0.008, 0.02);
  noseShape.lineTo(0.012, -0.025);
  noseShape.lineTo(0, -0.035);
  noseShape.lineTo(-0.012, -0.025);
  noseShape.closePath();
  const noseExtrude = { steps: 1, depth: 0.025, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.004, bevelSegments: 2 };
  const noseGeo = new THREE.ExtrudeGeometry(noseShape, noseExtrude);
  const noseMesh = new THREE.Mesh(noseGeo, faceMat);
  noseMesh.position.set(0, 0.005, 0.068);
  headGroup.add(noseMesh);

  // 1D. Eye Sockets (Left and Right)
  const eyeGeo = new THREE.SphereGeometry(0.012, 12, 8);
  const leftEye = new THREE.Mesh(eyeGeo, featureMat);
  leftEye.position.set(-0.034, 0.015, 0.072);
  headGroup.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeo, featureMat);
  rightEye.position.set(0.034, 0.015, 0.072);
  headGroup.add(rightEye);

  group.add(headGroup);

  // -------------------------------------------------------------
  // 2. Open-Top Visor Headband
  // -------------------------------------------------------------
  const bandArc = Math.PI * 0.72;
  const bandGeo = new THREE.CylinderGeometry(R, R, 0.024, 24, 1, true, -bandArc / 2, bandArc);
  const band = new THREE.Mesh(bandGeo, whiteBandMat);
  band.position.set(0, 0.022, 0);
  band.rotation.y = Math.PI / 2;
  group.add(band);

  // Red Wilson 'W' Logo centered on the front band
  const logoGeo = new THREE.PlaneGeometry(0.042, 0.042);
  const logoMesh = new THREE.Mesh(logoGeo, logoMat);
  logoMesh.position.set(0, 0.022, R * 1.01);
  group.add(logoMesh);

  // -------------------------------------------------------------
  // 3. Exaggerated Forward-Projecting White Visor Brim (Doubled Length)
  // -------------------------------------------------------------
  const brimShape = new THREE.Shape();
  brimShape.moveTo(-bw, 0);
  brimShape.quadraticCurveTo(-bw * 0.95, bl * 0.55, -bw * 0.45, bl * 0.92);
  brimShape.quadraticCurveTo(0, bl, bw * 0.45, bl * 0.92);
  brimShape.quadraticCurveTo(bw * 0.95, bl * 0.55, bw, 0);
  brimShape.quadraticCurveTo(0, 0.03, -bw, 0);

  const extrudeSettings = {
    steps: 1,
    depth: bThick,
    bevelEnabled: true,
    bevelThickness: 0.002,
    bevelSize: 0.002,
    bevelSegments: 2,
  };

  const brimGeo = new THREE.ExtrudeGeometry(brimShape, extrudeSettings);

  // Lateral downward curvature across the visor width
  const posAttr = brimGeo.attributes.position;
  for (let i = 0; i < posAttr.count; i++) {
    const x = posAttr.getX(i);
    const curveZ = -0.032 * (1 - Math.cos((x / bw) * Math.PI * 0.5));
    posAttr.setZ(i, posAttr.getZ(i) + curveZ);
  }
  brimGeo.computeVertexNormals();

  const brimMesh = new THREE.Mesh(brimGeo, whiteBrimMat);
  // Forward pitch: rotates +Y to +Z (pointing straight forward away from the face)
  const pitchAngle = Math.PI / 2 - 0.24;
  brimMesh.rotation.x = pitchAngle;
  brimMesh.position.set(0, 0.018, R * 0.85);
  group.add(brimMesh);

  // 4. Contrast Edge Border around the perimeter of the forward visor
  const edgeCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-bw, 0, 0),
    new THREE.Vector3(-bw * 0.8, bl * 0.45, -0.02),
    new THREE.Vector3(-bw * 0.4, bl * 0.9, -0.03),
    new THREE.Vector3(0, bl, -0.032),
    new THREE.Vector3(bw * 0.4, bl * 0.9, -0.03),
    new THREE.Vector3(bw * 0.8, bl * 0.45, -0.02),
    new THREE.Vector3(bw, 0, 0),
  ]);
  const edgeGeo = new THREE.TubeGeometry(edgeCurve, 36, 0.003, 6, false);
  const edgeMesh = new THREE.Mesh(edgeGeo, rimMat);
  edgeMesh.rotation.x = pitchAngle;
  edgeMesh.position.set(0, 0.018, R * 0.85);
  group.add(edgeMesh);

  group.frustumCulled = false;
  return group;
}

const capMesh = createCapMesh();
scene.add(capMesh);

// ---------------------------------------------------------------- 3D Tennis Racket Simulation
function createRacketMesh() {
  const group = new THREE.Group();

  const gripMat = new THREE.MeshStandardMaterial({
    color: 0xf1f5f9,
    roughness: 0.85,
    metalness: 0.05,
  });
  const capMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.4,
  });
  const frameMat = new THREE.MeshStandardMaterial({
    color: COL.volt,
    roughness: 0.3,
    metalness: 0.7,
  });
  const throatMat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    roughness: 0.35,
    metalness: 0.6,
  });
  const stringMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
  });
  const stringGridMat = new THREE.LineBasicMaterial({
    color: 0x94a3b8,
    transparent: true,
    opacity: 0.4,
  });

  const gripLen = 0.20;
  const gripRadius = 0.015;
  const shaftLen = 0.15;
  const headHeight = 0.32;
  const headWidth = 0.23;

  // 1. Butt Cap
  const buttCap = new THREE.Mesh(
    new THREE.CylinderGeometry(gripRadius * 1.18, gripRadius * 1.3, 0.016, 12),
    capMat
  );
  buttCap.position.y = 0.008;
  group.add(buttCap);

  // 2. Handle / Grip
  const grip = new THREE.Mesh(
    new THREE.CylinderGeometry(gripRadius, gripRadius * 1.1, gripLen, 12),
    gripMat
  );
  grip.position.y = 0.016 + gripLen / 2;
  group.add(grip);

  // 3. V-Throat & Shaft
  const throatY = 0.016 + gripLen;
  const branchLen = Math.hypot(shaftLen, headWidth * 0.38);
  const branchAngle = Math.atan2(headWidth * 0.38, shaftLen);

  const throatL = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.008, branchLen, 8), throatMat);
  throatL.position.set(-headWidth * 0.19, throatY + shaftLen / 2, 0);
  throatL.rotation.z = branchAngle;
  group.add(throatL);

  const throatR = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.008, branchLen, 8), throatMat);
  throatR.position.set(headWidth * 0.19, throatY + shaftLen / 2, 0);
  throatR.rotation.z = -branchAngle;
  group.add(throatR);

  // 4. Elliptical Racket Head
  const headCenterY = throatY + shaftLen + headHeight / 2 - 0.02;
  const curve = new THREE.EllipseCurve(
    0, 0,
    headWidth / 2, headHeight / 2,
    0, 2 * Math.PI,
    false,
    0
  );
  const points2D = curve.getPoints(36);
  const path3D = new THREE.CatmullRomCurve3(points2D.map(p => new THREE.Vector3(p.x, p.y, 0)), true);
  const hoopGeo = new THREE.TubeGeometry(path3D, 36, 0.007, 8, true);
  const hoop = new THREE.Mesh(hoopGeo, frameMat);
  hoop.position.y = headCenterY;
  group.add(hoop);

  // 5. String Bed
  const stringBedGeo = new THREE.ShapeGeometry(new THREE.Shape(points2D));
  const stringBed = new THREE.Mesh(stringBedGeo, stringMat);
  stringBed.position.y = headCenterY;
  group.add(stringBed);

  // 6. String Grid Lines
  const stringLinesGeo = new THREE.BufferGeometry();
  const linePositions = [];
  for (let sx = -headWidth * 0.38; sx <= headWidth * 0.38; sx += 0.026) {
    const normX = sx / (headWidth / 2);
    if (Math.abs(normX) < 0.98) {
      const halfH = (headHeight / 2) * Math.sqrt(Math.max(0, 1 - normX * normX));
      linePositions.push(sx, headCenterY - halfH, 0, sx, headCenterY + halfH, 0);
    }
  }
  for (let sy = -headHeight * 0.42; sy <= headHeight * 0.42; sy += 0.03) {
    const normY = sy / (headHeight / 2);
    if (Math.abs(normY) < 0.98) {
      const halfW = (headWidth / 2) * Math.sqrt(Math.max(0, 1 - normY * normY));
      linePositions.push(-halfW, headCenterY + sy, 0, halfW, headCenterY + sy, 0);
    }
  }
  stringLinesGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  const stringsMesh = new THREE.LineSegments(stringLinesGeo, stringGridMat);
  group.add(stringsMesh);

  group.frustumCulled = false;
  return group;
}

const racketMesh = createRacketMesh();
scene.add(racketMesh);

// Wrist trails (dominant = volt, off-hand = cyan)
function makeTrail(hex) {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(TRAIL_LEN * 3);
  const col = new Float32Array(TRAIL_LEN * 3);
  const c = new THREE.Color(hex);
  for (let i = 0; i < TRAIL_LEN; i++) {
    const f = Math.pow(i / (TRAIL_LEN - 1), 1.6);
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
const _p = new THREE.Vector3();

function precomputeWristTrails(fList) {
  const dIdx = dominantHand === 'Right' ? IDX.R_WR : IDX.L_WR;
  const oIdx = dominantHand === 'Right' ? IDX.L_WR : IDX.R_WR;
  const res = { dom: [], off: [] };
  for (const fr of fList) {
    if (fr.detected && fr.landmarks) {
      res.dom.push(mpToWorld(fr.landmarks[dIdx], new THREE.Vector3()).clone());
      res.off.push(mpToWorld(fr.landmarks[oIdx], new THREE.Vector3()).clone());
    } else {
      const l = res.dom.length;
      res.dom.push(l ? res.dom[l - 1].clone() : new THREE.Vector3());
      res.off.push(l ? res.off[l - 1].clone() : new THREE.Vector3());
    }
  }
  return res;
}

function updateActiveSmoothing() {
  activeFrames = showSmooth ? smoothFrames : rawFrames;
  wristWorld = showSmooth ? wristWorldSmooth : wristWorldRaw;
  const btn = document.getElementById('t-smooth');
  if (btn) btn.classList.toggle('on', showSmooth);
  renderFrame(cur);
}

function loadData(payload, label) {
  if (!payload || payload.meta?.schema !== 'tennis-mediapipe/landmarks3d@1')
    throw new Error('Not a landmarks3d@1 file.');
  data = payload;
  rawFrames = payload.frames;
  nFrames = rawFrames.length;
  fps = payload.meta.fps || 30;
  dominantHand = payload.meta.dominant_hand || 'Right';
  cur = 0; playing = false;

  smoothFrames = smoothFramesData(rawFrames);
  buildBones(payload.connections);

  wristWorldRaw = precomputeWristTrails(rawFrames);
  wristWorldSmooth = precomputeWristTrails(smoothFrames);
  updateActiveSmoothing();

  document.getElementById('scrub').max = nFrames - 1;
  const syn = payload.meta.synthetic;
  document.getElementById('badge').innerHTML =
    `${syn ? '<span class="synthetic">SYNTHETIC DEMO</span>' : '<span class="real">REAL DATA</span>'}<br>${label}<br>${nFrames} frames · ${fps} fps`;
  updatePlayBtn();
  renderFrame(0);
}

// ---------------------------------------------------------------- CSV input
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
  const byFrame = new Map();
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

  // Eye-level horizontal gaze tilt angle (head stability)
  const eyeVec = new THREE.Vector3().subVectors(P[IDX.R_EYE], P[IDX.L_EYE]);
  const eyeLen = eyeVec.length();
  let eyeTilt = 0;
  if (eyeLen > 1e-4) {
    eyeTilt = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(eyeVec.y / eyeLen, -1, 1)));
  }

  return { elbow, knee, tilt, eyeTilt };
}

// ---------------------------------------------------------------- rendering
const _fa = new THREE.Vector3();
const _h = new THREE.Vector3();
const _yWorld = new THREE.Vector3();
const _xWorld = new THREE.Vector3();
const _zWorld = new THREE.Vector3();
const _nPalm = new THREE.Vector3();
const _rotMat = new THREE.Matrix4();

function updateRacketPose(fr) {
  if (!fr || !fr.detected || !fr.landmarks || !showRacket) {
    racketMesh.visible = false;
    return;
  }

  const P = fr.landmarks.map(p => mpToWorld(p, new THREE.Vector3()));
  const isRight = dominantHand === 'Right';
  const wIdx = isRight ? IDX.R_WR : IDX.L_WR;
  const eIdx = isRight ? IDX.R_EL : IDX.L_EL;
  const pIdx = isRight ? IDX.R_PINKY : IDX.L_PINKY;
  const iIdx = isRight ? IDX.R_INDEX : IDX.L_INDEX;
  const tIdx = isRight ? IDX.R_THUMB : IDX.L_THUMB;

  const pW = P[wIdx], pE = P[eIdx], pP = P[pIdx], pI = P[iIdx], pT = P[tIdx];

  // Forearm pointing vector (from elbow to wrist)
  _fa.subVectors(pW, pE).normalize();

  // Hand center & direction (from wrist to knuckles)
  const handCenter = new THREE.Vector3().add(pP).add(pI).add(pT).multiplyScalar(1 / 3);
  _h.subVectors(handCenter, pW);
  const handLen = _h.length();

  // Racket long axis (handle pointing direction)
  if (handLen > 0.02) {
    _h.normalize();
    _yWorld.copy(_fa).multiplyScalar(0.35).addScaledVector(_h, 0.65).normalize();
  } else {
    _yWorld.copy(_fa);
  }

  // Palm normal calculation
  const vIndexPinky = new THREE.Vector3().subVectors(pI, pP);
  const vThumbWrist = new THREE.Vector3().subVectors(pT, pW);
  _nPalm.crossVectors(vIndexPinky, vThumbWrist);

  if (_nPalm.lengthSq() < 1e-4) {
    _nPalm.crossVectors(_yWorld, new THREE.Vector3(0, 1, 0));
    if (_nPalm.lengthSq() < 1e-4) _nPalm.set(0, 0, 1);
  }
  _nPalm.normalize();
  if (!isRight) _nPalm.negate();

  // Construct orthonormal coordinate basis: [X: side width, Y: long axis / shaft, Z: face normal]
  _xWorld.crossVectors(_yWorld, _nPalm).normalize();
  _zWorld.crossVectors(_xWorld, _yWorld).normalize();
  _xWorld.crossVectors(_yWorld, _zWorld).normalize();

  _rotMat.makeBasis(_xWorld, _yWorld, _zWorld);
  racketMesh.quaternion.setFromRotationMatrix(_rotMat);

  // Position handle firmly inside the player's palm
  racketMesh.position.copy(pW).sub(_yWorld.clone().multiplyScalar(0.06));
  racketMesh.visible = true;
}

// ---------------------------------------------------------------- Head & Tennis Sports Cap Pose
const _uRight = new THREE.Vector3();
const _uUp = new THREE.Vector3();
const _uFwd = new THREE.Vector3();
const _rotMatCap = new THREE.Matrix4();
const _pCap = new THREE.Vector3();

function updateCapPose(fr) {
  if (!fr || !fr.detected || !fr.landmarks || !showCap) {
    capMesh.visible = false;
    return;
  }

  const P = fr.landmarks.map(p => mpToWorld(p, new THREE.Vector3()));
  const pLeye = P[IDX.L_EYE];
  const pReye = P[IDX.R_EYE];
  const pNose = P[IDX.NOSE];
  const pLsho = P[IDX.L_SH];
  const pRsho = P[IDX.R_SH];

  const eyeMid = new THREE.Vector3().addVectors(pLeye, pReye).multiplyScalar(0.5);
  const shoMid = new THREE.Vector3().addVectors(pLsho, pRsho).multiplyScalar(0.5);

  const eyeDist = pLeye.distanceTo(pReye);
  const scale = Math.max(0.65, Math.min(1.8, eyeDist / 0.115));

  // 1. Right vector: from Left Eye to Right Eye (towards player's right)
  _uRight.subVectors(pReye, pLeye).normalize();

  // 2. Up vector: from Shoulder midpoint to Eye midpoint
  _uUp.subVectors(eyeMid, shoMid).normalize();

  // 3. Forward vector: gaze direction perpendicular to up and right vectors
  _uFwd.crossVectors(_uUp, _uRight).normalize();

  // Align forward vector with nose direction
  const noseDir = new THREE.Vector3().subVectors(pNose, eyeMid);
  if (_uFwd.dot(noseDir) < 0) {
    _uFwd.negate();
  }

  // Orthogonalize basis
  _uRight.crossVectors(_uUp, _uFwd).normalize();
  _uUp.crossVectors(_uFwd, _uRight).normalize();

  _rotMatCap.makeBasis(_uRight, _uUp, _uFwd);
  capMesh.quaternion.setFromRotationMatrix(_rotMatCap);
  capMesh.scale.setScalar(scale);

  // Position mannequin face & visor centered at eye level
  _pCap.copy(eyeMid)
    .addScaledVector(_uUp, 0.012 * scale)
    .addScaledVector(_uFwd, -0.010 * scale);
  capMesh.position.copy(_pCap);
  capMesh.visible = true;
}

function renderFrame(fFloat) {
  if (activeFrames.length === 0) return;
  const i = Math.max(0, Math.min(nFrames - 1, Math.round(fFloat)));
  const fr = activeFrames[i];
  const ok = fr && fr.detected && fr.landmarks;
  jointMesh.visible = !!ok;
  if (boneLines) boneLines.visible = !!ok && showBones;
  if (neckLine) neckLine.visible = !!ok && showBones;
  trailDom.visible = trailOff.visible = !!ok && showTrails;
  document.getElementById('nopose').style.display = ok ? 'none' : 'block';

  if (ok) {
    const vis = fr.visibility || [];
    for (let j = 0; j < NUM_LM; j++) {
      mpToWorld(fr.landmarks[j], _p);
      dummy.position.copy(_p);

      // Clean Face Filtering
      let s = 1.0;
      if (j === IDX.L_EYE || j === IDX.R_EYE) {
        s = 1.15; // Distinct clean eye markers
      } else if (j === IDX.NOSE) {
        s = 0.55; // Subtle nose marker
      } else if (j <= 10) {
        s = 0.0;  // Hide other facial landmark clutter
      } else {
        s = (vis[j] ?? 1) < 0.4 ? 0.35 : 1.0;
      }

      dummy.scale.setScalar(s);
      dummy.updateMatrix();
      jointMesh.setMatrixAt(j, dummy.matrix);
    }
    jointMesh.instanceMatrix.needsUpdate = true;

    // Bone segments
    const pairs = boneLines.userData.pairs;
    pairs.forEach(([a, b], k) => {
      mpToWorld(fr.landmarks[a], _p);
      bonePos.set([_p.x, _p.y, _p.z], k * 6);
      mpToWorld(fr.landmarks[b], _p);
      bonePos.set([_p.x, _p.y, _p.z], k * 6 + 3);
    });
    boneLines.geometry.attributes.position.needsUpdate = true;

    // Clean Neck connection (Eye midpoint to shoulder midpoint)
    const pLeye = mpToWorld(fr.landmarks[IDX.L_EYE], new THREE.Vector3());
    const pReye = mpToWorld(fr.landmarks[IDX.R_EYE], new THREE.Vector3());
    const pLsho = mpToWorld(fr.landmarks[IDX.L_SH], new THREE.Vector3());
    const pRsho = mpToWorld(fr.landmarks[IDX.R_SH], new THREE.Vector3());
    const eyeMid = new THREE.Vector3().addVectors(pLeye, pReye).multiplyScalar(0.5);
    const shoMid = new THREE.Vector3().addVectors(pLsho, pRsho).multiplyScalar(0.5);
    neckPos.set([eyeMid.x, eyeMid.y, eyeMid.z, shoMid.x, shoMid.y, shoMid.z], 0);
    neckLine.geometry.attributes.position.needsUpdate = true;

    // 3D Tennis Racket & Sports Cap Pose Updates
    updateRacketPose(fr);
    updateCapPose(fr);

    // Trails
    for (const [line, arr] of [[trailDom, wristWorld.dom], [trailOff, wristWorld.off]]) {
      const pos = line.geometry.attributes.position.array;
      for (let k = 0; k < TRAIL_LEN; k++) {
        const src = arr[Math.max(0, i - (TRAIL_LEN - 1 - k))];
        pos.set([src.x, src.y, src.z], k * 3);
      }
      line.geometry.attributes.position.needsUpdate = true;
    }

    const { elbow, knee, tilt, eyeTilt } = frameAngles(fr);
    document.getElementById('h-elbow').textContent = `${elbow.toFixed(0)}°`;
    document.getElementById('h-knee').textContent = `${knee.toFixed(0)}°`;
    document.getElementById('h-torso').textContent = `${tilt.toFixed(0)}°`;
    const hEye = document.getElementById('h-eye');
    if (hEye) hEye.textContent = `${eyeTilt >= 0 ? '+' : ''}${eyeTilt.toFixed(1)}°`;
  } else {
    racketMesh.visible = false;
    capMesh.visible = false;
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

// Toggles
document.getElementById('t-racket').onclick = (e) => {
  showRacket = !showRacket;
  e.target.classList.toggle('on', showRacket);
  renderFrame(cur);
};
document.getElementById('t-cap').onclick = (e) => {
  showCap = !showCap;
  e.target.classList.toggle('on', showCap);
  renderFrame(cur);
};
document.getElementById('t-smooth').onclick = (e) => {
  showSmooth = !showSmooth;
  e.target.classList.toggle('on', showSmooth);
  updateActiveSmoothing();
};
document.getElementById('t-trails').onclick = (e) => {
  showTrails = !showTrails;
  e.target.classList.toggle('on', showTrails);
  renderFrame(cur);
};
document.getElementById('t-bones').onclick = (e) => {
  showBones = !showBones;
  e.target.classList.toggle('on', showBones);
  renderFrame(cur);
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

// CSV upload
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

// CSV paste panel
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
  if (e.key === 'r' || e.key === 'R') {
    showRacket = !showRacket;
    document.getElementById('t-racket')?.classList.toggle('on', showRacket);
    renderFrame(cur);
  }
  if (e.key === 'c' || e.key === 'C') {
    showCap = !showCap;
    document.getElementById('t-cap')?.classList.toggle('on', showCap);
    renderFrame(cur);
  }
  if (e.key === 's' || e.key === 'S') {
    showSmooth = !showSmooth;
    document.getElementById('t-smooth')?.classList.toggle('on', showSmooth);
    updateActiveSmoothing();
  }
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
