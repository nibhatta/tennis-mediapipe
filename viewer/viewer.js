/**
 * 3D Pose Lab — Three.js viewer for tennis-mediapipe landmark data.
 *
 * Renders an immaculate volumetric 3D biomechanical mannequin with:
 * - Solid pearlescent white bone cylinders and smooth joint spheres
 * - Smooth egg-shaped ellipsoid head, neck, anatomical spine & torso cage
 * - Rectangular frame tennis racket simulation with dynamic wrist orientation
 * - Clean, subtle light mesh court ground (no heavy lines or net)
 * - Biomechanical ground-contact & airborne jump elevation calibration algorithm
 * - Multi-pass Savitzky-Golay trajectory smoothing
 * - Continuous sub-frame cubic Catmull-Rom spline interpolation for 60/120fps fluid playback
 * - Full 360-degree unconstrained orbit controls
 * - Wrist swing motion trails and live biomechanical readouts
 *
 * Coordinate mapping (MediaPipe normalized -> Three.js world):
 *   X = (x - 0.5) * SCALE          (left-to-right stays left-to-right)
 *   Y = (0.5 - y) * SCALE          (flip: MediaPipe y grows downward)
 *   Z = -z * SCALE                 (MediaPipe -z is toward the camera)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------- palette & settings
const COL = {
  bg: 0x050508,       // Pitch black studio background
  boneWhite: 0xf3f4f6, // Pearlescent studio white
  jointWhite: 0xf8fafc,
  volt: 0xc6e014,      // Accent for dominant trail / telemetry
  cyan: 0x38bdf8,      // Accent for off-hand trail
  grid: 0x1e293b,
  courtLine: 0xffffff,
  dim: 0x8b96a8,
};
const SCALE = 3.0;          // World units per normalized unit
const TRAIL_LEN = 30;       // Wrist trail length in frames
const SPEEDS = [0.25, 0.5, 1, 2];
const COURT_Y = -0.78;      // Standard 3D court level

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

// Volumetric mannequin bone connections
const MANNEQUIN_BONES = [
  // Upper Torso
  { id: 'neck', isSpecial: 'neck' },                          // Shoulder center -> Head base
  { id: 'spine', isSpecial: 'spine' },                        // Shoulder center -> Hip center
  { a: IDX.L_SH, b: IDX.R_SH },                               // Shoulder bar
  { a: IDX.L_HIP, b: IDX.R_HIP },                             // Hip bar
  // Side Torso
  { a: IDX.L_SH, b: IDX.L_HIP },                              // Left torso bar
  { a: IDX.R_SH, b: IDX.R_HIP },                              // Right torso bar
  // Left Arm
  { a: IDX.L_SH, b: IDX.L_EL },                               // Left upper arm
  { a: IDX.L_EL, b: IDX.L_WR },                               // Left forearm
  // Right Arm
  { a: IDX.R_SH, b: IDX.R_EL },                               // Right upper arm
  { a: IDX.R_EL, b: IDX.R_WR },                               // Right forearm
  // Left Leg
  { a: IDX.L_HIP, b: IDX.L_KN },                              // Left thigh
  { a: IDX.L_KN, b: IDX.L_AN },                               // Left shin
  { a: IDX.L_AN, b: IDX.L_HEEL },                             // Left heel
  { a: IDX.L_HEEL, b: IDX.L_FOOT },                           // Left foot base
  { a: IDX.L_AN, b: IDX.L_FOOT },                             // Left instep
  // Right Leg
  { a: IDX.R_HIP, b: IDX.R_KN },                              // Right thigh
  { a: IDX.R_KN, b: IDX.R_AN },                               // Right shin
  { a: IDX.R_AN, b: IDX.R_HEEL },                             // Right heel
  { a: IDX.R_HEEL, b: IDX.R_FOOT },                           // Right foot base
  { a: IDX.R_AN, b: IDX.R_FOOT },                             // Right instep
];

// Joint sphere indices to render
const MANNEQUIN_JOINTS = [
  IDX.L_SH, IDX.R_SH,
  IDX.L_EL, IDX.R_EL,
  IDX.L_WR, IDX.R_WR,
  IDX.L_HIP, IDX.R_HIP,
  IDX.L_KN, IDX.R_KN,
  IDX.L_AN, IDX.R_AN,
  IDX.L_HEEL, IDX.R_HEEL,
  IDX.L_FOOT, IDX.R_FOOT,
];

// ---------------------------------------------------------------- Advanced Multi-Pass Smoothing
const SG_11 = [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36].map(v => v / 429);
const SG_7 = [-2, 3, 6, 7, 6, 3, -2].map(v => v / 21);

function applySavitzkyGolay1D(arr, coeffs) {
  const n = arr.length;
  const half = Math.floor(coeffs.length / 2);
  if (n < coeffs.length) return arr.slice();
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

function smoothFramesData(rawFramesList) {
  const n = rawFramesList.length;
  if (n < 5) return rawFramesList;

  const smoothed = rawFramesList.map(f => ({
    frame: f.frame,
    t: f.t,
    detected: f.detected,
    visibility: f.visibility ? [...f.visibility] : null,
    landmarks: f.landmarks ? f.landmarks.map(p => (p ? [...p] : null)) : null,
  }));

  const detIdxs = [];
  for (let i = 0; i < n; i++) {
    if (rawFramesList[i].detected && rawFramesList[i].landmarks) {
      detIdxs.push(i);
    }
  }

  const m = detIdxs.length;
  if (m < 5) return smoothed;

  const coeffs = m >= 11 ? SG_11 : SG_7;

  for (let j = 0; j < 33; j++) {
    for (let d = 0; d < 3; d++) {
      const series = new Float64Array(m);
      for (let k = 0; k < m; k++) {
        series[k] = rawFramesList[detIdxs[k]].landmarks[j][d];
      }
      // Pass 1: Savitzky-Golay polynomial filter
      const pass1 = applySavitzkyGolay1D(series, coeffs);
      // Pass 2: Gentle Gaussian low-pass smoothing
      const pass2 = new Float64Array(m);
      for (let k = 0; k < m; k++) {
        const pPrev = pass1[Math.max(0, k - 1)];
        const pCurr = pass1[k];
        const pNext = pass1[Math.min(m - 1, k + 1)];
        pass2[k] = pPrev * 0.2 + pCurr * 0.6 + pNext * 0.2;
      }

      for (let k = 0; k < m; k++) {
        smoothed[detIdxs[k]].landmarks[j][d] = pass2[k];
      }
    }
  }

  return smoothed;
}

// ---------------------------------------------------------------- Biomechanical Grounding & Airborne Elevation Algorithm
function calibrateKinematicElevation(framesList) {
  const n = framesList.length;
  if (n === 0) return new Float32Array(0);

  const rawOffsets = new Float32Array(n);
  const frameStats = [];

  for (let i = 0; i < n; i++) {
    const fr = framesList[i];
    if (!fr.detected || !fr.landmarks) {
      frameStats.push(null);
      continue;
    }
    const lms = fr.landmarks;

    const pLAnkle = mpToWorldVec(lms[IDX.L_AN], new THREE.Vector3());
    const pRAnkle = mpToWorldVec(lms[IDX.R_AN], new THREE.Vector3());
    const pLHeel = mpToWorldVec(lms[IDX.L_HEEL], new THREE.Vector3());
    const pRHeel = mpToWorldVec(lms[IDX.R_HEEL], new THREE.Vector3());
    const pLFoot = mpToWorldVec(lms[IDX.L_FOOT], new THREE.Vector3());
    const pRFoot = mpToWorldVec(lms[IDX.R_FOOT], new THREE.Vector3());

    const pLHip = mpToWorldVec(lms[IDX.L_HIP], new THREE.Vector3());
    const pRHip = mpToWorldVec(lms[IDX.R_HIP], new THREE.Vector3());
    const hipMidY = (pLHip.y + pRHip.y) * 0.5;

    const pLKnee = mpToWorldVec(lms[IDX.L_KN], new THREE.Vector3());
    const pRKnee = mpToWorldVec(lms[IDX.R_KN], new THREE.Vector3());

    // Lowest foot point in raw world space
    const minFootY = Math.min(pLAnkle.y, pRAnkle.y, pLHeel.y, pRHeel.y, pLFoot.y, pRFoot.y);

    // Anatomical leg span
    const currentLegSpan = hipMidY - minFootY;

    // Knee bend angles
    const v1L = new THREE.Vector3().subVectors(pLHip, pLKnee).normalize();
    const v2L = new THREE.Vector3().subVectors(pLAnkle, pLKnee).normalize();
    const kneeBendL = 180 - THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(v1L.dot(v2L), -1, 1)));

    const v1R = new THREE.Vector3().subVectors(pRHip, pRKnee).normalize();
    const v2R = new THREE.Vector3().subVectors(pRAnkle, pRKnee).normalize();
    const kneeBendR = 180 - THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(v1R.dot(v2R), -1, 1)));

    const maxKneeBend = Math.max(kneeBendL, kneeBendR);

    frameStats.push({
      minFootY,
      hipMidY,
      currentLegSpan,
      maxKneeBend,
    });
  }

  // Step 2: Establish Ground Baseline Level
  // Collect feet Y values during stable grounded preparation & trophy load frames
  const groundedFootYs = [];
  for (let i = 0; i < n; i++) {
    const s = frameStats[i];
    if (!s) continue;
    if (s.maxKneeBend > 20 || i < n * 0.35 || i > n * 0.8) {
      groundedFootYs.push(s.minFootY);
    }
  }
  if (groundedFootYs.length === 0) {
    for (let i = 0; i < n; i++) {
      if (frameStats[i]) groundedFootYs.push(frameStats[i].minFootY);
    }
  }
  groundedFootYs.sort((a, b) => a - b);
  // Median grounded foot level in raw coordinates
  const rawGroundY = groundedFootYs[Math.floor(groundedFootYs.length * 0.45)] || 0;

  // Step 3: Compute Ground Offsets & Airborne Jump Flight
  for (let i = 0; i < n; i++) {
    const s = frameStats[i];
    if (!s) {
      rawOffsets[i] = COURT_Y - rawGroundY;
      continue;
    }

    const groundedOffset = COURT_Y - s.minFootY;
    const footLift = s.minFootY - rawGroundY;

    // Biomechanical airborne condition:
    // When the lowest foot is elevated above the baseline ground plane AND
    // knees are extending / in post-trophy flight phase (explosive upward leg drive):
    if (footLift > 0.04 && s.maxKneeBend < 40) {
      // Body is airborne! Ground reference stays at COURT_Y, allowing feet to lift into the air
      rawOffsets[i] = COURT_Y - rawGroundY;
    } else {
      // Grounded on court: Pin lowest foot firmly to the court mesh floor (COURT_Y)
      rawOffsets[i] = groundedOffset;
    }
  }

  // Step 4: Multi-pass smooth the vertical ground offset trajectory
  const smoothedOffsets = applySavitzkyGolay1D(rawOffsets, SG_11);
  return smoothedOffsets;
}

// ---------------------------------------------------------------- state
let data = null;
let rawFrames = [];
let smoothFrames = [];
let activeFrames = [];
let verticalOffsets = new Float32Array(0);
let nFrames = 0;
let fps = 30;
let cur = 0;
let playing = false;
let loop = true;
let speedIdx = 2;
let lastT = 0;
let scrubbing = false;
let wasPlaying = false;
let showTrails = true;
let showBones = true;
let showRacket = true;
let showSmooth = true;
let dominantHand = 'Right';
let wristWorld = { dom: [], off: [] };

// ---------------------------------------------------------------- three setup
const container = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(COL.bg);

const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.1, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.15, 0);
controls.minDistance = 0.8;
controls.maxDistance = 16;
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

// Studio 3-Point Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.85));

const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
keyLight.position.set(4, 7, 5);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xdbeafe, 1.0);
fillLight.position.set(-5, 4, 3);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 1.2);
rimLight.position.set(0, 5, -6);
scene.add(rimLight);

// ---------------------------------------------------------------- Light Mesh Court Environment
function createTennisCourtMesh() {
  const courtGroup = new THREE.Group();

  // Dark ground plane
  const floorGeo = new THREE.PlaneGeometry(20, 24);
  const floorMat = new THREE.MeshStandardMaterial({
    color: 0x050508,
    roughness: 0.95,
    metalness: 0.05,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = COURT_Y - 0.002;
  courtGroup.add(floor);

  // Light wireframe grid mesh representing the court
  const gridHelper = new THREE.GridHelper(16, 32, 0x475569, 0x1e293b);
  gridHelper.position.y = COURT_Y;
  courtGroup.add(gridHelper);

  // Small origin marker
  const originMarker = new THREE.Mesh(
    new THREE.ConeGeometry(0.025, 0.06, 8),
    new THREE.MeshBasicMaterial({ color: 0xfacc15 })
  );
  originMarker.position.set(0, COURT_Y + 0.03, 0);
  courtGroup.add(originMarker);

  return courtGroup;
}
scene.add(createTennisCourtMesh());

// ---------------------------------------------------------------- Volumetric 3D Mannequin Skeleton Objects
const NUM_BONES = MANNEQUIN_BONES.length;
const NUM_JOINTS = MANNEQUIN_JOINTS.length + 2; // + shoulder mid & hip mid

// Studio Bone & Joint Material
const mannequinMat = new THREE.MeshStandardMaterial({
  color: COL.boneWhite,
  roughness: 0.28,
  metalness: 0.06,
});

// 1. Instanced Bone Cylinders
const boneRadius = 0.018;
const boneCylinderGeo = new THREE.CylinderGeometry(boneRadius, boneRadius, 1.0, 20);
boneCylinderGeo.translate(0, 0.5, 0); // Origin at cylinder base
const boneCylinderMesh = new THREE.InstancedMesh(boneCylinderGeo, mannequinMat, NUM_BONES);
boneCylinderMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
boneCylinderMesh.frustumCulled = false;
scene.add(boneCylinderMesh);

// 2. Instanced Joint Spheres
const jointRadius = 0.025;
const jointSphereGeo = new THREE.SphereGeometry(jointRadius, 20, 16);
const jointSphereMesh = new THREE.InstancedMesh(jointSphereGeo, mannequinMat, NUM_JOINTS);
jointSphereMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
jointSphereMesh.frustumCulled = false;
scene.add(jointSphereMesh);

// 3. Egg-Shaped Mannequin Head
const headGeo = new THREE.SphereGeometry(1.0, 32, 24);
headGeo.scale(0.062, 0.095, 0.075);
const headMesh = new THREE.Mesh(headGeo, mannequinMat);
headMesh.frustumCulled = false;
scene.add(headMesh);

// ---------------------------------------------------------------- 3D Rectangular Tennis Racket
function createRectangularRacketMesh() {
  const group = new THREE.Group();

  const handleMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8,
    roughness: 0.4,
    metalness: 0.2,
  });
  const frameMat = new THREE.MeshStandardMaterial({
    color: COL.jointWhite,
    roughness: 0.25,
    metalness: 0.08,
  });
  const stringMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.15,
    side: THREE.DoubleSide,
  });

  const gripLen = 0.18;
  const gripRadius = 0.012;
  const headW = 0.17;
  const headH = 0.24;
  const tubeR = 0.0075;

  // 1. Handle Rod
  const gripGeo = new THREE.CylinderGeometry(gripRadius, gripRadius, gripLen, 16);
  gripGeo.translate(0, gripLen / 2, 0);
  const grip = new THREE.Mesh(gripGeo, handleMat);
  group.add(grip);

  // 2. Rectangular 3D Frame Head
  const rectPts = [
    new THREE.Vector3(-headW / 2, 0, 0),
    new THREE.Vector3(-headW / 2, headH, 0),
    new THREE.Vector3(headW / 2, headH, 0),
    new THREE.Vector3(headW / 2, 0, 0),
  ];
  const rectCurve = new THREE.CatmullRomCurve3(rectPts, true, 'catmullrom', 0.05);
  const frameGeo = new THREE.TubeGeometry(rectCurve, 32, tubeR, 8, true);
  const frame = new THREE.Mesh(frameGeo, frameMat);
  frame.position.y = gripLen;
  group.add(frame);

  // 3. Translucent String Face Plane
  const stringGeo = new THREE.PlaneGeometry(headW - tubeR * 2, headH - tubeR * 2);
  const stringBed = new THREE.Mesh(stringGeo, stringMat);
  stringBed.position.set(0, gripLen + headH / 2, 0);
  group.add(stringBed);

  group.frustumCulled = false;
  return group;
}

const racketMesh = createRectangularRacketMesh();
scene.add(racketMesh);

// ---------------------------------------------------------------- Wrist Trails
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

// ---------------------------------------------------------------- coordinate & sub-frame interpolation
function mpToWorldVec(lm, out) {
  out.set((lm[0] - 0.5) * SCALE, (0.5 - lm[1]) * SCALE, -lm[2] * SCALE);
  return out;
}

const _interpP = Array.from({ length: 33 }, () => new THREE.Vector3());

function catmullRom1D(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (
    (2 * p1) +
    (-p0 + p2) * t +
    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
  );
}

function getInterpolatedLandmarks(fFloat) {
  if (activeFrames.length === 0) return null;
  const clamped = THREE.MathUtils.clamp(fFloat, 0, nFrames - 1);
  const i = Math.floor(clamped);
  const alpha = clamped - i;

  const f0 = activeFrames[Math.max(0, i - 1)];
  const f1 = activeFrames[i];
  const f2 = activeFrames[Math.min(nFrames - 1, i + 1)];
  const f3 = activeFrames[Math.min(nFrames - 1, i + 2)];

  if (!f1 || !f1.detected || !f1.landmarks) return null;

  const lm0 = f0 && f0.landmarks ? f0.landmarks : f1.landmarks;
  const lm1 = f1.landmarks;
  const lm2 = f2 && f2.landmarks ? f2.landmarks : f1.landmarks;
  const lm3 = f3 && f3.landmarks ? f3.landmarks : lm2;

  // Interpolate vertical ground alignment offset
  const v0 = verticalOffsets[Math.max(0, i - 1)] || 0;
  const v1 = verticalOffsets[i] || 0;
  const v2 = verticalOffsets[Math.min(nFrames - 1, i + 1)] || 0;
  const v3 = verticalOffsets[Math.min(nFrames - 1, i + 2)] || 0;
  const yOffset = catmullRom1D(v0, v1, v2, v3, alpha);

  for (let j = 0; j < 33; j++) {
    const p0 = lm0[j], p1 = lm1[j], p2 = lm2[j], p3 = lm3[j];
    const x = catmullRom1D(p0[0], p1[0], p2[0], p3[0], alpha);
    const y = catmullRom1D(p0[1], p1[1], p2[1], p3[1], alpha);
    const z = catmullRom1D(p0[2], p1[2], p2[2], p3[2], alpha);
    mpToWorldVec([x, y, z], _interpP[j]);
    _interpP[j].y += yOffset; // Grounded on court mesh / airborne jump elevation!
  }

  return _interpP;
}

function precomputeWristTrails(fList, offsets) {
  const dIdx = dominantHand === 'Right' ? IDX.R_WR : IDX.L_WR;
  const oIdx = dominantHand === 'Right' ? IDX.L_WR : IDX.R_WR;
  const res = { dom: [], off: [] };
  for (let i = 0; i < fList.length; i++) {
    const fr = fList[i];
    const yOff = offsets[i] || 0;
    if (fr.detected && fr.landmarks) {
      const pDom = mpToWorldVec(fr.landmarks[dIdx], new THREE.Vector3());
      pDom.y += yOff;
      const pOff = mpToWorldVec(fr.landmarks[oIdx], new THREE.Vector3());
      pOff.y += yOff;
      res.dom.push(pDom);
      res.off.push(pOff);
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
  verticalOffsets = calibrateKinematicElevation(activeFrames);
  wristWorld = precomputeWristTrails(activeFrames, verticalOffsets);
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
  let curVal = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') { curVal += '"'; i++; }
        else quoted = false;
      } else curVal += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { out.push(curVal); curVal = ''; }
    else curVal += c;
  }
  out.push(curVal);
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
    throw new Error('No usable landmark rows found.');

  const maxF = Math.max(...byFrame.keys());
  let parsedFps = 30;
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
        parsedFps = Math.round(1000 / med) / 1000;
        for (const std of [23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 120]) {
          if (Math.abs(parsedFps - std) / std < 0.02) { parsedFps = std; break; }
        }
      }
    }
  }

  const framesList = [];
  let detectedCount = 0;
  for (let f = 0; f <= maxF; f++) {
    const e = byFrame.get(f);
    if (e && e.rows.size === 33) {
      const lms = [], vis = [];
      for (let j = 0; j < 33; j++) { lms.push(e.rows.get(j)); vis.push(e.vis.get(j) ?? 1.0); }
      framesList.push({
        frame: f, t: e.t !== null ? e.t : Math.round(f / parsedFps * 1000) / 1000,
        detected: true, landmarks: lms, visibility: vis,
      });
      detectedCount++;
    } else {
      framesList.push({
        frame: f, t: Math.round(f / parsedFps * 1000) / 1000,
        detected: false, landmarks: null, visibility: null,
      });
    }
  }
  if (detectedCount === 0)
    throw new Error('No complete frames found: every frame needs all 33 landmarks.');

  return {
    meta: {
      schema: 'tennis-mediapipe/landmarks3d@1',
      fps: parsedFps, frame_count: framesList.length,
      duration_sec: Math.round(framesList.length / parsedFps * 1000) / 1000,
      source: label || 'pasted.csv',
      synthetic: /sample/i.test(label || ''),
      dominant_hand: 'Right',
      coordinate_system: 'mediapipe_normalized',
      generated: new Date().toISOString(),
      notes: 'Parsed from CSV in 3D Pose Lab.',
    },
    landmark_names: LANDMARK_NAMES,
    frames: framesList,
  };
}

function computeKinematicAngles(P) {
  const v1 = new THREE.Vector3(), v2 = new THREE.Vector3();
  function bend3(ai, bi, ci) {
    v1.subVectors(P[ai], P[bi]); v2.subVectors(P[ci], P[bi]);
    const cos = THREE.MathUtils.clamp(v1.normalize().dot(v2.normalize()), -1, 1);
    return 180 - THREE.MathUtils.radToDeg(Math.acos(cos));
  }
  const isRight = dominantHand === 'Right';
  const elbow = bend3(isRight ? IDX.R_SH : IDX.L_SH, isRight ? IDX.R_EL : IDX.L_EL, isRight ? IDX.R_WR : IDX.L_WR);
  const knee = bend3(isRight ? IDX.R_HIP : IDX.L_HIP, isRight ? IDX.R_KN : IDX.L_KN, isRight ? IDX.R_AN : IDX.L_AN);
  const hip = new THREE.Vector3().addVectors(P[IDX.L_HIP], P[IDX.R_HIP]).multiplyScalar(0.5);
  const sho = new THREE.Vector3().addVectors(P[IDX.L_SH], P[IDX.R_SH]).multiplyScalar(0.5);
  const torso = new THREE.Vector3().subVectors(sho, hip).normalize();
  const tilt = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(torso.y, -1, 1)));

  const eyeVec = new THREE.Vector3().subVectors(P[IDX.R_EYE], P[IDX.L_EYE]);
  const eyeLen = eyeVec.length();
  let eyeTilt = 0;
  if (eyeLen > 1e-4) {
    eyeTilt = THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(eyeVec.y / eyeLen, -1, 1)));
  }

  return { elbow, knee, tilt, eyeTilt };
}

// ---------------------------------------------------------------- volumetric skeleton rendering
const _dummy = new THREE.Object3D();
const _vecY = new THREE.Vector3(0, 1, 0);
const _boneDir = new THREE.Vector3();
const _quat = new THREE.Quaternion();

function setCylinderTransform(instancedMesh, index, pA, pB, radius = 0.018) {
  _boneDir.subVectors(pB, pA);
  const len = _boneDir.length();
  if (len < 1e-4) {
    _dummy.position.set(0, -999, 0);
    _dummy.scale.set(0, 0, 0);
    _dummy.updateMatrix();
    instancedMesh.setMatrixAt(index, _dummy.matrix);
    return;
  }
  _boneDir.normalize();
  _quat.setFromUnitVectors(_vecY, _boneDir);

  _dummy.position.copy(pA);
  _dummy.quaternion.copy(_quat);
  _dummy.scale.set(radius / 0.018, len, radius / 0.018);
  _dummy.updateMatrix();
  instancedMesh.setMatrixAt(index, _dummy.matrix);
}

function setJointTransform(instancedMesh, index, pos, radius = 0.025) {
  _dummy.position.copy(pos);
  _dummy.quaternion.identity();
  _dummy.scale.setScalar(radius / 0.025);
  _dummy.updateMatrix();
  instancedMesh.setMatrixAt(index, _dummy.matrix);
}

// Racket positioning
const _fa = new THREE.Vector3();
const _h = new THREE.Vector3();
const _yWorld = new THREE.Vector3();
const _xWorld = new THREE.Vector3();
const _zWorld = new THREE.Vector3();
const _nPalm = new THREE.Vector3();
const _rotMat = new THREE.Matrix4();

function updateRacketPose(P) {
  if (!showRacket || !P) {
    racketMesh.visible = false;
    return;
  }

  const isRight = dominantHand === 'Right';
  const wIdx = isRight ? IDX.R_WR : IDX.L_WR;
  const eIdx = isRight ? IDX.R_EL : IDX.L_EL;
  const pIdx = isRight ? IDX.R_PINKY : IDX.L_PINKY;
  const iIdx = isRight ? IDX.R_INDEX : IDX.L_INDEX;
  const tIdx = isRight ? IDX.R_THUMB : IDX.L_THUMB;

  const pW = P[wIdx], pE = P[eIdx], pP = P[pIdx], pI = P[iIdx], pT = P[tIdx];

  _fa.subVectors(pW, pE).normalize();
  const handCenter = new THREE.Vector3().add(pP).add(pI).add(pT).multiplyScalar(1 / 3);
  _h.subVectors(handCenter, pW);
  const handLen = _h.length();

  if (handLen > 0.02) {
    _h.normalize();
    _yWorld.copy(_fa).multiplyScalar(0.35).addScaledVector(_h, 0.65).normalize();
  } else {
    _yWorld.copy(_fa);
  }

  const vIndexPinky = new THREE.Vector3().subVectors(pI, pP);
  const vThumbWrist = new THREE.Vector3().subVectors(pT, pW);
  _nPalm.crossVectors(vIndexPinky, vThumbWrist);

  if (_nPalm.lengthSq() < 1e-4) {
    _nPalm.crossVectors(_yWorld, new THREE.Vector3(0, 1, 0));
    if (_nPalm.lengthSq() < 1e-4) _nPalm.set(0, 0, 1);
  }
  _nPalm.normalize();
  if (!isRight) _nPalm.negate();

  _xWorld.crossVectors(_yWorld, _nPalm).normalize();
  _zWorld.crossVectors(_xWorld, _yWorld).normalize();
  _xWorld.crossVectors(_yWorld, _zWorld).normalize();

  _rotMat.makeBasis(_xWorld, _yWorld, _zWorld);
  racketMesh.quaternion.setFromRotationMatrix(_rotMat);
  racketMesh.position.copy(pW).sub(_yWorld.clone().multiplyScalar(0.04));
  racketMesh.visible = true;
}

function renderFrame(fFloat) {
  if (activeFrames.length === 0) return;

  const P = getInterpolatedLandmarks(fFloat);
  const ok = !!P;

  boneCylinderMesh.visible = ok && showBones;
  jointSphereMesh.visible = ok && showBones;
  headMesh.visible = ok && showBones;
  trailDom.visible = trailOff.visible = ok && showTrails;
  document.getElementById('nopose').style.display = ok ? 'none' : 'block';

  if (ok) {
    const shoMid = new THREE.Vector3().addVectors(P[IDX.L_SH], P[IDX.R_SH]).multiplyScalar(0.5);
    const hipMid = new THREE.Vector3().addVectors(P[IDX.L_HIP], P[IDX.R_HIP]).multiplyScalar(0.5);
    const eyeMid = new THREE.Vector3().addVectors(P[IDX.L_EYE], P[IDX.R_EYE]).multiplyScalar(0.5);

    // Spine direction & Head base
    const spineDir = new THREE.Vector3().subVectors(shoMid, hipMid).normalize();
    const headBase = new THREE.Vector3().copy(shoMid).addScaledVector(spineDir, 0.14);
    const headCenter = new THREE.Vector3().copy(shoMid).addScaledVector(spineDir, 0.23);

    // 1. Render Bone Cylinders
    MANNEQUIN_BONES.forEach((bone, bIdx) => {
      let pA, pB;
      if (bone.isSpecial === 'neck') {
        pA = shoMid;
        pB = headBase;
      } else if (bone.isSpecial === 'spine') {
        pA = hipMid;
        pB = shoMid;
      } else {
        pA = P[bone.a];
        pB = P[bone.b];
      }
      setCylinderTransform(boneCylinderMesh, bIdx, pA, pB, boneRadius);
    });
    boneCylinderMesh.instanceMatrix.needsUpdate = true;

    // 2. Render Joint Spheres
    MANNEQUIN_JOINTS.forEach((jIdx, k) => {
      setJointTransform(jointSphereMesh, k, P[jIdx], jointRadius);
    });
    setJointTransform(jointSphereMesh, MANNEQUIN_JOINTS.length, shoMid, jointRadius);
    setJointTransform(jointSphereMesh, MANNEQUIN_JOINTS.length + 1, hipMid, jointRadius);
    jointSphereMesh.instanceMatrix.needsUpdate = true;

    // 3. Render Egg-Shaped Head
    headMesh.position.copy(headCenter);
    const headUp = new THREE.Vector3().subVectors(eyeMid, shoMid).normalize();
    const headRight = new THREE.Vector3().subVectors(P[IDX.R_EYE], P[IDX.L_EYE]).normalize();
    const headFwd = new THREE.Vector3().crossVectors(headUp, headRight).normalize();
    const headBasis = new THREE.Matrix4().makeBasis(headRight, headUp, headFwd);
    headMesh.quaternion.setFromRotationMatrix(headBasis);
    headMesh.visible = showBones;

    // 4. Render Rectangular Tennis Racket
    updateRacketPose(P);

    // 5. Render Fluid Wrist Trails
    const iFloat = THREE.MathUtils.clamp(fFloat, 0, nFrames - 1);
    const currIdx = Math.round(iFloat);
    for (const [line, arr] of [[trailDom, wristWorld.dom], [trailOff, wristWorld.off]]) {
      const pos = line.geometry.attributes.position.array;
      for (let k = 0; k < TRAIL_LEN; k++) {
        const src = arr[Math.max(0, currIdx - (TRAIL_LEN - 1 - k))];
        pos.set([src.x, src.y, src.z], k * 3);
      }
      line.geometry.attributes.position.needsUpdate = true;
    }

    // 6. Live Telemetry
    const { elbow, knee, tilt, eyeTilt } = computeKinematicAngles(P);
    document.getElementById('h-elbow').textContent = `${elbow.toFixed(0)}°`;
    document.getElementById('h-knee').textContent = `${knee.toFixed(0)}°`;
    document.getElementById('h-torso').textContent = `${tilt.toFixed(0)}°`;
    const hEye = document.getElementById('h-eye');
    if (hEye) hEye.textContent = `${eyeTilt >= 0 ? '+' : ''}${eyeTilt.toFixed(1)}°`;
  } else {
    racketMesh.visible = false;
  }

  const intFrame = Math.round(fFloat);
  if (!scrubbing) document.getElementById('scrub').value = intFrame;
  document.getElementById('timelabel').innerHTML =
    `<b>F ${intFrame}</b> / ${nFrames - 1} · ${(fFloat / fps).toFixed(2)}s`;
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

// ---------------------------------------------------------------- continuous 60fps main loop
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
