# 3D Pose Lab — Three.js viewer

An interactive 3D companion to the Tennis Shot Technique Analyzer.
It renders the MediaPipe 33-landmark skeleton in true 3D: orbit around the
player with touch/mouse, scrub frame-by-frame through the swing, and watch
the hitting wrist's motion trail carve its path through space.

## How data flows

```
swing.mp4
  ├─► python export_3d_json.py swing.mp4 -o output/landmarks_3d.json
  │     └─► viewer/index.html  →  "Load JSON"  →  explore in 3D
  │
  └─► Streamlit app (app.py), Tab 3 "Render & Export Video"
        └─► output/<name>_landmarks3d.json   (written next to the MP4 + CSVs)
              └─► viewer/index.html  →  "Load JSON"  →  explore in 3D
```

`export_3d_json.py` reuses the existing `TennisVideoProcessor.scan_video()`
pipeline — no changes to the detection code were needed. It pivots the
per-landmark history into a `tennis-mediapipe/landmarks3d@1` JSON document.
The same payload builder (`TennisVideoProcessor.build_landmarks3d_payload`)
is used by the Streamlit app: Tab 2 offers "Download 3D JSON (Pose Lab)",
and Tab 3's video export writes `<name>_landmarks3d.json` next to the
annotated MP4 automatically, so the 3D scene always matches the rendered video.

The document layout:

- `meta` — fps, frame count, source file, dominant hand, coordinate system,
  and a `synthetic` flag
- `landmark_names` — the 33 MediaPipe names
- `connections` — the project's own bone sets (upper / lower / head)
- `frames[]` — per frame: `{frame, t, detected, landmarks: [[x,y,z] × 33],
  visibility: [...]}`. Undetected frames carry `landmarks: null`.

MediaPipe normalized coordinates are mapped to Three.js world space as
`X = (x-0.5)·S`, `Y = (0.5-y)·S`, `Z = -z·S`.

## Loading data: JSON, CSV file, or pasted CSV

The viewer accepts three input shapes, all flowing through the same render path:

1. **Load JSON** — a `landmarks_3d.json` file (schema above).
2. **Load CSV** — upload a 33-landmark CSV file. This is the exact CSV the
   analyzer already exports ("Download Full 33-Landmark 3D CSV"):
   `frame,timestamp_sec,landmark_index,landmark_name,x,y,z,visibility`.
   `landmark_index` or `landmark_name` (or both) identify the joint;
   `visibility` and `timestamp_sec` are optional. Playback fps is estimated
   from `timestamp_sec` when present, otherwise 30.
3. **Paste CSV** — paste the same CSV text into the panel. "Fill sample CSV"
   drops in the bundled demo data so you can try it instantly.

Every frame needs all 33 landmarks (indices 0–32); frames with fewer are
shown as "no pose", and gaps in frame numbers are padded the same way.
Pasting the *telemetry* CSV (joint angles) gives a clear error explaining
which CSV to use instead.

## Running it

The viewer is a static site (Three.js via CDN). Because browsers block
`fetch()` on `file://`, serve it over HTTP:

```bash
cd viewer
python3 -m http.server 8080
# open http://localhost:8080
```

It loads `sample/sample_landmarks.json` by default. Use **Load JSON** to open
a `landmarks_3d.json` exported from a real swing video, **Load CSV** for a
landmarks CSV file, or **Paste CSV** to paste CSV text directly.

## Demo data

`sample/` contains `generate_sample_landmarks.py`, which procedurally
animates a simple skeleton through a forehand-style swing and writes a
schema-identical JSON with `meta.synthetic: true`. It is clearly badged
"SYNTHETIC DEMO" in the viewer — regenerate or replace it any time:

```bash
cd viewer/sample
python generate_sample_landmarks.py -o sample_landmarks.json
python generate_sample_landmarks.py --format csv -o sample_landmarks.csv
```

## Controls

- **Orbit**: one-finger drag / mouse drag with full 360° unconstrained rotation (overhead, level, and underneath) · **Zoom**: pinch / wheel · **Pan**: two-finger drag / right-drag
- **Camera presets & toggles** (right rail):
  - Front / Side / Back / Top / Reset
  - **🎾 Racket**: simulates realistic 3D rectangular frame tennis racket matching stroke orientation
  - **✨ Smooth**: Multi-pass Savitzky-Golay filtering and sub-frame cubic Catmull-Rom spline interpolation
  - **Trails**: dual wrist swing motion trails (dominant volt + non-dominant cyan)
  - **Bones**: volumetric studio white mannequin skeleton
  - **Spin**: turntable auto-rotation
- **Transport** (bottom deck): first / prev / play-pause / next / last, frame scrubber, 0.25×–2× speed, loop toggle
- **Keyboard**: Space = play/pause, ←/→ = step frame, R = toggle racket, S = toggle smoothing
- **HUD** (top left): live hitting-elbow bend, loading-knee bend, torso tilt, and eye-level tilt computed continuously in 3D
- **Studio Mannequin Anatomy**: Solid pearlescent white bone cylinders, smooth joint spheres, egg-shaped head, neck, anatomical spine & torso cage, and 3D tennis court & net background environment.
