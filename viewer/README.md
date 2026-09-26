# 3D Pose Lab — Three.js viewer

An interactive 3D companion to the Tennis Shot Technique Analyzer.
It renders the MediaPipe 33-landmark skeleton in true 3D: orbit around the
player with touch/mouse, scrub frame-by-frame through the swing, and watch
the hitting wrist's motion trail carve its path through space.

## How data flows

```
swing.mp4
  └─► python export_3d_json.py swing.mp4 -o output/landmarks_3d.json
        └─► viewer/index.html  →  "Load JSON"  →  explore in 3D
```

`export_3d_json.py` reuses the existing `TennisVideoProcessor.scan_video()`
pipeline — no changes to the detection code were needed. It pivots the
per-landmark history into a `tennis-mediapipe/landmarks3d@1` JSON document:

- `meta` — fps, frame count, source file, dominant hand, coordinate system,
  and a `synthetic` flag
- `landmark_names` — the 33 MediaPipe names
- `connections` — the project's own bone sets (upper / lower / head)
- `frames[]` — per frame: `{frame, t, detected, landmarks: [[x,y,z] × 33],
  visibility: [...]}`. Undetected frames carry `landmarks: null`.

MediaPipe normalized coordinates are mapped to Three.js world space as
`X = (x-0.5)·S`, `Y = (0.5-y)·S`, `Z = -z·S`.

## Running it

The viewer is a static site (Three.js via CDN). Because browsers block
`fetch()` on `file://`, serve it over HTTP:

```bash
cd viewer
python3 -m http.server 8080
# open http://localhost:8080
```

It loads `sample/sample_landmarks.json` by default. Use **Load JSON** to open
a `landmarks_3d.json` exported from a real swing video.

## Demo data

`sample/` contains `generate_sample_landmarks.py`, which procedurally
animates a simple skeleton through a forehand-style swing and writes a
schema-identical JSON with `meta.synthetic: true`. It is clearly badged
"SYNTHETIC DEMO" in the viewer — regenerate or replace it any time:

```bash
cd viewer/sample
python generate_sample_landmarks.py -o sample_landmarks.json
```

## Controls

- **Orbit**: one-finger drag / mouse drag · **Zoom**: pinch / wheel ·
  **Pan**: two-finger drag / right-drag
- **Camera presets** (right rail): Front / Side / Back / Top / Reset, plus
  Trails, Bones, and auto-Spin toggles
- **Transport** (bottom deck): first / prev / play-pause / next / last,
  frame scrubber, 0.25×–2× speed, loop toggle
- **Keyboard**: Space = play/pause, ←/→ = step frame
- **HUD** (top left): live hitting-elbow bend, loading-knee bend, torso tilt,
  computed in 3D from the landmark positions each frame
