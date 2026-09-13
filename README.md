# 🎾 Tennis Shot Technique Analyzer (MediaPipe AI)

A local Python web application designed to analyze tennis shot mechanics using Google MediaPipe Pose Detection, OpenCV, Pandas, and Plotly.

---

## 🌟 Key Features

1. **Local Video Management**:
   - Automatically scans and indexes video files placed in `./videos/` (`.mp4`, `.mov`, `.avi`, `.mkv`).
   - Clean UI selector and video metadata inspector (Resolution, FPS, Duration, Frame count).
   - Informative empty-state guide when `./videos/` is empty.

2. **Google MediaPipe 33-Landmark 3D Pose AI**:
   - Detects all 33 full-body 3D landmarks in real time.
   - Computes tennis-specific biomechanical kinematics:
     - **Elbow Flexion/Extension**: Shoulder $\rightarrow$ Elbow $\rightarrow$ Wrist (hitting arm vs non-dominant arm).
     - **Shoulder Elevation / Abduction**: Hip $\rightarrow$ Shoulder $\rightarrow$ Elbow.
     - **Knee Bend Angle**: Hip $\rightarrow$ Knee $\rightarrow$ Ankle for loading and extension power transfer.
     - **Torso Spine Tilt**: Angle of spine vector relative to vertical upright axis.
     - **Shoulder Line Tilt & Hip Rotation**: Upper and lower body coiling/alignment.
     - **Head Stability & Eye Level**: Eye-line horizontal tilt and vertical gaze tracking.
   - **Handedness Customization**: Right-handed or Left-handed configuration with full dual-side tracking.
   - **Angle Measurement Mode ("Amount of Bend")**:
     - **Amount of Bend (Default)**: Intuitive physical coaching convention where **$0^\circ$ = completely straight limb**, **$90^\circ$ = square L-bend**, and deeper loading knee bend = $50^\circ - 70^\circ$.
     - **Interior Angle**: Standard geometric convention where **$180^\circ$ = locked straight** and **$90^\circ$ = square bend**.
     - Seamlessly toggleable in the sidebar with instant live updates across the frame inspector, HUD badges, and Plotly charts.

3. **Pro Coaching HUD Aesthetics**:
   - Semi-transparent Heads-Up Display (HUD) overlay card showing live frame metrics.
   - Color-coded anatomy: Right joints (Amber/Orange), Left joints (Cyan), Spine/Center (Tennis Volt Yellow).
   - Colored joint angle arcs & degrees directly displayed on joint vertices.
   - **Dual Wrist Motion Path Trails**: Simultaneous trajectory ribbons for both the hitting wrist and the non-dominant wrist with fading velocity trail effect.
   - Granular UI toggles to show/hide Upper Body, Lower Body, Head/Eyes, Joint Angles, Wrist Trails, and HUD badge.

4. **Frame Inspector & Kinematic Telemetry**:
   - **Frame Scrubber**: Slider and frame stepping controls (`⏮ -5`, `◀ -1`, `+1 ▶`, `+5 ⏭`) for frame-by-frame analysis.
   - **Kinematic Curves (Plotly)**: Synchronized time-series plots for elbow angle, knee loading, torso tilt, and eye-level tilt.
   - **Export Engine**:
     - Export annotated MP4 video to `./output/` with selectable playback speeds (`0.25x Super Slow-Mo`, `0.5x Slow-Mo`, `1.0x Normal`, `2.0x Fast`).
     - Uses H.264 encoding via `imageio-ffmpeg` / OpenCV for immediate in-browser and QuickTime playback.
     - Dual-level CSV export: Primary biomechanical telemetry CSV (`{name}_telemetry.csv`) and complete 33-landmark 3D coordinate CSV (`{name}_landmarks3d.csv`).

---

## 🚀 Setup & Installation (macOS / Apple Silicon & Intel)

### 1. Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12.
- Terminal shell (zsh or bash).

### 2. Clone or Navigate to Directory
```bash
cd /Users/nikbat/tennis-mediapipe
```

### 3. Create and Activate Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎬 How to Run

### Step 1: Add Tennis Video Files
Place one or more video files of tennis shots (forehands, backhands, serves) in the local `videos/` folder:
```bash
cp /path/to/your/tennis_shot.mp4 ./videos/
```

### Step 2: Launch the Web Dashboard
```bash
streamlit run app.py
```
Streamlit will automatically open your default web browser at `http://localhost:8501`.

---

## 📂 Project Architecture

```
tennis-mediapipe/
├── detector.py          # MediaPipe Pose pipeline & vector angle mathematics
├── video_processor.py   # Video I/O, HUD rendering engine, H.264 & CSV export
├── app.py               # Streamlit web dashboard with 3 interactive tabs
├── requirements.txt     # Dependency specifications
├── README.md            # Documentation & setup instructions
├── videos/              # Local folder for input tennis video files
└── output/              # Local folder for generated annotated videos & CSVs
```

---

## 📊 Exported CSV Schema

### 1. Primary Telemetry CSV (`{video}_telemetry.csv`)
| Column | Description |
|---|---|
| `frame` | Frame index (0-indexed) |
| `timestamp_sec` | Elapsed time in seconds |
| `dominant_hand` | "Right" or "Left" |
| `hitting_elbow_angle` | Flexion/extension angle of hitting arm elbow (degrees) |
| `non_dominant_elbow_angle` | Flexion/extension angle of non-dominant elbow (degrees) |
| `hitting_shoulder_angle` | Abduction/elevation angle of hitting shoulder (degrees) |
| `non_dominant_shoulder_angle` | Abduction/elevation angle of non-dominant shoulder (degrees) |
| `loading_knee_angle` | Flexion angle of loading leg knee (degrees) |
| `non_dominant_knee_angle` | Flexion angle of non-dominant knee (degrees) |
| `torso_tilt_deg` | Tilt of spine relative to vertical axis (degrees) |
| `shoulder_tilt_deg` | Horizontal tilt of shoulder axis (degrees) |
| `hip_tilt_deg` | Horizontal tilt of hip axis (degrees) |
| `eye_level_tilt_deg` | Tilt of eye-line relative to horizontal (degrees) |
| `head_y_norm` | Normalized vertical coordinate of head center |
| `dom_wrist_x`, `dom_wrist_y` | Pixel coordinates of dominant hitting wrist |
| `non_dom_wrist_x`, `non_dom_wrist_y` | Pixel coordinates of non-dominant wrist |

### 2. Full 3D Landmarks CSV (`{video}_landmarks3d.csv`)
Contains normalized $(x, y, z)$ coordinates and detection visibility for all 33 skeletal joints across every frame of the video.
