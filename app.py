"""
Tennis Shot Technique Analyzer - Streamlit Web Dashboard.
Powered by Google MediaPipe Pose Detection, OpenCV, Pandas, and Plotly.
"""

import os
import glob
import time
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from detector import TennisPoseDetector
from video_processor import TennisVideoProcessor


# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tennis Technique Analyzer | MediaPipe AI",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Pro Coaching CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main background & accent styling */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }

    /* Header Banner */
    .app-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #C6E00F 0%, #38BDF8 60%, #FFFFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(198, 224, 15, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 1.1rem;
        color: #C6E00F;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Video Banner Alert */
    .empty-video-card {
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 36px 24px;
        text-align: center;
        margin: 30px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border-radius: 8px;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(198, 224, 15, 0.15) !important;
        border-color: #C6E00F !important;
        color: #C6E00F !important;
        font-weight: 600;
    }

    /* High-Contrast Legible Buttons */
    .stButton > button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #334155 !important;
        color: #C6E00F !important;
        border-color: #C6E00F !important;
        box-shadow: 0 4px 14px 0 rgba(198, 224, 15, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        background-color: #0F172A !important;
        transform: translateY(0px) !important;
    }

    /* Primary Action Buttons */
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #C6E00F 0%, #A6C208 100%) !important;
        color: #0B0F19 !important;
        border: 1px solid #C6E00F !important;
        font-weight: 700 !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #D4ED15 0%, #B8D608 100%) !important;
        color: #000000 !important;
        box-shadow: 0 4px 18px 0 rgba(198, 224, 15, 0.45) !important;
    }

    /* Disabled Buttons */
    .stButton > button:disabled,
    .stButton > button[disabled] {
        background-color: #1E293B !important;
        color: #64748B !important;
        border-color: #334155 !important;
        opacity: 0.5 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stDownloadButton > button:hover {
        background-color: #334155 !important;
        color: #38BDF8 !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 4px 14px 0 rgba(56, 189, 248, 0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Silent Global Keyboard Arrow Navigation (±5 frames)
# Works in both maximized/fullscreen image view and standard inspector view
components.html(
    """
    <script>
    (function() {
        try {
            const topWin = window.parent || window;
            const topDoc = topWin.document || document;

            function handleArrowNav(e) {
                // Ignore if user is currently focused on an editable form input
                const target = e.target;
                if (target) {
                    const tag = (target.tagName || '').toLowerCase();
                    if (tag === 'input' || tag === 'textarea' || tag === 'select' || target.isContentEditable) {
                        return;
                    }
                }

                if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
                    const btns = Array.from(topDoc.querySelectorAll('button'));
                    const next5 = btns.find(b => b.textContent && (b.textContent.includes('+5') || b.textContent.includes('+5 ⏭')));
                    if (next5) {
                        e.preventDefault();
                        e.stopPropagation();
                        next5.click();
                    }
                } else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
                    const btns = Array.from(topDoc.querySelectorAll('button'));
                    const prev5 = btns.find(b => b.textContent && (b.textContent.includes('-5') || b.textContent.includes('⏮ -5')));
                    if (prev5) {
                        e.preventDefault();
                        e.stopPropagation();
                        prev5.click();
                    }
                }
            }

            // Bind to top window capture phase
            if (!topWin._tennisArrowNavAttached) {
                topWin._tennisArrowNavAttached = true;
                topWin.addEventListener('keydown', handleArrowNav, true);
                topDoc.addEventListener('keydown', handleArrowNav, true);
            }
        } catch(err) {
            console.warn("Arrow nav listener setup:", err);
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)

VIDEOS_DIR = "./videos"
OUTPUT_DIR = "./output"
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_available_videos() -> List[str]:
    """Lists video files located in ./videos/."""
    valid_extensions = (".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm")
    all_files = os.listdir(VIDEOS_DIR) if os.path.exists(VIDEOS_DIR) else []
    video_files = [f for f in all_files if f.lower().endswith(valid_extensions)]
    return sorted(video_files)


@st.cache_resource
def get_detector(dominant_hand: str, angle_mode: str = "bend") -> TennisPoseDetector:
    """Returns a cached TennisPoseDetector instance."""
    return TennisPoseDetector(dominant_hand=dominant_hand, angle_mode=angle_mode)


@st.cache_resource
def get_processor(_detector: TennisPoseDetector) -> TennisVideoProcessor:
    """Returns a cached TennisVideoProcessor instance."""
    return TennisVideoProcessor(detector=_detector)


@st.cache_data
def get_video_thumbnail(video_path: str) -> Optional[np.ndarray]:
    """Extracts a scaled thumbnail image (240px wide) from video."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_f = min(15, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame = cap.read()
    cap.release()
    if ret:
        h, w = frame.shape[:2]
        thumb_w = 240
        thumb_h = max(1, int(h * (thumb_w / w)))
        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return None


def load_frame_at_index(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Directly extracts a specific frame index from video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">🎾 Tennis Shot Technique Analyzer</h1>
        <p class="app-subtitle">
            Biomechanical Pose Estimation & Joint Kinematics powered by Google MediaPipe 33-Landmark Pose AI
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Sidebar: Video Management & Biomechanical Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚙️ Video & Shot Settings")

available_videos = get_available_videos()

if not available_videos:
    selected_video_name = None
    selected_video_path = None
    st.sidebar.warning("⚠️ No video files found in `./videos/`")
    st.sidebar.info("👉 Drop `.mp4`, `.mov`, or `.avi` files into `./videos/` folder.")
else:
    # Check if a video was selected via visual thumbnail gallery
    default_idx = 0
    if "selected_video_override" in st.session_state and st.session_state.selected_video_override in available_videos:
        default_idx = available_videos.index(st.session_state.selected_video_override)

    selected_video_name = st.sidebar.selectbox(
        "Select Local Video",
        available_videos,
        index=default_idx,
        help="Select a video file from the local ./videos/ folder",
    )
    selected_video_path = os.path.join(VIDEOS_DIR, selected_video_name)

    # Show Thumbnail Preview in Sidebar
    sidebar_thumb = get_video_thumbnail(selected_video_path)
    if sidebar_thumb is not None:
        st.sidebar.image(
            sidebar_thumb,
            use_container_width=True,
            caption=f"Preview: {selected_video_name}",
        )

    # Inspect Video Metadata
    try:
        meta = TennisVideoProcessor.get_video_metadata(selected_video_path)
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(15,23,42,0.6); padding: 10px 14px; border-radius: 8px; font-size: 0.85rem; margin-top: 6px; border: 1px solid rgba(255,255,255,0.06);">
                <b>Resolution:</b> {meta['width']}x{meta['height']}<br>
                <b>Framerate:</b> {meta['fps']} FPS<br>
                <b>Total Frames:</b> {meta['frame_count']}<br>
                <b>Duration:</b> {meta['duration_sec']}s
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.sidebar.error(f"Error reading video metadata: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🥋 Player Stance & Biomechanics")

dominant_hand = st.sidebar.radio(
    "Dominant Hand (Hitting Arm)",
    ["Right", "Left"],
    index=0,
    horizontal=True,
    help="Assigns dominant arm for stroke metrics (forehand/serve) and loading leg.",
)

angle_mode_choice = st.sidebar.radio(
    "Angle Measurement Mode",
    ["Amount of Bend (0° = Straight)", "Interior Angle (180° = Straight)"],
    index=0,
    help="Amount of Bend: 0° = straight arm/leg, 90° = square bend, deep knee bend = 50°-70°.",
)
mode_key = "bend" if "Bend" in angle_mode_choice else "interior"

trail_length = st.sidebar.slider(
    "Wrist Motion Trail Length",
    min_value=5,
    max_value=120,
    value=30,
    step=5,
    help="Number of historical frames to render in wrist swing path trail (supports up to 120 frames).",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Landmark Overlay Toggles")

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    toggle_upper = st.checkbox("Upper Body", value=True)
    toggle_lower = st.checkbox("Lower Body", value=True)
    toggle_head = st.checkbox("Head / Eyes", value=True)

with col_t2:
    toggle_angles = st.checkbox("Joint Angles", value=True)
    toggle_trails = st.checkbox("Wrist Trails", value=True)
    toggle_hud = st.checkbox("Coaching HUD", value=True)

head_style_choice = st.sidebar.selectbox(
    "Head / Face Overlay Style",
    [
        "🎯 Gaze Stabilizer (Head Center + Eye Axis)",
        "👀 Eyeline Only",
        "⭕ Minimal Head Halo",
        "🌐 Full 10-Point Mesh (Close-Up View)",
        "🚫 Off (Body Skeleton Only)",
    ],
    index=0,
    help="Gaze Stabilizer replaces the clump of 10 face dots with a single clean head marker and horizontal eye-tilt stabilizer axis.",
)

if "Gaze" in head_style_choice:
    head_style = "gaze_bar"
elif "Eyeline" in head_style_choice:
    head_style = "eyeline_only"
elif "Halo" in head_style_choice:
    head_style = "minimal_circle"
elif "Full" in head_style_choice:
    head_style = "full_mesh"
else:
    head_style = "off"

toggles = {
    "show_upper_body": toggle_upper,
    "show_lower_body": toggle_lower,
    "show_head": toggle_head,
    "head_style": head_style,
    "show_angles": toggle_angles,
    "show_wrist_trails": toggle_trails,
    "show_hud_telemetry": toggle_hud,
}


# -----------------------------------------------------------------------------
# Main Content Area
# -----------------------------------------------------------------------------
if not selected_video_path or not os.path.exists(selected_video_path):
    st.markdown(
        """
        <div class="empty-video-card">
            <h2 style="color: #94A3B8; font-weight: 600; margin-bottom: 12px;">📁 No Video Files Found</h2>
            <p style="color: #64748B; max-width: 520px; margin: 0 auto 20px auto; line-height: 1.6;">
                The designated <code>./videos/</code> folder is currently empty.
                Please copy or place your tennis shot video files (<code>.mp4</code>, <code>.mov</code>, <code>.avi</code>)
                into the local directory:
            </p>
            <div style="background: rgba(15,23,42,0.9); display: inline-block; padding: 12px 24px; border-radius: 8px; border: 1px solid rgba(198, 224, 15, 0.3);">
                <code style="color: #C6E00F; font-size: 1rem;">/Users/nikbat/tennis-mediapipe/videos/</code>
            </div>
            <p style="color: #64748B; font-size: 0.85rem; margin-top: 20px;">
                Once videos are added to that folder, refresh this page or select from the sidebar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# Initialize detector and processor
detector = get_detector(dominant_hand=dominant_hand, angle_mode=mode_key)
detector.set_dominant_hand(dominant_hand)
detector.set_angle_mode(mode_key)
processor = get_processor(detector)

# Session state management for scanned telemetry
if "scanned_video" not in st.session_state:
    st.session_state.scanned_video = None
if "telemetry_df" not in st.session_state:
    st.session_state.telemetry_df = None
if "landmarks_history" not in st.session_state:
    st.session_state.landmarks_history = None
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0

# If video selection changed, invalidate previous scan
if st.session_state.scanned_video != selected_video_name:
    st.session_state.scanned_video = None
    st.session_state.telemetry_df = None
    st.session_state.landmarks_history = None
    st.session_state.current_frame_idx = 0


# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Scan & Frame Inspector",
    "📈 Kinematic Curves & Telemetry",
    "🎬 Video Export & Player",
])


# =============================================================================
# TAB 1: Scan & Frame Inspector
# =============================================================================
with tab1:
    col_scan, col_meta = st.columns([3, 2])
    with col_scan:
        st.markdown("#### ⚡ Biomechanical Pre-Scan")
        st.write("Extract 33 full-body 3D landmarks and calculate all joint kinematics across the shot.")

        scan_btn = st.button("🚀 Run Pre-Scan on Video", type="primary", use_container_width=True)

    with col_meta:
        if st.session_state.telemetry_df is not None:
            st.success(f"✅ Pre-scan complete: {len(st.session_state.telemetry_df)} frames indexed.")
        else:
            st.info("💡 Run pre-scan to enable fast frame scrubbing and kinematic curve plotting.")

    # Execute Pre-Scan
    if scan_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(p: float, text: str):
            progress_bar.progress(p)
            status_text.text(text)

        with st.spinner("Processing video frames with MediaPipe Pose..."):
            start_t = time.time()
            df_telemetry, lms_history = processor.scan_video(
                selected_video_path, progress_callback=update_progress
            )
            elapsed = time.time() - start_t

        progress_bar.empty()
        status_text.empty()
        st.session_state.telemetry_df = df_telemetry
        st.session_state.landmarks_history = lms_history
        st.session_state.scanned_video = selected_video_name
        st.success(f"🎉 Analyzed {len(df_telemetry)} frames in {elapsed:.1f}s!")
        st.rerun()

    st.markdown("---")

    # Frame Inspector Controls
    total_frames = meta["frame_count"]
    fps = meta["fps"]

    st.markdown("#### 🎞️ Interactive Frame Inspector")

    # Stepping Buttons & Slider Row
    c_prev5, c_prev1, c_slider, c_next1, c_next5 = st.columns([1, 1, 6, 1, 1])

    with c_prev5:
        if st.button("⏮ -5", use_container_width=True):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 5)
    with c_prev1:
        if st.button("◀ -1", use_container_width=True):
            st.session_state.current_frame_idx = max(0, st.session_state.current_frame_idx - 1)
    with c_next1:
        if st.button("+1 ▶", use_container_width=True):
            st.session_state.current_frame_idx = min(total_frames - 1, st.session_state.current_frame_idx + 1)
    with c_next5:
        if st.button("+5 ⏭", use_container_width=True):
            st.session_state.current_frame_idx = min(total_frames - 1, st.session_state.current_frame_idx + 5)

    with c_slider:
        selected_frame_idx = st.slider(
            "Frame Navigation",
            min_value=0,
            max_value=max(0, total_frames - 1),
            value=st.session_state.current_frame_idx,
            step=1,
            label_visibility="collapsed",
        )
        st.session_state.current_frame_idx = selected_frame_idx

    curr_idx = st.session_state.current_frame_idx
    timestamp_sec = curr_idx / fps if fps > 0 else 0.0

    # Quick Stroke Navigation if scanned
    if st.session_state.telemetry_df is not None:
        detected_frames = st.session_state.telemetry_df.dropna(subset=["hitting_elbow_angle"])["frame"].tolist()
        if detected_frames:
            col_j1, col_j2, col_j3 = st.columns([2, 2, 3])
            with col_j1:
                prev_hits = [f for f in detected_frames if f < curr_idx]
                if st.button("🎯 Jump to Prev Stroke", disabled=len(prev_hits) == 0, use_container_width=True):
                    st.session_state.current_frame_idx = prev_hits[-1]
                    st.rerun()
            with col_j2:
                next_hits = [f for f in detected_frames if f > curr_idx]
                if st.button("🎯 Jump to Next Stroke", disabled=len(next_hits) == 0, use_container_width=True):
                    st.session_state.current_frame_idx = next_hits[0]
                    st.rerun()
            with col_j3:
                is_curr_detected = curr_idx in detected_frames
                if is_curr_detected:
                    st.markdown("<div style='padding: 8px 12px; background: rgba(198, 224, 15, 0.15); border: 1px solid #C6E00F; border-radius: 6px; font-size: 0.85rem; color: #C6E00F;'>✅ Active Stroke Pose Tracked</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='padding: 8px 12px; background: rgba(148, 163, 184, 0.1); border-radius: 6px; font-size: 0.85rem; color: #94A3B8;'>ℹ️ {len(detected_frames)} stroke frames indexed</div>", unsafe_allow_html=True)

    # Retrieve current frame from disk
    raw_frame = load_frame_at_index(selected_video_path, curr_idx)

    if raw_frame is not None:
        # Run MediaPipe pose on current frame with static_mode for robust random scrubbing
        frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        results = detector.process_frame(frame_rgb, static_mode=True)
        h, w = raw_frame.shape[:2]
        frame_metrics = detector.extract_metrics(
            results, frame_width=w, frame_height=h, frame_idx=curr_idx, timestamp_sec=timestamp_sec
        )

        # Build wrist trail history up to current frame from telemetry_df if scanned
        wrist_trails = {"dominant": [], "non_dominant": []}
        if st.session_state.telemetry_df is not None:
            sub_df = st.session_state.telemetry_df.iloc[max(0, curr_idx - trail_length): curr_idx + 1]
            for _, row in sub_df.iterrows():
                if not np.isnan(row.get("dom_wrist_x", np.nan)):
                    wrist_trails["dominant"].append((int(row["dom_wrist_x"]), int(row["dom_wrist_y"])))
                if not np.isnan(row.get("non_dom_wrist_x", np.nan)):
                    wrist_trails["non_dominant"].append((int(row["non_dom_wrist_x"]), int(row["non_dom_wrist_y"])))

        # Render HUD on frame
        annotated_frame = processor.render_hud_frame(
            raw_frame,
            frame_metrics,
            wrist_trails=wrist_trails,
            toggles=toggles,
            frame_idx=curr_idx,
            fps=fps,
        )

        # Display Viewport & Live Telemetry KPI Cards
        col_view, col_kpi = st.columns([3, 2])

        with col_view:
            st.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Frame {curr_idx:04d} / {total_frames} ({timestamp_sec:.2f}s)",
            )

        with col_kpi:
            st.markdown("##### 📊 Live Kinematic Readout")

            if frame_metrics:
                hit_elbow = frame_metrics.get("hitting_elbow_angle", np.nan)
                off_elbow = frame_metrics.get("non_dominant_elbow_angle", np.nan)
                load_knee = frame_metrics.get("loading_knee_angle", np.nan)
                off_knee = frame_metrics.get("non_dominant_knee_angle", np.nan)
                torso = frame_metrics.get("torso_tilt_deg", np.nan)
                head_tilt = frame_metrics.get("eye_level_tilt_deg", np.nan)

                if mode_key == "bend":
                    card1_label = f"🎾 HITTING ELBOW BEND ({dominant_hand.upper()})"
                    card1_sub = f"Non-dominant: {off_elbow:.1f}° bend | 0° = straight arm | 90° = square L-bend"
                    card2_label = "🦵 LOADING KNEE BEND"
                    card2_sub = f"Non-dominant knee: {off_knee:.1f}° bend | 0° = standing straight | 50° - 70° = deep trophy load"
                else:
                    card1_label = f"🎾 HITTING ELBOW ANGLE ({dominant_hand.upper()})"
                    card1_sub = f"Non-dominant: {off_elbow:.1f}° | Optimal stroke range: 90° - 165°"
                    card2_label = "🦵 LOADING KNEE BEND ANGLE"
                    card2_sub = f"Non-dominant knee: {off_knee:.1f}° | Deep loading: 100° - 130°"

                # Card 1: Hitting Arm Elbow
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{card1_label}</div>
                        <div class="metric-value">{hit_elbow:.1f}<span class="metric-unit">°</span></div>
                        <div class="metric-sub">{card1_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Card 2: Loading Leg Knee Bend
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{card2_label}</div>
                        <div class="metric-value">{load_knee:.1f}<span class="metric-unit">°</span></div>
                        <div class="metric-sub">{card2_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Card 3: Torso Spine Tilt
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">📐 Torso Tilt (Spine Angle)</div>
                        <div class="metric-value">{torso:.1f}<span class="metric-unit">°</span></div>
                        <div class="metric-sub">Upright axis deviation | Power transfer tilt</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Card 4: Head Stability & Eye Level
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">👀 Eye-Line Horizontal Tilt</div>
                        <div class="metric-value">{head_tilt:.1f}<span class="metric-unit">°</span></div>
                        <div class="metric-sub">Head stability metric | 0° represents level gaze</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("⚠️ No body pose detected in current frame.")
    else:
        st.error(f"Could not read frame {curr_idx} from video.")


# =============================================================================
# TAB 2: Kinematic Curves & Telemetry
# =============================================================================
with tab2:
    st.markdown("#### 📈 Kinematic Trajectories & Biomechanical Curves")

    if st.session_state.telemetry_df is None:
        st.info("ℹ️ Please run the **Biomechanical Pre-Scan** in Tab 1 to generate kinematic curves.")
    else:
        df = st.session_state.telemetry_df

        # Dynamically determine bend vs interior series for Plotly
        if mode_key == "bend":
            sub_title_elbow = "Elbow Bend (°): Hitting Arm vs Non-Dominant Arm (0° = Straight)"
            sub_title_knee = "Knee Bend (°): Loading Leg vs Support Leg (0° = Straight)"
            y_axis_title = "Bend (°)"
            if "hitting_elbow_bend" in df.columns:
                plot_hit_elbow = df["hitting_elbow_bend"]
                plot_non_elbow = df["non_dominant_elbow_bend"]
                plot_load_knee = df["loading_knee_bend"]
                plot_non_knee = df["non_dominant_knee_bend"]
            else:
                plot_hit_elbow = (180.0 - df["hitting_elbow_angle"]).clip(lower=0.0)
                plot_non_elbow = (180.0 - df["non_dominant_elbow_angle"]).clip(lower=0.0)
                plot_load_knee = (180.0 - df["loading_knee_angle"]).clip(lower=0.0)
                plot_non_knee = (180.0 - df["non_dominant_knee_angle"]).clip(lower=0.0)
        else:
            sub_title_elbow = "Elbow Interior Angle (°): Hitting Arm vs Non-Dominant Arm (180° = Straight)"
            sub_title_knee = "Knee Interior Angle (°): Loading Leg vs Support Leg (180° = Straight)"
            y_axis_title = "Interior Angle (°)"
            plot_hit_elbow = df["hitting_elbow_angle"]
            plot_non_elbow = df["non_dominant_elbow_angle"]
            plot_load_knee = df["loading_knee_angle"]
            plot_non_knee = df["non_dominant_knee_angle"]

        # Interactive Plotly Subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                sub_title_elbow,
                sub_title_knee,
                "Torso Tilt & Eye-Level Tilt (°)",
            ),
        )

        # 1. Elbow Plot
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=plot_hit_elbow,
                mode="lines",
                name=f"Hitting Elbow ({dominant_hand})",
                line=dict(color="#C6E00F", width=2.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=plot_non_elbow,
                mode="lines",
                name="Non-Dominant Elbow",
                line=dict(color="#38BDF8", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # 2. Knee Plot
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=plot_load_knee,
                mode="lines",
                name=f"Loading Knee ({dominant_hand})",
                line=dict(color="#F59E0B", width=2.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=plot_non_knee,
                mode="lines",
                name="Non-Dominant Knee",
                line=dict(color="#94A3B8", width=1.5, dash="dot"),
            ),
            row=2,
            col=1,
        )

        # 3. Torso & Head Stability Plot
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=df["torso_tilt_deg"],
                mode="lines",
                name="Torso Tilt",
                line=dict(color="#EC4899", width=2.0),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp_sec"],
                y=df["eye_level_tilt_deg"],
                mode="lines",
                name="Eye-Level Tilt",
                line=dict(color="#A855F7", width=1.5),
            ),
            row=3,
            col=1,
        )

        # Current frame vertical cursor line
        curr_time = st.session_state.current_frame_idx / fps if fps > 0 else 0.0
        for r in (1, 2, 3):
            fig.add_vline(
                x=curr_time,
                line_width=1.5,
                line_dash="dash",
                line_color="#FFFFFF",
                row=r,
                col=1,
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15,23,42,0.4)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            height=650,
            margin=dict(l=40, r=30, t=50, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
        fig.update_yaxes(title_text="Degrees (°)")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📋 Telemetry Data Table & CSV Downloads")

        # Telemetry Data Table Preview
        st.dataframe(df.head(50), use_container_width=True)

        # CSV Download Actions
        col_d1, col_d2 = st.columns(2)

        clean_csv = df.to_csv(index=False).encode("utf-8")
        with col_d1:
            st.download_button(
                label="📥 Download Primary Telemetry CSV",
                data=clean_csv,
                file_name=f"{os.path.splitext(selected_video_name)[0]}_telemetry.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_d2:
            if st.session_state.landmarks_history:
                lms_df = pd.DataFrame(st.session_state.landmarks_history)
                lms_csv = lms_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full 33-Landmark 3D CSV",
                    data=lms_csv,
                    file_name=f"{os.path.splitext(selected_video_name)[0]}_landmarks3d.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =============================================================================
# TAB 3: Full Video Export & Player
# =============================================================================
with tab3:
    st.markdown("#### 🎬 Export Annotated Video & In-Browser Player")
    st.write(
        "Render the complete shot with skeletal overlays, color-coded dual wrist motion trails, "
        "and the Pro Coaching HUD overlay to a browser-compatible H.264 MP4 file."
    )

    col_exp_cfg, col_exp_btn = st.columns([2, 1])

    with col_exp_cfg:
        speed_option = st.select_slider(
            "Playback Speed Multiplier (Slow-Motion Control)",
            options=[0.25, 0.5, 1.0, 2.0],
            value=1.0,
            format_func=lambda s: f"{s}x {'(Super Slow-Mo)' if s == 0.25 else '(Slow-Mo)' if s == 0.5 else '(Normal)' if s == 1.0 else '(Fast)'}",
            help="Adjusts output video playback speed. 0.25x or 0.5x slows down the swing for biomechanical review.",
        )

    base_name = os.path.splitext(selected_video_name)[0]
    output_video_filename = f"{base_name}_annotated_{speed_option}x.mp4"
    output_video_filepath = os.path.join(OUTPUT_DIR, output_video_filename)
    output_telemetry_csv = os.path.join(OUTPUT_DIR, f"{base_name}_telemetry.csv")
    output_landmarks_csv = os.path.join(OUTPUT_DIR, f"{base_name}_landmarks3d.csv")

    with col_exp_btn:
        st.write("")
        st.write("")
        export_btn = st.button("🚀 Render & Export Video", type="primary", use_container_width=True)

    if export_btn:
        exp_progress = st.progress(0.0)
        exp_status = st.empty()

        def update_export_progress(p: float, text: str):
            exp_progress.progress(p)
            exp_status.text(text)

        with st.spinner("Rendering annotated frames with Pro Coaching HUD..."):
            start_exp = time.time()
            # 1. Export video
            processor.export_annotated_video(
                video_path=selected_video_path,
                output_path=output_video_filepath,
                speed_multiplier=speed_option,
                toggles=toggles,
                trail_length=trail_length,
                progress_callback=update_export_progress,
            )

            # 2. Export CSV files if telemetry is available
            if st.session_state.telemetry_df is not None:
                TennisVideoProcessor.export_telemetry_csv(
                    st.session_state.telemetry_df, output_telemetry_csv
                )
            if st.session_state.landmarks_history is not None:
                TennisVideoProcessor.export_landmarks_csv(
                    st.session_state.landmarks_history, output_landmarks_csv
                )

            total_render_t = time.time() - start_exp

        exp_progress.empty()
        exp_status.empty()
        st.success(f"🎉 Export complete in {total_render_t:.1f}s!")

    # Check if annotated video already exists in ./output/
    if os.path.exists(output_video_filepath):
        st.markdown("---")
        st.markdown("##### 📺 Annotated Video Playback")

        col_player, col_details = st.columns([3, 2])

        with col_player:
            # Embedded HTML5 Video Player
            st.video(output_video_filepath)

        with col_details:
            st.markdown("##### 📁 Exported Files")
            st.markdown(
                f"""
                <div style="background: rgba(15,23,42,0.7); padding: 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08); font-size: 0.88rem;">
                    <b>Annotated Video:</b><br>
                    <code style="color: #C6E00F;">{output_video_filepath}</code><br><br>
                    <b>Telemetry CSV:</b><br>
                    <code style="color: #38BDF8;">{output_telemetry_csv}</code><br><br>
                    <b>3D Landmarks CSV:</b><br>
                    <code style="color: #F59E0B;">{output_landmarks_csv}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with open(output_video_filepath, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                label="⬇️ Download Processed MP4",
                data=video_bytes,
                file_name=output_video_filename,
                mime="video/mp4",
                use_container_width=True,
            )
