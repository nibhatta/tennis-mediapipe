"""
Tennis Video Processor & Pro Coaching HUD Drawing Engine.
Handles video stream inspection, full landmark pre-scan caching,
color-coded skeletal overlays, joint angle arcs, dual wrist motion trails,
and browser-compatible H.264 video + telemetry CSV export.
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any, Callable
import cv2
import numpy as np
import pandas as pd
from detector import TennisPoseDetector

try:
    import imageio
except ImportError:
    imageio = None


class TennisVideoProcessor:
    """
    Manages video frame extraction, pose overlay rendering with Pro Coaching HUD aesthetics,
    and output rendering (MP4 & CSV).
    """

    # HUD Color Palette (BGR format for OpenCV)
    COLOR_BG_CARD = (20, 24, 33)          # Dark Slate Card background
    COLOR_BG_BORDER = (55, 65, 81)        # Slate Border
    COLOR_VOLT_YELLOW = (20, 224, 198)    # Tennis Volt / Neon Green-Yellow
    COLOR_CYAN = (240, 200, 30)           # Cyan for Left side / Non-dominant
    COLOR_ORANGE = (30, 140, 255)         # Orange/Amber for Right side
    COLOR_WHITE = (245, 245, 245)
    COLOR_RED = (60, 60, 230)
    COLOR_TEXT_DIM = (160, 175, 190)

    def __init__(self, detector: Optional[TennisPoseDetector] = None):
        self.detector = detector or TennisPoseDetector()

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Reads FPS, total frames, resolution, and duration from video."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        cap.release()
        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
        }

    def scan_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Pre-scans the entire video through MediaPipe Pose.
        Returns:
            - telemetry_df: Pandas DataFrame with per-frame biomechanical angles and coordinates.
            - landmarks_history: List of full 33-landmark coordinate records for full 3D CSV export.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        telemetry_records = []
        landmarks_history = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process_frame(frame_rgb)
            metrics = self.detector.extract_metrics(
                results,
                frame_width=width,
                frame_height=height,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )

            if metrics is not None:
                # Save primary telemetry record (clean scalar values for DF)
                clean_record = {k: v for k, v in metrics.items() if not k.startswith("_")}
                telemetry_records.append(clean_record)

                # Save raw 33-landmark snapshot
                raw_lms = metrics.get("_landmarks_raw")
                if raw_lms:
                    for lm_idx, lm in enumerate(raw_lms):
                        landmarks_history.append({
                            "frame": frame_idx,
                            "timestamp_sec": round(timestamp_sec, 3),
                            "landmark_index": lm_idx,
                            "landmark_name": TennisPoseDetector.LANDMARK_NAMES[lm_idx],
                            "x": round(lm.x, 5),
                            "y": round(lm.y, 5),
                            "z": round(lm.z, 5),
                            "visibility": round(lm.visibility, 3),
                        })
            else:
                # Null record for undetected frames to preserve time index
                telemetry_records.append({
                    "frame": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "dominant_hand": self.detector.dominant_hand,
                    "hitting_elbow_angle": np.nan,
                    "non_dominant_elbow_angle": np.nan,
                    "hitting_shoulder_angle": np.nan,
                    "non_dominant_shoulder_angle": np.nan,
                    "loading_knee_angle": np.nan,
                    "non_dominant_knee_angle": np.nan,
                    "torso_tilt_deg": np.nan,
                    "shoulder_tilt_deg": np.nan,
                    "hip_tilt_deg": np.nan,
                    "eye_level_tilt_deg": np.nan,
                    "head_y_norm": np.nan,
                    "dom_wrist_x": np.nan,
                    "dom_wrist_y": np.nan,
                    "non_dom_wrist_x": np.nan,
                    "non_dom_wrist_y": np.nan,
                })

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_callback(progress, f"Scanning frame {frame_idx}/{total_frames}...")

        cap.release()
        telemetry_df = pd.DataFrame(telemetry_records)
        return telemetry_df, landmarks_history

    def render_hud_frame(
        self,
        frame_bgr: np.ndarray,
        metrics: Optional[Dict[str, Any]],
        wrist_trails: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        toggles: Optional[Dict[str, bool]] = None,
        frame_idx: int = 0,
        fps: float = 30.0,
    ) -> np.ndarray:
        """
        Overlays the Pro Coaching HUD onto an individual video frame:
        - Segmented skeleton (upper body, lower body, head)
        - Angle arcs & degree callouts
        - Dual wrist motion trails (hitting vs non-dominant)
        - Translucent telemetry badge card
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Default toggles if not provided
        toggles = toggles or {
            "show_upper_body": True,
            "show_lower_body": True,
            "show_head": True,
            "show_angles": True,
            "show_wrist_trails": True,
            "show_hud_telemetry": True,
        }

        # 1. Render Dual Wrist Trails
        if toggles.get("show_wrist_trails") and wrist_trails:
            self._draw_wrist_trails(annotated, wrist_trails)

        if not metrics or "_coords_pixel" not in metrics:
            # Draw HUD card with 'No Pose Detected' indicator
            if toggles.get("show_hud_telemetry"):
                self._draw_hud_card(annotated, metrics, frame_idx, fps, detected=False)
            return annotated

        coords_px = metrics["_coords_pixel"]
        visibilities = metrics.get("_visibilities", {})

        head_style = toggles.get("head_style", "gaze_bar") if toggles.get("show_head", True) else "off"

        # 2. Draw Skeletal Connections
        if toggles.get("show_upper_body"):
            self._draw_connections(
                annotated, coords_px, visibilities, TennisPoseDetector.UPPER_BODY_CONNECTIONS
            )
        if toggles.get("show_lower_body"):
            self._draw_connections(
                annotated, coords_px, visibilities, TennisPoseDetector.LOWER_BODY_CONNECTIONS
            )

        # Draw Head / Face Overlay based on selected coaching style
        self._draw_head_overlay(annotated, coords_px, visibilities, head_style=head_style)

        # 3. Draw Body Landmark Joint Dots (excludes face 0..10 unless full_mesh is selected)
        for idx, pt in coords_px.items():
            if idx <= 10 and head_style != "full_mesh":
                continue
            vis = visibilities.get(idx, 1.0)
            if vis < 0.4:
                continue

            # Determine body region color
            if idx in (TennisPoseDetector.RIGHT_WRIST, TennisPoseDetector.RIGHT_ELBOW, TennisPoseDetector.RIGHT_SHOULDER, TennisPoseDetector.RIGHT_HIP, TennisPoseDetector.RIGHT_KNEE, TennisPoseDetector.RIGHT_ANKLE):
                color = self.COLOR_ORANGE
            elif idx in (TennisPoseDetector.LEFT_WRIST, TennisPoseDetector.LEFT_ELBOW, TennisPoseDetector.LEFT_SHOULDER, TennisPoseDetector.LEFT_HIP, TennisPoseDetector.LEFT_KNEE, TennisPoseDetector.LEFT_ANKLE):
                color = self.COLOR_CYAN
            else:
                color = self.COLOR_VOLT_YELLOW

            cv2.circle(annotated, (pt[0], pt[1]), 5, color, -1)
            cv2.circle(annotated, (pt[0], pt[1]), 7, (255, 255, 255), 1)

        # 4. Draw Joint Angle Callouts
        if toggles.get("show_angles"):
            self._draw_joint_angle_callouts(annotated, metrics)

        # 5. Draw Pro Coaching HUD Card
        if toggles.get("show_hud_telemetry"):
            self._draw_hud_card(annotated, metrics, frame_idx, fps, detected=True)

        return annotated

    def _draw_head_overlay(
        self,
        img: np.ndarray,
        coords_px: Dict[int, np.ndarray],
        visibilities: Dict[int, float],
        head_style: str = "gaze_bar",
    ):
        """Renders head and eye stability overlays with customizable coaching styles."""
        if head_style == "off":
            return

        nose = coords_px.get(TennisPoseDetector.NOSE)
        left_eye = coords_px.get(TennisPoseDetector.LEFT_EYE)
        right_eye = coords_px.get(TennisPoseDetector.RIGHT_EYE)
        left_ear = coords_px.get(TennisPoseDetector.LEFT_EAR)
        right_ear = coords_px.get(TennisPoseDetector.RIGHT_EAR)
        l_shoulder = coords_px.get(TennisPoseDetector.LEFT_SHOULDER)
        r_shoulder = coords_px.get(TennisPoseDetector.RIGHT_SHOULDER)

        # 1. Clean Gaze Stabilizer (Default - single clean head dot + horizontal gaze stabilizer line)
        if head_style == "gaze_bar":
            head_pts = [p for p in (nose, left_eye, right_eye, left_ear, right_ear) if p is not None]
            if not head_pts:
                return
            head_cx = int(np.mean([p[0] for p in head_pts]))
            head_cy = int(np.mean([p[1] for p in head_pts]))

            # Connect neck (mid-shoulder to head center)
            if l_shoulder is not None and r_shoulder is not None:
                neck_x = int((l_shoulder[0] + r_shoulder[0]) / 2)
                neck_y = int((l_shoulder[1] + r_shoulder[1]) / 2)
                cv2.line(img, (neck_x, neck_y), (head_cx, head_cy), self.COLOR_VOLT_YELLOW, 2, cv2.LINE_AA)

            # Draw Gaze Axis / Eye line
            if left_eye is not None and right_eye is not None:
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                p1 = (int(left_eye[0] - dx * 0.4), int(left_eye[1] - dy * 0.4))
                p2 = (int(right_eye[0] + dx * 0.4), int(right_eye[1] + dy * 0.4))
                cv2.line(img, p1, p2, (255, 230, 0), 2, cv2.LINE_AA)
                cv2.circle(img, tuple(left_eye), 3, (255, 255, 255), -1)
                cv2.circle(img, tuple(right_eye), 3, (255, 255, 255), -1)

            # Single clean head center marker
            cv2.circle(img, (head_cx, head_cy), 5, self.COLOR_VOLT_YELLOW, -1)
            cv2.circle(img, (head_cx, head_cy), 7, (255, 255, 255), 1)

        # 2. Minimal Eyeline Only
        elif head_style == "eyeline_only":
            if left_eye is not None and right_eye is not None:
                cv2.line(img, tuple(left_eye), tuple(right_eye), (255, 230, 0), 2, cv2.LINE_AA)
                cv2.circle(img, tuple(left_eye), 3, self.COLOR_CYAN, -1)
                cv2.circle(img, tuple(right_eye), 3, self.COLOR_ORANGE, -1)

        # 3. Minimal Head Halo / Circle
        elif head_style == "minimal_circle":
            head_pts = [p for p in (nose, left_eye, right_eye, left_ear, right_ear) if p is not None]
            if head_pts:
                head_cx = int(np.mean([p[0] for p in head_pts]))
                head_cy = int(np.mean([p[1] for p in head_pts]))
                cv2.circle(img, (head_cx, head_cy), 14, self.COLOR_VOLT_YELLOW, 2, cv2.LINE_AA)
                cv2.circle(img, (head_cx, head_cy), 3, (255, 255, 255), -1)

        # 4. Full 10-point Mesh (legacy)
        elif head_style == "full_mesh":
            self._draw_connections(img, coords_px, visibilities, TennisPoseDetector.HEAD_CONNECTIONS)
            for h_idx in range(11):
                if h_idx in coords_px and visibilities.get(h_idx, 1.0) > 0.4:
                    pt = tuple(coords_px[h_idx])
                    cv2.circle(img, pt, 4, self.COLOR_VOLT_YELLOW, -1)
                    cv2.circle(img, pt, 5, (255, 255, 255), 1)

    def _draw_connections(
        self,
        img: np.ndarray,
        coords_px: Dict[int, np.ndarray],
        visibilities: Dict[int, float],
        connections: List[Tuple[int, int]],
    ):
        """Draws skeletal line segments with antialiasing."""
        for start_idx, end_idx in connections:
            if start_idx in coords_px and end_idx in coords_px:
                vis1 = visibilities.get(start_idx, 1.0)
                vis2 = visibilities.get(end_idx, 1.0)
                if vis1 > 0.4 and vis2 > 0.4:
                    pt1 = tuple(coords_px[start_idx])
                    pt2 = tuple(coords_px[end_idx])

                    # Color lines based on lateral side
                    if start_idx in (TennisPoseDetector.RIGHT_ELBOW, TennisPoseDetector.RIGHT_WRIST, TennisPoseDetector.RIGHT_KNEE, TennisPoseDetector.RIGHT_ANKLE):
                        line_color = self.COLOR_ORANGE
                    elif start_idx in (TennisPoseDetector.LEFT_ELBOW, TennisPoseDetector.LEFT_WRIST, TennisPoseDetector.LEFT_KNEE, TennisPoseDetector.LEFT_ANKLE):
                        line_color = self.COLOR_CYAN
                    else:
                        line_color = self.COLOR_VOLT_YELLOW

                    cv2.line(img, pt1, pt2, line_color, 2, cv2.LINE_AA)

    @staticmethod
    def _smooth_trajectory(points: List[Tuple[int, int]], window_size: int = 5) -> List[Tuple[int, int]]:
        """
        Applies Gaussian kernel smoothing to reduce coordinate tracking jitter,
        followed by Catmull-Rom spline subdivision for a silky smooth swing trajectory.
        """
        if len(points) < 3:
            return points

        pts_arr = np.array(points, dtype=np.float32)
        n = len(pts_arr)
        smoothed = np.copy(pts_arr)

        # 1. Gaussian Weighted Smoothing
        half_w = min(window_size // 2, (n - 1) // 2)
        if half_w >= 1:
            sigma = 1.2
            k = np.exp(-0.5 * (np.arange(-half_w, half_w + 1) / sigma) ** 2)
            k /= k.sum()

            for dim in (0, 1):
                padded = np.pad(pts_arr[:, dim], (half_w, half_w), mode="edge")
                conv = np.convolve(padded, k, mode="valid")
                smoothed[:, dim] = conv[:n]

        # Anchor head point exactly at active detected position
        smoothed[-1] = pts_arr[-1]

        # 2. Catmull-Rom Spline Subdivision for silky continuous curve
        if n >= 4:
            dense_pts = []
            for i in range(len(smoothed) - 1):
                p0 = smoothed[max(0, i - 1)]
                p1 = smoothed[i]
                p2 = smoothed[i + 1]
                p3 = smoothed[min(len(smoothed) - 1, i + 2)]

                # Subdivide each segment into 3 sub-points
                for t_step in range(3):
                    t = t_step / 3.0
                    t2 = t * t
                    t3 = t2 * t
                    pos = 0.5 * (
                        (2 * p1)
                        + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                    )
                    dense_pts.append((int(round(pos[0])), int(round(pos[1]))))
            dense_pts.append((int(round(smoothed[-1][0])), int(round(smoothed[-1][1]))))
            return dense_pts

        return [(int(round(p[0])), int(round(p[1]))) for p in smoothed]

    def _draw_wrist_trails(
        self, img: np.ndarray, wrist_trails: Dict[str, List[Tuple[int, int]]]
    ):
        """
        Draws motion path ribbons with motion smoothing, spline curves,
        and explicit non-linear transparency fading (from 8% opacity at tail to 100% at head).
        """
        trails_config = [
            ("dominant", self.COLOR_VOLT_YELLOW, (255, 255, 200), True),
            ("non_dominant", self.COLOR_CYAN, (255, 255, 255), False),
        ]

        for key, color, glow_color, is_dom in trails_config:
            raw_pts = wrist_trails.get(key, [])
            if len(raw_pts) < 2:
                if len(raw_pts) == 1:
                    cv2.circle(img, raw_pts[0], 5, color, -1)
                    cv2.circle(img, raw_pts[0], 7, (255, 255, 255), 1)
                continue

            # Smooth trajectory
            smooth_pts = self._smooth_trajectory(raw_pts)
            n_pts = len(smooth_pts)
            if n_pts < 2:
                continue

            # 1. Subtle Outer Glow Under-Ribbon
            glow_overlay = img.copy()
            for i in range(1, n_pts):
                prog = i / n_pts
                thickness = max(2, int(2 + 6 * (prog ** 1.2)))
                cv2.line(glow_overlay, smooth_pts[i - 1], smooth_pts[i], color, thickness, cv2.LINE_AA)
            cv2.addWeighted(glow_overlay, 0.22, img, 0.78, 0, img)

            # 2. Main Core Ribbon with Explicit Gradient Transparency Fading
            num_batches = min(16, n_pts - 1)
            batch_size = max(1, (n_pts - 1) // num_batches)

            for b in range(num_batches):
                start_i = b * batch_size + 1
                end_i = min(n_pts, (b + 1) * batch_size + 1)
                if start_i >= n_pts:
                    break

                batch_prog = end_i / n_pts
                # Explicit non-linear alpha curve: 8% at tail up to 98% at head
                alpha = max(0.08, min(0.98, (batch_prog ** 1.3) * 0.90 + 0.08))
                thickness = max(1, int(1 + (4 if is_dom else 3) * batch_prog))

                seg_overlay = img.copy()
                for i in range(start_i, end_i):
                    cv2.line(seg_overlay, smooth_pts[i - 1], smooth_pts[i], color, thickness, cv2.LINE_AA)
                    # Add subtle trail dots along the trajectory
                    if i % max(1, (n_pts // 10)) == 0 and i < n_pts - 1:
                        dot_rad = max(2, int(3.5 * batch_prog))
                        cv2.circle(seg_overlay, smooth_pts[i], dot_rad, color, -1)

                cv2.addWeighted(seg_overlay, alpha, img, 1.0 - alpha, 0, img)

            # 3. Bright Glowing Head at Current Wrist Position
            head_pt = smooth_pts[-1]
            cv2.circle(img, head_pt, 6 if is_dom else 5, color, -1)
            cv2.circle(img, head_pt, 8 if is_dom else 7, glow_color, 2, cv2.LINE_AA)
            cv2.circle(img, head_pt, 2, (255, 255, 255), -1)

    def _draw_joint_angle_callouts(self, img: np.ndarray, metrics: Dict[str, Any]):
        """Draws high-visibility angle labels directly at elbow and knee vertices."""
        coords_px = metrics["_coords_pixel"]

        callouts = [
            (
                "Elbow",
                coords_px.get(TennisPoseDetector.RIGHT_ELBOW),
                metrics.get("hitting_elbow_angle" if metrics["dominant_hand"] == "Right" else "non_dominant_elbow_angle"),
                self.COLOR_ORANGE,
            ),
            (
                "Elbow",
                coords_px.get(TennisPoseDetector.LEFT_ELBOW),
                metrics.get("non_dominant_elbow_angle" if metrics["dominant_hand"] == "Right" else "hitting_elbow_angle"),
                self.COLOR_CYAN,
            ),
            (
                "Knee",
                coords_px.get(TennisPoseDetector.RIGHT_KNEE),
                metrics.get("loading_knee_angle" if metrics["dominant_hand"] == "Right" else "non_dominant_knee_angle"),
                self.COLOR_ORANGE,
            ),
            (
                "Knee",
                coords_px.get(TennisPoseDetector.LEFT_KNEE),
                metrics.get("non_dominant_knee_angle" if metrics["dominant_hand"] == "Right" else "loading_knee_angle"),
                self.COLOR_CYAN,
            ),
        ]

        is_bend = metrics.get("angle_mode", "bend") == "bend"
        suffix = "° bend" if is_bend else "°"

        for name, pt, angle_val, color in callouts:
            if pt is None or angle_val is None or np.isnan(angle_val):
                continue

            text = f"{int(round(angle_val))}{suffix}"
            tx, ty = int(pt[0] + 12), int(pt[1] - 8)

            # Draw background badge for text readability
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 1), self.COLOR_BG_CARD, -1)
            cv2.rectangle(img, (tx - 3, ty - th - 3), (tx + tw + 3, ty + baseline + 1), color, 1)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_hud_card(
        self,
        img: np.ndarray,
        metrics: Optional[Dict[str, Any]],
        frame_idx: int,
        fps: float,
        detected: bool,
    ):
        """Draws a sleek translucent HUD card at the top-left of the viewport."""
        card_w = 260
        card_h = 135
        card_x = 18
        card_y = 18

        # Create overlay for translucent effect
        overlay = img.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_BG_CARD, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_BG_BORDER, 1)

        # Header: Tennis Biomechanics HUD
        time_sec = frame_idx / fps if fps > 0 else 0.0
        cv2.putText(img, "TENNIS TECHNIQUE HUD", (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_VOLT_YELLOW, 1, cv2.LINE_AA)
        cv2.putText(img, f"F:{frame_idx:04d} | {time_sec:05.2f}s", (card_x + 155, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.line(img, (card_x + 10, card_y + 30), (card_x + card_w - 10, card_y + 30), self.COLOR_BG_BORDER, 1)

        if not detected or metrics is None:
            cv2.putText(img, "Pose: Searching...", (card_x + 12, card_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_ORANGE, 1, cv2.LINE_AA)
            return

        # Metrics rows
        is_bend = metrics.get("angle_mode", "bend") == "bend"
        hit_elbow = metrics.get("hitting_elbow_angle")
        load_knee = metrics.get("loading_knee_angle")
        torso = metrics.get("torso_tilt_deg")
        dom_hand = metrics.get("dominant_hand", "Right")

        elbow_str = f"{hit_elbow:.1f}°" if hit_elbow is not None and not np.isnan(hit_elbow) else "--"
        knee_str = f"{load_knee:.1f}°" if load_knee is not None and not np.isnan(load_knee) else "--"
        torso_str = f"{torso:.1f}°" if torso is not None and not np.isnan(torso) else "--"

        row1_y = card_y + 52
        row2_y = card_y + 76
        row3_y = card_y + 100
        row4_y = card_y + 122

        # Row 1: Hitting Arm Elbow Bend
        row1_label = f"Elbow Bend ({dom_hand[0]}):" if is_bend else f"Hit Elbow ({dom_hand[0]}):"
        cv2.putText(img, row1_label, (card_x + 12, row1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, elbow_str, (card_x + 175, row1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_VOLT_YELLOW, 1, cv2.LINE_AA)

        # Row 2: Loading Knee Bend
        row2_label = "Knee Bend:" if is_bend else "Loading Knee:"
        cv2.putText(img, row2_label, (card_x + 12, row2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, knee_str, (card_x + 175, row2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)

        # Row 3: Torso Tilt
        cv2.putText(img, "Torso Tilt:", (card_x + 12, row3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, torso_str, (card_x + 175, row3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_CYAN, 1, cv2.LINE_AA)

        # Row 4: Wrist Trail Legend
        cv2.circle(img, (card_x + 16, row4_y - 4), 4, self.COLOR_VOLT_YELLOW, -1)
        cv2.putText(img, "Hit Wrist", (card_x + 26, row4_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.circle(img, (card_x + 120, row4_y - 4), 4, self.COLOR_CYAN, -1)
        cv2.putText(img, "Off Wrist", (card_x + 130, row4_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def export_annotated_video(
        self,
        video_path: str,
        output_path: str,
        speed_multiplier: float = 1.0,
        toggles: Optional[Dict[str, bool]] = None,
        trail_length: int = 25,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """
        Renders annotated video with full skeletal overlays and HUD to output_path.
        Supports playback speeds (0.25x, 0.5x, 1.0x, 2.0x).
        Uses imageio-ffmpeg for browser-compatible H.264 MP4 encoding.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine target export FPS
        # Adjusting the output FPS creates true slow-mo or speed-up playback in any player
        export_fps = max(5.0, fps * speed_multiplier)

        # Prepare writer: use imageio-ffmpeg if available, else OpenCV VideoWriter
        writer = None
        use_imageio = imageio is not None

        if use_imageio:
            try:
                writer = imageio.get_writer(
                    output_path,
                    fps=export_fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                )
            except Exception:
                writer = None

        if writer is None:
            # Fallback to OpenCV
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, export_fps, (width, height))

        wrist_trails = {"dominant": [], "non_dominant": []}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process_frame(frame_rgb)
            metrics = self.detector.extract_metrics(
                results,
                frame_width=width,
                frame_height=height,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
            )

            # Update motion trails
            if metrics:
                dom_x, dom_y = metrics["dom_wrist_x"], metrics["dom_wrist_y"]
                non_dom_x, non_dom_y = metrics["non_dom_wrist_x"], metrics["non_dom_wrist_y"]
                wrist_trails["dominant"].append((dom_x, dom_y))
                wrist_trails["non_dominant"].append((non_dom_x, non_dom_y))
                if len(wrist_trails["dominant"]) > trail_length:
                    wrist_trails["dominant"].pop(0)
                if len(wrist_trails["non_dominant"]) > trail_length:
                    wrist_trails["non_dominant"].pop(0)

            annotated_frame = self.render_hud_frame(
                frame,
                metrics,
                wrist_trails=wrist_trails,
                toggles=toggles,
                frame_idx=frame_idx,
                fps=fps,
            )

            if use_imageio and hasattr(writer, "append_data"):
                frame_rgb_out = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb_out)
            else:
                writer.write(annotated_frame)

            frame_idx += 1
            if progress_callback and total_frames > 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_callback(progress, f"Rendering annotated video: {frame_idx}/{total_frames}...")

        cap.release()
        if writer is not None:
            if hasattr(writer, "close"):
                writer.close()
            elif hasattr(writer, "release"):
                writer.release()

        return output_path

    @staticmethod
    def export_telemetry_csv(telemetry_df: pd.DataFrame, output_csv_path: str) -> str:
        """Exports primary tennis angle & kinematics telemetry DataFrame to CSV."""
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        telemetry_df.to_csv(output_csv_path, index=False)
        return output_csv_path

    @staticmethod
    def export_landmarks_csv(landmarks_history: List[Dict[str, Any]], output_csv_path: str) -> str:
        """Exports complete 33-landmark coordinate records to CSV."""
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df = pd.DataFrame(landmarks_history)
        df.to_csv(output_csv_path, index=False)
        return output_csv_path

    # ------------------------------------------------------------------
    # 3D Pose Lab (Three.js viewer) export
    # ------------------------------------------------------------------
    LANDMARKS3D_SCHEMA = "tennis-mediapipe/landmarks3d@1"
    NUM_LANDMARKS = 33

    @staticmethod
    def build_landmarks3d_payload(
        landmarks_history: List[Dict[str, Any]],
        fps: float,
        frame_count: int,
        source_name: str,
        dominant_hand: str = "Right",
        synthetic: bool = False,
    ) -> Dict[str, Any]:
        """
        Pivots the per-landmark scan history into the per-frame
        "tennis-mediapipe/landmarks3d@1" payload consumed by the Three.js
        3D viewer (viewer/index.html). Frames without a full 33-landmark
        set carry landmarks=None so the viewer shows its "no pose" state.
        """
        from datetime import datetime, timezone

        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for rec in landmarks_history:
            by_frame.setdefault(rec["frame"], []).append(rec)

        frames = []
        for frame_idx in range(frame_count):
            recs = by_frame.get(frame_idx, [])
            if len(recs) == TennisVideoProcessor.NUM_LANDMARKS:
                recs.sort(key=lambda r: r["landmark_index"])
                frames.append(
                    {
                        "frame": frame_idx,
                        "t": round(recs[0]["timestamp_sec"], 3),
                        "detected": True,
                        "landmarks": [[r["x"], r["y"], r["z"]] for r in recs],
                        "visibility": [r["visibility"] for r in recs],
                    }
                )
            else:
                frames.append(
                    {
                        "frame": frame_idx,
                        "t": round(frame_idx / fps, 3) if fps else 0.0,
                        "detected": False,
                        "landmarks": None,
                        "visibility": None,
                    }
                )

        return {
            "meta": {
                "schema": TennisVideoProcessor.LANDMARKS3D_SCHEMA,
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": round(frame_count / fps, 3) if fps else 0.0,
                "source": source_name,
                "synthetic": synthetic,
                "dominant_hand": dominant_hand,
                "coordinate_system": "mediapipe_normalized",
                "generated": datetime.now(timezone.utc).isoformat(),
                "notes": (
                    "x: 0..1 left-to-right, y: 0..1 top-to-bottom, "
                    "z: depth (~0 at mid-hip, negative toward camera). "
                    "Load this file in viewer/index.html (3D Pose Lab)."
                ),
            },
            "landmark_names": TennisPoseDetector.LANDMARK_NAMES,
            "connections": {
                "upper": [list(c) for c in TennisPoseDetector.UPPER_BODY_CONNECTIONS],
                "lower": [list(c) for c in TennisPoseDetector.LOWER_BODY_CONNECTIONS],
                "head": [list(c) for c in TennisPoseDetector.HEAD_CONNECTIONS],
            },
            "frames": frames,
        }

    @staticmethod
    def export_landmarks_json(
        landmarks_history: List[Dict[str, Any]],
        output_json_path: str,
        fps: float,
        frame_count: int,
        source_name: str,
        dominant_hand: str = "Right",
        synthetic: bool = False,
    ) -> str:
        """Writes the landmarks3d@1 JSON payload for the 3D Pose Lab viewer."""
        import json

        payload = TennisVideoProcessor.build_landmarks3d_payload(
            landmarks_history, fps, frame_count, source_name, dominant_hand, synthetic
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(payload, f)
        return output_json_path
