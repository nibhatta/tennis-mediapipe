"""
Tennis Pose Detector & Biomechanical Telemetry Engine.
Uses Google MediaPipe Pose Detection to extract 33 3D skeletal landmarks
and calculate key tennis technique metrics: joint angles, torso tilt,
shoulder rotation, head stability, and handedness-specific kinematics.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None


class TennisPoseDetector:
    """
    Analyzes body pose from video frames using MediaPipe Pose.
    Computes key joint angles, posture metrics, and lateral kinematic data.
    """

    # MediaPipe Landmark Index Constants
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Groupings for UI overlay toggling
    UPPER_BODY_CONNECTIONS = [
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (LEFT_WRIST, LEFT_PINKY),
        (LEFT_WRIST, LEFT_INDEX),
        (LEFT_WRIST, LEFT_THUMB),
        (LEFT_PINKY, LEFT_INDEX),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (RIGHT_WRIST, RIGHT_PINKY),
        (RIGHT_WRIST, RIGHT_INDEX),
        (RIGHT_WRIST, RIGHT_THUMB),
        (RIGHT_PINKY, RIGHT_INDEX),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
    ]

    LOWER_BODY_CONNECTIONS = [
        (LEFT_HIP, LEFT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (LEFT_ANKLE, LEFT_HEEL),
        (LEFT_HEEL, LEFT_FOOT_INDEX),
        (LEFT_ANKLE, LEFT_FOOT_INDEX),
        (RIGHT_HIP, RIGHT_KNEE),
        (RIGHT_KNEE, RIGHT_ANKLE),
        (RIGHT_ANKLE, RIGHT_HEEL),
        (RIGHT_HEEL, RIGHT_FOOT_INDEX),
        (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
    ]

    HEAD_CONNECTIONS = [
        (NOSE, LEFT_EYE_INNER),
        (LEFT_EYE_INNER, LEFT_EYE),
        (LEFT_EYE, LEFT_EYE_OUTER),
        (LEFT_EYE_OUTER, LEFT_EAR),
        (NOSE, RIGHT_EYE_INNER),
        (RIGHT_EYE_INNER, RIGHT_EYE),
        (RIGHT_EYE, RIGHT_EYE_OUTER),
        (RIGHT_EYE_OUTER, RIGHT_EAR),
        (MOUTH_LEFT, MOUTH_RIGHT),
    ]

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        dominant_hand: str = "Right",
        angle_mode: str = "bend",
    ):
        """
        Initialize the MediaPipe Pose estimation pipeline.

        Args:
            min_detection_confidence: Confidence threshold for initial pose detection.
            min_tracking_confidence: Confidence threshold for landmark tracking.
            model_complexity: 0 (Fastest), 1 (Balanced), 2 (Most accurate).
            dominant_hand: 'Right' or 'Left'.
            angle_mode: 'bend' (0°=straight, 90°=L-bend) or 'interior' (180°=straight).
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.dominant_hand = dominant_hand.capitalize()
        self.angle_mode = angle_mode.lower()

        self._pose = None
        self._pose_static = None
        if mp is not None:
            self.mp_pose = mp.solutions.pose
            self._init_mediapipe()

    def _init_mediapipe(self):
        """Instantiate MediaPipe Pose instance."""
        if hasattr(self, "mp_pose"):
            self._pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def set_dominant_hand(self, dominant_hand: str):
        """Update dominant hand (Right or Left)."""
        self.dominant_hand = dominant_hand.capitalize()

    def set_angle_mode(self, angle_mode: str):
        """Update angle mode ('bend' for 0°=straight, or 'interior' for 180°=straight)."""
        self.angle_mode = angle_mode.lower()

    @staticmethod
    def calculate_bend_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculates amount of bend (flexion angle from straight) in degrees.
        0° = limb locked straight (180° interior angle).
        90° = right angle / L-shape bend.
        180° = fully folded / closed.
        """
        interior = TennisPoseDetector.calculate_angle(a, b, c)
        return round(max(0.0, 180.0 - interior), 2)

    @staticmethod
    def calculate_angle(
        a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        """
        Calculates the 2D/3D planar angle at vertex b between segments ba and bc in degrees.
        Uses vector dot product: cos(theta) = (ba . bc) / (|ba| * |bc|).

        Args:
            a: First point (e.g. shoulder) [x, y] or [x, y, z]
            b: Vertex point (e.g. elbow) [x, y] or [x, y, z]
            c: Third point (e.g. wrist) [x, y] or [x, y, z]

        Returns:
            Angle in degrees [0.0, 180.0].
        """
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        c = np.array(c, dtype=np.float64)

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-7 or norm_bc < 1e-7:
            return 0.0

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cosine)))
        return round(angle_deg, 2)

    @staticmethod
    def calculate_line_angle_vertical(p_bottom: np.ndarray, p_top: np.ndarray) -> float:
        """
        Calculates tilt angle in degrees of vector (p_top - p_bottom) relative to vertical upright axis.
        Upright = 0 degrees.
        """
        p_bottom = np.array(p_bottom[:2], dtype=np.float64)
        p_top = np.array(p_top[:2], dtype=np.float64)
        v = p_top - p_bottom  # In image coords, top has smaller y

        # Vertical upright in image coords points straight up: [0, -1]
        upright = np.array([0.0, -1.0])
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-7:
            return 0.0

        cosine = np.dot(v, upright) / norm_v
        cosine = np.clip(cosine, -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cosine))), 2)

    @staticmethod
    def calculate_line_angle_horizontal(p_left: np.ndarray, p_right: np.ndarray) -> float:
        """
        Calculates tilt angle in degrees of segment between p_left and p_right relative to horizontal.
        Horizontal = 0 degrees.
        """
        dx = float(p_right[0] - p_left[0])
        dy = float(p_right[1] - p_left[1])
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        return round(angle_deg, 2)

    def process_frame(self, frame_rgb: np.ndarray, static_mode: bool = False) -> Optional[Any]:
        """
        Runs MediaPipe Pose inference on an RGB frame.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3)
            static_mode: If True, uses static image mode for reliable random-access frame inspection.

        Returns:
            MediaPipe Pose results object or None.
        """
        if static_mode:
            if self._pose_static is None and hasattr(self, "mp_pose") and self.mp_pose is not None:
                self._pose_static = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=self.model_complexity,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
            return self._pose_static.process(frame_rgb) if self._pose_static else None
        else:
            if self._pose is None:
                self._init_mediapipe()
            if self._pose is None:
                return None
            return self._pose.process(frame_rgb)

    def extract_metrics(
        self,
        results: Any,
        frame_width: int,
        frame_height: int,
        frame_idx: int = 0,
        timestamp_sec: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts all 33 landmarks and computes tennis-specific kinematics.

        Returns a dictionary with both high-level tennis telemetry and raw coordinates,
        or None if no pose is detected.
        """
        if not results or not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        coords_2d = {}
        coords_3d = {}
        coords_pixel = {}
        visibilities = {}

        for idx, lm in enumerate(landmarks):
            coords_2d[idx] = np.array([lm.x, lm.y])
            coords_3d[idx] = np.array([lm.x, lm.y, lm.z])
            coords_pixel[idx] = np.array([int(lm.x * frame_width), int(lm.y * frame_height)])
            visibilities[idx] = lm.visibility

        # 1. Elbow angles (Shoulder -> Elbow -> Wrist)
        r_elbow_int = self.calculate_angle(
            coords_3d[self.RIGHT_SHOULDER],
            coords_3d[self.RIGHT_ELBOW],
            coords_3d[self.RIGHT_WRIST],
        )
        l_elbow_int = self.calculate_angle(
            coords_3d[self.LEFT_SHOULDER],
            coords_3d[self.LEFT_ELBOW],
            coords_3d[self.LEFT_WRIST],
        )

        # 2. Shoulder elevation/abduction angles (Hip -> Shoulder -> Elbow)
        right_shoulder_angle = self.calculate_angle(
            coords_3d[self.RIGHT_HIP],
            coords_3d[self.RIGHT_SHOULDER],
            coords_3d[self.RIGHT_ELBOW],
        )
        left_shoulder_angle = self.calculate_angle(
            coords_3d[self.LEFT_HIP],
            coords_3d[self.LEFT_SHOULDER],
            coords_3d[self.LEFT_ELBOW],
        )

        # 3. Knee bend angles (Hip -> Knee -> Ankle)
        r_knee_int = self.calculate_angle(
            coords_3d[self.RIGHT_HIP],
            coords_3d[self.RIGHT_KNEE],
            coords_3d[self.RIGHT_ANKLE],
        )
        l_knee_int = self.calculate_angle(
            coords_3d[self.LEFT_HIP],
            coords_3d[self.LEFT_KNEE],
            coords_3d[self.LEFT_ANKLE],
        )

        # Amount of Bend (0° = locked straight, 90° = square bend)
        r_elbow_bend = round(max(0.0, 180.0 - r_elbow_int), 2)
        l_elbow_bend = round(max(0.0, 180.0 - l_elbow_int), 2)
        r_knee_bend = round(max(0.0, 180.0 - r_knee_int), 2)
        l_knee_bend = round(max(0.0, 180.0 - l_knee_int), 2)

        # Mode selection
        current_mode = getattr(self, "angle_mode", "bend")
        if current_mode == "bend":
            right_elbow_angle = r_elbow_bend
            left_elbow_angle = l_elbow_bend
            right_knee_angle = r_knee_bend
            left_knee_angle = l_knee_bend
        else:
            right_elbow_angle = r_elbow_int
            left_elbow_angle = l_elbow_int
            right_knee_angle = r_knee_int
            left_knee_angle = l_knee_int

        # 4. Torso tilt (Mid-hip to Mid-shoulder axis vs vertical)
        mid_hip = (coords_2d[self.LEFT_HIP] + coords_2d[self.RIGHT_HIP]) / 2.0
        mid_shoulder = (coords_2d[self.LEFT_SHOULDER] + coords_2d[self.RIGHT_SHOULDER]) / 2.0
        torso_tilt_deg = self.calculate_line_angle_vertical(mid_hip, mid_shoulder)

        # 5. Shoulder line tilt & Hip rotation/tilt
        shoulder_tilt_deg = self.calculate_line_angle_horizontal(
            coords_2d[self.LEFT_SHOULDER], coords_2d[self.RIGHT_SHOULDER]
        )
        hip_tilt_deg = self.calculate_line_angle_horizontal(
            coords_2d[self.LEFT_HIP], coords_2d[self.RIGHT_HIP]
        )

        # 6. Head stability: Eye level tilt & Head vertical coordinate
        eye_level_tilt_deg = self.calculate_line_angle_horizontal(
            coords_2d[self.LEFT_EYE], coords_2d[self.RIGHT_EYE]
        )
        head_y_norm = float(coords_2d[self.NOSE][1])

        # 7. Handedness mapping (Dominant vs Non-Dominant)
        is_right_handed = self.dominant_hand == "Right"
        if is_right_handed:
            dominant_wrist_idx = self.RIGHT_WRIST
            non_dominant_wrist_idx = self.LEFT_WRIST
            hitting_elbow_angle = right_elbow_angle
            non_dominant_elbow_angle = left_elbow_angle
            hitting_shoulder_angle = right_shoulder_angle
            non_dominant_shoulder_angle = left_shoulder_angle
            loading_knee_angle = right_knee_angle
            non_dominant_knee_angle = left_knee_angle
            hitting_elbow_bend = r_elbow_bend
            non_dominant_elbow_bend = l_elbow_bend
            loading_knee_bend = r_knee_bend
            non_dominant_knee_bend = l_knee_bend
        else:
            dominant_wrist_idx = self.LEFT_WRIST
            non_dominant_wrist_idx = self.RIGHT_WRIST
            hitting_elbow_angle = left_elbow_angle
            non_dominant_elbow_angle = right_elbow_angle
            hitting_shoulder_angle = left_shoulder_angle
            non_dominant_shoulder_angle = right_shoulder_angle
            loading_knee_angle = left_knee_angle
            non_dominant_knee_angle = right_knee_angle
            hitting_elbow_bend = l_elbow_bend
            non_dominant_elbow_bend = r_elbow_bend
            loading_knee_bend = l_knee_bend
            non_dominant_knee_bend = r_knee_bend

        dominant_wrist_px = coords_pixel[dominant_wrist_idx]
        non_dominant_wrist_px = coords_pixel[non_dominant_wrist_idx]

        metrics = {
            "frame": frame_idx,
            "timestamp_sec": round(timestamp_sec, 3),
            "dominant_hand": self.dominant_hand,
            "angle_mode": current_mode,
            # Primary Hitting / Stroke Telemetry
            "hitting_elbow_angle": hitting_elbow_angle,
            "non_dominant_elbow_angle": non_dominant_elbow_angle,
            "hitting_shoulder_angle": hitting_shoulder_angle,
            "non_dominant_shoulder_angle": non_dominant_shoulder_angle,
            "loading_knee_angle": loading_knee_angle,
            "non_dominant_knee_angle": non_dominant_knee_angle,
            "hitting_elbow_bend": hitting_elbow_bend,
            "non_dominant_elbow_bend": non_dominant_elbow_bend,
            "loading_knee_bend": loading_knee_bend,
            "non_dominant_knee_bend": non_dominant_knee_bend,
            # Torso & Head Stability
            "torso_tilt_deg": torso_tilt_deg,
            "shoulder_tilt_deg": shoulder_tilt_deg,
            "hip_tilt_deg": hip_tilt_deg,
            "eye_level_tilt_deg": eye_level_tilt_deg,
            "head_y_norm": round(head_y_norm, 4),
            # Wrist positions (pixel)
            "dom_wrist_x": int(dominant_wrist_px[0]),
            "dom_wrist_y": int(dominant_wrist_px[1]),
            "non_dom_wrist_x": int(non_dominant_wrist_px[0]),
            "non_dom_wrist_y": int(non_dominant_wrist_px[1]),
            # Internal coordinates for HUD renderer
            "_coords_pixel": coords_pixel,
            "_coords_3d": coords_3d,
            "_visibilities": visibilities,
            "_landmarks_raw": landmarks,
        }

        return metrics

    def close(self):
        """Release MediaPipe Pose resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
