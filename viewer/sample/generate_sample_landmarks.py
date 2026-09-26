#!/usr/bin/env python3
"""
Generate SYNTHETIC demo landmark data for the Three.js 3D viewer.

This does NOT come from a real video or from MediaPipe. It animates a simple
forward-kinematics skeleton through a forehand-style swing (ready -> backswing
-> contact -> follow-through -> recover) so the viewer can be explored
immediately, before running export_3d_json.py on a real recording.

Output schema is identical to export_3d_json.py ("tennis-mediapipe/landmarks3d@1")
with meta.synthetic = true and an explicit note.

Usage:
    python generate_sample_landmarks.py -o sample_landmarks.json
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone

# Mirrors TennisPoseDetector (detector.py) so this script runs without
# OpenCV/MediaPipe installed. Keep in sync with the repo's connection sets.
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
UPPER_BODY_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
]
LOWER_BODY_CONNECTIONS = [
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
HEAD_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]

FPS = 30
N_FRAMES = 90

# Body proportions (meters, y-up world; person faces +z, i.e. the camera)
SHOULDER_W = 0.21   # half-width
HIP_W = 0.11        # half-width
UPPER_ARM = 0.30
FOREARM = 0.28
SPINE = 0.50        # mid-hip -> mid-shoulder
HEAD_ABOVE = 0.24   # mid-shoulder -> nose
HIP_Y = 0.98
ANKLE_Y = 0.10

# Swing keyframes: (crouch, yaw, armAz, armEl, elbowBend, lArmAz, lArmEl, sway, lean)
# yaw > 0 coils the shoulders (right shoulder back); armAz sweeps the hitting arm.
KEYS = [
    # ready
    dict(crouch=0.35, yaw=0.15, armAz=-0.35, armEl=-0.55, elbowBend=0.55,
         lArmAz=0.35, lArmEl=-0.45, sway=0.00, lean=0.06),
    # backswing (coil)
    dict(crouch=0.60, yaw=0.95, armAz=-1.50, armEl=0.10, elbowBend=0.75,
         lArmAz=0.95, lArmEl=-0.10, sway=-0.05, lean=0.10),
    # contact
    dict(crouch=0.28, yaw=-0.30, armAz=0.95, armEl=-0.05, elbowBend=0.12,
         lArmAz=-0.55, lArmEl=-0.35, sway=0.06, lean=0.04),
    # follow-through
    dict(crouch=0.38, yaw=-0.95, armAz=2.00, armEl=0.35, elbowBend=0.95,
         lArmAz=-0.95, lArmEl=-0.25, sway=0.03, lean=0.05),
    # recover (= ready)
    dict(crouch=0.35, yaw=0.15, armAz=-0.35, armEl=-0.55, elbowBend=0.55,
         lArmAz=0.35, lArmEl=-0.45, sway=0.00, lean=0.06),
]
# Frame boundaries for each keyframe (inclusive start of each pose).
# 5 keyframes -> 4 segments: coil up, swing to contact, follow through, recover.
BOUNDS = [0, 20, 45, 58, N_FRAMES]


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def pose_at(frame: int) -> dict:
    """Interpolate keyframe parameters for a frame index."""
    for k in range(len(BOUNDS) - 1):
        if BOUNDS[k] <= frame < BOUNDS[k + 1]:
            a, b = KEYS[k], KEYS[k + 1]
            t = smoothstep((frame - BOUNDS[k]) / (BOUNDS[k + 1] - BOUNDS[k]))
            return {key: a[key] + (b[key] - a[key]) * t for key in a}
    return dict(KEYS[-1])


def yaw_rot(x: float, z: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return x * c + z * s, -x * s + z * c


def arm_dir(az: float, el: float):
    """Unit vector for upper-arm direction. az=0 -> forward (+z), el>0 -> raised."""
    return (
        math.sin(az) * math.cos(el),
        math.sin(el),
        math.cos(az) * math.cos(el),
    )


def skeleton(p: dict):
    """Return dict of 33 MediaPipe-indexed (x, y, z) points in y-up meters."""
    crouch, yaw = p["crouch"], p["yaw"]
    hip_y = HIP_Y - crouch * 0.18
    sh_y = hip_y + SPINE
    cx = p["sway"]

    pts = {}

    def torso_point(dx, dy, dz=0.0):
        rx, rz = yaw_rot(dx, dz, yaw)
        return (cx + rx, hip_y + dy, rz)

    # Core
    mid_hip = (cx, hip_y, 0.0)
    mid_sh = torso_point(p["lean"] * 0.4, SPINE)
    sh_l = torso_point(-SHOULDER_W, SPINE)
    sh_r = torso_point(SHOULDER_W, SPINE)
    hip_l = (cx - HIP_W * 0.95, hip_y, 0.0)
    hip_r = (cx + HIP_W * 0.95, hip_y, 0.0)

    pts[23], pts[24] = hip_l, hip_r
    pts[11], pts[12] = sh_l, sh_r

    # Arms (right = hitting arm)
    for side, sh, az, el, bend, idx_base in (
        ("R", sh_r, p["armAz"], p["armEl"], p["elbowBend"], (14, 16, 18, 20, 22)),
        ("L", sh_l, p["lArmAz"], p["lArmEl"], 0.45, (13, 15, 17, 19, 21)),
    ):
        d = arm_dir(az, el)
        elbow = (sh[0] + d[0] * UPPER_ARM, sh[1] + d[1] * UPPER_ARM, sh[2] + d[2] * UPPER_ARM)
        # forearm folds downward/inward with elbow bend
        d2 = (d[0] * (1 - bend * 0.55), d[1] - bend * 0.85, d[2] * (1 - bend * 0.55))
        n = math.sqrt(sum(v * v for v in d2)) or 1.0
        d2 = tuple(v / n for v in d2)
        wrist = (elbow[0] + d2[0] * FOREARM, elbow[1] + d2[1] * FOREARM, elbow[2] + d2[2] * FOREARM)
        e_idx, w_idx, pinky, index, thumb = idx_base
        pts[e_idx] = elbow
        pts[w_idx] = wrist
        pts[pinky] = (wrist[0] + d2[0] * 0.09 - 0.015, wrist[1] + d2[1] * 0.09, wrist[2] + d2[2] * 0.09)
        pts[index] = (wrist[0] + d2[0] * 0.10 + 0.015, wrist[1] + d2[1] * 0.10, wrist[2] + d2[2] * 0.10)
        pts[thumb] = (wrist[0] + d2[0] * 0.06, wrist[1] + d2[1] * 0.06 + 0.02, wrist[2] + d2[2] * 0.06 + 0.015)

    # Legs (feet planted, knees push forward with crouch)
    for hip, k_idx, a_idx, heel_idx, foot_idx in (
        (hip_l, 25, 27, 29, 31),
        (hip_r, 26, 28, 30, 32),
    ):
        ax = hip[0] * 0.9 + (0.13 if hip[0] < cx else -0.13) * 0.0
        # keep it simple: ankles near fixed stance positions
        ankle = (hip[0] + (0.02 if hip[0] < cx else -0.02), ANKLE_Y, 0.01)
        knee = (
            (hip[0] + ankle[0]) / 2 + 0.015 + crouch * 0.07,
            (hip[1] + ankle[1]) / 2 - 0.02,
            (hip[2] + ankle[2]) / 2 + 0.05 + crouch * 0.07,
        )
        pts[k_idx] = knee
        pts[a_idx] = ankle
        pts[heel_idx] = (ankle[0], ankle[1] - 0.03, ankle[2] - 0.06)
        pts[foot_idx] = (ankle[0], ankle[1] - 0.075, ankle[2] + 0.10)

    # Head (nose/eyes/ears/mouth relative to nose)
    nose = (mid_sh[0] + p["lean"] * 0.15, mid_sh[1] + HEAD_ABOVE, mid_sh[2] + 0.03)
    pts[0] = nose
    pts[1] = (nose[0] - 0.025, nose[1] + 0.025, nose[2] + 0.012)
    pts[2] = (nose[0] - 0.045, nose[1] + 0.03, nose[2] + 0.008)
    pts[3] = (nose[0] - 0.065, nose[1] + 0.028, nose[2])
    pts[4] = (nose[0] + 0.025, nose[1] + 0.025, nose[2] + 0.012)
    pts[5] = (nose[0] + 0.045, nose[1] + 0.03, nose[2] + 0.008)
    pts[6] = (nose[0] + 0.065, nose[1] + 0.028, nose[2])
    pts[7] = (nose[0] - 0.075, nose[1] + 0.01, nose[2] - 0.01)
    pts[8] = (nose[0] + 0.075, nose[1] + 0.01, nose[2] - 0.01)
    pts[9] = (nose[0] - 0.03, nose[1] - 0.035, nose[2] + 0.008)
    pts[10] = (nose[0] + 0.03, nose[1] - 0.035, nose[2] + 0.008)

    assert len(pts) == 33, f"expected 33 landmarks, got {len(pts)}"
    return pts


# MediaPipe normalized framing: view ~3.4 m wide, centered at (0, 0.9 m)
VIEW_W = 3.4
CENTER_Y = 0.9


def to_mediapipe(pt):
    x, y, z = pt
    return [round(0.5 + x / VIEW_W, 5), round(0.5 - (y - CENTER_Y) / VIEW_W, 5), round(-z / VIEW_W, 5)]


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic demo landmarks.")
    ap.add_argument("-o", "--output", default="sample_landmarks.json")
    ap.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="json: landmarks3d@1 payload for the viewer; "
             "csv: per-landmark rows (frame,timestamp_sec,landmark_index,"
             "landmark_name,x,y,z,visibility), same layout as the repo's "
             "export_landmarks_csv, for copy-paste / upload into the viewer.",
    )
    args = ap.parse_args()

    # Per-frame MediaPipe-normalized landmarks (shared by both formats).
    frame_landmarks = []
    for i in range(N_FRAMES):
        p = pose_at(i)
        skel = skeleton(p)
        frame_landmarks.append([to_mediapipe(skel[j]) for j in range(33)])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.format == "csv":
        import csv

        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "timestamp_sec", "landmark_index", "landmark_name",
                        "x", "y", "z", "visibility"])
            for i, lms in enumerate(frame_landmarks):
                t = round(i / FPS, 3)
                for idx, (x, y, z) in enumerate(lms):
                    w.writerow([i, t, idx, LANDMARK_NAMES[idx], x, y, z, 1.0])
        print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB, "
              f"{N_FRAMES} frames x 33 landmarks)")
        return

    frames = []
    for i, lms in enumerate(frame_landmarks):
        frames.append(
            {
                "frame": i,
                "t": round(i / FPS, 3),
                "detected": True,
                "landmarks": lms,
                "visibility": [1.0] * 33,
            }
        )

    payload = {
        "meta": {
            "schema": "tennis-mediapipe/landmarks3d@1",
            "fps": float(FPS),
            "frame_count": N_FRAMES,
            "duration_sec": round(N_FRAMES / FPS, 3),
            "source": "synthetic_demo_forehand",
            "synthetic": True,
            "dominant_hand": "Right",
            "coordinate_system": "mediapipe_normalized",
            "generated": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "SYNTHETIC DEMO DATA - not from a real video. Procedurally animated "
                "forehand-style swing for exploring the Three.js viewer. Run "
                "export_3d_json.py on a real swing recording for actual analysis."
            ),
        },
        "landmark_names": LANDMARK_NAMES,
        "connections": {
            "upper": [list(c) for c in UPPER_BODY_CONNECTIONS],
            "lower": [list(c) for c in LOWER_BODY_CONNECTIONS],
            "head": [list(c) for c in HEAD_CONNECTIONS],
        },
        "frames": frames,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1024:.0f} KB, {N_FRAMES} frames)")


if __name__ == "__main__":
    main()
