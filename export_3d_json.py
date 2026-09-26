#!/usr/bin/env python3
"""
Export per-frame MediaPipe 3D pose landmarks to a JSON file for the
Three.js 3D viewer (viewer/index.html).

The viewer expects the "tennis-mediapipe/landmarks3d@1" schema:

{
  "meta": {
    "schema": "tennis-mediapipe/landmarks3d@1",
    "fps": 30.0,
    "frame_count": 150,
    "duration_sec": 5.0,
    "source": "my_swing.mp4",
    "synthetic": false,
    "dominant_hand": "Right",
    "coordinate_system": "mediapipe_normalized",
    "generated": "2026-09-26T...",
    "notes": "..."
  },
  "landmark_names": ["nose", "left_eye_inner", ...],   # 33 MediaPipe names
  "connections": {"upper": [[11, 12], ...], "lower": [...], "head": [...]},
  "frames": [
    {
      "frame": 0,
      "t": 0.0,
      "detected": true,
      "landmarks": [[x, y, z], ... 33 entries ...],   # MediaPipe normalized coords
      "visibility": [0.99, ... 33 entries ...]
    },
    ...
  ]
}

Coordinate system (MediaPipe normalized):
  x: 0 (left edge) .. 1 (right edge)
  y: 0 (top) .. 1 (bottom)
  z: depth, ~0 at mid-hip; negative values are closer to the camera.
Frames where no pose was detected carry "detected": false and
"landmarks": null so the viewer can show a "no pose" state.

Usage:
    python export_3d_json.py path/to/swing.mp4 -o output/landmarks_3d.json
    python export_3d_json.py path/to/swing.mp4 --dominant-hand Left
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import TennisPoseDetector
from video_processor import TennisVideoProcessor

SCHEMA = "tennis-mediapipe/landmarks3d@1"
NUM_LANDMARKS = 33


def build_payload(
    video_path: str,
    landmarks_history,
    fps: float,
    frame_count: int,
    dominant_hand: str,
) -> dict:
    # Pivot the per-landmark history into per-frame records.
    by_frame = {}
    for rec in landmarks_history:
        f = rec["frame"]
        by_frame.setdefault(f, []).append(rec)

    frames = []
    for frame_idx in range(frame_count):
        recs = by_frame.get(frame_idx, [])
        if len(recs) == NUM_LANDMARKS:
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
            "schema": SCHEMA,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": round(frame_count / fps, 3) if fps else 0.0,
            "source": os.path.basename(video_path),
            "synthetic": False,
            "dominant_hand": dominant_hand,
            "coordinate_system": "mediapipe_normalized",
            "generated": datetime.now(timezone.utc).isoformat(),
            "notes": (
                "x: 0..1 left-to-right, y: 0..1 top-to-bottom, "
                "z: depth (~0 at mid-hip, negative toward camera). "
                "Produced by tennis-mediapipe export_3d_json.py"
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export MediaPipe 3D pose landmarks to JSON for the Three.js viewer."
    )
    ap.add_argument("video", help="Input video file (e.g. a tennis swing recording).")
    ap.add_argument(
        "-o",
        "--output",
        default=os.path.join("output", "landmarks_3d.json"),
        help="Destination JSON path (default: output/landmarks_3d.json).",
    )
    ap.add_argument(
        "--dominant-hand",
        default="Right",
        choices=["Right", "Left"],
        help="Player's dominant hand (drives which wrist trail is highlighted).",
    )
    ap.add_argument(
        "--angle-mode",
        default="bend",
        choices=["bend", "interior"],
        help="Angle convention used during scanning (kept for telemetry parity).",
    )
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    detector = TennisPoseDetector(
        dominant_hand=args.dominant_hand, angle_mode=args.angle_mode
    )
    processor = TennisVideoProcessor(detector)

    vid_meta = processor.get_video_metadata(args.video)
    fps = vid_meta["fps"]
    frame_count = vid_meta["frame_count"]
    print(f"Scanning {args.video}: {frame_count} frames @ {fps} fps ...")

    def _progress(p, msg):
        print(f"\r  {p * 100:5.1f}%  {msg}", end="", flush=True)

    _telemetry_df, landmarks_history = processor.scan_video(
        args.video, progress_callback=_progress
    )
    print()

    detected = sum(1 for r in landmarks_history) // NUM_LANDMARKS
    print(f"Pose detected on {detected}/{frame_count} frames.")

    payload = build_payload(
        args.video, landmarks_history, fps, frame_count, detector.dominant_hand
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)
    size_kb = os.path.getsize(args.output) / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")
    print("Open viewer/index.html and load this file to explore the swing in 3D.")


if __name__ == "__main__":
    main()
