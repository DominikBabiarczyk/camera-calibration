"""Calibrate camera from a 20-frame chessboard sequence using OpenCV.

This script expects frames with resolution 640x480 containing a chessboard
created like in `generate_dataset/generate_chessboard.py`:
- 9x6 squares
- 48 px square size

It returns a vector compatible with the sequence LSTM target order:
[fx, fy, cx, cy, k1, k2, p1, p2, k3]
where fx/fy/cx/cy are normalized exactly like in the sequence dataset:
fx/width, fy/height, cx/width, cy/height.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_SQUARES_X = 9
NUM_SQUARES_Y = 6
SQUARE_SIZE_PX = 48.0
REQUIRED_FRAMES = 20

# OpenCV uses the number of inner corners.
PATTERN_SIZE = (NUM_SQUARES_X - 1, NUM_SQUARES_Y - 1)  # (8, 5)


def build_object_points() -> np.ndarray:
    """Create 3D object points for the checkerboard pattern."""
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : PATTERN_SIZE[0], 0 : PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_PX
    return objp


def calibrate_from_sequence(frame_paths: list[Path]) -> dict[str, object]:
    """Calibrate camera using exactly 20 frame paths.

    Args:
        frame_paths: Exactly 20 image paths (640x480) containing a checkerboard.

    Returns:
        Dictionary with RMS error, raw OpenCV parameters, and LSTM-compatible vector.
    """
    if len(frame_paths) != REQUIRED_FRAMES:
        raise ValueError(
            f"Expected exactly {REQUIRED_FRAMES} frames, got {len(frame_paths)}."
        )

    objp = build_object_points()
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        h, w = image.shape[:2]
        if (w, h) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            raise ValueError(
                f"Invalid image size for {path}: got {w}x{h}, expected "
                f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            PATTERN_SIZE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 3:
        raise RuntimeError(
            "Not enough valid checkerboard detections for calibration. "
            f"Detected corners in {len(obj_points)}/{REQUIRED_FRAMES} frames."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        None,
        None,
    )

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    dist = dist_coeffs.ravel()
    k1 = float(dist[0]) if dist.size > 0 else 0.0
    k2 = float(dist[1]) if dist.size > 1 else 0.0
    p1 = float(dist[2]) if dist.size > 2 else 0.0
    p2 = float(dist[3]) if dist.size > 3 else 0.0
    k3 = float(dist[4]) if dist.size > 4 else 0.0

    lstm_vector = [
        fx / IMAGE_WIDTH,
        fy / IMAGE_HEIGHT,
        cx / IMAGE_WIDTH,
        cy / IMAGE_HEIGHT,
        k1,
        k2,
        p1,
        p2,
        k3,
    ]

    return {
        "rms_reprojection_error": float(rms),
        "detected_frames": len(obj_points),
        "required_frames": REQUIRED_FRAMES,
        "pattern_inner_corners": [PATTERN_SIZE[0], PATTERN_SIZE[1]],
        "square_size_px": SQUARE_SIZE_PX,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "vector_order": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
        "vector_lstm_compatible": lstm_vector,
        "raw_parameters": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "k3": k3,
        },
    }


def pick_20_frames(frames_dir: Path, pattern: str, start_index: int) -> list[Path]:
    """Pick a contiguous 20-frame window from a directory."""
    all_frames = sorted(frames_dir.glob(pattern))
    if not all_frames:
        raise FileNotFoundError(
            f"No frames found in {frames_dir} with pattern '{pattern}'."
        )

    end_index = start_index + REQUIRED_FRAMES
    selected = all_frames[start_index:end_index]
    if len(selected) != REQUIRED_FRAMES:
        raise ValueError(
            f"Cannot select {REQUIRED_FRAMES} frames from index {start_index}. "
            f"Found only {len(selected)} frames in that range."
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV calibration from a 20-frame checkerboard sequence"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing frames (e.g., frame_000001.jpg)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_*.jpg",
        help="Glob pattern for frame files",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in sorted frame list for selecting 20 frames",
    )
    args = parser.parse_args()

    frame_paths = pick_20_frames(args.frames_dir, args.pattern, args.start_index)
    result = calibrate_from_sequence(frame_paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
