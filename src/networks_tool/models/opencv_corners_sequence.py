from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class OpenCVCornersCalibrationModel(nn.Module):
    """Classical OpenCV calibration wrapper with a model-like API.

    This class is designed for the case where the input is already a sequence of
    checkerboard corners, not raw images for CNN feature extraction. It keeps the
    full pipeline in one file:
    - checkerboard corner extraction from images,
    - corner refinement and validation,
    - OpenCV calibration from 2D-3D correspondences,
    - conversion to the same 9-value vector used by the sequence LSTM target.

    The output vector order is:
    [fx, fy, cx, cy, k1, k2, p1, p2, k3]

    The first four values are normalized exactly like the sequence dataset:
    fx / width, fy / height, cx / width, cy / height.
    """

    vector_order = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]

    def __init__(
        self,
        num_outputs: int = 9,
        image_width: int = 640,
        image_height: int = 480,
        num_squares_x: int = 9,
        num_squares_y: int = 6,
        square_size: float = 48.0,
        required_frames: int = 20,
    ):
        super().__init__()
        if num_outputs != 9:
            raise ValueError("OpenCVCornersCalibrationModel returns exactly 9 outputs.")

        self.num_outputs = num_outputs
        self.image_width = image_width
        self.image_height = image_height
        self.num_squares_x = num_squares_x
        self.num_squares_y = num_squares_y
        self.square_size = square_size
        self.required_frames = required_frames
        self.pattern_size = (num_squares_x - 1, num_squares_y - 1)
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

    def build_object_points(self) -> np.ndarray:
        """Create the checkerboard 3D points on the z=0 plane."""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.pattern_size[0], 0 : self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def extract_corners_from_image(self, image: np.ndarray) -> np.ndarray | None:
        """Detect and refine checkerboard corners in a single image."""
        height, width = image.shape[:2]
        if (width, height) != (self.image_width, self.image_height):
            raise ValueError(
                f"Expected image size {self.image_width}x{self.image_height}, got {width}x{height}."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None

        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            self.subpix_criteria,
        )
        return refined.reshape(-1, 2).astype(np.float32)

    def extract_corners_from_paths(self, frame_paths: list[Path]) -> list[np.ndarray]:
        """Load images and extract refined checkerboard corners."""
        if len(frame_paths) != self.required_frames:
            raise ValueError(
                f"Expected exactly {self.required_frames} frame paths, got {len(frame_paths)}."
            )

        corner_sets: list[np.ndarray] = []
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {frame_path}")

            corners = self.extract_corners_from_image(image)
            if corners is not None:
                corner_sets.append(corners)

        return corner_sets

    def preprocess_corners(self, corner_sets: list[np.ndarray]) -> np.ndarray:
        """Validate corner shapes and stack them into a sequence array."""
        if len(corner_sets) < 3:
            raise RuntimeError(
                "Not enough valid checkerboard detections for calibration. "
                f"Detected corners in {len(corner_sets)}/{self.required_frames} frames."
            )

        expected_points = self.pattern_size[0] * self.pattern_size[1]
        processed: list[np.ndarray] = []
        for corners in corner_sets:
            corners_array = np.asarray(corners, dtype=np.float32)
            if corners_array.shape != (expected_points, 2):
                raise ValueError(
                    "Invalid corner array shape. Expected "
                    f"({expected_points}, 2), got {corners_array.shape}."
                )
            processed.append(corners_array)

        return np.stack(processed, axis=0)

    def calibrate_from_corners(self, corner_sets: list[np.ndarray] | np.ndarray) -> dict[str, object]:
        """Run classical OpenCV calibration on detected checkerboard corners."""
        if isinstance(corner_sets, np.ndarray):
            if corner_sets.ndim != 3:
                raise ValueError("Expected corner array with shape (T, P, 2).")
            corners_sequence = corner_sets.astype(np.float32)
            if corners_sequence.shape[0] < 3:
                raise RuntimeError(
                    "Not enough valid checkerboard detections for calibration. "
                    f"Detected corners in {corners_sequence.shape[0]}/{self.required_frames} frames."
                )
            expected_points = self.pattern_size[0] * self.pattern_size[1]
            if corners_sequence.shape[1:] != (expected_points, 2):
                raise ValueError(
                    "Invalid corner array shape. Expected "
                    f"(T, {expected_points}, 2), got {corners_sequence.shape}."
                )
        else:
            corners_sequence = self.preprocess_corners(corner_sets)

        objp = self.build_object_points()
        obj_points = [objp.copy() for _ in range(corners_sequence.shape[0])]
        img_points = [corners.reshape(-1, 1, 2) for corners in corners_sequence]

        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_points,
            img_points,
            (self.image_width, self.image_height),
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

        vector = np.array(
            [
                fx / self.image_width,
                fy / self.image_height,
                cx / self.image_width,
                cy / self.image_height,
                k1,
                k2,
                p1,
                p2,
                k3,
            ],
            dtype=np.float32,
        )

        return {
            "rms_reprojection_error": float(rms),
            "detected_frames": int(corners_sequence.shape[0]),
            "vector_order": list(self.vector_order),
            "vector_lstm_compatible": vector,
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

    def predict_from_images(self, frame_paths: list[Path]) -> dict[str, object]:
        """Full end-to-end pipeline from images to LSTM-compatible vector."""
        corner_sets = self.extract_corners_from_paths(frame_paths)
        return self.calibrate_from_corners(corner_sets)

    def forward(self, corners: torch.Tensor) -> torch.Tensor:
        """Compute the 9-value calibration vector from batched corner tensors.

        Args:
            corners: Tensor of shape (T, P, 2) or (B, T, P, 2).

        Returns:
            Tensor of shape (1, 9) or (B, 9).
        """
        if corners.ndim == 3:
            corners = corners.unsqueeze(0)
        if corners.ndim != 4:
            raise ValueError("Expected corners tensor with shape (T, P, 2) or (B, T, P, 2).")

        device = corners.device
        results = []
        corners_np = corners.detach().cpu().numpy()
        for sample in corners_np:
            calibration = self.calibrate_from_corners(sample)
            results.append(calibration["vector_lstm_compatible"])

        return torch.tensor(np.stack(results, axis=0), dtype=torch.float32, device=device)