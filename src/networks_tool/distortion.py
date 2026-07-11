"""Synthetic distortion generation using the Brown-Conrady model.

This module applies known radial distortion to undistorted images,
creating (image, parameters) training pairs. The key insight is that
we can generate infinite training data from any set of undistorted images.

The Brown-Conrady radial distortion model:
    r_distorted = r * (1 + k1 * r^2 + k2 * r^4)

where r is the distance from the principal point (image center).
"""

import cv2
import numpy as np


def apply_radial_distortion(
    image: np.ndarray,
    focal_length: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    """Apply radial distortion to an undistorted image.

    Args:
        image: Undistorted input image (H, W, 3).
        focal_length: Focal length in pixels.
        k1: First radial distortion coefficient.
        k2: Second radial distortion coefficient.

    Returns:
        Distorted image of the same shape.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Build the camera matrix
    camera_matrix = np.array(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    # cv2.undistort goes distorted→undistorted, so we use initUndistortRectifyMap
    # in reverse: we want to CREATE distortion, so we swap the logic.
    # We generate a map from undistorted coords to distorted coords.
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        R=None,
        newCameraMatrix=camera_matrix,
        size=(w, h),
        m1type=cv2.CV_32FC1,
    )
    distorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    return distorted


def sample_random_parameters(
    image_width: int,
    focal_range: tuple[float, float],
    k1_range: tuple[float, float],
    k2_range: tuple[float, float],
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Sample random calibration parameters for synthetic data generation.

    Args:
        image_width: Width of the image (used to scale focal length).
        focal_range: (min, max) for normalized focal length (f / width).ta
        k1_range: (min, max) for k1 distortion coefficient.
        k2_range: (min, max) for k2 distortion coefficient.
        rng: NumPy random generator for reproducibility.

    Returns:
        (focal_length_pixels, k1, k2)
    """
    if rng is None:
        rng = np.random.default_rng()

    f_norm = rng.uniform(*focal_range)
    focal_length = f_norm * image_width
    k1 = rng.uniform(*k1_range)
    k2 = rng.uniform(*k2_range)

    return focal_length, k1, k2
