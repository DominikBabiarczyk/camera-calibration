"""Generate augmented chessboard images with realistic perspective variations.

Usage:
    python augment_perspectives.py --input input.png --outdir outputs --config perspective_config.yaml --count 50 --sequences 10 --seed 42

The script loads the YAML configuration, samples parameters, builds a homography
from a simple pinhole/camera model + plane geometry, and writes warped images.
Each sequence preserves static camera/distortion parameters while varying the
board perspective across frames.
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sample_range(cfg: Dict[str, Any], key: str, rng: np.random.RandomState) -> float:
    entry = cfg.get(key)
    if entry is None:
        raise KeyError(f"Missing key in config: {key}")
    lo = entry.get("min")
    hi = entry.get("max")
    return float(rng.uniform(lo, hi))


def rotation_matrix_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    # angles in radians
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    # apply Rx then Ry then Rz
    return Rz @ Ry @ Rx


def build_projected_corners(
    w: int,
    h: int,
    params: Dict[str, float],
) -> np.ndarray:
    """Project 4 image corners through a simple camera+plane model and return
    their destination 2D coordinates.

    Returns an array shape (4,2) in pixel coordinates in order tl,tr,br,bl.
    """
    # source image corners centered in world plane (Z=0)
    cx = w / 2.0
    cy = h / 2.0
    src = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=float)
    # express corners in plane coordinates centered at origin
    plane_pts = np.column_stack((src[:, 0] - cx, src[:, 1] - cy, np.zeros(4)))

    # rotations (degrees->radians) - pitch (x), yaw (y), roll (z)
    pitch = np.deg2rad(params.get("pitch_deg", 0.0))
    yaw = np.deg2rad(params.get("yaw_deg", 0.0))
    roll = np.deg2rad(params.get("roll_deg", 0.0))
    R = rotation_matrix_from_euler(pitch, yaw, roll)

    # translation (x,y) as fraction of image dims -> pixels
    tx = params.get("translate_x", 0.0) * w
    ty = params.get("translate_y", 0.0) * h
    # distance (z) derived from zoom: larger zoom -> smaller distance
    zoom = params.get("zoom", 1.0)
    if abs(zoom) < 1e-8:
        raise ValueError("Invalid sampled 'zoom'=0. Set non-zero min/max in the config.")
    # choose a base distance in pixels (rough heuristic)
    base_distance = max(w, h) * 1.5
    tz = base_distance / zoom

    t = np.array([tx, ty, tz])

    # focal length
    base_f = max(w, h) * 0.9
    focal_length_scale = params.get("focal_length_scale", 1.0)
    if abs(focal_length_scale) < 1e-8:
        raise ValueError("Invalid sampled 'focal_length_scale'=0. Set non-zero min/max in the config.")
    f = base_f * focal_length_scale

    # project
    dst = []
    for P in plane_pts:
        P_cam = R @ P + t
        x = (f * (P_cam[0] / P_cam[2])) + cx
        y = (f * (P_cam[1] / P_cam[2])) + cy
        dst.append([x, y])
    dst = np.array(dst, dtype=float)

    # apply perspective skew (small fractional perturbation based on image size)
    skew = params.get("perspective_skew", 0.0)
    if abs(skew) > 1e-6:
        # push corners toward/away to increase projective effect
        pert = np.array([[ -1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        dst += (pert * (skew * np.array([[w, h]])))

    # corner jitter in pixels
    corner_shift = params.get("corner_shift_px", 0.0)
    if corner_shift > 0:
        rng = params.get("_rng")
        if rng is not None:
            jitter = rng.uniform(-corner_shift, corner_shift, size=(4, 2))
            dst += jitter

    # apply shear (affine) in normalized units
    sx = params.get("shear_x", 0.0)
    sy = params.get("shear_y", 0.0)
    if abs(sx) > 1e-6 or abs(sy) > 1e-6:
        shear_m = np.array([[1.0, sx], [sy, 1.0]])
        dst = ((dst - np.array([cx, cy])) @ shear_m.T) + np.array([cx, cy])

    return dst


def build_camera_matrix(w: int, h: int, params: Dict[str, float]) -> np.ndarray:
    fx = params["fx"]
    fy = params["fy"]
    cx = params["cx"]
    cy = params["cy"]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def euler_to_rvec(pitch: float, yaw: float, roll: float) -> np.ndarray:
    R = rotation_matrix_from_euler(pitch, yaw, roll)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3, 1)


def generate_board_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    xs = np.arange(cols, dtype=float) * square_size
    ys = np.arange(rows, dtype=float) * square_size
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return np.column_stack((grid, np.zeros((grid.shape[0], 1), dtype=float)))


def sample_camera_static_params(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    p["fx"] = sample_range(cfg, "fx", rng) * max(w, h)
    p["fy"] = sample_range(cfg, "fy", rng) * max(w, h)
    p["cx"] = sample_range(cfg, "principal_point_x", rng) * w
    p["cy"] = sample_range(cfg, "principal_point_y", rng) * h
    p["k1"] = sample_range(cfg, "k1", rng)
    p["k2"] = sample_range(cfg, "k2", rng)
    p["p1"] = sample_range(cfg, "p1", rng)
    p["p2"] = sample_range(cfg, "p2", rng)
    p["k3"] = sample_range(cfg, "k3", rng)
    p["square_size"] = sample_range(cfg, "square_size", rng) * max(w, h)
    p["board_cols"] = int(round(sample_range(cfg, "board_cols", rng)))
    p["board_rows"] = int(round(sample_range(cfg, "board_rows", rng)))
    return p


def sample_camera_dynamic_params(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    pitch = np.deg2rad(sample_range(cfg, "pitch_deg", rng))
    yaw = np.deg2rad(sample_range(cfg, "yaw_deg", rng))
    roll = np.deg2rad(sample_range(cfg, "roll_deg", rng))
    p["rvec"] = euler_to_rvec(pitch, yaw, roll)
    p["tvec"] = np.array([
        sample_range(cfg, "tvec_x", rng) * w,
        sample_range(cfg, "tvec_y", rng) * h,
        sample_range(cfg, "tvec_z", rng) * max(w, h),
    ], dtype=float).reshape(3, 1)
    return p


def project_chessboard_grid(w: int, h: int, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    cols = params["board_cols"] + 1
    rows = params["board_rows"] + 1
    board_points = generate_board_object_points(cols, rows, params["square_size"])
    camera_matrix = build_camera_matrix(w, h, params)
    dist_coeffs = np.array([params["k1"], params["k2"], params["p1"], params["p2"], params["k3"]], dtype=float)
    img_points, _ = cv2.projectPoints(board_points, params["rvec"], params["tvec"], camera_matrix, dist_coeffs)
    img_points = img_points.reshape(rows, cols, 2)

    xs = np.linspace(0.0, float(w), num=cols, dtype=float)
    ys = np.linspace(0.0, float(h), num=rows, dtype=float)
    src_grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    return src_grid, img_points


def build_homography_from_camera_params(w: int, h: int, params: Dict[str, Any]) -> np.ndarray:
    camera_matrix = build_camera_matrix(w, h, params)
    R, _ = cv2.Rodrigues(params["rvec"])
    cx = w / 2.0
    cy = h / 2.0
    # source image coordinates are in pixel space [0,w]x[0,h]. We must center
    # them around the camera/world origin before applying the plane homography.
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=float)
    H = camera_matrix @ np.hstack((R[:, :2], params["tvec"])) @ T
    return H


def distort_image(img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    h, w = img.shape[:2]
    K = build_camera_matrix(w, h, params)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = (
        params["k1"],
        params["k2"],
        params["p1"],
        params["p2"],
        params["k3"],
    )
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.indices((h, w), dtype=np.float32)
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3)
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * (x ** 2))
    y_dist = y * radial + p1 * (r2 + 2.0 * (y ** 2)) + 2.0 * p2 * x * y
    map_x[:] = (x_dist * fx) + cx
    map_y[:] = (y_dist * fy) + cy
    border_color = tuple(int(x) for x in params.get("_border_color", [255, 255, 255]))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)


def augment_image_camera(
    img: np.ndarray,
    params: Dict[str, Any],
    out_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    out_w, out_h = (w, h) if out_shape is None else out_shape
    H = build_homography_from_camera_params(w, h, params)
    border_color = tuple(int(x) for x in params.get("_border_color", [255, 255, 255]))
    warped = cv2.warpPerspective(
        img,
        H,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )
    return distort_image(warped, params)


def build_homography_from_params(w: int, h: int, params: Dict[str, float]) -> np.ndarray:
    src = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    dst = build_projected_corners(w, h, params).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H


def augment_image(
    img: np.ndarray,
    params: Dict[str, float],
    out_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    out_w, out_h = (w, h) if out_shape is None else out_shape
    H = build_homography_from_params(w, h, params)
    # choose border mode from params
    border_mode_name = params.get("_border_mode", "constant")
    border_color = params.get("_border_color", [0, 0, 0])
    if border_mode_name == "reflect":
        border = cv2.BORDER_REFLECT
        warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border)
    elif border_mode_name == "replicate":
        border = cv2.BORDER_REPLICATE
        warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=border)
    else:
        # constant: use borderColor (B,G,R)
        bgr = tuple(int(x) for x in border_color)
        warped = cv2.warpPerspective(
            img, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bgr
        )
    return warped


def dst_within_bounds(dst: np.ndarray, w: int, h: int, margin: float = 1.0) -> bool:
    # dst: (4,2)
    if dst.shape[0] == 0:
        return False
    x_ok = np.logical_and(dst[:, 0] >= -margin, dst[:, 0] <= (w + margin))
    y_ok = np.logical_and(dst[:, 1] >= -margin, dst[:, 1] <= (h + margin))
    return bool(np.all(np.logical_and(x_ok, y_ok)))


def sample_params(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState) -> Dict[str, float]:
    p: Dict[str, float] = {}
    # sample known keys if present
    for key in [
        "zoom",
        "pitch_deg",
        "yaw_deg",
        "roll_deg",
        "translate_x",
        "translate_y",
        "perspective_skew",
        "corner_shift_px",
        "focal_length_scale",
        "shear_x",
        "shear_y",
        "radial_distortion",
    ]:
        if key in cfg:
            p[key] = sample_range(cfg, key, rng)
    # add rng so downstream functions can jitter corners reproducibly
    p["_rng"] = rng
    return p


def sample_camera_static_params_constrained(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState) -> Dict[str, Any]:
    border_mode = cfg.get("border_mode", "constant")
    border_color = cfg.get("border_color", [255, 255, 255])
    params = sample_camera_static_params(cfg, w, h, rng)
    params["_border_mode"] = border_mode
    params["_border_color"] = border_color
    return params


def sample_camera_dynamic_params_constrained(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState, static_params: Dict[str, Any]) -> Dict[str, Any]:
    allow_oob = bool(cfg.get("allow_out_of_bounds", False))
    max_attempts = int(cfg.get("max_resample_attempts", 50))
    border_mode = cfg.get("border_mode", "constant")
    border_color = cfg.get("border_color", [255, 255, 255])

    for attempt in range(max_attempts if not allow_oob else 1):
        params = {**static_params, **sample_camera_dynamic_params(cfg, w, h, rng)}
        params["_border_mode"] = border_mode
        params["_border_color"] = border_color
        src = np.array([[[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]], dtype=np.float32)
        H = build_homography_from_camera_params(w, h, params)
        dst = cv2.perspectiveTransform(src, H)[0]
        if allow_oob or dst_within_bounds(dst, w, h, margin=1.0):
            return params
    params = {**static_params, **sample_camera_dynamic_params(cfg, w, h, rng)}
    params["_border_mode"] = border_mode
    params["_border_color"] = border_color
    return params


def sample_params_constrained(cfg: Dict[str, Any], w: int, h: int, rng: np.random.RandomState) -> Dict[str, Any]:
    mode = cfg.get("mode", "perspective")
    if mode == "camera":
        return sample_camera_static_params_constrained(cfg, w, h, rng)

    allow_oob = bool(cfg.get("allow_out_of_bounds", False))
    max_attempts = int(cfg.get("max_resample_attempts", 50))
    border_mode = cfg.get("border_mode", "constant")
    border_color = cfg.get("border_color", [255, 255, 255])

    for attempt in range(max_attempts if not allow_oob else 1):
        params = sample_params(cfg, w, h, rng)
        params["_border_mode"] = border_mode
        params["_border_color"] = border_color
        dst = build_projected_corners(w, h, params)
        if allow_oob or dst_within_bounds(dst, w, h, margin=1.0):
            return params
    params["_border_mode"] = border_mode
    params["_border_color"] = border_color
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--count", type=int, default=None, help="Number of frames per sequence")
    ap.add_argument("--sequences", type=int, default=None, help="Number of sequences to generate")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    count = args.count or int(cfg.get("samples", 50))
    sequences = args.sequences if args.sequences is not None else int(cfg.get("sequences", 1))
    seed = args.seed if args.seed is not None else cfg.get("seed")

    # create RNG
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")
    h, w = img.shape[:2]

    os.makedirs(args.outdir, exist_ok=True)

    mode = cfg.get("mode", "perspective")
    border_mode_cfg = cfg.get("border_mode", "constant")

    for seq_idx in range(sequences):
        sequence_outdir = (
            args.outdir
            if sequences == 1
            else os.path.join(args.outdir, f"sequence_{seq_idx:03d}")
        )
        os.makedirs(sequence_outdir, exist_ok=True)

        static_camera_params: Optional[Dict[str, Any]] = None
        if mode == "camera":
            static_camera_params = sample_camera_static_params_constrained(cfg, w, h, rng)
            camera_labels = {
                key: static_camera_params[key]
                for key in ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
                if key in static_camera_params
            }
            camera_params_path = os.path.join(sequence_outdir, "camera_params.yaml")
            with open(camera_params_path, "w") as f:
                yaml.safe_dump(camera_labels, f, sort_keys=False)

        for i in range(count):
            if mode == "camera":
                params = sample_camera_dynamic_params_constrained(cfg, w, h, rng, static_camera_params)
            else:
                params = sample_params_constrained(cfg, w, h, rng)
                params["_rng"] = np.random.RandomState(rng.randint(0, 2 ** 31))

            if mode == "camera":
                warped = augment_image_camera(img, params)
            else:
                if not cfg.get("allow_out_of_bounds", False) and not dst_within_bounds(
                    build_projected_corners(w, h, params), w, h
                ):
                    params["_border_mode"] = border_mode_cfg if border_mode_cfg != "constant" else "reflect"
                warped = augment_image(img, params)

            out_path = os.path.join(sequence_outdir, f"aug_{i:03d}.png")
            cv2.imwrite(out_path, warped)


if __name__ == "__main__":
    main()
