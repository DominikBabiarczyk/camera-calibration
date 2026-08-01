"""Synthetic calibration dataset.

Generates training pairs by applying known distortion to undistorted images.
Each sample is (distorted_image, [f_normalized, k1, k2]).
"""

from pathlib import Path
import logging
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .config import TrainingConfig
from .distortion import apply_radial_distortion, sample_random_parameters


logger = logging.getLogger(__name__)


def _sample_range(cfg: dict[str, Any], key: str, rng: np.random.RandomState) -> float:
    entry = cfg.get(key)
    if entry is None:
        raise KeyError(f"Missing key in config: {key}")
    return float(rng.uniform(entry["min"], entry["max"]))


def _rotation_matrix_from_euler(pitch: float, yaw: float, roll: float) -> np.ndarray:
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])
    rot_y = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    rot_z = np.array([
        [np.cos(roll), -np.sin(roll), 0.0],
        [np.sin(roll), np.cos(roll), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rot_z @ rot_y @ rot_x


def _build_camera_matrix(image_w: int, image_h: int, params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [params["fx"], 0.0, params["cx"]],
            [0.0, params["fy"], params["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _euler_to_rvec(pitch: float, yaw: float, roll: float) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(_rotation_matrix_from_euler(pitch, yaw, roll))
    return rotation.reshape(3, 1)


def _generate_board_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    xs = np.arange(cols, dtype=float) * square_size
    ys = np.arange(rows, dtype=float) * square_size
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return np.column_stack((grid, np.zeros((grid.shape[0], 1), dtype=float)))


def _dst_within_bounds(dst: np.ndarray, image_w: int, image_h: int, margin: float = 1.0) -> bool:
    if dst.shape[0] == 0:
        return False
    x_ok = np.logical_and(dst[:, 0] >= -margin, dst[:, 0] <= (image_w + margin))
    y_ok = np.logical_and(dst[:, 1] >= -margin, dst[:, 1] <= (image_h + margin))
    return bool(np.all(np.logical_and(x_ok, y_ok)))


def _sample_camera_static_params(
    cfg: dict[str, Any],
    image_w: int,
    image_h: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "fx": _sample_range(cfg, "fx", rng) * max(image_w, image_h),
        "fy": _sample_range(cfg, "fy", rng) * max(image_w, image_h),
        "cx": _sample_range(cfg, "principal_point_x", rng) * image_w,
        "cy": _sample_range(cfg, "principal_point_y", rng) * image_h,
        "k1": _sample_range(cfg, "k1", rng),
        "k2": _sample_range(cfg, "k2", rng),
        "p1": _sample_range(cfg, "p1", rng),
        "p2": _sample_range(cfg, "p2", rng),
        "k3": _sample_range(cfg, "k3", rng),
    }
    return params


def _sample_camera_dynamic_params(
    cfg: dict[str, Any],
    image_w: int,
    image_h: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    pitch = np.deg2rad(_sample_range(cfg, "pitch_deg", rng))
    yaw = np.deg2rad(_sample_range(cfg, "yaw_deg", rng))
    roll = np.deg2rad(_sample_range(cfg, "roll_deg", rng))
    return {
        "rvec": _euler_to_rvec(pitch, yaw, roll),
        "tvec": np.array(
            [
                _sample_range(cfg, "tvec_x", rng) * image_w,
                _sample_range(cfg, "tvec_y", rng) * image_h,
                _sample_range(cfg, "tvec_z", rng) * max(image_w, image_h),
            ],
            dtype=float,
        ).reshape(3, 1),
    }


def _project_chessboard_grid(
    image_w: int,
    image_h: int,
    params: dict[str, Any],
) -> np.ndarray:
    cols = params["board_cols"] + 1
    rows = params["board_rows"] + 1
    board_points = _generate_board_object_points(cols, rows, params["square_size"])
    camera_matrix = _build_camera_matrix(image_w, image_h, params)
    dist_coeffs = np.array(
        [params["k1"], params["k2"], params["p1"], params["p2"], params["k3"]],
        dtype=float,
    )
    image_points, _ = cv2.projectPoints(
        board_points,
        params["rvec"],
        params["tvec"],
        camera_matrix,
        dist_coeffs,
    )
    return image_points.reshape(rows, cols, 2)


def _build_homography_from_camera_params(
    image_w: int,
    image_h: int,
    params: dict[str, Any],
) -> np.ndarray:
    camera_matrix = _build_camera_matrix(image_w, image_h, params)
    rotation_matrix, _ = cv2.Rodrigues(params["rvec"])
    translate_to_center = np.array(
        [
            [1.0, 0.0, -(image_w / 2.0)],
            [0.0, 1.0, -(image_h / 2.0)],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return camera_matrix @ np.hstack((rotation_matrix[:, :2], params["tvec"])) @ translate_to_center


def _sample_camera_static_params_constrained(
    cfg: dict[str, Any],
    image_w: int,
    image_h: int,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    params = _sample_camera_static_params(cfg, image_w, image_h, rng)
    params["_border_mode"] = cfg.get("border_mode", "constant")
    params["_border_color"] = cfg.get("border_color", [255, 255, 255])
    return params


def _sample_camera_dynamic_params_constrained(
    cfg: dict[str, Any],
    image_w: int,
    image_h: int,
    rng: np.random.RandomState,
    static_params: dict[str, Any],
) -> dict[str, Any]:
    allow_out_of_bounds = bool(cfg.get("allow_out_of_bounds", False))
    max_resample_attempts = int(cfg.get("max_resample_attempts", 50))

    source_corners = np.array([[[0.0, 0.0], [image_w, 0.0], [image_w, image_h], [0.0, image_h]]], dtype=np.float32)
    for _ in range(max_resample_attempts if not allow_out_of_bounds else 1):
        params = {**static_params, **_sample_camera_dynamic_params(cfg, image_w, image_h, rng)}
        homography = _build_homography_from_camera_params(image_w, image_h, params)
        projected = cv2.perspectiveTransform(source_corners, homography)[0]
        if allow_out_of_bounds or _dst_within_bounds(projected, image_w, image_h, margin=1.0):
            return params

    return {**static_params, **_sample_camera_dynamic_params(cfg, image_w, image_h, rng)}


class SyntheticCalibrationDataset(Dataset):
    """Dataset that generates distorted images with known parameters on-the-fly.

    Each undistorted source image produces `samples_per_image` distorted variants,
    each with randomly sampled (focal_length, k1, k2).
    """

    def __init__(self, image_paths: list[Path], config: TrainingConfig, seed: int = 42):
        self.image_paths = image_paths
        self.config = config
        self.samples_per_image = config.samples_per_image
        self.rng = np.random.default_rng(seed)

        # Standard ImageNet normalization (for pretrained ResNet)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_idx = idx // self.samples_per_image

        # Load the undistorted source image at its original resolution.
        img = cv2.imread(str(self.image_paths[image_idx]))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_paths[image_idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        w = img.shape[1]

        # Sample random calibration parameters
        focal_px, k1, k2 = sample_random_parameters(
            image_width=w,
            focal_range=self.config.focal_length_range,
            k1_range=self.config.k1_range,
            k2_range=self.config.k2_range,
            rng=self.rng,
        )

        # Apply distortion
        distorted = apply_radial_distortion(img, focal_px, k1, k2)

        # Normalize focal length by image width for scale-invariance
        f_normalized = focal_px / w

        # Target vector: [f_normalized, k1, k2]
        target = torch.tensor([f_normalized, k1, k2], dtype=torch.float32)

        # Transform image to tensor
        img_tensor = self.transform(distorted)

        return img_tensor, target


class SequenceCalibrationDataset(Dataset):
    """Dataset that loads image sequences and sequence-level YAML labels."""

    def __init__(self, sequence_dirs: list[Path], config: TrainingConfig):
        self.sequence_dirs = sequence_dirs
        self.config = config
        self.sequence_length = config.sequence_length
        self.sequence_step = config.sequence_step

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.samples: list[tuple[list[Path], torch.Tensor]] = []

        for seq_dir in self.sequence_dirs:
            label_path = seq_dir / "camera_params.yaml"
            if not label_path.exists():
                continue

            frame_paths = sorted(
                p for p in seq_dir.iterdir()
                if p.suffix.lower() in extensions
            )
            if len(frame_paths) < self.sequence_length:
                continue

            first_frame = cv2.imread(str(frame_paths[0]))
            if first_frame is None:
                raise FileNotFoundError(f"Cannot read image: {frame_paths[0]}")
            image_h, image_w = first_frame.shape[:2]

            with open(label_path, "r") as f:
                params = yaml.safe_load(f)

            target = torch.tensor(
                [
                    params["fx"] / image_w,
                    params["fy"] / image_h,
                    params["cx"] / image_w,
                    params["cy"] / image_h,
                    params["k1"],
                    params["k2"],
                    params["p1"],
                    params["p2"],
                    params["k3"],
                ],
                dtype=torch.float32,
            )

            for start in range(0, len(frame_paths) - self.sequence_length + 1, self.sequence_step):
                window = frame_paths[start : start + self.sequence_length]
                self.samples.append((window, target))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid sequences found in {sequence_dirs}. "
                "Each sequence folder must contain a camera_params.yaml and enough images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_paths, target = self.samples[idx]
        frames = []
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {frame_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(img))

        sequence_tensor = torch.stack(frames, dim=0)
        return sequence_tensor, target


class CornerSequenceCalibrationDataset(Dataset):
    """Dataset that extracts refined checkerboard corners from fixed frame windows."""

    def __init__(self, sequence_dirs: list[Path], config: TrainingConfig, split_name: str = "dataset"):
        self.sequence_dirs = sequence_dirs
        self.config = config
        self.sequence_length = config.sequence_length
        self.split_name = split_name
        self.pattern_size = (
            config.corner_num_squares_x - 1,
            config.corner_num_squares_y - 1,
        )
        self.corner_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        sequence_iter = self.sequence_dirs
        if tqdm is not None:
            sequence_iter = tqdm(
                self.sequence_dirs,
                desc=f"Preparing {self.split_name} corner sequences",
                unit="seq",
            )

        for seq_dir in sequence_iter:
            label_path = seq_dir / "camera_params.yaml"

            precomputed_path = seq_dir / "corner_sequence.npz"
            if precomputed_path.exists():
                sample = self._load_precomputed_corner_sequence(precomputed_path)
                if sample is None:
                    continue
                self.samples.append(sample)
                continue

            if not label_path.exists():
                continue

            frame_paths = sorted(
                p for p in seq_dir.iterdir()
                if p.suffix.lower() in extensions
            )
            if len(frame_paths) < self.sequence_length:
                continue

            selected_paths = frame_paths[:self.sequence_length]
            first_frame = cv2.imread(str(selected_paths[0]))
            if first_frame is None:
                raise FileNotFoundError(f"Cannot read image: {selected_paths[0]}")
            image_h, image_w = first_frame.shape[:2]

            with open(label_path, "r") as f:
                params = yaml.safe_load(f)

            target = torch.tensor(
                [
                    params["fx"] / image_w,
                    params["fy"] / image_h,
                    params["cx"] / image_w,
                    params["cy"] / image_h,
                    params["k1"],
                    params["k2"],
                    params["p1"],
                    params["p2"],
                    params["k3"],
                ],
                dtype=torch.float32,
            )

            corner_sequence = self._extract_corner_sequence(
                selected_paths,
                image_w=image_w,
                image_h=image_h,
            )
            if corner_sequence is None:
                continue

            self.samples.append((corner_sequence, target))

        if not self.samples:
            raise FileNotFoundError(
                f"No valid corner sequences found in {sequence_dirs}. "
                "Each sequence folder must contain camera_params.yaml and at least "
                f"{self.sequence_length} images with a detectable checkerboard."
            )

    def _load_precomputed_corner_sequence(
        self,
        precomputed_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        data = np.load(precomputed_path)
        if "features" not in data:
            logger.warning("Skipping %s because it has no 'features' array.", precomputed_path)
            return None
        features = data["features"].astype(np.float32)
        if features.ndim != 3:
            logger.warning("Skipping %s because features do not have shape (T, P, F).", precomputed_path)
            return None
        if features.shape[0] < self.sequence_length:
            return None

        features = features[:self.sequence_length]
        if "target" in data:
            target = torch.tensor(data["target"].astype(np.float32), dtype=torch.float32)
        else:
            label_path = precomputed_path.with_name("camera_params.yaml")
            if not label_path.exists():
                logger.warning("Skipping %s because target and camera_params.yaml are missing.", precomputed_path)
                return None

            with open(label_path, "r") as f:
                params = yaml.safe_load(f)

            image_w = float(data.get("image_width", 640.0))
            image_h = float(data.get("image_height", 480.0))
            target = torch.tensor(
                [
                    params["fx"] / image_w,
                    params["fy"] / image_h,
                    params["cx"] / image_w,
                    params["cy"] / image_h,
                    params["k1"],
                    params["k2"],
                    params["p1"],
                    params["p2"],
                    params["k3"],
                ],
                dtype=torch.float32,
            )

        return torch.tensor(features, dtype=torch.float32), target

    def _extract_corner_sequence(
        self,
        frame_paths: list[Path],
        image_w: int,
        image_h: int,
    ) -> torch.Tensor | None:
        sequence_features: list[np.ndarray] = []

        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {frame_path}")

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
                self.corner_criteria,
            ).reshape(-1, 2).astype(np.float32)

            normalized = refined.copy()
            normalized[:, 0] /= image_w
            normalized[:, 1] /= image_h

            centroid = normalized.mean(axis=0, keepdims=True)
            centered = normalized - centroid
            scale = centered.std(axis=0, keepdims=True)
            scale = np.maximum(scale, 1e-6)
            standardized = centered / scale

            features = np.concatenate([normalized, standardized], axis=1)
            sequence_features.append(features)

        return torch.tensor(np.stack(sequence_features, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class SyntheticCornerSequenceCalibrationDataset(Dataset):
    """Dataset that generates corner sequences on-the-fly without writing files."""

    def __init__(
        self,
        num_sequences: int,
        config: TrainingConfig,
        seed: int,
        deterministic: bool,
        split_name: str,
    ):
        self.num_sequences = num_sequences
        self.config = config
        self.sequence_length = config.sequence_length
        self.seed = seed
        self.deterministic = deterministic
        self.split_name = split_name
        self.image_w = config.synthetic_image_width
        self.image_h = config.synthetic_image_height
        self.board_cols = config.synthetic_board_cols
        self.board_rows = config.synthetic_board_rows
        self.square_size = config.synthetic_square_size

        with open(config.synthetic_corner_config_path, "r") as f:
            self.synthetic_cfg = yaml.safe_load(f)

        if self.synthetic_cfg.get("mode") != "camera":
            raise ValueError(
                "Synthetic corner-sequence generation requires a config with mode: camera."
            )

        if self.num_sequences <= 0:
            raise ValueError(f"{self.split_name} sequence count must be positive, got {self.num_sequences}.")

    def __len__(self) -> int:
        return self.num_sequences

    def _make_rng(self, idx: int) -> np.random.RandomState:
        if self.deterministic:
            return np.random.RandomState(self.seed + idx)

        worker = get_worker_info()
        worker_offset = 0 if worker is None else worker.id * 1_000_003
        random_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        return np.random.RandomState(self.seed + worker_offset + random_seed)

    def _build_target(self, static_params: dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                static_params["fx"] / float(self.image_w),
                static_params["fy"] / float(self.image_h),
                static_params["cx"] / float(self.image_w),
                static_params["cy"] / float(self.image_h),
                static_params["k1"],
                static_params["k2"],
                static_params["p1"],
                static_params["p2"],
                static_params["k3"],
            ],
            dtype=torch.float32,
        )

    def _build_corner_features(self, projected_grid: np.ndarray) -> np.ndarray:
        inner_corners = projected_grid[:-1, :-1, :].reshape(-1, 2).astype(np.float32)
        normalized = inner_corners.copy()
        normalized[:, 0] /= float(self.image_w)
        normalized[:, 1] /= float(self.image_h)

        centroid = normalized.mean(axis=0, keepdims=True)
        centered = normalized - centroid
        scale = np.maximum(centered.std(axis=0, keepdims=True), 1e-6)
        standardized = centered / scale
        return np.concatenate([normalized, standardized], axis=1).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._make_rng(idx)

        static_params = _sample_camera_static_params_constrained(
            self.synthetic_cfg,
            self.image_w,
            self.image_h,
            rng,
        )
        static_params["board_cols"] = self.board_cols
        static_params["board_rows"] = self.board_rows
        static_params["square_size"] = self.square_size

        sequence_features: list[np.ndarray] = []
        for _ in range(self.sequence_length):
            frame_params = _sample_camera_dynamic_params_constrained(
                self.synthetic_cfg,
                self.image_w,
                self.image_h,
                rng,
                static_params,
            )
            projected_grid = _project_chessboard_grid(self.image_w, self.image_h, frame_params)
            sequence_features.append(self._build_corner_features(projected_grid))

        sequence_tensor = torch.tensor(np.stack(sequence_features, axis=0), dtype=torch.float32)
        return sequence_tensor, self._build_target(static_params)


def create_data_loaders(
    config: TrainingConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation data loaders from source images.

    Args:
        config: Training configuration.

    Returns:
        (train_loader, val_loader)
    """
    source_dir = Path(config.source_images_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if config.synthetic_corner_sequence_mode:
        train_count = config.max_source_items
        val_count = config.max_validation_items
        if train_count is None:
            raise ValueError("Synthetic corner-sequence mode requires --max-source-items for train sequence count.")
        if val_count is None:
            raise ValueError("Synthetic corner-sequence mode requires --max-validation-items for validation sequence count.")

        train_ds = SyntheticCornerSequenceCalibrationDataset(
            num_sequences=train_count,
            config=config,
            seed=config.synthetic_seed,
            deterministic=False,
            split_name="train",
        )
        val_ds = SyntheticCornerSequenceCalibrationDataset(
            num_sequences=val_count,
            config=config,
            seed=config.synthetic_seed + 1_000_000,
            deterministic=True,
            split_name="val",
        )
    elif config.corner_sequence_mode:
        sequence_dirs = sorted(
            p for p in source_dir.iterdir() if p.is_dir()
        )
        if config.max_source_items is not None:
            sequence_dirs = sequence_dirs[:config.max_source_items]
        if not sequence_dirs:
            raise FileNotFoundError(
                f"No sequence directories found in {source_dir}. "
                "Each sequence should be a subfolder containing images and camera_params.yaml."
            )

        split_idx = int(len(sequence_dirs) * config.train_split)
        train_dirs = sequence_dirs[:split_idx]
        val_dirs = sequence_dirs[split_idx:]
        if not val_dirs:
            val_dirs = train_dirs[-1:]
            train_dirs = train_dirs[:-1]
        if config.max_validation_items is not None:
            val_dirs = val_dirs[:config.max_validation_items]

        train_ds = CornerSequenceCalibrationDataset(train_dirs, config, split_name="train")
        val_ds = CornerSequenceCalibrationDataset(val_dirs, config, split_name="val")
    elif config.sequence_mode:
        sequence_dirs = sorted(
            p for p in source_dir.iterdir() if p.is_dir()
        )
        if config.max_source_items is not None:
            sequence_dirs = sequence_dirs[:config.max_source_items]
        if not sequence_dirs:
            raise FileNotFoundError(
                f"No sequence directories found in {source_dir}. "
                "Each sequence should be a subfolder containing images and camera_params.yaml."
            )

        split_idx = int(len(sequence_dirs) * config.train_split)
        train_dirs = sequence_dirs[:split_idx]
        val_dirs = sequence_dirs[split_idx:]
        if not val_dirs:
            val_dirs = train_dirs[-1:]
            train_dirs = train_dirs[:-1]
        if config.max_validation_items is not None:
            val_dirs = val_dirs[:config.max_validation_items]

        train_ds = SequenceCalibrationDataset(train_dirs, config)
        val_ds = SequenceCalibrationDataset(val_dirs, config)
    else:
        image_paths = sorted(
            p for p in source_dir.rglob("*") if p.suffix.lower() in extensions
        )
        if config.max_source_items is not None:
            image_paths = image_paths[:config.max_source_items]
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {source_dir}. "
                "Place undistorted images there to generate synthetic training data."
            )

        split_idx = int(len(image_paths) * config.train_split)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        if not val_paths:
            val_paths = train_paths[-1:]
            train_paths = train_paths[:-1]
        if config.max_validation_items is not None:
            val_paths = val_paths[:config.max_validation_items]

        train_ds = SyntheticCalibrationDataset(train_paths, config, seed=42)
        val_ds = SyntheticCalibrationDataset(val_paths, config, seed=123)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
