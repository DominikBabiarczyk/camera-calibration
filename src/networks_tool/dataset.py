"""Synthetic calibration dataset.

Generates training pairs by applying known distortion to undistorted images.
Each sample is (distorted_image, [f_normalized, k1, k2]).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import yaml

from .config import TrainingConfig
from .distortion import apply_radial_distortion, sample_random_parameters


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

        # Load and resize the undistorted source image
        img = cv2.imread(str(self.image_paths[image_idx]))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_paths[image_idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.config.image_size)

        w = self.config.image_size[0]

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

            with open(label_path, "r") as f:
                params = yaml.safe_load(f)

            image_w, image_h = self.config.image_size
            max_dim = max(image_w, image_h)
            target = torch.tensor(
                [
                    params["fx"] / max_dim,
                    params["fy"] / max_dim,
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

            frame_paths = sorted(
                p for p in seq_dir.iterdir()
                if p.suffix.lower() in extensions
            )
            if len(frame_paths) < self.sequence_length:
                continue

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
            img = cv2.resize(img, self.config.image_size)
            frames.append(self.transform(img))

        sequence_tensor = torch.stack(frames, dim=0)
        return sequence_tensor, target


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

    if config.sequence_mode:
        sequence_dirs = sorted(
            p for p in source_dir.iterdir() if p.is_dir()
        )
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

        train_ds = SequenceCalibrationDataset(train_dirs, config)
        val_ds = SequenceCalibrationDataset(val_dirs, config)
    else:
        image_paths = sorted(
            p for p in source_dir.rglob("*") if p.suffix.lower() in extensions
        )
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
