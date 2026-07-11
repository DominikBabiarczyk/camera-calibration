"""Configuration for training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    """All hyperparameters and paths in one place."""

    # --- Paths ---
    # Directory with undistorted source images (e.g., SUN360 crops or your own photos)
    source_images_dir: Path = Path("data/photos")
    output_dir: Path = Path("outputs/calibration_net")
    checkpoint_path: Path = Path("outputs/calibration_net/best_model.pth")

    # --- Dataset ---
    image_size: tuple[int, int] = (224, 224)
    # How many synthetic distorted samples to generate per source image
    samples_per_image: int = 10
    train_split: float = 0.8
    sequence_mode: bool = False
    sequence_length: int = 5
    sequence_step: int = 1
    model_name: str = "resnet18_single"

    # --- Distortion parameter ranges (Brown-Conrady model) ---
    # Focal length range (normalized by image width)
    focal_length_range: tuple[float, float] = (0.5, 2.0)
    # Radial distortion k1 range
    k1_range: tuple[float, float] = (-0.5, 0.5)
    # Radial distortion k2 range
    k2_range: tuple[float, float] = (-0.3, 0.3)

    # --- Training ---
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4

    # --- Output vector definition ---
    # The network predicts: [focal_length_normalized, k1, k2]
    num_outputs: int = 3
    output_names: list[str] = field(
        default_factory=lambda: ["focal_length", "k1", "k2"]
    )
