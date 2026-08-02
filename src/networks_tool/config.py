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
    samples_per_image: int = 20
    train_split: float = 0.8
    max_source_items: int | None = None
    max_validation_items: int | None = None
    sequence_mode: bool = False
    corner_sequence_mode: bool = False
    synthetic_corner_sequence_mode: bool = False
    fisheye_corner_sequence_mode: bool = False
    sequence_length: int = 5
    sequence_step: int = 1
    model_name: str = "resnet18_single"
    corner_num_squares_x: int = 9
    corner_num_squares_y: int = 6
    synthetic_corner_config_path: Path = Path(
        "generate_dataset/creating_various_perspectives/camera_calibration_config.yaml"
    )
    synthetic_image_width: int = 640
    synthetic_image_height: int = 480
    synthetic_board_cols: int = 8
    synthetic_board_rows: int = 5
    synthetic_square_size: float = 48.0
    synthetic_seed: int = 42
    fisheye_calibration_result_path: Path = Path("extern/XHOG-007_charuco/result.npz")
    fisheye_image_width: int = 1920
    fisheye_image_height: int = 1080
    fisheye_board_squares_x: int = 10
    fisheye_board_squares_y: int = 10
    fisheye_square_size_mm: float = 30.0
    fisheye_intrinsics_jitter: float = 0.10
    fisheye_principal_point_jitter: float = 0.02
    fisheye_distortion_jitter: float = 0.10

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
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # --- Output vector definition ---
    # The network predicts: [focal_length_normalized, k1, k2]
    num_outputs: int = 3
    output_names: list[str] = field(
        default_factory=lambda: ["focal_length", "k1", "k2"]
    )
