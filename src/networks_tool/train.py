"""Training loop for CalibrationNet.

Usage:
    python -m src.networks_tool.train
    python -m src.networks_tool.train --epochs 100 --lr 0.0001
"""

import argparse
import logging
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .config import TrainingConfig
from .dataset import create_data_loaders
from .model import CalibrationNet, LogCoshLoss
from .models import (
    CornerGRUSequenceCalibrationNet,
    CNNLSTMCalibrationNet,
    CNNTransformerCalibrationNet,
    CornerGRUCalibrationNet,
    EfficientNetB0CalibrationNet,
    FisheyeCornerGRUSequenceCalibrationNet,
    ResNet18CalibrationNet,
    ResNet50CalibrationNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def train(config: TrainingConfig) -> None:
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info(
        "Early stopping: patience=%d, min_delta=%.2e",
        config.early_stopping_patience,
        config.early_stopping_min_delta,
    )

    # Data
    train_loader, val_loader = create_data_loaders(config)
    logger.info(
        "Train samples: %d, Val samples: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Model
    model_map = {
        "resnet18_single": ResNet18CalibrationNet,
        "resnet50_single": ResNet50CalibrationNet,
        "efficientnet_b0_single": EfficientNetB0CalibrationNet,
        "corner_gru_sequence": CornerGRUSequenceCalibrationNet,
        "cnn_lstm_sequence": CNNLSTMCalibrationNet,
        "cnn_transformer_sequence": CNNTransformerCalibrationNet,
        "corner_gru_sequence": CornerGRUCalibrationNet,
        "fisheye_corner_gru_sequence": FisheyeCornerGRUSequenceCalibrationNet,
    }
    model_cls = model_map.get(config.model_name, CalibrationNet)
    if config.model_name == "fisheye_corner_gru_sequence":
        model = model_cls(
            num_outputs=config.num_outputs,
            num_points=(config.fisheye_board_squares_x - 1)
            * (config.fisheye_board_squares_y - 1),
        ).to(device)
    else:
        model = model_cls(num_outputs=config.num_outputs).to(device)
    criterion = LogCoshLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.num_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs} [train]", unit="batch") if tqdm else train_loader
        for images, targets in train_iter:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            if tqdm:
                train_iter.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{config.num_epochs} [val]", unit="batch") if tqdm else val_loader
        with torch.no_grad():
            for images, targets in val_iter:
                images, targets = images.to(device), targets.to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item() * images.size(0)

                if tqdm:
                    val_iter.set_postfix(loss=loss.item())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        logger.info(
            "Epoch %3d/%d | Train Loss: %.6f | Val Loss: %.6f | LR: %.2e",
            epoch,
            config.num_epochs,
            train_loss,
            val_loss,
            scheduler.get_last_lr()[0],
        )

        # Save best model
        if best_val_loss - val_loss > config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                config.checkpoint_path,
            )
            logger.info("  ✓ Saved best model (val_loss=%.6f)", val_loss)
        else:
            epochs_without_improvement += 1
            logger.info(
                "  No validation improvement for %d/%d epoch(s)",
                epochs_without_improvement,
                config.early_stopping_patience,
            )

            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    epochs_without_improvement,
                )
                break

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibration model")
    parser.add_argument("--source-dir", type=str, help="Path to source image directory or sequence root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-image", type=int, default=10)
    parser.add_argument("--max-source-items", type=int, default=None, help="Limit source images or sequence folders before train/val split")
    parser.add_argument("--max-validation-items", type=int, default=None, help="Limit validation images or sequence folders after train/val split")
    parser.add_argument("--sequence", action="store_true", help="Enable sequence dataset mode")
    parser.add_argument("--corner-sequence", action="store_true", help="Enable checkerboard-corner sequence dataset mode")
    parser.add_argument(
        "--synthetic-corner-sequence",
        action="store_true",
        help="Generate checkerboard-corner sequences on-the-fly during training without saving files",
    )
    parser.add_argument(
        "--fisheye-corner-sequence",
        action="store_true",
        help="Generate fisheye corner sequences from a calibrated NPZ file.",
    )
    parser.add_argument(
        "--fisheye-calibration-result",
        type=str,
        default="extern/XHOG-007_charuco/result.npz",
        help="NPZ containing camera_matrix and four fisheye distortion coefficients.",
    )
    parser.add_argument("--fisheye-image-width", type=int, default=1920)
    parser.add_argument("--fisheye-image-height", type=int, default=1080)
    parser.add_argument("--fisheye-board-squares-x", type=int, default=10)
    parser.add_argument("--fisheye-board-squares-y", type=int, default=10)
    parser.add_argument("--fisheye-square-size-mm", type=float, default=30.0)
    parser.add_argument("--sequence-length", type=int, default=5, help="Number of frames per training sequence")
    parser.add_argument("--sequence-step", type=int, default=1, help="Stride between sequence windows")
    parser.add_argument(
        "--synthetic-corner-config",
        type=str,
        default="generate_dataset/creating_various_perspectives/camera_calibration_config.yaml",
        help="YAML config used for on-the-fly synthetic corner-sequence generation",
    )
    parser.add_argument("--synthetic-image-width", type=int, default=640, help="Synthetic sequence image width")
    parser.add_argument("--synthetic-image-height", type=int, default=480, help="Synthetic sequence image height")
    parser.add_argument("--synthetic-board-cols", type=int, default=8, help="Number of inner checkerboard corners in x")
    parser.add_argument("--synthetic-board-rows", type=int, default=5, help="Number of inner checkerboard corners in y")
    parser.add_argument("--synthetic-square-size", type=float, default=48.0, help="Synthetic checkerboard square size in pixels")
    parser.add_argument("--synthetic-seed", type=int, default=42, help="Base random seed for on-the-fly synthetic sequence generation")
    parser.add_argument("--model-name", type=str, default="resnet18_single", help="Model name to train")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for training outputs")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path for the best-model checkpoint")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop training after this many epochs without sufficient val loss improvement",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum val loss improvement required to reset early stopping",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        samples_per_image=args.samples_per_image,
        max_source_items=args.max_source_items,
        max_validation_items=args.max_validation_items,
        sequence_mode=args.sequence,
        corner_sequence_mode=args.corner_sequence,
        synthetic_corner_sequence_mode=args.synthetic_corner_sequence,
        fisheye_corner_sequence_mode=args.fisheye_corner_sequence,
        sequence_length=args.sequence_length,
        sequence_step=args.sequence_step,
        synthetic_corner_config_path=Path(args.synthetic_corner_config),
        synthetic_image_width=args.synthetic_image_width,
        synthetic_image_height=args.synthetic_image_height,
        synthetic_board_cols=args.synthetic_board_cols,
        synthetic_board_rows=args.synthetic_board_rows,
        synthetic_square_size=args.synthetic_square_size,
        synthetic_seed=args.synthetic_seed,
        fisheye_calibration_result_path=Path(args.fisheye_calibration_result),
        fisheye_image_width=args.fisheye_image_width,
        fisheye_image_height=args.fisheye_image_height,
        fisheye_board_squares_x=args.fisheye_board_squares_x,
        fisheye_board_squares_y=args.fisheye_board_squares_y,
        fisheye_square_size_mm=args.fisheye_square_size_mm,
        model_name=args.model_name,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    if config.fisheye_corner_sequence_mode:
        config.num_outputs = 8
        config.output_names = ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]
    elif config.sequence_mode or config.corner_sequence_mode or config.synthetic_corner_sequence_mode:
        config.num_outputs = 9
        config.output_names = [
            "fx",
            "fy",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
        ]
    if args.source_dir:
        config.source_images_dir = Path(args.source_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.checkpoint_path:
        config.checkpoint_path = Path(args.checkpoint_path)

    train(config)


if __name__ == "__main__":
    main()
