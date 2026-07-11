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
    CNNLSTMCalibrationNet,
    CNNTransformerCalibrationNet,
    EfficientNetB0CalibrationNet,
    ResNet18CalibrationNet,
    ResNet50CalibrationNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def train(config: TrainingConfig) -> None:
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

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
        "cnn_lstm_sequence": CNNLSTMCalibrationNet,
        "cnn_transformer_sequence": CNNTransformerCalibrationNet,
    }
    model_cls = model_map.get(config.model_name, CalibrationNet)
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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

    logger.info("Training complete. Best val loss: %.6f", best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibration model")
    parser.add_argument("--source-dir", type=str, help="Path to source image directory or sequence root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence", action="store_true", help="Enable sequence dataset mode")
    parser.add_argument("--sequence-length", type=int, default=5, help="Number of frames per training sequence")
    parser.add_argument("--model-name", type=str, default="resnet18_single", help="Model name to train")
    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        sequence_mode=args.sequence,
        sequence_length=args.sequence_length,
        model_name=args.model_name,
    )
    if config.sequence_mode:
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

    train(config)


if __name__ == "__main__":
    main()
