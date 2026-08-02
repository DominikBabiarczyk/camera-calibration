"""Evaluation and inference for CalibrationNet.

Compares neural network predictions with OpenCV checkerboard calibration.

Usage:
    python -m src.networks_tool.evaluate --image path/to/image.jpg
    python -m src.networks_tool.evaluate --image-dir path/to/images/
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import TrainingConfig
from .model import CalibrationNet
from .models import (
    CNNLSTMCalibrationNet,
    CNNTransformerCalibrationNet,
    CornerGRUCalibrationNet,
    FisheyeCornerGRUSequenceCalibrationNet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a trained calibration model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: TrainingConfig = checkpoint["config"]

    model_map = {
        "resnet18_single": CalibrationNet,
        "resnet50_single": CalibrationNet,
        "efficientnet_b0_single": CalibrationNet,
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
        )
    elif config.model_name == "corner_gru_sequence":
        model = model_cls(num_outputs=config.num_outputs)
    else:
        model = model_cls(num_outputs=config.num_outputs, pretrained=False)
    strict = config.model_name != "corner_gru_sequence"
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if config.model_name == "corner_gru_sequence":
        logger.warning(
            "Loaded GRU checkpoint with strict=False; cnn/frame_proj weights are missing."
        )
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model from epoch %d (val_loss=%.6f)",
        checkpoint["epoch"],
        checkpoint["val_loss"],
    )
    return model


def predict_single_image(
    model: CalibrationNet,
    image_path: Path,
    image_size: tuple[int, int] = (224, 224),
    device: torch.device | None = None,
) -> dict[str, float]:
    """Predict camera parameters for a single image.

    Args:
        model: Trained CalibrationNet.
        image_path: Path to the image.
        image_size: Expected input size.
        device: Torch device.

    Returns:
        Dict with keys: focal_length, k1, k2 (and focal_length_px).
    """
    if device is None:
        device = next(model.parameters()).device

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    original_w = img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).squeeze(0).cpu().numpy()

    f_normalized, k1, k2 = pred
    return {
        "focal_length_normalized": float(f_normalized),
        "focal_length_px": float(f_normalized * original_w),
        "k1": float(k1),
        "k2": float(k2),
    }


def opencv_checkerboard_calibration(
    image_paths: list[Path],
    pattern_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> dict[str, float] | None:
    """Run OpenCV checkerboard calibration as ground truth baseline.

    Args:
        image_paths: Paths to checkerboard images.
        pattern_size: Number of inner corners (cols, rows).
        square_size: Size of a square in real-world units.

    Returns:
        Dict with focal_length, k1, k2 or None if calibration fails.
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_shape: tuple[int, int] | None = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape[::-1]  # (w, h)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            obj_points.append(objp)
            img_points.append(corners_refined)

    if len(obj_points) < 3 or image_shape is None:
        logger.warning("Not enough valid checkerboard images for calibration.")
        return None

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None
    )

    fx = camera_matrix[0, 0]
    k1 = dist_coeffs[0, 0]
    k2 = dist_coeffs[0, 1]

    logger.info("OpenCV calibration RMS error: %.4f", ret)
    return {
        "focal_length_px": float(fx),
        "focal_length_normalized": float(fx / image_shape[0]),
        "k1": float(k1),
        "k2": float(k2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CalibrationNet")
    parser.add_argument("--image", type=str, help="Single image to predict")
    parser.add_argument("--image-dir", type=str, help="Directory of images to predict")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/calibration_net/best_model.pth",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)

    if args.image:
        result = predict_single_image(model, Path(args.image), device=device)
        logger.info("Predictions for %s:", args.image)
        for key, value in result.items():
            logger.info("  %s: %.6f", key, value)

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() in extensions:
                result = predict_single_image(model, img_path, device=device)
                logger.info("%s → f=%.3f, k1=%.4f, k2=%.4f",
                            img_path.name,
                            result["focal_length_normalized"],
                            result["k1"],
                            result["k2"])


if __name__ == "__main__":
    main()
