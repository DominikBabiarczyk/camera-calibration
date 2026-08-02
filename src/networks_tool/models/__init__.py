"""Collection of model architectures for calibration experiments.

This package contains both single-frame regression baselines and sequence-aware
models that can process a stack of frames.
"""

from .corner_gru_sequence import CornerGRUSequenceCalibrationNet
from .cnn_rnn_sequence import CNNLSTMCalibrationNet
from .cnn_transformer_sequence import CNNTransformerCalibrationNet
from .corner_gru_sequence import CornerGRUCalibrationNet
from .fisheye_corner_gru_sequence import FisheyeCornerGRUSequenceCalibrationNet
from .efficientnet_b0_single import EfficientNetB0CalibrationNet
from .opencv_corners_sequence import OpenCVCornersCalibrationModel
from .resnet18_single import ResNet18CalibrationNet
from .resnet50_single import ResNet50CalibrationNet

__all__ = [
    "CornerGRUSequenceCalibrationNet",
    "ResNet18CalibrationNet",
    "ResNet50CalibrationNet",
    "EfficientNetB0CalibrationNet",
    "CNNLSTMCalibrationNet",
    "CNNTransformerCalibrationNet",
    "CornerGRUCalibrationNet",
    "OpenCVCornersCalibrationModel",
    "FisheyeCornerGRUSequenceCalibrationNet",
]
