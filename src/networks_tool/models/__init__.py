"""Collection of model architectures for calibration experiments.

This package contains both single-frame regression baselines and sequence-aware
models that can process a stack of frames.
"""

from .cnn_rnn_sequence import CNNLSTMCalibrationNet
from .cnn_transformer_sequence import CNNTransformerCalibrationNet
from .corner_gru_sequence import CornerGRUCalibrationNet
from .efficientnet_b0_single import EfficientNetB0CalibrationNet
from .resnet18_single import ResNet18CalibrationNet
from .resnet50_single import ResNet50CalibrationNet

__all__ = [
    "ResNet18CalibrationNet",
    "ResNet50CalibrationNet",
    "EfficientNetB0CalibrationNet",
    "CNNLSTMCalibrationNet",
    "CNNTransformerCalibrationNet",
    "CornerGRUCalibrationNet",
]
