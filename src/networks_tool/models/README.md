# Calibration Model Prototypes

This folder contains model architectures for camera calibration experiments.
Each file implements one model class. Use them as standalone baselines or as
alternatives to the existing `CalibrationNet`.

## Included models

- `resnet18_single.py`
  - Baseline single-frame model using ResNet-18.
- `resnet50_single.py`
  - Larger single-frame model using ResNet-50.
- `efficientnet_b0_single.py`
  - EfficientNet-B0 single-frame regression model.
- `cnn_rnn_sequence.py`
  - Sequence model with a CNN backbone and LSTM temporal aggregator.
  - Good for testing temporal consistency on short video clips.
- `cnn_transformer_sequence.py`
  - Sequence model with CNN feature extraction and Transformer encoder.
  - Useful for learning frame relationships across a sequence.

## How to use

Import the models directly from the package:

```python
from src.networks_tool.models import (
    ResNet18CalibrationNet,
    ResNet50CalibrationNet,
    EfficientNetB0CalibrationNet,
    CNNLSTMCalibrationNet,
    CNNTransformerCalibrationNet,
)
```

For sequential models, provide tensors of shape `(B, T, 3, H, W)`.
For single-frame models, provide `(B, 3, H, W)`.

## Suggested tests

- Compare `ResNet18CalibrationNet` with `ResNet50CalibrationNet` to measure
  trade-off between capacity and speed.
- Try `EfficientNetB0CalibrationNet` when you want a smaller, efficient model.
- Use `CNNLSTMCalibrationNet` for sequences where distortion is constant and
  perspective changes between frames.
- Use `CNNTransformerCalibrationNet` for longer frame sequences requiring
  richer temporal attention.
