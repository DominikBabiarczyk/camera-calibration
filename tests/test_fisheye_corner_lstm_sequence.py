import torch

from src.networks_tool.models import FisheyeCornerLSTMSequenceCalibrationNet


def test_fisheye_corner_lstm_output_shape_and_backward() -> None:
    model = FisheyeCornerLSTMSequenceCalibrationNet()
    inputs = torch.randn(2, 10, 81, 4)

    outputs = model(inputs)
    outputs.sum().backward()

    assert outputs.shape == (2, 8)
    assert all(parameter.grad is not None for parameter in model.parameters())