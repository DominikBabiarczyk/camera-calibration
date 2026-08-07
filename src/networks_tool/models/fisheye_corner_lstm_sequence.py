from .corner_lstm_sequence import CornerLSTMSequenceCalibrationNet


class FisheyeCornerLSTMSequenceCalibrationNet(CornerLSTMSequenceCalibrationNet):
    """Corner LSTM regressor for OpenCV's fisheye calibration model.

    Output: ``[fx, fy, cx, cy, k1, k2, k3, k4]``. Intrinsics are normalized
    by the synthetic image dimensions.
    """

    def __init__(self, num_outputs: int = 8, num_points: int = 81, **kwargs):
        kwargs["num_points"] = num_points
        super().__init__(num_outputs=num_outputs, **kwargs)