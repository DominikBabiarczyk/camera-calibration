from .corner_ssm_sequence import CornerSSMSequenceCalibrationNet


class FisheyeCornerSSMSequenceCalibrationNet(CornerSSMSequenceCalibrationNet):
    """Corner SSM regressor for OpenCV's four-coefficient fisheye model."""

    def __init__(self, num_outputs: int = 8, num_points: int = 81, **kwargs):
        kwargs["num_points"] = num_points
        super().__init__(num_outputs=num_outputs, **kwargs)