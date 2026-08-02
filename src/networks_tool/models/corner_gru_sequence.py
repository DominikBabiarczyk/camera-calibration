import torch
import torch.nn as nn


class CornerGRUSequenceCalibrationNet(nn.Module):
    """Compact sequence model operating on refined checkerboard corners.

    Input is a tensor of shape (B, T, P, F), where:
    - T is the number of frames in the fixed sequence,
    - P is the number of checkerboard inner corners,
    - F is the per-corner feature size.

    In the current dataset, each corner has 4 features:
    [x_norm, y_norm, x_standardized, y_standardized].
    """

    def __init__(
        self,
        num_outputs: int = 9,
        num_points: int = 40,
        point_feature_dim: int = 4,
        frame_hidden_dim: int = 96,
        temporal_hidden_dim: int = 128,
    ):
        super().__init__()
        frame_input_dim = num_points * point_feature_dim

        self.frame_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(frame_input_dim, frame_hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(frame_hidden_dim),
            nn.Linear(frame_hidden_dim, frame_hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.temporal_encoder = nn.GRU(
            input_size=frame_hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.regressor = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for processed checkerboard corner sequences.

        Args:
            x: Tensor of shape (B, T, P, F).

        Returns:
            Tensor of shape (B, num_outputs).
        """
        batch_size, seq_len, num_points, feature_dim = x.shape
        x = x.view(batch_size * seq_len, num_points, feature_dim)
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.view(batch_size, seq_len, -1)

        _, hidden = self.temporal_encoder(frame_features)
        return self.regressor(hidden[-1])



# Backwards-compatible name used by older training and evaluation code.
CornerGRUCalibrationNet = CornerGRUSequenceCalibrationNet
