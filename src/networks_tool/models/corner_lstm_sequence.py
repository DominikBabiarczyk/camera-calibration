import torch
import torch.nn as nn


class CornerLSTMSequenceCalibrationNet(nn.Module):
    """LSTM sequence model operating on refined checkerboard corners.

    Input has shape ``(B, T, P, F)``. Each frame is encoded independently,
    then the resulting feature sequence is aggregated by an LSTM.
    """

    def __init__(
        self,
        num_outputs: int = 9,
        num_points: int = 40,
        point_feature_dim: int = 4,
        frame_hidden_dim: int = 96,
        temporal_hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
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

        self.temporal_encoder = nn.LSTM(
            input_size=frame_hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(temporal_hidden_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict calibration parameters from ``(B, T, P, F)`` corners."""
        batch_size, seq_len, num_points, feature_dim = x.shape
        x = x.reshape(batch_size * seq_len, num_points, feature_dim)
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.reshape(batch_size, seq_len, -1)

        _, (hidden, _) = self.temporal_encoder(frame_features)
        return self.regressor(hidden[-1])