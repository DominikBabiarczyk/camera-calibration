import torch
import torch.nn as nn
from torchvision import models


class CornerGRUCalibrationNet(nn.Module):
    """Sequence-aware model: CNN encoder + GRU for camera calibration."""

    def __init__(self, num_outputs: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features

        self.frame_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 160),
            nn.ReLU(inplace=True),
        )

        self.frame_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Linear(96, 96),
        )

        self.temporal_encoder = nn.GRU(
            input_size=96,
            hidden_size=128,
            num_layers=1,
            batch_first=False,
            bidirectional=False,
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a frame sequence.

        Args:
            x: Tensor of shape (B, T, 3, H, W).

        Returns:
            Tensor of shape (B, num_outputs).
        """
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x).view(batch_size * seq_len, -1)
        features = self.frame_proj(features)
        features = self.frame_encoder(features)
        features = features.view(seq_len, batch_size, -1)

        output, _ = self.temporal_encoder(features)
        final_state = output[-1]
        return self.regressor(final_state)
