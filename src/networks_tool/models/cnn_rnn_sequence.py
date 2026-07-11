import torch
import torch.nn as nn
from torchvision import models


class CNNLSTMCalibrationNet(nn.Module):
    """Sequence-aware model: CNN encoder + LSTM for camera calibration."""

    def __init__(self, num_outputs: int = 3, pretrained: bool = True, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features
        self.frame_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=0.2,
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
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
        features = self.cnn(x)
        features = self.frame_proj(features.view(batch_size * seq_len, -1))
        features = features.view(seq_len, batch_size, -1)

        output, _ = self.lstm(features)
        final_state = output[-1]
        return self.regressor(final_state)
