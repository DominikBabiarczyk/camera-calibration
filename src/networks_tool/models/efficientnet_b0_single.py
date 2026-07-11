import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0CalibrationNet(nn.Module):
    """Single-frame calibration regressor using EfficientNet-B0."""

    def __init__(self, num_outputs: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.classifier[1].in_features

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of images.

        Args:
            x: Tensor of shape (B, 3, H, W).

        Returns:
            Tensor of shape (B, num_outputs).
        """
        features = self.features(x)
        return self.regressor(features)
