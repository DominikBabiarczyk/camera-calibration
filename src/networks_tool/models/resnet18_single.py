import torch
import torch.nn as nn
from torchvision import models


class ResNet18CalibrationNet(nn.Module):
    """Single-frame calibration regressor using ResNet-18."""

    def __init__(self, num_outputs: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features

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
