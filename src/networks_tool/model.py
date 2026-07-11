"""CalibrationNet — ResNet-18 based camera parameter regression.

Architecture:
    ResNet-18 (pretrained on ImageNet) → AdaptiveAvgPool → FC layers → [f, k1, k2]

Why ResNet-18?
    - Lightweight enough for a research project (11M params)
    - Pretrained features capture useful low/mid-level patterns
      (edges, curves — exactly what distortion affects)
    - Easy to swap for ResNet-50 or EfficientNet if needed
"""

import torch
import torch.nn as nn
from torchvision import models


class CalibrationNet(nn.Module):
    """Predicts camera intrinsic parameters from a single image.

    Output vector: [focal_length_normalized, k1, k2]
        - focal_length_normalized: focal length / image_width (typically 0.5-2.0)
        - k1: first radial distortion coefficient
        - k2: second radial distortion coefficient
    """

    def __init__(self, num_outputs: int = 3, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet-18, remove the classification head
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Everything except the final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features  # 512 for ResNet-18

        # Regression head with dropout for regularization
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
        """Forward pass.

        Args:
            x: Batch of images, shape (B, 3, H, W).

        Returns:
            Predicted parameters, shape (B, num_outputs).
        """
        features = self.features(x)
        return self.regressor(features)


class LogCoshLoss(nn.Module):
    """Log-cosh loss — smooth approximation of L1 loss.

    Behaves like L2 for small errors, like L1 for large errors.
    Used in DeepCalib (Bogdan et al., 2018) for its robustness.

    $$\\mathcal{L}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i} \\log(\\cosh(y_i - \\hat{y}_i))$$
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff)))
