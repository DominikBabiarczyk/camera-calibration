import torch
import torch.nn as nn
from torchvision import models


class CNNTransformerCalibrationNet(nn.Module):
    """Sequence-aware model: CNN encoder + Transformer encoder for calibration."""

    def __init__(
        self,
        num_outputs: int = 3,
        pretrained: bool = True,
        embedding_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        max_seq_len: int = 16,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.cnn = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = backbone.fc.in_features
        self.frame_proj = nn.Linear(feature_dim, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.2,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a sequence of frames.

        Args:
            x: Tensor of shape (B, T, 3, H, W).

        Returns:
            Tensor of shape (B, num_outputs).
        """
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x).flatten(1)
        features = self.frame_proj(features)
        features = features.view(batch_size, seq_len, -1)

        if seq_len > self.pos_embedding.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.pos_embedding.size(0)}")

        features = features + self.pos_embedding[:seq_len].unsqueeze(0)
        features = features.transpose(0, 1)
        encoded = self.transformer(features)
        pooled = encoded.mean(dim=0)
        return self.regressor(pooled)
