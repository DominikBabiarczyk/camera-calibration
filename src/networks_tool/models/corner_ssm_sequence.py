import torch
import torch.nn as nn


class DiagonalStateSpaceLayer(nn.Module):
    """Small diagonal state-space layer with a stable recurrent scan.

    For each time step this layer applies the discrete state-space update

    ``h_t = a * h_(t-1) + (1 - a) * B(x_t)``

    followed by ``y_t = C(h_t) + D(x_t)``. The decay ``a`` is learned per
    state dimension and constrained to ``(0, 1)`` for stable dynamics.
    """

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, state_dim)
        self.output_projection = nn.Linear(state_dim, input_dim)
        self.skip_projection = nn.Linear(input_dim, input_dim)
        self.logit_decay = nn.Parameter(torch.full((state_dim,), 1.5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        state = inputs.new_zeros(batch_size, self.logit_decay.numel())
        decay = torch.sigmoid(self.logit_decay)
        outputs = []
        for step in range(sequence_length):
            projected = self.input_projection(inputs[:, step])
            state = decay * state + (1.0 - decay) * projected
            outputs.append(self.output_projection(state) + self.skip_projection(inputs[:, step]))
        return torch.stack(outputs, dim=1)


class CornerSSMSequenceCalibrationNet(nn.Module):
    """State-space sequence model operating on checkerboard corners.

    Input has shape ``(B, T, P, F)`` and output has shape ``(B, num_outputs)``.
    Compared with a gated LSTM, the temporal block uses a diagonal state update
    with fewer parameters and no input/forget/output gates.
    """

    def __init__(
        self,
        num_outputs: int = 9,
        num_points: int = 40,
        point_feature_dim: int = 4,
        frame_hidden_dim: int = 96,
        state_dim: int = 128,
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
        self.temporal_encoder = DiagonalStateSpaceLayer(frame_hidden_dim, state_dim)
        self.temporal_norm = nn.LayerNorm(frame_hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(frame_hidden_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_points, feature_dim = x.shape
        x = x.reshape(batch_size * sequence_length, num_points, feature_dim)
        frame_features = self.frame_encoder(x)
        frame_features = frame_features.reshape(batch_size, sequence_length, -1)
        sequence_features = self.temporal_encoder(frame_features)
        final_features = self.temporal_norm(sequence_features[:, -1])
        return self.regressor(final_features)